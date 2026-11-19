"""Roadmap item 5 (motion), step 0: how much of the stream is motion, on every video.

Encodes (no decode) DAVIS TEST in full and the first 31 frames of every UVG video with the
current codec (A2 P-coder + wide I-coder, GOP 30, untuned) at all five lambdas, and tallies
I-frame, motion and residual bytes. For every coded motion field it also measures the
empirical entropy of the vectors as coded today (raw dy, dx) and of the difference from the
H.264 median predictor (left, top, top-right), pooled per dataset and lambda: a rough estimate
of what neighbour prediction alone could save, with the fields unchanged.

    p8_motion_share.py [--output outputs/p8_motion/share.json]
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_parity as bp  # noqa: E402
import p1_gate3 as g3  # noqa: E402
import p1_pframe as p1  # noqa: E402
import p4_gate3 as g4  # noqa: E402
import p4_wide_latent as p4  # noqa: E402

from nvc.evaluation.sequences import discover_sequences  # noqa: E402
from nvc.training.checkpoint import load_model_from_checkpoint  # noqa: E402
from nvc.utils.config import load_default_config  # noqa: E402
from nvc.utils.device import get_device  # noqa: E402
from nvc.utils.seed import seed_everything  # noqa: E402

LAMBDAS = ["1e-2", "3e-3", "1e-3", "3e-4", "1e-4"]
NEW_DIR = Path("outputs/p6_low_rate/vimeo")


def median_prediction(field: np.ndarray) -> np.ndarray:
    """H.264-style predictor per block from left (A), top (B), top-right (C, else top-left);
    missing neighbours count as zero, a lone available neighbour is used as is."""
    _, by, bx = field.shape
    pred = np.zeros_like(field)
    for y in range(by):
        for x in range(bx):
            avail = []
            if x > 0:
                avail.append(field[:, y, x - 1])
            if y > 0:
                avail.append(field[:, y - 1, x])
                if x + 1 < bx:
                    avail.append(field[:, y - 1, x + 1])
                elif x > 0:
                    avail.append(field[:, y - 1, x - 1])
            if len(avail) == 1:
                pred[:, y, x] = avail[0]
            elif len(avail) >= 2:
                stack = np.stack(avail + [np.zeros(2, dtype=field.dtype)] * (3 - len(avail)))
                pred[:, y, x] = np.median(stack, axis=0).astype(field.dtype)
    return pred


def entropy_bits(counter: Counter) -> float:
    total = sum(counter.values())
    p = np.array(list(counter.values()), dtype=np.float64) / total
    return float(-(p * np.log2(p)).sum() * total)


def main() -> int:
    defaults = load_default_config()
    parser = bp.build_arg_parser(defaults)
    parser.add_argument("--uvg-manifest", type=Path, default=Path("C:/uvg_scratch/uvg/manifest.json"))
    parser.add_argument("--uvg-frames", type=int, default=31)
    parser.add_argument("--output", type=Path, default=Path("outputs/p8_motion/share.json"))
    args = parser.parse_args()
    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    mc = bp._load_script("m10h_motion_compensation")
    m21 = bp._load_script("m21_refinement")
    md = bp._load_script("m11_data")
    fields: list[np.ndarray] = []
    real_estimate = mc.estimate_block_motion

    def capture(reference, current, **kwargs):
        motion = real_estimate(reference, current, **kwargs)
        fields.append(motion.cpu().numpy())
        return motion

    mc.estimate_block_motion = capture
    tables = g3.cumulative_tables()
    train_full = discover_sequences(args.manifest, split="train")
    motion_train = m21.broad_train_sequences(args.manifest,
                                             frames_per_sequence=args.train_frames_per_sequence)
    sets = {"davis_test": discover_sequences(args.manifest, split="test"),
            "uvg_1080p_31f": discover_sequences(args.uvg_manifest, split="test",
                                                max_frames_per_sequence=args.uvg_frames)}
    out: dict = {"lambdas": LAMBDAS, "results": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for lam in LAMBDAS:
        rig_lam = lam if lam in g3.BITS else "1e-3"
        models_dir = p4.VIMEO_DIR if lam in p1.LAMBDAS else NEW_DIR
        intra, _ = load_model_from_checkpoint(Path(p1.CHECKPOINT.format(rig_lam)), device=device)
        intra.eval()
        pcoder = p4.load_vimeo("p", lam, device, models_dir)
        icoder = p4.load_vimeo("i", lam, device, models_dir)
        identity = g4.identity_of(pcoder, icoder)
        rig = g3.build_rig(args, m21, md, intra, rig_lam, g3.BITS[rig_lam], device, train_full,
                           motion_train)
        for set_name, seqs in sets.items():
            tally = {"i": 0, "motion": 0, "residual": 0, "p_frames": 0, "pixels": 0}
            raw, pred = Counter(), Counter()
            per_video = {}
            for seq in seqs:
                frames = seq.load_frames()
                fields.clear()
                path = Path(tempfile.mkdtemp(prefix="p8_")) / "s.nvct"
                enc = g4.encode_sequence(mc, m21, intra, icoder, pcoder, frames, path, rig, tables,
                                         gop=30, block_size=args.block_size,
                                         search_range=args.search_range, identity=identity)
                path.unlink()
                v = {"i": 0, "motion": 0, "residual": 0}
                for r in enc["records"]:
                    if r["type"] == "I":
                        v["i"] += r["residual"]
                    else:
                        v["motion"] += r["motion"]
                        v["residual"] += r["residual"]
                        tally["p_frames"] += 1
                for f in fields:
                    d = f - median_prediction(f)
                    for c in range(2):
                        raw.update(((c, int(s)) for s in f[c].ravel()))
                        pred.update(((c, int(s)) for s in d[c].ravel()))
                for k in v:
                    tally[k] += v[k]
                tally["pixels"] += frames.shape[0] * frames.shape[2] * frames.shape[3]
                per_video[seq.sequence_id] = {**v, "motion_share_of_p": v["motion"] / max(v["motion"] + v["residual"], 1)}
                del frames
            total = tally["i"] + tally["motion"] + tally["residual"]
            raw_bits = sum(entropy_bits(Counter({k: n for k, n in raw.items() if k[0] == c})) for c in range(2))
            pred_bits = sum(entropy_bits(Counter({k: n for k, n in pred.items() if k[0] == c})) for c in range(2))
            res = {
                "bpp": total * 8 / tally["pixels"],
                "share_of_stream": {"i": tally["i"] / total, "motion": tally["motion"] / total,
                                    "residual": tally["residual"] / total},
                "motion_share_of_p": tally["motion"] / max(tally["motion"] + tally["residual"], 1),
                "motion_bytes_coded": tally["motion"],
                "motion_entropy_bytes_raw": raw_bits / 8,
                "motion_entropy_bytes_median_pred": pred_bits / 8,
                "zero_mvd_fraction": sum(n for k, n in pred.items() if k[1] == 0) / max(sum(pred.values()), 1),
                "per_video": per_video,
            }
            out["results"][f"{set_name}/{lam}"] = res
            print(f"== {set_name}/{lam}: bpp {res['bpp']:.4f}  motion {100 * res['share_of_stream']['motion']:.1f}% of stream, "
                  f"{100 * res['motion_share_of_p']:.1f}% of P bytes; motion coded {tally['motion']:,} B, "
                  f"entropy raw {raw_bits / 8:,.0f} B, median-pred {pred_bits / 8:,.0f} B "
                  f"(zero mvd {100 * res['zero_mvd_fraction']:.0f}%)", flush=True)
            args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
