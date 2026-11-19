"""Roadmap item 4 (speed): where the encode and decode time of the current codec goes.

Codes a few frames of one DAVIS TEST and one UVG 1080p sequence exactly as `p4_gate3.py` does
(A2 P-coder + wide I-coder, GOP 30, untuned) and times every stage by wrapping the functions
`p4_gate3` calls. CUDA is synchronised around each timed call, so the stage times add up to
wall time (the sync itself costs a little). Each sequence is coded twice and the second pass
is reported, so cuDNN autotuning and first-call costs are excluded.

Also records motion bytes vs residual bytes per P-frame (the low-rate write-up's open question).

Usage:
    p7_speed_profile.py [--lambdas 1e-2 1e-3] [--frames 31] [--output outputs/p7_speed/profile.json]
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

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

TIMES: dict[str, float] = defaultdict(float)
CALLS: dict[str, int] = defaultdict(int)
DEPTH = [0]   # only the outermost timed call is charged, so nested stages are not double-counted


def timed(name, fn, sync):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        if DEPTH[0]:
            return fn(*a, **k)
        DEPTH[0] += 1
        try:
            if sync:
                torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*a, **k)
            if sync:
                torch.cuda.synchronize()
            TIMES[name] += time.perf_counter() - t
            CALLS[name] += 1
            return out
        finally:
            DEPTH[0] -= 1
    return wrapper


def instrument(mc, sync):
    """Wrap the leaf stages. encode_frame/decode_frame are NOT wrapped: their remainder
    (total minus the leaves) is the encoder network and glue."""
    for name in ("estimate_block_motion", "encode_motion_payload", "decode_motion_payload",
                 "warp_blocks"):
        setattr(mc, name, timed(f"motion/{name}", getattr(mc, name), sync))
    g4._y_side = timed("net/y_prior", g4._y_side, sync)
    g4._x_side = timed("net/x_prior", g4._x_side, sync)
    g4._reconstruct = timed("net/synthesis+refine", g4._reconstruct, sync)
    g4.encode_symbols = timed("range/encode_symbols", g4.encode_symbols, sync)
    g4.decode_symbols = timed("range/decode_symbols", g4.decode_symbols, sync)
    g3.scale_index = timed("range/scale_index", g3.scale_index, sync)
    p1.laplace_bits = timed("range/ideal_bits(report only)", p1.laplace_bits, sync)


def run(mc, m21, intra, icoder, pcoder, frames, rig, tables, identity, args, sync):
    path = Path(tempfile.mkdtemp(prefix="p7_")) / "s.nvct"
    TIMES.clear(); CALLS.clear()
    if sync:
        torch.cuda.synchronize()
    t = time.perf_counter()
    encoded = g4.encode_sequence(mc, m21, intra, icoder, pcoder, frames, path, rig, tables,
                                 gop=30, block_size=args.block_size,
                                 search_range=args.search_range, identity=identity)
    if sync:
        torch.cuda.synchronize()
    encode_wall = time.perf_counter() - t
    enc = dict(TIMES)
    TIMES.clear(); CALLS.clear()
    t = time.perf_counter()
    decoded = g4.decode_sequence(mc, m21, intra, icoder, pcoder, path, rig, tables,
                                 identity=identity)
    if sync:
        torch.cuda.synchronize()
    decode_wall = time.perf_counter() - t
    dec = dict(TIMES)
    exact = torch.equal(decoded, encoded["reconstructions"])
    path.unlink()
    return encoded, encode_wall, enc, decode_wall, dec, exact


def main() -> int:
    defaults = load_default_config()
    parser = bp.build_arg_parser(defaults)
    parser.add_argument("--lambdas", nargs="+", default=["1e-2", "1e-3"])
    parser.add_argument("--frames", type=int, default=31, help="31 = one GOP of 30 + the next I")
    parser.add_argument("--models-dir-new", type=Path, default=Path("outputs/p6_low_rate/vimeo"))
    parser.add_argument("--uvg-manifest", type=Path, default=Path("C:/uvg_scratch/uvg/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/p7_speed/profile.json"))
    args = parser.parse_args()
    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    sync = device.type == "cuda"
    mc = bp._load_script("m10h_motion_compensation")
    m21 = bp._load_script("m21_refinement")
    md = bp._load_script("m11_data")
    instrument(mc, sync)
    tables = g3.cumulative_tables()
    train_full = discover_sequences(args.manifest, split="train")
    motion_train = m21.broad_train_sequences(args.manifest,
                                             frames_per_sequence=args.train_frames_per_sequence)
    sets = {"davis_256": discover_sequences(args.manifest, split="test", max_sequences=1,
                                            max_frames_per_sequence=args.frames)}
    if args.uvg_manifest.is_file():
        sets["uvg_1080p"] = discover_sequences(args.uvg_manifest, split="test", max_sequences=1,
                                               max_frames_per_sequence=args.frames)
    out = {"device": torch.cuda.get_device_name(0) if sync else str(device),
           "frames": args.frames, "gop": 30, "results": {}}
    for lam in args.lambdas:
        rig_lam = lam if lam in g3.BITS else "1e-3"
        models_dir = p4.VIMEO_DIR if lam in p1.LAMBDAS else args.models_dir_new
        intra, _ = load_model_from_checkpoint(Path(p1.CHECKPOINT.format(rig_lam)), device=device)
        intra.eval()
        pcoder = p4.load_vimeo("p", lam, device, models_dir)
        icoder = p4.load_vimeo("i", lam, device, models_dir)
        identity = g4.identity_of(pcoder, icoder)
        rig = g3.build_rig(args, m21, md, intra, rig_lam, g3.BITS[rig_lam], device, train_full,
                           motion_train)
        for set_name, seqs in sets.items():
            seq = seqs[0]
            frames = seq.load_frames()
            n = frames.shape[0]
            run(mc, m21, intra, icoder, pcoder, frames, rig, tables, identity, args, sync)  # warm-up
            encoded, ew, enc, dw, dec, exact = run(mc, m21, intra, icoder, pcoder, frames, rig,
                                                   tables, identity, args, sync)
            p_recs = [r for r in encoded["records"] if r["type"] == "P"]
            i_recs = [r for r in encoded["records"] if r["type"] == "I"]
            enc["net/encoder+glue(remainder)"] = ew - sum(enc.values())
            dec["net/glue(remainder)"] = dw - sum(dec.values())
            key = f"{set_name}/{lam}"
            out["results"][key] = {
                "sequence": seq.sequence_id, "frames": n, "size": list(frames.shape[2:]),
                "decode_exact": exact,
                "encode_s_per_frame": ew / n, "decode_s_per_frame": dw / n,
                "encode_fps": n / ew, "decode_fps": n / dw,
                "encode_stages_s_per_frame": {k: v / n for k, v in sorted(enc.items(), key=lambda kv: -kv[1])},
                "decode_stages_s_per_frame": {k: v / n for k, v in sorted(dec.items(), key=lambda kv: -kv[1])},
                "bytes_per_p_frame": {
                    "motion": sum(r["motion"] for r in p_recs) / max(len(p_recs), 1),
                    "residual": sum(r["residual"] for r in p_recs) / max(len(p_recs), 1)},
                "bytes_per_i_frame": sum(r["residual"] for r in i_recs) / max(len(i_recs), 1),
            }
            r = out["results"][key]
            print(f"== {key} {seq.sequence_id} {n} frames {tuple(frames.shape[2:])} exact={exact}")
            print(f"   encode {r['encode_s_per_frame']:.3f} s/frame ({r['encode_fps']:.1f} fps), "
                  f"decode {r['decode_s_per_frame']:.3f} s/frame ({r['decode_fps']:.1f} fps)")
            for side in ("encode", "decode"):
                total = r[f"{side}_s_per_frame"]
                for k, v in r[f"{side}_stages_s_per_frame"].items():
                    print(f"   {side:6s} {k:34s} {v * 1000:8.1f} ms  {100 * v / total:5.1f}%")
            print(f"   P-frame bytes: motion {r['bytes_per_p_frame']['motion']:.0f}  "
                  f"residual {r['bytes_per_p_frame']['residual']:.0f};  I-frame {r['bytes_per_i_frame']:.0f}",
                  flush=True)
            del frames
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
