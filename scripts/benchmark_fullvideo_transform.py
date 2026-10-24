"""Full-video BD-rate for any transform, through one fixed entropy stack.

WHAT THIS ANSWERS, AND WHAT IT DOES NOT
----------------------------------------
The Stage 1 transform work is measured intra-only. The 64-channel GDN/residual
transform, quantization-aware fine-tuned, reached +46.5% intra BD-rate against
x264 all-intra where the deployed 593k transform sits at +176.2%
(`outputs/stage1_vimeo/colab_run4_64ch_qat/`). Nobody knows what that is worth
on full video, which is what the project is actually judged on.

Getting the *deployed* answer means refitting the whole P-frame lineage to the
new latent - M10K's z_ref-conditioned model, M10L's 512-prototype codebook,
M11's channel-autoregressive context model, then M13/M14/M22's recalibrations -
because every one of them is fitted to the old transform and is refused by the
provenance check. That is a milestone-sized program.

This script answers the narrower question first, and cheaply: **holding the
entropy stack fixed, how much does the transform change full-video BD-rate?**
It codes real GOP-10 streams with P-frames, motion and latent residuals, using
only what `calibrate_grids` produces - the intra grid, the residual grid, the
motion table and empirical entropy models for each. No learned context model, no
codebook, none of the M13/M14/M22 refinements.

Consequences, stated plainly so no one over-reads the output:

  * The absolute BD-rate here is WORSE than the deployed codec's +450.7%, for
    every transform, because this stack is deliberately weaker than the deployed
    one. It is not comparable to Table II of the paper.
  * It is only meaningful as a DIFFERENCE between two transforms measured
    through this same stack. Run it on the deployed checkpoint as well as the
    new one; the delta is attributable to the transform, because nothing else
    moved.
  * A transform that wins here is very likely to win with the full stack too,
    but the margin will differ: the learned stack exploits residual structure
    this one ignores.

Classical anchors are read from the committed Stage 0 run rather than
re-encoded, so x264/x265 are the same bytes the paper reports.

Usage - run it twice, once per transform, then compare:

  ./.venv/Scripts/python.exe scripts/benchmark_fullvideo_transform.py \\
      --checkpoint outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt \\
      --output-name fullvideo_deployed.json

  ./.venv/Scripts/python.exe scripts/benchmark_fullvideo_transform.py \\
      --checkpoint outputs/stage1_vimeo/colab_run4_64ch_qat/best.pt \\
      --output-name fullvideo_64ch_qat.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

ROOT = Path(__file__).resolve().parents[1]

# The committed Stage 0 run supplies the classical curves, so the anchors here
# are byte-identical to the ones the paper reports.
ANCHORS = Path("outputs/benchmarks/parity_s0/parity.json")
# Default arms to compare against: x264/x265 in their own GOP structure, and
# forced to this codec's (I every ten, no B).
ANCHOR_ARMS = ("h264", "h265", "h264_lowdelay", "h265_lowdelay")
DEPLOYED_CHECKPOINT = Path(
    "outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEPLOYED_CHECKPOINT,
                        help="Any checkpoint; the architecture comes from the file.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/benchmarks/fullvideo_transform"))
    parser.add_argument("--output-name", type=str, default="fullvideo.json")
    parser.add_argument("--rate-points", type=int, nargs="+", default=[5, 4, 3])
    parser.add_argument("--gop", type=int, default=10,
                        help="Coding GOP. 10 matches the deployed codec and the "
                             "forced-GOP classical arms.")
    parser.add_argument("--calibration-frames", type=int, default=400,
                        help="Matches the deployed recipe; coverage matters more "
                             "than count (see M15).")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--anchors", type=Path, default=ANCHORS)
    parser.add_argument("--anchor-arms", type=str, nargs="+", default=list(ANCHOR_ARMS))
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="Smoke runs only.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Smoke runs only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def simple_rig(mc, model, train_sequences, *, bits: int, gop: int,
               calibration_frames: int, block_size: int,
               search_range: int) -> dict[str, Any]:
    """Everything a GOP-`gop` stream needs, and nothing learned.

    `calibrate_grids` fits the intra grid, the residual grid and the motion
    table by running this model over TRAIN frames, and returns an empirical
    entropy model for each. It loads no trained entropy model and checks no
    provenance, which is exactly why it works for a transform whose latent the
    deployed stack has never seen.
    """
    calibration = mc.calibrate_grids(
        model, train_sequences, bits=bits, mode="per_channel", gop_size=gop,
        block_size=block_size, search_range=search_range, reference_mode="mc",
        max_frames=calibration_frames)
    return {
        "bits": bits,
        "intra_params": calibration["intra_params"],
        "intra_entropy_model": calibration["intra_entropy_model"],
        "residual_params": calibration["residual_params"],
        "residual_entropy_model": calibration["residual_entropy_model"],
        "motion_entropy_model": calibration["motion_entropy_model"],
        "provenance": calibration["provenance"],
    }


def anchor_curve(path: Path, arm: str, convention: str, metric: str,
                 parity) -> list[tuple[float, float]]:
    """One classical arm's rate-quality curve from the committed Stage 0 run."""
    report = json.loads(path.read_text(encoding="utf-8"))
    points = [p for p in report["points"].values() if p["arm"] == arm]
    if len(points) < 2:
        raise SystemExit(
            f"{path} has fewer than two {arm!r} points, so it cannot supply a "
            "reference curve. Check --anchor-arms against that report.")
    return parity.curve(points, convention, metric)


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    parity = _load_script("benchmark_parity")
    mc = _load_script("m10h_motion_compensation")

    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    architecture = checkpoint.get("architecture", "BaselineAutoencoder")
    qat = {k: v for k, v in (checkpoint.get("extra") or {}).items()
           if k.startswith("qat_")}

    sequences = discover_sequences(args.manifest, split="test",
                                   max_sequences=args.max_sequences,
                                   max_frames_per_sequence=args.max_frames)
    frame_cache = {s.sequence_id: s.load_frames() for s in sequences}
    train_sequences = discover_sequences(args.manifest, split="train")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "stage": "full-video BD-rate for one transform, fixed simple entropy stack",
        "caveat": ("Entropy stack is calibrate_grids only - no learned context "
                   "model, no codebook, no M13/M14/M22 refinements. Absolute "
                   "BD-rate is therefore worse than the deployed codec's and is "
                   "NOT comparable to it. Meaningful only as a difference "
                   "between transforms measured through this same stack."),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "checkpoint": str(args.checkpoint), "architecture": architecture,
            "model_config": checkpoint.get("model_config"),
            "parameters": model.num_parameters(),
            "qat": qat or None,
            "rate_points": args.rate_points, "gop": args.gop,
            "calibration_frames": args.calibration_frames,
            "block_size": args.block_size, "search_range": args.search_range,
            "anchors": str(args.anchors), "anchor_arms": args.anchor_arms,
            "primary_convention": parity.PRIMARY_CONVENTION,
            "max_sequences": args.max_sequences, "max_frames": args.max_frames,
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": str(device), "platform": platform.platform()},
        "dataset": {"split": "test", "sequences": [s.sequence_id for s in sequences],
                    "frames": sum(s.frame_count for s in sequences)},
        "points": {},
    }

    print("=" * 100)
    print(f"FULL-VIDEO, SIMPLE STACK - {architecture}, {model.num_parameters():,} parameters"
          + (f", QAT {qat['qat_bits']}-bit/{qat['qat_mode']}" if qat else ", no QAT"))
    print(f"{len(sequences)} sequences, {report['dataset']['frames']} frames, "
          f"GOP {args.gop}, primary convention: {parity.PRIMARY_CONVENTION}")
    print("=" * 100, flush=True)

    stream_dir = Path(tempfile.mkdtemp(prefix="fullvideo_"))
    for bits in args.rate_points:
        rig = simple_rig(mc, model, train_sequences, bits=bits, gop=args.gop,
                         calibration_frames=args.calibration_frames,
                         block_size=args.block_size, search_range=args.search_range)
        rows, exact = [], True
        for sequence in sequences:
            frames = frame_cache[sequence.sequence_id]
            path = stream_dir / f"{bits}bit_{sequence.sequence_id}.nvct"

            started = time.perf_counter()
            encoded = mc.encode_sequence(
                model, frames, path,
                intra_params=rig["intra_params"],
                intra_entropy_model=rig["intra_entropy_model"],
                residual_params=rig["residual_params"],
                residual_entropy_model=rig["residual_entropy_model"],
                motion_entropy_model=rig["motion_entropy_model"],
                mode="mc", gop_size=args.gop, block_size=args.block_size,
                search_range=args.search_range)
            encode_seconds = time.perf_counter() - started
            size = path.stat().st_size

            started = time.perf_counter()
            decoded = mc.decode_sequence(
                model, path,
                intra_entropy_model=rig["intra_entropy_model"],
                residual_entropy_model=rig["residual_entropy_model"],
                motion_entropy_model=rig["motion_entropy_model"])
            decode_seconds = time.perf_counter() - started
            path.unlink()

            # `decode_sequence` returns the frames directly; `encode_sequence`
            # returns a record whose reconstruction key is `encoder_reconstructions`.
            # The decoder must reach the encoder's own reconstruction from the
            # file alone, or the stream is not self-contained.
            # Both to CPU first: the encoder keeps its reconstructions on the
            # compute device while the decoder returns them on the host, and
            # torch.equal refuses a cross-device comparison.
            decoded = decoded.cpu()
            exact &= torch.equal(decoded, encoded["encoder_reconstructions"].cpu())

            scores = parity.score_frames(
                parity.to_uint8_frames(decoded), frames, device=device)
            rows.append({"sequence": sequence.sequence_id, "bytes": size,
                         "pixels": sequence.total_pixels,
                         "encode_seconds": encode_seconds,
                         "decode_seconds": decode_seconds,
                         # The encoder's own accounting, kept because the motion
                         # share of the stream is a number the paper quotes and
                         # which a new transform can move.
                         "motion_bytes": encoded["motion_bytes"],
                         "residual_bytes": encoded["residual_bytes"],
                         "i_frame_residual_bytes": encoded["i_frame_residual_bytes"],
                         "p_frame_residual_bytes": encoded["p_frame_residual_bytes"],
                         "container_overhead_bytes": encoded["container_overhead_bytes"],
                         **scores})

        point = parity.summarize_point(
            f"nvc_simple/{bits}bit", "nvc_simple", f"{bits}bit", rows,
            extra={"bits": bits, "decode_reconstruction_exact": exact,
                   "calibration_provenance": rig["provenance"]})
        motion = sum(r["motion_bytes"] for r in rows)
        point["byte_split"] = {
            "motion_bytes": motion,
            "residual_bytes": sum(r["residual_bytes"] for r in rows),
            "i_frame_residual_bytes": sum(r["i_frame_residual_bytes"] for r in rows),
            "p_frame_residual_bytes": sum(r["p_frame_residual_bytes"] for r in rows),
            "container_overhead_bytes": sum(r["container_overhead_bytes"] for r in rows),
            "motion_share_of_stream": motion / point["total_bytes"],
        }
        report["points"][point["label"]] = point
        quality = point["quality"][parity.PRIMARY_CONVENTION]
        print(f"  {bits}-bit  {point['total_bytes']:>11,} B  bpp {point['bpp']:.4f}  "
              f"PSNR {quality['psnr']:.3f}  MS-SSIM {quality['msssim']:.4f}  "
              f"motion {100 * point['byte_split']['motion_share_of_stream']:.1f}%  "
              f"decode-exact={exact}", flush=True)

    nvc_points = list(report["points"].values())
    report["comparisons"] = {}
    for arm in args.anchor_arms:
        for convention in parity.CONVENTIONS:
            for metric in ("psnr", "msssim"):
                base = anchor_curve(args.anchors, arm, convention, metric, parity)
                test = parity.curve(nvc_points, convention, metric)
                report["comparisons"][f"{arm}/{convention}/{metric}"] = {
                    "bd_rate_percent": parity.bd_rate(base, test),
                    "quality_overlap": parity.overlap(base, test),
                }

    report["valid"] = all(p.get("decode_reconstruction_exact", True)
                          for p in report["points"].values())
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report_path = args.output_dir / args.output_name
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\nBD-rate (primary convention, PSNR):")
    for arm in args.anchor_arms:
        value = report["comparisons"][f"{arm}/{parity.PRIMARY_CONVENTION}/psnr"]
        headline = value["bd_rate_percent"]
        print(f"  vs {arm:<16} "
              + (f"{headline:+.1f}%" if headline is not None else "n/a (no overlap)"))
    print(f"run valid: {report['valid']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
