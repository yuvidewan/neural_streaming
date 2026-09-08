"""M10G evaluation: intra-only versus temporal, on identical frames and bytes.

THE COMPARISON
---------------
    A. INTRA-ONLY  - every frame coded independently (GOP = 1)
    B. TEMPORAL    - I-frame then P-frames (GOP = 10, configurable)

Both arms run through the SAME container, the SAME frozen model at the frozen
operating point lambda = 3.0e-4, and the SAME intra quantization grid and
entropy model. Arm A is produced by running the temporal coder with `gop_size=1`,
which makes every frame an I-frame. That is deliberate: it means the two arms
differ ONLY in whether P-frames exist, and container overhead is accounted
identically on both sides rather than being compared across two different file
formats.

BYTE ACCOUNTING
----------------
Stream BPP is `total container bytes * 8 / total decoded pixels` - every byte in
the file, including the header, both quantization blocks, and per-frame records.
I-frame overhead is inside that number, not excluded from it. Reporting only the
P-frame payload would flatter the temporal arm by hiding the I-frames it depends
on, so I-frame bytes, P-frame bytes and container overhead are all reported
separately as well as in the total.

CHECKPOINT SELECTION
---------------------
The model is the M10F lambda = 3e-4 run's `best.pt` - the M10G Part A convention
(best validation checkpoint), not the final snapshot. Selection used validation
history only; no test metric was consulted to choose it.

Example usage (PowerShell, from the project root, with .venv activated):

    .venv\\Scripts\\python.exe scripts\\m10g_evaluate.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

from nvc.evaluation.perceptual_metrics import msssim
from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device

DEFAULT_OUTPUT_DIR = Path("outputs/m10g_temporal_baseline")
FROZEN_LAMBDA = 3.0e-4
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10G: intra-only vs temporal baseline on the DAVIS test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--reference-mode", choices=["latent", "reencode"], default="latent")
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def _psnr_by_gop_position(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean PSNR at each distance from the last I-frame, pooled over sequences."""
    buckets: dict[int, list[float]] = {}
    for row in rows:
        for position, value in zip(row["gop_positions"], row["per_frame_psnr_db"]):
            if math.isfinite(value):
                buckets.setdefault(position, []).append(value)
    return [{"gop_position": position,
             "frames": len(buckets[position]),
             "mean_psnr_db": statistics.fmean(buckets[position])}
            for position in sorted(buckets)]


def classify_drift(positions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Decide the SHAPE of PSNR-vs-GOP-position from the data.

    Two failure modes look similar in a single average but need opposite fixes,
    so this distinguishes them rather than assuming one:

      * ACCUMULATION - PSNR keeps falling along the P-chain. Each P-frame's
        quantization error compounds on an already-degraded reference. Fixed by
        periodic refresh, error feedback, or a shorter GOP.
      * FIXED PENALTY - PSNR drops once at the first P-frame and then stays
        flat. Error is NOT compounding; being a P-frame simply costs a fixed
        amount of quality. Fixed by better residual coding, not by GOP length.

    The distinction is decidable here because this coder always forms its
    residual against the TRUE current latent, so each frame re-targets z_t and
    previous error is corrected rather than carried - the flat shape is the
    prediction, and the measurement is what confirms it.
    """
    if len(positions) < 3 or positions[0]["gop_position"] != 0:
        return None
    i_psnr = positions[0]["mean_psnr_db"]
    p_positions = positions[1:]
    first_p, last_p = p_positions[0]["mean_psnr_db"], p_positions[-1]["mean_psnr_db"]
    p_chain_drop = last_p - first_p
    step = first_p - i_psnr

    values = [entry["mean_psnr_db"] for entry in p_positions]
    spread = max(values) - min(values)
    # "Monotone enough to call it accumulation" - the chain must lose clearly
    # more across its length than it wobbles.
    accumulating = p_chain_drop < 0 and abs(p_chain_drop) > spread * 0.5 and abs(p_chain_drop) > 0.5

    if accumulating:
        shape = "ACCUMULATION along the P-chain"
        interpretation = [
            "PSNR keeps falling with distance from the I-frame: quantization error",
            "is compounding on an already-degraded reference. A shorter GOP or a",
            "refresh mechanism would help; better residual coding alone would not.",
        ]
    else:
        shape = "FIXED per-P-frame penalty (no accumulation)"
        interpretation = [
            f"PSNR drops {abs(step):.2f} dB once at the first P-frame and then stays flat",
            f"({p_chain_drop:+.2f} dB across the whole chain, {spread:.2f} dB of wobble).",
            "Error is NOT compounding - the coder re-targets the true latent every",
            "frame, so each P-frame's error is corrected rather than carried. The cost",
            "is the residual quantization grid itself, so a SHORTER GOP WOULD NOT HELP;",
            "the fix is a better residual representation, which is what a learned",
            "temporal model would provide.",
        ]
    return {
        "shape": shape,
        "i_frame_psnr_db": i_psnr,
        "first_p_psnr_db": first_p,
        "last_p_psnr_db": last_p,
        "i_to_first_p_drop_db": step,
        "p_chain_drop_db": p_chain_drop,
        "p_chain_spread_db": spread,
        "accumulating": accumulating,
        "interpretation": interpretation,
    }


@torch.no_grad()
def run_arm(temporal, model, sequences, calibration, *, gop_size: int, reference_mode: str,
            stream_dir: Path, label: str) -> dict[str, Any]:
    """Code every sequence at one GOP size and account for every byte."""
    rows = []
    for sequence in sequences:
        frames = sequence.load_frames()
        path = stream_dir / f"{label}_{sequence.sequence_id}.nvct"
        encoded = temporal.encode_sequence(
            model, frames, path,
            intra_params=calibration["intra_params"],
            intra_entropy_model=calibration["intra_entropy_model"],
            residual_params=calibration["residual_params"],
            residual_entropy_model=calibration["residual_entropy_model"],
            gop_size=gop_size, reference_mode=reference_mode)
        decoded, decoded_latents = temporal.decode_sequence_with_models(
            model, path,
            intra_entropy_model=calibration["intra_entropy_model"],
            residual_entropy_model=calibration["residual_entropy_model"],
            reference_mode=reference_mode, return_latents=True)

        symmetric = torch.equal(encoded["encoder_latents"], decoded_latents)
        decoded_cpu = decoded.cpu()
        mse = torch.mean((decoded_cpu - frames) ** 2).item()
        psnr = float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)
        quality = float(msssim(decoded_cpu.clamp(0, 1), frames).mean())

        # Drift diagnostic: per-frame PSNR indexed by position within the GOP.
        # If quantization error accumulates along the P-chain, PSNR falls
        # monotonically with distance from the last I-frame - which is a
        # different problem from "the residual is expensive" and needs a
        # different fix, so the two must not be conflated.
        per_frame_psnr = []
        for index in range(frames.shape[0]):
            frame_mse = torch.mean((decoded_cpu[index] - frames[index]) ** 2).item()
            per_frame_psnr.append(
                float("inf") if frame_mse <= 0 else 10.0 * math.log10(1.0 / frame_mse))
        gop_positions = [index % gop_size for index in range(frames.shape[0])]

        total_bytes = encoded["container_bytes"]
        rows.append({
            "sequence": sequence.sequence_id,
            "frames": encoded["frame_count"],
            "i_frames": encoded["i_frames"], "p_frames": encoded["p_frames"],
            "i_frame_payload_bytes": encoded["i_frame_payload_bytes"],
            "p_frame_payload_bytes": encoded["p_frame_payload_bytes"],
            "payload_bytes": encoded["payload_bytes"],
            "container_bytes": total_bytes,
            "container_overhead_bytes": total_bytes - encoded["payload_bytes"],
            "total_pixels": sequence.total_pixels,
            "raw_rgb_bytes": sequence.raw_rgb_bytes(),
            "stream_bpp": total_bytes * 8 / sequence.total_pixels,
            "compression_ratio": sequence.raw_rgb_bytes() / total_bytes,
            "mean_psnr_db": psnr,
            "mean_msssim": quality,
            "encoder_decoder_latents_identical": symmetric,
            "per_frame_psnr_db": per_frame_psnr,
            "gop_positions": gop_positions,
        })

    total_bytes = sum(r["container_bytes"] for r in rows)
    total_pixels = sum(r["total_pixels"] for r in rows)
    total_raw = sum(r["raw_rgb_bytes"] for r in rows)
    return {
        "arm": label, "gop_size": gop_size, "reference_mode": reference_mode,
        "sequences": rows,
        "total_frames": sum(r["frames"] for r in rows),
        "total_i_frames": sum(r["i_frames"] for r in rows),
        "total_p_frames": sum(r["p_frames"] for r in rows),
        "total_i_frame_payload_bytes": sum(r["i_frame_payload_bytes"] for r in rows),
        "total_p_frame_payload_bytes": sum(r["p_frame_payload_bytes"] for r in rows),
        "total_payload_bytes": sum(r["payload_bytes"] for r in rows),
        "total_container_bytes": total_bytes,
        "total_container_overhead_bytes": sum(r["container_overhead_bytes"] for r in rows),
        "total_pixels": total_pixels,
        "stream_bpp": total_bytes * 8 / total_pixels,
        "compression_ratio": total_raw / total_bytes,
        "mean_psnr_db": statistics.fmean(r["mean_psnr_db"] for r in rows),
        "mean_msssim": statistics.fmean(r["mean_msssim"] for r in rows),
        "all_encoder_decoder_symmetric": all(r["encoder_decoder_latents_identical"] for r in rows),
        "psnr_by_gop_position": _psnr_by_gop_position(rows),
    }


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    temporal = _load_script("m10g_temporal_baseline")
    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    train_sequences = discover_sequences(args.manifest, split="train")
    test_sequences = discover_sequences(
        args.manifest, split="test", max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames_per_sequence)
    if not train_sequences or not test_sequences:
        print("[ERROR] no sequences discovered from the manifest", file=sys.stderr)
        return 1

    total_test_frames = sum(s.frame_count for s in test_sequences)
    print("=" * 112)
    print("M10G - INTRA-ONLY vs TEMPORAL BASELINE (DAVIS test split)")
    print("=" * 112)
    print(f"  checkpoint      : {args.checkpoint}")
    print(f"  selection       : best validation checkpoint (M10G Part A convention)")
    print(f"  frozen lambda   : {FROZEN_LAMBDA:.1e}")
    print(f"  quantization    : {args.bits}-bit / {args.mode}")
    print(f"  sequences       : {len(test_sequences)}   frames: {total_test_frames}")
    print(f"  GOP (arm B)     : {args.gop}   reference mode: {args.reference_mode}")

    calibration = temporal.calibrate_temporal_grids(
        model, train_sequences, bits=args.bits, mode=args.mode, gop_size=args.gop,
        max_frames=args.calibration_frames)
    provenance = calibration["provenance"]
    print(f"\n  calibrated on TRAIN split only: {provenance['intra_frames']} intra frames, "
          f"{provenance['residual_frames']} residual frames")

    stream_dir = args.output_dir / "benchmark_streams"
    arms = {}
    for label, gop in (("A_intra_only", 1), ("B_temporal", args.gop)):
        print(f"\n  running arm {label} (GOP={gop}) ...")
        arms[label] = run_arm(
            temporal, model, test_sequences, calibration,
            gop_size=gop, reference_mode=args.reference_mode,
            stream_dir=stream_dir, label=label)

    a, b = arms["A_intra_only"], arms["B_temporal"]
    print()
    print("=" * 112)
    print("TOTAL-STREAM ACCOUNTING (every byte in the container, I-frame overhead included)")
    print("=" * 112)
    print(f"{'':<28} {'A intra-only':>16} {'B temporal':>16} {'change':>12}")

    def line(name, key, fmt="{:>16,.0f}", pct=True):
        av, bv = a[key], b[key]
        change = ((bv - av) / av * 100) if (pct and av) else None
        print(f"{name:<28} {fmt.format(av):>16} {fmt.format(bv):>16} "
              f"{(f'{change:+.2f}%' if change is not None else '-'):>12}")

    line("frames", "total_frames", "{:>16,d}", pct=False)
    line("I-frames", "total_i_frames", "{:>16,d}", pct=False)
    line("P-frames", "total_p_frames", "{:>16,d}", pct=False)
    line("I-frame payload bytes", "total_i_frame_payload_bytes")
    line("P-frame payload bytes", "total_p_frame_payload_bytes")
    line("container overhead bytes", "total_container_overhead_bytes")
    line("TOTAL container bytes", "total_container_bytes")
    print(f"{'stream BPP':<28} {a['stream_bpp']:>16.4f} {b['stream_bpp']:>16.4f} "
          f"{(b['stream_bpp'] - a['stream_bpp']) / a['stream_bpp'] * 100:>+11.2f}%")
    print(f"{'compression ratio':<28} {a['compression_ratio']:>16.2f} "
          f"{b['compression_ratio']:>16.2f} "
          f"{(b['compression_ratio'] - a['compression_ratio']) / a['compression_ratio'] * 100:>+11.2f}%")
    print(f"{'mean PSNR dB':<28} {a['mean_psnr_db']:>16.3f} {b['mean_psnr_db']:>16.3f} "
          f"{b['mean_psnr_db'] - a['mean_psnr_db']:>+11.3f}")
    print(f"{'mean MS-SSIM':<28} {a['mean_msssim']:>16.4f} {b['mean_msssim']:>16.4f} "
          f"{b['mean_msssim'] - a['mean_msssim']:>+11.4f}")

    print()
    print("  Per-frame averages:")
    for label, arm in (("A", a), ("B", b)):
        i_per = arm["total_i_frame_payload_bytes"] / max(arm["total_i_frames"], 1)
        p_per = (arm["total_p_frame_payload_bytes"] / arm["total_p_frames"]
                 if arm["total_p_frames"] else None)
        print(f"    {label}: I = {i_per:>10,.0f} B/frame   "
              f"P = {(f'{p_per:,.0f} B/frame' if p_per else 'n/a'):>18}")

    print()
    print("=" * 112)
    print("PER-SEQUENCE (temporal arm vs intra-only, same frames)")
    print("=" * 112)
    print(f"{'sequence':<24} {'frames':>7} {'A BPP':>9} {'B BPP':>9} {'change':>9} "
          f"{'A PSNR':>8} {'B PSNR':>8} {'helps?':>7}")
    per_sequence = []
    helped = 0
    for row_a, row_b in zip(a["sequences"], b["sequences"]):
        change = (row_b["stream_bpp"] - row_a["stream_bpp"]) / row_a["stream_bpp"] * 100
        helps = change < 0
        helped += int(helps)
        per_sequence.append({
            "sequence": row_a["sequence"], "frames": row_a["frames"],
            "intra_bpp": row_a["stream_bpp"], "temporal_bpp": row_b["stream_bpp"],
            "bpp_change_percent": change,
            "intra_psnr": row_a["mean_psnr_db"], "temporal_psnr": row_b["mean_psnr_db"],
            "intra_msssim": row_a["mean_msssim"], "temporal_msssim": row_b["mean_msssim"],
            "temporal_helps": helps,
        })
        print(f"{row_a['sequence']:<24} {row_a['frames']:>7} {row_a['stream_bpp']:>9.4f} "
              f"{row_b['stream_bpp']:>9.4f} {change:>+8.2f}% {row_a['mean_psnr_db']:>8.2f} "
              f"{row_b['mean_psnr_db']:>8.2f} {'yes' if helps else 'no':>7}")
    print()
    print(f"  temporal coding reduced bitrate on {helped}/{len(per_sequence)} sequences")

    print()
    print("=" * 112)
    print("DRIFT DIAGNOSTIC - mean PSNR by distance from the last I-frame (temporal arm)")
    print("=" * 112)
    print(f"{'GOP position':<14} {'frames':>8} {'mean PSNR dB':>14}  (0 = I-frame)")
    for entry in b["psnr_by_gop_position"]:
        print(f"{entry['gop_position']:<14} {entry['frames']:>8} {entry['mean_psnr_db']:>14.3f}")
    positions = b["psnr_by_gop_position"]
    drift = classify_drift(positions)
    if drift:
        print()
        print(f"  shape   : {drift['shape']}")
        print(f"  I-frame : {drift['i_frame_psnr_db']:.3f} dB")
        print(f"  P-frames: {drift['first_p_psnr_db']:.3f} dB at the first P, "
              f"{drift['last_p_psnr_db']:.3f} dB at the last")
        print(f"  step at first P : {drift['i_to_first_p_drop_db']:+.3f} dB")
        print(f"  slope across Ps : {drift['p_chain_drop_db']:+.3f} dB "
              f"(first P -> last P)")
        for line in drift["interpretation"]:
            print(f"  {line}")

    print()
    print(f"  encoder/decoder reference symmetry: A={'PASS' if a['all_encoder_decoder_symmetric'] else 'FAIL'}"
          f"  B={'PASS' if b['all_encoder_decoder_symmetric'] else 'FAIL'}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "M10G temporal benchmark",
        "frozen_lambda": FROZEN_LAMBDA,
        "checkpoint": str(args.checkpoint),
        "checkpoint_selection": "best validation checkpoint (M10G Part A convention)",
        "quantization": {"bits": args.bits, "mode": args.mode},
        "calibration_provenance": provenance,
        "bpp_definition": (
            "total container bytes * 8 / total decoded pixels - every byte in the file, "
            "I-frame overhead included. P-frame payload alone is reported separately but "
            "is never the headline number."
        ),
        "arms": arms,
        "per_sequence_comparison": per_sequence,
        "drift_diagnostic": {
            "psnr_by_gop_position": b["psnr_by_gop_position"],
            "classification": classify_drift(b["psnr_by_gop_position"]),
            "note": (
                "Mean PSNR at each distance from the last I-frame, and the SHAPE that "
                "implies. Accumulation (a falling chain) and a fixed per-P-frame penalty "
                "(one step then flat) need opposite fixes, so they are distinguished from "
                "the data rather than assumed."
            ),
        },
        "sequences_helped_by_temporal": helped,
        "sequences_total": len(per_sequence),
    }
    path = args.output_dir / "temporal_benchmark.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
