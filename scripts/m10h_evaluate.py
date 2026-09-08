"""M10H evaluation: does motion compensation pay for itself?

FOUR PATHS, TWO RATE POINTS, ONE CONTAINER
--------------------------------------------
    intra    every frame independent (GOP=1)                    rate-accounted
    prev     P-frames against x_hat_{t-1}          (M10G)       rate-accounted
    mc       P-frames against Warp(x_hat_{t-1})    (M10H)       rate-accounted
    oracle   P-frames against a dense-flow warp    DIAGNOSTIC   NOT accounted

All four run through the same `.nvct` v2 container, the same frozen model at
lambda = 3.0e-4, and the same calibration procedure, so container overhead is
accounted identically and the arms differ only in how the reference is formed.
`intra` is the same coder at GOP=1 rather than a different codec.

Two rate points come from the quantization bit depth (8-bit and 4-bit), the
knob the existing calibration and entropy-coding infrastructure already
supports at these exact settings. No lambda sweep; the intra operating point
stays frozen. Two points is enough for a BD-rate integration over the shared
quality range, which M10G could not do with one.

THE ORACLE IS QUARANTINED
--------------------------
Its dense float flow is never transmitted, so its "bitrate" omits the cost of
the very thing that makes it work. It is reported in its own clearly-marked
section, excluded from BD-rate, and never described as a compression result.
It exists to answer one question: if motion were free and dense, would motion
compensation fix the high-motion failure? That distinguishes a motion
ESTIMATION/REPRESENTATION bottleneck from a RESIDUAL-CODEC bottleneck.

Example usage (PowerShell, from the project root, with .venv activated):

    .venv\\Scripts\\python.exe scripts\\m10h_evaluate.py
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

DEFAULT_OUTPUT_DIR = Path("outputs/m10h_motion_compensation")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
FROZEN_LAMBDA = 3.0e-4
# 8-bit and 4-bit. 6-bit was measured and REJECTED as the second point: it
# moves rate ~28% but PSNR only 0.02-0.13 dB, so the two points form a
# near-vertical curve with no shared quality range for a BD-rate integration
# to span. 4-bit is where quantization actually costs quality (M10G intra:
# 29.63 dB at 8-bit vs 27.98 at 4-bit), which is what makes a curve.
RATE_POINTS = (8, 4)
# (arm label, coder mode, gop) - `intra` is the same coder at GOP=1.
ARMS = (
    ("intra", "prev", 1),
    ("prev", "prev", 10),
    ("mc", "mc", 10),
    ("oracle", "oracle", 10),
)
# M10G called these out as the motion-sensitive sequences.
WATCH = ("bmx-bumps", "drift-chicane", "schoolgirls", "gold-fish")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10H: intra vs previous-frame vs motion-compensated, two rate points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--quant-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--arms", nargs="+", default=[a[0] for a in ARMS])
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


@torch.no_grad()
def run_arm(mc, model, sequences, calibration, *, mode: str, gop: int, bits: int,
            block_size: int, search_range: int, stream_dir: Path, label: str) -> dict[str, Any]:
    rows = []
    for sequence in sequences:
        frames = sequence.load_frames()
        path = stream_dir / f"{label}_{bits}bit_{sequence.sequence_id}.nvct"
        encoded = mc.encode_sequence(
            model, frames, path,
            intra_params=calibration["intra_params"],
            intra_entropy_model=calibration["intra_entropy_model"],
            residual_params=calibration["residual_params"],
            residual_entropy_model=calibration["residual_entropy_model"],
            motion_entropy_model=calibration["motion_entropy_model"],
            mode=mode, gop_size=gop, block_size=block_size, search_range=search_range)

        symmetric = None
        if mc.is_rate_accounted(mode):
            _, decoded_latents = mc.decode_sequence(
                model, path,
                intra_entropy_model=calibration["intra_entropy_model"],
                residual_entropy_model=calibration["residual_entropy_model"],
                motion_entropy_model=calibration["motion_entropy_model"],
                return_latents=True)
            symmetric = torch.equal(encoded["encoder_latents"], decoded_latents)

        recon = encoded["encoder_reconstructions"]
        mse = torch.mean((recon - frames) ** 2).item()
        psnr = float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)
        quality = float(msssim(recon.clamp(0, 1), frames).mean())
        total_bytes = encoded["container_bytes"]
        rows.append({
            "sequence": sequence.sequence_id, "frames": encoded["frame_count"],
            "i_frames": encoded["i_frames"], "p_frames": encoded["p_frames"],
            "motion_bytes": encoded["motion_bytes"],
            "residual_bytes": encoded["residual_bytes"],
            "i_frame_residual_bytes": encoded["i_frame_residual_bytes"],
            "p_frame_residual_bytes": encoded["p_frame_residual_bytes"],
            "container_overhead_bytes": encoded["container_overhead_bytes"],
            "container_bytes": total_bytes,
            "total_pixels": sequence.total_pixels,
            "raw_rgb_bytes": sequence.raw_rgb_bytes(),
            "stream_bpp": total_bytes * 8 / sequence.total_pixels,
            "compression_ratio": sequence.raw_rgb_bytes() / total_bytes,
            "mean_psnr_db": psnr, "mean_msssim": quality,
            "encoder_decoder_latents_identical": symmetric,
        })

    total_bytes = sum(r["container_bytes"] for r in rows)
    total_pixels = sum(r["total_pixels"] for r in rows)
    accounted_total = sum(r["motion_bytes"] + r["residual_bytes"] + r["container_overhead_bytes"]
                          for r in rows)
    return {
        "arm": label, "mode": mode, "gop": gop, "bits": bits,
        "rate_accounted": mc.is_rate_accounted(mode),
        "sequences": rows,
        "total_frames": sum(r["frames"] for r in rows),
        "total_i_frames": sum(r["i_frames"] for r in rows),
        "total_p_frames": sum(r["p_frames"] for r in rows),
        "total_motion_bytes": sum(r["motion_bytes"] for r in rows),
        "total_residual_bytes": sum(r["residual_bytes"] for r in rows),
        "total_i_frame_residual_bytes": sum(r["i_frame_residual_bytes"] for r in rows),
        "total_p_frame_residual_bytes": sum(r["p_frame_residual_bytes"] for r in rows),
        "total_container_overhead_bytes": sum(r["container_overhead_bytes"] for r in rows),
        "total_container_bytes": total_bytes,
        # Every byte on disk must be one of motion, residual or overhead.
        "byte_accounting_closes": accounted_total == total_bytes,
        "total_pixels": total_pixels,
        "stream_bpp": total_bytes * 8 / total_pixels,
        "compression_ratio": sum(r["raw_rgb_bytes"] for r in rows) / total_bytes,
        "mean_psnr_db": statistics.fmean(r["mean_psnr_db"] for r in rows),
        "mean_msssim": statistics.fmean(r["mean_msssim"] for r in rows),
        "all_encoder_decoder_symmetric": (
            all(r["encoder_decoder_latents_identical"] for r in rows)
            if mc.is_rate_accounted(mode) else None),
    }


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    mc = _load_script("m10h_motion_compensation")
    m10e = _load_script("m10e_evaluate")  # BD-rate integration, reused not copied
    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    train_sequences = discover_sequences(args.manifest, split="train")
    test_sequences = discover_sequences(
        args.manifest, split="test", max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames_per_sequence)
    total_frames = sum(s.frame_count for s in test_sequences)

    print("=" * 118)
    print("M10H - MOTION-COMPENSATED TEMPORAL BASELINE (DAVIS test split)")
    print("=" * 118)
    print(f"  checkpoint  : {args.checkpoint}  (best-validation, M10G Part A convention)")
    print(f"  frozen lambda: {FROZEN_LAMBDA:.1e}   GOP {args.gop}")
    print(f"  motion      : {args.block_size}x{args.block_size} blocks, full search "
          f"+/-{args.search_range} px, integer-pel, SAD, entropy-coded into the stream")
    print(f"  rate points : {args.rate_points} bit  ({args.quant_mode})")
    print(f"  sequences   : {len(test_sequences)}   frames: {total_frames}")

    stream_dir = args.output_dir / "benchmark_streams"
    selected = [a for a in ARMS if a[0] in set(args.arms)]
    results: dict[tuple[str, int], dict[str, Any]] = {}
    calibrations: dict[tuple[str, int], dict[str, Any]] = {}

    for bits in args.rate_points:
        for label, mode, gop in selected:
            print(f"\n  calibrating + running {label:<7} @ {bits}-bit ...", flush=True)
            # The residual grid is always fitted at the P-frame GOP, even for the
            # intra arm (which codes at GOP=1 and never uses it): a GOP=1
            # calibration pass would contain no residuals at all, and forcing one
            # would change what the OTHER arms are calibrated against.
            calibration = mc.calibrate_grids(
                model, train_sequences, bits=bits, mode=args.quant_mode, gop_size=args.gop,
                block_size=args.block_size, search_range=args.search_range,
                reference_mode=mode, max_frames=args.calibration_frames)
            calibrations[(label, bits)] = calibration["provenance"]
            results[(label, bits)] = run_arm(
                mc, model, test_sequences, calibration, mode=mode, gop=gop, bits=bits,
                block_size=args.block_size, search_range=args.search_range,
                stream_dir=stream_dir, label=label)

    print()
    print("=" * 118)
    print("FULL BYTE ACCOUNTING (every byte on disk; motion is inside the total)")
    print("=" * 118)
    print(f"{'arm':<8} {'bits':>5} {'I bytes':>12} {'P resid':>12} {'motion':>11} "
          f"{'overhead':>9} {'TOTAL':>12} {'BPP':>8} {'PSNR':>8} {'MS-SSIM':>8} {'closes':>7}")
    for bits in args.rate_points:
        for label, _, _ in selected:
            arm = results.get((label, bits))
            if arm is None:
                continue
            tag = "" if arm["rate_accounted"] else "  <- NOT rate-accounted"
            print(f"{label:<8} {bits:>5} {arm['total_i_frame_residual_bytes']:>12,} "
                  f"{arm['total_p_frame_residual_bytes']:>12,} {arm['total_motion_bytes']:>11,} "
                  f"{arm['total_container_overhead_bytes']:>9,} "
                  f"{arm['total_container_bytes']:>12,} {arm['stream_bpp']:>8.4f} "
                  f"{arm['mean_psnr_db']:>8.3f} {arm['mean_msssim']:>8.4f} "
                  f"{'yes' if arm['byte_accounting_closes'] else 'NO':>7}{tag}")

    # --- motion cost share ---------------------------------------------------
    print()
    print("=" * 118)
    print("WHAT MOTION COSTS")
    print("=" * 118)
    motion_share = []
    for bits in args.rate_points:
        arm = results.get(("mc", bits))
        if arm is None:
            continue
        share = arm["total_motion_bytes"] / arm["total_container_bytes"] * 100
        per_p = arm["total_motion_bytes"] / max(arm["total_p_frames"], 1)
        prev = results.get(("prev", bits))
        residual_saved = (prev["total_p_frame_residual_bytes"]
                          - arm["total_p_frame_residual_bytes"]) if prev else None
        motion_share.append({
            "bits": bits, "motion_bytes": arm["total_motion_bytes"],
            "motion_share_percent": share, "motion_bytes_per_p_frame": per_p,
            "residual_bytes_saved_vs_prev": residual_saved,
            "net_gain_bytes": (residual_saved - arm["total_motion_bytes"])
            if residual_saved is not None else None,
        })
        print(f"  {bits}-bit: motion = {arm['total_motion_bytes']:,} B "
              f"({share:.2f}% of the stream, {per_p:.0f} B per P-frame)")
        if residual_saved is not None:
            net = residual_saved - arm["total_motion_bytes"]
            print(f"          residual saved vs prev-frame = {residual_saved:+,} B  "
                  f"-> NET {net:+,} B ({'motion pays for itself' if net > 0 else 'motion costs more than it saves'})")

    # --- BD-rate over the two rate points ------------------------------------
    print()
    print("=" * 118)
    print("BD-RATE over the two rate points (rate-accounted arms only)")
    print("=" * 118)

    def curve(label, metric="mean_psnr_db"):
        points = []
        for bits in args.rate_points:
            arm = results.get((label, bits))
            if arm is not None:
                points.append((arm["stream_bpp"], arm[metric]))
        return points

    bd_rows = []
    accounted = [label for label, mode, _ in selected if mc.is_rate_accounted(mode)]
    if "intra" in accounted:
        base = curve("intra")
        base_ms = curve("intra", "mean_msssim")
        print(f"{'arm':<10} {'vs intra (PSNR)':>18} {'vs intra (MS-SSIM)':>20}")
        for label in accounted:
            if label == "intra":
                continue
            psnr_bd = m10e._bd_rate_linear(base, curve(label))
            ms_bd = m10e._bd_rate_linear(base_ms, curve(label, "mean_msssim"))
            bd_rows.append({"arm": label, "bd_rate_psnr_vs_intra": psnr_bd,
                            "bd_rate_msssim_vs_intra": ms_bd})
            print(f"{label:<10} {(f'{psnr_bd:+.2f}%' if psnr_bd is not None else 'n/a'):>18} "
                  f"{(f'{ms_bd:+.2f}%' if ms_bd is not None else 'n/a'):>20}")
        if "prev" in accounted and "mc" in accounted:
            mc_vs_prev = m10e._bd_rate_linear(curve("prev"), curve("mc"))
            bd_rows.append({"arm": "mc_vs_prev", "bd_rate_psnr_vs_prev": mc_vs_prev})
            print(f"\n  mc vs prev (PSNR BD-rate): "
                  f"{(f'{mc_vs_prev:+.2f}%' if mc_vs_prev is not None else 'n/a')}")
        print("\n  Two rate points only - the BD integration spans the overlapping quality")
        print("  range of two points per curve, not a full RD sweep. Directional, not precise.")

    # --- per-sequence, focused on the motion-sensitive clips ------------------
    reference_bits = args.rate_points[0]
    print()
    print("=" * 118)
    print(f"PER-SEQUENCE at the {reference_bits}-bit rate point (BPP / PSNR), "
          "motion-sensitive clips marked *")
    print("=" * 118)
    per_sequence = []
    if all((label, reference_bits) in results for label, _, _ in selected):
        print(f"{'sequence':<16} " + " ".join(
            f"{label + ' BPP':>12} {label + ' PSNR':>12}" for label, _, _ in selected))
        for index, sequence in enumerate(test_sequences):
            mark = "*" if sequence.sequence_id in WATCH else " "
            cells, record = [], {"sequence": sequence.sequence_id,
                                 "motion_sensitive": sequence.sequence_id in WATCH}
            for label, _, _ in selected:
                row = results[(label, reference_bits)]["sequences"][index]
                cells.append(f"{row['stream_bpp']:>12.4f} {row['mean_psnr_db']:>12.2f}")
                record[f"{label}_bpp"] = row["stream_bpp"]
                record[f"{label}_psnr"] = row["mean_psnr_db"]
                record[f"{label}_motion_bytes"] = row["motion_bytes"]
            per_sequence.append(record)
            print(f"{mark}{sequence.sequence_id:<15} " + " ".join(cells))

        print()
        print("  mc vs prev, per sequence:")
        helped = 0
        for record in per_sequence:
            d_bpp = (record["mc_bpp"] - record["prev_bpp"]) / record["prev_bpp"] * 100
            d_psnr = record["mc_psnr"] - record["prev_psnr"]
            record["mc_vs_prev_bpp_percent"] = d_bpp
            record["mc_vs_prev_psnr_db"] = d_psnr
            better = d_psnr > 0.05 or (d_bpp < -0.5 and d_psnr > -0.05)
            record["mc_better_than_prev"] = better
            helped += int(better)
            mark = "*" if record["motion_sensitive"] else " "
            print(f"   {mark}{record['sequence']:<16} dBPP {d_bpp:>+7.2f}%  "
                  f"dPSNR {d_psnr:>+6.2f} dB  {'better' if better else ''}")
        print(f"\n  motion compensation improved {helped}/{len(per_sequence)} sequences vs prev-frame")

    # --- oracle, quarantined --------------------------------------------------
    oracle_block = {}
    if any(label == "oracle" for label, _, _ in selected):
        print()
        print("=" * 118)
        print("ORACLE DIAGNOSTIC - dense flow, NOT transmitted, NOT a compression result")
        print("=" * 118)
        for bits in args.rate_points:
            arm = results.get(("oracle", bits))
            prev = results.get(("prev", bits))
            mcarm = results.get(("mc", bits))
            if arm is None:
                continue
            print(f"  {bits}-bit  PSNR: prev {prev['mean_psnr_db']:.3f} -> "
                  f"mc {mcarm['mean_psnr_db']:.3f} -> oracle {arm['mean_psnr_db']:.3f} dB")
            oracle_block[f"{bits}bit"] = {
                "prev_psnr": prev["mean_psnr_db"], "mc_psnr": mcarm["mean_psnr_db"],
                "oracle_psnr": arm["mean_psnr_db"],
                "oracle_residual_bytes": arm["total_residual_bytes"],
                "mc_residual_bytes": mcarm["total_residual_bytes"],
                "note": "oracle bytes EXCLUDE its motion field, which is never transmitted",
            }
        print("\n  Read this as: how much of the gap could a better motion field close,")
        print("  if motion were free? It bounds motion estimation/representation, and")
        print("  says nothing about whether that motion could be affordably coded.")

    print()
    for bits in args.rate_points:
        for label, mode, _ in selected:
            arm = results.get((label, bits))
            if arm and arm["rate_accounted"]:
                assert arm["byte_accounting_closes"], f"{label}@{bits} byte accounting failed"
    print("  byte accounting closes for every rate-accounted arm (motion + residual + overhead = file size)")
    symmetry = {f"{label}@{bits}": results[(label, bits)]["all_encoder_decoder_symmetric"]
                for bits in args.rate_points for label, mode, _ in selected
                if (label, bits) in results and mc.is_rate_accounted(mode)}
    print(f"  encoder/decoder reference symmetry: "
          f"{'PASS' if all(symmetry.values()) else 'FAIL -> ' + str(symmetry)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "M10H motion-compensated temporal benchmark",
        "frozen_lambda": FROZEN_LAMBDA, "checkpoint": str(args.checkpoint),
        "gop": args.gop, "block_size": args.block_size, "search_range": args.search_range,
        "rate_points_bits": list(args.rate_points),
        "bpp_definition": ("total container bytes * 8 / total decoded pixels - motion, residual "
                           "and container overhead all included"),
        "motion_representation": {
            "estimator": "full-search block matching, SAD",
            "block_size": args.block_size, "search_range_px": args.search_range,
            "precision": "integer pixel (no interpolation)",
            "boundary": "replicate",
            "entropy_coding": "project arithmetic coder, 2 tables (dy, dx)",
            "transmitted": True,
        },
        "calibration_provenance": {f"{k[0]}@{k[1]}bit": v for k, v in calibrations.items()},
        "arms": {f"{k[0]}@{k[1]}bit": v for k, v in results.items()},
        "motion_cost": motion_share,
        "bd_rate": bd_rows,
        "per_sequence_at_reference_rate_point": per_sequence,
        "oracle_diagnostic": oracle_block,
        "encoder_decoder_symmetry": symmetry,
    }
    path = args.output_dir / "motion_benchmark.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
