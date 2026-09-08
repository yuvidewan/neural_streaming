"""M10J evaluation: does reference-conditioned entropy coding reduce real bytes?

The offline analysis measured a held-out, random-corrected conditional entropy
reduction of ~1.9-3.4% (see `m10j_entropy_analysis.py`). This script asks the
only question that settles it: how much of that survives the deployed arithmetic
coder.

The three temporal arms are coded in ONE closed-loop pass per sequence, so their
residual symbols, motion payloads and reconstructions are identical by
construction, not by assertion. `intra` is the same coder at GOP=1, carried
along for the RD comparison.

Reported side by side:

    H(R | channel)          the marginal model's theoretical cost
    H(R | channel, C)       the conditional model's
    actual residual bytes   what the arithmetic coder really emitted

The gap between the last two is the whole point: a theoretical entropy reduction
that does not show up in bytes is an entropy-coder bottleneck, and is classified
as such rather than reported as a win.

Example usage (PowerShell, from the project root, with .venv activated):

    .venv\\Scripts\\python.exe scripts\\m10j_evaluate.py
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

import numpy as np
import torch

from nvc.evaluation.perceptual_metrics import msssim
from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device

DEFAULT_OUTPUT_DIR = Path("outputs/m10j_conditional_entropy")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
FROZEN_LAMBDA = 3.0e-4
RATE_POINTS = (5, 4, 3)
WATCH = ("bmx-bumps", "drone", "cat-girl", "drift-chicane", "gold-fish", "schoolgirls")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10J: marginal vs conditional entropy coding on identical symbols.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--quant-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--table-frames", type=int, default=600)
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def _entropy_bits(counts: np.ndarray) -> float:
    smoothed = counts + 1.0
    probabilities = smoothed / smoothed.sum(axis=-1, keepdims=True)
    per_cell = -(probabilities * np.log2(probabilities)).sum(axis=-1)
    weights = counts.sum(axis=-1)
    total = weights.sum()
    return float((per_cell * weights).sum() / total) if total else 0.0


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    mc = _load_script("m10h_motion_compensation")
    ce = _load_script("m10j_conditional_entropy")
    m10e = _load_script("m10e_evaluate")
    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    train_sequences = discover_sequences(args.manifest, split="train")
    test_sequences = discover_sequences(
        args.manifest, split="test", max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames_per_sequence)
    arm_names = ["marginal"] + list(ce.DEPLOYED_CONTEXTS)

    print("=" * 124)
    print("M10J - CONDITIONAL ENTROPY CODING ON IDENTICAL RESIDUAL SYMBOLS (DAVIS test)")
    print("=" * 124)
    print(f"  frozen model : {args.checkpoint}   lambda {FROZEN_LAMBDA:.1e}")
    print(f"  arms         : intra (GOP=1), {', '.join(arm_names)}  (temporal, GOP={args.gop})")
    print(f"  rate points  : {args.rate_points} bit")
    print(f"  sequences    : {len(test_sequences)}  frames: "
          f"{sum(s.frame_count for s in test_sequences)}")
    print("  No training, no new weights, no checkpoint selection - this is an entropy-model")
    print("  experiment on the frozen M10H codec.")

    stream_dir = args.output_dir / "benchmark_streams"
    stream_dir.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[str, int], dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    entropy_rows: list[dict[str, Any]] = []
    per_sequence_rows: list[dict[str, Any]] = []

    for bits in args.rate_points:
        print(f"\n  fitting tables and coding at {bits}-bit ...", flush=True)
        calibration = mc.calibrate_grids(
            model, train_sequences, bits=bits, mode=args.quant_mode, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            reference_mode="mc", max_frames=args.calibration_frames)
        symbols, references = ce.collect_training_symbols(
            mc, model, train_sequences, calibration, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, device=device,
            max_frames=args.table_frames)

        arms = {}
        for scheme in arm_names:
            context_model = ce.fit_context_model(scheme, references)
            arms[scheme] = ce.build_conditional_entropy_model(
                symbols, references, context_model, bits=bits)
            provenance[f"{bits}bit_{scheme}"] = arms[scheme]["provenance"]

        # Theoretical cost of each model on the TRAIN symbols it was fitted to.
        channels = symbols[0].shape[0]
        alphabet = 2 ** bits
        for scheme in arm_names:
            context_model = arms[scheme]["context_model"]
            counts = np.zeros((channels, context_model.cardinality, alphabet), dtype=np.int64)
            for symbol_frame, reference in zip(symbols, references):
                contexts = context_model.contexts(reference)
                flat_s = symbol_frame.reshape(channels, -1)
                flat_c = contexts.reshape(channels, -1)
                for channel in range(channels):
                    index = flat_c[channel] * alphabet + flat_s[channel]
                    counts[channel] += np.bincount(
                        index, minlength=context_model.cardinality * alphabet
                    ).reshape(context_model.cardinality, alphabet)
            entropy_rows.append({"bits": bits, "arm": scheme,
                                 "entropy_bits_per_symbol": _entropy_bits(counts)})

        # intra control: the same coder at GOP=1
        intra_rows = []
        for sequence in test_sequences:
            frames = sequence.load_frames()
            path = stream_dir / f"intra_{bits}bit_{sequence.sequence_id}.nvct"
            encoded = mc.encode_sequence(
                model, frames, path,
                intra_params=calibration["intra_params"],
                intra_entropy_model=calibration["intra_entropy_model"],
                residual_params=calibration["residual_params"],
                residual_entropy_model=calibration["residual_entropy_model"],
                motion_entropy_model=calibration["motion_entropy_model"],
                mode="prev", gop_size=1, block_size=args.block_size,
                search_range=args.search_range)
            recon = encoded["encoder_reconstructions"]
            mse = torch.mean((recon - frames) ** 2).item()
            intra_rows.append({
                "sequence": sequence.sequence_id, "frames": encoded["frame_count"],
                "motion_bytes": 0, "residual_bytes": encoded["residual_bytes"],
                "p_frame_ideal_bits": 0.0,
                "i_frame_residual_bytes": encoded["i_frame_residual_bytes"],
                "p_frame_residual_bytes": 0,
                "container_overhead_bytes": encoded["container_overhead_bytes"],
                "container_bytes": encoded["container_bytes"],
                "total_pixels": sequence.total_pixels,
                "raw_rgb_bytes": sequence.raw_rgb_bytes(),
                "stream_bpp": encoded["container_bytes"] * 8 / sequence.total_pixels,
                "mean_psnr_db": 10.0 * math.log10(1.0 / mse),
                "mean_msssim": float(msssim(recon.clamp(0, 1), frames).mean()),
            })

        # temporal arms: one shared closed loop
        arm_rows = {arm: [] for arm in arm_names}
        symbols_identical, recon_identical, motion_identical = True, True, True
        for index, sequence in enumerate(test_sequences):
            frames = sequence.load_frames()
            paths = {arm: stream_dir / f"{arm}_{bits}bit_{sequence.sequence_id}.nvct"
                     for arm in arm_names}
            result = ce.encode_multi(
                mc, model, frames, arms, paths,
                intra_params=calibration["intra_params"],
                intra_entropy_model=calibration["intra_entropy_model"],
                residual_params=calibration["residual_params"],
                motion_entropy_model=calibration["motion_entropy_model"],
                gop_size=args.gop, block_size=args.block_size, search_range=args.search_range)

            recon = result["reconstructions"]
            mse = torch.mean((recon - frames) ** 2).item()
            psnr = 10.0 * math.log10(1.0 / mse)
            quality = float(msssim(recon.clamp(0, 1), frames).mean())
            motion_identical &= len({result["arms"][a]["motion_bytes"]
                                     for a in arm_names}) == 1

            record = {"sequence": sequence.sequence_id, "bits": bits}
            for arm in arm_names:
                stats = result["arms"][arm]
                decoded, decoded_symbols = ce.decode_sequence(
                    mc, model, paths[arm],
                    intra_entropy_model=calibration["intra_entropy_model"],
                    residual_entropy_model=arms[arm]["entropy_model"],
                    context_model=arms[arm]["context_model"],
                    motion_entropy_model=calibration["motion_entropy_model"],
                    return_symbols=True)
                symbols_identical &= all(
                    np.array_equal(a.reshape(-1), b.reshape(-1))
                    for a, b in zip(result["symbols"], decoded_symbols))
                recon_identical &= torch.equal(decoded.cpu(), recon)
                arm_rows[arm].append({
                    "sequence": sequence.sequence_id, "frames": result["frame_count"],
                    **{k: stats[k] for k in ("motion_bytes", "residual_bytes",
                                             "i_frame_residual_bytes",
                                             "p_frame_residual_bytes",
                                             "container_overhead_bytes", "container_bytes",
                                             "i_frames", "p_frames", "p_frame_ideal_bits")},
                    "total_pixels": sequence.total_pixels,
                    "raw_rgb_bytes": sequence.raw_rgb_bytes(),
                    "stream_bpp": stats["container_bytes"] * 8 / sequence.total_pixels,
                    "mean_psnr_db": psnr, "mean_msssim": quality,
                })
                record[f"{arm}_residual_bytes"] = stats["residual_bytes"]
                record[f"{arm}_bpp"] = stats["container_bytes"] * 8 / sequence.total_pixels
            record["psnr"] = psnr
            record["watched"] = sequence.sequence_id in WATCH
            per_sequence_rows.append(record)

        def aggregate(name, rows):
            total = sum(r["container_bytes"] for r in rows)
            pixels = sum(r["total_pixels"] for r in rows)
            p_frames = sum(r.get("p_frames", 0) for r in rows)
            return {
                "arm": name, "bits": bits, "sequences": rows,
                "total_motion_bytes": sum(r["motion_bytes"] for r in rows),
                "total_residual_bytes": sum(r["residual_bytes"] for r in rows),
                "total_i_frame_residual_bytes": sum(r["i_frame_residual_bytes"] for r in rows),
                "total_p_frame_residual_bytes": sum(r["p_frame_residual_bytes"] for r in rows),
                "total_container_overhead_bytes": sum(r["container_overhead_bytes"] for r in rows),
                "total_container_bytes": total, "total_pixels": pixels,
                "stream_bpp": total * 8 / pixels,
                "compression_ratio": sum(r["raw_rgb_bytes"] for r in rows) / total,
                "mean_psnr_db": statistics.fmean(r["mean_psnr_db"] for r in rows),
                "mean_msssim": statistics.fmean(r["mean_msssim"] for r in rows),
                "byte_accounting_closes": sum(
                    r["motion_bytes"] + r["residual_bytes"] + r["container_overhead_bytes"]
                    for r in rows) == total,
                "residual_bits_per_p_frame": (
                    sum(r["p_frame_residual_bytes"] for r in rows) * 8 / p_frames
                    if p_frames else 0.0),
                "p_frame_ideal_bits": sum(r.get("p_frame_ideal_bits", 0.0) for r in rows),
            }

        results[("intra", bits)] = aggregate("intra", intra_rows)
        for arm in arm_names:
            results[(arm, bits)] = aggregate(arm, arm_rows[arm])
        print(f"    symbols identical across arms: {symbols_identical} | "
              f"reconstruction identical: {recon_identical} | motion identical: {motion_identical}")
        if not (symbols_identical and recon_identical and motion_identical):
            print("[ERROR] the arms diverged; rate results are not interpretable",
                  file=sys.stderr)
            return 1

    print()
    print("=" * 124)
    print("FULL BYTE ACCOUNTING")
    print("=" * 124)
    print(f"{'arm':<17} {'bits':>5} {'I bytes':>12} {'P resid':>12} {'motion':>10} "
          f"{'TOTAL':>12} {'BPP':>8} {'PSNR':>8} {'MS-SSIM':>8} {'d resid vs marg':>16}")
    for bits in args.rate_points:
        base = results[("marginal", bits)]
        for arm in ["intra"] + arm_names:
            a = results[(arm, bits)]
            delta = ("" if arm in ("intra", "marginal") else
                     f"{(a['total_residual_bytes'] - base['total_residual_bytes']) / base['total_residual_bytes'] * 100:+15.2f}%")
            print(f"{arm:<17} {bits:>5} {a['total_i_frame_residual_bytes']:>12,} "
                  f"{a['total_p_frame_residual_bytes']:>12,} {a['total_motion_bytes']:>10,} "
                  f"{a['total_container_bytes']:>12,} {a['stream_bpp']:>8.4f} "
                  f"{a['mean_psnr_db']:>8.3f} {a['mean_msssim']:>8.4f} {delta:>16}")

    print()
    print("=" * 124)
    print("MODEL COST vs EMITTED BYTES - is the coder capturing the modelling gain?")
    print("=" * 124)
    print(f"{'bits':>5} {'arm':<17} {'ideal P bits':>16} {'vs marginal':>12} "
          f"{'P resid bytes':>15} {'vs marginal':>12} {'coder overhead':>15} {'realised':>10}")
    theory_rows = []
    for bits in args.rate_points:
        base = results[("marginal", bits)]
        base_ideal = base["p_frame_ideal_bits"]
        base_bytes = base["total_p_frame_residual_bytes"]
        for arm in arm_names:
            a = results[(arm, bits)]
            ideal_gain = (base_ideal - a["p_frame_ideal_bits"]) / base_ideal * 100
            byte_gain = (base_bytes - a["total_p_frame_residual_bytes"]) / base_bytes * 100
            overhead = (a["total_p_frame_residual_bytes"] * 8 - a["p_frame_ideal_bits"])                 / a["p_frame_ideal_bits"] * 100
            realised = byte_gain / ideal_gain * 100 if abs(ideal_gain) > 1e-9 else float("nan")
            theory_rows.append({
                "bits": bits, "arm": arm,
                "ideal_p_frame_bits": a["p_frame_ideal_bits"],
                "ideal_reduction_percent": ideal_gain,
                "p_frame_residual_bytes": a["total_p_frame_residual_bytes"],
                "byte_reduction_percent": byte_gain,
                "coder_overhead_percent": overhead,
                "realised_fraction_percent": realised})
            print(f"{bits:>5} {arm:<17} {a['p_frame_ideal_bits']:>16,.0f} "
                  f"{ideal_gain:>+11.2f}% {a['total_p_frame_residual_bytes']:>15,} "
                  f"{byte_gain:>+11.2f}% {overhead:>+14.2f}% "
                  f"{(f'{realised:.0f}%' if realised == realised else 'n/a'):>10}")
    print()
    print("  'ideal P bits' is the Shannon cost of the SAME coded symbols under each arm's own")
    print("  tables. 'realised' = deployed byte reduction / ideal-bit reduction; near 100% means")
    print("  the arithmetic coder captured the modelling gain, and well below it would indicate")
    print("  an entropy-coder bottleneck.")

    print()
    print("=" * 124)
    print("BD-RATE over three rate points")
    print("=" * 124)

    def curve(arm, metric="mean_psnr_db"):
        return [(results[(arm, b)]["stream_bpp"], results[(arm, b)][metric])
                for b in args.rate_points]

    bd_rows = []
    print(f"{'comparison':<34} {'PSNR BD-rate':>15} {'MS-SSIM BD-rate':>18}")
    for test, base in [(a, "marginal") for a in ce.DEPLOYED_CONTEXTS] + \
                      [(a, "intra") for a in arm_names]:
        psnr_bd = m10e._bd_rate_linear(curve(base), curve(test))
        ms_bd = m10e._bd_rate_linear(curve(base, "mean_msssim"), curve(test, "mean_msssim"))
        bd_rows.append({"test": test, "base": base, "bd_rate_psnr": psnr_bd,
                        "bd_rate_msssim": ms_bd})
        print(f"{test + ' vs ' + base:<34} "
              f"{(f'{psnr_bd:+.2f}%' if psnr_bd is not None else 'n/a'):>15} "
              f"{(f'{ms_bd:+.2f}%' if ms_bd is not None else 'n/a'):>18}")

    reference_bits = args.rate_points[1] if len(args.rate_points) > 1 else args.rate_points[0]
    print()
    print("=" * 124)
    print(f"PER-SEQUENCE residual bytes at {reference_bits}-bit (* = motion-sensitive)")
    print("=" * 124)
    print(f"{'sequence':<18} " + " ".join(f"{a:>18}" for a in arm_names) +
          f" {'best gain':>11}")
    for record in [r for r in per_sequence_rows if r["bits"] == reference_bits]:
        base = record["marginal_residual_bytes"]
        gains = {a: (record[f"{a}_residual_bytes"] - base) / base * 100 for a in arm_names}
        best = min(gains.values())
        record["residual_gain_percent"] = gains
        print(f"{'*' if record['watched'] else ' '}{record['sequence']:<17} " +
              " ".join(f"{record[f'{a}_residual_bytes']:>18,}" for a in arm_names) +
              f" {best:>+10.2f}%")

    closes = all(results[k]["byte_accounting_closes"] for k in results)
    print(f"\n  byte accounting closes everywhere: {closes}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "M10J conditional entropy benchmark",
        "frozen_lambda": FROZEN_LAMBDA, "checkpoint": str(args.checkpoint),
        "rate_points_bits": list(args.rate_points),
        "arms": {f"{k[0]}@{k[1]}bit": {kk: vv for kk, vv in v.items() if kk != "sequences"}
                 for k, v in results.items()},
        "per_sequence": {f"{k[0]}@{k[1]}bit": v["sequences"] for k, v in results.items()},
        "theory_vs_deployed": theory_rows,
        "entropy_bits": entropy_rows,
        "bd_rate": bd_rows,
        "per_sequence_summary": per_sequence_rows,
        "calibration_provenance": provenance,
        "invariants": {"symbols_identical_across_arms": True,
                       "reconstruction_identical_across_arms": True,
                       "motion_identical_across_arms": True,
                       "byte_accounting_closes": closes},
        "note": ("All temporal arms are produced in ONE closed-loop pass per sequence, so "
                 "identical symbols, motion and reconstruction are properties of the code "
                 "path rather than assertions checked afterwards."),
    }
    path = args.output_dir / "conditional_entropy_benchmark.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
