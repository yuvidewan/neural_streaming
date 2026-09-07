"""M10C evaluation: fresh calibration and real `.nvc` benchmarking of every snapshot.

Answers the one question M10C exists for: does M10A-L's -6.59% BD-rate survive
to convergence, or was it a pilot-only effect of the kind M9 Section 9F saw?

PAIRED BY TRAINING BUDGET
--------------------------
Every retained snapshot of both arms is calibrated and benchmarked
independently, and the control is only ever compared against the rate-aware arm
at the SAME step count. A partially-trained rate-aware model is never set
against a fully-trained control.

FRESH CALIBRATION PER SNAPSHOT
-------------------------------
Each snapshot gets its own train-split-only calibration at each bit depth -
never another snapshot's, never another arm's. That matters more here than
usual: the two arms' latents diverge in scale as training proceeds, so a shared
grid would quietly favour whichever model happened to match it. Calibration and
benchmarking are `m9_final_calibrate_benchmark.py`'s own `_calibrate`/
`_benchmark`, the same functions M9, M10A and M10B used, so the numbers stay
comparable across milestones and neither of those files is modified.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\m10c_evaluate.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from nvc.utils.config import load_default_config

DEFAULT_OUTPUT_DIR = Path("outputs/m10c_convergence")
BIT_DEPTHS = (8, 6, 4)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot_models(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """One benchmark model per (arm, snapshot). Keys carry arm and step so the
    pairing is unambiguous downstream and in every output file."""
    models = []
    for arm in summary["arms"]:
        for snapshot in arm["snapshots"]:
            models.append({
                "key": f"{arm['name']}@{snapshot['step']}",
                "label": f"{arm['name']} at {snapshot['step']} steps (lambda {arm['lambda']:.4e})",
                "checkpoint": Path(snapshot["path"]),
                "arm": arm["name"], "lambda": arm["lambda"], "step": snapshot["step"],
            })
    return models


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10C: calibrate and benchmark every retained snapshot, paired by budget.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage", choices=["calibrate", "benchmark", "analyse", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--calibration-batches", type=int, default=50)
    parser.add_argument("--mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow-clipping", action="store_true")
    return parser


def _bits_of(row: dict[str, Any]) -> int | None:
    configuration = str(row.get("codec_configuration", ""))
    for bits in BIT_DEPTHS:
        if f"{bits}bit" in configuration:
            return bits
    return None


def _bd_rate_linear(base, test):
    """Piecewise-linear BD-rate over the overlapping PSNR range.

    The conservative methodology settled on in M9/M10A: with three operating
    points a polynomial interpolates exactly and swings on curvature the data
    does not pin down. None when the curves do not overlap in PSNR.
    """
    import numpy as np

    base_psnr = np.array([p[1] for p in base]); base_rate = np.log10([p[0] for p in base])
    test_psnr = np.array([p[1] for p in test]); test_rate = np.log10([p[0] for p in test])
    order = np.argsort(base_psnr); base_psnr, base_rate = base_psnr[order], base_rate[order]
    order = np.argsort(test_psnr); test_psnr, test_rate = test_psnr[order], test_rate[order]
    low = max(base_psnr.min(), test_psnr.min()); high = min(base_psnr.max(), test_psnr.max())
    if high - low <= 1e-9:
        return None
    grid = np.linspace(low, high, 2000)
    return float((10 ** float(np.mean(
        np.interp(grid, test_psnr, test_rate) - np.interp(grid, base_psnr, base_rate)
    )) - 1) * 100)


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        print(f"[ERROR] --manifest not found: {args.manifest}", file=sys.stderr)
        return 1
    summary_path = args.output_dir / "training_summary.json"
    if not summary_path.is_file():
        print(f"[ERROR] training_summary.json not found: {summary_path}. Run "
              "scripts/m10c_convergence.py first.", file=sys.stderr)
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    m9 = _load_script("m9_final_calibrate_benchmark")
    models = snapshot_models(summary)
    calibration_dir = args.output_dir / "calibration"
    benchmark_dir = args.output_dir / "benchmarks"

    if args.stage in ("calibrate", "all"):
        result = m9._calibrate(args, models, calibration_dir)
        (args.output_dir / "calibration_report.json").write_text(
            json.dumps({
                "method": "per_channel_percentile", "lower_percentile": 0.1,
                "upper_percentile": 99.9, "calibration_batches": args.calibration_batches,
                "calibration_frames": args.calibration_batches * args.batch_size,
                "batch_size": args.batch_size, "calibration_split": "train", "seed": args.seed,
                "clip_guard_percent": m9.CLIP_GUARD_PERCENT,
                "rows": result["rows"], "failures": result.get("failures", []),
            }, indent=2),
            encoding="utf-8",
        )
        if not result["ok"] and not args.allow_clipping:
            print("\n[STOP] calibration failed the clipping guard:", file=sys.stderr)
            for failure in result.get("failures", []):
                print(f"  - {failure}", file=sys.stderr)
            return 1

    if args.stage in ("benchmark", "all"):
        result = m9._benchmark(args, models, calibration_dir, benchmark_dir)
        if not result["ok"]:
            return 1
        (args.output_dir / "benchmark_aggregate.json").write_text(
            json.dumps({
                "split": args.split, "bit_depths": list(BIT_DEPTHS),
                "note": "aggregate_bpp is MEASURED .nvc payload, not the training proxy.",
                "rows": result["rows"],
            }, indent=2),
            encoding="utf-8",
        )

    if args.stage in ("analyse", "all"):
        aggregate_path = args.output_dir / "benchmark_aggregate.json"
        if not aggregate_path.is_file():
            print(f"[ERROR] benchmark_aggregate.json not found: {aggregate_path}", file=sys.stderr)
            return 1
        rows = json.loads(aggregate_path.read_text(encoding="utf-8"))["rows"]
        measurements: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            bits = _bits_of(row)
            if bits is not None:
                measurements[(row["model"], bits)] = row

        steps = summary["snapshot_steps"]
        proxy = {
            (arm["name"], snapshot["step"]): snapshot
            for arm in summary["arms"] for snapshot in arm["snapshots"]
        }

        lines: list[str] = []

        def emit(text: str = "") -> None:
            print(text)
            lines.append(text)

        emit("=" * 104)
        emit("M10C - ACTUAL .nvc BENCHMARK BY TRAINING BUDGET (DAVIS test, 719 frames)")
        emit("=" * 104)
        emit(f"{'step':>7} {'arm':<8} {'bits':>5} {'BPP':>9} {'PSNR dB':>9} {'MS-SSIM':>9} "
             f"{'bytes/frm':>10} {'ratio':>7}")
        for step in steps:
            for arm in ("CTRL", "M10C-L"):
                for bits in BIT_DEPTHS:
                    row = measurements.get((f"{arm}@{step}", bits))
                    if row is None:
                        continue
                    emit(f"{step:>7} {arm:<8} {bits:>5} {row['aggregate_bpp']:>9.4f} "
                         f"{row['mean_psnr']:>9.3f} {row['mean_msssim']:>9.4f} "
                         f"{row.get('bytes_per_frame', 0):>10.1f} "
                         f"{row.get('compression_ratio', 0):>7.2f}")

        emit()
        emit("=" * 104)
        emit("PAIRED COMPARISON - M10C-L vs CTRL at the SAME training budget")
        emit("=" * 104)
        emit(f"{'step':>7} {'bits':>5} {'CTRL BPP':>10} {'L BPP':>10} {'dBPP%':>9} "
             f"{'CTRL PSNR':>10} {'L PSNR':>9} {'dPSNR':>9} {'dMS-SSIM':>10}")
        paired: list[dict[str, Any]] = []
        for step in steps:
            for bits in BIT_DEPTHS:
                control = measurements.get((f"CTRL@{step}", bits))
                rate = measurements.get((f"M10C-L@{step}", bits))
                if control is None or rate is None:
                    continue
                entry = {
                    "step": step, "bits": bits,
                    "ctrl_bpp": control["aggregate_bpp"], "rate_bpp": rate["aggregate_bpp"],
                    "bpp_change_percent": (rate["aggregate_bpp"] - control["aggregate_bpp"])
                    / control["aggregate_bpp"] * 100,
                    "ctrl_psnr": control["mean_psnr"], "rate_psnr": rate["mean_psnr"],
                    "psnr_delta_db": rate["mean_psnr"] - control["mean_psnr"],
                    "ctrl_msssim": control["mean_msssim"], "rate_msssim": rate["mean_msssim"],
                    "msssim_delta": rate["mean_msssim"] - control["mean_msssim"],
                }
                paired.append(entry)
                emit(f"{step:>7} {bits:>5} {entry['ctrl_bpp']:>10.4f} {entry['rate_bpp']:>10.4f} "
                     f"{entry['bpp_change_percent']:>+8.2f}% {entry['ctrl_psnr']:>10.3f} "
                     f"{entry['rate_psnr']:>9.3f} {entry['psnr_delta_db']:>+9.3f} "
                     f"{entry['msssim_delta']:>+10.4f}")

        emit()
        emit("=" * 104)
        emit("BD-RATE vs CTRL BY TRAINING BUDGET  (the M10C question)")
        emit("=" * 104)
        emit(f"{'step':>7} {'BD-rate (PSNR)':>16} {'BD-rate (MS-SSIM)':>19}   verdict at this budget")
        bd_by_step: list[dict[str, Any]] = []

        def curve(model: str, metric: str = "mean_psnr"):
            return [(measurements[(model, b)]["aggregate_bpp"], measurements[(model, b)][metric])
                    for b in BIT_DEPTHS if (model, b) in measurements]

        for step in steps:
            control_curve = curve(f"CTRL@{step}")
            rate_curve = curve(f"M10C-L@{step}")
            if len(control_curve) < 2 or len(rate_curve) < 2:
                continue
            bd_psnr = _bd_rate_linear(control_curve, rate_curve)
            bd_msssim = _bd_rate_linear(
                curve(f"CTRL@{step}", "mean_msssim"), curve(f"M10C-L@{step}", "mean_msssim"))
            verdict = ("better" if bd_psnr is not None and bd_psnr < -1.0 else
                       "worse" if bd_psnr is not None and bd_psnr > 1.0 else "no material difference")
            bd_by_step.append({"step": step, "bd_rate_psnr": bd_psnr,
                               "bd_rate_msssim": bd_msssim, "verdict": verdict})
            emit(f"{step:>7} {('n/a' if bd_psnr is None else f'{bd_psnr:+.2f}%'):>16} "
                 f"{('n/a' if bd_msssim is None else f'{bd_msssim:+.2f}%'):>19}   {verdict}")

        emit()
        emit("  M10A measured -6.59% at ~500 steps. Whether that number holds, shrinks or")
        emit("  reverses down this column IS the M10C result.")

        # --- proxy vs actual, by budget --------------------------------------
        emit()
        emit("=" * 104)
        emit("PROXY vs ACTUAL ALIGNMENT THROUGH CONVERGENCE")
        emit("=" * 104)
        emit("  (only the rate-aware arm: the control's estimator is unfitted by construction)")
        emit(f"{'step':>7} {'proxy R':>9} {'BPP 8':>9} {'BPP 6':>9} {'BPP 4':>9} "
             f"{'latent|.|':>10} {'bin width':>10}")
        alignment_rows = []
        for step in steps:
            snapshot = proxy.get(("M10C-L", step))
            if snapshot is None or (f"M10C-L@{step}", 8) not in measurements:
                continue
            entry = {
                "step": step, "proxy_R": snapshot["val_rate_bpp_proxy"],
                "latent_abs_mean": snapshot.get("latent_abs_mean"),
                "bin_width": snapshot.get("bin_width"),
                **{f"bpp_{b}bit": measurements[(f"M10C-L@{step}", b)]["aggregate_bpp"]
                   for b in BIT_DEPTHS},
            }
            alignment_rows.append(entry)
            emit(f"{step:>7} {entry['proxy_R']:>9.4f} {entry['bpp_8bit']:>9.4f} "
                 f"{entry['bpp_6bit']:>9.4f} {entry['bpp_4bit']:>9.4f} "
                 f"{(entry['latent_abs_mean'] or float('nan')):>10.4f} "
                 f"{(entry['bin_width'] or float('nan')):>10.4f}")

        # Does a falling proxy R still mean falling real BPP, at every budget?
        alignment = {}
        if len(alignment_rows) >= 3:
            proxies = [r["proxy_R"] for r in alignment_rows]
            for bits in BIT_DEPTHS:
                actual = [r[f"bpp_{bits}bit"] for r in alignment_rows]
                proxy_order = sorted(range(len(proxies)), key=lambda i: proxies[i])
                actual_order = sorted(range(len(actual)), key=lambda i: actual[i])
                alignment[f"{bits}bit"] = {
                    "rank_agreement_across_budgets": proxy_order == actual_order,
                }
            emit()
            for key, value in alignment.items():
                emit(f"  proxy/actual rank agreement across budgets, {key}: "
                     f"{value['rank_agreement_across_budgets']}")
            emit("  (this asks a different question from M10A/M10B: there the ranking was")
            emit("   across lambdas at one budget; here it is across budgets at one lambda.)")

        # --- Pareto ----------------------------------------------------------
        emit()
        emit("=" * 104)
        emit("PARETO FRONTIER over ALL snapshots and bit depths")
        emit("=" * 104)
        pareto: dict[str, list[str]] = {}
        for bits in BIT_DEPTHS:
            points = {
                key[0]: (measurements[key]["aggregate_bpp"], measurements[key]["mean_psnr"])
                for key in measurements if key[1] == bits
            }
            frontier = [
                model for model, (bpp, quality) in points.items()
                if not any(
                    other != model and other_bpp <= bpp and other_quality >= quality
                    and (other_bpp < bpp or other_quality > quality)
                    for other, (other_bpp, other_quality) in points.items()
                )
            ]
            pareto[f"{bits}bit"] = sorted(frontier)
            emit(f"  {bits}-bit non-dominated: {', '.join(sorted(frontier))}")

        report = {
            "phase": "M10C evaluation (RD by training budget)",
            "split": args.split, "snapshot_steps": steps,
            "note": (
                "aggregate_bpp is MEASURED .nvc payload. Comparisons are paired by "
                "training budget; a snapshot is only compared against the other arm's "
                "snapshot at the same step count."
            ),
            "measurements": [
                {"model": key[0], "bits": key[1],
                 **{k: v for k, v in measurements[key].items() if not isinstance(v, (dict, list))}}
                for key in sorted(measurements, key=lambda k: (k[0], -k[1]))
            ],
            "paired_comparison": paired,
            "bd_rate_by_step": bd_by_step,
            "proxy_vs_actual_by_step": {"rows": alignment_rows, "rank_agreement": alignment},
            "pareto_frontier": pareto,
        }
        (args.output_dir / "rd_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (args.output_dir / "rd_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with (args.output_dir / "rd_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["model", "bits", "aggregate_bpp", "mean_psnr", "mean_msssim",
                            "bytes_per_frame", "compression_ratio", "total_bytes", "total_frames",
                            "encode_seconds_per_frame", "decode_seconds_per_frame"],
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in report["measurements"]:
                writer.writerow(row)
        emit()
        emit(f"Analysis: {args.output_dir / 'rd_analysis.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
