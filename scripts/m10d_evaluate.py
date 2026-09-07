"""M10D evaluation: fresh calibration, real `.nvc` benchmark, converged lambda comparison.

Decides whether any lambda near the M10C operating point improves the deployed
Pareto frontier over M10C-L, using the deployed codec as the authority. The
differentiable proxy is reported alongside and its ordering checked, but it
never selects the winner.

WHAT IS BENCHMARKED
--------------------
The final 18,120-step checkpoint of every arm. Earlier snapshots are retained
by the training script for the convergence record but are not benchmarked here -
M10C already established the shape of the budget curve, and M10D's question is
about lambda at convergence.

Calibration and benchmarking are `m9_final_calibrate_benchmark.py`'s own
`_calibrate`/`_benchmark`, the same functions M9, M10A, M10B and M10C used, so
the numbers stay comparable across milestones and that file is not modified.

M10C-L IS THE REFERENCE, AND ALSO A REPLICATE
----------------------------------------------
M10D's CENTER arm is configuration-identical to M10C-L. The difference between
their deployed results is therefore a direct measurement of run-to-run
nondeterminism on the final metric - the noise floor against which every other
M10D difference has to be judged. That comparison is computed explicitly rather
than left implicit, because without it a 1-2% BD-rate gap cannot be called real.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\m10d_evaluate.py
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

DEFAULT_OUTPUT_DIR = Path("outputs/m10d_lambda_refinement")
M10C_DIR = Path("outputs/m10c_convergence")
BIT_DEPTHS = (8, 6, 4)
REPORT_ORDER = ("CTRL", "LOW", "CENTER", "HIGH", "VERY_HIGH")
# The M10C-L final checkpoint, carried in for direct comparison.
M10C_REFERENCE = "M10C-L@18120"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def final_models(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """One benchmark model per arm: its final 18,120-step snapshot."""
    models = []
    for arm in summary["arms"]:
        final = arm["snapshots"][-1]
        models.append({
            "key": arm["name"],
            "label": f"{arm['name']} (lambda {arm['lambda']:.4e}) @ {final['step']} steps",
            "checkpoint": Path(final["path"]),
            "lambda": arm["lambda"], "step": final["step"],
            "checkpoint_sha256": final["sha256"],
        })
    return models


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10D: calibrate and benchmark the converged lambda arms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--m10c-dir", type=Path, default=M10C_DIR)
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
    """Piecewise-linear BD-rate over the overlapping PSNR range - the
    conservative methodology used since M9. None when the curves do not
    overlap, where the metric is undefined rather than zero."""
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


def _pearson(a, b):
    n = len(a)
    if n < 3:
        return None
    mean_a, mean_b = sum(a) / n, sum(b) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den = (sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)) ** 0.5
    return num / den if den > 0 else None


def _spearman(a, b):
    n = len(a)
    if n < 3:
        return None

    def ranks(values):
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    return _pearson(ranks(a), ranks(b))


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
              "scripts/m10d_lambda_refinement.py first.", file=sys.stderr)
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    m9 = _load_script("m9_final_calibrate_benchmark")
    models = final_models(summary)
    calibration_dir = args.output_dir / "calibration"
    benchmark_dir = args.output_dir / "benchmarks"

    if args.stage in ("calibrate", "all"):
        result = m9._calibrate(args, models, calibration_dir)
        # Calibration provenance, recorded so a later reader can verify which
        # checkpoint each grid was fitted to and on which split.
        for row in result["rows"]:
            model = next((m for m in models if m["key"] == row["model"]), None)
            if model is not None:
                row["checkpoint_sha256"] = model["checkpoint_sha256"]
                row["lambda"] = model["lambda"]
                row["training_steps"] = model["step"]
        (args.output_dir / "calibration_report.json").write_text(
            json.dumps({
                "method": "per_channel_percentile", "lower_percentile": 0.1,
                "upper_percentile": 99.9, "calibration_batches": args.calibration_batches,
                "calibration_frames": args.calibration_batches * args.batch_size,
                "batch_size": args.batch_size, "calibration_split": "train", "seed": args.seed,
                "clip_guard_percent": m9.CLIP_GUARD_PERCENT,
                "provenance_note": (
                    "Every grid is fitted to its OWN checkpoint on the TRAIN split only. "
                    "No calibration is shared between arms or reused from another milestone."
                ),
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

        measurements: dict[tuple[str, int], dict[str, Any]] = {}
        for row in json.loads(aggregate_path.read_text(encoding="utf-8"))["rows"]:
            bits = _bits_of(row)
            if bits is not None:
                measurements[(row["model"], bits)] = row

        # Carry in M10C-L's final row for the direct comparison / noise floor.
        m10c_aggregate = args.m10c_dir / "benchmark_aggregate.json"
        if m10c_aggregate.is_file():
            for row in json.loads(m10c_aggregate.read_text(encoding="utf-8"))["rows"]:
                bits = _bits_of(row)
                if bits is not None and row["model"] == M10C_REFERENCE:
                    measurements[("M10C-L", bits)] = row

        proxy = {arm["name"]: arm for arm in summary["arms"]}
        present = [m for m in REPORT_ORDER if (m, 8) in measurements]
        lines: list[str] = []

        def emit(text: str = "") -> None:
            print(text)
            lines.append(text)

        emit("=" * 104)
        emit("M10D - ACTUAL .nvc BENCHMARK AT CONVERGENCE (18,120 steps, DAVIS test, 719 frames)")
        emit("=" * 104)
        emit(f"{'arm':<10} {'lambda':>11} {'bits':>5} {'BPP':>9} {'PSNR dB':>9} {'MS-SSIM':>9} "
             f"{'bytes/frm':>10} {'ratio':>7}")
        for model in present:
            for bits in BIT_DEPTHS:
                row = measurements[(model, bits)]
                emit(f"{model:<10} {proxy[model]['lambda']:>11.4e} {bits:>5} "
                     f"{row['aggregate_bpp']:>9.4f} {row['mean_psnr']:>9.3f} "
                     f"{row['mean_msssim']:>9.4f} {row.get('bytes_per_frame', 0):>10.1f} "
                     f"{row.get('compression_ratio', 0):>7.2f}")

        emit()
        emit("=" * 104)
        emit("vs CTRL (same budget, same seed, only lambda differs)")
        emit("=" * 104)
        emit(f"{'arm':<10} {'bits':>5} {'BPP':>9} {'dBPP%':>9} {'PSNR':>9} {'dPSNR':>9} "
             f"{'MS-SSIM':>9} {'dMS-SSIM':>10}")
        deltas = []
        for model in present:
            if model == "CTRL":
                continue
            for bits in BIT_DEPTHS:
                row, base = measurements[(model, bits)], measurements[("CTRL", bits)]
                entry = {
                    "arm": model, "lambda": proxy[model]["lambda"], "bits": bits,
                    "bpp": row["aggregate_bpp"], "ctrl_bpp": base["aggregate_bpp"],
                    "bpp_change_percent": (row["aggregate_bpp"] - base["aggregate_bpp"])
                    / base["aggregate_bpp"] * 100,
                    "psnr": row["mean_psnr"], "psnr_delta_db": row["mean_psnr"] - base["mean_psnr"],
                    "msssim": row["mean_msssim"],
                    "msssim_delta": row["mean_msssim"] - base["mean_msssim"],
                }
                deltas.append(entry)
                emit(f"{model:<10} {bits:>5} {entry['bpp']:>9.4f} "
                     f"{entry['bpp_change_percent']:>+8.2f}% {entry['psnr']:>9.3f} "
                     f"{entry['psnr_delta_db']:>+9.3f} {entry['msssim']:>9.4f} "
                     f"{entry['msssim_delta']:>+10.4f}")

        def curve(model: str, metric: str = "mean_psnr"):
            return [(measurements[(model, b)]["aggregate_bpp"], measurements[(model, b)][metric])
                    for b in BIT_DEPTHS if (model, b) in measurements]

        emit()
        emit("=" * 104)
        emit("BD-RATE (piecewise-linear; negative = fewer bits at equal quality)")
        emit("=" * 104)
        emit(f"{'arm':<10} {'lambda':>11} {'vs CTRL (PSNR)':>16} {'vs CTRL (MS-SSIM)':>19} "
             f"{'vs M10C-L':>12}")
        bd_rows = []
        for model in present:
            if model == "CTRL":
                continue
            entry = {
                "arm": model, "lambda": proxy[model]["lambda"],
                "bd_rate_psnr_vs_ctrl": _bd_rate_linear(curve("CTRL"), curve(model)),
                "bd_rate_msssim_vs_ctrl": _bd_rate_linear(
                    curve("CTRL", "mean_msssim"), curve(model, "mean_msssim")),
                "bd_rate_psnr_vs_m10c_l": (
                    _bd_rate_linear(curve("M10C-L"), curve(model))
                    if ("M10C-L", 8) in measurements else None
                ),
            }
            bd_rows.append(entry)

            def fmt(v):
                return "n/a" if v is None else f"{v:+.2f}%"

            emit(f"{model:<10} {entry['lambda']:>11.4e} {fmt(entry['bd_rate_psnr_vs_ctrl']):>16} "
                 f"{fmt(entry['bd_rate_msssim_vs_ctrl']):>19} "
                 f"{fmt(entry['bd_rate_psnr_vs_m10c_l']):>12}")

        # --- the replicate / noise floor -------------------------------------
        replicate: dict[str, Any] = {}
        if ("M10C-L", 8) in measurements and ("CENTER", 8) in measurements:
            emit()
            emit("=" * 104)
            emit("NONDETERMINISM NOISE FLOOR - CENTER vs M10C-L (configuration-identical replicate)")
            emit("=" * 104)
            emit(f"{'bits':>5} {'M10C-L BPP':>12} {'CENTER BPP':>12} {'dBPP%':>9} "
                 f"{'M10C-L PSNR':>13} {'CENTER PSNR':>13} {'dPSNR':>9}")
            for bits in BIT_DEPTHS:
                reference, centre = measurements[("M10C-L", bits)], measurements[("CENTER", bits)]
                entry = {
                    "bits": bits,
                    "m10c_bpp": reference["aggregate_bpp"], "center_bpp": centre["aggregate_bpp"],
                    "bpp_change_percent": (centre["aggregate_bpp"] - reference["aggregate_bpp"])
                    / reference["aggregate_bpp"] * 100,
                    "m10c_psnr": reference["mean_psnr"], "center_psnr": centre["mean_psnr"],
                    "psnr_delta_db": centre["mean_psnr"] - reference["mean_psnr"],
                }
                replicate[f"{bits}bit"] = entry
                emit(f"{bits:>5} {entry['m10c_bpp']:>12.4f} {entry['center_bpp']:>12.4f} "
                     f"{entry['bpp_change_percent']:>+8.2f}% {entry['m10c_psnr']:>13.3f} "
                     f"{entry['center_psnr']:>13.3f} {entry['psnr_delta_db']:>+9.3f}")
            spread = max(abs(v["bpp_change_percent"]) for v in replicate.values())
            replicate["max_abs_bpp_change_percent"] = spread
            emit()
            emit(f"  Same lambda, same seed, same budget -> |dBPP| up to {spread:.2f}%.")
            emit("  Any M10D difference smaller than this cannot be attributed to lambda.")

        # --- proxy vs actual --------------------------------------------------
        emit()
        emit("=" * 104)
        emit("PROXY vs ACTUAL AT CONVERGENCE")
        emit("=" * 104)
        emit("  (the lambda=0 control is excluded: its estimator is unfitted by construction)")
        ranked = [m for m in present if proxy[m]["lambda"] > 0]
        proxy_values = [proxy[m]["final_val_rate_bpp_proxy"] for m in ranked]
        emit(f"{'arm':<10} {'proxy R':>9} | " + " ".join(f"{'BPP ' + str(b):>10}" for b in BIT_DEPTHS))
        for model in ranked:
            emit(f"{model:<10} {proxy[model]['final_val_rate_bpp_proxy']:>9.4f} | "
                 + " ".join(f"{measurements[(model, b)]['aggregate_bpp']:>10.4f}" for b in BIT_DEPTHS))

        proxy_rank = sorted(ranked, key=lambda m: proxy[m]["final_val_rate_bpp_proxy"])
        emit()
        emit(f"  proxy R ranking (low->high): {' < '.join(proxy_rank)}")
        per_depth = {}
        for bits in BIT_DEPTHS:
            actual = [measurements[(m, bits)]["aggregate_bpp"] for m in ranked]
            actual_rank = sorted(ranked, key=lambda m: measurements[(m, bits)]["aggregate_bpp"])
            per_depth[f"{bits}bit"] = {
                "actual_ranking": actual_rank,
                "rank_agreement_with_proxy": actual_rank == proxy_rank,
                "pearson": _pearson(proxy_values, actual),
                "spearman": _spearman(proxy_values, actual),
            }
            block = per_depth[f"{bits}bit"]
            emit(f"  {bits}-bit actual        : {' < '.join(actual_rank)}")
            emit(f"        agree={block['rank_agreement_with_proxy']}  "
                 f"spearman={block['spearman']:+.3f}  pearson={block['pearson']:+.3f}")
        emit()
        emit(f"  n={len(ranked)} rate-aware arms: correlations are indicative only, not a")
        emit("  statistically strong claim. The ordering is the usable signal.")

        # --- Pareto -----------------------------------------------------------
        emit()
        emit("=" * 104)
        emit("PARETO FRONTIER (M10D arms + M10C-L reference)")
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
            "phase": "M10D evaluation (converged lambda refinement)",
            "split": args.split,
            "note": (
                "aggregate_bpp is MEASURED .nvc payload. final_val_rate_bpp_proxy is the "
                "training-time Laplace estimate against each arm's own tracked bin width. "
                "Only their ordering is compared; the deployed measurement is authoritative."
            ),
            "m10c_reference": M10C_REFERENCE,
            "measurements": [
                {"model": key[0], "bits": key[1],
                 **{k: v for k, v in measurements[key].items() if not isinstance(v, (dict, list))}}
                for key in sorted(measurements, key=lambda k: (k[0], -k[1]))
            ],
            "deltas_vs_ctrl": deltas,
            "bd_rate": bd_rows,
            "nondeterminism_replicate": replicate,
            "proxy_vs_actual": {"proxy_ranking": proxy_rank, "per_bit_depth": per_depth},
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
