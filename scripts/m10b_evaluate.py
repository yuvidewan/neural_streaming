"""M10B: fresh calibration, real `.nvc` benchmark, and the five-model RD comparison.

WHAT THIS ANSWERS
------------------
M10A found the useful rate-distortion region sits at or below its smallest
lambda (9.0757e-04, BD-rate -6.59% vs control; the two larger lambdas were
worse). M10B sweeps below that - 3e-4, 1e-4, 3e-5 - to find out whether the
frontier keeps improving or whether M10A-L is already near the optimum.

REUSE, NOT REIMPLEMENTATION
----------------------------
Calibration and benchmarking are `scripts/m9_final_calibrate_benchmark.py`'s
own `_calibrate`/`_benchmark`, imported and called with a different model list -
the same functions M10A used. That is what makes M9, M10A and M10B numbers
comparable at all, and neither of those files is modified.

WHAT IS AND IS NOT RE-RUN
--------------------------
Only the three new arms are trained, calibrated and benchmarked. The lambda=0
control and M10A-L are RETAINED from M10A: their checkpoints, calibrations and
benchmark rows already exist, were produced by these same scripts under this
same protocol, and re-running them would only add nondeterminism between two
copies of the same experiment. Their rows are read from
`outputs/m10a_pilot/`.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\m10b_evaluate.py
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

DEFAULT_OUTPUT_DIR = Path("outputs/m10b_pilot")
M10A_DIR = Path("outputs/m10a_pilot")
BIT_DEPTHS = (8, 6, 4)
# Report order: control, then descending lambda, so the sweep reads left to right.
REPORT_ORDER = ("CTRL", "M10A-L", "M10B-1", "M10B-2", "M10B-3")
RETAINED = {"CTRL": "M10A-CTRL", "M10A-L": "M10A-L"}


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_set(output_dir: Path) -> list[dict[str, Any]]:
    """Only the three NEW arms - the retained two are not recalibrated."""
    return [
        {"key": "M10B-1", "label": "M10B-1 (lambda 3e-04)",
         "checkpoint": output_dir / "lambda_3e-04" / "best.pt"},
        {"key": "M10B-2", "label": "M10B-2 (lambda 1e-04)",
         "checkpoint": output_dir / "lambda_1e-04" / "best.pt"},
        {"key": "M10B-3", "label": "M10B-3 (lambda 3e-05)",
         "checkpoint": output_dir / "lambda_3e-05" / "best.pt"},
    ]


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10B: fresh calibration, real .nvc benchmark, five-model RD comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--m10a-dir", type=Path, default=M10A_DIR)
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
    """Piecewise-linear BD-rate, the conservative methodology M10A settled on.

    Deliberately not a polynomial fit: with three operating points a quadratic
    interpolates exactly and swings on curvature the data does not pin down
    (M9 saw the two disagree by more than 2x). Returns None when the curves do
    not overlap in PSNR, where the metric is undefined rather than zero.
    """
    import numpy as np

    base_psnr = np.array([p[1] for p in base])
    base_rate = np.log10([p[0] for p in base])
    test_psnr = np.array([p[1] for p in test])
    test_rate = np.log10([p[0] for p in test])
    order = np.argsort(base_psnr); base_psnr, base_rate = base_psnr[order], base_rate[order]
    order = np.argsort(test_psnr); test_psnr, test_rate = test_psnr[order], test_rate[order]

    low = max(base_psnr.min(), test_psnr.min())
    high = min(base_psnr.max(), test_psnr.max())
    if high - low <= 1e-9:
        return None
    grid = np.linspace(low, high, 2000)
    difference = np.interp(grid, test_psnr, test_rate) - np.interp(grid, base_psnr, base_rate)
    return float((10 ** float(np.mean(difference)) - 1) * 100)


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

    m9 = _load_script("m9_final_calibrate_benchmark")
    models = _model_set(args.output_dir)
    calibration_dir = args.output_dir / "calibration"
    benchmark_dir = args.output_dir / "benchmarks"

    if args.stage in ("calibrate", "all"):
        result = m9._calibrate(args, models, calibration_dir)
        (args.output_dir / "calibration_report.json").write_text(
            json.dumps(
                {
                    "method": "per_channel_percentile",
                    "lower_percentile": 0.1, "upper_percentile": 99.9,
                    "calibration_batches": args.calibration_batches,
                    "calibration_frames": args.calibration_batches * args.batch_size,
                    "batch_size": args.batch_size, "calibration_split": "train",
                    "seed": args.seed, "clip_guard_percent": m9.CLIP_GUARD_PERCENT,
                    "rows": result["rows"], "failures": result.get("failures", []),
                },
                indent=2,
            ),
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
            json.dumps(
                {
                    "split": args.split, "bit_depths": list(BIT_DEPTHS),
                    "note": "aggregate_bpp is MEASURED .nvc payload, not the training proxy.",
                    "rows": result["rows"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if args.stage in ("analyse", "all"):
        new_path = args.output_dir / "benchmark_aggregate.json"
        m10a_path = args.m10a_dir / "benchmark_aggregate.json"
        for label, path in (("M10B benchmark_aggregate.json", new_path),
                            ("M10A benchmark_aggregate.json", m10a_path)):
            if not path.is_file():
                print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
                return 1

        measurements: dict[tuple[str, int], dict[str, Any]] = {}
        for row in json.loads(new_path.read_text(encoding="utf-8"))["rows"]:
            bits = _bits_of(row)
            if bits is not None:
                measurements[(row["model"], bits)] = row
        # Retained rows from M10A, renamed into this report's namespace.
        for row in json.loads(m10a_path.read_text(encoding="utf-8"))["rows"]:
            bits = _bits_of(row)
            for local, m10a_key in RETAINED.items():
                if bits is not None and row["model"] == m10a_key:
                    measurements[(local, bits)] = row

        proxy: dict[str, dict[str, Any]] = {}
        for arm in json.loads(
            (args.output_dir / "training_summary.json").read_text(encoding="utf-8")
        )["arms"]:
            proxy[arm["name"]] = arm
        for arm in json.loads(
            (args.m10a_dir / "training_summary.json").read_text(encoding="utf-8")
        )["arms"]:
            if arm["name"] in RETAINED:
                proxy[arm["name"]] = arm

        lines: list[str] = []

        def emit(text: str = "") -> None:
            print(text)
            lines.append(text)

        present = [m for m in REPORT_ORDER if (m, 8) in measurements and m in proxy]

        emit("=" * 100)
        emit("M10B - ACTUAL .nvc BENCHMARK (DAVIS test split, 719 frames)")
        emit("=" * 100)
        emit(f"{'model':<8} {'lambda':>10} {'bits':>5} {'BPP':>9} {'PSNR dB':>9} {'MS-SSIM':>9} "
             f"{'bytes/frm':>10} {'ratio':>7} {'enc s/fr':>9} {'dec s/fr':>9}")
        for model in present:
            for bits in BIT_DEPTHS:
                row = measurements[(model, bits)]
                emit(f"{model:<8} {proxy[model]['lambda']:>10.4e} {bits:>5} "
                     f"{row['aggregate_bpp']:>9.4f} {row['mean_psnr']:>9.3f} "
                     f"{row['mean_msssim']:>9.4f} {row.get('bytes_per_frame', 0):>10.1f} "
                     f"{row.get('compression_ratio', 0):>7.2f} "
                     f"{row.get('encode_seconds_per_frame', 0):>9.4f} "
                     f"{row.get('decode_seconds_per_frame', 0):>9.4f}")

        for baseline in ("CTRL", "M10A-L"):
            if baseline not in present:
                continue
            emit()
            emit("=" * 100)
            emit(f"vs {baseline}")
            emit("=" * 100)
            emit(f"{'model':<8} {'bits':>5} {'BPP':>9} {'dBPP%':>9} {'PSNR':>9} {'dPSNR':>9} "
                 f"{'MS-SSIM':>9} {'dMS-SSIM':>10}")
            for model in present:
                if model == baseline:
                    continue
                for bits in BIT_DEPTHS:
                    row, base = measurements[(model, bits)], measurements[(baseline, bits)]
                    emit(f"{model:<8} {bits:>5} {row['aggregate_bpp']:>9.4f} "
                         f"{(row['aggregate_bpp'] - base['aggregate_bpp']) / base['aggregate_bpp'] * 100:>+8.2f}% "
                         f"{row['mean_psnr']:>9.3f} {row['mean_psnr'] - base['mean_psnr']:>+9.3f} "
                         f"{row['mean_msssim']:>9.4f} {row['mean_msssim'] - base['mean_msssim']:>+10.4f}")

        def curve(model: str, metric: str = "mean_psnr"):
            return [(measurements[(model, b)]["aggregate_bpp"], measurements[(model, b)][metric])
                    for b in BIT_DEPTHS if (model, b) in measurements]

        emit()
        emit("=" * 100)
        emit("BD-RATE (piecewise-linear; negative = fewer bits at equal quality)")
        emit("=" * 100)
        emit(f"{'model':<8} {'vs CTRL (PSNR)':>16} {'vs M10A-L (PSNR)':>18} {'vs CTRL (MS-SSIM)':>19}")
        bd_rows = []
        for model in present:
            if model == "CTRL":
                continue
            entry = {
                "model": model, "lambda": proxy[model]["lambda"],
                "bd_rate_psnr_vs_ctrl": _bd_rate_linear(curve("CTRL"), curve(model)),
                "bd_rate_psnr_vs_m10a_l": (
                    _bd_rate_linear(curve("M10A-L"), curve(model)) if model != "M10A-L" else 0.0
                ),
                "bd_rate_msssim_vs_ctrl": _bd_rate_linear(
                    curve("CTRL", "mean_msssim"), curve(model, "mean_msssim")),
            }
            bd_rows.append(entry)

            def fmt(v):
                return "n/a" if v is None else f"{v:+.2f}%"

            emit(f"{model:<8} {fmt(entry['bd_rate_psnr_vs_ctrl']):>16} "
                 f"{fmt(entry['bd_rate_psnr_vs_m10a_l']):>18} "
                 f"{fmt(entry['bd_rate_msssim_vs_ctrl']):>19}")

        # --- proxy vs actual ------------------------------------------------
        emit()
        emit("=" * 100)
        emit("PROXY vs ACTUAL")
        emit("=" * 100)
        # The control's estimator never trains (0.0 * rate has zero gradient), so
        # its proxy R is an unfitted-prior value and must not enter a ranking.
        ranked = [m for m in present if proxy[m]["lambda"] > 0]
        proxy_values = [proxy[m]["val_rate_bpp_proxy"] for m in ranked]
        emit("  (the lambda=0 control is excluded: its estimator is unfitted by construction)")
        emit(f"{'model':<8} {'proxy R':>9} | " + " ".join(f"{'BPP ' + str(b):>10}" for b in BIT_DEPTHS))
        for model in ranked:
            emit(f"{model:<8} {proxy[model]['val_rate_bpp_proxy']:>9.4f} | "
                 + " ".join(f"{measurements[(model, b)]['aggregate_bpp']:>10.4f}" for b in BIT_DEPTHS))

        proxy_rank = sorted(ranked, key=lambda m: proxy[m]["val_rate_bpp_proxy"])
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
                "actual_bpp_spread_percent": (max(actual) - min(actual)) / min(actual) * 100,
            }
            block = per_depth[f"{bits}bit"]
            emit(f"  {bits}-bit actual            : {' < '.join(actual_rank)}")
            emit(f"        agree={block['rank_agreement_with_proxy']}  "
                 f"spearman={block['spearman']:+.3f}  pearson={block['pearson']:+.3f}  "
                 f"spread={block['actual_bpp_spread_percent']:.2f}%")
        emit()
        emit(f"  n={len(ranked)}: correlations are indicative only.")

        # --- Pareto frontier -------------------------------------------------
        emit()
        emit("=" * 100)
        emit("PARETO FRONTIER - is any new point non-dominated?")
        emit("=" * 100)
        pareto: dict[str, Any] = {}
        for bits in BIT_DEPTHS:
            points = {m: (measurements[(m, bits)]["aggregate_bpp"],
                          measurements[(m, bits)]["mean_psnr"]) for m in present}
            frontier = []
            for model, (bpp, quality) in points.items():
                dominated_by = [
                    other for other, (other_bpp, other_quality) in points.items()
                    if other != model and other_bpp <= bpp and other_quality >= quality
                    and (other_bpp < bpp or other_quality > quality)
                ]
                if not dominated_by:
                    frontier.append(model)
            pareto[f"{bits}bit"] = sorted(frontier)
            emit(f"  {bits}-bit non-dominated: {', '.join(sorted(frontier))}")

        report = {
            "phase": "M10B (low-lambda scale-tracked pilot)",
            "split": args.split,
            "note": (
                "aggregate_bpp is MEASURED .nvc payload. val_rate_bpp_proxy is the "
                "training-time Laplace estimate against each arm's own tracked bin "
                "width. Only their ordering is compared."
            ),
            "retained_from_m10a": list(RETAINED),
            "measurements": [
                {"model": m, "bits": b, "lambda": proxy[m]["lambda"],
                 **{k: v for k, v in measurements[(m, b)].items() if not isinstance(v, (dict, list))}}
                for m in present for b in BIT_DEPTHS if (m, b) in measurements
            ],
            "bd_rate": bd_rows,
            "proxy_vs_actual": {"proxy_ranking": proxy_rank, "per_bit_depth": per_depth},
            "pareto_frontier": pareto,
        }
        (args.output_dir / "rd_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (args.output_dir / "rd_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with (args.output_dir / "rd_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["model", "lambda", "bits", "aggregate_bpp", "mean_psnr", "mean_msssim",
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
