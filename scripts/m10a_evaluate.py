"""M10A Phases 5-7: fresh calibration, real `.nvc` benchmark, proxy-vs-actual analysis.

REUSE, NOT REIMPLEMENTATION
----------------------------
The calibration and benchmark stages are `scripts/m9_final_calibrate_benchmark.py`'s
own `_calibrate` and `_benchmark`, imported and called with a different model
list. Those functions already take `models` as a parameter; only the model set
was hard-coded. So M10A runs through byte-identical calibration and benchmark
code paths to M9 - which is what makes the two milestones' numbers comparable
at all - and that file is not modified.

WHAT IS AND IS NOT RE-RUN
--------------------------
The four M10A arms are calibrated fresh (train split only, per model, per bit
depth) and benchmarked for real. M8-QAT and M9-L/M/H are NOT retrained or
re-benchmarked: their numbers were produced by these same scripts under the
same protocol in M9 and are read from `outputs/m9_final/benchmark_aggregate.json`.
Re-running them would burn GPU time to reproduce numbers already on disk.

THE CENTRAL COMPARISON
-----------------------
    proxy R   the differentiable Laplace estimate from training, measured
              against each arm's own tracked bin width
    BPP       measured `.nvc` payload bytes x 8 / pixels

M9 established these disagreed badly: proxy R spanned 3.7x across lambdas while
actual BPP moved 0.7% in the wrong direction. M10A asks whether scale tracking
fixes that. Rank agreement and correlation are reported per bit depth, with an
explicit note that n=4 is far too small for a strong statistical claim.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\m10a_evaluate.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

from nvc.utils.config import load_default_config

DEFAULT_OUTPUT_DIR = Path("outputs/m10a_pilot")
M9_AGGREGATE = Path("outputs/m9_final/benchmark_aggregate.json")
BIT_DEPTHS = (8, 6, 4)
M10A_ORDER = ("M10A-CTRL", "M10A-L", "M10A-M", "M10A-H")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_set(output_dir: Path) -> list[dict[str, Any]]:
    return [
        {"key": "M10A-CTRL", "label": "M10A control (lambda=0)",
         "checkpoint": output_dir / "control_lambda0" / "best.pt"},
        {"key": "M10A-L", "label": "M10A-L (lambda 9.0757e-04)",
         "checkpoint": output_dir / "lambda_9e-4" / "best.pt"},
        {"key": "M10A-M", "label": "M10A-M (lambda 2.8700e-03)",
         "checkpoint": output_dir / "lambda_2.87e-3" / "best.pt"},
        {"key": "M10A-H", "label": "M10A-H (lambda 9.0757e-03)",
         "checkpoint": output_dir / "lambda_9e-3" / "best.pt"},
    ]


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10A Phases 5-7: fresh calibration, real .nvc benchmark, proxy-vs-actual.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--m9-aggregate", type=Path, default=M9_AGGREGATE)
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


def _spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation, computed directly - scipy is not a project dependency."""
    n = len(a)
    if n < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
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


def _pearson(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 3:
        return None
    mean_a, mean_b = sum(a) / n, sum(b) / n
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b))
    return numerator / denominator if denominator > 0 else None


def _bits_of(row: dict[str, Any]) -> int | None:
    configuration = str(row.get("codec_configuration", ""))
    for bits in BIT_DEPTHS:
        if f"{bits}bit" in configuration:
            return bits
    return None


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
                    "note": (
                        "aggregate_bpp is MEASURED .nvc payload, not the training-time "
                        "Laplace proxy."
                    ),
                    "rows": result["rows"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if args.stage in ("analyse", "all"):
        aggregate_path = args.output_dir / "benchmark_aggregate.json"
        training_path = args.output_dir / "training_summary.json"
        for label, path in (("benchmark_aggregate.json", aggregate_path),
                            ("training_summary.json", training_path)):
            if not path.is_file():
                print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
                return 1

        rows = json.loads(aggregate_path.read_text(encoding="utf-8"))["rows"]
        training = json.loads(training_path.read_text(encoding="utf-8"))
        measurements = {
            (row["model"], _bits_of(row)): row for row in rows if _bits_of(row) is not None
        }
        proxy = {
            f"M10A-{arm['name']}" if arm["name"] == "CTRL" else arm["name"]: arm
            for arm in training["arms"]
        }

        lines: list[str] = []

        def emit(text: str = "") -> None:
            print(text)
            lines.append(text)

        emit("=" * 92)
        emit("M10A - ACTUAL .nvc BENCHMARK (DAVIS test split)")
        emit("=" * 92)
        emit(f"{'model':<11} {'bits':>5} {'BPP':>9} {'PSNR dB':>9} {'MS-SSIM':>9} "
             f"{'bytes/frame':>12} {'enc s/fr':>9} {'dec s/fr':>9}")
        for model in M10A_ORDER:
            for bits in BIT_DEPTHS:
                row = measurements.get((model, bits))
                if row is None:
                    continue
                emit(f"{model:<11} {bits:>5} {row['aggregate_bpp']:>9.4f} {row['mean_psnr']:>9.3f} "
                     f"{row['mean_msssim']:>9.4f} {row.get('bytes_per_frame', 0):>12.1f} "
                     f"{row.get('encode_seconds_per_frame', 0):>9.4f} "
                     f"{row.get('decode_seconds_per_frame', 0):>9.4f}")

        # --- A/B/C/D: rankings, agreement, correlation --------------------
        emit()
        emit("=" * 92)
        emit("PROXY vs ACTUAL")
        emit("=" * 92)
        names = [m for m in M10A_ORDER if m in proxy and (m, 8) in measurements]
        proxy_values = [proxy[m]["val_rate_bpp_proxy"] for m in names]
        emit(f"{'model':<11} {'proxy R':>9} | "
             + " ".join(f"{'BPP ' + str(b) + 'bit':>10}" for b in BIT_DEPTHS))
        for model in names:
            emit(f"{model:<11} {proxy[model]['val_rate_bpp_proxy']:>9.4f} | "
                 + " ".join(f"{measurements[(model, b)]['aggregate_bpp']:>10.4f}" for b in BIT_DEPTHS))

        proxy_rank = sorted(names, key=lambda m: proxy[m]["val_rate_bpp_proxy"])
        emit()
        emit(f"  proxy R ranking (low->high): {' < '.join(proxy_rank)}")
        analysis: dict[str, Any] = {"proxy_ranking": proxy_rank, "per_bit_depth": {}}
        for bits in BIT_DEPTHS:
            actual = [measurements[(m, bits)]["aggregate_bpp"] for m in names]
            actual_rank = sorted(names, key=lambda m: measurements[(m, bits)]["aggregate_bpp"])
            agree = actual_rank == proxy_rank
            pearson = _pearson(proxy_values, actual)
            spearman = _spearman(proxy_values, actual)
            analysis["per_bit_depth"][f"{bits}bit"] = {
                "actual_ranking": actual_rank,
                "rank_agreement_with_proxy": agree,
                "pearson": pearson,
                "spearman": spearman,
                "actual_bpp_spread_percent": (max(actual) - min(actual)) / min(actual) * 100,
            }
            emit(f"  {bits}-bit actual ranking      : {' < '.join(actual_rank)}"
                 f"   agree={agree}"
                 f"   pearson={pearson:+.3f}" if pearson is not None else "")
            if spearman is not None:
                emit(f"        spearman={spearman:+.3f}   "
                     f"actual BPP spread={analysis['per_bit_depth'][f'{bits}bit']['actual_bpp_spread_percent']:.2f}%")
        emit()
        emit("  n=4: correlations are indicative only, far too few points for a")
        emit("  statistically strong claim. Treat the rankings as the primary evidence.")

        # --- E: scale-normalisation diagnostic ---------------------------
        emit()
        emit("=" * 92)
        emit("SCALE-NORMALISATION DIAGNOSTIC - how much did each quantity actually move?")
        emit("=" * 92)
        emit(f"{'model':<11} {'latent |.|':>11} {'latent rng':>11} {'bin width':>10} "
             f"{'proxy R':>9} {'BPP 4bit':>10}")
        for model in names:
            arm = proxy[model]
            emit(f"{model:<11} {arm['final_latent_abs_mean']:>11.4f} "
                 f"{arm['final_latent_range']:>11.2f} {arm['final_bin_width']:>10.4f} "
                 f"{arm['val_rate_bpp_proxy']:>9.4f} "
                 f"{measurements[(model, 4)]['aggregate_bpp']:>10.4f}")

        def spread(values: list[float]) -> float:
            return max(values) / min(values) if min(values) > 0 else float("nan")

        spreads = {
            "latent_abs_mean": spread([proxy[m]["final_latent_abs_mean"] for m in names]),
            "bin_width": spread([proxy[m]["final_bin_width"] for m in names]),
            "proxy_R": spread(proxy_values),
            "actual_bpp_4bit": spread([measurements[(m, 4)]["aggregate_bpp"] for m in names]),
        }
        emit()
        for key, value in spreads.items():
            emit(f"  max/min spread, {key:<18}: {value:.3f}x")
        analysis["spreads"] = spreads

        # --- G: versus M9's frozen-bin-width behaviour -------------------
        if args.m9_aggregate.is_file():
            m9_rows = json.loads(args.m9_aggregate.read_text(encoding="utf-8"))["rows"]
            m9_measurements = {
                (row["model"], _bits_of(row)): row for row in m9_rows if _bits_of(row) is not None
            }
            m9_training = args.m9_aggregate.parent / "training_summary.json"
            m9_proxy = {}
            if m9_training.is_file():
                for arm in json.loads(m9_training.read_text(encoding="utf-8"))["arms"]:
                    key = "M9-CTRL" if arm["name"] == "CTRL" else arm["name"]
                    m9_proxy[key] = arm["val_rate_bpp_proxy"]

            emit()
            emit("=" * 92)
            emit("G. M9 (frozen bin width) vs M10A (scale-tracked) - same lambdas")
            emit("=" * 92)
            emit(f"{'lambda':<9} {'M9 proxy':>9} {'M9 BPP4':>9} | {'M10A proxy':>11} {'M10A BPP4':>10}")
            comparison = []
            for m9_key, m10_key, label in (
                ("M9-L", "M10A-L", "9.08e-4"), ("M9-M", "M10A-M", "2.87e-3"),
                ("M9-H", "M10A-H", "9.08e-3"),
            ):
                if (m9_key, 4) not in m9_measurements or (m10_key, 4) not in measurements:
                    continue
                entry = {
                    "lambda": label,
                    "m9_proxy": m9_proxy.get(m9_key),
                    "m9_bpp_4bit": m9_measurements[(m9_key, 4)]["aggregate_bpp"],
                    "m9_psnr_4bit": m9_measurements[(m9_key, 4)]["mean_psnr"],
                    "m10a_proxy": proxy[m10_key]["val_rate_bpp_proxy"],
                    "m10a_bpp_4bit": measurements[(m10_key, 4)]["aggregate_bpp"],
                    "m10a_psnr_4bit": measurements[(m10_key, 4)]["mean_psnr"],
                }
                comparison.append(entry)
                emit(f"{label:<9} {entry['m9_proxy']:>9.4f} {entry['m9_bpp_4bit']:>9.4f} | "
                     f"{entry['m10a_proxy']:>11.4f} {entry['m10a_bpp_4bit']:>10.4f}")
            if comparison:
                m9_spread = spread([e["m9_proxy"] for e in comparison])
                m10_spread = spread([e["m10a_proxy"] for e in comparison])
                emit()
                emit(f"  proxy R spread across the three lambdas: M9 {m9_spread:.2f}x  ->  "
                     f"M10A {m10_spread:.2f}x")
                emit("  A smaller proxy spread is the intended effect: under a frozen bin width")
                emit("  the proxy paid out for shrinkage the codec ignores; tracked, it does not.")
                analysis["m9_comparison"] = {
                    "rows": comparison,
                    "m9_proxy_spread": m9_spread,
                    "m10a_proxy_spread": m10_spread,
                }

        report = {
            "phase": "M10A Phases 5-7",
            "split": args.split,
            "note": (
                "aggregate_bpp is MEASURED .nvc payload. val_rate_bpp_proxy is the "
                "training-time Laplace estimate against each arm's own tracked bin width. "
                "They are different quantities; only their ORDERING is compared."
            ),
            "measurements": [
                {"model": m, "bits": b, **{
                    k: v for k, v in measurements[(m, b)].items() if not isinstance(v, (dict, list))
                }}
                for m in M10A_ORDER for b in BIT_DEPTHS if (m, b) in measurements
            ],
            "proxy_vs_actual": analysis,
        }
        (args.output_dir / "rd_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (args.output_dir / "rd_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with (args.output_dir / "rd_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["model", "bits", "aggregate_bpp", "mean_psnr", "mean_msssim",
                            "bytes_per_frame", "total_bytes", "total_frames",
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
