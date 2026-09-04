"""Milestone 9 final RD analysis: tables, deltas, plots, proxy-vs-actual.

Reads what the training and benchmark stages already wrote and produces the
comparison the milestone is judged on. It computes nothing new about the models
- it only joins, differences and plots existing measurements, so every number
here is traceable to `training_summary.json` or `benchmark_aggregate.json`.

TWO DIFFERENT QUANTITIES, NEVER CONFLATED
------------------------------------------
    R_proxy   the differentiable Laplace training estimate, bits per input
              pixel, from training. Never called bitrate.
    BPP       measured `.nvc` payload bytes x 8 / pixels, from real encode.

They are reported in separate tables. The one place they meet is the explicit
proxy-vs-actual section, which asks whether R_proxy *ranked* the models the
same way BPP did - a correlation question, not an equality claim.

WHY THE CONTROL MATTERS FOR EVERY DELTA
----------------------------------------
The M9 models were fine-tuned on DAVIS; the M8 QAT checkpoint they start from
was trained on Vimeo. So an M8 -> M9 delta mixes "the rate term worked" with
"30 more epochs on DAVIS helped". The lambda=0 control received identical
treatment minus the rate term, so `M9-x vs CTRL` isolates the rate term and
`CTRL vs M8` measures the fine-tuning confound on its own. Both are reported.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\m9_final_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_DIR = Path("outputs/m9_final")
BIT_DEPTHS = (8, 6, 4)
REPORT_ORDER = ("M7", "M8-QAT", "M9-CTRL", "M9-L", "M9-M", "M9-H")


def build_arg_parser(defaults=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Milestone 9 final RD analysis: tables, deltas and plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def _bits_of(row: dict[str, Any]) -> int | None:
    """Bit depth from the aggregate record's codec_configuration, e.g.
    'nvc/4bit-per_channel'."""
    configuration = str(row.get("codec_configuration", ""))
    for bits in BIT_DEPTHS:
        if f"{bits}bit" in configuration:
            return bits
    return None


def _index(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    table: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        bits = _bits_of(row)
        if bits is not None:
            table[(row["model"], bits)] = row
    return table


def _delta_table(
    measurements: dict[tuple[str, int], dict[str, Any]], baseline: str, targets: tuple[str, ...]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for target in targets:
        for bits in BIT_DEPTHS:
            base = measurements.get((baseline, bits))
            new = measurements.get((target, bits))
            if base is None or new is None:
                continue
            bpp_change = (new["aggregate_bpp"] - base["aggregate_bpp"]) / base["aggregate_bpp"] * 100
            out.append({
                "baseline": baseline,
                "model": target,
                "bits": bits,
                "baseline_bpp": base["aggregate_bpp"],
                "model_bpp": new["aggregate_bpp"],
                "bpp_change_percent": bpp_change,
                "baseline_psnr": base["mean_psnr"],
                "model_psnr": new["mean_psnr"],
                "psnr_delta_db": new["mean_psnr"] - base["mean_psnr"],
                "baseline_msssim": base["mean_msssim"],
                "model_msssim": new["mean_msssim"],
                "msssim_delta": new["mean_msssim"] - base["mean_msssim"],
            })
    return out


def _bd_rate(base: list[tuple[float, float]], test: list[tuple[float, float]]) -> float | None:
    """Bjontegaard delta-rate: average % bitrate change at equal quality.

    Negative is better (the same PSNR for fewer bits). The standard
    formulation: fit PSNR -> log10(bitrate) for each curve, integrate the
    difference over the overlapping PSNR range, and exponentiate the mean.

    With three operating points a quadratic is the right fit order (a cubic
    would interpolate exactly and integrate its own noise). Returns None when
    the two curves do not overlap in PSNR, where the metric is undefined
    rather than zero - that is the case for a model whose whole curve sits
    below the baseline's.
    """
    import numpy as np

    base_rate = np.log10([point[0] for point in base])
    base_psnr = np.array([point[1] for point in base])
    test_rate = np.log10([point[0] for point in test])
    test_psnr = np.array([point[1] for point in test])

    low = max(base_psnr.min(), test_psnr.min())
    high = min(base_psnr.max(), test_psnr.max())
    if high - low <= 1e-9:
        return None

    base_fit = np.polyfit(base_psnr, base_rate, 2)
    test_fit = np.polyfit(test_psnr, test_rate, 2)
    base_integral = np.polyval(np.polyint(base_fit), [low, high])
    test_integral = np.polyval(np.polyint(test_fit), [low, high])
    average = ((test_integral[1] - test_integral[0]) - (base_integral[1] - base_integral[0])) / (high - low)
    return float((10 ** average - 1) * 100)


def _bd_rate_linear(base: list[tuple[float, float]], test: list[tuple[float, float]]) -> float | None:
    """Piecewise-linear BD-rate over the overlapping PSNR range.

    Reported alongside `_bd_rate` because with only three operating points a
    quadratic is an EXACT fit (3 points, 3 coefficients) and does no smoothing,
    so it can swing a long way on curvature the data does not really pin down -
    on this project's own numbers the two disagree by more than 2x for one
    model. The linear estimate interpolates only between measured points and
    never extrapolates curvature, so it is the conservative floor. Where the
    two differ, the honest reading is "somewhere in this range, at least the
    linear value".
    """
    import numpy as np

    base_psnr = np.array([point[1] for point in base])
    base_rate = np.log10([point[0] for point in base])
    test_psnr = np.array([point[1] for point in test])
    test_rate = np.log10([point[0] for point in test])

    order = np.argsort(base_psnr)
    base_psnr, base_rate = base_psnr[order], base_rate[order]
    order = np.argsort(test_psnr)
    test_psnr, test_rate = test_psnr[order], test_rate[order]

    low = max(base_psnr.min(), test_psnr.min())
    high = min(base_psnr.max(), test_psnr.max())
    if high - low <= 1e-9:
        return None

    grid = np.linspace(low, high, 2000)
    difference = np.interp(grid, test_psnr, test_rate) - np.interp(grid, base_psnr, base_rate)
    return float((10 ** float(np.mean(difference)) - 1) * 100)


def _curve(measurements: dict[tuple[str, int], dict[str, Any]], model: str,
           metric: str) -> list[tuple[float, float]]:
    return [
        (measurements[(model, bits)]["aggregate_bpp"], measurements[(model, bits)][metric])
        for bits in BIT_DEPTHS if (model, bits) in measurements
    ]


def _write_plots(measurements: dict[tuple[str, int], dict[str, Any]], path_prefix: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    styles = {
        "M7": ("#7f7f7f", "o--"), "M8-QAT": ("#1f77b4", "s-"),
        "M9-CTRL": ("#2ca02c", "^:"), "M9-L": ("#ff7f0e", "d-"),
        "M9-M": ("#d62728", "v-"), "M9-H": ("#9467bd", "P-"),
    }
    for model in REPORT_ORDER:
        points = [measurements[(model, b)] for b in BIT_DEPTHS if (model, b) in measurements]
        if not points:
            continue
        colour, marker = styles.get(model, ("#333333", "o-"))
        bpp = [p["aggregate_bpp"] for p in points]
        axes[0].plot(bpp, [p["mean_psnr"] for p in points], marker, color=colour, label=model)
        axes[1].plot(bpp, [p["mean_msssim"] for p in points], marker, color=colour, label=model)

    axes[0].set_xlabel("measured .nvc BPP")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("M9 final: PSNR vs actual .nvc bitrate (DAVIS test)")
    axes[1].set_xlabel("measured .nvc BPP")
    axes[1].set_ylabel("MS-SSIM")
    axes[1].set_title("M9 final: MS-SSIM vs actual .nvc bitrate")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path_prefix.with_name(path_prefix.name + "_rd.png"), dpi=140)
    plt.close(figure)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    aggregate_path = args.output_dir / "benchmark_aggregate.json"
    training_path = args.output_dir / "training_summary.json"
    for label, path in (("benchmark_aggregate.json", aggregate_path),
                        ("training_summary.json", training_path)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    measurements = _index(aggregate["rows"])
    if not measurements:
        print("[ERROR] no NVC measurements found in benchmark_aggregate.json", file=sys.stderr)
        return 1

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 96)
    emit("M9 FINAL - ACTUAL .nvc BENCHMARK (DAVIS test split)")
    emit("=" * 96)
    emit(f"{'model':<9} {'bits':>5} {'BPP':>9} {'PSNR dB':>9} {'MS-SSIM':>9} "
         f"{'bytes/frame':>12} {'enc s/fr':>9} {'dec s/fr':>9}")
    for model in REPORT_ORDER:
        for bits in BIT_DEPTHS:
            row = measurements.get((model, bits))
            if row is None:
                continue
            emit(f"{model:<9} {bits:>5} {row['aggregate_bpp']:>9.4f} {row['mean_psnr']:>9.3f} "
                 f"{row['mean_msssim']:>9.4f} {row.get('bytes_per_frame', 0):>12.1f} "
                 f"{row.get('encode_seconds_per_frame', 0):>9.4f} "
                 f"{row.get('decode_seconds_per_frame', 0):>9.4f}")

    deltas_vs_m8 = _delta_table(measurements, "M8-QAT", ("M9-CTRL", "M9-L", "M9-M", "M9-H"))
    deltas_vs_m7 = _delta_table(measurements, "M7", ("M8-QAT", "M9-L", "M9-M", "M9-H"))
    deltas_vs_ctrl = _delta_table(measurements, "M9-CTRL", ("M9-L", "M9-M", "M9-H"))

    for title, table in (
        ("M8-QAT -> M9 (mixes the rate term with DAVIS fine-tuning)", deltas_vs_m8),
        ("M9 CONTROL -> M9 (isolates the rate term alone)", deltas_vs_ctrl),
        ("M7 -> M9", deltas_vs_m7),
    ):
        emit()
        emit("=" * 96)
        emit(title)
        emit("=" * 96)
        emit(f"{'model':<9} {'bits':>5} {'BPP':>9} {'vs base':>9} {'PSNR dB':>9} "
             f"{'dPSNR':>8} {'MS-SSIM':>9} {'dMS-SSIM':>10}")
        for row in table:
            emit(f"{row['model']:<9} {row['bits']:>5} {row['model_bpp']:>9.4f} "
                 f"{row['bpp_change_percent']:>+8.2f}% {row['model_psnr']:>9.3f} "
                 f"{row['psnr_delta_db']:>+8.3f} {row['model_msssim']:>9.4f} "
                 f"{row['msssim_delta']:>+10.4f}")

    # --- BD-rate ---------------------------------------------------------
    # The standard way to answer "lower BPP at comparable quality, or higher
    # quality at comparable BPP" with one number per curve pair, instead of
    # arguing from a single operating point.
    emit()
    emit("=" * 96)
    emit("BD-RATE vs baselines (negative = fewer bits for the same quality; whole-curve)")
    emit("=" * 96)
    bd_rows = []
    emit("Two fit orders reported: piecewise-linear (conservative, no extrapolated")
    emit("curvature) and quadratic (the exact 3-point fit). Trust the linear floor.")
    emit(f"{'model':<9} {'vs M8-QAT lin':>15} {'vs M8-QAT quad':>15} {'vs CTRL lin':>13} "
         f"{'MS-SSIM lin':>13}")
    for model in ("M9-CTRL", "M9-L", "M9-M", "M9-H"):
        if not _curve(measurements, model, "mean_psnr"):
            continue
        entry = {"model": model}
        for label, baseline, metric in (
            ("psnr_vs_m8_qat", "M8-QAT", "mean_psnr"),
            ("psnr_vs_control", "M9-CTRL", "mean_psnr"),
            ("msssim_vs_m8_qat", "M8-QAT", "mean_msssim"),
            ("psnr_vs_m7", "M7", "mean_psnr"),
        ):
            base_curve = _curve(measurements, baseline, metric)
            test_curve = _curve(measurements, model, metric)
            entry[f"bd_rate_{label}_linear"] = _bd_rate_linear(base_curve, test_curve)
            entry[f"bd_rate_{label}_quadratic"] = _bd_rate(base_curve, test_curve)
        bd_rows.append(entry)

        def fmt(value):
            return "n/a" if value is None else f"{value:+.2f}%"

        emit(f"{model:<9} {fmt(entry['bd_rate_psnr_vs_m8_qat_linear']):>15} "
             f"{fmt(entry['bd_rate_psnr_vs_m8_qat_quadratic']):>15} "
             f"{fmt(entry['bd_rate_psnr_vs_control_linear']):>13} "
             f"{fmt(entry['bd_rate_msssim_vs_m8_qat_linear']):>13}")

    # --- proxy vs actual ------------------------------------------------
    emit()
    emit("=" * 96)
    emit("PROXY vs ACTUAL - did the training-time R rank the models the way real BPP did?")
    emit("=" * 96)
    proxy_by_name = {
        {"M9-L": "M9-L", "M9-M": "M9-M", "M9-H": "M9-H", "CTRL": "M9-CTRL"}[arm["name"]]:
            arm["val_rate_bpp_proxy"]
        for arm in training["arms"]
    }
    emit(f"{'model':<9} {'R_proxy':>9} | {'BPP 8bit':>9} {'BPP 6bit':>9} {'BPP 4bit':>9}")
    proxy_rows = []
    for model in ("M9-CTRL", "M9-L", "M9-M", "M9-H"):
        if model not in proxy_by_name:
            continue
        bpps = {b: measurements[(model, b)]["aggregate_bpp"]
                for b in BIT_DEPTHS if (model, b) in measurements}
        proxy_rows.append({"model": model, "r_proxy": proxy_by_name[model], **{f"bpp_{b}bit": v for b, v in bpps.items()}})
        emit(f"{model:<9} {proxy_by_name[model]:>9.4f} | "
             + " ".join(f"{bpps.get(b, float('nan')):>9.4f}" for b in BIT_DEPTHS))

    correlations = {}
    if len(proxy_rows) >= 3:
        proxies = [row["r_proxy"] for row in proxy_rows]
        for bits in BIT_DEPTHS:
            key = f"bpp_{bits}bit"
            actuals = [row.get(key) for row in proxy_rows]
            if any(value is None for value in actuals):
                continue
            # Rank agreement over so few points is what matters here, not a
            # least-squares fit: the question is whether the proxy ORDERED the
            # models the way the real coder did.
            proxy_order = sorted(range(len(proxies)), key=lambda i: proxies[i])
            actual_order = sorted(range(len(actuals)), key=lambda i: actuals[i])
            correlations[f"{bits}bit_rank_agreement"] = proxy_order == actual_order
        emit()
        for key, value in correlations.items():
            emit(f"  proxy/actual {key}: {value}")

    report = {
        "phase": "M9 final RD analysis",
        "split": aggregate.get("split"),
        "note": (
            "aggregate_bpp is MEASURED .nvc payload. val_rate_bpp_proxy is the "
            "training-time Laplace estimate. They are different quantities; the "
            "proxy section asks only whether the proxy ranked models as BPP did."
        ),
        "measurements": [
            {"model": model, "bits": bits, **{
                k: v for k, v in measurements[(model, bits)].items()
                if not isinstance(v, (dict, list))
            }}
            for model in REPORT_ORDER for bits in BIT_DEPTHS if (model, bits) in measurements
        ],
        "deltas_vs_m8_qat": deltas_vs_m8,
        "deltas_vs_control": deltas_vs_ctrl,
        "deltas_vs_m7": deltas_vs_m7,
        "bd_rate": bd_rows,
        "proxy_vs_actual": {"rows": proxy_rows, "rank_agreement": correlations},
    }
    (args.output_dir / "rd_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

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

    (args.output_dir / "rd_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plotted = False
    if not args.no_plot:
        plotted = _write_plots(measurements, args.output_dir / "m9_final")

    emit()
    emit(f"Analysis: {args.output_dir / 'rd_analysis.json'}")
    if plotted:
        emit(f"Plots:    {args.output_dir / 'm9_final_rd.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
