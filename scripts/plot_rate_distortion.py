"""Plot rate-distortion curves from a benchmark run.

Reads a run's results.json and draws PSNR-vs-BPP and MS-SSIM-vs-BPP curves
comparing every codec that was measured. Nothing is hardcoded: if a value
is not in results.json, it does not appear on the plot.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\plot_rate_distortion.py --run-dir outputs\\benchmarks\\<run>

Run `python scripts\\plot_rate_distortion.py --help` for the full option list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Distinct marker per codec so the curves stay readable in grayscale and for
# color-vision deficiency, rather than relying on hue alone.
_MARKERS = {"nvc": "o", "h264": "s", "h265": "^"}
_LABELS = {"nvc": "NVC (neural, intra-only)", "h264": "H.264", "h265": "H.265"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot rate-distortion curves from a benchmark run's results.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Benchmark run directory containing results.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write plots (default: <run-dir>/plots).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def _series_from_aggregate(aggregate: list[dict]) -> dict[str, list[dict]]:
    """Group aggregate rows by codec, sorted along the rate axis."""
    series: dict[str, list[dict]] = {}
    for row in aggregate:
        series.setdefault(row["codec"], []).append(row)
    for rows in series.values():
        rows.sort(key=lambda row: row["aggregate_bpp"])
    return series


def _plot_metric(
    series: dict[str, list[dict]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int,
) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(8, 5.5))
    plotted = False

    for codec, rows in sorted(series.items()):
        # Keep each row paired with its own point explicitly - zipping the
        # (unfiltered) `rows` against separately-filtered x/y lists would
        # shift every row after the first missing metric onto the wrong
        # point's label.
        points = [(r, r["aggregate_bpp"], r[metric_key]) for r in rows
                  if r.get(metric_key) is not None]
        if not points:
            continue
        plotted = True
        x_values = [point[1] for point in points]
        y_values = [point[2] for point in points]
        axes.plot(
            x_values, y_values,
            marker=_MARKERS.get(codec, "d"), markersize=6,
            linewidth=1.6, label=_LABELS.get(codec, codec),
        )
        for row, x_value, y_value in points:
            axes.annotate(
                row["codec_configuration"], (x_value, y_value),
                textcoords="offset points", xytext=(6, -10), fontsize=7.5, alpha=0.8,
            )

    if not plotted:
        plt.close(figure)
        return False

    axes.set_xlabel("Bits per pixel (total compressed file size)")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    results_path = args.run_dir / "results.json"
    if not results_path.is_file():
        print(f"[ERROR] results.json not found: {results_path}", file=sys.stderr)
        return 1

    document = json.loads(results_path.read_text(encoding="utf-8"))
    aggregate = document.get("aggregate", [])
    if not aggregate:
        print(f"[ERROR] {results_path} contains no aggregate results to plot.", file=sys.stderr)
        return 1

    output_dir = args.output_dir or (args.run_dir / "plots")
    series = _series_from_aggregate(aggregate)

    metadata = document.get("metadata", {})
    dataset = metadata.get("dataset", "dataset")
    frames = metadata.get("total_frames", "?")
    subtitle = f"{dataset}, {len(metadata.get('sequence_ids', []))} sequence(s), {frames} frames"

    written = []
    if _plot_metric(
        series, "mean_psnr", "Mean PSNR (dB)",
        f"Rate-distortion: PSNR vs. BPP\n{subtitle}",
        output_dir / "rd_psnr_vs_bpp.png", args.dpi,
    ):
        written.append(output_dir / "rd_psnr_vs_bpp.png")

    if _plot_metric(
        series, "mean_msssim", "Mean MS-SSIM",
        f"Rate-distortion: MS-SSIM vs. BPP\n{subtitle}",
        output_dir / "rd_msssim_vs_bpp.png", args.dpi,
    ):
        written.append(output_dir / "rd_msssim_vs_bpp.png")
    else:
        print("[WARN] No MS-SSIM values in results - skipping the MS-SSIM plot.")

    if not written:
        print("[ERROR] Nothing could be plotted from these results.", file=sys.stderr)
        return 1

    print("Plots written:")
    for path in written:
        print(f"  {path}")
    if metadata.get("methodology_note"):
        print("\nReminder: " + metadata["methodology_note"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
