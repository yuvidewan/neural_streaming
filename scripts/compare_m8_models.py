"""Milestone 8 model comparison: overlay several benchmark_rd.py runs.

plot_rate_distortion.py groups curves by *codec* ("nvc"/"h264"/"h265")
within a single run directory - it has no notion of "model identity" across
several NVC checkpoints, since every M7-style run only ever benchmarked one
checkpoint. Comparing baseline vs. control vs. QAT means overlaying THREE
separate run directories (each produced by its own benchmark_rd.py
invocation) on one axis, keyed by model rather than by codec. That's a
different grouping key, not a bigger version of the same script, so this is
a small dedicated script rather than a modification of plot_rate_distortion.py
- see TESTING.md.

Reads each run's results.json (already real encode/decode measurements -
nothing here re-measures anything), builds:
  - comparison.csv / comparison.json: one row per (model, bit depth)
  - rd_psnr_vs_bpp.png, rd_msssim_vs_bpp.png: one curve per model

"Payload BPP" (entropy-coded bytes only, header excluded) is derived
arithmetically from the .nvc fixed header format (nvc_format.py:
FIXED_HEADER_SIZE + num_channels * 8 bytes/channel for per-channel mode),
not re-measured or estimated - the header layout is a project constant.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\compare_m8_models.py `
        --run baseline=outputs\\benchmarks\\m8_qat_close_out\\baseline `
        --run control=outputs\\benchmarks\\m8_qat_close_out\\control `
        --run qat=outputs\\benchmarks\\m8_qat_close_out\\qat `
        --output-dir outputs\\benchmarks\\m8_qat_close_out\\comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

from nvc.compression.nvc_format import FIXED_HEADER_SIZE

# struct.Struct("<ff") in nvc_format.py: one (scale, zero_point) float32 pair
# per channel in the header's quantization-parameter block.
_PARAM_BYTES_PER_CHANNEL = struct.calcsize("<ff")

_MODEL_STYLE = {
    # (matplotlib color, marker) - assigned in insertion order for any model
    # name not listed here, so this comparison isn't limited to 3 models.
}
_DEFAULT_COLORS = ["#1b6ca8", "#c0392b", "#27833f", "#8e44ad", "#d68910"]
_DEFAULT_MARKERS = ["o", "s", "^", "D", "v"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay several benchmark_rd.py NVC run directories, keyed by model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run", action="append", required=True, dest="runs", metavar="NAME=PATH",
        help="A model's benchmark_rd.py run directory as name=path. Repeat for each model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=150)
    return parser


def _parse_runs(raw: list[str]) -> dict[str, Path]:
    runs = {}
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"[ERROR] --run must be NAME=PATH, got: {item!r}")
        name, _, path = item.partition("=")
        runs[name] = Path(path)
    return runs


class _RunLoadError(Exception):
    """A run directory failed a runtime check - reported via [ERROR] + exit 1,
    not SystemExit, per this project's script contract (see TESTING.md)."""


def _load_run(name: str, run_dir: Path) -> dict:
    results_path = run_dir / "results.json"
    metadata_path = run_dir / "metadata.json"
    if not results_path.is_file():
        raise _RunLoadError(f"{name}: results.json not found at {results_path}")
    document = json.loads(results_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}

    calibration_path = metadata.get("calibration")
    latent_channels = None
    quant_mode = None
    if calibration_path and Path(calibration_path).is_file():
        calib_doc = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        calib_meta = calib_doc.get("calibration_metadata", {})
        latent_channels = calib_meta.get("latent_channels")
        quant_mode = calib_meta.get("mode")

    width = metadata.get("resolution", {}).get("width")
    height = metadata.get("resolution", {}).get("height")
    calibration_fit = metadata.get("calibration_fit", {})

    rows = []
    for row in document.get("aggregate", []):
        if row["codec"] != "nvc":
            continue
        bits = row.get("quantization_bits")
        payload_bpp = None
        payload_bytes = None
        if latent_channels and quant_mode == "per_channel" and width and height:
            params_bytes = latent_channels * _PARAM_BYTES_PER_CHANNEL
            header_bytes_per_frame = FIXED_HEADER_SIZE + params_bytes
            payload_bytes = row["total_bytes"] - header_bytes_per_frame * row["total_frames"]
            payload_bpp = payload_bytes * 8 / (row["total_frames"] * width * height)

        fit = calibration_fit.get(f"{bits}bit", {})
        rows.append({
            "model": name,
            "bits": bits,
            "sequences": row["sequences"],
            "total_frames": row["total_frames"],
            "total_bytes": row["total_bytes"],
            "mean_encoded_bytes_per_frame": row["total_bytes"] / row["total_frames"],
            "payload_bytes": payload_bytes,
            "aggregate_bpp": row["aggregate_bpp"],
            "payload_bpp": payload_bpp,
            "compression_ratio": row["compression_ratio"],
            "mean_psnr": row["mean_psnr"],
            "mean_msssim": row["mean_msssim"],
            "pooled_psnr": row["pooled_psnr"],
            "entropy_model_id": row.get("entropy_model_id"),
            "calibration_clipped_percent": fit.get("clipped_percent"),
            "calibration_clipped_low_percent": fit.get("clipped_low_percent"),
            "calibration_clipped_high_percent": fit.get("clipped_high_percent"),
            "calibration_fits": fit.get("fits"),
            "checkpoint": metadata.get("checkpoint"),
            "checkpoint_epoch": metadata.get("checkpoint_epoch"),
        })
    rows.sort(key=lambda r: r["bits"], reverse=True)
    return {"rows": rows, "run_dir": str(run_dir)}


def _write_table(all_rows: list[dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    csv_path = output_dir / "comparison.csv"

    json_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    fieldnames = list(all_rows[0].keys()) if all_rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    return json_path, csv_path


def _plot_metric(
    runs: dict[str, dict], metric_key: str, ylabel: str, title: str,
    output_path: Path, dpi: int,
) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(8, 5.5))
    plotted = False

    for index, (name, run) in enumerate(sorted(runs.items())):
        rows = [r for r in run["rows"] if r.get(metric_key) is not None]
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["aggregate_bpp"])
        plotted = True
        color = _DEFAULT_COLORS[index % len(_DEFAULT_COLORS)]
        marker = _DEFAULT_MARKERS[index % len(_DEFAULT_MARKERS)]
        axes.plot(
            [r["aggregate_bpp"] for r in rows], [r[metric_key] for r in rows],
            marker=marker, markersize=7, linewidth=1.8, color=color, label=name,
        )
        for row in rows:
            axes.annotate(
                f"{row['bits']}bit", (row["aggregate_bpp"], row[metric_key]),
                textcoords="offset points", xytext=(6, -10), fontsize=7.5, alpha=0.85,
            )

    if not plotted:
        plt.close(figure)
        return False

    axes.set_xlabel("Bits per pixel (total compressed .nvc file size)")
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

    run_paths = _parse_runs(args.runs)
    if len(run_paths) < 2:
        parser.error("Pass at least two --run NAME=PATH entries to compare.")

    runs = {}
    all_rows: list[dict] = []
    for name, path in run_paths.items():
        try:
            run = _load_run(name, path)
        except _RunLoadError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        if not run["rows"]:
            print(f"[ERROR] {name}: no NVC aggregate rows found in {path}", file=sys.stderr)
            return 1
        runs[name] = run
        all_rows.extend(run["rows"])

    json_path, csv_path = _write_table(all_rows, args.output_dir)

    written = []
    if _plot_metric(
        runs, "mean_psnr", "Mean PSNR (dB)",
        "Milestone 8: PSNR vs. BPP (baseline vs. control vs. QAT)",
        args.output_dir / "rd_psnr_vs_bpp.png", args.dpi,
    ):
        written.append(args.output_dir / "rd_psnr_vs_bpp.png")

    if _plot_metric(
        runs, "mean_msssim", "Mean MS-SSIM",
        "Milestone 8: MS-SSIM vs. BPP (baseline vs. control vs. QAT)",
        args.output_dir / "rd_msssim_vs_bpp.png", args.dpi,
    ):
        written.append(args.output_dir / "rd_msssim_vs_bpp.png")

    print(f"Compared {len(runs)} model(s): {', '.join(sorted(runs))}")
    print(f"Table:  {json_path}")
    print(f"        {csv_path}")
    for path in written:
        print(f"Plot:   {path}")
    if not written:
        print("[ERROR] Nothing could be plotted.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
