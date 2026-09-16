"""Rate-distortion plot for the Stage 0 scoreboard (`benchmark_parity.py`).

Reads `parity.json` and draws PSNR and MS-SSIM against pooled BPP for every arm,
under the primary convention recorded in the report itself.

Run:
  ./.venv/Scripts/python.exe scripts/plot_parity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ARM_STYLE = {
    "h264": ("H.264 (default)", "#1f5fa8", "-", "o"),
    "h265": ("H.265 (default)", "#7a3fa0", "-", "s"),
    "h264_lowdelay": ("H.264 (I every 10, no B)", "#1f5fa8", "--", "o"),
    "h265_lowdelay": ("H.265 (I every 10, no B)", "#7a3fa0", "--", "s"),
    "nvc_deployed": ("NVC deployed", "#b4232b", "-", "D"),
    "nvc_m22": ("NVC + M22 grid", "#e07b22", "-", "D"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path,
                        default=Path("outputs/benchmarks/parity_s0/parity.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/benchmarks/parity_s0/parity_rd.png"))
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter

    report = json.loads(args.report.read_text(encoding="utf-8"))
    convention = report["configuration"]["primary_convention"]
    arms: dict[str, list[dict]] = {}
    for point in report["points"].values():
        arms.setdefault(point["arm"], []).append(point)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, metric, label in ((axes[0], "psnr", "PSNR (dB)"),
                                (axes[1], "msssim", "MS-SSIM")):
        for arm, (name, color, style, marker) in ARM_STYLE.items():
            points = sorted(arms.get(arm, []), key=lambda p: p["bpp"])
            if not points:
                continue
            axis.plot([p["bpp"] for p in points],
                      [p["quality"][convention][metric] for p in points],
                      linestyle=style, marker=marker, color=color, label=name,
                      markersize=4 if arm.startswith("h26") else 6, linewidth=1.4)
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5, 1.0]))
        axis.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel("bits per pixel (log)")
        axis.set_ylabel(label)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")
    dataset = report.get("dataset", {})
    figure.suptitle(f"DAVIS TEST, {len(dataset.get('sequences', []))} sequences, "
                    f"{dataset.get('frames')} frames, {convention} quality - single pass")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    print(f"Plot: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
