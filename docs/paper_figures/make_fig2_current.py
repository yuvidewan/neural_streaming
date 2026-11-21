"""Fig. 2 for the updated paper: the current codec against x264/x265, DAVIS TEST and UVG 1080p.

Every point is read from a committed report, nothing is typed in:
- ours: outputs/p8_motion/{test,uvg}_motion.json (A2 wide-latent codec, bit-aware motion, GOP 30)
- DAVIS anchors: parity_classical_{gop30,444_gop30} (CRF 20+) plus the *_hq_gop30 runs (CRF 8-18)
- UVG anchors: uvg_classical_{420,444}_gop30
Quality is sequence-mean PSNR in RGB, the convention every BD-rate in the paper uses.

Run from the repo root:  python docs/paper_figures/make_fig2_current.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path("outputs/benchmarks")
OURS = {"DAVIS": Path("outputs/p8_motion/test_motion.json"), "UVG": Path("outputs/p8_motion/uvg_motion.json")}
ANCHORS = {
    "DAVIS": {"420": [B / "parity_classical_gop30/parity_classical_gop30.json",
                      B / "parity_classical_420_hq_gop30/parity.json"],
              "444": [B / "parity_classical_444_gop30/parity_classical_444_gop30.json",
                      B / "parity_classical_444_hq_gop30/parity.json"]},
    "UVG": {"420": [B / "uvg_classical_420_gop30/parity.json"],
            "444": [B / "uvg_classical_444_gop30/parity.json"]},
}
SERIES = [("h264", "420", "x264, 4:2:0", "#7d8896", "--"), ("h264_444", "444", "x264, 4:4:4", "#1f2933", "-"),
          ("h265_444", "444", "x265, 4:4:4", "#8a6fb0", "-")]


def points(path: Path) -> list[dict]:
    pts = json.loads(path.read_text(encoding="utf-8"))["points"]
    return list(pts.values()) if isinstance(pts, dict) else pts


def curve(pts, arm=None):
    return sorted((p["bpp"], p["quality"]["sequence_mean"]["psnr"]) for p in pts if arm is None or p["arm"] == arm)


def first_existing(paths):
    found = [p for p in paths if p.is_file()]
    if not found:
        raise SystemExit(f"none of {paths} exist")
    return found


plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8})
fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.7), dpi=300)
for ax, (name, ours_path), panel in zip(axes, OURS.items(), "ab"):
    ours = curve(points(ours_path))
    lo, hi = ours[0][1] - 1.5, ours[-1][1] + 1.0
    for arm, space, label, color, style in SERIES:
        pts = [p for f in first_existing(ANCHORS[name][space]) for p in points(f)]
        c = [(b, q) for b, q in curve(pts, arm) if lo <= q <= hi]
        ax.plot(*zip(*c), style, color=color, lw=1.1, marker="o", ms=2.2, label=label)
    ax.plot(*zip(*ours), "-", color="#0f766e", lw=1.8, marker="s", ms=3.2, label="Proposed")
    ax.set_xscale("log")
    ax.set_ylim(lo, hi)
    ax.set_xlabel("bits per pixel (log scale)")
    ax.set_ylabel("PSNR (dB)")
    res = "256×256, 719 frames" if name == "DAVIS" else "1920×1072, 672 frames"
    ax.set_title(f"({panel}) {name} test, {res}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
fig.tight_layout(pad=0.5)
out = Path("docs/paper_figures/fig2_current_rd.png")
fig.savefig(out, dpi=300, facecolor="white")
print("wrote", out)
