"""Figures for the paper (2026-11-26 revision): architecture and RD curves.

    figs.py <figs_dir>

Fig. 1 is the current codec. Fig. 2 reads the committed result files: our MSE model set
(outputs/p13_audit_fixes for DAVIS and UVG, outputs/p11_mcljcv for MCL-JCV with the unused
1,024-byte header removed, which docs/PLAN_audit2_fixes.md proves exact) and the x264/x265
anchors (DAVIS at 30 fps, UVG and MCL-JCV at their real frame rates).
"""
import copy
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path(r"C:\Users\Aditya\OneDrive\Desktop\Github\Real-Time Neural Video Compression")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
COLUMN_IN = 3.30
BLOCKS = 1024

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.0,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "lines.linewidth": 1.0, "lines.markersize": 3.0, "savefig.dpi": 600,
})


def points(*paths):
    out = {}
    for path in paths:
        report = json.loads((REPO / path).read_text(encoding="utf-8"))
        items = report["points"]
        out.update(items if isinstance(items, dict) else {str(i): p for i, p in enumerate(items)})
    return list(out.values())


def without_blocks(pts):
    pts = copy.deepcopy(pts)
    for p in pts:
        p["total_bytes"] -= BLOCKS * len(p["per_sequence"])
        p["bpp"] = p["total_bytes"] * 8 / p["total_pixels"]
    return pts


def curve(pts, arm=None):
    return sorted((p["bpp"], p["quality"]["sequence_mean"]["psnr"]) for p in pts
                  if arm is None or p.get("arm") == arm)


B = "outputs/benchmarks/"
PANELS = [
    ("(a) UVG, 7 videos, 1920×1072",
     points(B + "uvg_classical_420_gop30_realfps/parity.json", B + "uvg_classical_444_gop30_realfps/parity.json"),
     points("outputs/p13_audit_fixes/uvg_motion.json"), (0.02, 1.2), (30, 43)),
    ("(b) MCL-JCV, 30 videos, 1920×1072 (never used for a choice)",
     points(B + "mcljcv_classical_420_gop30_realfps/parity.json", B + "mcljcv_classical_444_gop30_realfps/parity.json"),
     without_blocks(points("outputs/p11_mcljcv/mcljcv_mse.json")), (0.02, 1.2), (30, 43)),
    ("(c) DAVIS TEST, 9 videos, 256×256",
     points(B + "parity_classical_gop30/parity_classical_gop30.json", B + "parity_classical_420_hq_gop30/parity.json",
            B + "parity_classical_444_gop30/parity_classical_444_gop30.json", B + "parity_classical_444_hq_gop30/parity.json"),
     points("outputs/p13_audit_fixes/test_motion.json"), (0.05, 2.0), (26, 40)),
]
SERIES = [
    ("h264", "H.264 (x264), 4:2:0", "#1f5aa6", "o", "-"),
    ("h264_444", "H.264 (x264), 4:4:4", "#1f5aa6", "o", ":"),
    ("h264_lowdelay", "H.264, 4:2:0, low-delay", "#7f8c99", "^", "-"),
    ("h265", "H.265 (x265), 4:2:0", "#2e7d32", "s", "-"),
]

fig, axes = plt.subplots(3, 1, figsize=(COLUMN_IN, 4.6))
for ax, (title, anchors, ours, xlim, ylim) in zip(axes, PANELS):
    for arm, label, color, marker, style in SERIES:
        x, y = zip(*[(b, q) for b, q in curve(anchors, arm) if xlim[0] * 0.8 <= b <= xlim[1] * 1.2])
        ax.plot(x, y, style, color=color, marker=marker, markersize=2.2, label=label)
    x, y = zip(*curve(ours))
    ax.plot(x, y, "-", color="#b22222", marker="D", label="Ours (MSE models)", linewidth=1.4)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, pad=2)
    ax.grid(True, which="both", color="#e3e3e3", linewidth=0.4)
    ax.set_ylabel("PSNR (dB)", labelpad=1)
axes[-1].set_xlabel("Bits per pixel (log scale)", labelpad=1)
axes[0].legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#cccccc",
               borderpad=0.3, labelspacing=0.2, handlelength=1.8)
fig.tight_layout(pad=0.3, h_pad=0.5)
fig.savefig(OUT / "fig2_rd.png")
plt.close(fig)

# ------------------------------------------------------------------ architecture
fig = plt.figure(figsize=(COLUMN_IN, 2.75))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 90)
ax.axis("off")
FONT = 5.5


def box(x, y, w, h, text, face, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                linewidth=0.7, facecolor=face, edgecolor=edge))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FONT,
            family="DejaVu Sans", linespacing=1.25)


def arrow(x0, y0, x1, y1, color="#333333", dashed=False):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
                                 linewidth=0.7, color=color, linestyle="--" if dashed else "-"))


GREY, BLUE, ORANGE, GREEN, PURPLE = (("#eef1f4", "#5d6d7e"), ("#e3edf9", "#2c5f9e"),
                                     ("#fdebd9", "#c0670f"), ("#e2f2e6", "#2f7d47"),
                                     ("#ece6f4", "#5b3f8f"))
box(14, 82, 72, 6, r"Frame $x_t$ (RGB). GOP 30: one I-frame, then 29 P-frames", *GREY)
box(2, 66, 38, 11, "I-frame: reference is a\nflat grey frame", *BLUE)
box(46, 66, 52, 11, "Block motion, 16×16, ±16 px, integer;\nsearch cost SAD + " r"$\lambda_m\cdot$" "bits;\nmedian-predicted vectors", *ORANGE)
box(46, 52, 52, 9, r"Warp $\hat{x}_{t-1}$ → $x_w$; context $c = F(x_w)$", *ORANGE)
box(2, 33, 96, 14, r"Conditional coder (9.1 M parameters, one per $\lambda$): $y = E(x_t, c)$" "\n"
    r"64 channels at 1/16 + 32 detail channels at 1/8 (given the 1/16 latent)" "\n"
    r"unit-step rounding around the prior mean; Laplace prior $p(\hat{y} \mid c)$", *BLUE)
box(2, 18, 96, 9, "C range coder, 64 Laplace scale tables  →  .nvct v2 stream (bit-exact decode)", *GREEN)
box(2, 3, 62, 10, r"Decoder: $\hat{x}_t = D(\hat{y}, c) + R(D(\hat{y}, c), x_w, c)$", *PURPLE)
box(70, 4, 28, 8, r"Reference $\hat{x}_t$", *PURPLE)

arrow(30, 82, 21, 77.3)
arrow(70, 82, 72, 77.3)
arrow(72, 66, 72, 61.3)
arrow(21, 66, 21, 47.3)
arrow(72, 52, 72, 47.3)
arrow(50, 33, 50, 27.3)
arrow(33, 18, 33, 13.3)
arrow(64, 8, 69.6, 8)
ax.plot([98.4, 99.5, 99.5], [8, 8, 56.5], color="#5b3f8f", linewidth=0.7, linestyle="--")
arrow(99.5, 56.5, 98.4, 56.5, color="#5b3f8f")
fig.savefig(OUT / "fig1_arch.png")
plt.close(fig)
print("figures written to", OUT)
