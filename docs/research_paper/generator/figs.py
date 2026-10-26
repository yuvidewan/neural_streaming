"""Figures for the NCC 2027 revision: architecture, RD curves, per-frame views."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

REPO = Path(r"C:\Users\Aditya\OneDrive\Desktop\Github\Real-Time Neural Video Compression")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
COLUMN_IN = 3.30

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.3,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "lines.linewidth": 1.1, "lines.markersize": 3.2, "savefig.dpi": 600,
})


def load(path):
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def curve(points, metric="psnr"):
    return sorted((p["bpp"], p["quality"]["sequence_mean"][metric]) for p in points)


# ------------------------------------------------------------------ RD curves
refit = load("outputs/benchmarks/parity_refit/parity_64ch_qat.json")
stage0 = load("outputs/benchmarks/parity_s0/parity.json")
intra = load("outputs/benchmarks/parity_intra/parity_intra.json")
gate = load("outputs/stage1_vimeo/colab_run4_64ch_qat/gate_qat_epoch23")
m23 = load("outputs/m23_hyperprior/m23_deploy_test.json")
anchor = load("outputs/benchmarks/compressai_anchor/compressai_anchor.json")


def arm(report, name):
    points = report["points"]
    points = points.values() if isinstance(points, dict) else points
    return [p for p in points if p["arm"] == name]


final = [{"bpp": rp["candidate"]["bpp"], "quality": rp["candidate"]["quality"]}
         for rp in m23["rate_points"]]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COLUMN_IN, 3.6))
series = [
    (curve(arm(refit, "h264")), "H.264 (x264, default)", "#1f5aa6", "o", "-"),
    (curve(arm(refit, "h265")), "H.265 (x265, default)", "#2e7d32", "s", "-"),
    (curve(arm(refit, "h264_lowdelay")), "H.264, I every 10, no B", "#7f8c99", "^", "-"),
    (curve(arm(stage0, "nvc_m22")), "Initial codec (593 k)", "#d98c8c", "D", "--"),
    (curve(final), "Final codec", "#b22222", "D", "-"),
]
for data, label, color, marker, style in series:
    x, y = zip(*data)
    ax1.plot(x, y, style, color=color, marker=marker, label=label)
ax1.set_xlim(0, 0.85)
ax1.set_ylim(22, 34)
ax1.set_title("(a) Full video, GOP 10")

series = [
    (curve(arm(intra, "h264_intra")), "H.264 all-intra (x264)", "#1f5aa6", "o", "-"),
    (curve(arm(anchor, "bmshj2018-factorized")), "bmshj2018 factorized [CompressAI]", "#e08a00", "v", "-"),
    (curve(arm(anchor, "bmshj2018-hyperprior")), "bmshj2018 hyperprior [CompressAI]", "#8a5a00", "P", "-"),
    (curve(arm(intra, "nvc_deployed")), "Initial transform (593 k)", "#b39ddb", "s", "--"),
    (curve(arm(gate, "nvc_intra")),
     "Final transform (64-ch + QAT)", "#5e35b1", "s", "-"),
]
for data, label, color, marker, style in series:
    x, y = zip(*data)
    ax2.plot(x, y, style, color=color, marker=marker, label=label)
ax2.set_xlim(0, 1.0)
ax2.set_ylim(17, 36)
ax2.set_title("(b) All-intra (every frame an I-frame)")

for ax in (ax1, ax2):
    ax.grid(True, color="#d9d9d9", linewidth=0.4)
    ax.set_xlabel("Bits per pixel (BPP)", labelpad=1)
    ax.set_ylabel("PSNR (dB)", labelpad=1)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="#cccccc",
              borderpad=0.3, labelspacing=0.25, handlelength=1.8)
fig.tight_layout(pad=0.3, h_pad=0.6)
fig.savefig(OUT / "fig2_rd.png")
plt.close(fig)

# ------------------------------------------------------------------ architecture
fig = plt.figure(figsize=(COLUMN_IN, 2.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 86)
ax.axis("off")
FONT = 5.6


def box(x, y, w, h, text, face, edge, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                linewidth=0.7, facecolor=face, edgecolor=edge))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FONT,
            family="DejaVu Sans", weight="bold" if bold else "normal", linespacing=1.25)


def arrow(x0, y0, x1, y1, color="#333333", style="-|>", dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=6,
                                 linewidth=0.7, color=color, linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}"))


GREY, BLUE, ORANGE, GREEN, PURPLE = (("#eef1f4", "#5d6d7e"), ("#e3edf9", "#2c5f9e"),
                                     ("#fdebd9", "#c0670f"), ("#e2f2e6", "#2f7d47"),
                                     ("#ece6f4", "#5b3f8f"))
box(18, 78, 64, 6.5, r"Input frame $x_t$, 256×256 RGB (GOP of 10)", *GREY)
ax.text(26, 73.2, "I-frame", fontsize=FONT, color="#2c5f9e", style="italic",
        family="DejaVu Sans", ha="left")
ax.text(72.5, 73.2, "P-frame", fontsize=FONT, color="#c0670f", style="italic",
        family="DejaVu Sans", ha="right")
box(2, 59, 44, 11, "Analysis transform $E$: GDN +\nresidual blocks, stride 16, 7.2 M", *BLUE)
box(2, 44, 44, 11, r"Latent $z_t$ (64×16×16), per-channel" "\n" "range-fitted quantizer, 3/4/5 bit", *BLUE)
box(2, 31, 44, 9, "Static per-channel tables", *BLUE)
box(54, 59, 41, 11, "Block motion search, 16×16,\n±16 px, integer-pel (SAD)", *ORANGE)
box(54, 44, 41, 11, r"$z_{\mathrm{ref}} = E(W(\hat{x}_{t-1}, \widehat{mv}_t))$" "\n" r"$\Delta z = z_t - z_{\mathrm{ref}}$", *ORANGE)
box(54, 29, 41, 11, "G16 context + scale map\n→ 512-table shared codebook", *ORANGE)
box(10, 17.5, 80, 7.5, "Native C range coder → .nvct v3 container", *GREEN)
box(2, 3, 62, 10, r"Decoder: $\hat{z}_t = z_{\mathrm{ref}} + \Delta\hat{z}_t$, synthesis $D$" "\n" r"(IGDN + residual blocks, sigmoid) → $\hat{x}_t$", *PURPLE)
box(69, 4.5, 26, 7, r"Reference $\hat{x}_{t-1}$", *PURPLE)

arrow(24, 78, 24, 70.3)
arrow(74.5, 78, 74.5, 70.3)
arrow(24, 59, 24, 55.3)
arrow(24, 44, 24, 40.3)
arrow(74.5, 59, 74.5, 55.3)
arrow(74.5, 44, 74.5, 40.3)
arrow(24, 31, 40, 25.3)
arrow(76, 29, 60, 25.3)
arrow(50, 17.5, 40, 13.3)
arrow(64, 8, 68.7, 8)
ax.plot([95.3, 98.2, 98.2], [8, 8, 49.5], color="#5b3f8f", linewidth=0.7, linestyle="--")
arrow(98.2, 49.5, 95.5, 49.5, color="#5b3f8f")
fig.savefig(OUT / "fig1_arch.png")
plt.close(fig)

# ------------------------------------------------------------------ per-frame views, two rows
views = Image.open(sys.argv[2])
width, height = views.size
# The source grid has a header strip and four equal rows; keep the header and rows 1-2.
gray = views.convert("L")
row_bounds = []
inside = False
for yy in range(height):
    dark = sum(1 for xx in range(0, width, 7) if gray.getpixel((xx, yy)) < 235) > 30
    if dark and not inside:
        start, inside = yy, True
    elif not dark and inside:
        row_bounds.append((start, yy))
        inside = False
print("bands:", row_bounds)
bands = [b for b in row_bounds if b[1] - b[0] > 200]
cut = bands[1][1] + 12
views.crop((0, 40, width, cut)).save(OUT / "fig3_views.png")
print("views cropped to", width, "x", cut)
