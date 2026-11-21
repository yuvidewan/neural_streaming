"""Per-frame coding views for the paper: the four panels the demo page shows under the
players (motion vectors, prediction, residual, error), for several frames of one clip.

Everything is read from a clip the demo pipeline already encoded
(outputs/demo/clips/<clip>/<bundle>/): the decoded frames, the motion-compensated
predictions, and the motion field and residual map stored in result.json. Nothing is
drawn by hand; the panels are computed the same way the demo page computes them.

Usage:
  python make_perframe_figure.py <repo_root> <clip> <bundle_id> <out.png> <frame> [<frame> ...]
Frames are 1-based, as the demo page numbers them.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

repo, clip, bundle_id, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
frames = [int(f) for f in sys.argv[5:]]
clip_dir = repo / "outputs" / "demo" / "clips" / clip
result = json.loads((clip_dir / bundle_id / "result.json").read_text(encoding="utf-8"))
nvc = result["nvc"]
block = result["bundle"]["block_size"]
SCALE = 2                                          # draw at 512 px for clean arrows
INK, AMBER = (14, 42, 51), (232, 163, 61)

activity_max = max(max(max(row) for row in a) for a in nvc["activity"] if a)


def load(path):
    return np.asarray(Image.open(path).convert("RGB"))


def motion_panel(index):
    original = Image.fromarray(load(clip_dir / "original" / f"{index:03d}.png"))
    base = original.resize((256 * SCALE,) * 2, Image.NEAREST)
    dim = Image.blend(base, Image.new("RGB", base.size, INK), 0.6)
    draw = ImageDraw.Draw(dim)
    dy, dx = (np.array(m) for m in nvc["motion"][index])
    for r in range(dy.shape[0]):
        for c in range(dy.shape[1]):
            cx, cy = (c * block + block / 2) * SCALE, (r * block + block / 2) * SCALE
            ox, oy = dx[r, c] * SCALE, dy[r, c] * SCALE
            if ox == 0 and oy == 0:
                draw.ellipse([cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5], fill=(170, 130, 70))
                continue
            sx, sy = cx + ox, cy + oy            # the block came from (sx, sy) in the previous frame
            draw.line([sx, sy, cx, cy], fill=AMBER, width=3)
            angle = np.arctan2(cy - sy, cx - sx)
            head = [(cx, cy),
                    (cx - 9 * np.cos(angle - 0.5), cy - 9 * np.sin(angle - 0.5)),
                    (cx - 9 * np.cos(angle + 0.5), cy - 9 * np.sin(angle + 0.5))]
            draw.polygon(head, fill=AMBER)
    return np.asarray(dim)


def residual_panel(index):
    act = np.sqrt(np.array(nvc["activity"][index]) / activity_max).clip(0, 1)
    stops = np.array([[14, 42, 51], [15, 118, 110], [232, 163, 61], [255, 244, 220]], float)
    pos = act * (len(stops) - 1)
    low = np.minimum(pos.astype(int), len(stops) - 2)
    mix = (pos - low)[..., None]
    rgb = stops[low] + (stops[low + 1] - stops[low]) * mix
    return np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((256 * SCALE,) * 2, Image.NEAREST))


def error_panel(index):
    original = load(clip_dir / "original" / f"{index:03d}.png").astype(float)
    decoded = load(clip_dir / bundle_id / "nvc" / f"{index:03d}.png").astype(float)
    level = np.clip(np.abs(original - decoded).mean(axis=2) * 8, 0, 255).astype(np.uint8)
    return np.stack([level] * 3, axis=2)


columns = ["Motion vectors", "Prediction", "Residual (latent)", "Error ×8"]
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7})
# Compact layout for a page-limited version: two frames per row, 2 rows x 8 panels,
# same panels as the 4x4 figure (same clip, bundle, frames and per-panel computation).
per_row = 2
rows = (len(frames) + per_row - 1) // per_row
fig = plt.figure(figsize=(6.6, 1.16 * rows + 0.02))
gs = fig.add_gridspec(rows, per_row * 4 + (per_row - 1), width_ratios=([1] * 4 + [0.12]) * (per_row - 1) + [1] * 4,
                      left=0.004, right=0.996, top=0.86, bottom=0.004, wspace=0.03, hspace=0.42)
for k, frame in enumerate(frames):
    index = frame - 1
    if nvc["frame_types"][index] != "P":
        raise SystemExit(f"frame {frame} is an I-frame; choose P-frames")
    panels = [motion_panel(index),
              load(clip_dir / bundle_id / "pred" / f"{index:03d}.png"),
              residual_panel(index), error_panel(index)]
    row, slot = divmod(k, per_row)
    group = []
    for col, panel in enumerate(panels):
        ax = fig.add_subplot(gs[row, slot * 5 + col])
        ax.imshow(panel, interpolation="nearest" if col == 2 else "antialiased")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(columns[col], fontsize=6.5, pad=2)
        group.append(ax)
    first, last = group[-4].get_position(), group[-1].get_position()
    fig.text((first.x0 + last.x1) / 2, first.y1 + 0.085 / rows, f"Frame {frame}", ha="center", va="bottom",
             fontsize=7.5, fontweight="bold")
fig.savefig(out, dpi=300, facecolor="white")
print("wrote", out, "frames", frames, "clip", clip)
