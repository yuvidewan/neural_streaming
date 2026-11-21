"""Fig. 4 for the paper: one decoded P-frame of car-turn, proposed codec (3-bit, 5-bit) vs H.264 CRF 33.

Each panel is the full frame with a box, and under it that box enlarged 2x, so the differences between
the codecs are visible at column width. Everything is measured, nothing is drawn by hand:

- frames: the DAVIS test frames the benchmark uses (data/frames/test/<sequence>_NNNNNN.png, 256x256)
- proposed: the real codec bundles, encoded exactly as nvc.encode does; decoding is checked bit-exact
- H.264: libx264 CRF 33, preset medium, yuv420p, through FFmpeg (the demo's reference encoder)
- BPP is whole-file bits over the whole clip (the benchmark's convention); PSNR is for the shown frame

Run from the repo root, on the machine that has the codec bundles:

  python docs/paper_figures/make_fig4.py
  python docs/paper_figures/make_fig4.py --frame 37 --clip-frames 40 --zoom 112 100 128 64

Defaults: car-turn, frames 1-40, frame 37 (a P-frame: GOP 10 puts I-frames at 1, 11, 21, 31),
bundles outputs/codec_bundles/nvc_m22_{3,5}bit.pt. Stand-in bundles (outputs/demo/bundles) also run,
but the figure is then stamped PREVIEW, because their entropy model is untrained and the BPP is not real.
"""
from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from nvc.demo.reference import decode_to_frames, encode_h264, h264_available
from nvc.demo.trace import trace_encode
from nvc.video import VideoCodec
from nvc.video.container import FRAME_TYPE_I


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def to_uint8(frame: torch.Tensor) -> np.ndarray:
    return (frame.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def run_proposed(bundle: Path, frames_u8: np.ndarray, index: int, device: str):
    codec = VideoCodec.from_file(bundle, device=device)
    x = torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float().div(255.0)
    trace = trace_encode(codec, x)
    shown = trace.frames[index]
    if shown.frame_type == FRAME_TYPE_I:
        raise SystemExit(f"frame {index + 1} is an I-frame; choose a P-frame")
    decoded = codec.decode(trace.encoded.data)
    recon = np.stack([to_uint8(f) for f in trace.reconstructions])
    dec = np.stack([to_uint8(f) for f in decoded])
    if not np.array_equal(recon, dec):
        raise SystemExit(f"{bundle.name}: decoder output differs from the encoder's reconstruction")
    n, h, w = frames_u8.shape[:3]
    return {"bpp": len(trace.encoded.data) * 8 / (n * h * w), "psnr": psnr(dec[index], frames_u8[index]),
            "frame": dec[index], "frame_bytes": shown.motion_bytes + shown.residual_bytes}


def run_h264(frames_u8: np.ndarray, index: int, crf: int, fps: float):
    if not h264_available():
        raise SystemExit("FFmpeg with libx264 is required for the H.264 panel")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, f in enumerate(frames_u8):
            Image.fromarray(f).save(tmp / f"{i:03d}.png")
        total = encode_h264(tmp / "%03d.png", tmp / "clip.mp4", crf=crf, fps=fps)
        count = decode_to_frames(tmp / "clip.mp4", tmp / "dec")
        if count != len(frames_u8):
            raise SystemExit(f"H.264 decoded {count} frames, expected {len(frames_u8)}")
        frame = np.asarray(Image.open(tmp / "dec" / f"{index:03d}.png").convert("RGB"))
    n, h, w = frames_u8.shape[:3]
    return {"bpp": total * 8 / (n * h * w), "psnr": psnr(frame, frames_u8[index]), "frame": frame}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", default="car-turn")
    ap.add_argument("--frames-dir", type=Path, default=Path("data/frames/test"))
    ap.add_argument("--clip-frames", type=int, default=40)
    ap.add_argument("--frame", type=int, default=37, help="1-based frame to show (must be a P-frame)")
    ap.add_argument("--bundle-low", type=Path, default=Path("outputs/codec_bundles/nvc_m22_3bit.pt"))
    ap.add_argument("--bundle-high", type=Path, default=Path("outputs/codec_bundles/nvc_m22_5bit.pt"))
    ap.add_argument("--crf", type=int, default=33)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--zoom", type=int, nargs=4, default=[112, 100, 128, 64], metavar=("X", "Y", "W", "H"),
                    help="box enlarged 2x under each frame, in pixels of the 256x256 frame")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("docs/paper_figures/fig4_carturn.png"))
    args = ap.parse_args()

    paths = [args.frames_dir / f"{args.sequence}_{i:06d}.png" for i in range(1, args.clip_frames + 1)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"missing test frames, e.g. {missing[0]}")
    frames_u8 = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
    index = args.frame - 1
    preview = any("standin" in b.name for b in (args.bundle_low, args.bundle_high))

    low = run_proposed(args.bundle_low, frames_u8, index, args.device)
    high = run_proposed(args.bundle_high, frames_u8, index, args.device)
    ref = run_h264(frames_u8, index, args.crf, args.fps)
    bits_low, bits_high = (re.search(r"(\d+)bit", b.stem).group(1) for b in (args.bundle_low, args.bundle_high))

    panels = [
        ("(a) Original frame\n ", frames_u8[index]),
        (f"(b) Proposed, {bits_low}-bit\n{low['bpp']:.3f} BPP, {low['psnr']:.2f} dB", low["frame"]),
        (f"(c) Proposed, {bits_high}-bit\n{high['bpp']:.3f} BPP, {high['psnr']:.2f} dB", high["frame"]),
        (f"(d) H.264, CRF {args.crf}\n{ref['bpp']:.3f} BPP, {ref['psnr']:.2f} dB", ref["frame"]),
    ]
    zx, zy, zw, zh = args.zoom
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7.5})
    fig = plt.figure(figsize=(3.35, 3.35 * 1.66), dpi=300)
    gs = fig.add_gridspec(5, 2, height_ratios=[2, 1, 0.36, 2, 1], left=0.01, right=0.99, top=0.945, bottom=0.005,
                          wspace=0.04, hspace=0.05)
    for k, (label, img) in enumerate(panels):
        row, col = divmod(k, 2)
        ax = fig.add_subplot(gs[row * 3, col])
        ax.imshow(img)
        ax.add_patch(plt.Rectangle((zx - 0.5, zy - 0.5), zw, zh, fill=False, ec="#d62728", lw=0.9))
        ax.set_title(label, fontsize=7.5, pad=2.5, linespacing=1.15)
        axz = fig.add_subplot(gs[row * 3 + 1, col])
        axz.imshow(img[zy:zy + zh, zx:zx + zw], interpolation="nearest")
        for a in (ax, axz):
            a.set_xticks([]); a.set_yticks([])
        for spine in axz.spines.values():
            spine.set_edgecolor("#d62728"); spine.set_linewidth(0.9)
    if preview:
        fig.text(0.5, 0.5, "PREVIEW - demo stand-in bundles\nnot for the paper", ha="center", va="center",
                 fontsize=16, color="#d62728", alpha=0.55, rotation=30, fontweight="bold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, facecolor="white")

    print(f"wrote {args.out}{'  (PREVIEW: stand-in bundles)' if preview else ''}")
    print(f"clip: {args.sequence} frames 1-{args.clip_frames}, shown frame {args.frame} (P-frame)")
    for name, r in (("proposed low", low), ("proposed high", high), ("h264", ref)):
        print(f"  {name:13s} {r['bpp']:.4f} BPP  {r['psnr']:.2f} dB")
    ratio = low["bpp"] / ref["bpp"]
    print(f"caption numbers: Frame {args.frame} of a {args.clip_frames}-Frame Clip")
    print(f"paper claim 'H.264 CRF {args.crf} beats the {bits_low}-bit PSNR with a quarter of its bits': "
          f"PSNR {'yes' if ref['psnr'] > low['psnr'] else 'NO'} ({ref['psnr']:.2f} vs {low['psnr']:.2f} dB), "
          f"bits ratio {ratio:.1f}x")


if __name__ == "__main__":
    main()
