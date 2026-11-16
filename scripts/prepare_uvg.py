"""Turn the UVG 1080p test videos into a benchmark manifest.

UVG (Mercat et al., MMSys 2020, ultravideo.fi) is the standard test set of the learned video
coding literature: 7 videos, 1920x1080, 8-bit YUV 4:2:0. This script follows the literature's
usual protocol (DVC, DCVC): the first 96 frames of every video.

For each `<Name>.7z` in --archives it extracts the raw YUV, converts the first --frames frames
to RGB PNGs (BT.709, the colour matrix of HD video), centre-crops the height from 1080 to
1072 so both sides are multiples of 16, and deletes the YUV. It writes a manifest in the
format `nvc.evaluation.sequences.discover_sequences` reads, with every video in split "test".

Usage:
    prepare_uvg.py --archives C:/uvg_scratch --output C:/uvg_scratch/uvg [--frames 96]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

NAMES = ("Beauty", "Bosphorus", "HoneyBee", "Jockey", "ReadySetGo", "ShakeNDry", "YachtRide")
WIDTH, HEIGHT, CROPPED_HEIGHT = 1920, 1080, 1072


def convert(yuv: Path, out_dir: Path, name: str, frames: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    top = (HEIGHT - CROPPED_HEIGHT) // 2
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
         "-s", f"{WIDTH}x{HEIGHT}", "-i", str(yuv), "-frames:v", str(frames),
         "-vf", f"crop={WIDTH}:{CROPPED_HEIGHT}:0:{top},"
                "scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int",
         "-pix_fmt", "rgb24", "-start_number", "1", str(out_dir / f"{name}_%06d.png")],
        check=True)
    return len(list(out_dir.glob(f"{name}_*.png")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()
    import py7zr

    frames_dir = args.output / "frames"
    items = []
    for name in NAMES:
        done = sorted(frames_dir.glob(f"{name}_*.png"))
        if len(done) < args.frames:
            archive = args.archives / f"{name}.7z"
            work = args.output / "_yuv" / name
            work.mkdir(parents=True, exist_ok=True)
            print(f"[{name}] extracting {archive}", flush=True)
            with py7zr.SevenZipFile(archive) as handle:
                handle.extractall(work)
            yuv = max(work.rglob("*.yuv"), key=lambda p: p.stat().st_size)
            count = convert(yuv, frames_dir, name, args.frames)
            shutil.rmtree(work)
            if not args.keep_archives:
                archive.unlink()
        else:
            count = len(done)
        print(f"[{name}] {count} frames", flush=True)
        items.append({"source_type": "uvg_yuv420", "source_name": name, "split": "test",
                      "frame_count": count, "original_resolution": [WIDTH, HEIGHT],
                      "processed_resolution": [WIDTH, CROPPED_HEIGHT],
                      "frame_directory": "frames",
                      "frame_filename_pattern": name + "_{index:06d}.png"})
    manifest = {"source_type": "uvg", "settings": {"target_width": WIDTH,
                                                   "target_height": CROPPED_HEIGHT,
                                                   "frames_per_video": args.frames,
                                                   "colour_matrix": "bt709",
                                                   "crop": "centre, 1080 -> 1072 rows"},
                "items": items}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    shutil.rmtree(args.output / "_yuv", ignore_errors=True)
    print(f"manifest: {args.output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
