"""Turn the MCL-JCV 1080p source videos into a benchmark manifest (docs/PLAN_mcljcv.md).

MCL-JCV (Wang et al., ICIP 2016, USC MCL) is a standard 1080p test set of the learned video
coding literature: 30 videos, 1920x1080, 8-bit YUV 4:2:0. The release on Hugging Face
(`uscmcl/MCL-JCV_Dataset`) is 40 zips, about 79 GB, mostly H.264 test clips. Only the 30
`MCL_JCV/1080P/YUV_source/*.yuv` members are needed. This script reads each zip's central
directory and then just those members with HTTP range requests, decompresses the first
--frames frames as a stream, and pipes them straight into ffmpeg. No YUV is written to disk.

The protocol is the same as `prepare_uvg.py`:
- BT.709 to RGB;
- the height centre-cropped from 1080 to 1072;
- every video in split "test".
Videos that already have all their PNGs are skipped, so a rerun resumes.

Usage:
    prepare_mcljcv.py --output C:/mcljcv_scratch/mcljcv [--frames 96] [--only videoSRC01]
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import time
import zipfile
from pathlib import Path

import requests

REPO = "https://huggingface.co/datasets/uscmcl/MCL-JCV_Dataset/resolve/main/"
ZIP_NAME = "MCL_JCV-20250507T045226Z-{:03d}.zip"
ZIPS = range(3, 11)   # the zips holding the 1080p sources (checked 2026-11-21)
MEMBER = re.compile(r"MCL_JCV/1080P/YUV_source/(videoSRC\d\d)_1920x1080_(\d+)\.yuv$")
WIDTH, HEIGHT, CROPPED_HEIGHT = 1920, 1080, 1072
FRAME_BYTES = WIDTH * HEIGHT * 3 // 2
BLOCK = 16 << 20

session = requests.Session()
session.headers["User-Agent"] = "mcljcv-prepare/1.0"


class RemoteFile(io.RawIOBase):
    """A seekable read-only view of a remote file, fetched in BLOCK-sized range requests."""

    def __init__(self, url: str) -> None:
        response = session.head(url, allow_redirects=True, timeout=60)
        response.raise_for_status()
        self.url, self.size = response.url, int(response.headers["Content-Length"])
        self.pos, self.cache_start, self.cache = 0, 0, b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _fetch(self, start: int, length: int) -> bytes:
        end = min(start + length, self.size) - 1
        for attempt in range(8):
            try:
                response = session.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                                       timeout=120)
                response.raise_for_status()
                if len(response.content) == end - start + 1:
                    return response.content
            except requests.RequestException as error:
                print(f"    retry {attempt + 1}: {error}", flush=True)
            time.sleep(2 + 3 * attempt)
        raise IOError(f"range {start}-{end} of {self.url} failed")

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        offset = self.pos - self.cache_start
        if not (0 <= offset and offset + n <= len(self.cache)):
            self.cache_start, self.cache = self.pos, self._fetch(self.pos, max(n, BLOCK))
            offset = 0
        data = self.cache[offset:offset + n]
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def convert(stream, out_dir: Path, name: str, frames: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    top = (HEIGHT - CROPPED_HEIGHT) // 2
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
         "-s", f"{WIDTH}x{HEIGHT}", "-i", "pipe:0", "-frames:v", str(frames),
         "-vf", f"crop={WIDTH}:{CROPPED_HEIGHT}:0:{top},"
                "scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int",
         "-pix_fmt", "rgb24", "-start_number", "1", str(out_dir / f"{name}_%06d.png")],
        stdin=subprocess.PIPE)
    remaining = frames * FRAME_BYTES
    while remaining:
        chunk = stream.read(min(remaining, 4 << 20))
        if not chunk:
            break
        ffmpeg.stdin.write(chunk)
        remaining -= len(chunk)
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0 or remaining:
        raise RuntimeError(f"{name}: ffmpeg failed or the source ended early")
    return len(list(out_dir.glob(f"{name}_*.png")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--only", nargs="*", default=None, help="video names, for a smoke test")
    args = parser.parse_args()

    frames_dir = args.output / "frames"
    sources = {}
    for index in ZIPS:
        remote = RemoteFile(REPO + ZIP_NAME.format(index))
        archive = zipfile.ZipFile(remote)
        for info in archive.infolist():
            match = MEMBER.search(info.filename)
            if match:
                sources[match.group(1)] = (archive, info, int(match.group(2)))
    print(f"{len(sources)} 1080p sources found", flush=True)
    items = []
    for name in sorted(sources):
        if args.only and name not in args.only:
            continue
        archive, info, fps = sources[name]
        done = sorted(frames_dir.glob(f"{name}_*.png"))
        if len(done) < args.frames:
            started = time.time()
            with archive.open(info) as stream:
                count = convert(stream, frames_dir, name, args.frames)
            print(f"[{name}] {count} frames in {time.time() - started:.0f} s", flush=True)
        else:
            count = len(done)
            print(f"[{name}] {count} frames (already done)", flush=True)
        items.append({"source_type": "mcljcv_yuv420", "source_name": name, "split": "test",
                      "frame_count": count, "fps": fps, "original_resolution": [WIDTH, HEIGHT],
                      "processed_resolution": [WIDTH, CROPPED_HEIGHT],
                      "frame_directory": "frames",
                      "frame_filename_pattern": name + "_{index:06d}.png"})
    if args.only:
        return 0
    if len(items) != 30:
        raise SystemExit(f"expected 30 videos, got {len(items)}")
    manifest = {"source_type": "mcljcv", "settings": {"target_width": WIDTH,
                                                      "target_height": CROPPED_HEIGHT,
                                                      "frames_per_video": args.frames,
                                                      "colour_matrix": "bt709",
                                                      "crop": "centre, 1080 -> 1072 rows"},
                "items": items}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"manifest: {args.output / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
