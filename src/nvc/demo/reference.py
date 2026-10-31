"""H.264 at a matched bitrate, for the side-by-side pane.

The scoreboard compares codecs at equal QUALITY (BD-rate). A viewer cannot see
that; what it can see is equal BYTES. So this encodes the clip with libx264 at
the highest quality whose whole .mp4 file is no larger than the neural stream,
and decodes it back to frames.

libx264 is used the way `nvc.evaluation.codecs.FFmpegVideoCodec` uses it
(preset medium, yuv420p, constant rate factor), so a number here means the same
thing as a number in the benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nvc.evaluation.ffmpeg import find_ffmpeg, has_encoders, is_ffmpeg_available, run_command

ENCODER = "libx264"
PRESET = "medium"
PIX_FMT = "yuv420p"
CRF_BEST, CRF_WORST = 1, 51          # 0 is lossless mode, a different coder


@dataclass(frozen=True)
class H264Match:
    crf: int
    total_bytes: int
    target_bytes: int
    within_target: bool               # False when even the worst quality is larger


def h264_available() -> bool:
    return is_ffmpeg_available() and has_encoders([ENCODER])


def encode_h264(frame_pattern: Path, output: Path, *, crf: int, fps: float) -> int:
    """Encode `frame_pattern` (e.g. .../original/%03d.png) and return the file size."""
    run_command([
        find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", f"{fps:g}", "-start_number", "0", "-i", str(frame_pattern),
        "-c:v", ENCODER, "-crf", str(crf), "-preset", PRESET, "-pix_fmt", PIX_FMT,
        str(output),
    ], timeout=600)
    return output.stat().st_size


def match_bitrate(frame_pattern: Path, output: Path, *, target_bytes: int, fps: float) -> H264Match:
    """Leave in `output` the best-quality H.264 encode no larger than `target_bytes`.

    File size falls as CRF rises, so this is a binary search for the smallest CRF
    that fits. If nothing fits, the worst-quality encode is kept and flagged.
    """
    if target_bytes < 1:
        raise ValueError("target_bytes must be positive")
    low, high = CRF_BEST, CRF_WORST
    sizes: dict[int, int] = {}
    scratch = output.with_name(output.stem + ".search" + output.suffix)

    def size_at(crf: int) -> int:
        if crf not in sizes:
            sizes[crf] = encode_h264(frame_pattern, scratch, crf=crf, fps=fps)
        return sizes[crf]

    try:
        if size_at(CRF_WORST) > target_bytes:
            best = CRF_WORST
        else:
            while low < high:
                middle = (low + high) // 2
                if size_at(middle) <= target_bytes:
                    high = middle
                else:
                    low = middle + 1
            best = low
        total = encode_h264(frame_pattern, output, crf=best, fps=fps)
    finally:
        scratch.unlink(missing_ok=True)
    return H264Match(crf=best, total_bytes=total, target_bytes=target_bytes,
                     within_target=total <= target_bytes)


def decode_to_frames(video: Path, directory: Path) -> int:
    """Decode every frame of `video` to 000.png, 001.png, ... and return the count."""
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.png"):
        stale.unlink()
    run_command([
        find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
        "-start_number", "0", str(directory / "%03d.png"),
    ], timeout=600)
    return len(list(directory.glob("*.png")))
