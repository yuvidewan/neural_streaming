"""Video inspection utilities.

Wraps OpenCV's VideoCapture with explicit validation. Raw OpenCV failures
are silent (isOpened() == False, read() == (False, None)) rather than
raising exceptions, which produces confusing downstream errors. This
module turns those silent failures into clear, typed exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

from nvc.data.errors import DatasetSourceError


class VideoError(DatasetSourceError):
    """Base class for all video inspection/extraction errors."""


class VideoNotFoundError(VideoError):
    """The given path does not exist or is not a file."""


class UnsupportedVideoFormatError(VideoError):
    """The file extension is not in the allowed set of video formats."""


class CorruptVideoError(VideoError):
    """The file could not be opened or read as a valid video."""


@dataclass
class VideoMetadata:
    """Metadata read from a video container.

    fps, frame_count, and duration_seconds are None when the container
    does not report a usable value (a known OpenCV/codec quirk for some
    files) rather than raising an error - the video may still be entirely
    readable frame-by-frame.
    """

    filename: str
    path: Path
    width: int
    height: int
    fps: float | None
    frame_count: int | None
    duration_seconds: float | None
    file_size_bytes: int


def probe_video(
    path: str | Path,
    allowed_extensions: Iterable[str] | None = None,
) -> VideoMetadata:
    """Validate a video file and return its metadata.

    Raises VideoNotFoundError, UnsupportedVideoFormatError, or
    CorruptVideoError with a clear message instead of letting an invalid
    file produce a confusing OpenCV/numpy failure later on.
    """
    path = Path(path)

    if not path.exists() or not path.is_file():
        raise VideoNotFoundError(f"Video file not found: {path}")

    if allowed_extensions is not None and path.suffix.lower() not in {
        ext.lower() for ext in allowed_extensions
    }:
        raise UnsupportedVideoFormatError(
            f"Unsupported video extension '{path.suffix}' for {path.name} "
            f"(allowed: {sorted(allowed_extensions)})"
        )

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise CorruptVideoError(
                f"Could not open video (corrupt file or unsupported codec): {path}"
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise CorruptVideoError(
                f"Video reports invalid dimensions ({width}x{height}): {path}"
            )

        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        fps = fps_raw if fps_raw and fps_raw > 0 else None

        frame_count_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = frame_count_raw if frame_count_raw > 0 else None

        ok, _ = cap.read()
        if not ok:
            raise CorruptVideoError(f"Video opened but no frames could be read: {path}")
    finally:
        cap.release()

    duration = (frame_count / fps) if (frame_count and fps) else None

    return VideoMetadata(
        filename=path.name,
        path=path,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration,
        file_size_bytes=path.stat().st_size,
    )
