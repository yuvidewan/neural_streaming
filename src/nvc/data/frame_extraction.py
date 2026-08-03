"""Frame sampling, resizing, and lossless PNG export for a single video.

Frames are kept in OpenCV's native BGR channel order and written with
cv2.imwrite, which expects BGR and encodes it correctly - no BGR/RGB
conversion is needed here. (Converting to RGB before cv2.imwrite would
actually produce a file with swapped colors; conversion only matters if
someone later loads the raw array directly instead of going through the
saved PNG file.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from nvc.data.video_utils import VideoMetadata, probe_video


@dataclass
class FrameExtractionResult:
    """Summary of what extract_frames() produced for one video."""

    video_path: Path
    metadata: VideoMetadata
    output_dir: Path
    filename_prefix: str
    extracted_frame_count: int
    every_n_frames: int
    target_width: int
    target_height: int
    preserve_aspect_ratio: bool
    total_output_bytes: int


def _pick_interpolation(orig_w: int, orig_h: int, new_w: int, new_h: int) -> int:
    is_downscale = new_w <= orig_w and new_h <= orig_h
    return cv2.INTER_AREA if is_downscale else cv2.INTER_LINEAR


def resize_frame(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
    preserve_aspect_ratio: bool = True,
) -> np.ndarray:
    """Resize a frame to exactly (target_height, target_width, channels).

    preserve_aspect_ratio=True (default): resize so the frame fully covers
    the target box, then deterministically center-crop down to the exact
    target size. This never stretches the image.

    preserve_aspect_ratio=False: resize directly to the target size. This
    will stretch/distort the image if its aspect ratio differs from the
    target - only pass this explicitly when that's acceptable.
    """
    orig_height, orig_width = frame.shape[:2]

    if not preserve_aspect_ratio:
        interpolation = _pick_interpolation(orig_width, orig_height, target_width, target_height)
        return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)

    scale = max(target_width / orig_width, target_height / orig_height)
    new_width = max(target_width, round(orig_width * scale))
    new_height = max(target_height, round(orig_height * scale))
    interpolation = _pick_interpolation(orig_width, orig_height, new_width, new_height)
    resized = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)

    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    return resized[top : top + target_height, left : left + target_width]


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    target_width: int,
    target_height: int,
    every_n_frames: int = 1,
    preserve_aspect_ratio: bool = True,
    filename_prefix: str | None = None,
    allowed_extensions: list[str] | None = None,
) -> FrameExtractionResult:
    """Extract, resize, and save frames from one video as lossless PNGs.

    Frames are numbered sequentially starting at 1 in output order (i.e.
    after sampling), not by their original position in the source video:
    ``{prefix}_000001.png``, ``{prefix}_000002.png``, ...
    """
    if every_n_frames < 1:
        raise ValueError(f"every_n_frames must be >= 1, got {every_n_frames}")
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"target_width/target_height must be positive, got {target_width}x{target_height}"
        )

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # Validates the file and raises a clear VideoError subclass if it can't
    # be read; also gives us width/height/fps/frame_count for the manifest.
    metadata = probe_video(video_path, allowed_extensions=allowed_extensions)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    total_bytes = 0
    saved_count = 0
    try:
        frame_index = 0
        output_index = 1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % every_n_frames == 0:
                processed = resize_frame(frame, target_width, target_height, preserve_aspect_ratio)
                out_path = output_dir / f"{prefix}_{output_index:06d}.png"
                cv2.imwrite(str(out_path), processed)
                total_bytes += out_path.stat().st_size
                saved_count += 1
                output_index += 1
            frame_index += 1
    finally:
        cap.release()

    return FrameExtractionResult(
        video_path=video_path,
        metadata=metadata,
        output_dir=output_dir,
        filename_prefix=prefix,
        extracted_frame_count=saved_count,
        every_n_frames=every_n_frames,
        target_width=target_width,
        target_height=target_height,
        preserve_aspect_ratio=preserve_aspect_ratio,
        total_output_bytes=total_bytes,
    )
