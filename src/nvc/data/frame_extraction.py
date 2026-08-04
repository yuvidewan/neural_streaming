"""Frame sampling, resizing, and lossless PNG export.

Shared by both dataset sources (see sources.py):
- extract_frames(): reads frames from a video file via OpenCV VideoCapture
- extract_sequence_frames(): reads frames from a folder of image files

Both call the same resize_frame()/save_frame_png() code - the only
difference between them is how a raw frame is obtained.

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
from typing import Iterable

import cv2
import numpy as np

from nvc.data.sequence_utils import (
    DEFAULT_IMAGE_EXTENSIONS,
    CorruptImageError,
    list_sequence_images,
    probe_sequence,
)
from nvc.data.video_utils import VideoMetadata, probe_video


@dataclass
class FrameExtractionResult:
    """Summary of what extract_frames()/extract_sequence_frames() produced."""

    source_path: Path
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


def save_frame_png(
    frame: np.ndarray,
    output_dir: Path,
    filename_prefix: str,
    output_index: int,
    target_width: int,
    target_height: int,
    preserve_aspect_ratio: bool,
) -> tuple[Path, int]:
    """Resize one frame and save it as a lossless PNG.

    Shared by both source types so resize/crop/save behavior - including
    exact output filenames - is identical regardless of where the frame
    came from. Returns (output_path, bytes_written).
    """
    processed = resize_frame(frame, target_width, target_height, preserve_aspect_ratio)
    out_path = output_dir / f"{filename_prefix}_{output_index:06d}.png"
    cv2.imwrite(str(out_path), processed)
    return out_path, out_path.stat().st_size


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    target_width: int,
    target_height: int,
    every_n_frames: int = 1,
    preserve_aspect_ratio: bool = True,
    filename_prefix: str | None = None,
    allowed_extensions: Iterable[str] | None = None,
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
                _, num_bytes = save_frame_png(
                    frame, output_dir, prefix, output_index,
                    target_width, target_height, preserve_aspect_ratio,
                )
                total_bytes += num_bytes
                saved_count += 1
                output_index += 1
            frame_index += 1
    finally:
        cap.release()

    return FrameExtractionResult(
        source_path=video_path,
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


def extract_sequence_frames(
    sequence_dir: str | Path,
    output_dir: str | Path,
    *,
    target_width: int,
    target_height: int,
    every_n_frames: int = 1,
    preserve_aspect_ratio: bool = True,
    filename_prefix: str | None = None,
    allowed_image_extensions: Iterable[str] | None = None,
) -> FrameExtractionResult:
    """Resize and save frames from a folder of sequential images (e.g. one
    DAVIS-style JPEGImages/480p/<sequence>/ folder) as lossless PNGs.

    Mirrors extract_frames() exactly (same validation-first behavior, same
    sampling semantics, same output filename scheme) but reads frames one
    image file at a time instead of decoding a video stream.
    """
    if every_n_frames < 1:
        raise ValueError(f"every_n_frames must be >= 1, got {every_n_frames}")
    if target_width <= 0 or target_height <= 0:
        raise ValueError(
            f"target_width/target_height must be positive, got {target_width}x{target_height}"
        )

    sequence_dir = Path(sequence_dir)
    output_dir = Path(output_dir)
    extensions = allowed_image_extensions or DEFAULT_IMAGE_EXTENSIONS

    # Validates every image in the sequence up front (readable, consistent
    # dimensions) and raises a clear SequenceError subclass if not.
    metadata = probe_sequence(sequence_dir, allowed_extensions=extensions)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or sequence_dir.name

    images = list_sequence_images(sequence_dir, extensions)

    total_bytes = 0
    saved_count = 0
    output_index = 1
    for frame_index, image_path in enumerate(images):
        if frame_index % every_n_frames != 0:
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise CorruptImageError(f"Could not read image (corrupt or unsupported): {image_path}")
        _, num_bytes = save_frame_png(
            frame, output_dir, prefix, output_index,
            target_width, target_height, preserve_aspect_ratio,
        )
        total_bytes += num_bytes
        saved_count += 1
        output_index += 1

    return FrameExtractionResult(
        source_path=sequence_dir,
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
