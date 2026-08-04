"""Image-sequence (folder-of-frames) inspection utilities.

Handles datasets that ship as one-folder-per-sequence, e.g. DAVIS's
JPEGImages/480p/<sequence>/00000.jpg layout - but nothing here is
DAVIS-specific: any dataset using that same "each subfolder is one
sequence of numbered images" convention works.

Mirrors video_utils.py's shape (discover -> probe -> raise clear errors)
so the two source types feel consistent to work with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2

from nvc.data.errors import DatasetSourceError
from nvc.data.video_utils import VideoMetadata

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class SequenceError(DatasetSourceError):
    """Base class for all image-sequence validation/read errors."""


class SequenceNotFoundError(SequenceError):
    """The given path does not exist or is not a directory."""


class EmptySequenceError(SequenceError):
    """The sequence folder contains no supported image files."""


class CorruptImageError(SequenceError):
    """An image file could not be read/decoded."""


class InconsistentDimensionsError(SequenceError):
    """Images within one sequence do not all share the same width/height."""


def discover_sequences(root: str | Path) -> list[Path]:
    """Return every direct subdirectory of root, sorted for determinism.

    Each subdirectory is treated as one independent sequence (e.g.
    JPEGImages/480p/bear/, .../camel/, ...). No filtering by content
    happens here - a subfolder with no/bad images is still "discovered"
    and is reported as a clear per-sequence error later, rather than
    being silently dropped.
    """
    root = Path(root)
    if not root.is_dir():
        raise SequenceNotFoundError(f"Image sequence root not found: {root}")
    return sorted(p for p in root.iterdir() if p.is_dir())


def list_sequence_images(
    sequence_dir: str | Path,
    allowed_extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> list[Path]:
    """List image files directly inside sequence_dir, sorted by filename.

    Plain lexicographic sort (not numeric-gap validation): zero-padded
    names like 00000.jpg, 00001.jpg sort correctly this way, and per the
    spec, sequences do not need to be perfectly contiguous - whatever
    files are present are used in sorted order.
    """
    sequence_dir = Path(sequence_dir)
    allowed = {ext.lower() for ext in allowed_extensions}
    images = [
        p for p in sequence_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed
    ]
    return sorted(images)


def probe_sequence(
    sequence_dir: str | Path,
    allowed_extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> VideoMetadata:
    """Validate an image-sequence folder and return its metadata.

    Checks the folder exists, is non-empty, every image is readable, and
    every image shares the same dimensions as the first one. Raises a
    clear SequenceError subclass instead of a confusing None/AttributeError
    later on. Returns a VideoMetadata (fps/duration are None - a sequence
    has no inherent frame rate) so this fits the same shape frame
    extraction already uses for videos.
    """
    sequence_dir = Path(sequence_dir)
    if not sequence_dir.is_dir():
        raise SequenceNotFoundError(f"Sequence folder not found: {sequence_dir}")

    images = list_sequence_images(sequence_dir, allowed_extensions)
    if not images:
        raise EmptySequenceError(f"No supported images found in sequence folder: {sequence_dir}")

    width: int | None = None
    height: int | None = None
    total_bytes = 0
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            raise CorruptImageError(f"Could not read image (corrupt or unsupported): {image_path}")

        image_height, image_width = image.shape[:2]
        if width is None:
            width, height = image_width, image_height
        elif (image_width, image_height) != (width, height):
            raise InconsistentDimensionsError(
                f"{image_path.name} is {image_width}x{image_height}, but the sequence "
                f"'{sequence_dir.name}' started with {width}x{height} "
                f"({images[0].name}). All frames in a sequence must share one resolution."
            )
        total_bytes += image_path.stat().st_size

    return VideoMetadata(
        filename=sequence_dir.name,
        path=sequence_dir,
        width=width,
        height=height,
        fps=None,
        frame_count=len(images),
        duration_seconds=None,
        file_size_bytes=total_bytes,
    )
