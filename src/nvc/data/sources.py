"""Dataset source abstraction.

A DatasetSource hides how raw frames are obtained (decoding a video vs.
reading a folder of images) behind three small methods. ingest.py only
ever talks to this interface, so it doesn't care which kind of source
produced a given item - and adding a new source later (e.g. a directory
of image pairs) means writing one new class here, nothing else changes.

Deliberately minimal - a strategy pattern, not a plugin framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from nvc.data.dataset_prep import discover_videos
from nvc.data.frame_extraction import FrameExtractionResult, extract_frames, extract_sequence_frames
from nvc.data.sequence_utils import DEFAULT_IMAGE_EXTENSIONS, discover_sequences


class DatasetSource(ABC):
    """Strategy for turning one kind of raw input into extracted frames."""

    source_type: str

    @abstractmethod
    def discover(self, root: Path) -> list[Path]:
        """Return every processable item directly under root, sorted."""

    @abstractmethod
    def item_key(self, item: Path) -> str:
        """Stable name used for de-duplication and output frame filenames."""

    @abstractmethod
    def process(
        self,
        item: Path,
        output_dir: Path,
        *,
        target_width: int,
        target_height: int,
        every_n_frames: int,
        preserve_aspect_ratio: bool,
    ) -> FrameExtractionResult:
        """Validate, extract, resize, and save one item's frames.

        Raises a DatasetSourceError subclass if the item is invalid.
        """


class VideoDatasetSource(DatasetSource):
    """Frames come from decoding a video file."""

    source_type = "video"

    def __init__(self, extensions: Iterable[str]):
        self.extensions = tuple(extensions)

    def discover(self, root: Path) -> list[Path]:
        return discover_videos(root, self.extensions)

    def item_key(self, item: Path) -> str:
        return item.stem

    def process(self, item: Path, output_dir: Path, **kwargs) -> FrameExtractionResult:
        return extract_frames(item, output_dir, allowed_extensions=self.extensions, **kwargs)


class ImageSequenceDatasetSource(DatasetSource):
    """Frames come from reading an already-extracted folder of images."""

    source_type = "image_sequence"

    def __init__(self, extensions: Iterable[str] = DEFAULT_IMAGE_EXTENSIONS):
        self.extensions = tuple(extensions)

    def discover(self, root: Path) -> list[Path]:
        return discover_sequences(root)

    def item_key(self, item: Path) -> str:
        return item.name

    def process(self, item: Path, output_dir: Path, **kwargs) -> FrameExtractionResult:
        return extract_sequence_frames(
            item, output_dir, allowed_image_extensions=self.extensions, **kwargs
        )
