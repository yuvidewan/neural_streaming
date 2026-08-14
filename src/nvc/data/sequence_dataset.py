"""SequenceFrameDataset: a torch.utils.data.Dataset over the generic
sequence manifest schema produced by nvc.data.vimeo.build_sequence_manifest
(or, in principle, any future dataset-specific builder emitting the same
schema - this class has no Vimeo-specific knowledge at all).

Mirrors FrameDataset's contract deliberately: construction only parses the
manifest and builds a flat index (no image is opened until __getitem__), and
__getitem__ returns a single [3, H, W] float32 tensor in [0, 1] - so the
existing frame-based BaselineAutoencoder and training loop consume Vimeo
sequences exactly as they consume DAVIS frames, with zero changes to
model/training code. What FrameDataset gets from a manifest's frame
directory, this gets from (dataset_root / sequences_dirname / sequence_id /
frame_filename) - each frame is read directly from the original source
file, never a resized copy.

SEQUENCE IDENTITY
------------------
Flattening sequences into individual-frame samples would normally throw
away which sequence/position each frame came from. That information is
preserved via `sequence_id_at(index)` / `frame_index_at(index)` for later
use by temporal/inter-frame models - not used by anything in this
milestone, but kept alongside the flat index rather than discarded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

from nvc.data.image_io import read_image_as_tensor
from nvc.data.validation import DatasetValidationError, validate_frame_tensor


class SequenceFrameDataset(Dataset):
    """Lazily loads one frame at a time from a sequence-oriented manifest.

    Each `items[]` entry in the manifest (one sequence) expands to
    `frame_count` samples, ordered by `frame_filenames`. `crop_size`, when
    given, is enforced on every loaded tensor via `expected_height`/
    `expected_width` in `validate_frame_tensor` - since source frames are
    read directly from disk at their native resolution (never resized),
    this only holds if `transform` actually reduces every frame to
    `crop_size` x `crop_size` (e.g. via a fixed-size crop) or every source
    frame already shares that resolution.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        *,
        crop_size: int | None = None,
    ) -> None:
        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise DatasetValidationError(
                f"Sequence manifest not found: {manifest_path}. Run "
                "scripts/prepare_training_dataset.py first."
            )

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.split = split
        self.transform = transform
        self._crop_size = crop_size

        dataset_root = (manifest_path.parent / manifest["dataset_root"]).resolve()
        sequences_dir = dataset_root / manifest["sequences_dirname"]

        # One entry per (sequence, frame) pair - built once at construction,
        # not scanned again per __getitem__ call.
        self._frame_paths: list[Path] = []
        self._sequence_ids: list[str] = []
        self._frame_indices: list[int] = []

        for item in manifest["items"]:
            if item["split"] != split:
                continue
            seq_dir = sequences_dir / item["sequence_id"]
            for frame_index, filename in enumerate(item["frame_filenames"]):
                self._frame_paths.append(seq_dir / filename)
                self._sequence_ids.append(item["sequence_id"])
                self._frame_indices.append(frame_index)

        if not self._frame_paths:
            raise DatasetValidationError(
                f"No frames found for split '{split}' in manifest {manifest_path}."
            )

    def __len__(self) -> int:
        return len(self._frame_paths)

    def sequence_id_at(self, index: int) -> str:
        return self._sequence_ids[index]

    def frame_index_at(self, index: int) -> int:
        return self._frame_indices[index]

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self._frame_paths[index]

        # Unlike FrameDataset, the source frame is read at its native
        # resolution (never pre-resized during ingestion), so shape
        # validation must run AFTER the transform - a crop_size check
        # against the raw, uncropped frame would fail every sample.
        tensor = read_image_as_tensor(path)

        if self.transform is not None:
            tensor = self.transform(tensor)

        validate_frame_tensor(
            tensor,
            source=str(path),
            expected_height=self._crop_size,
            expected_width=self._crop_size,
        )

        return tensor
