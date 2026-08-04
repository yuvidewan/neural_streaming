"""FrameDataset: a torch.utils.data.Dataset over frames produced by the
Milestone 2/2.5 ingestion pipeline.

Reads data/processed/manifest.json directly (rather than globbing the
frames directory) so it works identically regardless of whether the
frames came from raw videos or an image-sequence dataset like DAVIS - the
manifest already made that distinction irrelevant downstream (Task 5 of
Milestone 2.5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2
import torch
from torch.utils.data import Dataset

from nvc.data.validation import DatasetValidationError, validate_frame_tensor

VALID_SPLITS = ("train", "val", "test")


class FrameDataset(Dataset):
    """Lazily loads one RGB frame per __getitem__ call - nothing is
    preloaded into memory at construction time, only the manifest (small
    JSON) and a flat list of file paths.

    Returns a single [3, H, W] float32 tensor in [0, 1] per sample (no
    label - this is for a reconstruction task, not classification).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise DatasetValidationError(
                f"Manifest not found: {manifest_path}. Run scripts/prepare_dataset.py first."
            )

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.split = split
        self.transform = transform

        settings = manifest.get("settings", {})
        self._expected_width: int | None = settings.get("target_width")
        self._expected_height: int | None = settings.get("target_height")

        self._frame_paths: list[Path] = []
        checked_dirs: set[Path] = set()
        for item in manifest["items"]:
            if item["split"] != split:
                continue

            frame_dir = (manifest_path.parent / item["frame_directory"]).resolve()
            if frame_dir not in checked_dirs:
                if not frame_dir.is_dir():
                    raise DatasetValidationError(
                        f"Frame directory for '{item['source_name']}' ({split} split) "
                        f"does not exist: {frame_dir}. Re-run scripts/prepare_dataset.py."
                    )
                checked_dirs.add(frame_dir)

            pattern = item["frame_filename_pattern"]
            for index in range(1, item["frame_count"] + 1):
                self._frame_paths.append(frame_dir / pattern.format(index=index))

        if not self._frame_paths:
            raise DatasetValidationError(
                f"No frames found for split '{split}' in manifest {manifest_path}."
            )

    def __len__(self) -> int:
        return len(self._frame_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self._frame_paths[index]

        # cv2.imread returns BGR (OpenCV's native in-memory order) or None
        # if the file is missing/corrupt/unsupported. Every other consumer
        # here (PyTorch models, matplotlib, torchvision) expects RGB, so
        # this conversion is required - unlike on the write side (see
        # frame_extraction.py), where cv2.imwrite already expects BGR.
        image = cv2.imread(str(path))
        if image is None:
            raise DatasetValidationError(
                f"Could not read image (missing, corrupt, or unsupported): {path}"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().to(torch.float32) / 255.0

        validate_frame_tensor(
            tensor,
            source=str(path),
            expected_height=self._expected_height,
            expected_width=self._expected_width,
        )

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor
