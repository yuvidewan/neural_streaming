"""DataLoader factory functions for FrameDataset.

num_workers defaults to 0 (single-process loading) - the Windows-safe
default. DataLoader worker processes on Windows are spawned, not forked,
which requires the calling script's entry point to be guarded by
`if __name__ == "__main__":`; scripts that have that guard can pass
num_workers > 0 explicitly.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nvc.data.frame_dataset import FrameDataset
from nvc.data.sequence_dataset import SequenceFrameDataset
from nvc.data.transforms import build_eval_transform, build_train_transform


def create_train_loader(
    manifest_path: str | Path,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    crop_size: int | None = None,
) -> DataLoader:
    """Shuffled loader over the train split. Shuffle order is reproducible
    for a fixed seed (a fresh torch.Generator is seeded per call).
    """
    dataset = FrameDataset(manifest_path, split="train", transform=build_train_transform(crop_size))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )


def _create_eval_loader(
    manifest_path: str | Path,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    crop_size: int | None,
) -> DataLoader:
    dataset = FrameDataset(manifest_path, split=split, transform=build_eval_transform(crop_size))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_val_loader(
    manifest_path: str | Path,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    crop_size: int | None = None,
) -> DataLoader:
    """Unshuffled, deterministic loader over the validation split."""
    return _create_eval_loader(
        manifest_path, "val",
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, crop_size=crop_size,
    )


def create_test_loader(
    manifest_path: str | Path,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    crop_size: int | None = None,
) -> DataLoader:
    """Unshuffled, deterministic loader over the test split."""
    return _create_eval_loader(
        manifest_path, "test",
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, crop_size=crop_size,
    )


# --- SequenceFrameDataset factories (Vimeo-90K and future sequence-oriented
# datasets). Mirror the FrameDataset factories above exactly, but know
# nothing about which dataset produced the manifest - see sequence_dataset.py
# and nvc.data.vimeo for why that separation matters.


def create_sequence_train_loader(
    manifest_path: str | Path,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    crop_size: int | None = None,
) -> DataLoader:
    """Shuffled loader over a sequence manifest's 'train' split."""
    dataset = SequenceFrameDataset(
        manifest_path, split="train",
        transform=build_train_transform(crop_size), crop_size=crop_size,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )


def _create_sequence_eval_loader(
    manifest_path: str | Path,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    crop_size: int | None,
) -> DataLoader:
    dataset = SequenceFrameDataset(
        manifest_path, split=split,
        transform=build_eval_transform(crop_size), crop_size=crop_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_sequence_val_loader(
    manifest_path: str | Path,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    crop_size: int | None = None,
) -> DataLoader:
    """Unshuffled, deterministic loader over a sequence manifest's 'val' split."""
    return _create_sequence_eval_loader(
        manifest_path, "val",
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, crop_size=crop_size,
    )


def create_sequence_test_loader(
    manifest_path: str | Path,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    crop_size: int | None = None,
) -> DataLoader:
    """Unshuffled, deterministic loader over a sequence manifest's 'test' split."""
    return _create_sequence_eval_loader(
        manifest_path, "test",
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, crop_size=crop_size,
    )
