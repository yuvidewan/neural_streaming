"""Shared synthetic test fixtures (not a test module itself).

Generates tiny videos and image sequences in-process so tests never need
an external dataset.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_synthetic_video(
    path: Path,
    *,
    num_frames: int = 10,
    width: int = 64,
    height: int = 48,
    fps: float = 10.0,
) -> Path:
    """Write a tiny synthetic .mp4 with solid-color frames for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), f"Could not open VideoWriter for {path}"
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=(i * 20) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def make_sequence(
    sequence_dir: Path,
    *,
    num_frames: int = 5,
    width: int = 64,
    height: int = 48,
    ext: str = ".jpg",
    start_index: int = 0,
) -> Path:
    """Create a synthetic DAVIS-style sequence folder: 00000.jpg, 00001.jpg, ..."""
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        frame = np.full((height, width, 3), fill_value=(i * 15) % 256, dtype=np.uint8)
        cv2.imwrite(str(sequence_dir / f"{start_index + i:05d}{ext}"), frame)
    return sequence_dir


def make_tiny_manifest(
    tmp_path: Path,
    *,
    num_sequences: int = 4,
    frames_per_sequence: int = 8,
    width: int = 32,
    height: int = 32,
    train_ratio: float = 0.5,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
    seed: int = 1,
) -> Path:
    """Build a tiny synthetic image-sequence dataset and ingest it for real
    via nvc.data.ingest.ingest_dataset(), returning the resulting
    manifest.json path.

    Reuses the actual ingestion pipeline (not a hand-written fake manifest)
    so Milestone 3 tests exercise the real manifest format instead of a
    second, drifting copy of it.
    """
    from nvc.data.ingest import ingest_dataset

    root = tmp_path / "sequences"
    for i in range(num_sequences):
        make_sequence(root / f"seq{i}", num_frames=frames_per_sequence, width=width, height=height)

    manifest_path = tmp_path / "processed" / "manifest.json"
    ingest_dataset(
        input_root=root,
        frames_root=tmp_path / "frames",
        manifest_path=manifest_path,
        source_type="image_sequence",
        target_width=width,
        target_height=height,
        every_n_frames=1,
        preserve_aspect_ratio=True,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    return manifest_path
