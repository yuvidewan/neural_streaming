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
