"""Shared synthetic test fixtures (not a test module itself).

Generates tiny videos, image sequences, checkpoints, and calibration files
in-process so tests never need an external dataset or a real training run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

# Deliberately tiny (a real model is latent_channels=64, base_channels=32) -
# every script test using this just needs a real, loadable, structurally
# valid checkpoint fast, not a good one.
TINY_MODEL_KWARGS: dict[str, Any] = {"in_channels": 3, "latent_channels": 4, "base_channels": 8}


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


def make_vimeo_dataset(
    root: Path,
    *,
    train_sequence_ids: list[str],
    test_sequence_ids: list[str],
    width: int = 448,
    height: int = 256,
    incomplete_ids: list[str] = (),
) -> Path:
    """Build a tiny synthetic Vimeo-90K Septuplet-style dataset on disk.

    Creates root/sequences/<id>/im1.png..im7.png for every id in
    train_sequence_ids + test_sequence_ids, plus sep_trainlist.txt and
    sep_testlist.txt. `incomplete_ids` (a subset of the above) get only 3 of
    the 7 required frames, for testing IncompleteVimeoSequenceError.

    Real sequence ids look like "00001/0001" (two path segments) - tests
    can pass ids with a "/" to exercise that nesting, or flat ids to check
    nothing hardcodes a fixed depth.
    """
    sequences_dir = root / "sequences"
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    for sequence_id in [*train_sequence_ids, *test_sequence_ids]:
        seq_dir = sequences_dir / sequence_id
        seq_dir.mkdir(parents=True, exist_ok=True)
        frame_count = 3 if sequence_id in incomplete_ids else 7
        for i in range(1, frame_count + 1):
            cv2.imwrite(str(seq_dir / f"im{i}.png"), frame)

    root.mkdir(parents=True, exist_ok=True)
    (root / "sep_trainlist.txt").write_text(
        "\n".join(train_sequence_ids) + "\n", encoding="utf-8"
    )
    (root / "sep_testlist.txt").write_text(
        "\n".join(test_sequence_ids) + "\n", encoding="utf-8"
    )
    return root


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


def make_tiny_checkpoint(
    path: Path,
    *,
    epoch: int = 1,
    history: list[dict[str, Any]] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    quantization_noise=None,
) -> Path:
    """Save a real, structurally valid BaselineAutoencoder checkpoint via the
    project's own save_checkpoint(), for scripts that need --checkpoint/
    --resume but don't need a *good* model (encode/decode/reconstruct
    scripts, calibration, benchmarking).

    Randomly initialized weights - correctness of the learned mapping is
    covered by tests/test_baseline_autoencoder.py; these tests only need
    "loads and runs without crashing, produces the declared shapes."
    """
    from nvc.models import BaselineAutoencoder
    from nvc.training import save_checkpoint

    kwargs = {**TINY_MODEL_KWARGS, **(model_kwargs or {})}
    model = BaselineAutoencoder(quantization_noise=quantization_noise, **kwargs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    save_checkpoint(
        path, model=model, optimizer=optimizer, epoch=epoch,
        history=history if history is not None else [],
        model_config=model.config_dict(),
    )
    return path


def make_tiny_calibration(
    path: Path,
    *,
    checkpoint_path: Path,
    bits: int = 4,
    mode: str = "per_channel",
    calibration_split: str = "train",
    num_channels: int = TINY_MODEL_KWARGS["latent_channels"],
    seed: int = 0,
) -> Path:
    """Write a real, structurally valid calibration file (quantization
    params + entropy model) without needing a real encoder pass over real
    data - synthesizes plausible per-channel latent statistics directly.

    Matches the exact document shape scripts/calibrate_quantizer.py writes
    (see nvc.compression.calibration.save_calibration), so any script that
    reads a calibration file via load_calibration() accepts this unchanged.
    """
    from nvc.compression.calibration import calibrate_quantization_params, save_calibration
    from nvc.compression.entropy_model import EmpiricalEntropyModel, _counts_to_frequencies

    rng = np.random.default_rng(seed)
    num_symbols = 2 ** bits
    # Fake but shape-correct calibration latents: [N, C, H, W].
    latents = torch.randn(64, num_channels, 4, 4, generator=torch.Generator().manual_seed(seed))
    params = calibrate_quantization_params(latents, bits=bits, mode=mode)

    counts = rng.integers(1, 100, size=(num_channels, num_symbols)).astype(np.float64)
    entropy_model = EmpiricalEntropyModel(_counts_to_frequencies(counts), bits=bits)

    save_calibration(
        path,
        params=params,
        entropy_model_data=entropy_model.to_dict(),
        metadata={
            "method": "per_channel_percentile",
            "bits": bits,
            "mode": mode,
            "latent_channels": num_channels,
            "calibration_frames": int(latents.shape[0]),
            "calibration_split": calibration_split,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": 1,
            "seed": 0,
            "entropy_model_id": entropy_model.model_id().hex(),
        },
    )
    return path
