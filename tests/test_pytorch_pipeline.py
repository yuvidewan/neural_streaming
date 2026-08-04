"""Tests for Milestone 3: FrameDataset, transforms, DataLoader factories,
dataset validation, basic metrics, device selection, and reproducibility.

Uses a tiny synthetic manifest built via the real ingest_dataset()
pipeline (see helpers.make_tiny_manifest) - not a hand-written fake
manifest - so these tests exercise the actual manifest format produced by
Milestone 2.5, not a second copy of it.
"""

from __future__ import annotations

import json
import random
import shutil

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from nvc.data.frame_dataset import FrameDataset
from nvc.data.ingest import ingest_dataset
from nvc.data.loaders import create_test_loader, create_train_loader, create_val_loader
from nvc.data.transforms import build_eval_transform, build_train_transform
from nvc.data.validation import DatasetValidationError, validate_frame_tensor
from nvc.evaluation.basic_metrics import mse, psnr
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

from helpers import make_tiny_manifest


# --- FrameDataset ---


def test_frame_dataset_sizes_match_manifest(tmp_path):
    manifest_path = make_tiny_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for split in ("train", "val", "test"):
        dataset = FrameDataset(manifest_path, split=split)
        assert len(dataset) == manifest["summary"]["frames_per_split"][split]


def test_frame_dataset_item_shape_dtype_range(tmp_path):
    manifest_path = make_tiny_manifest(tmp_path, width=40, height=24)
    dataset = FrameDataset(manifest_path, split="train")

    tensor = dataset[0]

    assert tensor.shape == (3, 24, 40)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_frame_dataset_rgb_channel_order_is_correct(tmp_path):
    # Write a frame with a known pure-red color using OpenCV's native BGR
    # array order: (B=0, G=0, R=255). If FrameDataset's BGR->RGB
    # conversion were missing or wrong, channel 0 would come out ~0 (blue)
    # instead of ~1.0 (red).
    seq_dir = tmp_path / "sequences" / "red"
    seq_dir.mkdir(parents=True)
    bgr_frame = np.zeros((20, 20, 3), dtype=np.uint8)
    bgr_frame[:, :] = (0, 0, 255)
    cv2.imwrite(str(seq_dir / "00000.jpg"), bgr_frame)

    manifest_path = tmp_path / "processed" / "manifest.json"
    ingest_dataset(
        input_root=tmp_path / "sequences",
        frames_root=tmp_path / "frames",
        manifest_path=manifest_path,
        source_type="image_sequence",
        target_width=20, target_height=20,
        every_n_frames=1, preserve_aspect_ratio=True,
        train_ratio=1.0, val_ratio=0.0, test_ratio=0.0, seed=1,
    )

    dataset = FrameDataset(manifest_path, split="train")
    tensor = dataset[0]  # [C, H, W]: C=0 is R, C=1 is G, C=2 is B

    assert tensor[0].mean().item() == pytest.approx(1.0, abs=1e-2)
    assert tensor[1].mean().item() == pytest.approx(0.0, abs=1e-2)
    assert tensor[2].mean().item() == pytest.approx(0.0, abs=1e-2)


def test_frame_dataset_missing_manifest_raises(tmp_path):
    with pytest.raises(DatasetValidationError):
        FrameDataset(tmp_path / "no_manifest.json", split="train")


def test_frame_dataset_invalid_split_raises(tmp_path):
    manifest_path = make_tiny_manifest(tmp_path)
    with pytest.raises(ValueError):
        FrameDataset(manifest_path, split="not_a_split")


def test_frame_dataset_missing_frame_folder_raises(tmp_path):
    manifest_path = make_tiny_manifest(tmp_path)
    shutil.rmtree(tmp_path / "frames" / "train")

    with pytest.raises(DatasetValidationError):
        FrameDataset(manifest_path, split="train")


def test_frame_dataset_corrupt_image_raises_on_getitem(tmp_path):
    manifest_path = make_tiny_manifest(tmp_path)
    dataset = FrameDataset(manifest_path, split="train")
    dataset._frame_paths[0].write_bytes(b"not a real png")

    with pytest.raises(DatasetValidationError):
        dataset[0]


# --- Transforms ---


def test_build_eval_transform_is_deterministic():
    transform = build_eval_transform(crop_size=8)
    tensor = torch.rand(3, 16, 16)

    out_a = transform(tensor.clone())
    out_b = transform(tensor.clone())

    assert out_a.shape == (3, 8, 8)
    assert torch.equal(out_a, out_b)


def test_build_train_transform_crop_output_shape():
    transform = build_train_transform(crop_size=8)
    tensor = torch.rand(3, 16, 16)

    out = transform(tensor)

    assert out.shape == (3, 8, 8)


def test_build_train_transform_no_crop_preserves_shape():
    transform = build_train_transform(crop_size=None)
    tensor = torch.rand(3, 16, 16)

    out = transform(tensor)

    assert out.shape == (3, 16, 16)


def test_transforms_exclude_disallowed_augmentations():
    train_transform = build_train_transform(crop_size=8)
    names = {type(t).__name__ for t in train_transform.transforms}
    for forbidden in ("RandomVerticalFlip", "ColorJitter", "RandomRotation", "RandomPerspective"):
        assert forbidden not in names


# --- DataLoader factory ---


def test_create_train_loader_shuffles_and_batches(tmp_path):
    manifest_path = make_tiny_manifest(tmp_path)
    loader = create_train_loader(manifest_path, batch_size=2, seed=7)

    assert isinstance(loader.sampler, RandomSampler)
    batch = next(iter(loader))
    assert batch.shape[0] == 2
    assert batch.shape[1] == 3


def test_create_val_and_test_loaders_do_not_shuffle(tmp_path):
    manifest_path = make_tiny_manifest(tmp_path)
    val_loader = create_val_loader(manifest_path, batch_size=2)
    test_loader = create_test_loader(manifest_path, batch_size=2)

    assert isinstance(val_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)


def test_train_loader_covers_full_dataset_per_epoch(tmp_path):
    manifest_path = make_tiny_manifest(tmp_path, num_sequences=4, frames_per_sequence=4)
    loader = create_train_loader(manifest_path, batch_size=3, seed=1)

    total_seen = sum(batch.shape[0] for batch in loader)
    assert total_seen == len(loader.dataset)


# --- Dataset validation ---


def test_validate_frame_tensor_wrong_dtype_raises():
    tensor = torch.zeros(3, 4, 4, dtype=torch.uint8)
    with pytest.raises(DatasetValidationError):
        validate_frame_tensor(tensor)


def test_validate_frame_tensor_wrong_channels_raises():
    tensor = torch.zeros(4, 4, 4, dtype=torch.float32)
    with pytest.raises(DatasetValidationError):
        validate_frame_tensor(tensor)


def test_validate_frame_tensor_wrong_dimensions_raises():
    tensor = torch.zeros(3, 10, 10, dtype=torch.float32)
    with pytest.raises(DatasetValidationError):
        validate_frame_tensor(tensor, expected_height=20, expected_width=20)


def test_validate_frame_tensor_out_of_range_raises():
    tensor = torch.full((3, 4, 4), 1.5, dtype=torch.float32)
    with pytest.raises(DatasetValidationError):
        validate_frame_tensor(tensor)


def test_validate_frame_tensor_accepts_valid_tensor():
    tensor = torch.rand(3, 4, 4, dtype=torch.float32)
    validate_frame_tensor(tensor, expected_height=4, expected_width=4)  # should not raise


# --- Basic metrics ---


def test_mse_and_psnr_identity_image():
    image = torch.rand(3, 8, 8)

    assert mse(image, image).item() == 0.0
    assert psnr(image, image).item() == float("inf")


def test_mse_known_difference():
    a = torch.zeros(1, 1, 1)
    b = torch.full((1, 1, 1), 0.5)

    assert mse(a, b).item() == pytest.approx(0.25)


def test_psnr_matches_formula_for_known_mse():
    a = torch.zeros(1, 1, 1)
    b = torch.full((1, 1, 1), 0.5)
    expected = 10.0 * torch.log10(torch.tensor(1.0) / 0.25)

    assert psnr(a, b).item() == pytest.approx(expected.item())


# --- Reproducibility ---


def test_seed_everything_reproducible_across_libraries():
    seed_everything(123)
    py_a, np_a, torch_a = random.random(), np.random.rand(), torch.rand(3)

    seed_everything(123)
    py_b, np_b, torch_b = random.random(), np.random.rand(), torch.rand(3)

    assert py_a == py_b
    assert np_a == np_b
    assert torch.equal(torch_a, torch_b)


# --- Device utility ---


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda")
