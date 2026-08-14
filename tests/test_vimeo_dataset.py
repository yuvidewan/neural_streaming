"""Tests for Milestone 6.5: Vimeo-90K discovery/validation, leakage-safe
official train/test splitting, deterministic subset selection, the generic
sequence manifest, SequenceFrameDataset, and a DAVIS regression check.

Uses tiny synthetic Vimeo-90K-style directories (helpers.make_vimeo_dataset)
- the real ~82 GB dataset is never required for these tests.
"""

from __future__ import annotations

import torch

from nvc.data.sequence_dataset import SequenceFrameDataset
from nvc.data.transforms import build_eval_transform, build_train_transform
from nvc.data.validation import DatasetValidationError
from nvc.data.vimeo import (
    IncompleteVimeoSequenceError,
    SEPTUPLET_FRAME_COUNT,
    VimeoLeakageError,
    VimeoListNotFoundError,
    VimeoRootNotFoundError,
    VimeoSequencesDirNotFoundError,
    build_sequence_manifest,
    find_vimeo_root,
    load_official_split_ids,
    parse_sequence_list,
    select_deterministic_subset,
    sequence_directory,
    validate_sequence,
    vimeo_dataset_statistics,
)

import pytest

from helpers import make_tiny_manifest, make_vimeo_dataset

_TRAIN_IDS = ["00001/0001", "00001/0002", "00002/0001", "00002/0002"]
_TEST_IDS = ["00003/0001", "00003/0002"]


def _tiny_vimeo(tmp_path, **overrides):
    kwargs = {
        "train_sequence_ids": _TRAIN_IDS,
        "test_sequence_ids": _TEST_IDS,
        "width": 448, "height": 256,
    }
    kwargs.update(overrides)
    return make_vimeo_dataset(tmp_path / "vimeo_septuplet", **kwargs)


# --- Directory discovery ---


def test_find_vimeo_root_accepts_a_valid_structure(tmp_path):
    root = _tiny_vimeo(tmp_path)
    assert find_vimeo_root(root) == root


def test_find_vimeo_root_missing_directory_raises(tmp_path):
    with pytest.raises(VimeoRootNotFoundError):
        find_vimeo_root(tmp_path / "does_not_exist")


def test_find_vimeo_root_missing_sequences_dir_raises(tmp_path):
    root = tmp_path / "vimeo_septuplet"
    root.mkdir()
    (root / "sep_trainlist.txt").write_text("00001/0001\n", encoding="utf-8")

    with pytest.raises(VimeoSequencesDirNotFoundError):
        find_vimeo_root(root)


def test_error_messages_mention_how_to_fix_it(tmp_path):
    try:
        find_vimeo_root(tmp_path / "missing")
    except VimeoRootNotFoundError as exc:
        assert "README" in str(exc) or "download" in str(exc).lower()


# --- List parsing ---


def test_parse_sequence_list_reads_and_sorts_ids(tmp_path):
    list_path = tmp_path / "list.txt"
    list_path.write_text("b/2\na/1\n\nc/3\n", encoding="utf-8")

    assert parse_sequence_list(list_path) == ["a/1", "b/2", "c/3"]


def test_parse_sequence_list_missing_file_raises(tmp_path):
    with pytest.raises(VimeoListNotFoundError):
        parse_sequence_list(tmp_path / "nope.txt")


def test_parse_sequence_list_deduplicates(tmp_path):
    list_path = tmp_path / "list.txt"
    list_path.write_text("a/1\nb/2\na/1\n", encoding="utf-8")

    assert parse_sequence_list(list_path) == ["a/1", "b/2"]


def test_train_list_parsing(tmp_path):
    root = _tiny_vimeo(tmp_path)
    assert parse_sequence_list(root / "sep_trainlist.txt") == sorted(_TRAIN_IDS)


def test_test_list_parsing(tmp_path):
    root = _tiny_vimeo(tmp_path)
    assert parse_sequence_list(root / "sep_testlist.txt") == sorted(_TEST_IDS)


def test_load_official_split_ids_missing_train_list_raises(tmp_path):
    root = tmp_path / "vimeo_septuplet"
    (root / "sequences").mkdir(parents=True)
    (root / "sep_testlist.txt").write_text("00001/0001\n", encoding="utf-8")

    with pytest.raises(VimeoListNotFoundError):
        load_official_split_ids(root)


# --- Leakage prevention (mandatory) ---


def test_official_train_test_ids_are_disjoint(tmp_path):
    root = _tiny_vimeo(tmp_path)
    splits = load_official_split_ids(root)

    assert set(splits["train"]) & set(splits["test"]) == set()
    assert set(splits["train"]) == set(_TRAIN_IDS)
    assert set(splits["test"]) == set(_TEST_IDS)


def test_overlapping_lists_are_rejected(tmp_path):
    # A corrupted/hand-edited list file must stop the pipeline, not leak.
    overlapping_test_ids = [*_TEST_IDS, _TRAIN_IDS[0]]
    root = _tiny_vimeo(tmp_path, test_sequence_ids=overlapping_test_ids)

    with pytest.raises(VimeoLeakageError):
        load_official_split_ids(root)


def test_max_sequences_subset_cannot_reach_into_the_test_list(tmp_path):
    root = _tiny_vimeo(tmp_path)

    manifest = build_sequence_manifest(
        root, tmp_path / "manifest.json", split="train", max_sequences=2, seed=1,
    )

    selected_ids = {item["sequence_id"] for item in manifest["items"]}
    assert selected_ids <= set(_TRAIN_IDS)
    assert selected_ids.isdisjoint(_TEST_IDS)


# --- Sequence validation ---


def test_7_frame_sequence_validates(tmp_path):
    root = _tiny_vimeo(tmp_path)

    info = validate_sequence(root, _TRAIN_IDS[0])

    assert len(info.frame_paths) == SEPTUPLET_FRAME_COUNT
    assert all(p.is_file() for p in info.frame_paths)


def test_incomplete_sequence_raises(tmp_path):
    root = _tiny_vimeo(tmp_path, incomplete_ids=[_TRAIN_IDS[0]])

    with pytest.raises(IncompleteVimeoSequenceError):
        validate_sequence(root, _TRAIN_IDS[0])


def test_missing_sequence_directory_raises(tmp_path):
    root = _tiny_vimeo(tmp_path)

    with pytest.raises(IncompleteVimeoSequenceError):
        validate_sequence(root, "99999/9999")


def test_validate_sequence_can_probe_resolution(tmp_path):
    root = _tiny_vimeo(tmp_path, width=64, height=48)

    info = validate_sequence(root, _TRAIN_IDS[0], probe_resolution=True)

    assert info.resolution == (64, 48)


def test_sequence_directory_reflects_nested_ids(tmp_path):
    root = _tiny_vimeo(tmp_path)

    assert sequence_directory(root, "00001/0001") == root / "sequences" / "00001" / "0001"


# --- Invalid sequence entries are skipped, not fatal ---


def test_build_manifest_skips_invalid_sequences_and_records_them(tmp_path):
    root = _tiny_vimeo(tmp_path, incomplete_ids=[_TRAIN_IDS[0]])

    manifest = build_sequence_manifest(
        root, tmp_path / "manifest.json", split="train", max_sequences=None, seed=1, validate=True,
    )

    skipped_ids = {error["sequence_id"] for error in manifest["errors"]}
    assert _TRAIN_IDS[0] in skipped_ids
    written_ids = {item["sequence_id"] for item in manifest["items"]}
    assert _TRAIN_IDS[0] not in written_ids
    assert len(written_ids) == len(_TRAIN_IDS) - 1


# --- Deterministic subset selection ---


def test_max_sequences_selects_the_requested_count():
    ids = [f"g/{i:04d}" for i in range(50)]
    subset = select_deterministic_subset(ids, max_sequences=10, seed=42)

    assert len(subset) == 10
    assert set(subset) <= set(ids)


def test_subset_selection_is_deterministic_for_the_same_seed():
    ids = [f"g/{i:04d}" for i in range(50)]

    first = select_deterministic_subset(ids, max_sequences=10, seed=42)
    second = select_deterministic_subset(ids, max_sequences=10, seed=42)

    assert first == second


def test_subset_selection_differs_for_a_different_seed():
    ids = [f"g/{i:04d}" for i in range(50)]

    a = select_deterministic_subset(ids, max_sequences=10, seed=1)
    b = select_deterministic_subset(ids, max_sequences=10, seed=2)

    assert a != b


def test_subset_selection_returns_everything_when_max_sequences_is_none():
    ids = [f"g/{i:04d}" for i in range(10)]

    assert select_deterministic_subset(ids, max_sequences=None, seed=1) == sorted(ids)


def test_subset_selection_returns_everything_when_max_exceeds_available():
    ids = [f"g/{i:04d}" for i in range(5)]

    assert select_deterministic_subset(ids, max_sequences=100, seed=1) == sorted(ids)


def test_subset_selection_rejects_negative_max_sequences():
    with pytest.raises(ValueError):
        select_deterministic_subset(["a/1"], max_sequences=-1, seed=1)


# --- Dataset statistics ---


def test_vimeo_dataset_statistics_uses_list_files_not_a_filesystem_scan(tmp_path):
    root = _tiny_vimeo(tmp_path)

    stats = vimeo_dataset_statistics(root, resolution_probe_samples=1)

    assert stats["num_train_sequences"] == len(_TRAIN_IDS)
    assert stats["num_test_sequences"] == len(_TEST_IDS)
    assert stats["num_sequences_total"] == len(_TRAIN_IDS) + len(_TEST_IDS)
    assert stats["frames_per_sequence"] == SEPTUPLET_FRAME_COUNT
    assert stats["total_source_frames"] == (len(_TRAIN_IDS) + len(_TEST_IDS)) * SEPTUPLET_FRAME_COUNT
    assert len(stats["source_resolution_samples"]) == 1


# --- Generic sequence manifest ---


def test_manifest_has_one_entry_per_sequence_not_per_frame(tmp_path):
    root = _tiny_vimeo(tmp_path)

    manifest = build_sequence_manifest(
        root, tmp_path / "manifest.json", split="train", max_sequences=None, seed=1,
    )

    assert len(manifest["items"]) == len(_TRAIN_IDS)
    assert manifest["summary"]["total_frames"] == len(_TRAIN_IDS) * SEPTUPLET_FRAME_COUNT


def test_manifest_rejects_unknown_split(tmp_path):
    root = _tiny_vimeo(tmp_path)

    with pytest.raises(ValueError):
        build_sequence_manifest(
            root, tmp_path / "manifest.json", split="val", max_sequences=None, seed=1,
        )


# --- SequenceFrameDataset: sequence identity ---


def test_sequence_identity_is_exposed_per_frame(tmp_path):
    root = _tiny_vimeo(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    build_sequence_manifest(root, manifest_path, split="train", max_sequences=None, seed=1)

    dataset = SequenceFrameDataset(manifest_path, split="train")

    assert len(dataset) == len(_TRAIN_IDS) * SEPTUPLET_FRAME_COUNT
    # First 7 samples all belong to the lexicographically first sequence, in order.
    first_id = sorted(_TRAIN_IDS)[0]
    for frame_index in range(SEPTUPLET_FRAME_COUNT):
        assert dataset.sequence_id_at(frame_index) == first_id
        assert dataset.frame_index_at(frame_index) == frame_index


# --- SequenceFrameDataset: lazy loading ---


def test_dataset_construction_does_not_read_any_image(tmp_path, monkeypatch):
    root = _tiny_vimeo(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    build_sequence_manifest(root, manifest_path, split="train", max_sequences=None, seed=1)

    import nvc.data.sequence_dataset as sequence_dataset_module

    def _fail(*args, **kwargs):
        raise AssertionError("read_image_as_tensor should not be called during construction")

    monkeypatch.setattr(sequence_dataset_module, "read_image_as_tensor", _fail)

    SequenceFrameDataset(manifest_path, split="train")  # must not raise


def test_getitem_loads_the_correct_frame_lazily(tmp_path):
    root = _tiny_vimeo(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    build_sequence_manifest(root, manifest_path, split="train", max_sequences=None, seed=1)
    dataset = SequenceFrameDataset(manifest_path, split="train")

    tensor = dataset[0]

    assert tensor.shape == (3, 256, 448)
    assert tensor.dtype == torch.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(DatasetValidationError):
        SequenceFrameDataset(tmp_path / "no_manifest.json", split="train")


def test_unknown_split_raises(tmp_path):
    root = _tiny_vimeo(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    build_sequence_manifest(root, manifest_path, split="train", max_sequences=None, seed=1)

    with pytest.raises(DatasetValidationError):
        SequenceFrameDataset(manifest_path, split="test")  # this manifest only has train


# --- Crop preprocessing (no distortion, no stretching) ---


def test_random_256_training_crop_matches_target_shape(tmp_path):
    root = _tiny_vimeo(tmp_path, width=448, height=256)
    manifest_path = tmp_path / "manifest.json"
    build_sequence_manifest(root, manifest_path, split="train", max_sequences=None, seed=1)

    dataset = SequenceFrameDataset(
        manifest_path, split="train", transform=build_train_transform(256), crop_size=256,
    )

    tensor = dataset[0]
    assert tensor.shape == (3, 256, 256)


def test_deterministic_validation_crop_is_reproducible(tmp_path):
    root = _tiny_vimeo(tmp_path, width=448, height=256)
    manifest_path = tmp_path / "manifest.json"
    build_sequence_manifest(root, manifest_path, split="test", max_sequences=None, seed=1)

    dataset = SequenceFrameDataset(
        manifest_path, split="test", transform=build_eval_transform(256), crop_size=256,
    )

    first = dataset[0]
    second = dataset[0]
    assert torch.equal(first, second)
    assert first.shape == (3, 256, 256)


def test_crop_never_stretches_the_aspect_ratio(tmp_path):
    # A crop can only select a contiguous sub-region, never resize. Prove it
    # with a frame whose color varies by column: resizing would interpolate
    # those values, while a correct center crop reproduces an EXACT
    # contiguous slice of the original pixels, unchanged.
    import cv2
    import numpy as np

    from nvc.data.image_io import read_image_as_tensor

    width, height = 448, 256
    root = tmp_path / "vimeo_septuplet"
    seq_dir = root / "sequences" / "00001" / "0001"
    seq_dir.mkdir(parents=True)

    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    gradient[:, :, 0] = np.arange(width, dtype=np.uint8)  # varies left-to-right
    for i in range(1, SEPTUPLET_FRAME_COUNT + 1):
        cv2.imwrite(str(seq_dir / f"im{i}.png"), gradient)
    (root / "sep_trainlist.txt").write_text("00001/0001\n", encoding="utf-8")
    (root / "sep_testlist.txt").write_text("", encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    build_sequence_manifest(root, manifest_path, split="train", max_sequences=None, seed=1)
    dataset = SequenceFrameDataset(
        manifest_path, split="train", transform=build_eval_transform(256), crop_size=256,
    )

    original = read_image_as_tensor(seq_dir / "im1.png")  # [3, 256, 448]
    left = (width - 256) // 2
    expected_slice = original[:, :, left : left + 256]

    assert torch.equal(dataset[0], expected_slice)


# --- DAVIS regression: unaffected by Vimeo additions ---


def test_davis_style_ingestion_and_frame_dataset_still_work(tmp_path):
    from nvc.data.frame_dataset import FrameDataset
    from nvc.data.loaders import create_train_loader, create_val_loader

    manifest_path = make_tiny_manifest(tmp_path)

    train_dataset = FrameDataset(manifest_path, split="train")
    assert len(train_dataset) > 0
    tensor = train_dataset[0]
    assert tensor.dtype == torch.float32
    assert tensor.dim() == 3

    train_loader = create_train_loader(manifest_path, batch_size=2, seed=1)
    val_loader = create_val_loader(manifest_path, batch_size=2)
    batch = next(iter(train_loader))
    assert batch.shape[1] == 3
    assert len(val_loader.dataset) > 0
