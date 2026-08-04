"""Tests for Milestone 2.5: image-sequence discovery/validation, the
DatasetSource abstraction, unified ingestion (video + image sequence),
manifest generation, and source-type auto-detection.

Uses tiny synthetic image sequences and videos generated in-process - no
external dataset (e.g. DAVIS) is required for these tests.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from nvc.data.ingest import detect_source_type, ingest_dataset
from nvc.data.sequence_utils import (
    CorruptImageError,
    EmptySequenceError,
    InconsistentDimensionsError,
    SequenceNotFoundError,
    discover_sequences,
    list_sequence_images,
    probe_sequence,
)

from helpers import make_sequence, make_synthetic_video


# --- Image sequence discovery ---


def test_discover_sequences_finds_one_folder_per_sequence(tmp_path):
    root = tmp_path / "480p"
    make_sequence(root / "bear", num_frames=3)
    make_sequence(root / "camel", num_frames=4)
    (root / "not_a_sequence_file.txt").write_text("irrelevant")

    sequences = discover_sequences(root)

    # The stray file is not a directory, so it's not treated as a sequence.
    assert [s.name for s in sequences] == ["bear", "camel"]


def test_discover_sequences_missing_root_raises(tmp_path):
    with pytest.raises(SequenceNotFoundError):
        discover_sequences(tmp_path / "does_not_exist")


# --- Sequence validation ---


def test_probe_sequence_empty_folder_raises(tmp_path):
    empty_dir = tmp_path / "empty_seq"
    empty_dir.mkdir()
    with pytest.raises(EmptySequenceError):
        probe_sequence(empty_dir)


def test_probe_sequence_corrupt_image_raises(tmp_path):
    seq_dir = make_sequence(tmp_path / "seq", num_frames=2)
    (seq_dir / "00002.jpg").write_bytes(b"not a real image")

    with pytest.raises(CorruptImageError):
        probe_sequence(seq_dir)


def test_probe_sequence_inconsistent_dimensions_raises(tmp_path):
    seq_dir = make_sequence(tmp_path / "seq", num_frames=3, width=64, height=48)
    # Overwrite one frame with a different resolution.
    odd_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(seq_dir / "00001.jpg"), odd_frame)

    with pytest.raises(InconsistentDimensionsError):
        probe_sequence(seq_dir)


def test_probe_sequence_jpg_sequence_reads_metadata(tmp_path):
    seq_dir = make_sequence(tmp_path / "seq", num_frames=6, width=80, height=60, ext=".jpg")
    metadata = probe_sequence(seq_dir)

    assert metadata.filename == "seq"
    assert metadata.width == 80
    assert metadata.height == 60
    assert metadata.frame_count == 6
    assert metadata.fps is None
    assert metadata.duration_seconds is None


def test_probe_sequence_png_sequence_reads_metadata(tmp_path):
    seq_dir = make_sequence(tmp_path / "seq", num_frames=4, width=32, height=32, ext=".png")
    metadata = probe_sequence(seq_dir)

    assert metadata.frame_count == 4
    assert metadata.width == 32


def test_list_sequence_images_mixed_extensions(tmp_path):
    seq_dir = tmp_path / "mixed_seq"
    seq_dir.mkdir()
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(seq_dir / "00000.jpg"), frame)
    cv2.imwrite(str(seq_dir / "00001.png"), frame)
    cv2.imwrite(str(seq_dir / "00002.jpeg"), frame)
    (seq_dir / "readme.txt").write_text("not an image")

    images = list_sequence_images(seq_dir)

    assert [p.name for p in images] == ["00000.jpg", "00001.png", "00002.jpeg"]

    # A mixed-extension sequence with consistent dimensions still probes fine.
    metadata = probe_sequence(seq_dir)
    assert metadata.frame_count == 3


# --- Unified ingestion: image-sequence source ---


def test_ingest_dataset_image_sequence_end_to_end(tmp_path):
    root = tmp_path / "JPEGImages" / "480p"
    for name, count in [("bear", 8), ("camel", 6), ("dog", 6), ("goat", 4)]:
        make_sequence(root / name, num_frames=count, width=64, height=48)

    frames_root = tmp_path / "frames"
    manifest_path = tmp_path / "processed" / "manifest.json"

    manifest = ingest_dataset(
        input_root=root,
        frames_root=frames_root,
        manifest_path=manifest_path,
        source_type="image_sequence",
        target_width=32,
        target_height=32,
        every_n_frames=1,
        preserve_aspect_ratio=True,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=3,
    )

    assert manifest["source_type"] == "image_sequence"
    assert manifest["source_type_auto_detected"] is False
    summary = manifest["summary"]
    assert summary["total_items_found"] == 4
    assert summary["total_items_processed"] == 4
    assert summary["total_items_skipped"] == 0
    assert summary["total_frames_extracted"] == 8 + 6 + 6 + 4

    # Manifest entries use the unified schema from Task 5.
    record = manifest["items"][0]
    for key in (
        "source_type", "source_name", "split", "frame_count",
        "original_resolution", "processed_resolution", "frame_directory",
    ):
        assert key in record
    assert record["source_type"] == "image_sequence"
    assert record["original_resolution"] == [64, 48]
    assert record["processed_resolution"] == [32, 32]

    # No leakage: each sequence's frames live only under its assigned split.
    for record in manifest["items"]:
        name = record["source_name"]
        split = record["split"]
        for other_split in ("train", "val", "test"):
            matches = list((frames_root / other_split).glob(f"{name}_*.png"))
            expected = record["frame_count"] if other_split == split else 0
            assert len(matches) == expected


def test_ingest_dataset_image_sequence_skips_bad_sequences(tmp_path):
    root = tmp_path / "480p"
    make_sequence(root / "good_one", num_frames=5)
    (root / "empty_one").mkdir()  # no images -> should be skipped, not fatal
    bad = make_sequence(root / "bad_dims", num_frames=3)
    cv2.imwrite(str(bad / "00099.jpg"), np.zeros((999, 999, 3), dtype=np.uint8))

    manifest = ingest_dataset(
        input_root=root,
        frames_root=tmp_path / "frames",
        manifest_path=tmp_path / "processed" / "manifest.json",
        source_type="image_sequence",
        target_width=16,
        target_height=16,
        every_n_frames=1,
        preserve_aspect_ratio=True,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=1,
    )

    assert manifest["summary"]["total_items_found"] == 3
    assert manifest["summary"]["total_items_processed"] == 1
    assert manifest["summary"]["total_items_skipped"] == 2
    skipped_names = {err["source_name"] for err in manifest["errors"]}
    assert skipped_names == {"empty_one", "bad_dims"}


# --- Unified ingestion: video source (compatibility with the existing pipeline) ---


def test_ingest_dataset_video_source_matches_video_pipeline_behavior(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_synthetic_video(raw_dir / "clip_a.mp4", num_frames=10)
    make_synthetic_video(raw_dir / "clip_b.mp4", num_frames=6)

    manifest = ingest_dataset(
        input_root=raw_dir,
        frames_root=tmp_path / "frames",
        manifest_path=tmp_path / "processed" / "manifest.json",
        source_type="video",
        target_width=32,
        target_height=32,
        every_n_frames=1,
        preserve_aspect_ratio=True,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=1,
    )

    assert manifest["source_type"] == "video"
    assert manifest["summary"]["total_items_found"] == 2
    assert manifest["summary"]["total_frames_extracted"] == 16
    assert all(item["source_type"] == "video" for item in manifest["items"])


# --- Auto source-type detection ---


def test_detect_source_type_video_when_videos_present(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_synthetic_video(raw_dir / "clip.mp4", num_frames=3)

    assert detect_source_type(raw_dir) == "video"


def test_detect_source_type_image_sequence_when_subfolders_have_images(tmp_path):
    root = tmp_path / "480p"
    make_sequence(root / "bear", num_frames=3)

    assert detect_source_type(root) == "image_sequence"


def test_detect_source_type_raises_when_ambiguous(tmp_path):
    empty_dir = tmp_path / "nothing_here"
    empty_dir.mkdir()

    with pytest.raises(ValueError):
        detect_source_type(empty_dir)


def test_ingest_dataset_auto_detects_and_records_it(tmp_path):
    root = tmp_path / "480p"
    make_sequence(root / "bear", num_frames=3)
    make_sequence(root / "camel", num_frames=3)

    manifest = ingest_dataset(
        input_root=root,
        frames_root=tmp_path / "frames",
        manifest_path=tmp_path / "processed" / "manifest.json",
        source_type=None,  # auto-detect
        target_width=16,
        target_height=16,
        every_n_frames=1,
        preserve_aspect_ratio=True,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=1,
    )

    assert manifest["source_type"] == "image_sequence"
    assert manifest["source_type_auto_detected"] is True
