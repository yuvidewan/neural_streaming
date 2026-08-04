"""Tests for Milestone 2: video metadata, frame extraction/sampling,
deterministic splitting, leakage prevention, manifest generation, and
error handling.

Uses tiny synthetic videos generated in-process (cv2.VideoWriter) instead
of requiring a real external dataset.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from nvc.data.dataset_prep import assign_splits, discover_videos, prepare_dataset
from nvc.data.frame_extraction import extract_frames, resize_frame
from nvc.data.video_utils import (
    CorruptVideoError,
    UnsupportedVideoFormatError,
    VideoNotFoundError,
    probe_video,
)

from helpers import make_synthetic_video


# --- Video metadata (video_utils.probe_video) ---


def test_probe_video_reads_expected_metadata(tmp_path):
    video_path = make_synthetic_video(
        tmp_path / "clip.mp4", num_frames=15, width=64, height=48, fps=10.0
    )
    metadata = probe_video(video_path)

    assert metadata.filename == "clip.mp4"
    assert metadata.width == 64
    assert metadata.height == 48
    assert metadata.fps == pytest.approx(10.0, rel=0.05)
    assert metadata.frame_count == 15
    assert metadata.duration_seconds == pytest.approx(1.5, rel=0.1)
    assert metadata.file_size_bytes > 0


def test_probe_video_missing_path_raises(tmp_path):
    with pytest.raises(VideoNotFoundError):
        probe_video(tmp_path / "does_not_exist.mp4")


def test_probe_video_unsupported_extension_raises(tmp_path):
    fake = tmp_path / "clip.txt"
    fake.write_text("not a video")
    with pytest.raises(UnsupportedVideoFormatError):
        probe_video(fake, allowed_extensions=[".mp4", ".avi"])


def test_probe_video_corrupt_file_raises(tmp_path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a real video file" * 10)
    with pytest.raises(CorruptVideoError):
        probe_video(corrupt)


# --- Frame extraction & sampling ---


def test_extract_frames_every_frame(tmp_path):
    video_path = make_synthetic_video(tmp_path / "clip.mp4", num_frames=10)
    out_dir = tmp_path / "out"
    result = extract_frames(video_path, out_dir, target_width=32, target_height=32, every_n_frames=1)

    assert result.extracted_frame_count == 10
    saved = sorted(out_dir.glob("*.png"))
    assert len(saved) == 10
    assert saved[0].name == "clip_000001.png"
    assert saved[-1].name == "clip_000010.png"


def test_extract_frames_sampling_every_n(tmp_path):
    video_path = make_synthetic_video(tmp_path / "clip.mp4", num_frames=10)
    out_dir = tmp_path / "out"
    result = extract_frames(video_path, out_dir, target_width=32, target_height=32, every_n_frames=3)

    # Original frame indices 0, 3, 6, 9 are kept -> 4 frames.
    assert result.extracted_frame_count == 4
    assert len(list(out_dir.glob("*.png"))) == 4


def test_extract_frames_target_dimensions(tmp_path):
    video_path = make_synthetic_video(tmp_path / "clip.mp4", num_frames=3, width=64, height=48)
    out_dir = tmp_path / "out"
    extract_frames(video_path, out_dir, target_width=100, target_height=50, every_n_frames=1)

    saved = sorted(out_dir.glob("*.png"))
    img = cv2.imread(str(saved[0]))
    assert img.shape[1] == 100  # width
    assert img.shape[0] == 50  # height


def test_resize_frame_preserve_aspect_ratio_no_distortion():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    out = resize_frame(frame, target_width=32, target_height=32, preserve_aspect_ratio=True)
    assert out.shape == (32, 32, 3)


def test_resize_frame_no_preserve_aspect_ratio_stretches():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    out = resize_frame(frame, target_width=20, target_height=20, preserve_aspect_ratio=False)
    assert out.shape == (20, 20, 3)


# --- Deterministic splitting & leakage prevention ---


def test_assign_splits_is_deterministic_for_same_seed(tmp_path):
    videos = [tmp_path / f"video_{i}.mp4" for i in range(10)]
    split_a = assign_splits(videos, 0.8, 0.1, 0.1, seed=42)
    split_b = assign_splits(videos, 0.8, 0.1, 0.1, seed=42)
    assert split_a == split_b


def test_assign_splits_covers_every_video_exactly_once(tmp_path):
    videos = [tmp_path / f"video_{i}.mp4" for i in range(7)]
    assignment = assign_splits(videos, 0.8, 0.1, 0.1, seed=1)
    assert set(assignment.keys()) == set(videos)
    assert all(split in ("train", "val", "test") for split in assignment.values())


def test_assign_splits_rejects_bad_ratios(tmp_path):
    videos = [tmp_path / "a.mp4"]
    with pytest.raises(ValueError):
        assign_splits(videos, 0.5, 0.3, 0.3, seed=1)  # sums to 1.1


def test_prepare_dataset_no_video_leakage_across_splits(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for i in range(6):
        make_synthetic_video(raw_dir / f"video_{i}.mp4", num_frames=8)

    frames_root = tmp_path / "frames"
    manifest_path = tmp_path / "processed" / "manifest.json"

    manifest = prepare_dataset(
        raw_dir=raw_dir,
        frames_root=frames_root,
        manifest_path=manifest_path,
        target_width=32,
        target_height=32,
        every_n_frames=1,
        preserve_aspect_ratio=True,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=7,
        supported_extensions=(".mp4",),
    )

    # Every video's frames must exist in exactly one split directory.
    for record in manifest["videos"]:
        stem = record["source_video"].rsplit(".", 1)[0]
        split = record["split"]
        for other_split in ("train", "val", "test"):
            matches = list((frames_root / other_split).glob(f"{stem}_*.png"))
            if other_split == split:
                assert len(matches) == record["extracted_frame_count"]
            else:
                assert len(matches) == 0


# --- Manifest generation ---


def test_prepare_dataset_manifest_is_valid_and_consistent(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_synthetic_video(raw_dir / "a.mp4", num_frames=6)
    make_synthetic_video(raw_dir / "b.mp4", num_frames=6)

    frames_root = tmp_path / "frames"
    manifest_path = tmp_path / "processed" / "manifest.json"

    manifest = prepare_dataset(
        raw_dir=raw_dir,
        frames_root=frames_root,
        manifest_path=manifest_path,
        target_width=32,
        target_height=32,
        every_n_frames=1,
        preserve_aspect_ratio=True,
        train_ratio=0.5,
        val_ratio=0.5,
        test_ratio=0.0,
        seed=1,
        supported_extensions=(".mp4",),
    )

    assert manifest_path.exists()
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk == manifest

    summary = manifest["summary"]
    assert summary["total_videos_found"] == 2
    assert summary["total_videos_processed"] == 2
    assert summary["total_videos_skipped"] == 0
    assert sum(summary["frames_per_split"].values()) == summary["total_frames_extracted"]
    assert summary["total_frames_extracted"] == 12


# --- Error handling: corrupt/invalid videos are skipped, not fatal ---


def test_prepare_dataset_skips_corrupt_video_but_processes_others(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_synthetic_video(raw_dir / "good.mp4", num_frames=5)
    (raw_dir / "bad.mp4").write_bytes(b"garbage not a video")

    frames_root = tmp_path / "frames"
    manifest_path = tmp_path / "processed" / "manifest.json"

    manifest = prepare_dataset(
        raw_dir=raw_dir,
        frames_root=frames_root,
        manifest_path=manifest_path,
        target_width=32,
        target_height=32,
        every_n_frames=1,
        preserve_aspect_ratio=True,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        seed=1,
        supported_extensions=(".mp4",),
    )

    assert manifest["summary"]["total_videos_processed"] == 1
    assert manifest["summary"]["total_videos_skipped"] == 1
    assert manifest["errors"][0]["source_video"] == "bad.mp4"


def test_prepare_dataset_raises_on_stem_collision(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_synthetic_video(raw_dir / "clip.mp4", num_frames=3)
    (raw_dir / "clip.avi").write_bytes(b"not a real avi")

    with pytest.raises(ValueError, match="duplicate filenames"):
        prepare_dataset(
            raw_dir=raw_dir,
            frames_root=tmp_path / "frames",
            manifest_path=tmp_path / "processed" / "manifest.json",
            target_width=32,
            target_height=32,
            every_n_frames=1,
            preserve_aspect_ratio=True,
            train_ratio=1.0,
            val_ratio=0.0,
            test_ratio=0.0,
            seed=1,
            supported_extensions=(".mp4", ".avi"),
        )


def test_discover_videos_filters_unsupported_extensions(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    make_synthetic_video(raw_dir / "clip.mp4", num_frames=2)
    (raw_dir / "notes.txt").write_text("irrelevant")

    videos = discover_videos(raw_dir, (".mp4",))
    assert [v.name for v in videos] == ["clip.mp4"]
