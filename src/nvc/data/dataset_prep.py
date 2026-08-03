"""Dataset-level orchestration: video discovery, video-level train/val/test
splitting, per-video frame extraction, and manifest/statistics generation.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from nvc.data.frame_extraction import extract_frames
from nvc.data.video_utils import VideoError

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SkippedVideo:
    """A video that could not be processed, and why."""

    video_path: Path
    reason: str


def discover_videos(raw_dir: Path, supported_extensions: tuple[str, ...]) -> list[Path]:
    """List candidate video files directly inside raw_dir, sorted for determinism.

    Sorting matters: relying on filesystem iteration order would make the
    later seeded split non-reproducible across platforms.
    """
    allowed = {ext.lower() for ext in supported_extensions}
    videos = [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed]
    return sorted(videos)


def _check_for_stem_collisions(videos: list[Path]) -> None:
    """Fail fast if two videos would produce colliding frame filenames.

    Extracted frames are named "{video.stem}_NNNNNN.png". Two source
    videos with the same stem (e.g. clip.mp4 and clip.mov) would silently
    overwrite each other's frames, so this is rejected up front instead.
    """
    seen: dict[str, Path] = {}
    collisions: list[tuple[Path, Path]] = []
    for video in videos:
        if video.stem in seen:
            collisions.append((seen[video.stem], video))
        else:
            seen[video.stem] = video
    if collisions:
        details = "; ".join(f"{a.name} vs {b.name}" for a, b in collisions)
        raise ValueError(
            "Videos with duplicate filenames (ignoring extension) would "
            f"overwrite each other's extracted frames: {details}. Rename one of the files."
        )


def assign_splits(
    videos: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[Path, str]:
    """Deterministically assign each video to train/val/test.

    Splitting happens at the video level (not per-frame): all frames from
    one source video always land in the same split. This is what prevents
    data leakage between splits - it is not an incidental detail.
    """
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {ratio_sum}"
        )
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("train_ratio, val_ratio, and test_ratio must each be >= 0")

    ordered = sorted(videos)  # fixed base order before shuffling, for reproducibility
    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    n_train = min(round(total * train_ratio), total)
    n_val = min(round(total * val_ratio), total - n_train)
    # test gets the remainder, guaranteeing every video is assigned exactly once

    assignment: dict[Path, str] = {}
    for video in shuffled[:n_train]:
        assignment[video] = "train"
    for video in shuffled[n_train : n_train + n_val]:
        assignment[video] = "val"
    for video in shuffled[n_train + n_val :]:
        assignment[video] = "test"
    return assignment


def _relpath_posix(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start)).as_posix()


def prepare_dataset(
    raw_dir: Path,
    frames_root: Path,
    manifest_path: Path,
    *,
    target_width: int,
    target_height: int,
    every_n_frames: int,
    preserve_aspect_ratio: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    supported_extensions: tuple[str, ...],
) -> dict:
    """Run the full dataset preparation pipeline and return the manifest dict.

    Discovers videos in raw_dir, splits them at the video level, extracts
    resized frames for each into frames_root/{train,val,test}, writes a
    JSON manifest to manifest_path, and returns that same manifest. Videos
    that fail to open/read are skipped and recorded under "errors" instead
    of aborting the whole run.
    """
    raw_dir = Path(raw_dir)
    frames_root = Path(frames_root)
    manifest_path = Path(manifest_path)

    videos = discover_videos(raw_dir, supported_extensions)
    if not videos:
        raise ValueError(f"No supported video files found in {raw_dir}")

    _check_for_stem_collisions(videos)
    assignment = assign_splits(videos, train_ratio, val_ratio, test_ratio, seed)

    video_records = []
    skipped: list[SkippedVideo] = []

    for video_path in videos:
        split = assignment[video_path]
        output_dir = frames_root / split
        try:
            result = extract_frames(
                video_path,
                output_dir,
                target_width=target_width,
                target_height=target_height,
                every_n_frames=every_n_frames,
                preserve_aspect_ratio=preserve_aspect_ratio,
                allowed_extensions=supported_extensions,
            )
        except VideoError as exc:
            skipped.append(SkippedVideo(video_path=video_path, reason=str(exc)))
            continue

        video_records.append(
            {
                "source_video": video_path.name,
                "split": split,
                "original_width": result.metadata.width,
                "original_height": result.metadata.height,
                "original_fps": result.metadata.fps,
                "original_frame_count": result.metadata.frame_count,
                "extracted_frame_count": result.extracted_frame_count,
                "every_n_frames": result.every_n_frames,
                "target_width": result.target_width,
                "target_height": result.target_height,
                "preserve_aspect_ratio": result.preserve_aspect_ratio,
                "frame_output_dir": _relpath_posix(output_dir, manifest_path.parent),
                "frame_filename_pattern": f"{result.filename_prefix}_{{index:06d}}.png",
                "output_bytes": result.total_output_bytes,
            }
        )

    frames_per_split = {name: 0 for name in SPLIT_NAMES}
    videos_per_split = {name: 0 for name in SPLIT_NAMES}
    total_output_bytes = 0
    for record in video_records:
        frames_per_split[record["split"]] += record["extracted_frame_count"]
        videos_per_split[record["split"]] += 1
        total_output_bytes += record["output_bytes"]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "target_width": target_width,
            "target_height": target_height,
            "every_n_frames": every_n_frames,
            "preserve_aspect_ratio": preserve_aspect_ratio,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_seed": seed,
            "supported_video_extensions": list(supported_extensions),
        },
        "videos": video_records,
        "errors": [{"source_video": s.video_path.name, "message": s.reason} for s in skipped],
        "summary": {
            "total_videos_found": len(videos),
            "total_videos_processed": len(video_records),
            "total_videos_skipped": len(skipped),
            "videos_per_split": videos_per_split,
            "frames_per_split": frames_per_split,
            "total_frames_extracted": sum(frames_per_split.values()),
            "total_output_bytes": total_output_bytes,
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
