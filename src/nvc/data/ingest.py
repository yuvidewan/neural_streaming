"""Source-agnostic dataset ingestion: video files and image-sequence
folders both flow through the same discovery -> split -> process ->
manifest pipeline, via the DatasetSource strategy in sources.py.

This is the entry point scripts/prepare_dataset.py uses. It reuses
dataset_prep.assign_splits() and relpath_posix() directly rather than
reimplementing splitting - the video-only Milestone 2 pipeline in
dataset_prep.py is unchanged and still works independently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from nvc.data.dataset_prep import SPLIT_NAMES, assign_splits, relpath_posix
from nvc.data.errors import DatasetSourceError
from nvc.data.sequence_utils import DEFAULT_IMAGE_EXTENSIONS
from nvc.data.sources import DatasetSource, ImageSequenceDatasetSource, VideoDatasetSource

DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
SOURCE_TYPES = ("video", "image_sequence")


@dataclass
class SkippedItem:
    """A dataset item that could not be processed, and why."""

    item_path: Path
    reason: str


def detect_source_type(
    input_root: Path,
    video_extensions: tuple[str, ...] = DEFAULT_VIDEO_EXTENSIONS,
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
) -> str:
    """Best-effort guess at whether input_root holds videos or image sequences.

    Heuristic: if input_root directly contains a supported video file, it's
    "video" (matches Milestone 2's data/raw/ layout). Otherwise, if any
    direct subdirectory contains supported images, it's "image_sequence"
    (matches DAVIS-style JPEGImages/480p/<sequence>/ layouts). Raises
    ValueError if neither pattern matches, so callers fail with a clear
    message instead of silently guessing wrong.
    """
    if not input_root.is_dir():
        raise ValueError(f"Input path does not exist or is not a directory: {input_root}")

    video_exts = {ext.lower() for ext in video_extensions}
    if any(p.is_file() and p.suffix.lower() in video_exts for p in input_root.iterdir()):
        return "video"

    image_exts = {ext.lower() for ext in image_extensions}
    for sub in input_root.iterdir():
        if sub.is_dir() and any(
            p.is_file() and p.suffix.lower() in image_exts for p in sub.iterdir()
        ):
            return "image_sequence"

    raise ValueError(
        f"Could not auto-detect a dataset type in {input_root}: no video files "
        "directly inside it, and no subfolders containing supported images. "
        "Pass --source-type explicitly."
    )


def _build_source(
    source_type: str, video_extensions: tuple[str, ...], image_extensions: tuple[str, ...]
) -> DatasetSource:
    if source_type == "video":
        return VideoDatasetSource(video_extensions)
    if source_type == "image_sequence":
        return ImageSequenceDatasetSource(image_extensions)
    raise ValueError(
        f"Unknown source_type {source_type!r} (expected one of {SOURCE_TYPES})"
    )


def _check_for_key_collisions(items: list[Path], source: DatasetSource) -> None:
    """Fail fast if two items would produce colliding output filenames."""
    seen: dict[str, Path] = {}
    collisions: list[tuple[Path, Path]] = []
    for item in items:
        key = source.item_key(item)
        if key in seen:
            collisions.append((seen[key], item))
        else:
            seen[key] = item
    if collisions:
        details = "; ".join(f"{a.name} vs {b.name}" for a, b in collisions)
        raise ValueError(
            "Items with duplicate names would overwrite each other's extracted "
            f"frames: {details}. Rename one of them."
        )


def ingest_dataset(
    input_root: Path,
    frames_root: Path,
    manifest_path: Path,
    *,
    source_type: str | None,
    target_width: int,
    target_height: int,
    every_n_frames: int,
    preserve_aspect_ratio: bool,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    video_extensions: tuple[str, ...] = DEFAULT_VIDEO_EXTENSIONS,
    image_extensions: tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
) -> dict:
    """Discover, split, extract, and describe a dataset regardless of source.

    If source_type is None, it is auto-detected from input_root. Splitting
    happens at the item level (one whole video, or one whole sequence
    folder, per split) via the same assign_splits() Milestone 2 uses, so
    leakage prevention behaves identically for both source types. Items
    that fail validation are skipped and recorded under "errors" rather
    than aborting the run.
    """
    input_root = Path(input_root)
    frames_root = Path(frames_root)
    manifest_path = Path(manifest_path)

    auto_detected = source_type is None
    resolved_source_type = (
        detect_source_type(input_root, video_extensions, image_extensions)
        if auto_detected
        else source_type
    )

    source = _build_source(resolved_source_type, video_extensions, image_extensions)

    items = source.discover(input_root)
    if not items:
        raise ValueError(f"No {resolved_source_type} items found in {input_root}")

    _check_for_key_collisions(items, source)
    assignment = assign_splits(items, train_ratio, val_ratio, test_ratio, seed)

    records: list[dict] = []
    skipped: list[SkippedItem] = []

    for item in items:
        split = assignment[item]
        output_dir = frames_root / split
        try:
            result = source.process(
                item,
                output_dir,
                target_width=target_width,
                target_height=target_height,
                every_n_frames=every_n_frames,
                preserve_aspect_ratio=preserve_aspect_ratio,
            )
        except DatasetSourceError as exc:
            skipped.append(SkippedItem(item_path=item, reason=str(exc)))
            continue

        records.append(
            {
                "source_type": source.source_type,
                "source_name": source.item_key(item),
                "split": split,
                "frame_count": result.extracted_frame_count,
                "original_resolution": [result.metadata.width, result.metadata.height],
                "processed_resolution": [result.target_width, result.target_height],
                "frame_directory": relpath_posix(output_dir, manifest_path.parent),
                "frame_filename_pattern": f"{result.filename_prefix}_{{index:06d}}.png",
                "every_n_frames": result.every_n_frames,
                "preserve_aspect_ratio": result.preserve_aspect_ratio,
                "original_fps": result.metadata.fps,
                "original_frame_count": result.metadata.frame_count,
                "output_bytes": result.total_output_bytes,
            }
        )

    frames_per_split = {name: 0 for name in SPLIT_NAMES}
    items_per_split = {name: 0 for name in SPLIT_NAMES}
    total_output_bytes = 0
    for record in records:
        frames_per_split[record["split"]] += record["frame_count"]
        items_per_split[record["split"]] += 1
        total_output_bytes += record["output_bytes"]

    total_frames = sum(frames_per_split.values())

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_type": resolved_source_type,
        "source_type_auto_detected": auto_detected,
        "settings": {
            "target_width": target_width,
            "target_height": target_height,
            "every_n_frames": every_n_frames,
            "preserve_aspect_ratio": preserve_aspect_ratio,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_seed": seed,
            "video_extensions": list(video_extensions),
            "image_extensions": list(image_extensions),
        },
        "items": records,
        "errors": [
            {"source_name": source.item_key(s.item_path), "message": s.reason} for s in skipped
        ],
        "summary": {
            "total_items_found": len(items),
            "total_items_processed": len(records),
            "total_items_skipped": len(skipped),
            "items_per_split": items_per_split,
            "frames_per_split": frames_per_split,
            "total_frames_extracted": total_frames,
            "average_frames_per_item": (total_frames / len(records)) if records else 0,
            "total_output_bytes": total_output_bytes,
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
