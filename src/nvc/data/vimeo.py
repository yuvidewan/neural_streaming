"""Vimeo-90K Septuplet dataset support: discovery, leakage-safe train/test
splitting, deterministic subsetting, and a lightweight sequence manifest.

This is the ONLY module that knows anything about Vimeo-90K's directory
layout. Everything downstream - SequenceFrameDataset, the DataLoader
factories, the training/model code - talks only to the generic sequence
manifest schema documented on `build_sequence_manifest`, never to a Vimeo
path directly. A future dataset (e.g. a different septuplet-style corpus)
only needs its own discovery module producing the same manifest schema;
nothing else in the pipeline would need to change.

WHY NOT THE EXISTING ingest.py PIPELINE
----------------------------------------
`nvc.data.ingest.ingest_dataset()` (Milestone 2/2.5) resizes and re-saves
every frame as a new PNG under `data/frames/`. That is the right approach
for DAVIS (6,208 frames), but running it over Vimeo-90K (~642,000 source
frames, ~82 GB) would silently create a second, ~82 GB copy of the dataset
on disk. This module instead builds a manifest that *references* the
original Vimeo files by (sequence_id, filename) and leaves resizing to a
crop applied lazily at __getitem__ time (see sequence_dataset.py) - the
source images are never copied or re-encoded.

DIRECTORY STRUCTURE
--------------------
The official Vimeo-90K Septuplet release looks like:

    vimeo_septuplet/
    ├── sequences/
    │   └── <group>/<sequence>/{im1.png, ..., im7.png}   (e.g. 00001/0001/)
    ├── sep_trainlist.txt      one "<group>/<sequence>" id per line
    └── sep_testlist.txt       one "<group>/<sequence>" id per line

Nothing here hardcodes the "<group>/<sequence>" two-level nesting: a
sequence's id is whatever relative path string appears in the official
list file, and its directory is `sequences/<that string>`. This also means
a differently-nested archive (or a single-level one, as a simplified
example structure might show) works unmodified, as long as the list files
name real subdirectories of `sequences/`.

LEAKAGE PREVENTION
-------------------
`sep_trainlist.txt` and `sep_testlist.txt` are treated as authoritative and
are never re-split, merged, or shuffled together. `load_official_split_ids`
asserts the two id sets are disjoint before returning them - a defensive
check on top of trusting the official files, not a substitute for it.
Subset selection (`select_deterministic_subset`) operates within one
split's id list only, so requesting a training subset can never reach into
the test list.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2

from nvc.data.dataset_prep import relpath_posix
from nvc.data.errors import DatasetSourceError

DATASET_NAME = "vimeo90k_septuplet"
SOURCE_TYPE = "vimeo_sequence"
SEQUENCES_DIRNAME = "sequences"
TRAIN_LIST_FILENAME = "sep_trainlist.txt"
TEST_LIST_FILENAME = "sep_testlist.txt"
SEPTUPLET_FRAME_COUNT = 7
SEPTUPLET_FILENAMES = tuple(f"im{i}.png" for i in range(1, SEPTUPLET_FRAME_COUNT + 1))
OFFICIAL_SPLITS = ("train", "test")

# How many sequences vimeo_dataset_statistics() opens an image for, to
# report actual source resolution without scanning the whole dataset.
_DEFAULT_RESOLUTION_PROBE_SAMPLES = 3


class VimeoDatasetError(DatasetSourceError):
    """Base class for all Vimeo-90K discovery/validation errors."""


class VimeoRootNotFoundError(VimeoDatasetError):
    """The given Vimeo root does not exist or is not a directory."""


class VimeoSequencesDirNotFoundError(VimeoDatasetError):
    """The root exists but has no sequences/ subdirectory."""


class VimeoListNotFoundError(VimeoDatasetError):
    """sep_trainlist.txt or sep_testlist.txt is missing."""


class VimeoLeakageError(VimeoDatasetError):
    """The official train and test id lists are not disjoint.

    This should never happen with an unmodified official release; it exists
    as a defense-in-depth check, not as evidence this is expected.
    """


class IncompleteVimeoSequenceError(VimeoDatasetError):
    """A sequence directory is missing one or more of the 7 required frames."""


@dataclass(frozen=True)
class VimeoSequenceInfo:
    sequence_id: str
    frame_paths: tuple[Path, ...]
    resolution: tuple[int, int] | None  # (width, height); None if not probed


def find_vimeo_root(root: str | Path) -> Path:
    """Validate that `root` looks like a Vimeo-90K Septuplet root.

    Checks the root directory and its sequences/ subdirectory exist. Does
    NOT require the list files to be present (a caller that only needs
    directory structure, e.g. a fallback filesystem scan, can still work
    without them) - use `load_official_split_ids` when the lists are
    required.
    """
    root = Path(root)
    if not root.is_dir():
        raise VimeoRootNotFoundError(
            f"Vimeo-90K root not found: {root}. Download the Vimeo-90K Septuplet "
            "dataset and point --vimeo-root (or configs/default.json's "
            "'vimeo_root') at the extracted folder. See README.md, "
            "'Large-Scale Training Dataset', for download and layout details."
        )
    sequences_dir = root / SEQUENCES_DIRNAME
    if not sequences_dir.is_dir():
        raise VimeoSequencesDirNotFoundError(
            f"Vimeo-90K root {root} has no '{SEQUENCES_DIRNAME}/' subdirectory. "
            f"Expected {sequences_dir}. Check the archive was extracted correctly "
            "and --vimeo-root points at the folder that directly contains "
            f"'{SEQUENCES_DIRNAME}/', '{TRAIN_LIST_FILENAME}', and '{TEST_LIST_FILENAME}'."
        )
    return root


def parse_sequence_list(list_path: Path) -> list[str]:
    """Parse a sep_*list.txt file into a sorted, de-duplicated id list.

    One id per line (e.g. "00001/0001"); blank lines are skipped. Sorted
    for the same reason discover_videos()/discover_sequences() sort their
    results: reproducible order independent of filesystem/file quirks.
    """
    if not list_path.is_file():
        raise VimeoListNotFoundError(
            f"Vimeo-90K list file not found: {list_path}. This file ships with "
            "the official dataset - re-download or re-extract if it's missing."
        )
    lines = list_path.read_text(encoding="utf-8").splitlines()
    ids = {line.strip() for line in lines if line.strip()}
    return sorted(ids)


def load_official_split_ids(root: Path) -> dict[str, list[str]]:
    """Load sep_trainlist.txt and sep_testlist.txt as the authoritative split.

    Raises VimeoLeakageError if the two lists share any id - a corrupted or
    hand-edited list file should stop the pipeline, not silently leak test
    sequences into training.
    """
    train_ids = parse_sequence_list(root / TRAIN_LIST_FILENAME)
    test_ids = parse_sequence_list(root / TEST_LIST_FILENAME)

    overlap = set(train_ids) & set(test_ids)
    if overlap:
        sample = ", ".join(sorted(overlap)[:5])
        raise VimeoLeakageError(
            f"{len(overlap)} sequence id(s) appear in BOTH {TRAIN_LIST_FILENAME} "
            f"and {TEST_LIST_FILENAME} (e.g. {sample}). Refusing to proceed - "
            "this would leak test sequences into training."
        )
    return {"train": train_ids, "test": test_ids}


def sequence_directory(root: Path, sequence_id: str) -> Path:
    return root / SEQUENCES_DIRNAME / sequence_id


def validate_sequence(
    root: Path, sequence_id: str, *, probe_resolution: bool = False
) -> VimeoSequenceInfo:
    """Check that a sequence directory has all 7 required frame files.

    With probe_resolution=True, also opens im1.png (only that one frame,
    not all 7) to report its resolution - a cheap per-sequence check, not a
    full-dataset scan.
    """
    seq_dir = sequence_directory(root, sequence_id)
    if not seq_dir.is_dir():
        raise IncompleteVimeoSequenceError(
            f"Sequence directory not found: {seq_dir} (id {sequence_id!r})"
        )

    frame_paths = tuple(seq_dir / name for name in SEPTUPLET_FILENAMES)
    missing = [p.name for p in frame_paths if not p.is_file()]
    if missing:
        raise IncompleteVimeoSequenceError(
            f"Sequence '{sequence_id}' is missing {len(missing)} of "
            f"{SEPTUPLET_FRAME_COUNT} required frames: {', '.join(missing)} "
            f"(directory: {seq_dir})"
        )

    resolution = None
    if probe_resolution:
        image = cv2.imread(str(frame_paths[0]))
        if image is None:
            raise IncompleteVimeoSequenceError(
                f"Could not read {frame_paths[0]} (corrupt or unsupported image)"
            )
        height, width = image.shape[:2]
        resolution = (width, height)

    return VimeoSequenceInfo(sequence_id=sequence_id, frame_paths=frame_paths, resolution=resolution)


def select_deterministic_subset(
    sequence_ids: list[str], max_sequences: int | None, seed: int
) -> list[str]:
    """Deterministically select up to `max_sequences` ids from a sorted list.

    Sorted input -> seeded shuffle -> truncate, mirroring
    dataset_prep.assign_splits()'s reproducibility pattern. Selection uses
    only ids and the seed - never anything about compression quality or
    model performance - so results cannot be "tuned" toward favorable
    frames by construction.
    """
    if max_sequences is not None and max_sequences < 0:
        raise ValueError(f"max_sequences must be >= 0, got {max_sequences}")

    ordered = sorted(sequence_ids)
    if max_sequences is None or max_sequences >= len(ordered):
        return ordered

    shuffled = ordered.copy()
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:max_sequences])


def vimeo_dataset_statistics(
    root: str | Path, *, resolution_probe_samples: int = _DEFAULT_RESOLUTION_PROBE_SAMPLES
) -> dict:
    """Report dataset-level statistics from the official list files.

    Deliberately does NOT walk the sequences/ directory tree (which can
    hold 90,000+ subdirectories) - counts come from the list files, which
    are authoritative and orders of magnitude cheaper to read. Only
    `resolution_probe_samples` actual images are opened, to report real
    (not assumed) source resolution.
    """
    root = find_vimeo_root(root)
    splits = load_official_split_ids(root)

    probe_ids = (splits["train"] + splits["test"])[:resolution_probe_samples]
    resolutions = []
    unreadable = []
    for sequence_id in probe_ids:
        try:
            info = validate_sequence(root, sequence_id, probe_resolution=True)
            resolutions.append(info.resolution)
        except VimeoDatasetError as exc:
            unreadable.append({"sequence_id": sequence_id, "message": str(exc)})

    total_sequences = len(splits["train"]) + len(splits["test"])
    return {
        "dataset": DATASET_NAME,
        "vimeo_root": str(root),
        "num_sequences_total": total_sequences,
        "num_train_sequences": len(splits["train"]),
        "num_test_sequences": len(splits["test"]),
        "frames_per_sequence": SEPTUPLET_FRAME_COUNT,
        "total_source_frames": total_sequences * SEPTUPLET_FRAME_COUNT,
        "source_resolution_samples": resolutions,
        "source_resolution_probe_unreadable": unreadable,
        "note": (
            "Sequence counts come from sep_trainlist.txt/sep_testlist.txt, not a "
            "filesystem scan. total_source_frames assumes every listed sequence has "
            f"the full {SEPTUPLET_FRAME_COUNT} frames - pass validate=True to "
            "build_sequence_manifest (or run with --full-scan) to confirm that "
            "for every sequence rather than a small sample."
        ),
    }


def build_sequence_manifest(
    root: str | Path,
    manifest_path: str | Path,
    *,
    split: str,
    max_sequences: int | None,
    seed: int,
    validate: bool | None = None,
    probe_resolution: bool = False,
) -> dict:
    """Build and write a generic sequence manifest for one official split.

    `validate=None` (default) validates automatically when a subset is
    requested (max_sequences is not None) since that bounds the cost to
    `max_sequences` sequence-directory checks; pass True/False to override.
    Invalid sequences are skipped and recorded under "errors", exactly like
    ingest.ingest_dataset() does for DAVIS - one bad sequence does not abort
    the whole run.

    MANIFEST SCHEMA (dataset-agnostic; consumed by SequenceFrameDataset):
        {
          "dataset": str, "source_type": str,
          "dataset_root": str (relative to the manifest's directory),
          "sequences_dirname": str,
          "settings": {...},
          "items": [
            {"dataset", "source_type", "sequence_id", "split",
             "frame_filenames": [...], "frame_count": int,
             "original_resolution": [w, h] | null}
          ],
          "errors": [{"sequence_id", "message"}],
          "summary": {...},
        }
    One entry per SEQUENCE, not per frame - for the full dataset that is
    tens of thousands of entries rather than hundreds of thousands.
    """
    if split not in OFFICIAL_SPLITS:
        raise ValueError(f"split must be one of {OFFICIAL_SPLITS}, got {split!r}")

    root = find_vimeo_root(root)
    manifest_path = Path(manifest_path)

    official_ids = load_official_split_ids(root)[split]
    selected_ids = select_deterministic_subset(official_ids, max_sequences, seed)
    should_validate = validate if validate is not None else (max_sequences is not None)

    items: list[dict] = []
    errors: list[dict] = []
    for sequence_id in selected_ids:
        if should_validate:
            try:
                info = validate_sequence(root, sequence_id, probe_resolution=probe_resolution)
            except VimeoDatasetError as exc:
                errors.append({"sequence_id": sequence_id, "message": str(exc)})
                continue
            resolution = list(info.resolution) if info.resolution else None
        else:
            resolution = None

        items.append({
            "dataset": DATASET_NAME,
            "source_type": SOURCE_TYPE,
            "sequence_id": sequence_id,
            "split": split,
            "frame_filenames": list(SEPTUPLET_FILENAMES),
            "frame_count": SEPTUPLET_FRAME_COUNT,
            "original_resolution": resolution,
        })

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_NAME,
        "source_type": SOURCE_TYPE,
        "dataset_root": relpath_posix(root, manifest_path.parent),
        "sequences_dirname": SEQUENCES_DIRNAME,
        "settings": {
            "split": split,
            "max_sequences": max_sequences,
            "seed": seed,
            "validated": should_validate,
            "probe_resolution": probe_resolution,
            "official_ids_available": len(official_ids),
        },
        "items": items,
        "errors": errors,
        "summary": {
            "sequences_selected": len(selected_ids),
            "sequences_written": len(items),
            "sequences_skipped": len(errors),
            "total_frames": len(items) * SEPTUPLET_FRAME_COUNT,
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
