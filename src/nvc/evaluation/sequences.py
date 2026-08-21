"""Benchmark sequence discovery and validation.

The rate-distortion benchmark is sequence-oriented (H.264/H.265 encode a
*sequence*, not a bag of frames), so this module groups the flat frame
dataset back into ordered sequences.

Sequence membership comes from `data/processed/manifest.json`, not from
parsing frame filenames. That matters: DAVIS sequence names contain
hyphens and underscores ('bmx-bumps', 'drift-chicane'), so splitting
'drift-chicane_000007.png' on a separator is a heuristic that silently
mis-groups. The manifest already records each item's directory, filename
pattern and exact frame count - it is authoritative, and reusing it keeps
this consistent with FrameDataset, which reads the same file.

Nothing here copies the dataset. A sequence is a list of paths into the
existing frames directory; only the temporary FFmpeg working files (see
`materialize_sequence_for_ffmpeg`) are ever written, and they live under a
caller-supplied scratch directory that is safe to delete.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from nvc.data.image_io import read_image_as_tensor
from nvc.data.validation import DatasetValidationError

# FFmpeg reads image sequences via a printf-style pattern; frames are
# rewritten (hard-linked where possible) to this contiguous naming so the
# original dataset's numbering gaps or prefixes cannot confuse it.
FFMPEG_FRAME_PATTERN = "frame_%06d.png"
_FFMPEG_FRAME_TEMPLATE = "frame_{index:06d}.png"


@dataclass(frozen=True)
class BenchmarkSequence:
    """One ordered, validated image sequence to benchmark."""

    dataset: str
    sequence_id: str
    split: str
    frame_paths: tuple[Path, ...]
    width: int
    height: int

    @property
    def frame_count(self) -> int:
        return len(self.frame_paths)

    @property
    def total_pixels(self) -> int:
        """Pixels across every frame - the denominator for sequence BPP."""
        return self.frame_count * self.width * self.height

    def raw_rgb_bytes(self, channels: int = 3) -> int:
        """Uncompressed uint8 RGB size, the compression-ratio baseline.

        Deliberately RGB, not YUV: this project's frames are RGB and the
        ratio is documented as being against raw uint8 RGB storage, which
        is NOT the same thing as uncompressed YUV video.
        """
        return self.total_pixels * channels

    def load_frames(self) -> torch.Tensor:
        """Load every frame as one [N, 3, H, W] float32 tensor in [0, 1]."""
        return torch.stack([read_image_as_tensor(p) for p in self.frame_paths])


def discover_sequences(
    manifest_path: str | Path,
    *,
    split: str = "test",
    dataset: str = "davis",
    max_sequences: int | None = None,
    max_frames_per_sequence: int | None = None,
) -> list[BenchmarkSequence]:
    """Build ordered BenchmarkSequences from a frame manifest.

    Sequences come back in the manifest's own (already sorted) order, never
    chosen by how well they compress. `max_sequences` /
    `max_frames_per_sequence` truncate deterministically from the front,
    which is what makes the smoke-test mode reproducible.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise DatasetValidationError(
            f"Manifest not found: {manifest_path}. Run scripts/prepare_dataset.py first."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    settings = manifest.get("settings", {})
    default_width = settings.get("target_width")
    default_height = settings.get("target_height")

    sequences: list[BenchmarkSequence] = []
    for item in manifest["items"]:
        if item["split"] != split:
            continue

        frame_directory = (manifest_path.parent / item["frame_directory"]).resolve()
        if not frame_directory.is_dir():
            raise DatasetValidationError(
                f"Frame directory for '{item['source_name']}' does not exist: "
                f"{frame_directory}. Re-run scripts/prepare_dataset.py."
            )

        pattern = item["frame_filename_pattern"]
        frame_count = item["frame_count"]
        if max_frames_per_sequence is not None:
            frame_count = min(frame_count, max_frames_per_sequence)

        # Frames are numbered from 1 by frame_extraction.save_frame_png.
        paths = [frame_directory / pattern.format(index=i) for i in range(1, frame_count + 1)]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise DatasetValidationError(
                f"Sequence '{item['source_name']}' is missing {len(missing)} frame file(s), "
                f"first: {missing[0]}. Re-run scripts/prepare_dataset.py."
            )
        if not paths:
            continue

        resolution = item.get("processed_resolution") or [default_width, default_height]
        sequences.append(BenchmarkSequence(
            dataset=dataset,
            sequence_id=item["source_name"],
            split=split,
            frame_paths=tuple(paths),
            width=int(resolution[0]),
            height=int(resolution[1]),
        ))

        if max_sequences is not None and len(sequences) >= max_sequences:
            break

    if not sequences:
        raise DatasetValidationError(
            f"No sequences found for split '{split}' in manifest {manifest_path}."
        )
    return sequences


def validate_sequence_frames(sequence: BenchmarkSequence) -> None:
    """Confirm every frame decodes and shares the sequence's resolution.

    Run before encoding rather than during: a mid-sequence resolution
    change would make FFmpeg silently rescale, quietly invalidating every
    per-frame comparison that follows.
    """
    for path in sequence.frame_paths:
        tensor = read_image_as_tensor(path)
        channels, height, width = tensor.shape
        if channels != 3:
            raise DatasetValidationError(
                f"{path}: expected 3 channels, got {channels}"
            )
        if (width, height) != (sequence.width, sequence.height):
            raise DatasetValidationError(
                f"{path} is {width}x{height}, but sequence '{sequence.sequence_id}' "
                f"is {sequence.width}x{sequence.height}. Every frame in a sequence "
                "must share one resolution."
            )


def materialize_sequence_for_ffmpeg(
    sequence: BenchmarkSequence, destination: str | Path
) -> tuple[Path, str]:
    """Stage a sequence as contiguously-numbered PNGs FFmpeg can glob.

    Returns (directory, pattern). Frames are hard-linked when the
    filesystem allows it and only copied as a fallback, so staging a
    sequence normally costs no additional disk space and no re-encoding -
    this never duplicates the dataset in any meaningful sense.

    Ordering is preserved exactly: output index N is the Nth frame of the
    sequence, regardless of the source dataset's own numbering.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(sequence.frame_paths, start=1):
        target = destination / _FFMPEG_FRAME_TEMPLATE.format(index=index)
        if target.exists():
            target.unlink()
        try:
            # Hard link: same bytes, one directory entry, no extra storage.
            target.hardlink_to(source)
        except (OSError, NotImplementedError):
            # Different volume, or a filesystem without hard links.
            shutil.copyfile(source, target)

    return destination, FFMPEG_FRAME_PATTERN
