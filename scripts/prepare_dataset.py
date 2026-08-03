"""Milestone 2 CLI: raw videos -> resized train/val/test PNG frame datasets + manifest.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\prepare_dataset.py --input data\\raw

    python scripts\\prepare_dataset.py `
        --input data\\raw `
        --width 256 `
        --height 256 `
        --every-n-frames 2 `
        --seed 42

Run `python scripts\\prepare_dataset.py --help` for the full option list.
This script only extracts and organizes frames - it does not train, encode,
or compress anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nvc.data.dataset_prep import prepare_dataset
from nvc.utils.config import load_default_config


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate, extract, resize, and split video frames from data/raw "
            "into train/val/test PNG datasets, and write a manifest.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=defaults.raw_data_dir,
        help="Directory containing raw source videos.",
    )
    parser.add_argument(
        "--output", type=Path, default=defaults.frames_dir,
        help="Directory where train/val/test frame folders are written.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json",
        help="Path to write the dataset manifest JSON.",
    )
    parser.add_argument(
        "--width", type=int, default=defaults.frame_width,
        help="Target frame width in pixels.",
    )
    parser.add_argument(
        "--height", type=int, default=defaults.frame_height,
        help="Target frame height in pixels.",
    )
    parser.add_argument(
        "--every-n-frames", type=int, default=defaults.every_n_frames,
        help="Keep 1 out of every N frames from each video (1 = every frame).",
    )
    parser.add_argument(
        "--no-preserve-aspect-ratio", dest="preserve_aspect_ratio",
        action="store_false", default=True,
        help=(
            "Resize directly to --width/--height, allowing distortion, instead "
            "of the default resize-then-center-crop behavior."
        ),
    )
    parser.add_argument(
        "--train-ratio", type=float, default=defaults.train_ratio,
        help="Fraction of videos (not frames) assigned to the training split.",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=defaults.val_ratio,
        help="Fraction of videos assigned to the validation split.",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=defaults.test_ratio,
        help="Fraction of videos assigned to the test split.",
    )
    parser.add_argument(
        "--seed", type=int, default=defaults.random_seed,
        help="Random seed controlling the video-level train/val/test split.",
    )
    return parser


def _print_report(manifest: dict) -> None:
    summary = manifest["summary"]
    print("=" * 50)
    print("Dataset Preparation Summary")
    print("=" * 50)
    print(f"Videos found:      {summary['total_videos_found']}")
    print(f"Videos processed:  {summary['total_videos_processed']}")
    print(f"Videos skipped:    {summary['total_videos_skipped']}")
    print()
    for split in ("train", "val", "test"):
        print(
            f"  {split:<5}: {summary['videos_per_split'][split]:>3} videos, "
            f"{summary['frames_per_split'][split]:>6} frames"
        )
    print()
    print(f"Total frames extracted: {summary['total_frames_extracted']}")
    print(f"Approx. output size:    {summary['total_output_bytes'] / (1024 * 1024):.2f} MB")

    if manifest["errors"]:
        print()
        print(f"Skipped videos ({len(manifest['errors'])}):")
        for err in manifest["errors"]:
            print(f"  [SKIPPED] {err['source_video']}: {err['message']}")

    print("=" * 50)


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    if args.every_n_frames < 1:
        parser.error("--every-n-frames must be >= 1")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive integers")

    if not args.input.is_dir():
        print(f"[ERROR] Input directory does not exist: {args.input}", file=sys.stderr)
        return 1

    try:
        manifest = prepare_dataset(
            raw_dir=args.input,
            frames_root=args.output,
            manifest_path=args.manifest,
            target_width=args.width,
            target_height=args.height,
            every_n_frames=args.every_n_frames,
            preserve_aspect_ratio=args.preserve_aspect_ratio,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            supported_extensions=defaults.supported_video_extensions,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    _print_report(manifest)
    print(f"\nManifest written to: {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
