"""Validate and/or select a reproducible subset of a large-scale training
dataset (currently Vimeo-90K Septuplet) without duplicating source frames.

This does NOT train anything, and does NOT resize/re-save frames the way
scripts/prepare_dataset.py does for DAVIS - it only writes a small manifest
that references the original Vimeo files. See README.md, "Large-Scale
Training Dataset", for why: running the PNG-extraction pipeline over
Vimeo-90K would silently create a second ~82 GB copy on disk.

Two modes:

    --validate-only    Report dataset statistics and spot-check a sample of
                        sequences for structural correctness. Writes nothing.

    (default)           Select a deterministic subset of one official split
                        (--split train|test) and write a sequence manifest
                        for it.

Example usage (PowerShell, from the project root, with .venv activated):

    # Validate a real download before using it
    python scripts\\prepare_training_dataset.py --dataset vimeo90k --validate-only

    # Build a reproducible 10,000-sequence training subset
    python scripts\\prepare_training_dataset.py `
        --dataset vimeo90k --split train --max-sequences 10000 --seed 42

    # The full official test split, fully validated
    python scripts\\prepare_training_dataset.py `
        --dataset vimeo90k --split test --full-scan

Run `python scripts\\prepare_training_dataset.py --help` for the full option list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nvc.data.vimeo import (
    OFFICIAL_SPLITS,
    VimeoDatasetError,
    build_sequence_manifest,
    load_official_split_ids,
    validate_sequence,
    vimeo_dataset_statistics,
)
from nvc.utils.config import load_default_config

_SUPPORTED_DATASETS = ("vimeo90k",)
_SPOT_CHECK_SAMPLE_SIZE = 20


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a large-scale training dataset, or select a reproducible "
            "subset of one official split into a sequence manifest."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=_SUPPORTED_DATASETS, default="vimeo90k",
        help="Which large-scale dataset to operate on.",
    )
    parser.add_argument(
        "--vimeo-root", type=Path, default=defaults.vimeo_root,
        help="Path to the extracted Vimeo-90K Septuplet folder (contains sequences/, "
             "sep_trainlist.txt, sep_testlist.txt). Not auto-downloaded - see README.md.",
    )
    parser.add_argument("--split", choices=OFFICIAL_SPLITS, default="train")
    parser.add_argument(
        "--max-sequences", type=int, default=None,
        help="Deterministically select at most this many sequences from --split "
             "(default: use every sequence in the official list).",
    )
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument(
        "--crop-size", type=int, default=defaults.frame_width,
        help="Training crop size, recorded for reference (SequenceFrameDataset applies "
             "the actual crop at load time; Vimeo frames are never resized on disk).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Manifest output path (default: configs/default.json's "
             "vimeo_manifest_path, suffixed with the split name).",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Report dataset statistics and spot-check sequences; do not write a manifest.",
    )
    parser.add_argument(
        "--full-scan", action="store_true",
        help="Validate every selected sequence (all 7 frame files present), not just a "
             "bounded sample. Cheap when combined with --max-sequences; can be slow over "
             "the full dataset (tens of thousands of directory checks).",
    )
    return parser


def _spot_check(root: Path, ids: list[str], seed: int, sample_size: int) -> dict:
    """Validate a small deterministic sample rather than the whole split."""
    from nvc.data.vimeo import select_deterministic_subset

    sample_ids = select_deterministic_subset(ids, min(sample_size, len(ids)), seed)
    ok, failed = [], []
    for sequence_id in sample_ids:
        try:
            validate_sequence(root, sequence_id)
            ok.append(sequence_id)
        except VimeoDatasetError as exc:
            failed.append({"sequence_id": sequence_id, "message": str(exc)})
    return {"sample_size": len(sample_ids), "passed": len(ok), "failed": failed}


def _run_validate_only(args) -> int:
    try:
        stats = vimeo_dataset_statistics(args.vimeo_root)
    except VimeoDatasetError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    try:
        official_ids = load_official_split_ids(args.vimeo_root)
    except VimeoDatasetError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("=" * 64)
    print("Vimeo-90K Septuplet - Dataset Validation")
    print("=" * 64)
    print(f"Root:                  {stats['vimeo_root']}")
    print(f"Total sequences:       {stats['num_sequences_total']:,}")
    print(f"Train sequences:       {stats['num_train_sequences']:,}  (sep_trainlist.txt)")
    print(f"Test sequences:        {stats['num_test_sequences']:,}  (sep_testlist.txt)")
    print(f"Frames per sequence:   {stats['frames_per_sequence']}")
    print(f"Total source frames:   {stats['total_source_frames']:,}  (assumes complete sequences)")
    if stats["source_resolution_samples"]:
        print(f"Sampled resolution:    {stats['source_resolution_samples']}")
    if stats["source_resolution_probe_unreadable"]:
        print(f"[WARN] {len(stats['source_resolution_probe_unreadable'])} sampled sequence(s) "
              "could not be probed - see JSON output for details.")

    overall_ok = True
    for split in OFFICIAL_SPLITS:
        ids = official_ids[split]
        if args.full_scan:
            print(f"\nValidating ALL {len(ids):,} '{split}' sequences (--full-scan)...")
            report = {"sample_size": len(ids), "passed": 0, "failed": []}
            for sequence_id in ids:
                try:
                    validate_sequence(args.vimeo_root, sequence_id)
                    report["passed"] += 1
                except VimeoDatasetError as exc:
                    report["failed"].append({"sequence_id": sequence_id, "message": str(exc)})
        else:
            report = _spot_check(args.vimeo_root, ids, args.seed, _SPOT_CHECK_SAMPLE_SIZE)
            print(f"\nSpot-checking {report['sample_size']} of {len(ids):,} "
                  f"'{split}' sequences (seed {args.seed})...")
        print(f"  passed: {report['passed']} / {report['sample_size']}")
        if report["failed"]:
            overall_ok = False
            for failure in report["failed"][:5]:
                print(f"  [FAIL] {failure['sequence_id']}: {failure['message']}")
            if len(report["failed"]) > 5:
                print(f"  ... and {len(report['failed']) - 5} more")

    print("=" * 64)
    if overall_ok:
        print("Validation passed. Dataset structure looks correct.")
    else:
        print("[WARN] Some sequences failed validation - see above.")
    print(json.dumps(stats, indent=2))
    return 0 if overall_ok else 1


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    if args.dataset != "vimeo90k":
        parser.error(f"Unsupported --dataset {args.dataset!r}")

    if args.validate_only:
        return _run_validate_only(args)

    output = args.output or defaults.vimeo_manifest_path.with_stem(
        f"{defaults.vimeo_manifest_path.stem}_{args.split}"
    )

    try:
        manifest = build_sequence_manifest(
            args.vimeo_root, output,
            split=args.split, max_sequences=args.max_sequences, seed=args.seed,
            validate=True if args.full_scan else None,
        )
    except VimeoDatasetError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    summary = manifest["summary"]
    settings = manifest["settings"]

    print("=" * 64)
    print("Training Dataset Subset Selection")
    print("=" * 64)
    print(f"Dataset:               {manifest['dataset']}")
    print(f"Split:                 {args.split}  ({settings['official_ids_available']:,} official sequences)")
    print(f"Requested max:         {args.max_sequences if args.max_sequences is not None else '(all)'}")
    print(f"Seed:                  {args.seed}")
    print(f"Crop size (recorded):  {args.crop_size}")
    print(f"Validated on selection: {settings['validated']}")
    print()
    print(f"Sequences selected:    {summary['sequences_selected']:,}")
    print(f"Sequences written:     {summary['sequences_written']:,}")
    print(f"Sequences skipped:     {summary['sequences_skipped']:,}")
    print(f"Total frames:          {summary['total_frames']:,}")
    if manifest["items"]:
        sample = [item["sequence_id"] for item in manifest["items"][:5]]
        print(f"First selected ids:    {sample}")
    if manifest["errors"]:
        print(f"\n[WARN] {len(manifest['errors'])} sequence(s) skipped:")
        for error in manifest["errors"][:5]:
            print(f"  [SKIPPED] {error['sequence_id']}: {error['message']}")
    print("=" * 64)
    print(f"Manifest written to: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
