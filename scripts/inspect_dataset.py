"""Inspect the prepared frame dataset: sizes, a sample batch's tensor
shape/dtype/pixel range, and the selected compute device. Optionally saves
a visual grid of sample training frames.

This does not train anything - it is a sanity-check tool for the PyTorch
data pipeline built in Milestone 3.

Usage (PowerShell, from the project root, with .venv activated):

    python scripts\\inspect_dataset.py
    python scripts\\inspect_dataset.py --visualize
    python scripts\\inspect_dataset.py --manifest data\\processed\\manifest.json --batch-size 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from nvc.data.loaders import create_test_loader, create_train_loader, create_val_loader
from nvc.data.validation import DatasetValidationError
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report train/val/test dataset sizes and a sample batch's tensor "
            "properties; optionally save a visual grid of sample frames."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json",
        help="Path to the dataset manifest produced by scripts/prepare_dataset.py.",
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--crop-size", type=int, default=defaults.random_crop_size)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save a grid of sample training frames to outputs/visualizations/dataset_grid.png",
    )
    parser.add_argument(
        "--grid-size", type=int, default=16,
        help="Number of images to include in the visualization grid.",
    )
    return parser


def _save_grid(batch: torch.Tensor, grid_size: int, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    images = batch[: min(grid_size, batch.shape[0])]
    nrow = max(1, int(images.shape[0] ** 0.5))
    grid = make_grid(images, nrow=nrow)
    # [C, H, W] -> [H, W, C] for imshow. Already RGB (FrameDataset converts
    # OpenCV's BGR on load), so no color conversion needed here.
    grid_np = grid.permute(1, 2, 0).clamp(0, 1).numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.title("Sample training frames (RGB)")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved visualization grid: {output_path}")


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        print(
            f"[ERROR] Manifest not found: {args.manifest}. Run scripts\\prepare_dataset.py first.",
            file=sys.stderr,
        )
        return 1

    try:
        train_loader = create_train_loader(
            args.manifest, batch_size=args.batch_size, num_workers=args.num_workers,
            seed=args.seed, crop_size=args.crop_size,
        )
        val_loader = create_val_loader(
            args.manifest, batch_size=args.batch_size, num_workers=args.num_workers,
            crop_size=args.crop_size,
        )
        test_loader = create_test_loader(
            args.manifest, batch_size=args.batch_size, num_workers=args.num_workers,
            crop_size=args.crop_size,
        )
    except DatasetValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)

    print("=" * 50)
    print("Dataset Inspection")
    print("=" * 50)
    print(f"Total frames:        {train_size + val_size + test_size}")
    print(f"Train:               {train_size} frames, {len(train_loader)} batches")
    print(f"Validation:          {val_size} frames, {len(val_loader)} batches")
    print(f"Test:                {test_size} frames, {len(test_loader)} batches")

    batch = next(iter(train_loader))
    print()
    print(f"Sample batch shape:  {tuple(batch.shape)}")
    print(f"Sample batch dtype:  {batch.dtype}")
    print(f"Sample pixel range:  [{batch.min().item():.4f}, {batch.max().item():.4f}]")

    print()
    device = get_device()
    print(f"Selected device:     {device}")
    print("=" * 50)

    if args.visualize:
        _save_grid(batch, args.grid_size, defaults.visualizations_dir / "dataset_grid.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
