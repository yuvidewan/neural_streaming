"""Reconstruction CLI: load a trained BaselineAutoencoder checkpoint,
encode and decode real test-split frames, report MSE/PSNR, and save an
Original | Reconstruction comparison visualization.

This does not train anything - it is inference-only, for sanity-checking a
checkpoint produced by scripts/train_autoencoder.py.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\reconstruct.py --checkpoint outputs\\checkpoints\\best.pt
    python scripts\\reconstruct.py --checkpoint outputs\\checkpoints\\latest.pt --num-samples 4

Run `python scripts\\reconstruct.py --help` for the full option list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from nvc.data.loaders import create_test_loader
from nvc.data.validation import DatasetValidationError
from nvc.evaluation.basic_metrics import mse, psnr
from nvc.models import BaselineAutoencoder
from nvc.training import load_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct test-split frames with a trained BaselineAutoencoder "
            "checkpoint, report MSE/PSNR, and save a comparison visualization."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to a checkpoint saved by scripts/train_autoencoder.py.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json",
        help="Path to the dataset manifest produced by scripts/prepare_dataset.py.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=8,
        help="Number of test-split frames to encode/decode.",
    )
    parser.add_argument(
        "--output", type=Path, default=defaults.visualizations_dir / "reconstructions.png",
        help="Path to save the Original | Reconstruction comparison image.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def _save_comparison_grid(originals: torch.Tensor, reconstructions: torch.Tensor, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = originals.shape[0]
    fig, axes = plt.subplots(n, 2, figsize=(4, 2 * n), squeeze=False)
    for i in range(n):
        axes[i][0].imshow(originals[i].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[i][0].axis("off")
        axes[i][1].imshow(reconstructions[i].permute(1, 2, 0).clamp(0, 1).numpy())
        axes[i][1].axis("off")
    axes[0][0].set_title("Original")
    axes[0][1].set_title("Reconstruction")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.manifest.is_file():
        print(
            f"[ERROR] Manifest not found: {args.manifest}. Run scripts\\prepare_dataset.py first.",
            file=sys.stderr,
        )
        return 1

    device = get_device() if args.device == "auto" else torch.device(args.device)

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model = BaselineAutoencoder(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    try:
        test_loader = create_test_loader(
            args.manifest, batch_size=args.num_samples, crop_size=defaults.random_crop_size,
        )
    except DatasetValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    batch = next(iter(test_loader)).to(device)
    with torch.no_grad():
        latent = model.encode(batch)
        reconstruction = model.decode(latent)

    batch_mse = mse(reconstruction, batch).item()
    batch_psnr = psnr(reconstruction, batch).item()

    _save_comparison_grid(batch.cpu(), reconstruction.cpu(), args.output)

    print("=" * 50)
    print("Reconstruction")
    print("=" * 50)
    print(f"Checkpoint:       {args.checkpoint} (epoch {checkpoint['epoch']})")
    print(f"Samples:          {batch.shape[0]}")
    print(f"MSE:              {batch_mse:.6f}")
    print(f"PSNR:             {batch_psnr:.2f} dB")
    print(f"Saved comparison: {args.output}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
