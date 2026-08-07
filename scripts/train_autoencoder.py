"""Baseline convolutional autoencoder training CLI.

Trains BaselineAutoencoder (src/nvc/models/autoencoder.py) to reconstruct
RGB video frames using plain MSE reconstruction loss - no quantization,
entropy coding, or stochastic latent variables yet (see README.md).

Example usage (PowerShell, from the project root, with .venv activated):

    # Quick CPU smoke test: exercises the full training path with real
    # gradient updates on a handful of batches. Not full training.
    python scripts\\train_autoencoder.py --epochs 1 --max-batches 5

    # A real training run (defaults come from configs/default.json)
    python scripts\\train_autoencoder.py --epochs 50

    # Resume from the latest checkpoint, continuing epoch numbering
    python scripts\\train_autoencoder.py --epochs 20 --resume outputs\\checkpoints\\latest.pt

Run `python scripts\\train_autoencoder.py --help` for the full option list.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from nvc.data.loaders import create_train_loader, create_val_loader
from nvc.data.validation import DatasetValidationError
from nvc.models import BaselineAutoencoder
from nvc.training import resume_training_state, save_checkpoint, train_one_epoch, validate_one_epoch
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train BaselineAutoencoder on the prepared frame dataset with "
            "plain MSE reconstruction loss."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json",
        help="Path to the dataset manifest produced by scripts/prepare_dataset.py.",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of epochs to run in this invocation (added on top of --resume's epoch, if given).",
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument(
        "--latent-channels", type=int, default=defaults.latent_channels,
        help="Number of channels in the spatial latent tensor.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, default=defaults.checkpoint_dir)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help=(
            "Limit each epoch to this many batches (train and val). For "
            "quick CPU smoke tests only - this is not full training."
        ),
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Checkpoint path to restore model/optimizer/epoch/history from before continuing.",
    )
    return parser


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return get_device()
    return torch.device(name)


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if not args.manifest.is_file():
        print(
            f"[ERROR] Manifest not found: {args.manifest}. Run scripts\\prepare_dataset.py first.",
            file=sys.stderr,
        )
        return 1

    seed_everything(args.seed)
    device = _resolve_device(args.device)

    try:
        train_loader = create_train_loader(
            args.manifest, batch_size=args.batch_size, num_workers=defaults.num_workers,
            seed=args.seed, crop_size=defaults.random_crop_size,
        )
        val_loader = create_val_loader(
            args.manifest, batch_size=args.batch_size, num_workers=defaults.num_workers,
            crop_size=defaults.random_crop_size,
        )
    except DatasetValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    model = BaselineAutoencoder(latent_channels=args.latent_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: list[dict] = []
    start_epoch = 1
    if args.resume is not None:
        if not args.resume.is_file():
            print(f"[ERROR] Checkpoint not found: {args.resume}", file=sys.stderr)
            return 1
        try:
            start_epoch, history = resume_training_state(
                args.resume, model=model, optimizer=optimizer, map_location=device,
            )
        except RuntimeError as exc:
            print(
                f"[ERROR] Could not resume from {args.resume} "
                f"(likely a --latent-channels mismatch with the checkpoint): {exc}",
                file=sys.stderr,
            )
            return 1
        print(
            f"[RESUME] Resumed from {args.resume}: next epoch is {start_epoch}, "
            f"{len(history)} epoch(s) of prior history loaded."
        )

    # Latent-space sanity report. This is a RAW dimensionality ratio, not a
    # compression ratio: the latent tensor is still float32 and has not
    # been quantized or entropy-coded.
    sample_batch = next(iter(train_loader)).to(device)
    with torch.no_grad():
        sample_latent = model.encode(sample_batch)
        sample_reconstruction = model.decode(sample_latent)
    input_elements = sample_batch[0].numel()
    latent_elements = sample_latent[0].numel()

    print("=" * 60)
    print("BaselineAutoencoder")
    print("=" * 60)
    print(f"Input shape:            {tuple(sample_batch.shape[1:])}")
    print(f"Latent shape:           {tuple(sample_latent.shape[1:])}")
    print(f"Reconstruction shape:   {tuple(sample_reconstruction.shape[1:])}")
    print(f"Trainable parameters:   {model.num_parameters():,}")
    print(f"Raw latent dimensionality ratio (input elements / latent elements): {input_elements / latent_elements:.2f}")
    print("  NOT a compression ratio - the latent is float32 and unquantized/uncoded.")
    print("=" * 60)

    if args.max_batches is not None:
        print(f"[SMOKE TEST] Limited to {args.max_batches} batch(es) per epoch - this is not full training.")

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_dir / "history.json"
    model_config = model.config_dict()
    best_val_loss = min((record["val_loss"] for record in history), default=float("inf"))

    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, max_batches=args.max_batches)
        val_metrics = validate_one_epoch(model, val_loader, device, max_batches=args.max_batches)
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_psnr": val_metrics["psnr"],
            "elapsed_seconds": elapsed,
        }
        history.append(record)

        print(
            f"[EPOCH {epoch}] train_mse={train_metrics['loss']:.6f} "
            f"val_mse={val_metrics['loss']:.6f} val_psnr={val_metrics['psnr']:.2f} dB "
            f"elapsed={elapsed:.1f}s"
        )

        save_checkpoint(
            checkpoint_dir / "latest.pt", model=model, optimizer=optimizer,
            epoch=epoch, history=history, model_config=model_config,
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                checkpoint_dir / "best.pt", model=model, optimizer=optimizer,
                epoch=epoch, history=history, model_config=model_config,
            )
            print(f"  [BEST] New best validation MSE: {best_val_loss:.6f} -> {checkpoint_dir / 'best.pt'}")

        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print()
    print(f"Checkpoints written to: {checkpoint_dir}")
    print(f"History written to:     {history_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
