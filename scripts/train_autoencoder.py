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

    # Milestone 8A: quantization-aware training (distortion-only; see
    # src/nvc/training/quantization_noise.py). --qat-calibration must be a
    # calibration file computed from the TRAIN split (scripts/calibrate_
    # quantizer.py's normal output) at the SAME bit depth as --qat-bits.
    # Use a checkpoint-dir distinct from the baseline run so latest.pt/
    # best.pt never collide with the non-QAT checkpoints:
    python scripts\\train_autoencoder.py --epochs 20 `
        --resume outputs\\checkpoints\\vimeo_epoch17_best.pt `
        --qat-enabled --qat-bits 4 `
        --qat-calibration outputs\\calibration\\vimeo_epoch17_4bit_qat_train.json `
        --checkpoint-dir outputs\\checkpoints\\vimeo_qat_noise

Run `python scripts\\train_autoencoder.py --help` for the full option list.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

from nvc.data.loaders import create_train_loader, create_val_loader
from nvc.data.validation import DatasetValidationError
from nvc.models import BaselineAutoencoder
from nvc.training import (
    QuantizationNoise,
    RateEstimator,
    resume_training_state,
    save_checkpoint,
    train_one_epoch,
    train_one_epoch_with_rate,
    validate_one_epoch,
    validate_one_epoch_with_rate,
)
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
    parser.add_argument(
        "--qat-enabled", action="store_true", default=defaults.qat_enabled,
        help=(
            "Milestone 8A: inject differentiable Uniform(-scale/2, scale/2) "
            "quantization noise into the latent during training (distortion-only; "
            "no rate/entropy loss). Requires --qat-calibration. Off by default - "
            "existing training behavior is unchanged unless this is passed."
        ),
    )
    parser.add_argument(
        "--qat-bits", type=int, default=defaults.qat_bits,
        help="Bit depth the quantization-noise relaxation targets. Must match "
             "--qat-calibration's own recorded bit depth.",
    )
    parser.add_argument(
        "--qat-mode", choices=["global", "per_channel"], default=defaults.qat_mode,
        help="Must match --qat-calibration's own recorded mode.",
    )
    parser.add_argument(
        "--qat-calibration", type=Path, default=defaults.qat_calibration_path,
        help="Calibration file (scripts/calibrate_quantizer.py output) computed "
             "from the TRAIN split, supplying the frozen training-time noise scale. "
             "Required when --qat-enabled is passed.",
    )
    parser.add_argument(
        "--rate-enabled", action="store_true", default=defaults.rate_enabled,
        help=(
            "Milestone 9A: add a differentiable rate proxy (nvc.training."
            "RateEstimator) and train on distortion + lambda * rate instead of "
            "distortion alone. Off by default - existing training behavior is "
            "unchanged unless this is passed."
        ),
    )
    parser.add_argument(
        "--rate-lambda", type=float, default=defaults.rate_lambda,
        help="Rate weight in D + lambda*R. Must be finite and >= 0; 0.0 (the "
             "default) reproduces the distortion-only objective exactly.",
    )
    parser.add_argument(
        "--rate-calibration", type=Path, default=defaults.rate_calibration_path,
        help="Calibration file supplying the rate estimator's bin width. Required "
             "when --rate-enabled is passed WITHOUT --qat-enabled. Must NOT be "
             "passed together with --qat-enabled - in that case the rate "
             "estimator reuses --qat-calibration's scale automatically, so there "
             "is only ever one quantization scale in play for a given run.",
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
    if args.qat_enabled and args.qat_calibration is None:
        parser.error("--qat-calibration is required when --qat-enabled is passed")
    if args.rate_enabled:
        if not math.isfinite(args.rate_lambda) or args.rate_lambda < 0:
            parser.error("--rate-lambda must be finite and >= 0")
        if args.qat_enabled and args.rate_calibration is not None:
            parser.error(
                "--rate-calibration must not be passed together with --qat-enabled - "
                "the rate estimator reuses --qat-calibration's scale automatically, "
                "so there is only ever one quantization scale for a given run."
            )
        if not args.qat_enabled and args.rate_calibration is None:
            parser.error("--rate-calibration is required when --rate-enabled is passed without --qat-enabled")

    seed_everything(args.seed)
    device = _resolve_device(args.device)

    quantization_noise = None
    if args.qat_enabled:
        if not args.qat_calibration.is_file():
            print(f"[ERROR] --qat-calibration not found: {args.qat_calibration}", file=sys.stderr)
            return 1
        try:
            quantization_noise = QuantizationNoise.from_calibration(
                args.qat_calibration, bits=args.qat_bits, mode=args.qat_mode,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

    rate_estimator = None
    if args.rate_enabled:
        if args.qat_enabled:
            # Reuse the SAME scale QuantizationNoise just loaded - never a
            # second, independently loaded copy (see --rate-calibration's
            # help text and rate_estimator.py's module docstring).
            rate_estimator = RateEstimator(
                quantization_noise.scale, bits=quantization_noise.bits, mode=quantization_noise.mode,
            )
        else:
            if not args.rate_calibration.is_file():
                print(f"[ERROR] --rate-calibration not found: {args.rate_calibration}", file=sys.stderr)
                return 1
            try:
                rate_estimator = RateEstimator.from_calibration(args.rate_calibration)
            except (ValueError, FileNotFoundError) as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                return 1
        rate_estimator = rate_estimator.to(device)

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

    model = BaselineAutoencoder(
        latent_channels=args.latent_channels, quantization_noise=quantization_noise,
    ).to(device)
    # The rate estimator's own loc/log_scale must be in the optimizer too -
    # it is a real nn.Module with learnable parameters (see rate_estimator.py),
    # unlike QuantizationNoise which has none.
    optimizer_params = list(model.parameters())
    if rate_estimator is not None:
        optimizer_params += list(rate_estimator.parameters())
    optimizer = torch.optim.Adam(optimizer_params, lr=args.learning_rate)

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

    if quantization_noise is not None:
        print(
            f"[QAT] Quantization-noise relaxation ENABLED - "
            f"{quantization_noise.bits}-bit / {quantization_noise.mode}, "
            f"scale from {args.qat_calibration}."
        )
    else:
        print("[QAT] Quantization-noise relaxation disabled (baseline training).")

    if rate_estimator is not None:
        scale_source = "--qat-calibration (shared with QAT)" if args.qat_enabled else str(args.rate_calibration)
        print(
            f"[RATE] Milestone 9A rate loss ENABLED - lambda={args.rate_lambda}, "
            f"{rate_estimator.bits}-bit / {rate_estimator.mode} bin width from {scale_source}. "
            f"Loss is distortion + {args.rate_lambda} * rate."
        )
    else:
        print("[RATE] Rate loss disabled (loss is plain MSE, as before Milestone 9).")

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_dir / "history.json"
    model_config = model.config_dict()
    best_val_loss = min((record["val_loss"] for record in history), default=float("inf"))

    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        t0 = time.time()
        if rate_estimator is not None:
            train_metrics = train_one_epoch_with_rate(
                model, train_loader, optimizer, device,
                rate_estimator=rate_estimator, lambda_rate=args.rate_lambda, max_batches=args.max_batches,
            )
            val_metrics = validate_one_epoch_with_rate(
                model, val_loader, device,
                rate_estimator=rate_estimator, lambda_rate=args.rate_lambda, max_batches=args.max_batches,
            )
        else:
            train_metrics = train_one_epoch(model, train_loader, optimizer, device, max_batches=args.max_batches)
            val_metrics = validate_one_epoch(model, val_loader, device, max_batches=args.max_batches)
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_psnr": val_metrics["psnr"],
            "elapsed_seconds": elapsed,
            # Milestone 8A: recorded every epoch (not just once in metadata)
            # so a history.json file can never be mistaken for the wrong
            # experiment type if baseline and QAT runs are ever compared
            # side by side or a history file gets copied out of context.
            "qat_enabled": quantization_noise is not None,
            "qat_bits": quantization_noise.bits if quantization_noise is not None else None,
            "qat_mode": quantization_noise.mode if quantization_noise is not None else None,
            # Milestone 9A: same reasoning - always recorded, None/False when
            # rate training wasn't used, so history.json is self-describing.
            "rate_enabled": rate_estimator is not None,
            "rate_lambda": args.rate_lambda if rate_estimator is not None else None,
            "train_distortion": train_metrics.get("distortion"),
            "train_rate_bpp": train_metrics.get("rate"),
            "val_distortion": val_metrics.get("distortion"),
            "val_rate_bpp": val_metrics.get("rate"),
        }
        history.append(record)

        rate_suffix = (
            f" train_rate={train_metrics['rate']:.4f}bpp val_rate={val_metrics['rate']:.4f}bpp"
            if rate_estimator is not None else ""
        )
        print(
            f"[EPOCH {epoch}] train_mse={train_metrics['loss']:.6f} "
            f"val_mse={val_metrics['loss']:.6f} val_psnr={val_metrics['psnr']:.2f} dB "
            f"elapsed={elapsed:.1f}s{rate_suffix}"
        )

        # Re-read fresh every epoch, not hoisted above the loop - the rate
        # estimator's own parameters are updated by optimizer.step() each
        # epoch just like the model's, so a stale state_dict captured once
        # before the loop would silently checkpoint epoch-1's rate params
        # forever after.
        checkpoint_extra = (
            {"rate_estimator_state_dict": rate_estimator.state_dict(), "rate_lambda": args.rate_lambda}
            if rate_estimator is not None else None
        )

        save_checkpoint(
            checkpoint_dir / "latest.pt", model=model, optimizer=optimizer,
            epoch=epoch, history=history, model_config=model_config, extra=checkpoint_extra,
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                checkpoint_dir / "best.pt", model=model, optimizer=optimizer,
                epoch=epoch, history=history, model_config=model_config, extra=checkpoint_extra,
            )
            print(f"  [BEST] New best validation MSE: {best_val_loss:.6f} -> {checkpoint_dir / 'best.pt'}")

        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print()
    print(f"Checkpoints written to: {checkpoint_dir}")
    print(f"History written to:     {history_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
