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
    resume_model_only,
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
        "--resume-model-only", action="store_true",
        help=(
            "Milestone 9C: with --resume, restore model weights/epoch/history but "
            "NOT optimizer state. Required when starting rate training from a "
            "checkpoint written before --rate-enabled existed (M7/M8), whose saved "
            "optimizer state has no entries for the rate estimator's parameters. "
            "Also the right choice for a controlled lambda sweep, where every arm "
            "should start from an identical, empty optimizer state."
        ),
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
        "--rate-lr", type=float, default=defaults.rate_lr,
        help=(
            "Milestone 9C.1: learning rate for the rate estimator's own loc/log_scale, "
            "in a SEPARATE optimizer parameter group from the model's. The model keeps "
            "--learning-rate; only the estimator uses this. Must be finite and > 0. "
            "Defaults high relative to the model's LR on purpose - 128 scalars fitting a "
            "density need O(1) movement, which the model's 1e-4 cannot deliver in a short "
            "run (see MILESTONE_9_PLAN.md, M9C.1). Ignored unless --rate-enabled."
        ),
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


def _describe_optimizer_mismatch(
    checkpoint_path: Path, optimizer: torch.optim.Optimizer, rate_enabled: bool
) -> str:
    """Explain WHY optimizer state could not be restored, in this run's terms.

    Two distinct checkpoint vintages both fail against a rate-enabled run, and
    telling them apart matters because the remedy line is the same but the
    reason a reader should accept it is not:

      * pre-M9 (M7/M8): one parameter group holding only the model's
        parameters - the rate estimator's loc/log_scale did not exist yet.
      * M9A/M9C: one parameter group holding model + rate parameters together,
        before M9C.1 split them so the estimator could take its own learning
        rate (see --rate-lr).

    Falls back to a structural description when the checkpoint is neither.
    """
    try:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        groups = saved["optimizer_state_dict"]["param_groups"]
        saved_shape = [len(group["params"]) for group in groups]
    except Exception:  # noqa: BLE001 - diagnosis only, never worth masking the real error
        return (
            " Pass --resume-model-only to restore the weights and start the "
            "optimizer fresh."
        )

    live_shape = [len(group["params"]) for group in optimizer.state_dict()["param_groups"]]
    remedy = (
        " Pass --resume-model-only to restore the weights and start the optimizer fresh."
    )
    if not rate_enabled:
        return f" Checkpoint optimizer groups {saved_shape}, this run's {live_shape}.{remedy}"
    if len(saved_shape) == 1 and len(live_shape) == 2 and saved_shape[0] == live_shape[0]:
        return (
            " This checkpoint predates --rate-enabled, so its saved optimizer state has "
            f"no entries for the rate estimator's parameters (groups {saved_shape} vs "
            f"{live_shape}).{remedy}"
        )
    if len(saved_shape) == 1 and len(live_shape) == 2 and saved_shape[0] == sum(live_shape):
        return (
            " This checkpoint is from M9A/M9C, which kept the model and rate-estimator "
            "parameters in ONE optimizer group; M9C.1 splits them so the estimator can "
            f"take its own --rate-lr (groups {saved_shape} vs {live_shape}).{remedy}"
        )
    return f" Checkpoint optimizer groups {saved_shape}, this run's {live_shape}.{remedy}"


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
    if args.resume_model_only and args.resume is None:
        parser.error("--resume-model-only only makes sense together with --resume")
    if args.rate_enabled:
        if not math.isfinite(args.rate_lambda) or args.rate_lambda < 0:
            parser.error("--rate-lambda must be finite and >= 0")
        # Strictly positive, and never silently falling back to the model's LR:
        # a zero or negative rate LR would freeze the estimator at its
        # initialization, which is exactly the M9C failure this flag exists to fix.
        if not math.isfinite(args.rate_lr) or args.rate_lr <= 0:
            parser.error("--rate-lr must be finite and > 0")
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
    # unlike QuantizationNoise which has none. Milestone 9C.1: they go into
    # their OWN parameter group at --rate-lr, because the two need very
    # different step sizes (see that flag's help and the config field's
    # comment). The model's own group keeps --learning-rate untouched.
    #
    # When rate training is off, this is a single unnamed group exactly as
    # before, so non-rate runs and their checkpoints are bit-identical to
    # every run this script has ever produced.
    if rate_estimator is not None:
        optimizer = torch.optim.Adam(
            [
                {"params": list(model.parameters()), "lr": args.learning_rate},
                {"params": list(rate_estimator.parameters()), "lr": args.rate_lr},
            ]
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: list[dict] = []
    start_epoch = 1
    if args.resume is not None:
        if not args.resume.is_file():
            print(f"[ERROR] Checkpoint not found: {args.resume}", file=sys.stderr)
            return 1
        try:
            if args.resume_model_only:
                start_epoch, history = resume_model_only(
                    args.resume, model=model, map_location=device,
                )
            else:
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
        except ValueError as exc:
            # Raised by optimizer.load_state_dict when this run's optimizer
            # owns a different number of parameters than the checkpoint's did.
            # The overwhelmingly common cause is starting rate training from a
            # pre-M9 checkpoint: the rate estimator's loc/log_scale are in the
            # optimizer here but have no saved state there. Say so, rather than
            # surfacing PyTorch's bare "parameter group" message.
            hint = _describe_optimizer_mismatch(
                args.resume, optimizer, rate_enabled=rate_estimator is not None,
            )
            print(f"[ERROR] Could not resume from {args.resume}: {exc}.{hint}", file=sys.stderr)
            return 1
        restored = "model weights only (optimizer state started fresh)" if args.resume_model_only else "model + optimizer"
        print(
            f"[RESUME] Resumed from {args.resume} ({restored}): next epoch is {start_epoch}, "
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
        # Milestone 9C.1: state both learning rates explicitly, since the whole
        # point of the split is that they differ and a reader must be able to
        # confirm from the log which one the estimator actually got.
        print(
            f"[RATE] Optimizer parameter groups: model lr={args.learning_rate}, "
            f"rate estimator (loc/log_scale) lr={args.rate_lr}."
        )
    else:
        print("[RATE] Rate loss disabled (loss is plain MSE, as before Milestone 9).")

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_dir / "history.json"
    model_config = model.config_dict()
    # Milestone 9: seed the best-so-far from history records produced under the
    # SAME objective as this run, not blindly from every record present.
    #
    # `val_loss` means different things in different runs: plain MSE for a
    # distortion-only run, `D + lambda*R` for a rate-aware one - and the two are
    # not on a comparable scale. Resuming the M8 QAT checkpoint into an M9 run
    # seeded best_val_loss with M8's pure-MSE minimum (4.55e-04, measured on
    # Vimeo), which no `D + lambda*R` epoch can ever beat, so `best.pt` was
    # silently never written for the entire run. Every M9C/M9C.1 pilot arm
    # produced only `latest.pt` because of this.
    #
    # The comparison criterion itself needs no change: `val_metrics["loss"]` for
    # a rate-enabled run already IS `D + lambda*R`, exactly the M9 objective.
    # Only the starting value was wrong.
    #
    # Backward compatible by construction: for a distortion-only run every
    # historical record matches (`rate_enabled` absent or False both sides), so
    # M7/M8-style runs keep the exact behaviour they have always had.
    rate_enabled = rate_estimator is not None

    def _same_objective(record: dict) -> bool:
        if bool(record.get("rate_enabled", False)) != rate_enabled:
            return False
        # Two rate runs at different lambdas are also different objectives.
        return not rate_enabled or record.get("rate_lambda") == args.rate_lambda

    comparable = [record["val_loss"] for record in history if _same_objective(record)]
    best_val_loss = min(comparable, default=float("inf"))
    if history and not comparable:
        print(
            f"[BEST] Prior history was produced under a different objective "
            f"({len(history)} record(s) ignored); best-checkpoint tracking starts fresh "
            f"for this run's own objective."
        )

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
            # Milestone 9C.1: recorded per epoch like every other rate/qat field,
            # so a history.json says which LR the estimator was trained at.
            "rate_lr": args.rate_lr if rate_estimator is not None else None,
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
            {
                "rate_estimator_state_dict": rate_estimator.state_dict(),
                "rate_lambda": args.rate_lambda,
                "rate_lr": args.rate_lr,
            }
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
            objective = "D + lambda*R" if rate_enabled else "MSE"
            print(f"  [BEST] New best validation {objective}: {best_val_loss:.6e} -> {checkpoint_dir / 'best.pt'}")

        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print()
    print(f"Checkpoints written to: {checkpoint_dir}")
    print(f"History written to:     {history_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
