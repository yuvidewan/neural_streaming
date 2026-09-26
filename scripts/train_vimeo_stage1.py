"""PARITY_ROADMAP Stage 1 - train the GDN/residual transform on Vimeo-90K.

Chunked, resumable training of `ResidualGDNAutoencoder` (8,437,827 parameters)
over the ten Kaggle Vimeo-90K chunks. Designed to be driven either from a local
machine or from `colab_train_stage1.ipynb`, which is a thin wrapper around this
file: mount Drive, install, run this with `--output-dir` pointed at Drive.

WHY A NEW SCRIPT AND NOT A FLAG ON THE OLD ONE
-----------------------------------------------
`train_vimeo_qat_combined.py` trains two BaselineAutoencoder arms (QAT and its
matched control) and is the record of Milestone 8B. It is not modified here.
Its chunk machinery - Kaggle download, collision-reconciling extraction,
symlinking one chunk at a time into `sequences/`, per-chunk split lists,
manifest building, progress bookkeeping - is reused read-only through the usual
`_load_script` helper, so there is exactly one implementation of the awkward
parts and this file only adds what Stage 1 needs.

THE TWO PHASES, AND WHY THERE ARE TWO
--------------------------------------
The rate proxy needs a quantization bin width, and a bin width comes from
calibrating a *trained* model - a from-scratch network has no meaningful latent
scale to calibrate. So Stage 1 trains the way the baseline lineage did (M7/M8
distortion-only, then M9+ rate):

  Phase A (default)   distortion only, from scratch.
      python scripts/train_vimeo_stage1.py --output-dir <dir>

  then calibrate the Phase A result:
      python scripts/calibrate_quantizer.py --checkpoint <dir>/checkpoints/best.pt \
          --manifest <a Vimeo train manifest> --bits 4 --mode per_channel \
          --output <dir>/calibration/stage1_4bit_train.json

  Phase B              D + lambda*R, continuing from Phase A's weights.
      python scripts/train_vimeo_stage1.py --output-dir <dir> --rate-enabled \
          --rate-calibration <dir>/calibration/stage1_4bit_train.json \
          --rate-lambda <value> --rate-track-scale --reset-progress

`--rate-track-scale` is recommended for Phase B for the reason M9F.5 found: with
a frozen bin width the encoder can lower the proxy rate just by shrinking the
latent, which the real quantizer (it recalibrates per model) does not reward.

RESUMING
--------
Interrupt and re-run at any time. `progress.json` records which chunks are done
and they are skipped; training continues from `checkpoints/latest.pt`, including
the optimizer state and the rate estimator's own parameters. This matters more
than usual here: Colab disconnects, and a full pass over ten chunks is long.

WHAT IT DOES NOT DO
-------------------
Measure anything. The Stage 1 gate is intra-only BD-rate from
`scripts/benchmark_parity.py --gop 1`, against the denominator in
`outputs/benchmarks/parity_intra/` (+176.2% PSNR for the current codec). A
checkpoint from this script is an input to that measurement, not a result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

from nvc.data.loaders import (
    create_sequence_test_loader,
    create_sequence_train_loader,
)
from nvc.models import ResidualGDNAutoencoder
from nvc.training import (
    QuantizationNoise,
    RateEstimator,
    save_checkpoint,
    train_one_epoch,
    train_one_epoch_with_rate,
    validate_one_epoch,
    validate_one_epoch_with_rate,
)
from nvc.training.checkpoint import load_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_CHUNKS = list(range(1, 11))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Stage 1 GDN/residual transform on chunked Vimeo-90K.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/stage1_vimeo"),
        help="Everything this run owns: checkpoints/, progress.json, history.json. "
             "Point it at Google Drive when running on Colab so a disconnect loses nothing.",
    )
    parser.add_argument("--chunks", type=int, nargs="+", default=DEFAULT_CHUNKS,
                        help="Kaggle chunk numbers to cycle through, in order.")
    parser.add_argument("--scratch-dir", type=Path, default=Path("/content/vimeo_scratch"),
                        help="Working area for the chunk being trained on. Wiped per chunk.")
    parser.add_argument("--vimeo-root", type=Path, default=None,
                        help="Where sequences/ is symlinked. Defaults to <scratch>/vimeo_root.")

    parser.add_argument("--epochs-per-chunk-max", type=int, default=4,
                        help="Ceiling per chunk; early stopping usually stops sooner.")
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument(
        "--early-stop-min-improvement", type=float, default=0.002,
        help=(
            "Fractional improvement in validation loss that counts as progress, "
            "RELATIVE to the chunk's best so far (0.002 = 0.2%%). Relative rather "
            "than absolute on purpose: an absolute threshold is a different bar at "
            "every scale. The previous 1e-5 absolute default was ~1%% of the val MSE "
            "after one epoch but ~5%% once MSE reached 2e-4, so late chunks stopped at "
            "the patience floor whether or not they were still learning - the metric's "
            "scale decided, not convergence."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16,
                        help="16 rather than the baseline's 32: the Stage 1 model is 14x larger "
                             "and peaks near 3 GiB at this batch and crop size.")
    parser.add_argument("--crop-size", type=int, default=256,
                        help="Vimeo frames are 448x256; must be divisible by 16.")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Adam's rate for the transform's own weights. 1e-4 is the "
                             "standard starting rate for this architecture class.")
    parser.add_argument(
        "--lr-decay-chunks", type=int, default=2,
        help=(
            "Run the LAST this-many chunks of the schedule at the decayed rate "
            "(--learning-rate * --lr-decay-factor). This is the final decay the "
            "reference recipes for this architecture class all end with - typically "
            "1e-4 down to 1e-5 - and it is worth a few tenths of a dB that no amount "
            "of extra epochs at the undecayed rate recovers. 0 disables it."
        ),
    )
    parser.add_argument("--lr-decay-factor", type=float, default=0.1,
                        help="Multiplier applied during --lr-decay-chunks.")
    parser.add_argument("--latent-channels", type=int, default=192)
    parser.add_argument("--base-channels", type=int, default=192)
    parser.add_argument("--residual-blocks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Colab is Linux, so >0 is safe; use 0 on Windows.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--rate-enabled", action="store_true",
                        help="Phase B: train on D + lambda*R instead of distortion alone. "
                             "Requires --rate-calibration.")
    parser.add_argument("--rate-lambda", type=float, default=0.0)
    parser.add_argument("--rate-lr", type=float, default=defaults.rate_lr)
    parser.add_argument("--rate-calibration", type=Path, default=None,
                        help="calibrate_quantizer.py output from the TRAIN split, supplying the "
                             "rate estimator's bin width.")
    parser.add_argument("--rate-track-scale", action="store_true",
                        help="Nudge the bin width toward each batch's own dynamic range "
                             "(M9F.5). Recommended for Phase B.")
    parser.add_argument("--rate-scale-momentum", type=float, default=defaults.rate_scale_momentum)

    parser.add_argument("--kaggle-dataset-owner", default="wangsally")
    parser.add_argument("--kaggle-dataset-prefix", default="vimeo-90k")
    parser.add_argument("--reuse-chunk", action="store_true",
                        help="Skip the download when a chunk's frames are already extracted "
                             "in --scratch-dir. Pairs with --keep-chunk to make a re-run after "
                             "a crash cost nothing. Off by default: on Colab the VM is wiped "
                             "between sessions, so the data is never there to reuse.")
    parser.add_argument("--keep-chunk", action="store_true",
                        help="Do not delete each chunk after training on it. Needs ~89GB free "
                             "for the full set, so off by default.")
    parser.add_argument("--reset-progress", action="store_true",
                        help="Clear the completed-chunk list so every chunk is visited again, "
                             "keeping the weights. This is how Phase B starts from Phase A.")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Cap batches per epoch. Smoke tests only.")
    return parser


def build_model(args, quantization_noise=None) -> ResidualGDNAutoencoder:
    return ResidualGDNAutoencoder(
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        residual_blocks=args.residual_blocks,
        quantization_noise=quantization_noise,
    )


def build_rate_estimator(args) -> RateEstimator | None:
    """The rate proxy for Phase B, or None for Phase A.

    The bin width comes from a calibration file rather than from anything this
    script invents, so the proxy's quantization scale is the same object the
    real quantizer uses - see rate_estimator.py's own docstring on why that
    correspondence is the whole point.
    """
    if not args.rate_enabled:
        return None
    if args.rate_calibration is None:
        raise SystemExit("--rate-enabled requires --rate-calibration (see this script's docstring)")
    if not args.rate_calibration.is_file():
        raise SystemExit(f"--rate-calibration not found: {args.rate_calibration}")
    # bits/mode come from the file's own record; from_calibration also refuses
    # anything not calibrated on the train split.
    noise = QuantizationNoise.from_calibration(args.rate_calibration)
    return RateEstimator(
        noise.scale, bits=noise.bits, mode=noise.mode,
        track_scale=args.rate_track_scale, scale_momentum=args.rate_scale_momentum,
    )


def prepare_chunk(chunks, chunk_number: int, chunk_dir: Path, args) -> Path:
    """Get this chunk's frames on disk, and return the folder holding its groups.

    `download_and_extract_chunk` wipes its scratch directory before downloading,
    so an interrupted run re-pays the whole 6-10GB download even when the frames
    are still sitting there. That is the right default on Colab, where the VM is
    wiped between sessions anyway, but locally it is minutes of wasted bandwidth
    every time a chunk crashes - hence --reuse-chunk.

    Reuse is only taken when the directory actually holds an extracted chunk:
    `_find_sequences_source_root` locates it by finding an im1.png, and raises if
    there is none, in which case this falls through to a normal download rather
    than training on a half-extracted tree.
    """
    if args.reuse_chunk and chunk_dir.is_dir():
        try:
            group_root = chunks._find_sequences_source_root(chunk_dir)
        except RuntimeError:
            print(f"[chunk {chunk_number}] --reuse-chunk: nothing extracted at {chunk_dir}, "
                  "downloading", flush=True)
        else:
            print(f"[chunk {chunk_number}] reusing already-extracted frames at {group_root}",
                  flush=True)
            return group_root
    return chunks.download_and_extract_chunk(
        chunk_number, chunk_dir,
        dataset_owner=args.kaggle_dataset_owner,
        dataset_prefix=args.kaggle_dataset_prefix)


def build_loaders(train_manifest: Path, val_manifest: Path, args):
    """Sequence loaders, not the frame ones.

    `build_chunk_manifests` writes a Vimeo SEQUENCE manifest - items carry
    `sequence_id` and `frame_filenames` - and `FrameDataset` cannot read one: it
    wants a `frame_directory` per item and dies with `KeyError: 'frame_directory'`.
    That failure only surfaces after a 9GB download, an extraction and a symlink
    pass, which is why this is a function with a test rather than four lines
    inline in `main`.

    The "val" loader is the chunk's own test split. It exists only to give this
    chunk a validation signal and is discarded with the chunk; it is not the
    official Vimeo split and not a benchmark. The real comparison is DAVIS, which
    nothing here trains on.
    """
    train_loader = create_sequence_train_loader(
        train_manifest, batch_size=args.batch_size, num_workers=args.num_workers,
        seed=args.seed, crop_size=args.crop_size)
    val_loader = create_sequence_test_loader(
        val_manifest, batch_size=args.batch_size, num_workers=args.num_workers,
        crop_size=args.crop_size)
    return train_loader, val_loader


def build_optimizer(model, rate_estimator, args) -> torch.optim.Optimizer:
    """Model parameters at --learning-rate; the estimator's own loc/log_scale in
    a separate group at --rate-lr, exactly as train_autoencoder.py does it
    (M9C.1: 128 scalars fitting a density need far larger steps than the model)."""
    if rate_estimator is None:
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    return torch.optim.Adam(
        [
            {"params": list(model.parameters()), "lr": args.learning_rate},
            {"params": list(rate_estimator.parameters()), "lr": args.rate_lr},
        ]
    )


def learning_rate_for_chunk(chunk_number: int, chunks: list[int], args) -> float:
    """The transform's learning rate while training on this chunk.

    Derived from the chunk's POSITION in the schedule rather than from a
    torch scheduler's internal state, for one practical reason: this script is
    built to be interrupted and resumed, and a position-derived rate is correct
    after a resume without any scheduler state to checkpoint or restore.

    A plateau scheduler would also be the wrong instrument here. Validation loss
    is measured on each chunk's OWN held-out split, so it is a different dataset
    every chunk and not comparable across them - "no improvement" would often
    mean "harder chunk", not "converged".
    """
    if args.lr_decay_chunks <= 0 or chunk_number not in chunks:
        return args.learning_rate
    position = chunks.index(chunk_number)
    if position >= max(0, len(chunks) - args.lr_decay_chunks):
        return args.learning_rate * args.lr_decay_factor
    return args.learning_rate


def set_transform_learning_rate(optimizer: torch.optim.Optimizer, rate: float) -> None:
    """Set the rate on the model's parameter group only.

    The rate estimator, when present, lives in its own group at --rate-lr and is
    left alone: its 2*C scalars are fitting a density and need O(1) movement
    (M9C.1), which decaying them would take away for no benefit. `build_optimizer`
    puts the model's group first.
    """
    optimizer.param_groups[0]["lr"] = rate


def train_one_chunk(
    *, model, optimizer, rate_estimator, train_loader, val_loader, device, args,
    start_epoch: int, chunk_number: int, history: list[dict[str, Any]],
    checkpoint_dir: Path, model_config: dict[str, Any], best_val_loss: float,
    checkpoint_extra,
) -> tuple[int, float]:
    """Up to --epochs-per-chunk-max epochs on one chunk, stopping early when the
    validation loss stops improving. Best-checkpoint tracking is GLOBAL across
    chunks; only the early-stopping decision is scoped to this chunk.
    """
    epoch = start_epoch
    best_chunk_val_loss = float("inf")
    stalled = 0

    for step in range(args.epochs_per_chunk_max):
        if rate_estimator is None:
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, device, max_batches=args.max_batches,
                progress_desc=f"chunk {chunk_number} epoch {epoch} train")
            val_metrics = validate_one_epoch(
                model, val_loader, device, max_batches=args.max_batches,
                progress_desc=f"chunk {chunk_number} epoch {epoch} val")
        else:
            train_metrics = train_one_epoch_with_rate(
                model, train_loader, optimizer, device, rate_estimator=rate_estimator,
                lambda_rate=args.rate_lambda, max_batches=args.max_batches,
                progress_desc=f"chunk {chunk_number} epoch {epoch} train")
            val_metrics = validate_one_epoch_with_rate(
                model, val_loader, device, rate_estimator=rate_estimator,
                lambda_rate=args.rate_lambda, max_batches=args.max_batches,
                progress_desc=f"chunk {chunk_number} epoch {epoch} val")

        val_loss = val_metrics["loss"]
        record = {
            "epoch": epoch, "chunk": chunk_number,
            "train_loss": train_metrics["loss"], "val_loss": val_loss,
            "val_psnr": val_metrics["psnr"],
            "rate_enabled": rate_estimator is not None,
            "rate_lambda": args.rate_lambda if rate_estimator is not None else None,
            # Recorded per epoch so the decay is visible in history.json rather
            # than something you have to infer from the chunk schedule.
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        # Only the rate loops report these; the distortion-only ones do not.
        for key in ("distortion", "rate"):
            if key in val_metrics:
                record[f"val_{key}"] = val_metrics[key]
        history.append(record)

        extra_note = ""
        if "rate" in val_metrics:
            extra_note = (f" val_distortion={val_metrics['distortion']:.6f}"
                          f" val_rate={val_metrics['rate']:.4f}")
        print(f"  chunk {chunk_number} epoch {epoch} (step {step + 1}/{args.epochs_per_chunk_max}): "
              f"train={train_metrics['loss']:.6f} val={val_loss:.6f} "
              f"val_psnr={val_metrics['psnr']:.2f} dB{extra_note}", flush=True)

        extra = checkpoint_extra()
        save_checkpoint(checkpoint_dir / "latest.pt", model=model, optimizer=optimizer,
                        epoch=epoch, history=history, model_config=model_config, extra=extra)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(checkpoint_dir / "best.pt", model=model, optimizer=optimizer,
                            epoch=epoch, history=history, model_config=model_config, extra=extra)
            print(f"    new global-best val loss {best_val_loss:.6f}", flush=True)
        (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        epoch += 1
        # Relative, so the bar means the same thing at 1e-3 and at 1e-5. The
        # first epoch's `inf` best is handled explicitly: inf * 0.0 is NaN, and
        # every comparison against NaN is False, which would count the opening
        # epoch as a stall.
        threshold = (best_chunk_val_loss * (1.0 - args.early_stop_min_improvement)
                     if math.isfinite(best_chunk_val_loss) else float("inf"))
        if val_loss < threshold:
            best_chunk_val_loss = val_loss
            stalled = 0
        else:
            stalled += 1
            if stalled >= args.early_stop_patience:
                print(f"  early stop on chunk {chunk_number} after {step + 1} epoch(s)", flush=True)
                break

    return epoch, best_val_loss


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    if args.crop_size % 16 != 0:
        raise SystemExit(f"--crop-size must be divisible by 16, got {args.crop_size}")
    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    chunks = _load_script("train_vimeo_qat_combined")

    output_dir = args.output_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    scratch_dir = args.scratch_dir
    vimeo_root = args.vimeo_root or (scratch_dir / "vimeo_root")

    rate_estimator = build_rate_estimator(args)
    if rate_estimator is not None:
        rate_estimator = rate_estimator.to(device)
    model = build_model(args).to(device)
    optimizer = build_optimizer(model, rate_estimator, args)
    model_config = model.config_dict()

    def checkpoint_extra():
        if rate_estimator is None:
            return None
        return {"rate_estimator_state_dict": rate_estimator.state_dict(),
                "rate_lambda": args.rate_lambda,
                "rate_track_scale": args.rate_track_scale}

    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_val_loss = float("inf")
    latest = checkpoint_dir / "latest.pt"
    if latest.is_file():
        checkpoint = load_checkpoint(latest, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        saved_rate = (checkpoint.get("extra") or {}).get("rate_estimator_state_dict")
        if rate_estimator is not None and saved_rate is not None:
            rate_estimator.load_state_dict(saved_rate)
        # The optimizer's parameter list changes between Phase A and Phase B
        # (the estimator's loc/log_scale join it), so a cross-phase resume must
        # start the optimizer fresh rather than fail - M9C's reasoning exactly.
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError:
            print("[resume] optimizer state does not match this objective; starting it fresh "
                  "(expected when moving from Phase A to Phase B)", flush=True)
        print(f"[resume] from {latest} at epoch {start_epoch}", flush=True)

    progress = chunks._load_or_init_progress(progress_path)
    if args.reset_progress:
        print(f"[progress] clearing {len(progress['completed_chunks'])} completed chunk(s); "
              "weights are kept", flush=True)
        progress["completed_chunks"] = []
    if progress.get("best_val_loss") is not None and not args.reset_progress:
        best_val_loss = progress["best_val_loss"]

    print("=" * 70)
    print(f"Stage 1 transform: {type(model).__name__}, {model.num_parameters():,} parameters")
    print(f"Objective: {'D + %g*R' % args.rate_lambda if rate_estimator is not None else 'distortion only (Phase A)'}")
    print(f"Chunks: {args.chunks}   done already: {progress['completed_chunks']}")
    print(f"Device: {device}   batch {args.batch_size} at {args.crop_size}x{args.crop_size}")
    if args.lr_decay_chunks > 0 and len(args.chunks) > args.lr_decay_chunks:
        decayed = [c for c in args.chunks
                   if learning_rate_for_chunk(c, list(args.chunks), args) != args.learning_rate]
        print(f"Learning rate: {args.learning_rate:g}, decayed to "
              f"{args.learning_rate * args.lr_decay_factor:g} on chunk(s) {decayed}")
    else:
        print(f"Learning rate: {args.learning_rate:g} throughout (no final decay)")
    print(f"Early stop: patience {args.early_stop_patience}, needs "
          f"{args.early_stop_min_improvement:.3%} relative improvement")
    print("=" * 70, flush=True)

    for chunk_number in args.chunks:
        if chunk_number in progress["completed_chunks"]:
            print(f"[chunk {chunk_number}] already done, skipping", flush=True)
            continue

        group_root = prepare_chunk(
            chunks, chunk_number, scratch_dir / f"chunk_{chunk_number}", args)
        chunks.relink_sequences_to_chunk(group_root, vimeo_root)
        sequence_ids = chunks.discover_complete_sequence_ids(vimeo_root / "sequences")
        chunks.write_chunk_split_lists(vimeo_root, sequence_ids, args.seed)
        train_manifest, val_manifest = chunks.build_chunk_manifests(
            vimeo_root, output_dir / "vimeo_manifest.json", args.seed)

        train_loader, val_loader = build_loaders(train_manifest, val_manifest, args)

        chunk_lr = learning_rate_for_chunk(chunk_number, list(args.chunks), args)
        set_transform_learning_rate(optimizer, chunk_lr)
        if chunk_lr != args.learning_rate:
            print(f"[chunk {chunk_number}] final decay: transform learning rate "
                  f"{args.learning_rate:g} -> {chunk_lr:g}", flush=True)

        start_epoch, best_val_loss = train_one_chunk(
            model=model, optimizer=optimizer, rate_estimator=rate_estimator,
            train_loader=train_loader, val_loader=val_loader, device=device, args=args,
            start_epoch=start_epoch, chunk_number=chunk_number, history=history,
            checkpoint_dir=checkpoint_dir, model_config=model_config,
            best_val_loss=best_val_loss, checkpoint_extra=checkpoint_extra)

        progress["completed_chunks"].append(chunk_number)
        progress["best_val_loss"] = best_val_loss
        chunks._save_progress(progress_path, progress)

        if not args.keep_chunk:
            shutil.rmtree(scratch_dir / f"chunk_{chunk_number}", ignore_errors=True)
            print(f"[chunk {chunk_number}] deleted to free disk", flush=True)

    print("=" * 70)
    print(f"Done. Best validation loss {best_val_loss:.6f}")
    print(f"Checkpoints: {checkpoint_dir}")
    print("Next: calibrate (Phase A -> B), or measure the gate with "
          "scripts/benchmark_parity.py --gop 1")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
