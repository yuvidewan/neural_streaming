"""Training/validation epoch loops for BaselineAutoencoder.

Kept separate from the model definition (src/nvc/models/) so a later
milestone can swap in a different model without rewriting the epoch loop,
and separate from checkpointing (checkpoint.py) so the two concerns don't
tangle.
"""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nvc.evaluation.basic_metrics import mse, psnr
from nvc.training.rate_estimator import RateEstimator


def _progress_total(loader: DataLoader, max_batches: int | None) -> int | None:
    """Batch count for a progress bar, respecting max_batches truncation.

    None (unknown length) is a valid DataLoader state - tqdm handles that
    fine, just without an ETA.
    """
    try:
        total = len(loader)
    except TypeError:
        return None
    return total if max_batches is None else min(max_batches, total)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_batches: int | None = None,
    progress_desc: str | None = None,
) -> dict[str, float]:
    """Run one training epoch with real gradient updates.

    max_batches caps the number of batches processed - used for quick CPU
    smoke tests, not full training. progress_desc is opt-in: pass a label
    (e.g. "chunk 3 epoch 2") to get a live tqdm bar with a running-loss
    postfix and an ETA for this epoch; omit it (the default) for the
    original silent behavior every existing caller relies on.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    iterator = (
        tqdm(loader, total=_progress_total(loader, max_batches), desc=progress_desc, leave=False)
        if progress_desc is not None else loader
    )

    for batch in iterator:
        batch = batch.to(device)

        optimizer.zero_grad()
        reconstruction = model(batch)
        loss = mse(reconstruction, batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        if progress_desc is not None:
            iterator.set_postfix(mse=f"{total_loss / num_batches:.6f}")
        if max_batches is not None and num_batches >= max_batches:
            break

    if progress_desc is not None:
        iterator.close()

    if num_batches == 0:
        raise ValueError("train_one_epoch: the data loader produced no batches")
    return {"loss": total_loss / num_batches}


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = None,
    progress_desc: str | None = None,
) -> dict[str, float]:
    """Run one validation epoch - no gradients, no parameter updates.

    progress_desc: see train_one_epoch - same opt-in tqdm behavior.
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0

    iterator = (
        tqdm(loader, total=_progress_total(loader, max_batches), desc=progress_desc, leave=False)
        if progress_desc is not None else loader
    )

    for batch in iterator:
        batch = batch.to(device)

        reconstruction = model(batch)
        batch_loss = mse(reconstruction, batch)
        batch_psnr = psnr(reconstruction, batch)

        total_loss += batch_loss.item()
        total_psnr += batch_psnr.item()
        num_batches += 1
        if progress_desc is not None:
            iterator.set_postfix(mse=f"{total_loss / num_batches:.6f}", psnr=f"{total_psnr / num_batches:.2f}dB")
        if max_batches is not None and num_batches >= max_batches:
            break

    if progress_desc is not None:
        iterator.close()

    if num_batches == 0:
        raise ValueError("validate_one_epoch: the data loader produced no batches")
    return {"loss": total_loss / num_batches, "psnr": total_psnr / num_batches}


def _check_lambda_rate(lambda_rate: float) -> None:
    if not math.isfinite(lambda_rate) or lambda_rate < 0:
        raise ValueError(f"lambda_rate must be finite and >= 0, got {lambda_rate}")


def train_one_epoch_with_rate(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    rate_estimator: RateEstimator,
    lambda_rate: float,
    max_batches: int | None = None,
    progress_desc: str | None = None,
) -> dict[str, float]:
    """Milestone 9A: one training epoch under `D + lambda * R`.

    Deliberately NOT a modification of `train_one_epoch` above - that
    function is untouched and remains the exact distortion-only path (see
    `tests/test_rate_estimator.py`'s lambda=0 equivalence test, which
    proves this function's model-parameter gradients match
    `train_one_epoch`'s exactly when `lambda_rate=0.0`).

    Does not call `model(batch)` (`BaselineAutoencoder.forward()`), which
    never returns the latent - instead this replicates forward()'s own
    encode -> [QAT noise, if attached] -> decode dispatch explicitly, using
    only the model's existing public `encode()`/`decode()`/
    `quantization_noise` (see `models/autoencoder.py`), so the rate
    estimator can see the same latent forward() would have used internally.
    `BaselineAutoencoder` itself is not modified anywhere for this.

    `lambda_rate` must be finite and >= 0 - see `_check_lambda_rate`.
    `rate_estimator` is evaluated every batch regardless of `lambda_rate`
    (so its own parameters keep receiving a gradient signal even at
    lambda=0, for numerical-stability testing) - but at lambda=0 that
    signal is multiplied by exactly 0.0 before being summed into
    `total_loss`, so it never affects `model`'s own gradients.
    """
    _check_lambda_rate(lambda_rate)

    model.train()
    total_loss = 0.0
    total_distortion = 0.0
    total_rate = 0.0
    num_batches = 0

    iterator = (
        tqdm(loader, total=_progress_total(loader, max_batches), desc=progress_desc, leave=False)
        if progress_desc is not None else loader
    )

    for batch in iterator:
        batch = batch.to(device)
        image_pixels = batch.shape[-2] * batch.shape[-1]

        optimizer.zero_grad()
        latent = model.encode(batch)
        if model.training and model.quantization_noise is not None:
            latent = model.quantization_noise.apply(latent)
        rate = rate_estimator(latent, image_pixels)
        reconstruction = model.decode(latent)
        distortion = mse(reconstruction, batch)
        loss = distortion + lambda_rate * rate
        loss.backward()
        optimizer.step()
        # After the step, not before: this batch's rate was scored against
        # the bin width as of the start of the step. No-op unless
        # rate_estimator was built with track_scale=True (see
        # rate_estimator.py, "SCALE TRACKING" - the MILESTONE_9_PLAN.md
        # Section 9F.5 fix). Training-only: validate_one_epoch_with_rate
        # below does NOT call this, so evaluating on a val batch never has
        # the side effect of moving tracked state.
        rate_estimator.update_bin_width(latent)

        total_loss += loss.item()
        total_distortion += distortion.item()
        total_rate += rate.item()
        num_batches += 1
        if progress_desc is not None:
            iterator.set_postfix(
                loss=f"{total_loss / num_batches:.6f}",
                d=f"{total_distortion / num_batches:.6f}",
                r=f"{total_rate / num_batches:.4f}bpp",
            )
        if max_batches is not None and num_batches >= max_batches:
            break

    if progress_desc is not None:
        iterator.close()

    if num_batches == 0:
        raise ValueError("train_one_epoch_with_rate: the data loader produced no batches")
    return {
        "loss": total_loss / num_batches,
        "distortion": total_distortion / num_batches,
        "rate": total_rate / num_batches,
    }


@torch.no_grad()
def validate_one_epoch_with_rate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    rate_estimator: RateEstimator,
    lambda_rate: float,
    max_batches: int | None = None,
    progress_desc: str | None = None,
) -> dict[str, float]:
    """Validation counterpart to `train_one_epoch_with_rate` - no gradients,
    no parameter updates. Same D + lambda*R bookkeeping, plus PSNR
    (matching `validate_one_epoch`'s own reporting)."""
    _check_lambda_rate(lambda_rate)

    model.eval()
    total_loss = 0.0
    total_distortion = 0.0
    total_rate = 0.0
    total_psnr = 0.0
    num_batches = 0

    iterator = (
        tqdm(loader, total=_progress_total(loader, max_batches), desc=progress_desc, leave=False)
        if progress_desc is not None else loader
    )

    for batch in iterator:
        batch = batch.to(device)
        image_pixels = batch.shape[-2] * batch.shape[-1]

        latent = model.encode(batch)
        if model.training and model.quantization_noise is not None:
            latent = model.quantization_noise.apply(latent)
        rate = rate_estimator(latent, image_pixels)
        reconstruction = model.decode(latent)
        distortion = mse(reconstruction, batch)
        loss = distortion + lambda_rate * rate
        batch_psnr = psnr(reconstruction, batch)

        total_loss += loss.item()
        total_distortion += distortion.item()
        total_rate += rate.item()
        total_psnr += batch_psnr.item()
        num_batches += 1
        if progress_desc is not None:
            iterator.set_postfix(
                loss=f"{total_loss / num_batches:.6f}", psnr=f"{total_psnr / num_batches:.2f}dB",
            )
        if max_batches is not None and num_batches >= max_batches:
            break

    if progress_desc is not None:
        iterator.close()

    if num_batches == 0:
        raise ValueError("validate_one_epoch_with_rate: the data loader produced no batches")
    return {
        "loss": total_loss / num_batches,
        "distortion": total_distortion / num_batches,
        "rate": total_rate / num_batches,
        "psnr": total_psnr / num_batches,
    }
