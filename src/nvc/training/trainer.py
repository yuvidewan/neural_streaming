"""Training/validation epoch loops for BaselineAutoencoder.

Kept separate from the model definition (src/nvc/models/) so a later
milestone can swap in a different model without rewriting the epoch loop,
and separate from checkpointing (checkpoint.py) so the two concerns don't
tangle.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nvc.evaluation.basic_metrics import mse, psnr


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
