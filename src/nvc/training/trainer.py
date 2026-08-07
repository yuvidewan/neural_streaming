"""Training/validation epoch loops for BaselineAutoencoder.

Kept separate from the model definition (src/nvc/models/) so a later
milestone can swap in a different model without rewriting the epoch loop,
and separate from checkpointing (checkpoint.py) so the two concerns don't
tangle.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from nvc.evaluation.basic_metrics import mse, psnr


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run one training epoch with real gradient updates.

    max_batches caps the number of batches processed - used for quick CPU
    smoke tests, not full training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        reconstruction = model(batch)
        loss = mse(reconstruction, batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        if max_batches is not None and num_batches >= max_batches:
            break

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
) -> dict[str, float]:
    """Run one validation epoch - no gradients, no parameter updates."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)

        reconstruction = model(batch)
        batch_loss = mse(reconstruction, batch)
        batch_psnr = psnr(reconstruction, batch)

        total_loss += batch_loss.item()
        total_psnr += batch_psnr.item()
        num_batches += 1
        if max_batches is not None and num_batches >= max_batches:
            break

    if num_batches == 0:
        raise ValueError("validate_one_epoch: the data loader produced no batches")
    return {"loss": total_loss / num_batches, "psnr": total_psnr / num_batches}
