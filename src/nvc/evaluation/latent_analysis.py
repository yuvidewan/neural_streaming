"""Extraction and statistical description of the trained encoder's latents.

Runs only the encoder half of a trained BaselineAutoencoder over a
DataLoader and collects the resulting latent tensors for analysis.

Memory behavior: batches are encoded on the compute device but each latent
is moved to CPU immediately, so GPU memory holds one batch at a time - the
full latent set is never accumulated in VRAM. On CPU the full DAVIS test
split is modest (719 frames x 64x16x16 float32 is about 47 MB), which is
what lets `latent_statistics` report an exact median rather than an
approximation.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

# |value| below this counts as "near zero" when reporting sparsity. The
# encoder's final layer is linear (no ReLU), so exact zeros are vanishingly
# rare; the near-zero fraction is the meaningful sparsity signal.
NEAR_ZERO_THRESHOLD = 1e-2


@torch.no_grad()
def extract_latents(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> torch.Tensor:
    """Encode every frame the loader yields; return latents as one CPU tensor.

    Returns shape [N, C, H, W]. Only `model.encode` is called - the decoder
    is never run, so this measures the representation itself.
    """
    model.eval()
    latents: list[torch.Tensor] = []

    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        latent = model.encode(batch.to(device))
        latents.append(latent.cpu())

    if not latents:
        raise ValueError("extract_latents: the data loader produced no batches")
    return torch.cat(latents, dim=0)


def latent_statistics(latents: torch.Tensor) -> dict:
    """Global and per-channel descriptive statistics for a latent set.

    `latents` is [N, C, H, W] (as returned by extract_latents).
    """
    if latents.dim() != 4:
        raise ValueError(
            f"latent_statistics expects a 4D [N, C, H, W] tensor, got shape {tuple(latents.shape)}"
        )

    flat = latents.flatten()
    num_channels = latents.shape[1]

    # Per-channel: collapse sample, height and width, keeping the channel axis.
    per_channel = latents.permute(1, 0, 2, 3).reshape(num_channels, -1)

    return {
        "num_samples": latents.shape[0],
        "latent_shape_per_sample": list(latents.shape[1:]),
        "elements_per_sample": int(latents[0].numel()),
        "total_elements": int(flat.numel()),
        "near_zero_threshold": NEAR_ZERO_THRESHOLD,
        "global": {
            "min": flat.min().item(),
            "max": flat.max().item(),
            "mean": flat.mean().item(),
            "std": flat.std().item(),
            "median": flat.median().item(),
            "percent_exactly_zero": (flat == 0).float().mean().item() * 100.0,
            "percent_near_zero": (flat.abs() < NEAR_ZERO_THRESHOLD).float().mean().item() * 100.0,
        },
        "per_channel": {
            "min": per_channel.min(dim=1).values.tolist(),
            "max": per_channel.max(dim=1).values.tolist(),
            "mean": per_channel.mean(dim=1).tolist(),
            "std": per_channel.std(dim=1).tolist(),
            "median": per_channel.median(dim=1).values.tolist(),
        },
    }
