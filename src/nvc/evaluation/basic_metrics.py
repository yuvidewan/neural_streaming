"""Minimal reconstruction-quality metrics: MSE and PSNR only.

MS-SSIM and other perceptual metrics are planned for a later milestone
once there's an actual reconstruction to evaluate - not implemented here.

Both functions operate on tensors of any matching shape (e.g. [C, H, W] or
[B, C, H, W]) already in [0, 1] range, matching what FrameDataset produces.
"""

from __future__ import annotations

import torch


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error. 0.0 for identical inputs."""
    return torch.mean((prediction - target) ** 2)


def psnr(prediction: torch.Tensor, target: torch.Tensor, *, max_value: float = 1.0) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio in dB.

    Returns +inf for identical inputs (MSE == 0) rather than relying on
    floating-point division-by-zero behavior - this is the correct
    mathematical limit, made explicit.
    """
    error = mse(prediction, target)
    if error == 0:
        return torch.tensor(float("inf"))
    return 10.0 * torch.log10((max_value ** 2) / error)
