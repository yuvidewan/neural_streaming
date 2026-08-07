"""Convolutional encoder: downsamples an RGB frame into a compact spatial
latent tensor using strided convolutions only - no pooling, attention, or
pretrained backbone.
"""

from __future__ import annotations

import torch
from torch import nn

_DOWNSAMPLE_FACTOR = 16  # four stride-2 convolutions: 2**4


class Encoder(nn.Module):
    """[B, in_channels, H, W] -> [B, latent_channels, H/16, W/16].

    H and W must each be divisible by 16 (four stride-2 convolutions with
    kernel_size=4, stride=2, padding=1, which halves spatial size exactly).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 64,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # No activation on the latent itself - it stays an unconstrained
            # real-valued tensor, which a later milestone will quantize.
            nn.Conv2d(c3, latent_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Encoder expects a [B, {self.in_channels}, H, W] tensor, got shape {tuple(x.shape)}"
            )
        _, _, height, width = x.shape
        if height % _DOWNSAMPLE_FACTOR != 0 or width % _DOWNSAMPLE_FACTOR != 0:
            raise ValueError(
                f"Encoder input height and width must each be divisible by "
                f"{_DOWNSAMPLE_FACTOR} (four stride-2 convolutions), got {height}x{width}"
            )
        return self.net(x)
