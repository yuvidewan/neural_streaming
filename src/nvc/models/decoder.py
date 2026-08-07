"""Convolutional decoder: upsamples a compact spatial latent tensor back
into an RGB frame using transposed convolutions, exactly mirroring
Encoder's four stride-2 downsampling steps so shapes reconstruct exactly
without any final resizing.
"""

from __future__ import annotations

import torch
from torch import nn


class Decoder(nn.Module):
    """[B, latent_channels, H, W] -> [B, out_channels, H*16, W*16]."""

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 64,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        c1, c2, c3 = base_channels * 4, base_channels * 2, base_channels
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c3, out_channels, kernel_size=4, stride=2, padding=1),
            # Sigmoid: reconstruction must land in [0, 1], matching
            # FrameDataset's pixel range - acceptable for this deterministic
            # baseline (no perceptual/adversarial loss yet).
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 4 or z.shape[1] != self.latent_channels:
            raise ValueError(
                f"Decoder expects a [B, {self.latent_channels}, H, W] tensor, got shape {tuple(z.shape)}"
            )
        return self.net(z)
