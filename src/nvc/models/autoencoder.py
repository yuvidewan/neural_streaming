"""BaselineAutoencoder: a deterministic (non-variational) convolutional
autoencoder combining Encoder and Decoder behind an explicit encode/decode
interface, so a later milestone can insert a quantizer + entropy coder
between them without restructuring this class.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from nvc.models.decoder import Decoder
from nvc.models.encoder import Encoder


class BaselineAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 64,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels

        self.encoder = Encoder(in_channels, latent_channels, base_channels)
        self.decoder = Decoder(in_channels, latent_channels, base_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict[str, Any]:
        """Architecture config sufficient to rebuild this exact model via
        BaselineAutoencoder(**config_dict()) - used when loading a
        checkpoint, so the caller doesn't need to know the training-time
        CLI arguments.
        """
        return {
            "in_channels": self.in_channels,
            "latent_channels": self.latent_channels,
            "base_channels": self.base_channels,
        }
