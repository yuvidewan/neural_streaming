"""BaselineAutoencoder: a deterministic (non-variational) convolutional
autoencoder combining Encoder and Decoder behind an explicit encode/decode
interface, so a later milestone can insert a quantizer + entropy coder
between them without restructuring this class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from nvc.models.decoder import Decoder
from nvc.models.encoder import Encoder

if TYPE_CHECKING:
    # Deferred to a type-checking-only import: nvc.training's package
    # __init__ imports checkpoint.py, which imports nvc.models - a real
    # runtime import here would be circular. `from __future__ import
    # annotations` (above) already makes every annotation in this file a
    # string, so the type hint below costs nothing at runtime either way.
    from nvc.training.quantization_noise import QuantizationNoise


class BaselineAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 64,
        base_channels: int = 32,
        *,
        quantization_noise: QuantizationNoise | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels

        self.encoder = Encoder(in_channels, latent_channels, base_channels)
        self.decoder = Decoder(in_channels, latent_channels, base_channels)

        # Milestone 8A (quantization-aware training). Deliberately NOT part
        # of config_dict()/model_config: it is a *training method*, not an
        # architectural choice, so a checkpoint rebuilt via
        # BaselineAutoencoder(**checkpoint["model_config"]) (the standard
        # inference path in nvc.training.checkpoint) never re-attaches it -
        # which is correct, since inference always runs in eval mode where
        # it would never fire anyway (see forward()). A QAT training script
        # must attach it explicitly on every invocation, exactly as
        # --latent-channels already must be re-supplied identically across
        # --resume runs. It is a plain object, not an nn.Module/Parameter,
        # so it is never captured by state_dict()/parameters()/.to(device).
        self.quantization_noise = quantization_noise

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        # Gated on nn.Module's own training/eval flag (model.train() /
        # model.eval()), not a forward() argument - every existing caller
        # already sets this correctly (train_one_epoch calls model.train(),
        # validate_one_epoch/load_model_from_checkpoint/reconstruction and
        # benchmark scripts all use eval mode), so the standard evaluation
        # and inference paths can never accidentally inject noise, and nothing
        # about the existing model(x) call signature changes.
        if self.training and self.quantization_noise is not None:
            z = self.quantization_noise.apply(z)
        return self.decode(z)

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
