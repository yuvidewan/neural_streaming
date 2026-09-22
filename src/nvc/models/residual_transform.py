"""The Stage 1 analysis/synthesis transform: GDN activations, residual
blocks, and enough capacity to be worth training.

WHAT THIS REPLACES, AND WHY
---------------------------
`BaselineAutoencoder` is four stride-2 convolutions with ReLU between them
and 593k parameters. The Stage 0 scoreboard measured what that costs:
+450.7% BD-rate against libx264, with the codec's rate-distortion slope
(1.3 dB per rate doubling, against H.264's 2.8 dB) pointing at the
transform rather than at the entropy coder. Twenty-two milestones of
quantizer and entropy work moved the total by a few percent each; the
architecture is the binding constraint.

This module is Stage 1 of PARITY_ROADMAP.md: the same four-stage,
stride-16 structure, with the three things the compression literature
actually changed.

  1. GDN/IGDN instead of ReLU (see gdn.py). A joint, smooth,
     channel-mixing nonlinearity that gaussianizes the latent instead of
     merely rectifying it.
  2. Residual blocks between the downsampling stages, so depth can grow
     without the gradient path growing with it.
  3. 192 channels rather than 32/64, which is where the parameter budget
     goes: ~8.5M at the default configuration against the baseline's 593k.

WHAT IT DELIBERATELY KEEPS
--------------------------
  - **Stride 16 exactly.** Four halvings in analysis, four doublings in
    synthesis. Everything downstream - the latent grids, the G16 context
    model, the motion block size, `nvc.video.codec.AUTOENCODER_STRIDE` -
    is written against a 16x downsample, and Stage 1's stated lever is the
    transform, not the latent geometry.
  - **The sigmoid on the reconstruction.** The rest of the pipeline treats
    decoded frames as living in [0, 1]: FrameDataset's range, the residual
    coder, the uint8 rounding in the scoreboard. Removing it is a separate
    change with its own blast radius, so it stays.
  - **The encode/decode/config_dict/num_parameters interface**, so the
    existing checkpoint, calibration and evaluation code paths take this
    model without modification.
  - **`quantization_noise`**, attached exactly as `BaselineAutoencoder`
    does it: a plain object, not a submodule, never part of config_dict(),
    and only ever applied in training mode.

SHAPE CONTRACT
--------------
Analysis maps [B, 3, H, W] -> [B, latent_channels, H/16, W/16] and
synthesis maps back, with H and W divisible by 16. The 5x5 stride-2
convolutions use padding 2 (giving exactly H/2 for even H) and the
transposed convolutions add output_padding 1 (giving exactly 2H), so the
round trip is shape-exact with no cropping or resizing anywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from nvc.models.gdn import GDN, IGDN

if TYPE_CHECKING:
    from nvc.training.quantization_noise import QuantizationNoise

DOWNSAMPLE_FACTOR = 16  # four stride-2 stages: 2**4

_KERNEL_SIZE = 5
_PADDING = 2


class ResidualBlock(nn.Module):
    """Conv3x3 -> LeakyReLU -> Conv3x3, added back to the input.

    No normalization layer: BatchNorm's running statistics would make the
    encoder's output depend on batch composition, which a codec cannot
    have - the same frame must encode to the same latent whether it
    arrives alone or in a batch of 32. LeakyReLU rather than ReLU because
    a residual branch that can only add non-negative corrections is a
    strictly weaker function, and the negative slope costs nothing.

    The second convolution starts at zero, so the block is exactly the
    identity at initialization and depth cannot hurt the starting point.
    """

    def __init__(self, channels: int, *, negative_slope: float = 0.1) -> None:
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.activation(self.conv1(x)))


def _residual_stack(channels: int, count: int) -> list[nn.Module]:
    return [ResidualBlock(channels) for _ in range(count)]


class AnalysisTransform(nn.Module):
    """[B, in_channels, H, W] -> [B, latent_channels, H/16, W/16].

    Residual blocks sit after the second and third downsampling stages -
    where the spatial resolution is already reduced (so they are cheap) but
    the representation is not yet the latent itself (so they still have
    room to work).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 192,
        base_channels: int = 192,
        residual_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        n = base_channels
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, n, _KERNEL_SIZE, stride=2, padding=_PADDING),
            GDN(n),
            nn.Conv2d(n, n, _KERNEL_SIZE, stride=2, padding=_PADDING),
            GDN(n),
            *_residual_stack(n, residual_blocks),
            nn.Conv2d(n, n, _KERNEL_SIZE, stride=2, padding=_PADDING),
            GDN(n),
            *_residual_stack(n, residual_blocks),
            # No activation on the latent: it stays an unconstrained
            # real-valued tensor for the quantizer to work on, exactly as
            # in the baseline encoder.
            nn.Conv2d(n, latent_channels, _KERNEL_SIZE, stride=2, padding=_PADDING),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"AnalysisTransform expects a [B, {self.in_channels}, H, W] tensor, "
                f"got shape {tuple(x.shape)}"
            )
        _, _, height, width = x.shape
        if height % DOWNSAMPLE_FACTOR != 0 or width % DOWNSAMPLE_FACTOR != 0:
            raise ValueError(
                f"AnalysisTransform input height and width must each be divisible by "
                f"{DOWNSAMPLE_FACTOR} (four stride-2 stages), got {height}x{width}"
            )
        return self.net(x)


class SynthesisTransform(nn.Module):
    """[B, latent_channels, H, W] -> [B, out_channels, H*16, W*16].

    Mirrors AnalysisTransform stage for stage, with IGDN where the analysis
    side has GDN and the residual stacks in the corresponding positions.
    """

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 192,
        base_channels: int = 192,
        residual_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        n = base_channels
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                latent_channels, n, _KERNEL_SIZE, stride=2, padding=_PADDING, output_padding=1
            ),
            IGDN(n),
            *_residual_stack(n, residual_blocks),
            nn.ConvTranspose2d(
                n, n, _KERNEL_SIZE, stride=2, padding=_PADDING, output_padding=1
            ),
            IGDN(n),
            *_residual_stack(n, residual_blocks),
            nn.ConvTranspose2d(
                n, n, _KERNEL_SIZE, stride=2, padding=_PADDING, output_padding=1
            ),
            IGDN(n),
            nn.ConvTranspose2d(
                n, out_channels, _KERNEL_SIZE, stride=2, padding=_PADDING, output_padding=1
            ),
            # See the module docstring: the [0, 1] output range is a
            # contract the rest of the pipeline is written against.
            nn.Sigmoid(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 4 or z.shape[1] != self.latent_channels:
            raise ValueError(
                f"SynthesisTransform expects a [B, {self.latent_channels}, H, W] tensor, "
                f"got shape {tuple(z.shape)}"
            )
        return self.net(z)


class ResidualGDNAutoencoder(nn.Module):
    """AnalysisTransform + SynthesisTransform behind BaselineAutoencoder's
    interface, so existing training, checkpoint and evaluation code can use
    it unchanged.

    The defaults (192 channels, 1 residual block per position) land at
    about 8.5M parameters, inside the 8-12M band Stage 1 asks for.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 192,
        base_channels: int = 192,
        residual_blocks: int = 1,
        *,
        quantization_noise: QuantizationNoise | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        self.residual_blocks = residual_blocks

        self.encoder = AnalysisTransform(
            in_channels, latent_channels, base_channels, residual_blocks
        )
        self.decoder = SynthesisTransform(
            in_channels, latent_channels, base_channels, residual_blocks
        )

        # Same contract as BaselineAutoencoder: a plain object, not an
        # nn.Module or Parameter, never captured by state_dict() or .to(),
        # never part of config_dict(), and only applied in training mode.
        self.quantization_noise = quantization_noise

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        if self.training and self.quantization_noise is not None:
            z = self.quantization_noise.apply(z)
        return self.decode(z)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config_dict(self) -> dict[str, Any]:
        """Architecture config sufficient to rebuild this exact model via
        ResidualGDNAutoencoder(**config_dict())."""
        return {
            "in_channels": self.in_channels,
            "latent_channels": self.latent_channels,
            "base_channels": self.base_channels,
            "residual_blocks": self.residual_blocks,
        }
