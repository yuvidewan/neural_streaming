"""Encoder / decoder network architectures.

The baseline transform (Milestones 1-22):

- encoder:      Encoder, a strided-convolution CNN downsampler
- decoder:      Decoder, a transposed-convolution CNN upsampler
- autoencoder:  BaselineAutoencoder, a deterministic (non-variational)
                autoencoder combining the two behind an explicit
                encode/decode interface

The Stage 1 transform (PARITY_ROADMAP.md), which exists because the
baseline is what the Stage 0 scoreboard identified as the binding
constraint on BD-rate:

- gdn:                 GDN / IGDN, the divisive-normalization activation
- residual_transform:  AnalysisTransform, SynthesisTransform,
                       ResidualBlock and ResidualGDNAutoencoder - the same
                       stride-16 structure with GDN, residual blocks and
                       ~8.5M parameters

Neither module does quantization or entropy coding; those live in
nvc.compression, and the video codec is in nvc.video.
"""

from .autoencoder import BaselineAutoencoder
from .decoder import Decoder
from .encoder import Encoder
from .gdn import GDN, IGDN
from .residual_transform import (
    AnalysisTransform,
    ResidualBlock,
    ResidualGDNAutoencoder,
    SynthesisTransform,
)

__all__ = [
    "Encoder",
    "Decoder",
    "BaselineAutoencoder",
    "GDN",
    "IGDN",
    "AnalysisTransform",
    "SynthesisTransform",
    "ResidualBlock",
    "ResidualGDNAutoencoder",
]
