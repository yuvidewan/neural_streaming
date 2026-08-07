"""Encoder / decoder network architectures.

- encoder:      Encoder, a strided-convolution CNN downsampler
- decoder:      Decoder, a transposed-convolution CNN upsampler
- autoencoder:  BaselineAutoencoder, a deterministic (non-variational)
                autoencoder combining the two behind an explicit
                encode/decode interface

This is a plain reconstruction baseline - no VAE, quantization, or entropy
coding here yet.
"""

from .autoencoder import BaselineAutoencoder
from .decoder import Decoder
from .encoder import Encoder

__all__ = ["Encoder", "Decoder", "BaselineAutoencoder"]
