"""Compression quality and performance metrics.

- basic_metrics:  MSE, PSNR (implemented)
- latent_analysis: encoder-latent extraction and descriptive statistics

Not implemented yet: MS-SSIM, bits-per-pixel (BPP), true compression ratio
(requires entropy coding), encoding/decoding time, and comparisons against
H.264/H.265.
"""

from .basic_metrics import mse, psnr, psnr_from_mse
from .latent_analysis import extract_latents, latent_statistics

__all__ = ["mse", "psnr", "psnr_from_mse", "extract_latents", "latent_statistics"]
