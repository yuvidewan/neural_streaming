"""Compression quality and performance metrics, and the benchmark harness.

- basic_metrics:      MSE, PSNR
- perceptual_metrics: MS-SSIM (Milestone 7)
- latent_analysis:    encoder-latent extraction and descriptive statistics
- ffmpeg:             FFmpeg/ffprobe discovery and safe invocation
- sequences:          benchmark sequence discovery and validation
- codecs:             NVC / H.264 / H.265 backends behind one interface
- rd_benchmark:       orchestration, aggregation, and the result schema

Not implemented: VMAF/LPIPS, and any temporal (inter-frame) neural coding.
"""

from .basic_metrics import mse, psnr, psnr_from_mse
from .latent_analysis import extract_latents, latent_statistics
from .perceptual_metrics import MetricInputError, msssim

__all__ = [
    "mse",
    "psnr",
    "psnr_from_mse",
    "msssim",
    "MetricInputError",
    "extract_latents",
    "latent_statistics",
]
