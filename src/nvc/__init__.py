"""Neural Video Compression (nvc) core package.

Sub-packages are added incrementally as functionality is implemented:
- data:        frame extraction and dataset utilities
- models:      encoder / decoder (VAE) architectures
- compression: quantization, entropy coding, .nvc serialization
- evaluation:  PSNR, MS-SSIM, and codec comparison metrics
- utils:       shared configuration and helper utilities

No compression, model, or streaming logic is implemented yet.
"""

__version__ = "0.0.1"
