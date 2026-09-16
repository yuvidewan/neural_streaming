"""Neural Video Compression (nvc).

A learned video codec: a convolutional autoencoder, block motion compensation, a
causal channel-context entropy model and a from-scratch range coder, writing the
`.nvct` v2 container.

    import nvc
    data = nvc.encode(frames, "outputs/codec_bundles/nvc_m22_3bit.pt")
    frames_hat = nvc.decode(data, "outputs/codec_bundles/nvc_m22_3bit.pt")

`frames` is [N, 3, H, W] float in [0, 1] or [N, H, W, 3] uint8, with H and W
divisible by 16. A bundle (`nvc.video.CodecBundle`) holds everything one operating
point needs; `scripts/export_codec_bundle.py` produces them.

Sub-packages:
- video:       the video codec - motion, container, entropy model, bundles
- compression: quantization, static entropy tables, range coder, still-image .nvc
- models:      the autoencoder
- training:    training loops, rate estimation, checkpoints
- evaluation:  PSNR, MS-SSIM, FFmpeg H.264/H.265 harness
- data:        frame extraction and dataset utilities
- utils:       configuration, device and seeding helpers

Where the codec stands against H.264/H.265 is measured in
`outputs/benchmarks/parity_s0/` - see README.md.
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["decode", "encode"]


def _codec(bundle, device):
    # Imported lazily so `import nvc` stays cheap and torch-free until used.
    from nvc.video import CodecBundle, VideoCodec

    if isinstance(bundle, VideoCodec):
        return bundle
    if isinstance(bundle, CodecBundle):
        return VideoCodec(bundle, device=device)
    return VideoCodec.from_file(bundle, device=device)


def encode(frames, bundle, *, device=None) -> bytes:
    """Encode a frame sequence to `.nvct` v2 bytes.

    `bundle` is a path to a codec bundle, a loaded `CodecBundle`, or a
    `VideoCodec` (reuse one to avoid reloading weights on every call).
    """
    return _codec(bundle, device).encode(frames).data


def decode(data, bundle, *, device=None):
    """Decode `.nvct` v2 bytes (or a file path) to [N, 3, H, W] frames in [0, 1]."""
    return _codec(bundle, device).decode(data)
