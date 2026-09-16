"""The video codec: frames <-> `.nvct` v2 bytes.

    I-frame   z = E(x) -> intra grid -> per-channel tables -> range coder
    P-frame   motion  = block search (x_hat_prev -> x), coded with the motion tables
              z_ref   = E(Warp(x_hat_prev, decoded motion))
              symbols = Q_residual(E(x) - z_ref)
              coded by the causal context model: prototype routed through the
              assignment codebook, arithmetic-coded with the recalibrated tables
    both      x_hat = D(z_hat); the NEXT P-frame references x_hat, never x

The loop is closed: the encoder warps with the motion vectors the decoder will
read back and references its own reconstruction, so encoder and decoder hold
identical state at every frame. This is the loop `scripts/m21_refinement.py`
ran at its identity candidate, which M13, M14, M21 and M22 all measured, and
`scripts/verify_promoted_codec.py` proves this module writes the same bytes.

Frames are [N, 3, H, W] float tensors in [0, 1] (or [N, H, W, 3] uint8 arrays),
with H and W divisible by both the block size and the autoencoder's 16x
downsampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from nvc.compression.codec import (
    decode_payload_to_latent,
    encode_latent_to_payload,
    latent_to_symbols,
    symbols_to_latent,
)
from nvc.video.bundle import CodecBundle
from nvc.video.container import (
    FRAME_TYPE_I,
    TemporalFormatError,
    TemporalStreamHeader,
    TemporalStreamReader,
    gop_frame_types,
    stream_to_bytes,
)
from nvc.video.entropy import decode_residual_frame, encode_residual_frame
from nvc.video.motion import (
    decode_motion_payload,
    deterministic_kernels,
    encode_motion_payload,
    estimate_block_motion,
    motion_alphabet_bits,
    warp_blocks,
)

AUTOENCODER_STRIDE = 16


@dataclass(frozen=True)
class EncodedVideo:
    """The stream plus its byte accounting."""

    data: bytes
    frame_types: tuple[int, ...]
    motion_bytes: tuple[int, ...]
    residual_bytes: tuple[int, ...]
    width: int
    height: int

    @property
    def total_bytes(self) -> int:
        return len(self.data)

    @property
    def bits_per_pixel(self) -> float:
        return len(self.data) * 8 / (len(self.frame_types) * self.width * self.height)


def as_frame_tensor(frames) -> torch.Tensor:
    """Accept [N, 3, H, W] float in [0, 1] or [N, H, W, 3] uint8; return the former."""
    if isinstance(frames, np.ndarray):
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[3] != 3:
            raise ValueError("numpy frames must be uint8 with shape [N, H, W, 3]")
        return torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).contiguous().float() / 255.0
    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"frames must be a torch.Tensor or numpy array, got {type(frames).__name__}")
    if frames.dim() != 4 or frames.shape[1] != 3:
        raise ValueError(f"frames must be [N, 3, H, W], got {tuple(frames.shape)}")
    if not frames.is_floating_point():
        raise ValueError("tensor frames must be floating point in [0, 1]")
    if frames.numel() and (float(frames.min()) < 0.0 or float(frames.max()) > 1.0):
        raise ValueError("tensor frames must lie in [0, 1]")
    return frames


class VideoCodec:
    """Encode and decode video with one frozen codec bundle."""

    def __init__(self, bundle: CodecBundle, *, device: str | torch.device | None = None) -> None:
        self.bundle = bundle
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = bundle.autoencoder().to(self.device).eval()
        self.context_model = bundle.context_model().to(self.device).eval()
        self.intra_params = bundle.intra_params()
        self.residual_params = bundle.residual_params()
        self.intra_entropy_model = bundle.intra_entropy_model()
        self.motion_entropy_model = bundle.motion_entropy_model()
        self.assign_codebook = bundle.assign_codebook()
        self.coding_codebook = bundle.coding_codebook()
        self.residual_identity = bytes.fromhex(bundle.residual_entropy_model_id)
        from nvc.video.entropy import zero_symbols
        channels = int(self.residual_params.scale.numel())
        self.zero = torch.from_numpy(zero_symbols(self.residual_params, channels))

    @classmethod
    def from_file(cls, path: str | Path, *, device: str | torch.device | None = None) -> "VideoCodec":
        return cls(CodecBundle.load(path), device=device)

    # --- encoder -------------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, frames, *, return_reconstructions: bool = False):
        """Encode a sequence. Returns `EncodedVideo`, or `(EncodedVideo,
        reconstructions)` when `return_reconstructions` is set - the frames the
        decoder will produce, bit-identical."""
        frames = as_frame_tensor(frames)
        bundle = self.bundle
        count, _, height, width = frames.shape
        if count < 1:
            raise ValueError("cannot encode an empty sequence")
        for size, label in ((bundle.block_size, "block size"), (AUTOENCODER_STRIDE, "autoencoder stride")):
            if height % size or width % size:
                raise ValueError(f"frame size {width}x{height} must be divisible by the {label} ({size})")
        if width > 0xFFFF or height > 0xFFFF or count > 0xFFFFFFFF:
            raise ValueError("sequence exceeds the .nvct v2 header limits")

        model = self.autoencoder
        types = gop_frame_types(count, bundle.gop_size)
        latent_shape = tuple(model.encode(frames[0:1].to(self.device)).shape[1:])
        header = TemporalStreamHeader(
            gop_size=bundle.gop_size, quantization_bits=bundle.bits,
            quantization_mode=bundle.quantization_mode, image_width=width, image_height=height,
            image_channels=3, latent_channels=latent_shape[0], latent_height=latent_shape[1],
            latent_width=latent_shape[2], frame_count=count,
            num_intra_quantization_params=self.intra_params.scale.numel(),
            num_residual_quantization_params=self.residual_params.scale.numel(),
            block_size=bundle.block_size, search_range=bundle.search_range,
            motion_bits=self.motion_entropy_model.bits, reference_mode="mc",
            intra_entropy_model_id=self.intra_entropy_model.model_id(),
            residual_entropy_model_id=self.residual_identity,
            motion_entropy_model_id=self.motion_entropy_model.model_id())

        records: list[tuple[int, bytes, bytes]] = []
        reconstructions: list[torch.Tensor] = []
        previous = None
        with deterministic_kernels():
            for index in range(count):
                frame = frames[index:index + 1].to(self.device)
                latent = model.encode(frame)
                if types[index] == FRAME_TYPE_I:
                    payload, _ = encode_latent_to_payload(
                        latent, params=self.intra_params, entropy_model=self.intra_entropy_model)
                    decoded, _ = decode_payload_to_latent(
                        payload, entropy_model=self.intra_entropy_model, params=self.intra_params,
                        shape=latent_shape)
                    reconstructed_latent = decoded.to(self.device)
                    records.append((FRAME_TYPE_I, b"", payload))
                else:
                    motion = estimate_block_motion(previous, frame, block_size=bundle.block_size,
                                                   search_range=bundle.search_range)
                    motion_payload = encode_motion_payload(
                        motion, search_range=bundle.search_range,
                        entropy_model=self.motion_entropy_model)
                    # Warp with the vectors the decoder will read back, not the estimate.
                    decoded_motion = decode_motion_payload(
                        motion_payload, (height // bundle.block_size, width // bundle.block_size),
                        search_range=bundle.search_range, entropy_model=self.motion_entropy_model)
                    warped = warp_blocks(previous, decoded_motion, block_size=bundle.block_size)
                    reference_latent = model.encode(warped)
                    symbols = latent_to_symbols(latent - reference_latent, self.residual_params)
                    payload = encode_residual_frame(
                        self.context_model, self.assign_codebook, self.coding_codebook,
                        reference_latent, symbols.reshape(latent_shape), self.zero)
                    records.append((types[index], motion_payload, payload))
                    reconstructed_latent = reference_latent + symbols_to_latent(
                        symbols, latent_shape, self.residual_params).to(self.device)
                reconstruction = model.decode(reconstructed_latent)
                previous = reconstruction
                if return_reconstructions:
                    reconstructions.append(reconstruction.detach().cpu())

        data = stream_to_bytes(header, self.intra_params, self.residual_params, records)
        encoded = EncodedVideo(data=data, frame_types=tuple(types),
                               motion_bytes=tuple(len(m) for _, m, _ in records),
                               residual_bytes=tuple(len(r) for _, _, r in records),
                               width=width, height=height)
        if return_reconstructions:
            return encoded, torch.cat(reconstructions, dim=0)
        return encoded

    # --- decoder -------------------------------------------------------------------------

    def _check_stream(self, header: TemporalStreamHeader) -> None:
        """Refuse a stream this bundle cannot decode, before any payload is read."""
        bundle = self.bundle
        expected = {
            "quantization_bits": bundle.bits,
            "quantization_mode": bundle.quantization_mode,
            "image_channels": 3,
            "latent_channels": int(bundle.context_model_config["latent_channels"]),
            "block_size": bundle.block_size,
            "search_range": bundle.search_range,
            "motion_bits": motion_alphabet_bits(bundle.search_range),
            "reference_mode": "mc",
        }
        wrong = {k: (getattr(header, k), v) for k, v in expected.items() if getattr(header, k) != v}
        if wrong:
            raise TemporalFormatError(
                "stream does not match this codec bundle: " +
                ", ".join(f"{k} is {got!r}, bundle has {want!r}" for k, (got, want) in wrong.items()))
        width, height = header.image_width, header.image_height
        for size in (bundle.block_size, AUTOENCODER_STRIDE):
            if width == 0 or height == 0 or width % size or height % size:
                raise TemporalFormatError(f"stream frame size {width}x{height} is not decodable")
        if (header.latent_height, header.latent_width) != (height // AUTOENCODER_STRIDE,
                                                           width // AUTOENCODER_STRIDE):
            raise TemporalFormatError("stream latent size does not match its frame size")
        for label, declared, actual in (
                ("intra", header.intra_entropy_model_id, self.intra_entropy_model.model_id()),
                ("residual", header.residual_entropy_model_id, self.residual_identity),
                ("motion", header.motion_entropy_model_id, self.motion_entropy_model.model_id())):
            if declared != actual:
                raise TemporalFormatError(
                    f"{label} entropy model mismatch: stream declares {declared.hex()}, "
                    f"this bundle has {actual.hex()}")

    @torch.no_grad()
    def decode(self, data: bytes | bytearray | memoryview | str | Path) -> torch.Tensor:
        """Decode a `.nvct` v2 stream (bytes or a file path) to [N, 3, H, W] frames."""
        reader = TemporalStreamReader(data)
        header = reader.header
        self._check_stream(header)
        model = self.autoencoder
        latent_shape = header.latent_shape
        reconstructions: list[torch.Tensor] = []
        previous = None
        with deterministic_kernels():
            for frame_type, motion_payload, residual_payload in reader:
                if frame_type == FRAME_TYPE_I:
                    latent, _ = decode_payload_to_latent(
                        residual_payload, entropy_model=self.intra_entropy_model,
                        params=reader.intra_params, shape=latent_shape)
                    reconstructed_latent = latent.to(self.device)
                else:
                    motion = decode_motion_payload(
                        motion_payload, header.block_grid, search_range=header.search_range,
                        entropy_model=self.motion_entropy_model)
                    warped = warp_blocks(previous, motion, block_size=header.block_size)
                    reference_latent = model.encode(warped)
                    symbols = decode_residual_frame(
                        self.context_model, self.assign_codebook, self.coding_codebook,
                        residual_payload, reference_latent, self.zero, shape=latent_shape)
                    reconstructed_latent = reference_latent + symbols_to_latent(
                        symbols, latent_shape, reader.residual_params).to(self.device)
                reconstruction = model.decode(reconstructed_latent)
                reconstructions.append(reconstruction.detach().cpu())
                previous = reconstruction
        return torch.cat(reconstructions, dim=0)
