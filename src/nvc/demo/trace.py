"""The encoder loop with its intermediates kept, for display.

`VideoCodec.encode` returns the stream and per-frame byte counts and discards
everything in between. A demo needs the in-between: the motion field, the warped
prediction, how much residual each latent position carried. `trace_encode` runs
the same loop with the same functions in the same order and keeps those.

It does NOT go through `VideoCodec.encode`, so the two could drift apart. The
test suite pins them together: `trace_encode(...).encoded.data` must equal
`codec.encode(...).data` byte for byte, and `process_clip` additionally decodes
every traced stream with the real decoder and compares the frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from nvc.compression.codec import (
    decode_payload_to_latent,
    encode_latent_to_payload,
    latent_to_symbols,
    symbols_to_latent,
)
from nvc.video.codec import AUTOENCODER_STRIDE, EncodedVideo, VideoCodec, as_frame_tensor
from nvc.video.container import (
    FRAME_TYPE_I,
    TemporalStreamHeader,
    gop_frame_types,
    stream_to_bytes,
)
from nvc.video.entropy import encode_residual_frame
from nvc.video.motion import (
    decode_motion_payload,
    deterministic_kernels,
    encode_motion_payload,
    estimate_block_motion,
    warp_blocks,
)


@dataclass(frozen=True)
class FrameTrace:
    """What the encoder did with one frame.

    `motion`, `prediction` and `activity` are None for an I-frame, which is coded
    on its own. `motion` is [2, blocks_y, blocks_x] of (dy, dx): the block at
    (row, column) was fetched from (row + dy, column + dx) in the previous
    reconstruction. `activity` is [latent_h, latent_w]: the mean distance of the
    residual symbols from the zero symbol at that position, i.e. how much the
    prediction missed by, in quantizer steps.
    """

    index: int
    frame_type: int
    motion_bytes: int
    residual_bytes: int
    reconstruction: torch.Tensor
    motion: np.ndarray | None = None
    prediction: torch.Tensor | None = None
    activity: np.ndarray | None = None


@dataclass(frozen=True)
class EncodeTrace:
    encoded: EncodedVideo
    frames: tuple[FrameTrace, ...]

    @property
    def reconstructions(self) -> torch.Tensor:
        return torch.stack([frame.reconstruction for frame in self.frames], dim=0)


@torch.no_grad()
def trace_encode(codec: VideoCodec, frames, *,
                 progress: Callable[[int, int], None] | None = None) -> EncodeTrace:
    """Encode `frames` exactly as `codec.encode` does, keeping each frame's trace.

    `progress(done, total)` is called after every frame.
    """
    frames = as_frame_tensor(frames)
    bundle = codec.bundle
    count, _, height, width = frames.shape
    if count < 1:
        raise ValueError("cannot encode an empty sequence")
    for size, label in ((bundle.block_size, "block size"), (AUTOENCODER_STRIDE, "autoencoder stride")):
        if height % size or width % size:
            raise ValueError(f"frame size {width}x{height} must be divisible by the {label} ({size})")

    model = codec.autoencoder
    types = gop_frame_types(count, bundle.gop_size)
    latent_shape = tuple(model.encode(frames[0:1].to(codec.device)).shape[1:])
    header = TemporalStreamHeader(
        gop_size=bundle.gop_size, quantization_bits=bundle.bits,
        quantization_mode=bundle.quantization_mode, image_width=width, image_height=height,
        image_channels=3, latent_channels=latent_shape[0], latent_height=latent_shape[1],
        latent_width=latent_shape[2], frame_count=count,
        num_intra_quantization_params=codec.intra_params.scale.numel(),
        num_residual_quantization_params=codec.residual_params.scale.numel(),
        block_size=bundle.block_size, search_range=bundle.search_range,
        motion_bits=codec.motion_entropy_model.bits, reference_mode="mc",
        intra_entropy_model_id=codec.intra_entropy_model.model_id(),
        residual_entropy_model_id=codec.residual_identity,
        motion_entropy_model_id=codec.motion_entropy_model.model_id())

    zero = codec.zero.numpy().reshape(-1, 1, 1)
    records: list[tuple[int, bytes, bytes]] = []
    traces: list[FrameTrace] = []
    previous = None
    with deterministic_kernels():
        for index in range(count):
            frame = frames[index:index + 1].to(codec.device)
            latent = model.encode(frame)
            motion_field = prediction = activity = None
            if types[index] == FRAME_TYPE_I:
                payload, _ = encode_latent_to_payload(
                    latent, params=codec.intra_params, entropy_model=codec.intra_entropy_model)
                decoded, _ = decode_payload_to_latent(
                    payload, entropy_model=codec.intra_entropy_model, params=codec.intra_params,
                    shape=latent_shape)
                reconstructed_latent = decoded.to(codec.device)
                motion_payload = b""
            else:
                motion = estimate_block_motion(previous, frame, block_size=bundle.block_size,
                                               search_range=bundle.search_range)
                motion_payload = encode_motion_payload(
                    motion, search_range=bundle.search_range,
                    entropy_model=codec.motion_entropy_model)
                decoded_motion = decode_motion_payload(
                    motion_payload, (height // bundle.block_size, width // bundle.block_size),
                    search_range=bundle.search_range, entropy_model=codec.motion_entropy_model)
                warped = warp_blocks(previous, decoded_motion, block_size=bundle.block_size)
                reference_latent = model.encode(warped)
                symbols = latent_to_symbols(latent - reference_latent, codec.residual_params)
                payload = encode_residual_frame(
                    codec.context_model, codec.assign_codebook, codec.coding_codebook,
                    reference_latent, symbols.reshape(latent_shape), codec.zero)
                reconstructed_latent = reference_latent + symbols_to_latent(
                    symbols, latent_shape, codec.residual_params).to(codec.device)
                motion_field = decoded_motion.cpu().numpy().astype(np.int64)
                prediction = warped[0].detach().cpu()
                activity = np.abs(symbols.reshape(latent_shape) - zero).mean(axis=0)
            records.append((types[index], motion_payload, payload))
            reconstruction = model.decode(reconstructed_latent)
            previous = reconstruction
            traces.append(FrameTrace(
                index=index, frame_type=int(types[index]), motion_bytes=len(motion_payload),
                residual_bytes=len(payload), reconstruction=reconstruction[0].detach().cpu(),
                motion=motion_field, prediction=prediction, activity=activity))
            if progress is not None:
                progress(index + 1, count)

    data = stream_to_bytes(header, codec.intra_params, codec.residual_params, records)
    encoded = EncodedVideo(data=data, frame_types=tuple(types),
                           motion_bytes=tuple(len(m) for _, m, _ in records),
                           residual_bytes=tuple(len(r) for _, _, r in records),
                           width=width, height=height)
    return EncodeTrace(encoded=encoded, frames=tuple(traces))
