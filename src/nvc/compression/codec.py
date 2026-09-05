"""End-to-end .nvc codec: frame <-> compressed bitstream.

    encode:  frame -> encoder -> latent -> fixed quantizer -> symbols
                   -> arithmetic coder -> .nvc bytes

    decode:  .nvc bytes -> arithmetic decoder -> symbols -> dequantizer
                        -> approximate latent -> decoder -> frame

Only the entropy-coding stage is new relative to Milestone 5, and it is
exactly lossless: the symbols that come out of `decode` are bit-identical to
the ones that went into `encode`, so the reconstructed frame is identical to
what Milestone 5's quantize-then-decode path produced. All distortion comes
from quantization, none from entropy coding - `verify_lossless_roundtrip`
asserts precisely that.

SYMBOL ORDER
------------
Symbols are serialized in channel-major (C, H, W) raster order. That order
is implicit and never stored: the decoder reconstructs it from the latent
dimensions in the header, which is also what lets each symbol be coded
against its own channel's frequency table with no side information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.compression.nvc_format import (
    NVCFormatError,
    NVCHeader,
    NVCReader,
    NVCStreamHeader,
    NVCWriter,
)
from nvc.compression.quantization import QuantizationParams, UniformQuantizer
from nvc.compression.range_coder import decode_symbols, encode_symbols


@dataclass(frozen=True)
class EncodeResult:
    data: bytes
    header: NVCHeader
    symbols: np.ndarray
    payload_bits: int
    header_bits: int

    @property
    def total_bits(self) -> int:
        return self.payload_bits + self.header_bits

    @property
    def total_bytes(self) -> int:
        return len(self.data)

    def bits_per_pixel(self, *, payload_only: bool = False) -> float:
        pixels = self.header.image_width * self.header.image_height
        return (self.payload_bits if payload_only else self.total_bits) / pixels

    def bits_per_symbol(self, *, payload_only: bool = True) -> float:
        bits = self.payload_bits if payload_only else self.total_bits
        return bits / self.header.symbol_count


def channel_table_index(latent_channels: int, latent_height: int, latent_width: int) -> np.ndarray:
    """Frequency-table index for every symbol position, in C-major order."""
    return np.repeat(np.arange(latent_channels, dtype=np.int64), latent_height * latent_width)


def latent_to_symbols(latent: torch.Tensor, params: QuantizationParams) -> np.ndarray:
    """Quantize a single-frame latent to a flat symbol array (C-major)."""
    if latent.shape[0] != 1:
        raise ValueError(
            f"Expected a single-frame latent with batch size 1, got {tuple(latent.shape)}"
        )
    quantizer = UniformQuantizer(params.bits, params.mode)
    quantized, _ = quantizer.quantize(latent, params.to(latent.device))
    return quantized[0].reshape(-1).cpu().numpy().astype(np.int64)


def symbols_to_latent(
    symbols: np.ndarray, shape: tuple[int, int, int], params: QuantizationParams
) -> torch.Tensor:
    """Inverse of `latent_to_symbols`: symbols -> approximate latent [1, C, H, W]."""
    quantizer = UniformQuantizer(params.bits, params.mode)
    quantized = torch.from_numpy(np.asarray(symbols, dtype=np.int64)).reshape(1, *shape)
    return quantizer.dequantize(quantized, params)


def encode_latent_to_payload(
    latent: torch.Tensor, *, params: QuantizationParams, entropy_model: EmpiricalEntropyModel,
) -> tuple[bytes, np.ndarray]:
    """Quantize and entropy-code one latent into raw payload bytes - no
    header. This is the core `encode_latent` and the `.nvcs` stream format
    both share: a stream's per-frame payload is byte-for-byte identical to
    a single-frame `.nvc`'s payload, only the header differs (once per
    stream instead of once per frame - see nvc_format.py).

    Returns (payload, symbols): symbols are returned alongside the payload
    so a caller building an `EncodeResult` (or verifying a lossless round
    trip) never has to quantize the same latent a second time to get them.
    """
    if entropy_model.bits != params.bits:
        raise ValueError(
            f"Entropy model is built for {entropy_model.bits}-bit symbols but the "
            f"quantization parameters are {params.bits}-bit"
        )

    latent_channels, latent_height, latent_width = latent.shape[1:]
    if entropy_model.num_tables != latent_channels:
        raise ValueError(
            f"Entropy model has {entropy_model.num_tables} tables but the latent has "
            f"{latent_channels} channels"
        )

    symbols = latent_to_symbols(latent, params)
    table_index = channel_table_index(latent_channels, latent_height, latent_width)
    payload = encode_symbols(symbols, entropy_model.cumulative, table_index)
    return payload, symbols


def encode_latent(
    latent: torch.Tensor,
    *,
    params: QuantizationParams,
    entropy_model: EmpiricalEntropyModel,
    image_shape: tuple[int, int, int],
) -> EncodeResult:
    """Quantize and entropy-code one latent into .nvc bytes.

    `image_shape` is the source frame's [C, H, W], recorded so the decoder
    can report and validate the original dimensions.
    """
    latent_channels, latent_height, latent_width = latent.shape[1:]
    payload, symbols = encode_latent_to_payload(latent, params=params, entropy_model=entropy_model)

    header = NVCHeader(
        quantization_bits=params.bits,
        quantization_mode=params.mode,
        image_width=image_shape[2],
        image_height=image_shape[1],
        image_channels=image_shape[0],
        latent_channels=latent_channels,
        latent_height=latent_height,
        latent_width=latent_width,
        symbol_count=len(symbols),
        payload_length=len(payload),
        entropy_model_id=entropy_model.model_id(),
        scales=tuple(params.scale.flatten().tolist()),
        zero_points=tuple(params.zero_point.flatten().tolist()),
    )
    data = NVCWriter.to_bytes(header, payload)

    return EncodeResult(
        data=data,
        header=header,
        symbols=symbols,
        payload_bits=len(payload) * 8,
        header_bits=header.header_size * 8,
    )


def decode_payload_to_latent(
    payload: bytes,
    *,
    entropy_model: EmpiricalEntropyModel,
    params: QuantizationParams,
    shape: tuple[int, int, int],
) -> tuple[torch.Tensor, np.ndarray]:
    """Inverse of `encode_latent_to_payload`: raw payload bytes -> (latent,
    symbols), given the quantization params and latent shape out-of-band
    (from a header - once-per-frame via `NVCHeader`, once-per-stream via
    `NVCStreamHeader`). The core `decode_latent` and the `.nvcs` stream
    format both share.
    """
    if entropy_model.bits != params.bits:
        raise ValueError(
            f"Entropy model is {entropy_model.bits}-bit but params are {params.bits}-bit"
        )
    latent_channels, latent_height, latent_width = shape
    table_index = channel_table_index(latent_channels, latent_height, latent_width)
    symbol_count = latent_channels * latent_height * latent_width
    symbols = decode_symbols(payload, symbol_count, entropy_model.cumulative, table_index)
    latent = symbols_to_latent(symbols, shape, params)
    return latent, symbols


def decode_latent(
    data: bytes, *, entropy_model: EmpiricalEntropyModel
) -> tuple[torch.Tensor, NVCHeader, np.ndarray]:
    """Parse .nvc bytes back into an approximate latent.

    Returns (latent, header, symbols). Quantization parameters come from the
    file itself; the entropy model must be supplied and is verified against
    the id recorded at encode time.
    """
    header, payload = NVCReader.from_bytes(data)

    if entropy_model.model_id() != header.entropy_model_id:
        raise NVCFormatError(
            "Entropy model mismatch: this file was encoded with model id "
            f"{header.entropy_model_id.hex()} but the supplied model is "
            f"{entropy_model.model_id().hex()}. Decoding would produce garbage."
        )
    if entropy_model.bits != header.quantization_bits:
        raise NVCFormatError(
            f"Entropy model is {entropy_model.bits}-bit but the file declares "
            f"{header.quantization_bits}-bit symbols"
        )

    params = QuantizationParams.from_dict({
        "bits": header.quantization_bits,
        "mode": header.quantization_mode,
        "scale": list(header.scales),
        "zero_point": list(header.zero_points),
    })
    latent, symbols = decode_payload_to_latent(
        payload,
        entropy_model=entropy_model,
        params=params,
        shape=(header.latent_channels, header.latent_height, header.latent_width),
    )
    return latent, header, symbols


@torch.no_grad()
def encode_frame(
    model: torch.nn.Module,
    frame: torch.Tensor,
    *,
    params: QuantizationParams,
    entropy_model: EmpiricalEntropyModel,
) -> EncodeResult:
    """Full encode: a [3, H, W] or [1, 3, H, W] frame in [0, 1] -> .nvc bytes."""
    if frame.dim() == 3:
        frame = frame.unsqueeze(0)
    if frame.dim() != 4 or frame.shape[0] != 1:
        raise ValueError(
            f"encode_frame expects one [3, H, W] or [1, 3, H, W] frame, got {tuple(frame.shape)}"
        )

    model.eval()
    latent = model.encode(frame)
    return encode_latent(
        latent,
        params=params,
        entropy_model=entropy_model,
        image_shape=tuple(frame.shape[1:]),
    )


@torch.no_grad()
def decode_frame(
    model: torch.nn.Module, data: bytes, *, entropy_model: EmpiricalEntropyModel
) -> tuple[torch.Tensor, NVCHeader]:
    """Full decode: .nvc bytes -> reconstructed [1, 3, H, W] frame in [0, 1]."""
    model.eval()
    device = next(model.parameters()).device
    latent, header, _ = decode_latent(data, entropy_model=entropy_model)
    reconstruction = model.decode(latent.to(device))

    if tuple(reconstruction.shape[1:]) != (
        header.image_channels, header.image_height, header.image_width
    ):
        raise NVCFormatError(
            f"Decoded frame shape {tuple(reconstruction.shape[1:])} does not match the "
            f"dimensions recorded in the header "
            f"({header.image_channels}, {header.image_height}, {header.image_width})"
        )
    return reconstruction, header


def build_stream_header(
    *,
    params: QuantizationParams,
    entropy_model: EmpiricalEntropyModel,
    image_shape: tuple[int, int, int],
    latent_shape: tuple[int, int, int],
    frame_count: int,
) -> NVCStreamHeader:
    """Build the once-per-stream header for `NVCStreamWriter` - the stream
    analog of the header `encode_latent` builds internally for a single
    frame. Every frame in the stream must share this exact
    model/calibration/shape (see nvc_format.py's "STREAM CONTAINER").
    """
    return NVCStreamHeader(
        quantization_bits=params.bits,
        quantization_mode=params.mode,
        image_width=image_shape[2],
        image_height=image_shape[1],
        image_channels=image_shape[0],
        latent_channels=latent_shape[0],
        latent_height=latent_shape[1],
        latent_width=latent_shape[2],
        frame_count=frame_count,
        entropy_model_id=entropy_model.model_id(),
        scales=tuple(params.scale.flatten().tolist()),
        zero_points=tuple(params.zero_point.flatten().tolist()),
    )


@torch.no_grad()
def encode_frame_payload(
    model: torch.nn.Module,
    frame: torch.Tensor,
    *,
    params: QuantizationParams,
    entropy_model: EmpiricalEntropyModel,
) -> bytes:
    """Full encode, stream variant: a [3, H, W] or [1, 3, H, W] frame in
    [0, 1] -> raw payload bytes, for `NVCStreamWriter.append_frame`. Shares
    `encode_frame`'s exact model-forward + quantize + entropy-code steps
    (via `encode_latent_to_payload`); only the header-per-frame step is
    skipped, since a stream writes that once via `build_stream_header`.
    """
    if frame.dim() == 3:
        frame = frame.unsqueeze(0)
    if frame.dim() != 4 or frame.shape[0] != 1:
        raise ValueError(
            f"encode_frame_payload expects one [3, H, W] or [1, 3, H, W] frame, "
            f"got {tuple(frame.shape)}"
        )

    model.eval()
    latent = model.encode(frame)
    payload, _ = encode_latent_to_payload(latent, params=params, entropy_model=entropy_model)
    return payload


@torch.no_grad()
def decode_frame_payload(
    model: torch.nn.Module,
    payload: bytes,
    *,
    params: QuantizationParams,
    entropy_model: EmpiricalEntropyModel,
    image_shape: tuple[int, int, int],
    latent_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Inverse of `encode_frame_payload`: one stream frame's payload bytes
    -> reconstructed [1, 3, H, W] frame in [0, 1]. `image_shape`/
    `latent_shape` come from the stream's header (`NVCStreamReader.header`),
    shared by every frame in the stream rather than re-read per frame.
    """
    model.eval()
    device = next(model.parameters()).device
    latent, _ = decode_payload_to_latent(
        payload, entropy_model=entropy_model, params=params, shape=latent_shape,
    )
    reconstruction = model.decode(latent.to(device))

    if tuple(reconstruction.shape[1:]) != image_shape:
        raise NVCFormatError(
            f"Decoded frame shape {tuple(reconstruction.shape[1:])} does not match the "
            f"stream's declared dimensions {image_shape}"
        )
    return reconstruction


def verify_lossless_roundtrip(encoded: EncodeResult, decoded_symbols: np.ndarray) -> None:
    """Assert entropy coding introduced no error at all.

    Quantization is lossy by design; entropy coding must not add a single
    bit of additional distortion on top of it.
    """
    if not np.array_equal(encoded.symbols, decoded_symbols):
        mismatches = int(np.sum(encoded.symbols != decoded_symbols))
        raise NVCFormatError(
            f"Entropy coding was not lossless: {mismatches} of "
            f"{len(encoded.symbols)} symbols differ after the round trip"
        )
