"""M10G Part C: a causal temporal coding baseline - I-frames and P-frames.

THE QUESTION THIS PREPARES FOR
-------------------------------
NVC is intra-only: every frame is coded independently, so it pays full price for
content that barely moved. Classical codecs exploit that redundancy through
inter-frame prediction; NVC currently does not. Before building a learned
temporal model, this milestone establishes the minimum scientifically useful
temporal path and asks what it actually buys.

This file is deliberately NOT a motion-compensated video codec. It prioritises
correctness, decoder symmetry, causality, reproducibility and honest byte
accounting over compression, exactly as M10G specifies.

DESIGN DECISION 1 - RESIDUALS LIVE IN THE LATENT DOMAIN, NOT PIXEL SPACE
-------------------------------------------------------------------------
The obvious formulation, `residual = x_t - x_hat_{t-1}` in pixel space, does not
fit this architecture. `Decoder` ends in `nn.Sigmoid()`, so a reconstruction is
bounded to [0, 1] - but a pixel residual lies in [-1, +1]. The decoder literally
cannot emit a signed residual, and forcing it to would need either an affine
remap (halving effective precision and mismatching the model's training
distribution) or a new output head (a trained architecture change, out of scope
here).

The latent has no such problem. `Encoder` ends on a bare `Conv2d` with a comment
that the latent "stays an unconstrained real-valued tensor" - so latent
differences are naturally signed, and the existing percentile calibration,
uniform quantizer and arithmetic coder handle them unchanged. Nothing in
`src/nvc/` is modified, and no new activation or head is invented.

So:

    I-frame:  z_t = E(x_t)                      -> quantize/code with INTRA grid
    P-frame:  dz  = z_t - z_ref                 -> quantize/code with RESIDUAL grid

with two separately calibrated grids, because intra latents and latent residuals
have completely different statistics (residuals concentrate near zero).

DESIGN DECISION 2 - THE REFERENCE IS A DECODED QUANTITY
--------------------------------------------------------
`z_ref` is the previously DECODED latent, `z_hat_{t-1}` - never the original
frame's latent. The encoder runs closed-loop: it dequantizes its own symbols and
predicts from that, so encoder and decoder hold bit-identical reference state by
construction rather than by hope.

Two reference modes are provided because it is a genuine design question:

    latent   (default) z_ref = z_hat_{t-1}, the dequantized latent directly.
    reencode           z_ref = E(D(z_hat_{t-1})), re-encoding the reconstructed
                       frame. Closer to the brief's "previous reconstructed
                       frame" wording and to a classical codec's reference
                       picture; costs an extra decode+encode per frame.

Both are strictly causal and both are symmetric. `latent` is the default because
it is cheaper and involves no extra lossy round trip.

DESIGN DECISION 3 - A SEPARATE PROTOTYPE CONTAINER (.nvct), NOT A CHANGE TO .nvc
---------------------------------------------------------------------------------
`.nvc` (single frame) and `.nvcs` (stream) are untouched and remain bit-exact.
`.nvcs` v1 cannot express this stream: it carries exactly ONE quantization
parameter block for the whole file and has no per-frame type field, whereas a
temporal stream needs two grids (intra + residual) and must mark every frame as
I or P. Extending `.nvcs` in place would change shipped format semantics, which
M10G forbids.

`.nvct` therefore follows `.nvcs`'s conventions deliberately - same magic-plus-
version prologue, same length-prefixed frame records, same truncation and
trailing-data strictness - so it is the existing direction continued in a
prototype, not a competing style. It lives here in `scripts/` rather than in
`src/nvc/` because it is a prototype: promoting it into the shipped package is a
later decision, once a learned temporal model justifies it.

CAUSALITY
----------
For frame t the encoder may use x_t and previously RECONSTRUCTED frames only.
Never a future frame, never an original past frame the decoder does not have.
Every sequence starts with an I-frame and reference state resets at sequence
boundaries, so no reference ever crosses from one DAVIS sequence into another.

Example usage (PowerShell, from the project root, with .venv activated):

    .venv\\Scripts\\python.exe scripts\\m10g_temporal_baseline.py --stage smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.codec import (
    decode_payload_to_latent,
    encode_latent_to_payload,
    latent_to_symbols,
)
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.compression.quantization import QuantizationParams
from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device

DEFAULT_OUTPUT_DIR = Path("outputs/m10g_temporal_baseline")
DEFAULT_GOP = 10
FRAME_TYPE_I = 0
FRAME_TYPE_P = 1
FRAME_TYPE_NAMES = {FRAME_TYPE_I: "I", FRAME_TYPE_P: "P"}

TEMPORAL_MAGIC = b"NVCT"
TEMPORAL_FORMAT_VERSION = 1
# magic, version, gop, bits, mode, coder, w, h, channels, lc, lh, lw,
# frame_count, n_intra_params, n_residual_params, intra_id, residual_id
_HEADER_STRUCT = struct.Struct("<4sBBBBBHHBHHHIHH8s8s")
_FRAME_RECORD_STRUCT = struct.Struct("<BI")  # frame_type, payload_length
# Module-level rather than a bare class attribute inside the dataclass: an
# un-annotated attribute in a @dataclass body makes dataclasses resolve the
# defining module, which fails when the module is loaded via importlib
# without being registered in sys.modules - exactly how the test suite
# loads every script.
TEMPORAL_HEADER_SIZE = _HEADER_STRUCT.size


class TemporalFormatError(Exception):
    """Raised on a malformed, truncated or unsupported .nvct stream."""


class CausalityViolationError(RuntimeError):
    """Raised when the coder is asked to use information a decoder cannot have."""


class TemporalStreamHeader:
    """.nvct version 1 fixed header - 44 bytes, then two quantization blocks.

    Layout (little-endian), mirroring `.nvcs`'s prologue conventions:

        0   4  char[]  magic, always b"NVCT"
        4   1  uint8   format_version (1)
        5   1  uint8   gop_size
        6   1  uint8   quantization_bits
        7   1  uint8   quantization_mode: 0=global, 1=per_channel
        8   1  uint8   entropy_coder_id (1 = static_arithmetic_v1)
        9   2  uint16  image_width
        11  2  uint16  image_height
        13  1  uint8   image_channels
        14  2  uint16  latent_channels
        16  2  uint16  latent_height
        18  2  uint16  latent_width
        20  4  uint32  frame_count
        24  2  uint16  num_intra_quantization_params
        26  2  uint16  num_residual_quantization_params
        28  8  bytes   intra_entropy_model_id
        36  8  bytes   residual_entropy_model_id

    Then the INTRA quantization block, then the RESIDUAL quantization block,
    each `num_* * 8` bytes of repeated { float32 scale; float32 zero_point }.

    Then `frame_count` repetitions of:
        uint8   frame_type (0 = I, 1 = P)
        uint32  payload_length
        bytes   payload

    Frame dependency is explicit and needs no extra field: an I-frame resets
    reference state, a P-frame references the immediately preceding decoded
    frame in the stream. A stream is therefore decodable from the header and
    the type bytes alone.
    """
    # A plain class rather than @dataclass on purpose: an un-annotated or
    # string-annotated field in a dataclass makes `dataclasses` resolve the
    # defining module through sys.modules, which fails when the module is
    # loaded via importlib without being registered there - exactly how this
    # project's test suite loads every script.
    def __init__(self, *, gop_size: int, quantization_bits: int,
                 quantization_mode: str, image_width: int, image_height: int,
                 image_channels: int, latent_channels: int, latent_height: int,
                 latent_width: int, frame_count: int,
                 num_intra_quantization_params: int,
                 num_residual_quantization_params: int,
                 intra_entropy_model_id: bytes, residual_entropy_model_id: bytes,
                 entropy_coder_id: int = 1,
                 format_version: int = TEMPORAL_FORMAT_VERSION) -> None:
        self.gop_size = gop_size
        self.quantization_bits = quantization_bits
        self.quantization_mode = quantization_mode
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels
        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.frame_count = frame_count
        self.num_intra_quantization_params = num_intra_quantization_params
        self.num_residual_quantization_params = num_residual_quantization_params
        self.intra_entropy_model_id = intra_entropy_model_id
        self.residual_entropy_model_id = residual_entropy_model_id
        self.entropy_coder_id = entropy_coder_id
        self.format_version = format_version

    @property
    def header_size(self) -> int:
        return TEMPORAL_HEADER_SIZE

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return (self.image_channels, self.image_height, self.image_width)

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (self.latent_channels, self.latent_height, self.latent_width)

    def pack(self) -> bytes:
        mode_code = 1 if self.quantization_mode == "per_channel" else 0
        return _HEADER_STRUCT.pack(
            TEMPORAL_MAGIC, self.format_version, self.gop_size, self.quantization_bits,
            mode_code, self.entropy_coder_id, self.image_width, self.image_height,
            self.image_channels, self.latent_channels, self.latent_height, self.latent_width,
            self.frame_count, self.num_intra_quantization_params,
            self.num_residual_quantization_params,
            self.intra_entropy_model_id[:8].ljust(8, b"\0"),
            self.residual_entropy_model_id[:8].ljust(8, b"\0"),
        )

    @classmethod
    def unpack(cls, data: bytes) -> "TemporalStreamHeader":
        if len(data) < _HEADER_STRUCT.size:
            raise TemporalFormatError(
                f"Truncated .nvct header: need {_HEADER_STRUCT.size} bytes, got {len(data)}"
            )
        (magic, version, gop, bits, mode_code, coder, width, height, channels,
         lc, lh, lw, frame_count, n_intra, n_residual, intra_id, residual_id) = \
            _HEADER_STRUCT.unpack_from(data, 0)
        if magic != TEMPORAL_MAGIC:
            raise TemporalFormatError(
                f"Bad magic bytes: expected {TEMPORAL_MAGIC!r}, got {magic!r}"
            )
        if version != TEMPORAL_FORMAT_VERSION:
            raise TemporalFormatError(
                f"Unsupported .nvct format version {version}; this build writes and reads "
                f"version {TEMPORAL_FORMAT_VERSION}"
            )
        if mode_code not in (0, 1):
            raise TemporalFormatError(f"Unknown quantization_mode code {mode_code}")
        if gop < 1:
            raise TemporalFormatError(f"gop_size must be >= 1, got {gop}")
        return cls(
            gop_size=gop, quantization_bits=bits,
            quantization_mode="per_channel" if mode_code == 1 else "global",
            image_width=width, image_height=height, image_channels=channels,
            latent_channels=lc, latent_height=lh, latent_width=lw,
            frame_count=frame_count, num_intra_quantization_params=n_intra,
            num_residual_quantization_params=n_residual,
            intra_entropy_model_id=intra_id, residual_entropy_model_id=residual_id,
            entropy_coder_id=coder, format_version=version,
        )


def _pack_quantization_block(params: QuantizationParams) -> bytes:
    scale = params.scale.detach().flatten().cpu().numpy().astype(np.float32)
    zero = params.zero_point.detach().flatten().cpu().numpy().astype(np.float32)
    if scale.size != zero.size:
        raise TemporalFormatError("scale and zero_point must have the same element count")
    return b"".join(struct.pack("<ff", float(s), float(z)) for s, z in zip(scale, zero))


def _unpack_quantization_block(data: bytes, count: int, *, bits: int, mode: str,
                               latent_channels: int) -> QuantizationParams:
    if len(data) < count * 8:
        raise TemporalFormatError(
            f"Truncated quantization block: need {count * 8} bytes, got {len(data)}"
        )
    values = [struct.unpack_from("<ff", data, i * 8) for i in range(count)]
    scale = torch.tensor([v[0] for v in values], dtype=torch.float32)
    zero = torch.tensor([v[1] for v in values], dtype=torch.float32)
    if mode == "per_channel":
        shape = (1, latent_channels, 1, 1)
    else:
        shape = (1, 1, 1, 1)
    return QuantizationParams(
        scale=scale.reshape(shape), zero_point=zero.reshape(shape), bits=bits, mode=mode,
    )


class TemporalStreamWriter:
    """Writes a .nvct header once, then typed, length-prefixed frame records."""

    def __init__(self, path: str | Path, header: TemporalStreamHeader,
                 intra_params: QuantizationParams, residual_params: QuantizationParams) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._expected = header.frame_count
        self._written = 0
        self._file = open(self._path, "wb")
        try:
            self._file.write(header.pack())
            self._file.write(_pack_quantization_block(intra_params))
            self._file.write(_pack_quantization_block(residual_params))
        except Exception:
            self._file.close()
            raise

    def append_frame(self, frame_type: int, payload: bytes) -> None:
        if frame_type not in (FRAME_TYPE_I, FRAME_TYPE_P):
            raise TemporalFormatError(f"Unknown frame_type {frame_type}")
        if self._written >= self._expected:
            raise TemporalFormatError(
                f"Stream header declares {self._expected} frame(s); append_frame called "
                f"again after all of them were written"
            )
        if self._written == 0 and frame_type != FRAME_TYPE_I:
            raise TemporalFormatError(
                "The first frame of a stream must be an I-frame - a P-frame would "
                "reference state the decoder does not have."
            )
        self._file.write(_FRAME_RECORD_STRUCT.pack(frame_type, len(payload)))
        self._file.write(payload)
        self._written += 1

    def close(self) -> None:
        written, expected = self._written, self._expected
        self._file.close()
        if written != expected:
            raise TemporalFormatError(
                f"Stream header declares {expected} frame(s) but only {written} were written"
            )

    def __enter__(self) -> "TemporalStreamWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self._file.close()


class TemporalStreamReader:
    """Reads a .nvct stream: header, both quantization blocks, then typed frames."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f".nvct stream file not found: {path}")
        data = path.read_bytes()
        self.header = TemporalStreamHeader.unpack(data)
        offset = self.header.header_size

        intra_bytes = self.header.num_intra_quantization_params * 8
        residual_bytes = self.header.num_residual_quantization_params * 8
        if len(data) < offset + intra_bytes + residual_bytes:
            raise TemporalFormatError("Truncated stream: quantization blocks incomplete")
        self.intra_params = _unpack_quantization_block(
            data[offset:offset + intra_bytes], self.header.num_intra_quantization_params,
            bits=self.header.quantization_bits, mode=self.header.quantization_mode,
            latent_channels=self.header.latent_channels)
        offset += intra_bytes
        self.residual_params = _unpack_quantization_block(
            data[offset:offset + residual_bytes], self.header.num_residual_quantization_params,
            bits=self.header.quantization_bits, mode=self.header.quantization_mode,
            latent_channels=self.header.latent_channels)
        offset += residual_bytes
        self._body = data[offset:]

    def __len__(self) -> int:
        return self.header.frame_count

    def __iter__(self) -> Iterator[tuple[int, bytes]]:
        offset, count, body = 0, 0, self._body
        while count < self.header.frame_count:
            if offset + _FRAME_RECORD_STRUCT.size > len(body):
                raise TemporalFormatError(
                    f"Truncated stream: expected {self.header.frame_count} frame(s), only "
                    f"{count} present before running out of data"
                )
            frame_type, length = _FRAME_RECORD_STRUCT.unpack_from(body, offset)
            offset += _FRAME_RECORD_STRUCT.size
            if frame_type not in (FRAME_TYPE_I, FRAME_TYPE_P):
                raise TemporalFormatError(
                    f"Frame {count} declares unknown frame_type {frame_type}"
                )
            if count == 0 and frame_type != FRAME_TYPE_I:
                raise TemporalFormatError("First frame of a stream must be an I-frame")
            if offset + length > len(body):
                raise TemporalFormatError(
                    f"Truncated frame {count}: declares {length} bytes, only "
                    f"{len(body) - offset} present"
                )
            yield frame_type, body[offset:offset + length]
            offset += length
            count += 1
        if offset != len(body):
            raise TemporalFormatError(
                f"Trailing data after the declared {self.header.frame_count} frame(s): "
                f"{len(body) - offset} extra byte(s)"
            )


def gop_frame_types(frame_count: int, gop_size: int) -> list[int]:
    """The GOP pattern for ONE sequence: I then gop_size-1 P frames, repeating.

    Always starts with I, so a sequence can never reference across its own
    boundary into the previous one.
    """
    if gop_size < 1:
        raise ValueError(f"gop_size must be >= 1, got {gop_size}")
    return [FRAME_TYPE_I if index % gop_size == 0 else FRAME_TYPE_P
            for index in range(frame_count)]


@torch.no_grad()
def encode_sequence(
    model: torch.nn.Module,
    frames: torch.Tensor,
    path: str | Path,
    *,
    intra_params: QuantizationParams,
    intra_entropy_model: EmpiricalEntropyModel,
    residual_params: QuantizationParams,
    residual_entropy_model: EmpiricalEntropyModel,
    gop_size: int = DEFAULT_GOP,
    reference_mode: str = "latent",
) -> dict[str, Any]:
    """Encode [N, 3, H, W] frames to a .nvct stream, closed-loop and causal.

    The encoder decodes its own output at every step and predicts from that, so
    the reference state it uses is exactly the state the decoder will hold.
    """
    if frames.dim() != 4:
        raise ValueError(f"Expected [N, 3, H, W] frames, got {tuple(frames.shape)}")
    if reference_mode not in ("latent", "reencode"):
        raise ValueError(f"Unknown reference_mode {reference_mode!r}")

    model.eval()
    device = next(model.parameters()).device
    frame_count = frames.shape[0]
    types = gop_frame_types(frame_count, gop_size)

    probe = model.encode(frames[0:1].to(device))
    latent_shape = tuple(probe.shape[1:])
    header = TemporalStreamHeader(
        gop_size=gop_size, quantization_bits=intra_params.bits,
        quantization_mode=intra_params.mode,
        image_width=frames.shape[3], image_height=frames.shape[2],
        image_channels=frames.shape[1],
        latent_channels=latent_shape[0], latent_height=latent_shape[1],
        latent_width=latent_shape[2], frame_count=frame_count,
        num_intra_quantization_params=intra_params.scale.numel(),
        num_residual_quantization_params=residual_params.scale.numel(),
        intra_entropy_model_id=intra_entropy_model.model_id(),
        residual_entropy_model_id=residual_entropy_model.model_id(),
    )

    records: list[dict[str, Any]] = []
    encoder_latents: list[torch.Tensor] = []
    encoder_reconstructions: list[torch.Tensor] = []
    reference: torch.Tensor | None = None
    with TemporalStreamWriter(path, header, intra_params, residual_params) as writer:
        for index in range(frame_count):
            frame = frames[index:index + 1].to(device)
            latent = model.encode(frame)
            frame_type = types[index]

            if frame_type == FRAME_TYPE_I:
                payload, _ = encode_latent_to_payload(
                    latent, params=intra_params, entropy_model=intra_entropy_model)
                decoded, _ = decode_payload_to_latent(
                    payload, entropy_model=intra_entropy_model, params=intra_params,
                    shape=latent_shape)
                reconstructed_latent = decoded.to(device)
            else:
                if reference is None:
                    raise CausalityViolationError(
                        f"Frame {index} is a P-frame but no reference exists - a decoder "
                        f"could not reproduce this."
                    )
                delta = latent - reference
                payload, _ = encode_latent_to_payload(
                    delta, params=residual_params, entropy_model=residual_entropy_model)
                decoded_delta, _ = decode_payload_to_latent(
                    payload, entropy_model=residual_entropy_model, params=residual_params,
                    shape=latent_shape)
                reconstructed_latent = reference + decoded_delta.to(device)

            writer.append_frame(frame_type, payload)
            reconstruction = model.decode(reconstructed_latent)
            encoder_latents.append(reconstructed_latent.detach().cpu())
            encoder_reconstructions.append(reconstruction.detach().cpu())
            records.append({
                "index": index, "frame_type": FRAME_TYPE_NAMES[frame_type],
                "payload_bytes": len(payload),
            })

            if reference_mode == "reencode":
                reference = model.encode(reconstruction)
            else:
                reference = reconstructed_latent

    total_payload = sum(r["payload_bytes"] for r in records)
    return {
        "path": str(path), "frame_count": frame_count, "gop_size": gop_size,
        "reference_mode": reference_mode,
        "frames": records,
        "payload_bytes": total_payload,
        "container_bytes": Path(path).stat().st_size,
        "i_frames": sum(1 for t in types if t == FRAME_TYPE_I),
        "p_frames": sum(1 for t in types if t == FRAME_TYPE_P),
        "i_frame_payload_bytes": sum(r["payload_bytes"] for r, t in zip(records, types)
                                     if t == FRAME_TYPE_I),
        "p_frame_payload_bytes": sum(r["payload_bytes"] for r, t in zip(records, types)
                                     if t == FRAME_TYPE_P),
        "latent_shape": latent_shape,
        # The encoder's own closed-loop state, for the symmetry invariant: a
        # decoder must arrive at exactly these latents from the bitstream alone.
        "encoder_latents": torch.cat(encoder_latents, dim=0),
        "encoder_reconstructions": torch.cat(encoder_reconstructions, dim=0),
    }


@torch.no_grad()
def decode_sequence_with_models(
    model: torch.nn.Module,
    path: str | Path,
    *,
    intra_entropy_model: EmpiricalEntropyModel,
    residual_entropy_model: EmpiricalEntropyModel,
    reference_mode: str = "latent",
    return_latents: bool = False,
):
    """Decode a .nvct stream to [N, 3, H, W] reconstructions.

    The entropy models are passed in rather than embedded: the container stores
    their 8-byte ids and they are verified against it, matching how `.nvc` and
    `.nvcs` already treat the calibration as out-of-band shared state.
    """
    model.eval()
    device = next(model.parameters()).device
    reader = TemporalStreamReader(path)
    header = reader.header

    for label, expected, supplied in (
        ("intra", header.intra_entropy_model_id, intra_entropy_model),
        ("residual", header.residual_entropy_model_id, residual_entropy_model),
    ):
        actual = supplied.model_id()
        if actual != expected:
            raise TemporalFormatError(
                f"{label} entropy model mismatch: stream declares {expected.hex()}, "
                f"supplied model is {actual.hex()}"
            )

    reconstructions: list[torch.Tensor] = []
    decoded_latents: list[torch.Tensor] = []
    reference: torch.Tensor | None = None
    for index, (frame_type, payload) in enumerate(reader):
        if frame_type == FRAME_TYPE_I:
            latent, _ = decode_payload_to_latent(
                payload, entropy_model=intra_entropy_model, params=reader.intra_params,
                shape=header.latent_shape)
            reconstructed_latent = latent.to(device)
        else:
            if reference is None:
                raise TemporalFormatError(
                    f"Frame {index} is a P-frame but no reference is available"
                )
            delta, _ = decode_payload_to_latent(
                payload, entropy_model=residual_entropy_model, params=reader.residual_params,
                shape=header.latent_shape)
            reconstructed_latent = reference + delta.to(device)

        reconstruction = model.decode(reconstructed_latent)
        if tuple(reconstruction.shape[1:]) != header.image_shape:
            raise TemporalFormatError(
                f"Decoded frame shape {tuple(reconstruction.shape[1:])} does not match the "
                f"stream's declared dimensions {header.image_shape}"
            )
        reconstructions.append(reconstruction)
        decoded_latents.append(reconstructed_latent.detach().cpu())
        reference = model.encode(reconstruction) if reference_mode == "reencode" \
            else reconstructed_latent

    frames = torch.cat(reconstructions, dim=0)
    if return_latents:
        return frames, torch.cat(decoded_latents, dim=0)
    return frames


@torch.no_grad()
def calibrate_temporal_grids(
    model: torch.nn.Module,
    sequences,
    *,
    bits: int,
    mode: str = "per_channel",
    gop_size: int = DEFAULT_GOP,
    max_frames: int = 400,
    lower_percentile: float = 0.1,
    upper_percentile: float = 99.9,
) -> dict[str, Any]:
    """Fit BOTH grids - intra latents and latent residuals - on TRAIN sequences.

    Residual statistics are collected the way the codec will actually produce
    them: closed-loop, against a dequantized reference, so the grid is fitted to
    the distribution it will really see rather than to an idealised
    `z_t - z_{t-1}`.
    """
    model.eval()
    device = next(model.parameters()).device
    intra_latents: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    seen = 0

    # Pass 1: intra grid, from I-frame latents only.
    for sequence in sequences:
        frames = sequence.load_frames()
        for index in range(frames.shape[0]):
            if seen >= max_frames:
                break
            intra_latents.append(model.encode(frames[index:index + 1].to(device)).cpu())
            seen += 1
        if seen >= max_frames:
            break
    if not intra_latents:
        raise ValueError("No calibration frames were collected")
    intra_stack = torch.cat(intra_latents, dim=0)
    intra_params = calibrate_quantization_params(
        intra_stack, bits=bits, mode=mode,
        lower_percentile=lower_percentile, upper_percentile=upper_percentile)
    intra_symbols = np.stack([
        latent_to_symbols(intra_stack[i:i + 1], intra_params).reshape(intra_stack.shape[1], -1)
        for i in range(intra_stack.shape[0])
    ])
    intra_entropy_model = EmpiricalEntropyModel.from_symbols(
        intra_symbols, bits=bits, num_tables=intra_stack.shape[1])

    # Pass 2: residual grid, closed-loop against the dequantized reference.
    latent_shape = tuple(intra_stack.shape[1:])
    seen = 0
    for sequence in sequences:
        frames = sequence.load_frames()
        reference = None
        types = gop_frame_types(frames.shape[0], gop_size)
        for index in range(frames.shape[0]):
            if seen >= max_frames:
                break
            latent = model.encode(frames[index:index + 1].to(device))
            if types[index] == FRAME_TYPE_I:
                payload, _ = encode_latent_to_payload(
                    latent, params=intra_params, entropy_model=intra_entropy_model)
                decoded, _ = decode_payload_to_latent(
                    payload, entropy_model=intra_entropy_model, params=intra_params,
                    shape=latent_shape)
                reference = decoded.to(device)
            else:
                residuals.append((latent - reference).cpu())
                # Reference advances using the TRUE latent here only because the
                # residual grid does not exist yet; this pass fits it. The coder
                # itself is always closed-loop.
                reference = latent
            seen += 1
        if seen >= max_frames:
            break
    if not residuals:
        raise ValueError("No residual frames were collected (is gop_size larger than every sequence?)")
    residual_stack = torch.cat(residuals, dim=0)
    residual_params = calibrate_quantization_params(
        residual_stack, bits=bits, mode=mode,
        lower_percentile=lower_percentile, upper_percentile=upper_percentile)
    residual_symbols = np.stack([
        latent_to_symbols(residual_stack[i:i + 1], residual_params).reshape(
            residual_stack.shape[1], -1)
        for i in range(residual_stack.shape[0])
    ])
    residual_entropy_model = EmpiricalEntropyModel.from_symbols(
        residual_symbols, bits=bits, num_tables=residual_stack.shape[1])

    return {
        "intra_params": intra_params, "intra_entropy_model": intra_entropy_model,
        "residual_params": residual_params, "residual_entropy_model": residual_entropy_model,
        "provenance": {
            "split": "train", "bits": bits, "mode": mode, "gop_size": gop_size,
            "lower_percentile": lower_percentile, "upper_percentile": upper_percentile,
            "intra_frames": int(intra_stack.shape[0]),
            "residual_frames": int(residual_stack.shape[0]),
            "intra_entropy_model_id": intra_entropy_model.model_id().hex(),
            "residual_entropy_model_id": residual_entropy_model.model_id().hex(),
            "note": (
                "Two separately fitted grids. Intra latents and latent residuals have "
                "different statistics, so one shared grid would misallocate levels for both."
            ),
        },
    }


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10G Part C: causal temporal (I/P) coding baseline on the .nvct prototype.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage", choices=["smoke", "calibrate", "all"], default="smoke")
    parser.add_argument("--gop", type=int, default=DEFAULT_GOP)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--reference-mode", choices=["latent", "reencode"], default="latent")
    parser.add_argument("--max-sequences", type=int, default=2)
    parser.add_argument("--max-frames-per-sequence", type=int, default=20)
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    device = get_device() if args.device == "auto" else torch.device(args.device)
    # Architecture comes from the checkpoint's own model_config - never
    # re-specified here, so a mismatched --latent-channels cannot silently
    # build the wrong model.
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    train_sequences = discover_sequences(
        args.manifest, split="train", max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames_per_sequence)
    test_sequences = discover_sequences(
        args.manifest, split="test", max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames_per_sequence)
    if not train_sequences or not test_sequences:
        print("[ERROR] no sequences discovered from the manifest", file=sys.stderr)
        return 1

    print("=" * 100)
    print("M10G PART C - CAUSAL TEMPORAL BASELINE (.nvct prototype)")
    print("=" * 100)
    print(f"  checkpoint     : {args.checkpoint}")
    print(f"  GOP            : {args.gop}   reference mode: {args.reference_mode}")
    print(f"  quantization   : {args.bits}-bit / {args.mode}")
    print(f"  residual domain: LATENT (the decoder's Sigmoid cannot emit signed pixel residuals)")

    calibration = calibrate_temporal_grids(
        model, train_sequences, bits=args.bits, mode=args.mode, gop_size=args.gop,
        max_frames=args.calibration_frames)
    print(f"\n  calibrated on TRAIN split: {calibration['provenance']['intra_frames']} intra "
          f"frames, {calibration['provenance']['residual_frames']} residual frames")
    print(f"  intra entropy model id   : {calibration['provenance']['intra_entropy_model_id'][:16]}")
    print(f"  residual entropy model id: {calibration['provenance']['residual_entropy_model_id'][:16]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = args.output_dir / "streams"

    # Diagnostic: is there temporal redundancy in this latent space at all?
    # Compares the order-0 empirical entropy of intra symbols against residual
    # symbols. This is what decides whether a learned temporal model has
    # anything to exploit - independent of how well this naive baseline codes.
    intra_entropy = float(
        calibration["intra_entropy_model"].entropy_bits_per_symbol().mean())
    residual_entropy = float(
        calibration["residual_entropy_model"].entropy_bits_per_symbol().mean())
    print()
    print("  TEMPORAL REDUNDANCY DIAGNOSTIC (order-0 entropy of the coded symbols)")
    print(f"    intra latent symbols    : {intra_entropy:.4f} bits/symbol")
    print(f"    latent residual symbols : {residual_entropy:.4f} bits/symbol")
    delta = residual_entropy - intra_entropy
    print(f"    difference              : {delta:+.4f} bits/symbol "
          f"({'residual is CHEAPER' if delta < 0 else 'residual is MORE expensive'})")

    results = []
    print()
    print(f"{'sequence':<22} {'frames':>7} {'I':>4} {'P':>4} {'I B/frm':>9} {'P B/frm':>9} "
          f"{'stream BPP':>11} {'PSNR':>7} {'sym?':>5} {'det?':>5}")
    for sequence in test_sequences:
        frames = sequence.load_frames()
        path = stream_dir / f"{sequence.sequence_id}.nvct"
        encoded = encode_sequence(
            model, frames, path,
            intra_params=calibration["intra_params"],
            intra_entropy_model=calibration["intra_entropy_model"],
            residual_params=calibration["residual_params"],
            residual_entropy_model=calibration["residual_entropy_model"],
            gop_size=args.gop, reference_mode=args.reference_mode)
        decoded, decoded_latents = decode_sequence_with_models(
            model, path,
            intra_entropy_model=calibration["intra_entropy_model"],
            residual_entropy_model=calibration["residual_entropy_model"],
            reference_mode=args.reference_mode, return_latents=True)

        # INVARIANT 1 - encoder/decoder symmetry, at the level the codec
        # actually controls: the decoder must reach exactly the encoder's
        # closed-loop reference latents from the bitstream alone.
        symmetric = torch.equal(encoded["encoder_latents"], decoded_latents)

        # INVARIANT 2 - determinism. Decoding the same stream twice must give
        # the same latents exactly. (Pixels are compared with a tolerance
        # instead: model.decode() is a GPU convolution, and this project has
        # already measured that identical GPU work varies in the last bits.)
        _, latents_again = decode_sequence_with_models(
            model, path,
            intra_entropy_model=calibration["intra_entropy_model"],
            residual_entropy_model=calibration["residual_entropy_model"],
            reference_mode=args.reference_mode, return_latents=True)
        deterministic = torch.equal(decoded_latents, latents_again)

        mse = torch.mean((decoded.cpu() - frames) ** 2).item()
        psnr = float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)
        total_bytes = encoded["container_bytes"]
        stream_bpp = total_bytes * 8 / sequence.total_pixels

        results.append({
            "sequence": sequence.sequence_id,
            **{k: v for k, v in encoded.items()
               if k not in ("encoder_latents", "encoder_reconstructions")},
            "encoder_decoder_latents_identical": symmetric,
            "deterministic_decode": deterministic,
            "mean_psnr_db": psnr,
            "stream_bpp": stream_bpp,
            "total_pixels": sequence.total_pixels,
            "i_bytes_per_frame": encoded["i_frame_payload_bytes"] / max(encoded["i_frames"], 1),
            "p_bytes_per_frame": encoded["p_frame_payload_bytes"] / max(encoded["p_frames"], 1),
            "container_overhead_bytes": total_bytes - encoded["payload_bytes"],
        })
        print(f"{sequence.sequence_id:<22} {encoded['frame_count']:>7} {encoded['i_frames']:>4} "
              f"{encoded['p_frames']:>4} "
              f"{encoded['i_frame_payload_bytes'] / max(encoded['i_frames'], 1):>9.0f} "
              f"{encoded['p_frame_payload_bytes'] / max(encoded['p_frames'], 1):>9.0f} "
              f"{stream_bpp:>11.4f} {psnr:>7.2f} "
              f"{'yes' if symmetric else 'NO':>5} {'yes' if deterministic else 'NO':>5}")

    all_symmetric = all(r["encoder_decoder_latents_identical"] for r in results)
    all_deterministic = all(r["deterministic_decode"] for r in results)
    print()
    print(f"  encoder/decoder reference symmetry (exact): {'PASS' if all_symmetric else 'FAIL'}")
    print(f"  deterministic decode (exact latents)      : "
          f"{'PASS' if all_deterministic else 'FAIL'}")
    print(f"  stream BPP counts EVERY byte in the container, I-frame overhead included.")

    report = {
        "phase": "M10G Part C (temporal baseline smoke test)",
        "container": ".nvct prototype v1",
        "gop_size": args.gop, "reference_mode": args.reference_mode,
        "residual_domain": "latent",
        "calibration_provenance": calibration["provenance"],
        "temporal_redundancy_diagnostic": {
            "intra_entropy_bits_per_symbol": intra_entropy,
            "residual_entropy_bits_per_symbol": residual_entropy,
            "difference_bits_per_symbol": delta,
            "residual_is_cheaper": bool(delta < 0),
            "note": (
                "Order-0 empirical entropy of the symbols each grid actually codes. "
                "This measures whether temporal redundancy is visible in THIS latent "
                "space under a per-channel percentile grid - not how well any "
                "particular codec exploits it."
            ),
        },
        "invariants": {
            "encoder_decoder_reference_symmetry": all_symmetric,
            "deterministic_decode": all_deterministic,
        },
        "sequences": results,
    }
    path = args.output_dir / "temporal_smoke.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
