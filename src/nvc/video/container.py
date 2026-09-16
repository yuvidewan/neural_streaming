"""The `.nvct` version 2 video container.

Promoted from `scripts/m10h_motion_compensation.py` (M10H). The byte layout is
unchanged, so every `.nvct` v2 stream the research code wrote is readable here
and vice versa:

    0   4  char[]  magic b"NVCT"
    4   1  uint8   format_version (2)
    5   1  uint8   gop_size
    6   1  uint8   quantization_bits
    7   1  uint8   quantization_mode: 0=global, 1=per_channel
    8   1  uint8   entropy_coder_id
    9   2  uint16  image_width
    11  2  uint16  image_height
    13  1  uint8   image_channels
    14  2  uint16  latent_channels
    16  2  uint16  latent_height
    18  2  uint16  latent_width
    20  4  uint32  frame_count
    24  2  uint16  num_intra_quantization_params
    26  2  uint16  num_residual_quantization_params
    28  1  uint8   block_size
    29  1  uint8   search_range
    30  1  uint8   motion_bits
    31  1  uint8   reference_mode (0=prev, 1=mc, 2=oracle)
    32  8  bytes   intra_entropy_model_id
    40  8  bytes   residual_entropy_model_id
    48  8  bytes   motion_entropy_model_id

then the intra and residual quantization blocks (float32 scale, float32 zero point
per parameter), then per frame: uint8 frame_type (0=I, 1=P), uint32 motion length,
uint32 residual length, the motion payload, the residual payload.

Differences from the research reader, all fail-closed: streams are read from
bytes as well as files, and the quantization block sizes are checked against the
declared latent channel count before anything is allocated from them.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
import torch

from nvc.compression.quantization import QuantizationParams

TEMPORAL_MAGIC = b"NVCT"
TEMPORAL_FORMAT_VERSION = 2
FRAME_TYPE_I = 0
FRAME_TYPE_P = 1
MODE_CODES = {"prev": 0, "mc": 1, "oracle": 2}
MODE_NAMES = {code: name for name, code in MODE_CODES.items()}

_HEADER_STRUCT = struct.Struct("<4sBBBBBHHBHHHIHHBBBB8s8s8s")
TEMPORAL_HEADER_SIZE = _HEADER_STRUCT.size
_FRAME_RECORD_STRUCT = struct.Struct("<BII")      # frame_type, motion_length, residual_length


class TemporalFormatError(Exception):
    """A malformed, truncated, unsupported or mismatched `.nvct` v2 stream."""


def gop_frame_types(frame_count: int, gop_size: int) -> list[int]:
    """I then gop_size - 1 P frames, repeating; always starts with an I-frame."""
    if gop_size < 1:
        raise ValueError(f"gop_size must be >= 1, got {gop_size}")
    return [FRAME_TYPE_I if index % gop_size == 0 else FRAME_TYPE_P
            for index in range(frame_count)]


class TemporalStreamHeader:
    """The 56-byte `.nvct` v2 header. See the module docstring for the layout."""

    def __init__(self, *, gop_size: int, quantization_bits: int, quantization_mode: str,
                 image_width: int, image_height: int, image_channels: int,
                 latent_channels: int, latent_height: int, latent_width: int,
                 frame_count: int, num_intra_quantization_params: int,
                 num_residual_quantization_params: int, block_size: int, search_range: int,
                 motion_bits: int, reference_mode: str, intra_entropy_model_id: bytes,
                 residual_entropy_model_id: bytes, motion_entropy_model_id: bytes,
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
        self.block_size = block_size
        self.search_range = search_range
        self.motion_bits = motion_bits
        self.reference_mode = reference_mode
        self.intra_entropy_model_id = intra_entropy_model_id
        self.residual_entropy_model_id = residual_entropy_model_id
        self.motion_entropy_model_id = motion_entropy_model_id
        self.entropy_coder_id = entropy_coder_id
        self.format_version = format_version

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (self.latent_channels, self.latent_height, self.latent_width)

    @property
    def block_grid(self) -> tuple[int, int]:
        return (self.image_height // self.block_size, self.image_width // self.block_size)

    def pack(self) -> bytes:
        return _HEADER_STRUCT.pack(
            TEMPORAL_MAGIC, self.format_version, self.gop_size, self.quantization_bits,
            1 if self.quantization_mode == "per_channel" else 0, self.entropy_coder_id,
            self.image_width, self.image_height, self.image_channels,
            self.latent_channels, self.latent_height, self.latent_width, self.frame_count,
            self.num_intra_quantization_params, self.num_residual_quantization_params,
            self.block_size, self.search_range, self.motion_bits,
            MODE_CODES[self.reference_mode],
            self.intra_entropy_model_id[:8].ljust(8, b"\0"),
            self.residual_entropy_model_id[:8].ljust(8, b"\0"),
            self.motion_entropy_model_id[:8].ljust(8, b"\0"))

    @classmethod
    def unpack(cls, data: bytes) -> "TemporalStreamHeader":
        if len(data) < TEMPORAL_HEADER_SIZE:
            raise TemporalFormatError(
                f"Truncated .nvct header: need {TEMPORAL_HEADER_SIZE} bytes, got {len(data)}")
        (magic, version, gop, bits, mode_code, coder, width, height, channels,
         lc, lh, lw, frame_count, n_intra, n_residual, block_size, search_range,
         motion_bits, mode, intra_id, residual_id, motion_id) = _HEADER_STRUCT.unpack_from(data, 0)
        if magic != TEMPORAL_MAGIC:
            raise TemporalFormatError(f"Bad magic bytes: expected {TEMPORAL_MAGIC!r}, got {magic!r}")
        if version != TEMPORAL_FORMAT_VERSION:
            raise TemporalFormatError(
                f"Unsupported .nvct format version {version}; this build reads version "
                f"{TEMPORAL_FORMAT_VERSION}")
        if mode_code not in (0, 1):
            raise TemporalFormatError(f"Unknown quantization_mode code {mode_code}")
        if mode not in MODE_NAMES:
            raise TemporalFormatError(f"Unknown reference mode code {mode}")
        if gop < 1:
            raise TemporalFormatError(f"gop_size must be >= 1, got {gop}")
        if block_size < 1:
            raise TemporalFormatError(f"block_size must be >= 1, got {block_size}")
        quantization_mode = "per_channel" if mode_code == 1 else "global"
        expected_params = lc if quantization_mode == "per_channel" else 1
        for label, count in (("intra", n_intra), ("residual", n_residual)):
            if count != expected_params:
                raise TemporalFormatError(
                    f"{label} quantization block declares {count} parameter(s); a "
                    f"{quantization_mode} stream with {lc} latent channels needs {expected_params}")
        return cls(
            gop_size=gop, quantization_bits=bits, quantization_mode=quantization_mode,
            image_width=width, image_height=height, image_channels=channels,
            latent_channels=lc, latent_height=lh, latent_width=lw, frame_count=frame_count,
            num_intra_quantization_params=n_intra, num_residual_quantization_params=n_residual,
            block_size=block_size, search_range=search_range, motion_bits=motion_bits,
            reference_mode=MODE_NAMES[mode], intra_entropy_model_id=intra_id,
            residual_entropy_model_id=residual_id, motion_entropy_model_id=motion_id,
            entropy_coder_id=coder, format_version=version)


def _pack_quantization_block(params: QuantizationParams) -> bytes:
    scale = params.scale.detach().flatten().cpu().numpy().astype(np.float32)
    zero = params.zero_point.detach().flatten().cpu().numpy().astype(np.float32)
    return b"".join(struct.pack("<ff", float(s), float(z)) for s, z in zip(scale, zero))


def _unpack_quantization_block(data: bytes, count: int, *, bits: int, mode: str,
                               latent_channels: int) -> QuantizationParams:
    if len(data) < count * 8:
        raise TemporalFormatError(
            f"Truncated quantization block: need {count * 8} bytes, got {len(data)}")
    values = [struct.unpack_from("<ff", data, i * 8) for i in range(count)]
    scale = torch.tensor([v[0] for v in values], dtype=torch.float32)
    zero = torch.tensor([v[1] for v in values], dtype=torch.float32)
    shape = (1, latent_channels, 1, 1) if mode == "per_channel" else (1, 1, 1, 1)
    return QuantizationParams(scale=scale.reshape(shape), zero_point=zero.reshape(shape),
                              bits=bits, mode=mode)


class TemporalStreamWriter:
    """Writes the header once, then typed frame records with explicit lengths.

    `target` is a path or a writable binary file object (e.g. `io.BytesIO`).
    """

    def __init__(self, target: str | Path | BinaryIO, header: TemporalStreamHeader,
                 intra_params: QuantizationParams, residual_params: QuantizationParams) -> None:
        self._owns_file = isinstance(target, (str, Path))
        if self._owns_file:
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file: BinaryIO = open(path, "wb")
        else:
            self._file = target
        self._expected = header.frame_count
        self._written = 0
        try:
            self._file.write(header.pack())
            self._file.write(_pack_quantization_block(intra_params))
            self._file.write(_pack_quantization_block(residual_params))
        except Exception:
            self._release()
            raise

    def append_frame(self, frame_type: int, motion_payload: bytes, residual_payload: bytes) -> None:
        if frame_type not in (FRAME_TYPE_I, FRAME_TYPE_P):
            raise TemporalFormatError(f"Unknown frame_type {frame_type}")
        if self._written >= self._expected:
            raise TemporalFormatError(
                f"Stream header declares {self._expected} frame(s); append_frame called again")
        if self._written == 0 and frame_type != FRAME_TYPE_I:
            raise TemporalFormatError("The first frame of a stream must be an I-frame")
        if frame_type == FRAME_TYPE_I and motion_payload:
            raise TemporalFormatError("An I-frame must carry no motion payload")
        self._file.write(_FRAME_RECORD_STRUCT.pack(
            frame_type, len(motion_payload), len(residual_payload)))
        self._file.write(motion_payload)
        self._file.write(residual_payload)
        self._written += 1

    def _release(self) -> None:
        if self._owns_file:
            self._file.close()

    def close(self) -> None:
        written, expected = self._written, self._expected
        self._release()
        if written != expected:
            raise TemporalFormatError(
                f"Stream header declares {expected} frame(s) but only {written} were written")


class TemporalStreamReader:
    """Reads a `.nvct` v2 stream from a path or from bytes. Rejects truncation
    anywhere and trailing data after the declared frames."""

    def __init__(self, source: str | Path | bytes | bytearray | memoryview) -> None:
        if isinstance(source, (bytes, bytearray, memoryview)):
            data = bytes(source)
        else:
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f".nvct stream file not found: {path}")
            data = path.read_bytes()
        self.header = TemporalStreamHeader.unpack(data)
        offset = TEMPORAL_HEADER_SIZE
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

    def __iter__(self) -> Iterator[tuple[int, bytes, bytes]]:
        offset, count, body = 0, 0, self._body
        while count < self.header.frame_count:
            if offset + _FRAME_RECORD_STRUCT.size > len(body):
                raise TemporalFormatError(
                    f"Truncated stream: expected {self.header.frame_count} frame(s), only "
                    f"{count} present before running out of data")
            frame_type, motion_length, residual_length = _FRAME_RECORD_STRUCT.unpack_from(
                body, offset)
            offset += _FRAME_RECORD_STRUCT.size
            if frame_type not in (FRAME_TYPE_I, FRAME_TYPE_P):
                raise TemporalFormatError(f"Frame {count} declares unknown frame_type {frame_type}")
            if count == 0 and frame_type != FRAME_TYPE_I:
                raise TemporalFormatError("First frame of a stream must be an I-frame")
            if frame_type == FRAME_TYPE_I and motion_length:
                raise TemporalFormatError(
                    f"Frame {count} is an I-frame but declares {motion_length} motion bytes")
            if offset + motion_length > len(body):
                raise TemporalFormatError(
                    f"Truncated motion payload on frame {count}: declares {motion_length} "
                    f"bytes, only {len(body) - offset} present")
            motion = body[offset:offset + motion_length]
            offset += motion_length
            if offset + residual_length > len(body):
                raise TemporalFormatError(
                    f"Truncated residual payload on frame {count}: declares {residual_length} "
                    f"bytes, only {len(body) - offset} present")
            residual = body[offset:offset + residual_length]
            offset += residual_length
            yield frame_type, motion, residual
            count += 1
        if offset != len(body):
            raise TemporalFormatError(
                f"Trailing data after the declared {self.header.frame_count} frame(s): "
                f"{len(body) - offset} extra byte(s)")


def stream_to_bytes(header: TemporalStreamHeader, intra_params: QuantizationParams,
                    residual_params: QuantizationParams,
                    frames: list[tuple[int, bytes, bytes]]) -> bytes:
    """Serialize a whole stream in memory."""
    buffer = io.BytesIO()
    writer = TemporalStreamWriter(buffer, header, intra_params, residual_params)
    for frame_type, motion, residual in frames:
        writer.append_frame(frame_type, motion, residual)
    writer.close()
    return buffer.getvalue()
