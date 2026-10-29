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

VERSION 3 adds transmitted side information
-------------------------------------------
M23 measured a transmitted per-position scale map worth +1.42/+1.76/+1.77% of
real coded bytes on top of the full deployed entropy stack, decoded from its own
payload (`outputs/m23_hyperprior/README.md`). v2 has nowhere to put it: its frame
record is a fixed (frame_type, motion_length, residual_length). v3 appends, and
appends only:

    header   v2's 56 bytes, then
    56  1  uint8   side_bins       (0 = this stream carries no side channel)
    57  1  uint8   side_stride
    58  8  bytes   side_entropy_model_id

    frame    uint8 frame_type, uint32 motion length, uint32 residual length,
             uint32 SIDE length, then the motion, residual and side payloads in
             that order.

The side payload comes last in the record but is decoded FIRST by a v3 decoder,
because the residual's entropy tables are selected by the context it carries. The
bin edges are deliberately NOT in the stream: what is transmitted is the bin
INDEX, which is all a decoder needs to pick a table.

`side_bins = 0` is how a v3 stream says it has no side channel, and such a stream
must declare zero-length side payloads - a v3 header is not a licence to write
one.

WHY PER-FRAME AND NOT PER-GOP
-----------------------------
Per-GOP would be cheaper. M23's coded validation measured the map at exactly 25
bytes per frame: 64 coarse values at 3 bits is 24 bytes of payload plus 1 byte of
range-coder flush, so coding one payload per GOP instead of one per frame
recovers ~4% of the map's cost. But the map is only 0.41-0.93% of the stream, so
that is 0.016-0.037% of the total - and a per-GOP block changes the stream's
interleaving, which the M12 resumable decoder reads. A per-frame payload is one
extra uint32 and leaves that structure alone. The cheaper layout is recorded here
with its measured value rather than taken, so the trade is a decision and not an
oversight.

BACKWARD COMPATIBILITY
----------------------
Additive, and the constants are load-bearing elsewhere: `TEMPORAL_HEADER_SIZE`
and `_FRAME_RECORD_STRUCT` keep their v2 values because `m10g_temporal_baseline`
and `m10h_motion_compensation` parse the layout with them directly. The writer
still defaults to v2 and still emits byte-identical v2 streams; `__iter__` still
yields 3-tuples, so every existing reader is untouched on both versions, and
v3-aware code opts in through `iter_frames_with_side()`.

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
# The default the writer emits, deliberately still 2: a v3 stream is only
# produced by a caller that asks for one.
TEMPORAL_FORMAT_VERSION = 2
TEMPORAL_FORMAT_VERSION_V3 = 3
SUPPORTED_FORMAT_VERSIONS = (2, 3)
FRAME_TYPE_I = 0
FRAME_TYPE_P = 1
MODE_CODES = {"prev": 0, "mc": 1, "oracle": 2}
MODE_NAMES = {code: name for name, code in MODE_CODES.items()}

_HEADER_STRUCT = struct.Struct("<4sBBBBBHHBHHHIHHBBBB8s8s8s")
# v2's values, and they stay v2's: m10g_temporal_baseline and
# m10h_motion_compensation import these and parse the layout with them.
TEMPORAL_HEADER_SIZE = _HEADER_STRUCT.size
_FRAME_RECORD_STRUCT = struct.Struct("<BII")      # frame_type, motion_length, residual_length

# v3's additions: side_bins, side_stride, side_entropy_model_id; and a frame
# record with a fourth length.
_HEADER_EXTENSION_STRUCT = struct.Struct("<BB8s")
TEMPORAL_HEADER_SIZE_V3 = _HEADER_STRUCT.size + _HEADER_EXTENSION_STRUCT.size
_FRAME_RECORD_STRUCT_V3 = struct.Struct("<BIII")


def header_size_for(format_version: int) -> int:
    if format_version == TEMPORAL_FORMAT_VERSION_V3:
        return TEMPORAL_HEADER_SIZE_V3
    return TEMPORAL_HEADER_SIZE


def frame_record_struct_for(format_version: int) -> struct.Struct:
    if format_version == TEMPORAL_FORMAT_VERSION_V3:
        return _FRAME_RECORD_STRUCT_V3
    return _FRAME_RECORD_STRUCT


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
                 format_version: int = TEMPORAL_FORMAT_VERSION,
                 side_bins: int = 0, side_stride: int = 0,
                 side_entropy_model_id: bytes = b"") -> None:
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
        self.side_bins = side_bins
        self.side_stride = side_stride
        self.side_entropy_model_id = side_entropy_model_id
        if format_version not in SUPPORTED_FORMAT_VERSIONS:
            raise TemporalFormatError(
                f"Unsupported .nvct format version {format_version}; this build writes "
                f"{SUPPORTED_FORMAT_VERSIONS}")
        if format_version == TEMPORAL_FORMAT_VERSION and (side_bins or side_stride
                                                          or side_entropy_model_id):
            raise TemporalFormatError(
                "A v2 header cannot describe a side channel; use format_version=3")
        if side_bins:
            # A power of two, because the side symbols are coded on a 2**bits
            # alphabet like every other table in this codec.
            if side_bins & (side_bins - 1):
                raise TemporalFormatError(
                    f"side_bins must be a power of two, got {side_bins}")
            if side_stride < 1:
                raise TemporalFormatError(
                    f"side_stride must be >= 1 when side_bins is set, got {side_stride}")

    @property
    def carries_side_information(self) -> bool:
        """True when this stream declares a side channel its frames may fill."""
        return (self.format_version == TEMPORAL_FORMAT_VERSION_V3
                and self.side_bins > 0)

    @property
    def header_size(self) -> int:
        return header_size_for(self.format_version)

    @property
    def side_grid(self) -> tuple[int, int]:
        """The coarse map's [height, width], ceil-divided as `coarsen` pools it."""
        if not self.carries_side_information:
            raise TemporalFormatError("This stream declares no side channel")
        stride = self.side_stride
        return ((self.latent_height + stride - 1) // stride,
                (self.latent_width + stride - 1) // stride)

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (self.latent_channels, self.latent_height, self.latent_width)

    @property
    def block_grid(self) -> tuple[int, int]:
        return (self.image_height // self.block_size, self.image_width // self.block_size)

    def pack(self) -> bytes:
        base = _HEADER_STRUCT.pack(
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
        if self.format_version != TEMPORAL_FORMAT_VERSION_V3:
            # Byte-identical to every v2 stream ever written, which is the
            # compatibility guarantee this change rests on.
            return base
        return base + _HEADER_EXTENSION_STRUCT.pack(
            self.side_bins, self.side_stride,
            self.side_entropy_model_id[:8].ljust(8, b"\0"))

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
        if version not in SUPPORTED_FORMAT_VERSIONS:
            raise TemporalFormatError(
                f"Unsupported .nvct format version {version}; this build reads versions "
                f"{SUPPORTED_FORMAT_VERSIONS}")
        side_bins, side_stride, side_id = 0, 0, b""
        if version == TEMPORAL_FORMAT_VERSION_V3:
            if len(data) < TEMPORAL_HEADER_SIZE_V3:
                raise TemporalFormatError(
                    f"Truncated .nvct v3 header: need {TEMPORAL_HEADER_SIZE_V3} bytes, "
                    f"got {len(data)}")
            side_bins, side_stride, side_id = _HEADER_EXTENSION_STRUCT.unpack_from(
                data, TEMPORAL_HEADER_SIZE)
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
            entropy_coder_id=coder, format_version=version,
            side_bins=side_bins, side_stride=side_stride, side_entropy_model_id=side_id)


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
        self._record_struct = frame_record_struct_for(header.format_version)
        self._header = header
        try:
            self._file.write(header.pack())
            self._file.write(_pack_quantization_block(intra_params))
            self._file.write(_pack_quantization_block(residual_params))
        except Exception:
            self._release()
            raise

    def append_frame(self, frame_type: int, motion_payload: bytes, residual_payload: bytes,
                     side_payload: bytes = b"") -> None:
        """Append one frame record.

        `side_payload` is v3 only and is refused otherwise, so a stream cannot
        acquire side information its header does not declare. Existing callers
        pass three arguments and write byte-identical v2 records.
        """
        if frame_type not in (FRAME_TYPE_I, FRAME_TYPE_P):
            raise TemporalFormatError(f"Unknown frame_type {frame_type}")
        if self._written >= self._expected:
            raise TemporalFormatError(
                f"Stream header declares {self._expected} frame(s); append_frame called again")
        if self._written == 0 and frame_type != FRAME_TYPE_I:
            raise TemporalFormatError("The first frame of a stream must be an I-frame")
        if frame_type == FRAME_TYPE_I and motion_payload:
            raise TemporalFormatError("An I-frame must carry no motion payload")
        if side_payload and not self._header.carries_side_information:
            raise TemporalFormatError(
                f"Frame {self._written} carries {len(side_payload)} side byte(s) but the "
                + ("header declares side_bins=0" if self._header.format_version
                   == TEMPORAL_FORMAT_VERSION_V3 else
                   f"stream is format version {self._header.format_version}, which has no "
                   "side channel"))
        if self._header.format_version == TEMPORAL_FORMAT_VERSION_V3:
            self._file.write(self._record_struct.pack(
                frame_type, len(motion_payload), len(residual_payload), len(side_payload)))
        else:
            self._file.write(self._record_struct.pack(
                frame_type, len(motion_payload), len(residual_payload)))
        self._file.write(motion_payload)
        self._file.write(residual_payload)
        self._file.write(side_payload)
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
    """Reads a `.nvct` v2 or v3 stream from a path or from bytes. Rejects
    truncation anywhere and trailing data after the declared frames.

    `__iter__` yields (frame_type, motion, residual) for BOTH versions, so every
    reader written against v2 keeps working unchanged. v3-aware callers use
    `iter_frames_with_side()`, which yields a fourth element (empty on v2).
    """

    def __init__(self, source: str | Path | bytes | bytearray | memoryview) -> None:
        if isinstance(source, (bytes, bytearray, memoryview)):
            data = bytes(source)
        else:
            path = Path(source)
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

    def __iter__(self) -> Iterator[tuple[int, bytes, bytes]]:
        """(frame_type, motion, residual) - the v2 shape, on either version."""
        for frame_type, motion, residual, _ in self.iter_frames_with_side():
            yield frame_type, motion, residual

    def iter_frames_with_side(self) -> Iterator[tuple[int, bytes, bytes, bytes]]:
        """(frame_type, motion, residual, side); `side` is empty on a v2 stream."""
        record = frame_record_struct_for(self.header.format_version)
        is_v3 = self.header.format_version == TEMPORAL_FORMAT_VERSION_V3
        offset, count, body = 0, 0, self._body
        while count < self.header.frame_count:
            if offset + record.size > len(body):
                raise TemporalFormatError(
                    f"Truncated stream: expected {self.header.frame_count} frame(s), only "
                    f"{count} present before running out of data")
            if is_v3:
                (frame_type, motion_length, residual_length,
                 side_length) = record.unpack_from(body, offset)
            else:
                frame_type, motion_length, residual_length = record.unpack_from(body, offset)
                side_length = 0
            offset += record.size
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
            if side_length and not self.header.carries_side_information:
                raise TemporalFormatError(
                    f"Frame {count} declares {side_length} side byte(s) but the header "
                    f"declares side_bins=0")
            if offset + side_length > len(body):
                raise TemporalFormatError(
                    f"Truncated side payload on frame {count}: declares {side_length} "
                    f"bytes, only {len(body) - offset} present")
            side = body[offset:offset + side_length]
            offset += side_length
            yield frame_type, motion, residual, side
            count += 1
        if offset != len(body):
            raise TemporalFormatError(
                f"Trailing data after the declared {self.header.frame_count} frame(s): "
                f"{len(body) - offset} extra byte(s)")


def stream_to_bytes(header: TemporalStreamHeader, intra_params: QuantizationParams,
                    residual_params: QuantizationParams,
                    frames: list[tuple[int, bytes, bytes]]) -> bytes:
    """Serialize a whole stream in memory.

    Each frame is (frame_type, motion, residual) or, for v3, a 4-tuple whose
    fourth element is the side payload. Accepting both keeps every existing
    caller's 3-tuples working.
    """
    buffer = io.BytesIO()
    writer = TemporalStreamWriter(buffer, header, intra_params, residual_params)
    for frame in frames:
        if len(frame) == 4:
            frame_type, motion, residual, side = frame
        elif len(frame) == 3:
            frame_type, motion, residual = frame
            side = b""
        else:
            raise TemporalFormatError(
                f"A frame must be (type, motion, residual) or (type, motion, residual, "
                f"side); got {len(frame)} element(s)")
        writer.append_frame(frame_type, motion, residual, side)
    writer.close()
    return buffer.getvalue()
