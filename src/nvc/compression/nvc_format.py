"""The .nvc binary container format, version 1.

BINARY LAYOUT
-------------
All multi-byte integers are LITTLE-ENDIAN and unsigned. No implicit padding
is used anywhere (every struct format string starts with '<').

    Fixed header - 37 bytes
    +--------+------+--------+-----------------------------------------------+
    | Offset | Size | Type   | Field                                         |
    +--------+------+--------+-----------------------------------------------+
    |      0 |    4 | char[] | magic, always b"NVC1"                         |
    |      4 |    1 | uint8  | format_version (currently 1)                  |
    |      5 |    1 | uint8  | quantization_bits                             |
    |      6 |    1 | uint8  | quantization_mode: 0=global, 1=per_channel    |
    |      7 |    1 | uint8  | entropy_coder_id (1 = static_arithmetic_v1)   |
    |      8 |    2 | uint16 | image_width                                   |
    |     10 |    2 | uint16 | image_height                                  |
    |     12 |    1 | uint8  | image_channels                                |
    |     13 |    2 | uint16 | latent_channels                               |
    |     15 |    2 | uint16 | latent_height                                 |
    |     17 |    2 | uint16 | latent_width                                  |
    |     19 |    4 | uint32 | symbol_count                                  |
    |     23 |    2 | uint16 | num_quantization_params                       |
    |     25 |    4 | uint32 | payload_length (bytes)                        |
    |     29 |    8 | bytes  | entropy_model_id (first 8 bytes of a SHA-256) |
    +--------+------+--------+-----------------------------------------------+

    Quantization parameter block - num_quantization_params * 8 bytes
        repeated { float32 scale; float32 zero_point; }

    Payload - payload_length bytes
        arithmetic-coded symbols, MSB-first bit packing

STREAM CONTAINER - .nvcs, version 1 (see NVCStreamHeader below)
-----------------------------------------------------------------
    Fixed header - 33 bytes
    +--------+------+--------+-----------------------------------------------+
    | Offset | Size | Type   | Field                                         |
    +--------+------+--------+-----------------------------------------------+
    |      0 |    4 | char[] | magic, always b"NVCS"                         |
    |      4 |    1 | uint8  | format_version (currently 1)                  |
    |      5 |    1 | uint8  | quantization_bits                             |
    |      6 |    1 | uint8  | quantization_mode: 0=global, 1=per_channel    |
    |      7 |    1 | uint8  | entropy_coder_id (1 = static_arithmetic_v1)   |
    |      8 |    2 | uint16 | image_width                                   |
    |     10 |    2 | uint16 | image_height                                  |
    |     12 |    1 | uint8  | image_channels                                |
    |     13 |    2 | uint16 | latent_channels                               |
    |     15 |    2 | uint16 | latent_height                                 |
    |     17 |    2 | uint16 | latent_width                                  |
    |     19 |    2 | uint16 | num_quantization_params                       |
    |     21 |    4 | uint32 | frame_count                                   |
    |     25 |    8 | bytes  | entropy_model_id (first 8 bytes of a SHA-256) |
    +--------+------+--------+-----------------------------------------------+

    Quantization parameter block - num_quantization_params * 8 bytes, ONCE
    for the whole stream (not repeated per frame - this is the entire point)
        repeated { float32 scale; float32 zero_point; }

    Then `frame_count` repetitions of:
        uint32  payload_length (bytes)
        bytes   payload_length bytes of arithmetic-coded symbols

A stream shares one model/calibration across every frame in it (fixed
latent shape, fixed quantization grid, fixed entropy model - exactly the
assumption every existing calibration/QAT/benchmark workflow already makes
per checkpoint), so `symbol_count` is never stored per frame either: it is
always `latent_channels * latent_height * latent_width`, identical for
every frame in the stream by construction.

This is a strictly ADDITIVE format living alongside the single-frame `.nvc`
format below - `NVCHeader`/`NVCWriter`/`NVCReader`/`encode_frame`/
`decode_frame` are completely unchanged, since a standalone single-image
encode (scripts/encode.py, scripts/decode.py) has no "stream" to amortize a
header across. Use the stream format for a sequence of frames sharing one
model+calibration (e.g. one DAVIS sequence in a benchmark run); use the
plain per-frame format for one-off images.

DESIGN NOTES
------------
Quantization parameters are embedded in every single-frame file so a .nvc
is self-describing for dequantization: a decoder needs no sidecar to turn
symbols back into a latent. At 64 channels that is 512 bytes, which is real
overhead on a single frame and is reported honestly as header cost rather
than hidden (see the BPP figures, which are given both payload-only and
total-file). This module's own design note used to say "a sequence-level
header shared across frames is the obvious fix once this codes video rather
than stills - that is a later milestone" - `NVCStreamHeader`/
`NVCStreamWriter`/`NVCStreamReader` (end of this module) are that fix,
added without touching anything else in this file.

The entropy model is NOT embedded: it is large (a 64x256 frequency table)
and constant across every frame from a given calibration. Instead the header
carries `entropy_model_id`, and the decoder verifies that the model it
loaded matches the one used to encode. A mismatch raises rather than
silently decoding garbage.

`format_version` exists so this layout can change without ambiguity; readers
reject versions they do not understand.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from nvc.compression.quantization import QUANTIZATION_MODES
from nvc.compression.range_coder import ARITHMETIC_CODER_ID

MAGIC = b"NVC1"
FORMAT_VERSION = 1

_HEADER_STRUCT = struct.Struct("<4sBBBBHHBHHHIHI8s")
FIXED_HEADER_SIZE = _HEADER_STRUCT.size  # 37
_PARAM_STRUCT = struct.Struct("<ff")

STREAM_MAGIC = b"NVCS"
STREAM_FORMAT_VERSION = 1

_STREAM_HEADER_STRUCT = struct.Struct("<4sBBBBHHBHHHHI8s")
STREAM_FIXED_HEADER_SIZE = _STREAM_HEADER_STRUCT.size  # 33
_STREAM_FRAME_LENGTH_STRUCT = struct.Struct("<I")

_MODE_TO_CODE = {"global": 0, "per_channel": 1}
_CODE_TO_MODE = {code: mode for mode, code in _MODE_TO_CODE.items()}


class NVCFormatError(Exception):
    """Raised for any malformed, truncated, or unsupported .nvc data.

    Every validation failure surfaces as this one exception type with a
    specific message - never a raw struct.error or a silently wrong decode.
    """


@dataclass(frozen=True)
class NVCHeader:
    quantization_bits: int
    quantization_mode: str
    image_width: int
    image_height: int
    image_channels: int
    latent_channels: int
    latent_height: int
    latent_width: int
    symbol_count: int
    payload_length: int
    entropy_model_id: bytes
    scales: tuple[float, ...]
    zero_points: tuple[float, ...]
    format_version: int = FORMAT_VERSION
    entropy_coder_id: int = ARITHMETIC_CODER_ID

    @property
    def header_size(self) -> int:
        """Total header bytes, including the quantization parameter block."""
        return FIXED_HEADER_SIZE + len(self.scales) * _PARAM_STRUCT.size

    def pack(self) -> bytes:
        if self.quantization_mode not in _MODE_TO_CODE:
            raise NVCFormatError(
                f"quantization_mode must be one of {QUANTIZATION_MODES}, "
                f"got {self.quantization_mode!r}"
            )
        if len(self.scales) != len(self.zero_points):
            raise NVCFormatError(
                f"scales ({len(self.scales)}) and zero_points ({len(self.zero_points)}) "
                "must have the same length"
            )
        if len(self.entropy_model_id) != 8:
            raise NVCFormatError(
                f"entropy_model_id must be exactly 8 bytes, got {len(self.entropy_model_id)}"
            )

        head = _HEADER_STRUCT.pack(
            MAGIC,
            self.format_version,
            self.quantization_bits,
            _MODE_TO_CODE[self.quantization_mode],
            self.entropy_coder_id,
            self.image_width,
            self.image_height,
            self.image_channels,
            self.latent_channels,
            self.latent_height,
            self.latent_width,
            self.symbol_count,
            len(self.scales),
            self.payload_length,
            self.entropy_model_id,
        )
        params = b"".join(
            _PARAM_STRUCT.pack(scale, zero_point)
            for scale, zero_point in zip(self.scales, self.zero_points)
        )
        return head + params

    @classmethod
    def unpack(cls, data: bytes) -> "NVCHeader":
        """Parse and fully validate a header. Raises NVCFormatError on any problem."""
        if len(data) < FIXED_HEADER_SIZE:
            raise NVCFormatError(
                f"File too small to contain a header: {len(data)} bytes, "
                f"need at least {FIXED_HEADER_SIZE}"
            )

        (
            magic, format_version, bits, mode_code, coder_id,
            image_width, image_height, image_channels,
            latent_channels, latent_height, latent_width,
            symbol_count, num_params, payload_length, model_id,
        ) = _HEADER_STRUCT.unpack(data[:FIXED_HEADER_SIZE])

        if magic != MAGIC:
            raise NVCFormatError(f"Bad magic bytes: expected {MAGIC!r}, got {magic!r}")
        if format_version != FORMAT_VERSION:
            raise NVCFormatError(
                f"Unsupported .nvc format version {format_version}; "
                f"this build reads version {FORMAT_VERSION}"
            )
        if coder_id != ARITHMETIC_CODER_ID:
            raise NVCFormatError(
                f"Unsupported entropy coder id {coder_id}; "
                f"this build implements id {ARITHMETIC_CODER_ID}"
            )
        if mode_code not in _CODE_TO_MODE:
            raise NVCFormatError(f"Unknown quantization mode code {mode_code}")
        if not 1 <= bits <= 16:
            raise NVCFormatError(f"quantization_bits out of range: {bits}")

        dimensions = {
            "image_width": image_width, "image_height": image_height,
            "image_channels": image_channels, "latent_channels": latent_channels,
            "latent_height": latent_height, "latent_width": latent_width,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise NVCFormatError(f"Invalid {name}: {value} (must be positive)")

        expected_symbols = latent_channels * latent_height * latent_width
        if symbol_count != expected_symbols:
            raise NVCFormatError(
                f"symbol_count {symbol_count} does not match latent dimensions "
                f"{latent_channels}x{latent_height}x{latent_width} (= {expected_symbols})"
            )

        mode = _CODE_TO_MODE[mode_code]
        expected_params = 1 if mode == "global" else latent_channels
        if num_params != expected_params:
            raise NVCFormatError(
                f"num_quantization_params {num_params} does not match mode '{mode}' "
                f"with {latent_channels} latent channels (expected {expected_params})"
            )

        params_size = num_params * _PARAM_STRUCT.size
        if len(data) < FIXED_HEADER_SIZE + params_size:
            raise NVCFormatError(
                f"Truncated quantization parameter block: need {params_size} bytes, "
                f"only {len(data) - FIXED_HEADER_SIZE} present"
            )

        scales, zero_points = [], []
        for index in range(num_params):
            offset = FIXED_HEADER_SIZE + index * _PARAM_STRUCT.size
            scale, zero_point = _PARAM_STRUCT.unpack_from(data, offset)
            if not scale > 0:
                raise NVCFormatError(
                    f"Invalid scale {scale} for parameter {index} (must be positive)"
                )
            scales.append(scale)
            zero_points.append(zero_point)

        return cls(
            quantization_bits=bits,
            quantization_mode=mode,
            image_width=image_width,
            image_height=image_height,
            image_channels=image_channels,
            latent_channels=latent_channels,
            latent_height=latent_height,
            latent_width=latent_width,
            symbol_count=symbol_count,
            payload_length=payload_length,
            entropy_model_id=model_id,
            scales=tuple(scales),
            zero_points=tuple(zero_points),
            format_version=format_version,
            entropy_coder_id=coder_id,
        )


class NVCWriter:
    """Serializes a header plus an entropy-coded payload into .nvc bytes."""

    @staticmethod
    def to_bytes(header: NVCHeader, payload: bytes) -> bytes:
        if len(payload) != header.payload_length:
            raise NVCFormatError(
                f"payload is {len(payload)} bytes but header declares "
                f"{header.payload_length}"
            )
        return header.pack() + payload

    @staticmethod
    def write(path: str | Path, header: NVCHeader, payload: bytes) -> int:
        """Write a .nvc file; returns the total number of bytes written."""
        data = NVCWriter.to_bytes(header, payload)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return len(data)


class NVCReader:
    """Parses .nvc bytes back into a validated header and payload."""

    @staticmethod
    def from_bytes(data: bytes) -> tuple[NVCHeader, bytes]:
        header = NVCHeader.unpack(data)
        start = header.header_size
        end = start + header.payload_length

        available = len(data) - start
        if available < header.payload_length:
            raise NVCFormatError(
                f"Truncated payload: header declares {header.payload_length} bytes "
                f"but only {available} are present"
            )
        if len(data) > end:
            raise NVCFormatError(
                f"Trailing data after payload: file has {len(data)} bytes, "
                f"expected {end}"
            )
        return header, data[start:end]

    @staticmethod
    def read(path: str | Path) -> tuple[NVCHeader, bytes]:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f".nvc file not found: {path}")
        return NVCReader.from_bytes(path.read_bytes())


@dataclass(frozen=True)
class NVCStreamHeader:
    """The once-per-stream analog of `NVCHeader` - see this module's
    docstring, "STREAM CONTAINER", for the exact layout and why
    `symbol_count`/`payload_length` are absent (the first is implicit, the
    second moves to a per-frame length prefix)."""

    quantization_bits: int
    quantization_mode: str
    image_width: int
    image_height: int
    image_channels: int
    latent_channels: int
    latent_height: int
    latent_width: int
    frame_count: int
    entropy_model_id: bytes
    scales: tuple[float, ...]
    zero_points: tuple[float, ...]
    format_version: int = STREAM_FORMAT_VERSION
    entropy_coder_id: int = ARITHMETIC_CODER_ID

    @property
    def symbol_count(self) -> int:
        """Implicit and identical for every frame in the stream - never stored."""
        return self.latent_channels * self.latent_height * self.latent_width

    @property
    def header_size(self) -> int:
        """Total header bytes, including the quantization parameter block."""
        return STREAM_FIXED_HEADER_SIZE + len(self.scales) * _PARAM_STRUCT.size

    def pack(self) -> bytes:
        if self.quantization_mode not in _MODE_TO_CODE:
            raise NVCFormatError(
                f"quantization_mode must be one of {QUANTIZATION_MODES}, "
                f"got {self.quantization_mode!r}"
            )
        if len(self.scales) != len(self.zero_points):
            raise NVCFormatError(
                f"scales ({len(self.scales)}) and zero_points ({len(self.zero_points)}) "
                "must have the same length"
            )
        if len(self.entropy_model_id) != 8:
            raise NVCFormatError(
                f"entropy_model_id must be exactly 8 bytes, got {len(self.entropy_model_id)}"
            )
        if self.frame_count < 0:
            raise NVCFormatError(f"frame_count must be non-negative, got {self.frame_count}")

        head = _STREAM_HEADER_STRUCT.pack(
            STREAM_MAGIC,
            self.format_version,
            self.quantization_bits,
            _MODE_TO_CODE[self.quantization_mode],
            self.entropy_coder_id,
            self.image_width,
            self.image_height,
            self.image_channels,
            self.latent_channels,
            self.latent_height,
            self.latent_width,
            len(self.scales),
            self.frame_count,
            self.entropy_model_id,
        )
        params = b"".join(
            _PARAM_STRUCT.pack(scale, zero_point)
            for scale, zero_point in zip(self.scales, self.zero_points)
        )
        return head + params

    @classmethod
    def unpack(cls, data: bytes) -> "NVCStreamHeader":
        """Parse and fully validate a stream header. Raises NVCFormatError
        on any problem - mirrors `NVCHeader.unpack`'s validation exactly,
        minus the symbol_count/payload_length checks that don't apply here."""
        if len(data) < STREAM_FIXED_HEADER_SIZE:
            raise NVCFormatError(
                f"File too small to contain a stream header: {len(data)} bytes, "
                f"need at least {STREAM_FIXED_HEADER_SIZE}"
            )

        (
            magic, format_version, bits, mode_code, coder_id,
            image_width, image_height, image_channels,
            latent_channels, latent_height, latent_width,
            num_params, frame_count, model_id,
        ) = _STREAM_HEADER_STRUCT.unpack(data[:STREAM_FIXED_HEADER_SIZE])

        if magic != STREAM_MAGIC:
            raise NVCFormatError(f"Bad magic bytes: expected {STREAM_MAGIC!r}, got {magic!r}")
        if format_version != STREAM_FORMAT_VERSION:
            raise NVCFormatError(
                f"Unsupported .nvcs format version {format_version}; "
                f"this build reads version {STREAM_FORMAT_VERSION}"
            )
        if coder_id != ARITHMETIC_CODER_ID:
            raise NVCFormatError(
                f"Unsupported entropy coder id {coder_id}; "
                f"this build implements id {ARITHMETIC_CODER_ID}"
            )
        if mode_code not in _CODE_TO_MODE:
            raise NVCFormatError(f"Unknown quantization mode code {mode_code}")
        if not 1 <= bits <= 16:
            raise NVCFormatError(f"quantization_bits out of range: {bits}")

        dimensions = {
            "image_width": image_width, "image_height": image_height,
            "image_channels": image_channels, "latent_channels": latent_channels,
            "latent_height": latent_height, "latent_width": latent_width,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise NVCFormatError(f"Invalid {name}: {value} (must be positive)")

        mode = _CODE_TO_MODE[mode_code]
        expected_params = 1 if mode == "global" else latent_channels
        if num_params != expected_params:
            raise NVCFormatError(
                f"num_quantization_params {num_params} does not match mode '{mode}' "
                f"with {latent_channels} latent channels (expected {expected_params})"
            )

        params_size = num_params * _PARAM_STRUCT.size
        if len(data) < STREAM_FIXED_HEADER_SIZE + params_size:
            raise NVCFormatError(
                f"Truncated quantization parameter block: need {params_size} bytes, "
                f"only {len(data) - STREAM_FIXED_HEADER_SIZE} present"
            )

        scales, zero_points = [], []
        for index in range(num_params):
            offset = STREAM_FIXED_HEADER_SIZE + index * _PARAM_STRUCT.size
            scale, zero_point = _PARAM_STRUCT.unpack_from(data, offset)
            if not scale > 0:
                raise NVCFormatError(
                    f"Invalid scale {scale} for parameter {index} (must be positive)"
                )
            scales.append(scale)
            zero_points.append(zero_point)

        return cls(
            quantization_bits=bits,
            quantization_mode=mode,
            image_width=image_width,
            image_height=image_height,
            image_channels=image_channels,
            latent_channels=latent_channels,
            latent_height=latent_height,
            latent_width=latent_width,
            frame_count=frame_count,
            entropy_model_id=model_id,
            scales=tuple(scales),
            zero_points=tuple(zero_points),
            format_version=format_version,
            entropy_coder_id=coder_id,
        )


class NVCStreamWriter:
    """Writes a stream header once, then a sequence of length-prefixed
    payloads. The per-stream analog of `NVCWriter`, for multiple frames
    sharing one model/calibration (see this module's docstring).

    `frame_count` must be known up front (it is part of the fixed header,
    written before any frame), matching how a real benchmark or encode run
    already knows how many frames a sequence has before starting.
    """

    def __init__(self, path: str | Path, header: NVCStreamHeader) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "wb")
        self._frames_written = 0
        self._expected_frames = header.frame_count
        try:
            self._file.write(header.pack())
        except Exception:
            self._file.close()
            raise

    def append_frame(self, payload: bytes) -> None:
        """Write one more frame's payload, length-prefixed. Raises
        NVCFormatError if this would exceed the frame_count declared in the
        header - a silent overrun would leave the file self-inconsistent
        (a reader stops at frame_count regardless of what bytes follow)."""
        if self._frames_written >= self._expected_frames:
            raise NVCFormatError(
                f"Stream header declares {self._expected_frames} frame(s); "
                f"append_frame called again after all of them were written"
            )
        self._file.write(_STREAM_FRAME_LENGTH_STRUCT.pack(len(payload)))
        self._file.write(payload)
        self._frames_written += 1

    def close(self) -> None:
        if self._frames_written != self._expected_frames:
            self._file.close()
            raise NVCFormatError(
                f"Stream header declares {self._expected_frames} frame(s) but only "
                f"{self._frames_written} were written before close()"
            )
        self._file.close()

    def __enter__(self) -> "NVCStreamWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self._file.close()  # don't mask the real exception with a frame-count mismatch


class NVCStreamReader:
    """Reads a stream header once, then yields each frame's payload bytes
    in order. The per-stream analog of `NVCReader`."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f".nvcs stream file not found: {path}")
        data = path.read_bytes()
        self.header = NVCStreamHeader.unpack(data)
        self._body = data[self.header.header_size :]

    def __iter__(self):
        offset = 0
        body = self._body
        count = 0
        while count < self.header.frame_count:
            if offset + _STREAM_FRAME_LENGTH_STRUCT.size > len(body):
                raise NVCFormatError(
                    f"Truncated stream: expected {self.header.frame_count} frame(s), "
                    f"only {count} present before running out of data"
                )
            (length,) = _STREAM_FRAME_LENGTH_STRUCT.unpack_from(body, offset)
            offset += _STREAM_FRAME_LENGTH_STRUCT.size
            if offset + length > len(body):
                raise NVCFormatError(
                    f"Truncated frame {count}: declares {length} bytes, only "
                    f"{len(body) - offset} present"
                )
            yield body[offset : offset + length]
            offset += length
            count += 1
        if offset != len(body):
            raise NVCFormatError(
                f"Trailing data after the declared {self.header.frame_count} frame(s): "
                f"{len(body) - offset} extra byte(s)"
            )

    def __len__(self) -> int:
        return self.header.frame_count
