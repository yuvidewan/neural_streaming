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

DESIGN NOTES
------------
Quantization parameters are embedded in every file so a .nvc is
self-describing for dequantization: a decoder needs no sidecar to turn
symbols back into a latent. At 64 channels that is 512 bytes, which is real
overhead on a single frame and is reported honestly as header cost rather
than hidden (see the BPP figures, which are given both payload-only and
total-file). A sequence-level header shared across frames is the obvious
fix once this codes video rather than stills - that is a later milestone.

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
