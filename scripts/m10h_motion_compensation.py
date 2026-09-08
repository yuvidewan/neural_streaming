"""M10H: a causal, fully rate-accounted motion-compensated temporal baseline.

THE QUESTION
-------------
M10G's temporal baseline cut bitrate 9.2% but lost 0.95 dB, and the loss was
almost entirely three high-motion sequences (bmx-bumps -6.33 dB, drone -3.48,
cat-girl -1.44) while six low-motion sequences improved at essentially no
quality cost. The diagnostic showed error does NOT accumulate along the P-chain,
so the problem is not GOP length. The obvious remaining suspect is that
`x_hat_{t-1}` is simply MISALIGNED with `x_t` when content moves.

M10H tests exactly that: warp the reference before differencing, and pay for the
motion field in the bitstream.

    x_warp = Warp(x_hat_{t-1}, mv_t)
    z_ref  = E(x_warp)
    dz     = z_t - z_ref            -> coded with the RESIDUAL grid
    decoder: x_warp = Warp(x_hat_{t-1}, decoded mv_t); z_hat_t = z_ref + dz_hat

Warping happens in PIXEL space (where motion is meaningful); the residual is
still coded in the LATENT domain, which is what M10G established as the correct
causal formulation for this architecture (the decoder's Sigmoid cannot emit
signed pixel residuals - see m10g_temporal_baseline.py).

MOTION IS PAID FOR. NOTHING IS FREE.
-------------------------------------
Every motion vector is quantized, entropy-coded, written into the stream, and
counted in the reported bitrate. The decoder reconstructs motion from those
bytes alone. No ground-truth flow, no encoder-only side information, no implicit
channel. An uncoded motion field would make the comparison meaningless as
compression, so the container carries an explicit motion payload length per
frame and the byte accounting reports motion and residual separately.

The ORACLE path (mode "oracle") is the one exception and is quarantined as
such: it warps with dense float flow that is NOT transmitted. It exists only to
answer "could motion compensation fix this at all?" and its bitrate is not a
compression result. `is_rate_accounted()` returns False for it, and the
evaluator refuses to rank it against the coded paths.

MOTION REPRESENTATION (fully specified, integers only)
-------------------------------------------------------
    estimator        full-search block matching, SAD criterion
    block size       16 x 16 pixels
    spatial res      one vector per block; at 256x256 that is 16 x 16 = 256
                     vectors, exactly one per latent spatial position
    precision        INTEGER pixel. No interpolation anywhere, so the warp is
                     bit-exact and reproducible on any machine.
    range            +/- SEARCH_RANGE pixels per component (default 16)
    coordinates      symbol = component + SEARCH_RANGE, so [0, 2*range]
    alphabet         2**motion_bits, the smallest power of two that holds it
    boundary         replicate (edge pixels extend outward)
    entropy coding   the project's existing arithmetic coder, two frequency
                     tables (one for dy, one for dx)
    tie-break        smallest SAD; ties broken by smallest |dy|+|dx|, then
                     smallest dy, then smallest dx - deterministic, and biased
                     toward zero motion, which is also the cheapest to code.

Nothing float ever enters the bitstream.

CONTAINER
----------
`.nvct` version 2. Version 1 (M10G) had one payload per frame; version 2 carries
a motion payload AND a residual payload with explicit lengths for each. M10G's
reader rejects version 2 by design (it pins version == 1), which is the correct
fail-closed behaviour rather than a regression. `.nvc` and `.nvcs` are untouched.

Example usage (PowerShell, from the project root, with .venv activated):

    .venv\\Scripts\\python.exe scripts\\m10h_motion_compensation.py --stage smoke
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.codec import (
    decode_payload_to_latent,
    encode_latent_to_payload,
    latent_to_symbols,
)
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.compression.quantization import QuantizationParams
from nvc.compression.range_coder import decode_symbols, encode_symbols
from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device

DEFAULT_OUTPUT_DIR = Path("outputs/m10h_motion_compensation")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
FROZEN_LAMBDA = 3.0e-4

DEFAULT_GOP = 10
DEFAULT_BLOCK_SIZE = 16
DEFAULT_SEARCH_RANGE = 16
SNAPSHOT_MODES = ("prev", "mc", "oracle")

FRAME_TYPE_I = 0
FRAME_TYPE_P = 1
FRAME_TYPE_NAMES = {FRAME_TYPE_I: "I", FRAME_TYPE_P: "P"}

MODE_CODES = {"prev": 0, "mc": 1, "oracle": 2}
MODE_NAMES = {code: name for name, code in MODE_CODES.items()}

TEMPORAL_MAGIC = b"NVCT"
TEMPORAL_FORMAT_VERSION = 2
_HEADER_STRUCT = struct.Struct("<4sBBBBBHHBHHHIHHBBBB8s8s8s")
TEMPORAL_HEADER_SIZE = _HEADER_STRUCT.size
# frame_type, motion_length, residual_length
_FRAME_RECORD_STRUCT = struct.Struct("<BII")


@contextlib.contextmanager
def deterministic_kernels():
    """Force bit-reproducible cuDNN algorithms for the duration of a codec call.

    REQUIRED for correctness in this design, not a nicety. Motion compensation
    puts the NETWORK in the reference path - both sides compute
    `z_ref = E(Warp(x_hat_{t-1}))`, and `x_hat_{t-1}` itself came from
    `model.decode(...)`. Measured on this machine: `encode()` is bit-exact by
    default but `decode()` is NOT (its transposed convolutions select
    nondeterministic algorithms), so without this the encoder and decoder drift
    apart along the P-chain and the reconstruction silently diverges.

    M10G's default path did not need this because its reference was a pure
    dequantize with no network in it.
    """
    previous_deterministic = torch.backends.cudnn.deterministic
    previous_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = previous_deterministic
        torch.backends.cudnn.benchmark = previous_benchmark


class TemporalFormatError(Exception):
    """Raised on a malformed, truncated or unsupported .nvct v2 stream."""


class CausalityViolationError(RuntimeError):
    """Raised when the coder is asked to use information a decoder cannot have."""


def is_rate_accounted(mode: str) -> bool:
    """Whether a mode's reported bytes are the whole story.

    False for "oracle", whose dense float flow is never transmitted. Kept as a
    function so the evaluator can refuse to rank an un-accounted path rather
    than relying on someone remembering the caveat.
    """
    if mode not in MODE_CODES:
        raise ValueError(f"Unknown mode {mode!r}")
    return mode != "oracle"


def motion_alphabet_bits(search_range: int) -> int:
    """Smallest power-of-two alphabet that holds [-range, +range]."""
    if search_range < 0:
        raise ValueError(f"search_range must be >= 0, got {search_range}")
    return max(1, math.ceil(math.log2(2 * search_range + 1)))


# --- motion estimation -------------------------------------------------------


@torch.no_grad()
def estimate_block_motion(
    reference: torch.Tensor,
    current: torch.Tensor,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    search_range: int = DEFAULT_SEARCH_RANGE,
) -> torch.Tensor:
    """Full-search block matching, SAD criterion, integer pixels.

    `reference` and `current` are [1, C, H, W] in [0, 1]. Returns a LongTensor
    [2, blocks_y, blocks_x] holding (dy, dx) per block, each in
    [-search_range, +search_range].

    Deterministic by construction: candidates are visited in a fixed order and
    ties are broken toward zero motion, so the same inputs always produce the
    same field on any machine. That matters more than it sounds - a
    nondeterministic estimator would make encoder and decoder disagree about
    the reference, and the failure would look like a codec bug.
    """
    if reference.shape != current.shape:
        raise ValueError(
            f"reference {tuple(reference.shape)} and current {tuple(current.shape)} "
            f"must have the same shape")
    if reference.dim() != 4 or reference.shape[0] != 1:
        raise ValueError(f"expected [1, C, H, W], got {tuple(reference.shape)}")
    _, _, height, width = reference.shape
    if height % block_size or width % block_size:
        raise ValueError(
            f"frame {height}x{width} must divide evenly into {block_size}x{block_size} blocks")

    blocks_y, blocks_x = height // block_size, width // block_size
    padded = F.pad(reference, (search_range,) * 4, mode="replicate")

    best_cost = None
    best_dy = torch.zeros(blocks_y, blocks_x, dtype=torch.long, device=reference.device)
    best_dx = torch.zeros_like(best_dy)
    # Tie-break key, minimised alongside cost: prefers zero motion, then the
    # lexicographically smallest (dy, dx).
    best_key = None

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            top, left = search_range + dy, search_range + dx
            shifted = padded[:, :, top:top + height, left:left + width]
            absdiff = (current - shifted).abs().sum(dim=1, keepdim=True)
            # Sum of absolute differences per block.
            cost = F.avg_pool2d(absdiff, block_size)[0, 0] * (block_size * block_size)
            key = abs(dy) + abs(dx)

            if best_cost is None:
                best_cost = cost.clone()
                best_key = torch.full_like(cost, float(key))
                best_dy.fill_(dy)
                best_dx.fill_(dx)
                continue

            better = cost < best_cost
            tied = (cost == best_cost) & (torch.full_like(cost, float(key)) < best_key)
            update = better | tied
            best_cost = torch.where(update, cost, best_cost)
            best_key = torch.where(update, torch.full_like(cost, float(key)), best_key)
            best_dy = torch.where(update, torch.full_like(best_dy, dy), best_dy)
            best_dx = torch.where(update, torch.full_like(best_dx, dx), best_dx)

    return torch.stack([best_dy, best_dx], dim=0)


@torch.no_grad()
def warp_blocks(reference: torch.Tensor, motion: torch.Tensor, *,
                block_size: int = DEFAULT_BLOCK_SIZE) -> torch.Tensor:
    """Apply an integer-pixel block motion field to a reference frame.

    Block (by, bx) is filled from the reference at an offset of (dy, dx), with
    replicate padding at the frame edge. Pure integer indexing - no
    interpolation - so encoder and decoder produce bit-identical output.
    """
    if reference.dim() != 4 or reference.shape[0] != 1:
        raise ValueError(f"expected [1, C, H, W], got {tuple(reference.shape)}")
    _, channels, height, width = reference.shape
    blocks_y, blocks_x = height // block_size, width // block_size
    if tuple(motion.shape) != (2, blocks_y, blocks_x):
        raise ValueError(
            f"motion {tuple(motion.shape)} does not match a {blocks_y}x{blocks_x} block grid")

    device = reference.device
    rows = torch.arange(height, device=device).view(height, 1).expand(height, width)
    columns = torch.arange(width, device=device).view(1, width).expand(height, width)
    dy = motion[0].to(device).repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    dx = motion[1].to(device).repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)

    source_rows = (rows + dy).clamp(0, height - 1)
    source_columns = (columns + dx).clamp(0, width - 1)
    flat = (source_rows * width + source_columns).reshape(-1)
    warped = reference.reshape(1, channels, -1)[:, :, flat].reshape(1, channels, height, width)
    return warped


@torch.no_grad()
def estimate_dense_flow_oracle(reference: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """DIAGNOSTIC ONLY - dense Farneback flow, never transmitted.

    Used exclusively by the "oracle" ablation to answer whether motion
    compensation *could* fix the high-motion failure if motion were free and
    dense. Its bytes are not in any stream, so any bitrate computed with it is
    not a compression result. See `is_rate_accounted`.
    """
    import cv2

    def to_gray(tensor: torch.Tensor) -> np.ndarray:
        array = tensor[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        return cv2.cvtColor((array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # BACKWARD flow: computed current -> reference, so that for each pixel of
    # the frame being predicted it says where to SAMPLE the reference. Computing
    # it reference -> current gives forward motion, which cannot be used to
    # gather without inversion - and silently produces a warp that is worse than
    # no warp at all. Block matching uses the same backward convention (it
    # searches the reference at p + (dy, dx) for the block at p).
    flow = cv2.calcOpticalFlowFarneback(
        to_gray(current), to_gray(reference), None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0)
    return torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).to(reference.device)


@torch.no_grad()
def warp_dense(reference: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """DIAGNOSTIC ONLY - bilinear dense warp for the oracle path."""
    _, _, height, width = reference.shape
    device = reference.device
    rows = torch.arange(height, device=device).view(1, height, 1).expand(1, height, width)
    columns = torch.arange(width, device=device).view(1, 1, width).expand(1, height, width)
    source_x = columns + flow[:, 0]
    source_y = rows + flow[:, 1]
    grid = torch.stack([
        2.0 * source_x / max(width - 1, 1) - 1.0,
        2.0 * source_y / max(height - 1, 1) - 1.0,
    ], dim=-1)
    return F.grid_sample(reference, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


# --- motion payload coding ---------------------------------------------------


def motion_to_symbols(motion: torch.Tensor, *, search_range: int) -> np.ndarray:
    """(dy, dx) -> non-negative symbols, dy table first then dx table."""
    shifted = motion.detach().cpu().numpy().astype(np.int64) + search_range
    if shifted.min() < 0 or shifted.max() > 2 * search_range:
        raise ValueError(
            f"motion vector outside +/-{search_range}: "
            f"[{shifted.min() - search_range}, {shifted.max() - search_range}]")
    return shifted.reshape(-1)


def symbols_to_motion(symbols: np.ndarray, shape: tuple[int, int],
                      *, search_range: int) -> torch.Tensor:
    blocks_y, blocks_x = shape
    array = np.asarray(symbols, dtype=np.int64).reshape(2, blocks_y, blocks_x) - search_range
    return torch.from_numpy(array)


def motion_table_index(blocks_y: int, blocks_x: int) -> np.ndarray:
    """Table 0 for every dy symbol, table 1 for every dx symbol."""
    return np.repeat(np.arange(2, dtype=np.int64), blocks_y * blocks_x)


def encode_motion_payload(motion: torch.Tensor, *, search_range: int,
                          entropy_model: EmpiricalEntropyModel) -> bytes:
    _, blocks_y, blocks_x = motion.shape
    symbols = motion_to_symbols(motion, search_range=search_range)
    return encode_symbols(symbols, entropy_model.cumulative,
                          motion_table_index(blocks_y, blocks_x))


def decode_motion_payload(payload: bytes, shape: tuple[int, int], *, search_range: int,
                          entropy_model: EmpiricalEntropyModel) -> torch.Tensor:
    blocks_y, blocks_x = shape
    symbols = decode_symbols(payload, 2 * blocks_y * blocks_x, entropy_model.cumulative,
                             motion_table_index(blocks_y, blocks_x))
    return symbols_to_motion(symbols, shape, search_range=search_range)


# --- .nvct version 2 container ----------------------------------------------


class TemporalStreamHeader:
    """.nvct version 2 - 56 bytes, then intra and residual quantization blocks.

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

    Then each frame:
        uint8   frame_type (0=I, 1=P)
        uint32  motion_payload_length   (0 for I-frames)
        uint32  residual_payload_length
        bytes   motion payload
        bytes   residual payload

    Both lengths are explicit, so motion and residual bytes are separable for
    accounting and a truncation in either is detectable.
    """

    def __init__(self, *, gop_size: int, quantization_bits: int, quantization_mode: str,
                 image_width: int, image_height: int, image_channels: int,
                 latent_channels: int, latent_height: int, latent_width: int,
                 frame_count: int, num_intra_quantization_params: int,
                 num_residual_quantization_params: int, block_size: int, search_range: int,
                 motion_bits: int, reference_mode: str,
                 intra_entropy_model_id: bytes, residual_entropy_model_id: bytes,
                 motion_entropy_model_id: bytes, entropy_coder_id: int = 1,
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
    def header_size(self) -> int:
        return TEMPORAL_HEADER_SIZE

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return (self.image_channels, self.image_height, self.image_width)

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
            self.motion_entropy_model_id[:8].ljust(8, b"\0"),
        )

    @classmethod
    def unpack(cls, data: bytes) -> "TemporalStreamHeader":
        if len(data) < TEMPORAL_HEADER_SIZE:
            raise TemporalFormatError(
                f"Truncated .nvct header: need {TEMPORAL_HEADER_SIZE} bytes, got {len(data)}")
        (magic, version, gop, bits, mode_code, coder, width, height, channels,
         lc, lh, lw, frame_count, n_intra, n_residual, block_size, search_range,
         motion_bits, mode, intra_id, residual_id, motion_id) = _HEADER_STRUCT.unpack_from(data, 0)
        if magic != TEMPORAL_MAGIC:
            raise TemporalFormatError(
                f"Bad magic bytes: expected {TEMPORAL_MAGIC!r}, got {magic!r}")
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
        return cls(
            gop_size=gop, quantization_bits=bits,
            quantization_mode="per_channel" if mode_code == 1 else "global",
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
    """Writes a .nvct v2 header once, then typed frame records with explicit
    motion and residual lengths."""

    def __init__(self, path, header: TemporalStreamHeader,
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

    def append_frame(self, frame_type: int, motion_payload: bytes, residual_payload: bytes) -> None:
        if frame_type not in (FRAME_TYPE_I, FRAME_TYPE_P):
            raise TemporalFormatError(f"Unknown frame_type {frame_type}")
        if self._written >= self._expected:
            raise TemporalFormatError(
                f"Stream header declares {self._expected} frame(s); append_frame called again")
        if self._written == 0 and frame_type != FRAME_TYPE_I:
            raise TemporalFormatError(
                "The first frame of a stream must be an I-frame - a P-frame would reference "
                "state the decoder does not have.")
        if frame_type == FRAME_TYPE_I and motion_payload:
            raise TemporalFormatError(
                "An I-frame must carry no motion payload - it has no temporal dependency.")
        self._file.write(_FRAME_RECORD_STRUCT.pack(
            frame_type, len(motion_payload), len(residual_payload)))
        self._file.write(motion_payload)
        self._file.write(residual_payload)
        self._written += 1

    def close(self) -> None:
        written, expected = self._written, self._expected
        self._file.close()
        if written != expected:
            raise TemporalFormatError(
                f"Stream header declares {expected} frame(s) but only {written} were written")

    def __enter__(self) -> "TemporalStreamWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self._file.close()


class TemporalStreamReader:
    """Reads a .nvct v2 stream. Rejects truncation in the header, either
    quantization block, a motion payload or a residual payload, and rejects
    trailing data."""

    def __init__(self, path) -> None:
        path = Path(path)
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
                raise TemporalFormatError(
                    f"Frame {count} declares unknown frame_type {frame_type}")
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


def gop_frame_types(frame_count: int, gop_size: int) -> list[int]:
    """I then gop_size-1 P frames, repeating; always starts with I so a
    sequence can never reference across its own boundary."""
    if gop_size < 1:
        raise ValueError(f"gop_size must be >= 1, got {gop_size}")
    return [FRAME_TYPE_I if index % gop_size == 0 else FRAME_TYPE_P
            for index in range(frame_count)]


# --- the coder ---------------------------------------------------------------


@torch.no_grad()
def encode_sequence(model, frames: torch.Tensor, path, *,
                    intra_params, intra_entropy_model,
                    residual_params, residual_entropy_model,
                    motion_entropy_model, mode: str = "mc",
                    gop_size: int = DEFAULT_GOP, block_size: int = DEFAULT_BLOCK_SIZE,
                    search_range: int = DEFAULT_SEARCH_RANGE) -> dict[str, Any]:
    """Encode [N, 3, H, W] frames closed-loop and causally.

    For every P-frame the encoder estimates motion from the previously
    RECONSTRUCTED frame to the current original, codes that motion into the
    stream, warps with the DECODED motion (not the estimated one), and forms
    the latent residual against the warped reference. Warping with the decoded
    field is what guarantees the decoder can reproduce the reference exactly.
    """
    if mode not in MODE_CODES:
        raise ValueError(f"Unknown mode {mode!r}")
    if frames.dim() != 4:
        raise ValueError(f"Expected [N, 3, H, W] frames, got {tuple(frames.shape)}")

    model.eval()
    device = next(model.parameters()).device
    frame_count = frames.shape[0]
    types = gop_frame_types(frame_count, gop_size)
    determinism = deterministic_kernels()
    determinism.__enter__()
    latent_shape = tuple(model.encode(frames[0:1].to(device)).shape[1:])
    blocks_y, blocks_x = frames.shape[2] // block_size, frames.shape[3] // block_size

    header = TemporalStreamHeader(
        gop_size=gop_size, quantization_bits=intra_params.bits,
        quantization_mode=intra_params.mode,
        image_width=frames.shape[3], image_height=frames.shape[2],
        image_channels=frames.shape[1], latent_channels=latent_shape[0],
        latent_height=latent_shape[1], latent_width=latent_shape[2],
        frame_count=frame_count,
        num_intra_quantization_params=intra_params.scale.numel(),
        num_residual_quantization_params=residual_params.scale.numel(),
        block_size=block_size, search_range=search_range,
        motion_bits=motion_entropy_model.bits, reference_mode=mode,
        intra_entropy_model_id=intra_entropy_model.model_id(),
        residual_entropy_model_id=residual_entropy_model.model_id(),
        motion_entropy_model_id=motion_entropy_model.model_id())

    records: list[dict[str, Any]] = []
    encoder_latents: list[torch.Tensor] = []
    encoder_reconstructions: list[torch.Tensor] = []
    previous_reconstruction: torch.Tensor | None = None

    with TemporalStreamWriter(path, header, intra_params, residual_params) as writer:
        for index in range(frame_count):
            frame = frames[index:index + 1].to(device)
            latent = model.encode(frame)
            frame_type = types[index]
            motion_payload = b""

            if frame_type == FRAME_TYPE_I:
                residual_payload, _ = encode_latent_to_payload(
                    latent, params=intra_params, entropy_model=intra_entropy_model)
                decoded, _ = decode_payload_to_latent(
                    residual_payload, entropy_model=intra_entropy_model, params=intra_params,
                    shape=latent_shape)
                reconstructed_latent = decoded.to(device)
            else:
                if previous_reconstruction is None:
                    raise CausalityViolationError(
                        f"Frame {index} is a P-frame but no reconstructed reference exists.")
                if mode == "prev":
                    warped = previous_reconstruction
                elif mode == "mc":
                    motion = estimate_block_motion(
                        previous_reconstruction, frame,
                        block_size=block_size, search_range=search_range)
                    motion_payload = encode_motion_payload(
                        motion, search_range=search_range, entropy_model=motion_entropy_model)
                    # Warp with the DECODED motion, so the encoder's reference is
                    # exactly what the decoder will rebuild from these bytes.
                    decoded_motion = decode_motion_payload(
                        motion_payload, (blocks_y, blocks_x), search_range=search_range,
                        entropy_model=motion_entropy_model)
                    warped = warp_blocks(previous_reconstruction, decoded_motion,
                                         block_size=block_size)
                else:  # oracle - diagnostic only, motion is NOT transmitted
                    flow = estimate_dense_flow_oracle(previous_reconstruction, frame)
                    warped = warp_dense(previous_reconstruction, flow)

                reference_latent = model.encode(warped)
                delta = latent - reference_latent
                residual_payload, _ = encode_latent_to_payload(
                    delta, params=residual_params, entropy_model=residual_entropy_model)
                decoded_delta, _ = decode_payload_to_latent(
                    residual_payload, entropy_model=residual_entropy_model,
                    params=residual_params, shape=latent_shape)
                reconstructed_latent = reference_latent + decoded_delta.to(device)

            writer.append_frame(frame_type, motion_payload, residual_payload)
            reconstruction = model.decode(reconstructed_latent)
            previous_reconstruction = reconstruction
            encoder_latents.append(reconstructed_latent.detach().cpu())
            encoder_reconstructions.append(reconstruction.detach().cpu())
            records.append({
                "index": index, "frame_type": FRAME_TYPE_NAMES[frame_type],
                "motion_bytes": len(motion_payload),
                "residual_bytes": len(residual_payload),
                "total_bytes": len(motion_payload) + len(residual_payload),
            })

    determinism.__exit__(None, None, None)
    container_bytes = Path(path).stat().st_size
    motion_total = sum(r["motion_bytes"] for r in records)
    residual_total = sum(r["residual_bytes"] for r in records)
    return {
        "path": str(path), "frame_count": frame_count, "gop_size": gop_size, "mode": mode,
        "rate_accounted": is_rate_accounted(mode),
        "block_size": block_size, "search_range": search_range,
        "frames": records,
        "i_frames": sum(1 for t in types if t == FRAME_TYPE_I),
        "p_frames": sum(1 for t in types if t == FRAME_TYPE_P),
        "i_frame_residual_bytes": sum(r["residual_bytes"] for r, t in zip(records, types)
                                      if t == FRAME_TYPE_I),
        "p_frame_residual_bytes": sum(r["residual_bytes"] for r, t in zip(records, types)
                                      if t == FRAME_TYPE_P),
        "motion_bytes": motion_total,
        "residual_bytes": residual_total,
        "payload_bytes": motion_total + residual_total,
        "container_bytes": container_bytes,
        "container_overhead_bytes": container_bytes - motion_total - residual_total,
        "latent_shape": latent_shape,
        "encoder_latents": torch.cat(encoder_latents, dim=0),
        "encoder_reconstructions": torch.cat(encoder_reconstructions, dim=0),
    }


@torch.no_grad()
def decode_sequence(model, path, *, intra_entropy_model, residual_entropy_model,
                    motion_entropy_model, return_latents: bool = False):
    """Decode a .nvct v2 stream using ONLY the stream and the model.

    The oracle mode is intentionally undecodable: its motion was never
    transmitted, so a stream claiming that mode is rejected rather than decoded
    into something that silently differs from the encoder's reconstruction.
    """
    model.eval()
    device = next(model.parameters()).device
    reader = TemporalStreamReader(path)
    header = reader.header

    if header.reference_mode == "oracle":
        raise TemporalFormatError(
            "This stream declares the ORACLE reference mode, whose dense flow is not "
            "transmitted. It is a diagnostic, not a decodable bitstream.")

    for label, expected, supplied in (
        ("intra", header.intra_entropy_model_id, intra_entropy_model),
        ("residual", header.residual_entropy_model_id, residual_entropy_model),
        ("motion", header.motion_entropy_model_id, motion_entropy_model),
    ):
        if supplied.model_id() != expected:
            raise TemporalFormatError(
                f"{label} entropy model mismatch: stream declares {expected.hex()}, "
                f"supplied model is {supplied.model_id().hex()}")

    reconstructions: list[torch.Tensor] = []
    decoded_latents: list[torch.Tensor] = []
    previous_reconstruction: torch.Tensor | None = None

    with deterministic_kernels():
      for index, (frame_type, motion_payload, residual_payload) in enumerate(reader):
          if frame_type == FRAME_TYPE_I:
              latent, _ = decode_payload_to_latent(
                  residual_payload, entropy_model=intra_entropy_model,
                  params=reader.intra_params, shape=header.latent_shape)
              reconstructed_latent = latent.to(device)
          else:
              if previous_reconstruction is None:
                  raise TemporalFormatError(
                      f"Frame {index} is a P-frame but no reference is available")
              if header.reference_mode == "prev":
                  warped = previous_reconstruction
              else:
                  motion = decode_motion_payload(
                      motion_payload, header.block_grid, search_range=header.search_range,
                      entropy_model=motion_entropy_model)
                  warped = warp_blocks(previous_reconstruction, motion,
                                       block_size=header.block_size)
              reference_latent = model.encode(warped)
              delta, _ = decode_payload_to_latent(
                  residual_payload, entropy_model=residual_entropy_model,
                  params=reader.residual_params, shape=header.latent_shape)
              reconstructed_latent = reference_latent + delta.to(device)

          reconstruction = model.decode(reconstructed_latent)
          if tuple(reconstruction.shape[1:]) != header.image_shape:
              raise TemporalFormatError(
                  f"Decoded frame shape {tuple(reconstruction.shape[1:])} does not match the "
                  f"stream's declared dimensions {header.image_shape}")
          reconstructions.append(reconstruction)
          decoded_latents.append(reconstructed_latent.detach().cpu())
          previous_reconstruction = reconstruction

    frames = torch.cat(reconstructions, dim=0)
    if return_latents:
        return frames, torch.cat(decoded_latents, dim=0)
    return frames


# --- calibration -------------------------------------------------------------


@torch.no_grad()
def calibrate_grids(model, sequences, *, bits: int, mode: str = "per_channel",
                    gop_size: int = DEFAULT_GOP, block_size: int = DEFAULT_BLOCK_SIZE,
                    search_range: int = DEFAULT_SEARCH_RANGE, reference_mode: str = "mc",
                    max_frames: int = 400, lower_percentile: float = 0.1,
                    upper_percentile: float = 99.9) -> dict[str, Any]:
    """Fit the intra grid, the residual grid and the motion frequency tables on
    TRAIN sequences only.

    The residual grid is fitted against the SAME kind of reference the coder
    will use (warped or not, per `reference_mode`), because a motion-compensated
    residual has a visibly different distribution from a plain frame difference
    and a grid fitted to the wrong one would misallocate its levels.
    """
    model.eval()
    device = next(model.parameters()).device

    intra_latents: list[torch.Tensor] = []
    seen = 0
    for sequence in sequences:
        frames = sequence.load_frames()
        for index in range(frames.shape[0]):
            if seen >= max_frames:
                break
            intra_latents.append(model.encode(frames[index:index + 1].to(device)).cpu())
            seen += 1
        if seen >= max_frames:
            break
    intra_stack = torch.cat(intra_latents, dim=0)
    intra_params = calibrate_quantization_params(
        intra_stack, bits=bits, mode=mode,
        lower_percentile=lower_percentile, upper_percentile=upper_percentile)
    intra_symbols = np.stack([
        latent_to_symbols(intra_stack[i:i + 1], intra_params).reshape(intra_stack.shape[1], -1)
        for i in range(intra_stack.shape[0])])
    intra_entropy_model = EmpiricalEntropyModel.from_symbols(
        intra_symbols, bits=bits, num_tables=intra_stack.shape[1])

    latent_shape = tuple(intra_stack.shape[1:])
    motion_bits = motion_alphabet_bits(search_range)
    residuals: list[torch.Tensor] = []
    motion_symbol_frames: list[np.ndarray] = []
    seen = 0

    for sequence in sequences:
        frames = sequence.load_frames()
        types = gop_frame_types(frames.shape[0], gop_size)
        previous_reconstruction = None
        for index in range(frames.shape[0]):
            if seen >= max_frames:
                break
            frame = frames[index:index + 1].to(device)
            latent = model.encode(frame)
            if types[index] == FRAME_TYPE_I:
                payload, _ = encode_latent_to_payload(
                    latent, params=intra_params, entropy_model=intra_entropy_model)
                decoded, _ = decode_payload_to_latent(
                    payload, entropy_model=intra_entropy_model, params=intra_params,
                    shape=latent_shape)
                previous_reconstruction = model.decode(decoded.to(device))
            elif previous_reconstruction is not None:
                if reference_mode == "prev":
                    warped = previous_reconstruction
                else:
                    motion = estimate_block_motion(
                        previous_reconstruction, frame,
                        block_size=block_size, search_range=search_range)
                    motion_symbol_frames.append(
                        motion_to_symbols(motion, search_range=search_range).reshape(2, -1))
                    warped = warp_blocks(previous_reconstruction, motion, block_size=block_size)
                residuals.append((latent - model.encode(warped)).cpu())
                # Calibration advances the reference from the TRUE latent, because
                # the residual grid being fitted here does not exist yet and cannot
                # be used to close the loop. The CODER is always closed-loop; this
                # makes calibration-time residuals very slightly optimistic relative
                # to coding-time ones, which is noted rather than hidden.
                previous_reconstruction = model.decode(latent)
            seen += 1
        if seen >= max_frames:
            break

    if not residuals:
        raise ValueError("No residual frames collected (is gop_size larger than every sequence?)")
    residual_stack = torch.cat(residuals, dim=0)
    residual_params = calibrate_quantization_params(
        residual_stack, bits=bits, mode=mode,
        lower_percentile=lower_percentile, upper_percentile=upper_percentile)
    residual_symbols = np.stack([
        latent_to_symbols(residual_stack[i:i + 1], residual_params).reshape(
            residual_stack.shape[1], -1)
        for i in range(residual_stack.shape[0])])
    residual_entropy_model = EmpiricalEntropyModel.from_symbols(
        residual_symbols, bits=bits, num_tables=residual_stack.shape[1])

    if motion_symbol_frames:
        motion_symbols = np.stack(motion_symbol_frames)
    else:
        # "prev" mode codes no motion; a uniform two-table model keeps the
        # container's shape identical across modes without ever being used.
        motion_symbols = np.stack([np.stack([
            np.arange(2 ** motion_bits) % (2 * search_range + 1),
            np.arange(2 ** motion_bits) % (2 * search_range + 1)])])
    motion_entropy_model = EmpiricalEntropyModel.from_symbols(
        motion_symbols, bits=motion_bits, num_tables=2)

    return {
        "intra_params": intra_params, "intra_entropy_model": intra_entropy_model,
        "residual_params": residual_params, "residual_entropy_model": residual_entropy_model,
        "motion_entropy_model": motion_entropy_model,
        "provenance": {
            "split": "train", "bits": bits, "mode": mode, "gop_size": gop_size,
            "reference_mode": reference_mode, "block_size": block_size,
            "search_range": search_range, "motion_bits": motion_bits,
            "lower_percentile": lower_percentile, "upper_percentile": upper_percentile,
            "intra_frames": int(intra_stack.shape[0]),
            "residual_frames": int(residual_stack.shape[0]),
            "motion_frames": len(motion_symbol_frames),
            "intra_entropy_model_id": intra_entropy_model.model_id().hex(),
            "residual_entropy_model_id": residual_entropy_model.model_id().hex(),
            "motion_entropy_model_id": motion_entropy_model.model_id().hex(),
        },
    }


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10H: causal motion-compensated temporal baseline (.nvct v2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage", choices=["smoke"], default="smoke")
    parser.add_argument("--mode", choices=list(MODE_CODES), default="mc")
    parser.add_argument("--gop", type=int, default=DEFAULT_GOP)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--quant-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--search-range", type=int, default=DEFAULT_SEARCH_RANGE)
    parser.add_argument("--calibration-frames", type=int, default=120)
    parser.add_argument("--max-sequences", type=int, default=2)
    parser.add_argument("--max-frames-per-sequence", type=int, default=20)
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
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    train_sequences = discover_sequences(
        args.manifest, split="train", max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames_per_sequence)
    test_sequences = discover_sequences(
        args.manifest, split="test", max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames_per_sequence)

    print("=" * 104)
    print("M10H - MOTION-COMPENSATED TEMPORAL BASELINE (.nvct v2)")
    print("=" * 104)
    print(f"  mode        : {args.mode}   rate-accounted: {is_rate_accounted(args.mode)}")
    print(f"  motion      : {args.block_size}x{args.block_size} blocks, full search "
          f"+/-{args.search_range} px, integer-pel, SAD")
    print(f"  motion bits : {motion_alphabet_bits(args.search_range)} "
          f"(alphabet {2 ** motion_alphabet_bits(args.search_range)})")
    print(f"  quantization: {args.bits}-bit / {args.quant_mode}   GOP {args.gop}")

    calibration = calibrate_grids(
        model, train_sequences, bits=args.bits, mode=args.quant_mode, gop_size=args.gop,
        block_size=args.block_size, search_range=args.search_range,
        reference_mode=args.mode, max_frames=args.calibration_frames)
    print(f"\n  calibrated on TRAIN only: {calibration['provenance']['intra_frames']} intra, "
          f"{calibration['provenance']['residual_frames']} residual, "
          f"{calibration['provenance']['motion_frames']} motion frames")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = args.output_dir / "streams"
    print()
    print(f"{'sequence':<22} {'frames':>7} {'motion B':>10} {'residual B':>11} {'total B':>10} "
          f"{'PSNR':>7} {'sym?':>5}")
    results = []
    for sequence in test_sequences:
        frames = sequence.load_frames()
        path = stream_dir / f"{args.mode}_{sequence.sequence_id}.nvct"
        encoded = encode_sequence(
            model, frames, path,
            intra_params=calibration["intra_params"],
            intra_entropy_model=calibration["intra_entropy_model"],
            residual_params=calibration["residual_params"],
            residual_entropy_model=calibration["residual_entropy_model"],
            motion_entropy_model=calibration["motion_entropy_model"],
            mode=args.mode, gop_size=args.gop, block_size=args.block_size,
            search_range=args.search_range)

        symmetric = None
        if is_rate_accounted(args.mode):
            _, decoded_latents = decode_sequence(
                model, path,
                intra_entropy_model=calibration["intra_entropy_model"],
                residual_entropy_model=calibration["residual_entropy_model"],
                motion_entropy_model=calibration["motion_entropy_model"],
                return_latents=True)
            symmetric = torch.equal(encoded["encoder_latents"], decoded_latents)

        recon = encoded["encoder_reconstructions"]
        mse = torch.mean((recon - frames) ** 2).item()
        psnr = float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)
        results.append({"sequence": sequence.sequence_id, "mean_psnr_db": psnr,
                        "encoder_decoder_latents_identical": symmetric,
                        **{k: v for k, v in encoded.items()
                           if k not in ("encoder_latents", "encoder_reconstructions")}})
        print(f"{sequence.sequence_id:<22} {encoded['frame_count']:>7} "
              f"{encoded['motion_bytes']:>10,} {encoded['residual_bytes']:>11,} "
              f"{encoded['container_bytes']:>10,} {psnr:>7.2f} "
              f"{('yes' if symmetric else 'n/a' if symmetric is None else 'NO'):>5}")

    report = {"phase": "M10H smoke", "mode": args.mode,
              "rate_accounted": is_rate_accounted(args.mode),
              "calibration_provenance": calibration["provenance"], "sequences": results}
    path = args.output_dir / f"smoke_{args.mode}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
