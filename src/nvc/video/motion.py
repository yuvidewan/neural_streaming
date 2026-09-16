"""Block motion estimation, warping and motion-payload coding.

Promoted from `scripts/m10h_motion_compensation.py` (M10H) with the arithmetic
unchanged: every function here must produce bit-identical output to its research
original, because the encoder and decoder both run it and any drift between them
corrupts every later frame of a GOP. `tests/test_video_codec.py` pins that
equivalence against the frozen script.

Motion is estimated and applied in PIXEL space with integer-pixel vectors, so the
warp is pure indexing (no interpolation) and identical on every backend.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.compression.range_coder import decode_symbols, encode_symbols

DEFAULT_BLOCK_SIZE = 16
DEFAULT_SEARCH_RANGE = 16


@contextlib.contextmanager
def deterministic_kernels():
    """Force bit-reproducible cuDNN algorithms for the duration of a codec call.

    Required for correctness, not a nicety: both sides compute
    `z_ref = E(Warp(x_hat_{t-1}))`, and `x_hat_{t-1}` came from the autoencoder's
    decoder, whose transposed convolutions pick nondeterministic cuDNN algorithms
    by default. Without this guard encoder and decoder drift apart along the
    P-chain and the reconstruction silently diverges.
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


def motion_alphabet_bits(search_range: int) -> int:
    """Smallest power-of-two alphabet that holds [-range, +range]."""
    if search_range < 0:
        raise ValueError(f"search_range must be >= 0, got {search_range}")
    return max(1, math.ceil(math.log2(2 * search_range + 1)))


@torch.no_grad()
def estimate_block_motion(reference: torch.Tensor, current: torch.Tensor, *,
                          block_size: int = DEFAULT_BLOCK_SIZE,
                          search_range: int = DEFAULT_SEARCH_RANGE) -> torch.Tensor:
    """Full-search block matching, SAD criterion, integer pixels.

    `reference` and `current` are [1, C, H, W] in [0, 1]. Returns a LongTensor
    [2, blocks_y, blocks_x] of (dy, dx) per block, each in [-search_range,
    +search_range]. Candidates are visited in a fixed order and ties break toward
    zero motion, so the same inputs give the same field on any machine.
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
    best_key = None       # tie-break: prefer zero motion, then smallest (dy, dx)

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            top, left = search_range + dy, search_range + dx
            shifted = padded[:, :, top:top + height, left:left + width]
            absdiff = (current - shifted).abs().sum(dim=1, keepdim=True)
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
    """Apply an integer-pixel block motion field to a reference frame, with
    replicate padding at the frame edge."""
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
    return reference.reshape(1, channels, -1)[:, :, flat].reshape(1, channels, height, width)


def motion_to_symbols(motion: torch.Tensor, *, search_range: int) -> np.ndarray:
    """(dy, dx) -> non-negative symbols, dy table first then dx table."""
    shifted = motion.detach().cpu().numpy().astype(np.int64) + search_range
    if shifted.min() < 0 or shifted.max() > 2 * search_range:
        raise ValueError(
            f"motion vector outside +/-{search_range}: "
            f"[{shifted.min() - search_range}, {shifted.max() - search_range}]")
    return shifted.reshape(-1)


def symbols_to_motion(symbols: np.ndarray, shape: tuple[int, int], *,
                      search_range: int) -> torch.Tensor:
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
    return encode_symbols(symbols, entropy_model.cumulative, motion_table_index(blocks_y, blocks_x))


def decode_motion_payload(payload: bytes, shape: tuple[int, int], *, search_range: int,
                          entropy_model: EmpiricalEntropyModel) -> torch.Tensor:
    blocks_y, blocks_x = shape
    symbols = decode_symbols(payload, 2 * blocks_y * blocks_x, entropy_model.cumulative,
                             motion_table_index(blocks_y, blocks_x))
    motion = symbols_to_motion(symbols, shape, search_range=search_range)
    if motion.abs().max() > search_range:
        # A symbol in (2 * range, alphabet) decodes to a vector outside the search
        # window; it cannot come from a legitimate encoder, so refuse it.
        raise ValueError(f"decoded motion vector outside +/-{search_range}")
    return motion
