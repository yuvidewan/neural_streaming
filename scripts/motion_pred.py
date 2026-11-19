"""Predicted block-motion coding and bit-aware motion search (docs/PLAN_motion_rate.md).

Coding: each block's (dy, dx) is sent as its difference (mvd) from the H.264 median predictor
of its causal neighbours: left A, top B, top-right C (top-left D when C is outside the frame).
- No neighbour available (the first block): predictor (0, 0).
- Only A available (the first row): predictor A.
- Otherwise: component-wise median of A, B, C, with an unavailable neighbour counted as (0, 0).

Each mvd component is range coded with a zero-mean discrete Laplace table on [-32, 32]. The
encoder picks one of SCALE_LEVELS scales per frame and per component and sends the two indices
first: payload = [scale index dy (1 byte)][scale index dx (1 byte)][range-coded mvd symbols,
every dy first, then every dx]. All integer, so encoder and decoder agree exactly.

Search: `search_rate_aware` picks per block, in raster order, the vector with the lowest
SAD + lambda_m * bits, bits being the mvd's cost under given scales and the predictor from
the vectors already chosen. Ties go to fewer bits, then smaller |dy|+|dx|, then the first
candidate in (dy, dx) order.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np
import torch

from nvc.compression.range_coder import decode_symbols, encode_symbols

MVD_RADIUS = 32                       # |mvd| <= 2 * search_range for search_range 16
SCALE_LEVELS = 64
SCALE_MIN, SCALE_MAX = 0.05, 16.0
FREQUENCY_TOTAL = 1 << 16
DEFAULT_SCALE_INDEX = 40              # first P-frame of a GOP: a broad table (scale ~1.5)
CODER_NAME = "motion_pred_v1"


@lru_cache(maxsize=None)
def scales() -> np.ndarray:
    return np.exp(np.linspace(np.log(SCALE_MIN), np.log(SCALE_MAX), SCALE_LEVELS))


@lru_cache(maxsize=None)
def tables() -> tuple[np.ndarray, np.ndarray]:
    """(cumulative [SCALE_LEVELS, 2R+2] int64, bits [SCALE_LEVELS, 2R+1] float64). `bits` is
    the exact cost of each symbol under the integer frequencies the range coder uses."""
    values = np.arange(-MVD_RADIUS, MVD_RADIUS + 1, dtype=np.float64)
    cumulative, bits = [], []
    for s in scales():
        a = np.abs(values)
        upper = np.exp(-(a + 0.5) / s)
        lower = np.where(a > 0, np.exp(-(a - 0.5) / s), 1.0)
        mass = np.where(a > 0, 0.5 * (lower - upper), 1.0 - upper)
        mass = mass / mass.sum()
        free = FREQUENCY_TOTAL - len(values)
        frequency = 1 + np.floor(mass * free).astype(np.int64)
        frequency[MVD_RADIUS] += FREQUENCY_TOTAL - int(frequency.sum())
        cumulative.append(np.concatenate([[0], np.cumsum(frequency)]))
        bits.append(-np.log2(frequency / FREQUENCY_TOTAL))
    return np.stack(cumulative).astype(np.int64), np.stack(bits)


def model_id() -> bytes:
    digest = hashlib.sha256()
    digest.update(CODER_NAME.encode())
    digest.update(tables()[0].tobytes())
    return digest.digest()[:8]


# --------------------------------------------------------------------------------------------
# Predictor
# --------------------------------------------------------------------------------------------


def _median3(a, b, c):
    return np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))


def predict_field(field: np.ndarray) -> np.ndarray:
    """Vectorised predictor for a known field [2, by, bx] int64 (the encoder's side)."""
    _, by, bx = field.shape
    zero = np.zeros_like(field)
    a = zero.copy(); a[:, :, 1:] = field[:, :, :-1]
    b = zero.copy(); b[:, 1:, :] = field[:, :-1, :]
    c = zero.copy(); c[:, 1:, :-1] = field[:, :-1, 1:]
    if bx > 1:
        c[:, 1:, -1] = field[:, :-1, -2]           # top-left D at the right edge
    pred = _median3(a, b, c)
    pred[:, 0, :] = a[:, 0, :]                     # first row: only A (zero at the first block)
    if bx == 1:
        pred[:, 1:, 0] = field[:, :-1, 0]          # a one-block-wide frame: only B
    return pred


def _predict_at(field: np.ndarray, y: int, x: int, bx: int) -> tuple[int, int]:
    """The same predictor for one block from the blocks before it (the decoder's side)."""
    if y == 0:
        if x == 0:
            return 0, 0
        return int(field[0, 0, x - 1]), int(field[1, 0, x - 1])
    if bx == 1:
        return int(field[0, y - 1, 0]), int(field[1, y - 1, 0])
    out = []
    for c in range(2):
        a = int(field[c, y, x - 1]) if x > 0 else 0
        b = int(field[c, y - 1, x])
        d = int(field[c, y - 1, x + 1]) if x + 1 < bx else int(field[c, y - 1, x - 1])
        out.append(sorted((a, b, d))[1])
    return out[0], out[1]


# --------------------------------------------------------------------------------------------
# Coding
# --------------------------------------------------------------------------------------------


def best_scales(mvd: np.ndarray) -> tuple[int, int]:
    """Per component, the table index with the fewest bits for these mvds."""
    _, bits = tables()
    out = []
    for c in range(2):
        counts = np.bincount(mvd[c].ravel() + MVD_RADIUS, minlength=2 * MVD_RADIUS + 1)
        out.append(int(np.argmin(bits @ counts)))
    return out[0], out[1]


def encode_field(motion: torch.Tensor) -> tuple[bytes, tuple[int, int]]:
    field = motion.detach().cpu().numpy().astype(np.int64)
    mvd = field - predict_field(field)
    if np.abs(mvd).max() > MVD_RADIUS:
        raise ValueError(f"mvd outside +-{MVD_RADIUS}")
    sy, sx = best_scales(mvd)
    cumulative, _ = tables()
    n = mvd[0].size
    index = np.concatenate([np.full(n, sy, np.int64), np.full(n, sx, np.int64)])
    body = encode_symbols((mvd.reshape(-1) + MVD_RADIUS).astype(np.int64), cumulative, index)
    return bytes([sy, sx]) + body, (sy, sx)


def decode_field(payload: bytes, shape: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
    by, bx = shape
    sy, sx = payload[0], payload[1]
    cumulative, _ = tables()
    n = by * bx
    index = np.concatenate([np.full(n, sy, np.int64), np.full(n, sx, np.int64)])
    mvd = (np.asarray(decode_symbols(payload[2:], 2 * n, cumulative, index), dtype=np.int64)
           - MVD_RADIUS).reshape(2, by, bx)
    field = np.zeros((2, by, bx), dtype=np.int64)
    for y in range(by):
        for x in range(bx):
            py, px = _predict_at(field, y, x, bx)
            field[0, y, x] = mvd[0, y, x] + py
            field[1, y, x] = mvd[1, y, x] + px
    return torch.from_numpy(field), (sy, sx)


# --------------------------------------------------------------------------------------------
# Bit-aware search
# --------------------------------------------------------------------------------------------


def search_rate_aware(cost: torch.Tensor, *, search_range: int, lambda_m: float,
                      scale_indices: tuple[int, int]) -> torch.Tensor:
    """`cost` [K, by, bx]: per-block SAD of every candidate in (dy, dx) visiting order, as
    computed by `m10h_motion_compensation.block_sad_costs`. Returns a LongTensor [2, by, bx]."""
    cost_np = cost.detach().double().cpu().numpy()
    k, by, bx = cost_np.shape
    span = 2 * search_range + 1
    if k != span * span:
        raise ValueError(f"{k} candidates for search range {search_range}")
    cand_dy = np.repeat(np.arange(-search_range, search_range + 1), span)
    cand_dx = np.tile(np.arange(-search_range, search_range + 1), span)
    key = (np.abs(cand_dy) + np.abs(cand_dx)).astype(np.float64)
    _, bits_table = tables()
    bits_y, bits_x = bits_table[scale_indices[0]], bits_table[scale_indices[1]]
    field = np.zeros((2, by, bx), dtype=np.int64)
    for y in range(by):
        for x in range(bx):
            py, px = _predict_at(field, y, x, bx)
            bits = bits_y[cand_dy - py + MVD_RADIUS] + bits_x[cand_dx - px + MVD_RADIUS]
            total = cost_np[:, y, x] + lambda_m * bits
            best = total == total.min()
            if best.sum() > 1:
                b = np.where(best, bits, np.inf)
                best &= b == b.min()
                if best.sum() > 1:
                    kk = np.where(best, key, np.inf)
                    best &= kk == kk.min()
            i = int(np.argmax(best))
            field[0, y, x], field[1, y, x] = cand_dy[i], cand_dx[i]
    return torch.from_numpy(field)
