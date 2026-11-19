"""B-frame oracle (docs/PLAN_bframes.md): frame order, two-sided block prediction, B payloads.

Frame order. Inside each closed GOP, anchors sit every M-th frame and on the GOP's last frame.
An anchor is a P-frame predicted from the previous anchor. The frames between two anchors are
B-frames, coded right after the later anchor in bisection order (the middle first, using the
two ends, then each half). The decoder needs no M: when a P record is followed by k B records,
the P's display index is the previous anchor + k + 1 and the B's follow the same bisection.

B payload: struct "<HHI" (mode frequencies f_fwd, f_bwd; f_avg = 2^16 - both; length of the
range-coded mode map), the mode map, uint32 length of the forward field, then the forward and
backward fields as `motion_pred` payloads. A mode is 0 = forward, 1 = backward, 2 = average.
Vectors a block does not use are set to their predictor, so their mvd is 0.
"""

from __future__ import annotations

import struct

import numpy as np
import torch
import torch.nn.functional as F

import motion_pred as mp
from nvc.compression.range_coder import decode_symbols, encode_symbols

FORWARD, BACKWARD, AVERAGE = 0, 1, 2
MODE_TOTAL = 1 << 16
MODE_BITS = float(np.log2(3))           # the mode cost the encoder's decision assumes
_HEAD = struct.Struct("<HHI")
_LEN = struct.Struct("<I")


# --------------------------------------------------------------------------------------------
# Frame order
# --------------------------------------------------------------------------------------------


def _bisect(lo: int, hi: int, out: list) -> None:
    if hi - lo < 2:
        return
    mid = (lo + hi) // 2
    out.append(("B", mid, (lo, hi)))
    _bisect(lo, mid, out)
    _bisect(mid, hi, out)


def coding_order(count: int, gop: int, m: int) -> list[tuple[str, int, tuple[int, ...]]]:
    """[(type, display index, references)] in coding order. m = 1 is the all-P codec."""
    out = []
    for start in range(0, count, gop):
        end = min(start + gop, count) - 1
        out.append(("I", start, ()))
        anchors = list(range(start + m, end, m)) + ([end] if end > start else [])
        previous = start
        for anchor in anchors:
            out.append(("P", anchor, (previous,)))
            _bisect(previous, anchor, out)
            previous = anchor
    return out


def display_order(types: list[str]) -> list[tuple[str, int, tuple[int, ...]]]:
    """The decoder's side of `coding_order`, from the frame types in coding order alone."""
    out, i, anchor = [], 0, None
    while i < len(types):
        kind = types[i]
        if kind == "I":
            index = 0 if anchor is None else anchor + 1
            out.append(("I", index, ()))
            anchor, i = index, i + 1
        elif kind == "P":
            k = 0
            while i + 1 + k < len(types) and types[i + 1 + k] == "B":
                k += 1
            index = anchor + k + 1
            out.append(("P", index, (anchor,)))
            bs: list = []
            _bisect(anchor, index, bs)
            if len(bs) != k:
                raise ValueError(f"{k} B records after a P cannot fill a gap of {index - anchor}")
            out.extend(bs)
            anchor, i = index, i + 1 + k
        else:
            raise ValueError("a B record without a P anchor before it")
    return out


# --------------------------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------------------------


def _expand(blocks: torch.Tensor, block_size: int) -> torch.Tensor:
    return blocks.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)


def combine(forward: torch.Tensor, backward: torch.Tensor, modes: torch.Tensor,
            block_size: int) -> torch.Tensor:
    """Per-block prediction from the two warps; identical on both sides (same float ops)."""
    pixel = _expand(modes.to(forward.device), block_size)
    average = (forward + backward) * 0.5
    return torch.where(pixel == FORWARD, forward, torch.where(pixel == BACKWARD, backward, average))


def block_sad(prediction: torch.Tensor, current: torch.Tensor, block_size: int) -> torch.Tensor:
    absdiff = (current - prediction).abs().sum(dim=1, keepdim=True)
    return F.avg_pool2d(absdiff, block_size)[0, 0] * (block_size * block_size)


def vector_bits(field: torch.Tensor, scales: tuple[int, int]) -> torch.Tensor:
    """Approximate per-block cost of a field's vectors under `scales` (encoder decision only)."""
    f = field.cpu().numpy().astype(np.int64)
    mvd = np.clip(f - mp.predict_field(f), -mp.MVD_RADIUS, mp.MVD_RADIUS) + mp.MVD_RADIUS
    _, bits = mp.tables()
    return torch.from_numpy(bits[scales[0]][mvd[0]] + bits[scales[1]][mvd[1]])


def choose_modes(current, forward, backward, block_size: int, lambda_m: float,
                 bits_forward: torch.Tensor, bits_backward: torch.Tensor) -> torch.Tensor:
    """Per block the lowest SAD + lambda_m * vector bits: a one-sided block frees the other
    vector, the average pays for both. Ties: forward, backward, average."""
    bf, bb = bits_forward.double(), bits_backward.double()
    sads = torch.stack([block_sad(forward, current, block_size).cpu().double() + lambda_m * bf,
                        block_sad(backward, current, block_size).cpu().double() + lambda_m * bb,
                        block_sad((forward + backward) * 0.5, current, block_size).cpu().double()
                        + lambda_m * (bf + bb)])
    return sads.argmin(dim=0).long()


def clear_unused(forward: torch.Tensor, backward: torch.Tensor, modes: torch.Tensor):
    """Raster order: a vector its block does not use becomes its predictor (mvd 0)."""
    fields = [forward.cpu().numpy().astype(np.int64).copy(),
              backward.cpu().numpy().astype(np.int64).copy()]
    m = modes.numpy()
    _, by, bx = fields[0].shape
    for y in range(by):
        for x in range(bx):
            for which, unused_when in ((0, BACKWARD), (1, FORWARD)):
                if m[y, x] == unused_when:
                    py, px = mp._predict_at(fields[which], y, x, bx)
                    fields[which][0, y, x], fields[which][1, y, x] = py, px
    return torch.from_numpy(fields[0]), torch.from_numpy(fields[1])


# --------------------------------------------------------------------------------------------
# Payload
# --------------------------------------------------------------------------------------------


def _mode_table(counts: np.ndarray) -> np.ndarray:
    n = max(int(counts.sum()), 1)
    frequency = 1 + np.floor(counts / n * (MODE_TOTAL - 3)).astype(np.int64)
    frequency[int(np.argmax(counts))] += MODE_TOTAL - int(frequency.sum())
    return frequency


def encode_payload(forward, backward, modes) -> tuple[bytes, tuple, tuple]:
    m = modes.numpy().astype(np.int64).reshape(-1)
    frequency = _mode_table(np.bincount(m, minlength=3))
    cumulative = np.concatenate([[0], np.cumsum(frequency)])[None].astype(np.int64)
    mode_body = encode_symbols(m, cumulative, np.zeros(m.size, np.int64))
    fwd, scales_f = mp.encode_field(forward)
    bwd, scales_b = mp.encode_field(backward)
    payload = (_HEAD.pack(int(frequency[0]), int(frequency[1]), len(mode_body)) + mode_body
               + _LEN.pack(len(fwd)) + fwd + bwd)
    return payload, scales_f, scales_b


def decode_payload(payload: bytes, grid: tuple[int, int]):
    by, bx = grid
    f0, f1, mode_length = _HEAD.unpack_from(payload, 0)
    frequency = np.array([f0, f1, MODE_TOTAL - f0 - f1], dtype=np.int64)
    cumulative = np.concatenate([[0], np.cumsum(frequency)])[None].astype(np.int64)
    offset = _HEAD.size
    modes = np.asarray(decode_symbols(payload[offset:offset + mode_length], by * bx, cumulative,
                                      np.zeros(by * bx, np.int64)), dtype=np.int64)
    offset += mode_length
    (fwd_length,) = _LEN.unpack_from(payload, offset)
    offset += _LEN.size
    forward, _ = mp.decode_field(payload[offset:offset + fwd_length], grid)
    backward, _ = mp.decode_field(payload[offset + fwd_length:], grid)
    return forward, backward, torch.from_numpy(modes.reshape(by, bx))
