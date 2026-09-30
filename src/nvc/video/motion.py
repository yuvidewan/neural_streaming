"""Block motion estimation, warping and motion-payload coding.

Promoted from `scripts/m10h_motion_compensation.py` (M10H) with the arithmetic
unchanged: every function here must produce bit-identical output to its research
original, because the encoder and decoder both run it and any drift between them
corrupts every later frame of a GOP. `tests/test_video_codec.py` pins that
equivalence against the frozen script.

Motion is estimated and applied in PIXEL space with integer-pixel vectors, so the
warp is pure indexing (no interpolation) and identical on every backend.

`estimate_block_motion` has two implementations that return bit-identical results: the
PyTorch reference (`estimate_block_motion_torch`) and a native C kernel
(`estimate_block_motion_native`, ~25-30x faster on a CPU). Only the ENCODER estimates
motion - the vectors are transmitted - so the choice never affects decodability. The
dispatcher picks native when it can and falls back to the reference otherwise; see
C_REWRITE_REPORT.md for the measurements.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import ctypes
import math
import os
import threading
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.compression.range_coder import decode_symbols, encode_symbols
from nvc.video import _native

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


def _validate_motion_inputs(reference: torch.Tensor, current: torch.Tensor, block_size: int) -> None:
    """The input checks every implementation shares, so every backend rejects the same
    inputs with the same message."""
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


# --- native backend ---------------------------------------------------------------------

_BACKENDS = ("auto", "native", "torch")
_warned_unavailable = False
_pool_lock = threading.Lock()
_pool: concurrent.futures.ThreadPoolExecutor | None = None
_pool_workers = 0


def _default_threads() -> int:
    """As many threads as PyTorch is currently allowed, so `torch.set_num_threads` also
    bounds this. The result never depends on the thread count."""
    return max(1, torch.get_num_threads())


def _thread_pool(workers: int) -> concurrent.futures.ThreadPoolExecutor:
    global _pool, _pool_workers
    with _pool_lock:
        if _pool is None or _pool_workers != workers:
            if _pool is not None:
                _pool.shutdown(wait=False)
            _pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="nvc-motion")
            _pool_workers = workers
        return _pool


def _native_ineligible(reference: torch.Tensor, current: torch.Tensor, block_size: int,
                       search_range: int) -> str | None:
    """Why the native kernel cannot take this call, or None if it can. Everything it does
    not handle goes to the reference, so this list is what "bit-identical" is scoped to."""
    if reference.device.type != "cpu" or current.device.type != "cpu":
        return "tensors are not on the CPU"
    if reference.dtype != torch.float32 or current.dtype != torch.float32:
        return "tensors are not float32"
    if reference.shape[1] != 3:
        return "the native kernel is specialised for 3 channels"
    if block_size < 1 or search_range < 0:
        return "block_size must be >= 1 and search_range >= 0"
    # NaN comparisons behave differently in the two implementations; pixels are finite in [0, 1].
    if not (bool(torch.isfinite(reference).all()) and bool(torch.isfinite(current).all())):
        return "non-finite pixel values"
    return None


def _run_native(lib, reference: torch.Tensor, current: torch.Tensor, *, block_size: int,
                search_range: int, threads: int, early_exit: bool, mode: int) -> torch.Tensor:
    """Run the kernel. `lib`, `early_exit` and `mode` are parameters so tests can drive a
    deliberately broken kernel, force the scalar path, or turn the early exit off."""
    height, width = reference.shape[2], reference.shape[3]
    float_ptr = ctypes.POINTER(ctypes.c_float)
    int_ptr = ctypes.POINTER(ctypes.c_int32)
    reference_np = np.ascontiguousarray(reference.detach().numpy()[0])
    current_np = np.ascontiguousarray(current.detach().numpy()[0])
    pad = np.empty((3, lib.nvc_bs_padded_height(height, search_range),
                    lib.nvc_bs_padded_width(width, search_range)), dtype=np.float32)
    status = lib.nvc_bs_pad(reference_np.ctypes.data_as(float_ptr), height, width, search_range,
                            pad.ctypes.data_as(float_ptr))
    if status != 0:
        raise RuntimeError(f"native motion padding rejected its arguments (status {status})")

    grid = (height // block_size, width // block_size)
    blocks = grid[0] * grid[1]
    out_dy = np.zeros(blocks, dtype=np.int32)
    out_dx = np.zeros(blocks, dtype=np.int32)
    current_ptr = current_np.ctypes.data_as(float_ptr)
    pad_ptr = pad.ctypes.data_as(float_ptr)
    dy_ptr = out_dy.ctypes.data_as(int_ptr)
    dx_ptr = out_dx.ctypes.data_as(int_ptr)

    def search(bounds: tuple[int, int]) -> int:
        return lib.nvc_bs_search(current_ptr, pad_ptr, height, width, search_range, block_size,
                                 bounds[0], bounds[1], dy_ptr, dx_ptr, int(early_exit), mode)

    # Blocks are independent and each range writes only its own outputs, so chunks can run on
    # threads (ctypes drops the GIL for the call). Many small chunks balance the load, since
    # early exit makes some blocks much cheaper than others.
    chunk = max(1, -(-blocks // (threads * 4)))
    ranges = [(begin, min(begin + chunk, blocks)) for begin in range(0, blocks, chunk)]
    if threads <= 1 or len(ranges) == 1:
        statuses = [search(bounds) for bounds in ranges]
    else:
        statuses = list(_thread_pool(threads).map(search, ranges))
    for status in statuses:
        if status == -2:
            raise RuntimeError("the native motion kernel was asked for AVX2, which this CPU lacks")
        if status != 0:
            raise RuntimeError(f"native motion search rejected its arguments (status {status})")
    return torch.from_numpy(np.stack([out_dy.reshape(grid), out_dx.reshape(grid)]).astype(np.int64))


@torch.no_grad()
def estimate_block_motion_native(reference: torch.Tensor, current: torch.Tensor, *,
                                 block_size: int = DEFAULT_BLOCK_SIZE,
                                 search_range: int = DEFAULT_SEARCH_RANGE,
                                 threads: int | None = None) -> torch.Tensor:
    """`estimate_block_motion_torch`'s result, computed by the native kernel. Strict: raises
    ValueError for an input the kernel does not handle (CPU float32, 3 channels, finite)
    and RuntimeError if the native library is unavailable. Most callers want
    `estimate_block_motion`, which falls back instead. `threads` defaults to
    `torch.get_num_threads()`; the result does not depend on it."""
    _validate_motion_inputs(reference, current, block_size)
    reason = _native_ineligible(reference, current, block_size, search_range)
    if reason is not None:
        raise ValueError(f"native motion search cannot take this input: {reason}")
    lib = _native.load()
    if lib is None:
        raise RuntimeError(f"native motion search unavailable: {_native.last_error()}")
    return _run_native(lib, reference, current, block_size=block_size, search_range=search_range,
                       threads=threads if threads is not None else _default_threads(),
                       early_exit=True, mode=_native.MODE_AUTO)


@torch.no_grad()
def estimate_block_motion(reference: torch.Tensor, current: torch.Tensor, *,
                          block_size: int = DEFAULT_BLOCK_SIZE,
                          search_range: int = DEFAULT_SEARCH_RANGE,
                          backend: str | None = None) -> torch.Tensor:
    """Full-search block matching, SAD criterion, integer pixels.

    `reference` and `current` are [1, C, H, W] in [0, 1]. Returns a LongTensor
    [2, blocks_y, blocks_x] of (dy, dx) per block, each in [-search_range,
    +search_range]. Candidates are visited in a fixed order and ties break toward
    zero motion, so the same inputs give the same field on any machine.

    `backend` is "auto" (default), "native" or "torch"; when None it is read from the
    NVC_MOTION_BACKEND environment variable, else "auto". "auto" uses the native kernel
    when the tensors are CPU float32 with 3 finite channels and the library builds, and
    otherwise the PyTorch reference - with a one-time RuntimeWarning if the library was
    the problem, since the fallback is ~25x slower. Both return bit-identical results.
    """
    global _warned_unavailable
    if backend is None:
        backend = os.environ.get("NVC_MOTION_BACKEND", "auto")
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
    if backend == "native":
        return estimate_block_motion_native(reference, current, block_size=block_size,
                                            search_range=search_range)
    if backend == "auto":
        _validate_motion_inputs(reference, current, block_size)
        if _native_ineligible(reference, current, block_size, search_range) is None:
            lib = _native.load()
            if lib is not None:
                return _run_native(lib, reference, current, block_size=block_size,
                                   search_range=search_range, threads=_default_threads(),
                                   early_exit=True, mode=_native.MODE_AUTO)
            if not _warned_unavailable:
                _warned_unavailable = True
                warnings.warn(
                    f"nvc.video: native motion search unavailable ({_native.last_error()}); "
                    f"using the PyTorch reference, which is roughly 25x slower on a CPU. "
                    f"Results are identical.", RuntimeWarning, stacklevel=2)
    return estimate_block_motion_torch(reference, current, block_size=block_size,
                                       search_range=search_range)


@torch.no_grad()
def estimate_block_motion_torch(reference: torch.Tensor, current: torch.Tensor, *,
                                block_size: int = DEFAULT_BLOCK_SIZE,
                                search_range: int = DEFAULT_SEARCH_RANGE) -> torch.Tensor:
    """The PyTorch reference implementation, unchanged from `scripts/m10h_motion_compensation.py`
    apart from sharing its input checks. This is the definition the native kernel is tested
    against, and the only implementation on GPUs and for non-float32 inputs."""
    _validate_motion_inputs(reference, current, block_size)
    _, _, height, width = reference.shape

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
