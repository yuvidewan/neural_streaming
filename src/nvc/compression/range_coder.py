"""Integer arithmetic coding, implemented from first principles.

No compression library is called here. The symbol-to-bitstream mapping is
written out explicitly so it can be explained and defended academically,
which is the whole point of choosing arithmetic coding over a black-box
`compress()` call.

THE ALGORITHM
-------------
Arithmetic coding represents an entire symbol sequence as a single number
inside [0, 1). Each symbol narrows the current interval in proportion to its
probability; likely symbols shrink it little (few bits), unlikely symbols
shrink it a lot (many bits). The final bitstream is any number inside the
last interval.

Doing that in exact arithmetic would need unbounded precision, so the
interval is held in fixed-point integers [low, high] within a 32-bit space
and renormalized as it narrows:

- If the interval sits entirely in the lower half, its leading bit is
  settled as 0: emit it and rescale.
- If entirely in the upper half, the leading bit is 1: emit it and rescale.
- Otherwise, if it straddles the midpoint but sits inside the middle half
  ([1/4, 3/4)), no leading bit is decided yet, but the interval is still too
  narrow. This is the classic UNDERFLOW case: rescale around the midpoint
  and remember one "pending" bit, to be emitted with the opposite polarity
  once the next real bit is known.

The decoder mirrors every one of these steps, so the two stay in lockstep.

WHY INTEGER FREQUENCIES
-----------------------
Interval subdivision uses integer cumulative frequencies rather than floats.
Encoder and decoder must compute bit-identical interval boundaries; floats
would risk platform-dependent rounding and desynchronize the decode. The
constraint `total <= MAX_TOTAL_FREQUENCY` (2**30) guarantees the
`span * cumulative` products cannot overflow the interval arithmetic.

PER-POSITION MODEL SELECTION
----------------------------
Each symbol names its own frequency table via `table_index`. This project
uses one table per latent channel, and because symbols are coded in a fixed
channel-major order, the decoder derives the same table indices without any
side information in the bitstream.

C BACKEND - THE ONLY RUNTIME PATH (Milestone 8B/8C)
----------------------------------------------------
The per-symbol loop above was originally pure Python, measured at
~100 ms/frame on real latent-sized data (16,384 symbols) - a tight,
sequential, per-element loop, exactly the pattern where Python's
per-iteration interpreter overhead dominates. It has been rewritten in C
(`_native/range_coder.c`, structurally identical to the original Python,
loaded via ctypes with automatic on-demand compilation - see
`_native/__init__.py`) and measured ~40x faster with byte-identical output
(see README, "Milestone 8B").

This build has deliberately **removed the Python fallback**: encode_symbols
and decode_symbols below always use the C backend, and raise a clear,
actionable RuntimeError (via `ensure_native_backend()`) if it cannot be
loaded or built, rather than silently degrading to the slow path. The
original Python implementation is kept below, in full, **commented out**
- not deleted, not live, not reachable from any code path - preserved
purely for reference/comparison/teaching (e.g. reading the two side by
side, or reinstating a fallback later if that's ever wanted). If you need
to actually run it, uncomment `_encode_symbols_python`/
`_decode_symbols_python` and the `_BitWriter`/`_BitReader` classes they
depend on, and the `bisect_right` import above them.
"""

from __future__ import annotations

import ctypes

import numpy as np

from . import _native

PRECISION = 32
WHOLE = 1 << PRECISION
HALF = WHOLE >> 1
QUARTER = WHOLE >> 2
THREE_QUARTERS = 3 * QUARTER

# Interval arithmetic stays exact only while the frequency total is below
# a quarter of the coding range.
MAX_TOTAL_FREQUENCY = QUARTER

ARITHMETIC_CODER_ID = 1
ARITHMETIC_CODER_NAME = "static_arithmetic_v1"


# --- Python reference implementation - COMMENTED OUT, NOT A RUNTIME PATH ---
#
# Kept for reference/comparison only (see the module docstring's "C BACKEND"
# section for why). Nothing below this point, down to "--- end of retired
# Python reference ---", executes in this build.
#
# from bisect import bisect_right
#
#
# class _BitWriter:
#     """Accumulates bits MSB-first into bytes."""
#
#     def __init__(self) -> None:
#         self._data = bytearray()
#         self._current = 0
#         self._filled = 0
#
#     def write(self, bit: int) -> None:
#         self._current = (self._current << 1) | bit
#         self._filled += 1
#         if self._filled == 8:
#             self._data.append(self._current)
#             self._current = 0
#             self._filled = 0
#
#     def finish(self) -> bytes:
#         """Zero-pad the final partial byte and return the payload."""
#         if self._filled:
#             self._data.append(self._current << (8 - self._filled))
#             self._current = 0
#             self._filled = 0
#         return bytes(self._data)
#
#
# class _BitReader:
#     """Reads bits MSB-first, yielding 0 forever past the end of the data.
#
#     Reading zeros past the end is deliberate and standard: the encoder's
#     termination sequence already pins the final value inside the correct
#     interval, so any trailing bits the decoder still asks for cannot change
#     the symbols it produces.
#     """
#
#     def __init__(self, data: bytes) -> None:
#         self._data = data
#         self._position = 0
#         self._bit = 0
#
#     def read(self) -> int:
#         if self._position >= len(self._data):
#             return 0
#         bit = (self._data[self._position] >> (7 - self._bit)) & 1
#         self._bit += 1
#         if self._bit == 8:
#             self._bit = 0
#             self._position += 1
#         return bit
#
#
# def _encode_symbols_python(
#     symbols: np.ndarray, cumulative: np.ndarray, table_index: np.ndarray
# ) -> bytes:
#     """Arithmetic-encode `symbols` into a byte payload. Assumes validated,
#     int64, correctly-shaped inputs.
#     """
#     # Python ints in plain lists beat numpy scalar indexing in this
#     # inherently sequential inner loop.
#     tables = [row.tolist() for row in cumulative]
#     symbol_list = symbols.tolist()
#     index_list = table_index.tolist()
#
#     writer = _BitWriter()
#     low = 0
#     high = WHOLE - 1
#     pending = 0
#
#     for symbol, table in zip(symbol_list, index_list):
#         cum = tables[table]
#         total = cum[-1]
#         span = high - low + 1
#
#         high = low + (span * cum[symbol + 1]) // total - 1
#         low = low + (span * cum[symbol]) // total
#
#         while True:
#             if high < HALF:
#                 writer.write(0)
#                 for _ in range(pending):
#                     writer.write(1)
#                 pending = 0
#             elif low >= HALF:
#                 writer.write(1)
#                 for _ in range(pending):
#                     writer.write(0)
#                 pending = 0
#                 low -= HALF
#                 high -= HALF
#             elif low >= QUARTER and high < THREE_QUARTERS:
#                 # Underflow: leading bit undecided, interval too narrow.
#                 pending += 1
#                 low -= QUARTER
#                 high -= QUARTER
#             else:
#                 break
#             low <<= 1
#             high = (high << 1) | 1
#
#     # Termination: emit enough bits to pin a value inside [low, high].
#     pending += 1
#     if low < QUARTER:
#         writer.write(0)
#         for _ in range(pending):
#             writer.write(1)
#     else:
#         writer.write(1)
#         for _ in range(pending):
#             writer.write(0)
#
#     return writer.finish()
#
#
# def _decode_symbols_python(
#     payload: bytes, num_symbols_to_decode: int, cumulative: np.ndarray, table_index: np.ndarray
# ) -> np.ndarray:
#     """Inverse of `_encode_symbols_python`. Assumes validated inputs."""
#     tables = [row.tolist() for row in cumulative]
#     index_list = table_index.tolist()
#
#     reader = _BitReader(payload)
#     low = 0
#     high = WHOLE - 1
#     value = 0
#     for _ in range(PRECISION):
#         value = (value << 1) | reader.read()
#
#     decoded = np.empty(num_symbols_to_decode, dtype=np.int64)
#
#     for position, table in enumerate(index_list):
#         cum = tables[table]
#         total = cum[-1]
#         span = high - low + 1
#
#         # Which cumulative-frequency slot does the current value fall into?
#         scaled = ((value - low + 1) * total - 1) // span
#         symbol = bisect_right(cum, scaled) - 1
#         decoded[position] = symbol
#
#         high = low + (span * cum[symbol + 1]) // total - 1
#         low = low + (span * cum[symbol]) // total
#
#         # Mirror the encoder's renormalization exactly.
#         while True:
#             if high < HALF:
#                 pass
#             elif low >= HALF:
#                 value -= HALF
#                 low -= HALF
#                 high -= HALF
#             elif low >= QUARTER and high < THREE_QUARTERS:
#                 value -= QUARTER
#                 low -= QUARTER
#                 high -= QUARTER
#             else:
#                 break
#             low <<= 1
#             high = (high << 1) | 1
#             value = (value << 1) | reader.read()
#
#     return decoded
#
# --- end of retired Python reference ---


def _validate_inputs(symbols: np.ndarray, cumulative: np.ndarray, table_index: np.ndarray) -> None:
    if cumulative.ndim != 2:
        raise ValueError(f"cumulative must be 2D [tables, symbols+1], got {cumulative.shape}")
    if len(symbols) != len(table_index):
        raise ValueError(
            f"symbols and table_index must be the same length, got "
            f"{len(symbols)} and {len(table_index)}"
        )
    totals = cumulative[:, -1]
    if np.any(totals > MAX_TOTAL_FREQUENCY):
        raise ValueError(
            f"frequency total {int(totals.max())} exceeds MAX_TOTAL_FREQUENCY "
            f"({MAX_TOTAL_FREQUENCY}); interval arithmetic would overflow"
        )
    if np.any(totals <= 0):
        raise ValueError("every frequency table must have a positive total")


# --- Native backend safeguard -----------------------------------------


def ensure_native_backend() -> None:
    """Raise a clear, actionable RuntimeError if the C range coder is not
    available. Called by `encode_symbols`/`decode_symbols` before every
    use - there is no fallback to degrade to, so a missing/broken native
    library must fail loudly here, not surface later as a confusing
    AttributeError/None-call deep inside ctypes plumbing.
    """
    if _native.load() is None:
        raise RuntimeError(
            "The native C range coder is not available, and this build has no "
            "Python fallback for arithmetic coding (by design - see "
            "range_coder.py's module docstring). "
            f"Reason: {_native.last_error()}. "
            "Install a C compiler (gcc/clang/cc) on PATH and retry - the library "
            "is compiled automatically on first use once one is found."
        )


# --- C backend (ctypes bridge into _native/range_coder.c) ------------------


def _encode_symbols_c(symbols: np.ndarray, cumulative: np.ndarray, table_index: np.ndarray) -> bytes:
    lib = _native.load()
    symbols_arr = np.ascontiguousarray(symbols, dtype=np.int64)
    cumulative_arr = np.ascontiguousarray(cumulative, dtype=np.int64)
    table_index_arr = np.ascontiguousarray(table_index, dtype=np.int64)
    table_width = cumulative_arr.shape[1]

    out_ptr = ctypes.POINTER(ctypes.c_uint8)()
    out_len = ctypes.c_int64()

    status = lib.rc_encode(
        symbols_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), ctypes.c_int64(len(symbols_arr)),
        cumulative_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), ctypes.c_int64(table_width),
        table_index_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(out_ptr), ctypes.byref(out_len),
    )
    if status != 0:
        raise MemoryError(
            "Native range coder failed to allocate its output buffer "
            "(rc_encode returned a non-zero status) - the process is likely out "
            "of memory."
        )
    try:
        return ctypes.string_at(out_ptr, out_len.value)
    finally:
        lib.rc_free(out_ptr)


def _decode_symbols_c(
    payload: bytes, num_symbols_to_decode: int, cumulative: np.ndarray, table_index: np.ndarray
) -> np.ndarray:
    lib = _native.load()
    cumulative_arr = np.ascontiguousarray(cumulative, dtype=np.int64)
    table_index_arr = np.ascontiguousarray(table_index, dtype=np.int64)
    table_width = cumulative_arr.shape[1]
    payload_arr = np.frombuffer(payload, dtype=np.uint8)
    out = np.empty(num_symbols_to_decode, dtype=np.int64)

    payload_ptr = (
        payload_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)) if payload_arr.size else None
    )
    status = lib.rc_decode(
        payload_ptr, ctypes.c_int64(len(payload)),
        ctypes.c_int64(num_symbols_to_decode),
        cumulative_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), ctypes.c_int64(table_width),
        table_index_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if status != 0:
        raise RuntimeError(
            f"Native range coder rc_decode rejected its inputs (status {status}) - "
            "this indicates a bug in the calling code, not bad input data (which "
            "encode_symbols/decode_symbols validate before ever reaching here)."
        )
    return out


# --- Public API --------------------------------------------------------


def encode_symbols(
    symbols: np.ndarray, cumulative: np.ndarray, table_index: np.ndarray
) -> bytes:
    """Arithmetic-encode `symbols` into a byte payload.

    `cumulative[t]` is the cumulative frequency array for table t (length
    num_symbols + 1, starting at 0). `table_index[i]` selects the table for
    symbols[i].

    Always uses the native C backend - see this module's docstring. Raises
    RuntimeError (via `ensure_native_backend()`) if it isn't available;
    there is no slower fallback to silently drop into.
    """
    symbols = np.asarray(symbols, dtype=np.int64)
    table_index = np.asarray(table_index, dtype=np.int64)
    cumulative = np.asarray(cumulative, dtype=np.int64)
    _validate_inputs(symbols, cumulative, table_index)

    if len(symbols) == 0:
        raise ValueError("encode_symbols: refusing to encode an empty symbol sequence")

    num_symbols = cumulative.shape[1] - 1
    if symbols.min() < 0 or symbols.max() >= num_symbols:
        raise ValueError(
            f"symbols must lie in [0, {num_symbols}), got [{symbols.min()}, {symbols.max()}]"
        )

    ensure_native_backend()
    return _encode_symbols_c(symbols, cumulative, table_index)


def decode_symbols(
    payload: bytes, num_symbols_to_decode: int, cumulative: np.ndarray, table_index: np.ndarray
) -> np.ndarray:
    """Inverse of `encode_symbols`; returns the original symbol array.

    Always uses the native C backend - see this module's docstring. Raises
    RuntimeError (via `ensure_native_backend()`) if it isn't available;
    there is no slower fallback to silently drop into.
    """
    if num_symbols_to_decode <= 0:
        raise ValueError(
            f"num_symbols_to_decode must be positive, got {num_symbols_to_decode}"
        )
    table_index = np.asarray(table_index, dtype=np.int64)
    cumulative = np.asarray(cumulative, dtype=np.int64)
    if len(table_index) != num_symbols_to_decode:
        raise ValueError(
            f"table_index has {len(table_index)} entries but "
            f"{num_symbols_to_decode} symbols were requested"
        )
    _validate_inputs(np.zeros(num_symbols_to_decode, dtype=np.int64), cumulative, table_index)

    ensure_native_backend()
    return _decode_symbols_c(payload, num_symbols_to_decode, cumulative, table_index)


# --- Resumable decoder (Milestone 12) -----------------------------------


class ResumableDecoder:
    """Decode a payload's symbols in successive GROUPS, resuming exactly
    where the previous group left off, instead of re-decoding from byte 0.

    `decode_symbols` above is stateless: every call walks the payload from
    its first bit. M11's channel-autoregressive decoder needed each channel
    GROUP's probability tables to depend on the groups decoded so far in the
    SAME frame, and the only way to get that through a stateless decoder was
    to call `decode_symbols(payload, count, ...)` again with a longer prefix
    for every group - re-walking every bit already consumed each time. For
    G groups covering N symbols that is O(N*G) bit-level work instead of
    O(N), and it is what made a context needing many groups (one per
    channel, let alone one per spatial position) impractically slow.

    This class exposes the decoder's state - the bit position and the
    interval integers low/high/value - as a handle that persists between
    calls:

        decoder = ResumableDecoder(payload)
        first_group  = decoder.decode_group(cumulative_1, table_index_1)
        second_group = decoder.decode_group(cumulative_2, table_index_2)
        decoder.close()

    or as a context manager:

        with ResumableDecoder(payload) as decoder:
            symbols = decoder.decode_group(cumulative, table_index)

    Each `decode_group` call decodes exactly `len(table_index)` symbols
    starting immediately after the previous call's last symbol (or the
    start of the payload, for the first call) - group boundaries may be any
    size, including 1. `decode_symbols(payload, n, cumulative, table_index)`
    in one call is exactly equivalent to
    `ResumableDecoder(payload).decode_group(cumulative, table_index)` in
    one call decoding all n symbols: `decode_symbols`'s C implementation
    (`rc_decode`) is now itself a thin wrapper over the same
    open/decode/close primitives this class calls, so the two paths are
    identical by construction, not merely by agreement in testing.

    Always uses the native C backend - see this module's docstring. Raises
    RuntimeError (via `ensure_native_backend()`) if it isn't available.
    """

    def __init__(self, payload: bytes) -> None:
        ensure_native_backend()
        self._lib = _native.load()
        # The C reader holds a raw pointer into this buffer for the handle's
        # whole lifetime - keep both the bytes and the numpy view over them
        # alive on the Python side so nothing is garbage-collected under it.
        self._payload = bytes(payload)
        self._payload_array = np.frombuffer(self._payload, dtype=np.uint8)
        payload_ptr = (
            self._payload_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            if self._payload_array.size else None
        )
        handle = self._lib.rc_decoder_open(payload_ptr, ctypes.c_int64(len(self._payload)))
        if not handle:
            raise MemoryError(
                "Native range coder failed to allocate a resumable decoder state "
                "(rc_decoder_open returned NULL) - the process is likely out of memory."
            )
        self._handle = handle
        self._closed = False
        self.symbols_decoded = 0

    def decode_group(self, cumulative: np.ndarray, table_index: np.ndarray) -> np.ndarray:
        """Decode the next `len(table_index)` symbols and advance the state.

        `cumulative`/`table_index` follow the same convention as
        `decode_symbols`: `cumulative[t]` is table t's cumulative frequency
        row, and `table_index[i]` selects the table for the i-th symbol OF
        THIS GROUP (not of the whole stream).
        """
        if self._closed:
            raise RuntimeError("decode_group called on a closed ResumableDecoder")
        table_index = np.asarray(table_index, dtype=np.int64)
        cumulative = np.asarray(cumulative, dtype=np.int64)
        n = len(table_index)
        if n == 0:
            raise ValueError("decode_group: refusing to decode an empty group")
        _validate_inputs(np.zeros(n, dtype=np.int64), cumulative, table_index)

        cumulative_arr = np.ascontiguousarray(cumulative, dtype=np.int64)
        table_index_arr = np.ascontiguousarray(table_index, dtype=np.int64)
        table_width = cumulative_arr.shape[1]
        out = np.empty(n, dtype=np.int64)

        status = self._lib.rc_decoder_decode(
            self._handle, ctypes.c_int64(n),
            cumulative_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), ctypes.c_int64(table_width),
            table_index_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        )
        if status != 0:
            raise RuntimeError(
                f"Native resumable range coder rejected its inputs (status {status}) - "
                "this indicates a bug in the calling code, not bad input data."
            )
        self.symbols_decoded += n
        return out

    def close(self) -> None:
        """Free the native decoder state. Safe to call more than once."""
        if not self._closed:
            self._lib.rc_decoder_close(self._handle)
            self._closed = True

    def __enter__(self) -> "ResumableDecoder":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort: __init__ can raise before self._closed is set (e.g.
        # ensure_native_backend() failing), so this must not assume any
        # attribute exists.
        try:
            self.close()
        except Exception:
            pass
