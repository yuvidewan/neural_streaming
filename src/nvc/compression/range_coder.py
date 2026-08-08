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
"""

from __future__ import annotations

from bisect import bisect_right

import numpy as np

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


class _BitWriter:
    """Accumulates bits MSB-first into bytes."""

    def __init__(self) -> None:
        self._data = bytearray()
        self._current = 0
        self._filled = 0

    def write(self, bit: int) -> None:
        self._current = (self._current << 1) | bit
        self._filled += 1
        if self._filled == 8:
            self._data.append(self._current)
            self._current = 0
            self._filled = 0

    def finish(self) -> bytes:
        """Zero-pad the final partial byte and return the payload."""
        if self._filled:
            self._data.append(self._current << (8 - self._filled))
            self._current = 0
            self._filled = 0
        return bytes(self._data)


class _BitReader:
    """Reads bits MSB-first, yielding 0 forever past the end of the data.

    Reading zeros past the end is deliberate and standard: the encoder's
    termination sequence already pins the final value inside the correct
    interval, so any trailing bits the decoder still asks for cannot change
    the symbols it produces.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._position = 0
        self._bit = 0

    def read(self) -> int:
        if self._position >= len(self._data):
            return 0
        bit = (self._data[self._position] >> (7 - self._bit)) & 1
        self._bit += 1
        if self._bit == 8:
            self._bit = 0
            self._position += 1
        return bit


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


def encode_symbols(
    symbols: np.ndarray, cumulative: np.ndarray, table_index: np.ndarray
) -> bytes:
    """Arithmetic-encode `symbols` into a byte payload.

    `cumulative[t]` is the cumulative frequency array for table t (length
    num_symbols + 1, starting at 0). `table_index[i]` selects the table for
    symbols[i].
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

    # Python ints in plain lists beat numpy scalar indexing in this
    # inherently sequential inner loop.
    tables = [row.tolist() for row in cumulative]
    symbol_list = symbols.tolist()
    index_list = table_index.tolist()

    writer = _BitWriter()
    low = 0
    high = WHOLE - 1
    pending = 0

    for symbol, table in zip(symbol_list, index_list):
        cum = tables[table]
        total = cum[-1]
        span = high - low + 1

        high = low + (span * cum[symbol + 1]) // total - 1
        low = low + (span * cum[symbol]) // total

        while True:
            if high < HALF:
                writer.write(0)
                for _ in range(pending):
                    writer.write(1)
                pending = 0
            elif low >= HALF:
                writer.write(1)
                for _ in range(pending):
                    writer.write(0)
                pending = 0
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:
                # Underflow: leading bit undecided, interval too narrow.
                pending += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1

    # Termination: emit enough bits to pin a value inside [low, high].
    pending += 1
    if low < QUARTER:
        writer.write(0)
        for _ in range(pending):
            writer.write(1)
    else:
        writer.write(1)
        for _ in range(pending):
            writer.write(0)

    return writer.finish()


def decode_symbols(
    payload: bytes, num_symbols_to_decode: int, cumulative: np.ndarray, table_index: np.ndarray
) -> np.ndarray:
    """Inverse of `encode_symbols`; returns the original symbol array."""
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

    tables = [row.tolist() for row in cumulative]
    index_list = table_index.tolist()

    reader = _BitReader(payload)
    low = 0
    high = WHOLE - 1
    value = 0
    for _ in range(PRECISION):
        value = (value << 1) | reader.read()

    decoded = np.empty(num_symbols_to_decode, dtype=np.int64)

    for position, table in enumerate(index_list):
        cum = tables[table]
        total = cum[-1]
        span = high - low + 1

        # Which cumulative-frequency slot does the current value fall into?
        scaled = ((value - low + 1) * total - 1) // span
        symbol = bisect_right(cum, scaled) - 1
        decoded[position] = symbol

        high = low + (span * cum[symbol + 1]) // total - 1
        low = low + (span * cum[symbol]) // total

        # Mirror the encoder's renormalization exactly.
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                value -= HALF
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:
                value -= QUARTER
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            value = (value << 1) | reader.read()

    return decoded
