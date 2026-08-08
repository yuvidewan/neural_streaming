"""Empirical (static, non-learned) entropy model over quantized symbols.

Estimates P(symbol) by counting symbol occurrences in calibration data. One
independent frequency table per latent channel: the Milestone 5 analysis
found channels with visibly different distributions, and symbols are coded
in channel-major order so the decoder always knows which table applies at
each position without any side information.

This is deliberately the simplest thing that works. No hyperprior, no
context model, no autoregressive conditioning, nothing learned - those are
later milestones. The model is fixed at calibration time and shared by
encoder and decoder.

TWO KINDS OF PROBABILITY
------------------------
- Float probabilities, used only to report empirical entropy in bits/symbol.
- Integer frequencies summing to exactly TOTAL_FREQUENCY (a power of two),
  which is what the arithmetic coder actually consumes. Integer frequencies
  are required because the coder's interval arithmetic must be exactly
  reproducible on the decode side; floats would drift.

The rounding from the first to the second is a real (small) source of
inefficiency, reported as "probability quantization" in the efficiency
breakdown.

ZERO-PROBABILITY SAFETY
-----------------------
A symbol with probability 0 is unencodable - the coder would be handed a
zero-width interval. Calibration data cannot be assumed to contain every
symbol that will ever occur, so:

1. Add-one (Laplace) smoothing is applied to raw counts, and
2. every integer frequency is floored at MIN_FREQUENCY = 1.

Together these guarantee every symbol in [0, 2**bits) stays encodable, at a
negligible cost in coding efficiency.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

# Frequencies sum to this per table. Must be a power of two well below the
# arithmetic coder's overflow bound (see range_coder.MAX_TOTAL_FREQUENCY).
TOTAL_FREQUENCY = 1 << 16

# Laplace smoothing added to every raw count before normalization.
SMOOTHING_COUNT = 1.0

# No symbol may ever have frequency 0; see "zero-probability safety" above.
MIN_FREQUENCY = 1

ENTROPY_MODEL_VERSION = 1
ENTROPY_MODEL_NAME = "empirical_per_channel_static_v1"


class EmpiricalEntropyModel:
    """Static per-channel symbol frequency tables for arithmetic coding."""

    def __init__(self, frequencies: np.ndarray, bits: int) -> None:
        """`frequencies` is an integer array of shape [num_tables, 2**bits]."""
        frequencies = np.asarray(frequencies, dtype=np.int64)
        num_symbols = 2 ** bits

        if frequencies.ndim != 2 or frequencies.shape[1] != num_symbols:
            raise ValueError(
                f"frequencies must have shape [num_tables, {num_symbols}] for {bits}-bit "
                f"symbols, got {frequencies.shape}"
            )
        if (frequencies < MIN_FREQUENCY).any():
            raise ValueError(
                f"every frequency must be >= {MIN_FREQUENCY} so no symbol is unencodable"
            )
        row_totals = frequencies.sum(axis=1)
        if not np.all(row_totals == TOTAL_FREQUENCY):
            raise ValueError(
                f"every table must sum to exactly {TOTAL_FREQUENCY}, got totals "
                f"{sorted(set(row_totals.tolist()))}"
            )

        self.bits = bits
        self.num_symbols = num_symbols
        self.frequencies = frequencies
        # cumulative[t, s] = sum of frequencies[t, :s]; length num_symbols + 1.
        self.cumulative = np.zeros((frequencies.shape[0], num_symbols + 1), dtype=np.int64)
        np.cumsum(frequencies, axis=1, out=self.cumulative[:, 1:])

    @property
    def num_tables(self) -> int:
        return self.frequencies.shape[0]

    @classmethod
    def from_symbols(
        cls, symbols: np.ndarray, *, bits: int, num_tables: int
    ) -> "EmpiricalEntropyModel":
        """Build a model by counting symbols.

        `symbols` is a [N, num_tables, ...] integer array, or any shape whose
        axis 1 is the table (channel) axis.
        """
        num_symbols = 2 ** bits
        symbols = np.asarray(symbols)
        if symbols.ndim < 2 or symbols.shape[1] != num_tables:
            raise ValueError(
                f"symbols axis 1 must be the table axis of size {num_tables}, "
                f"got shape {symbols.shape}"
            )
        if symbols.min() < 0 or symbols.max() >= num_symbols:
            raise ValueError(
                f"symbols must lie in [0, {num_symbols}), got "
                f"[{symbols.min()}, {symbols.max()}]"
            )

        per_table = np.moveaxis(symbols, 1, 0).reshape(num_tables, -1)
        counts = np.stack([
            np.bincount(row, minlength=num_symbols) for row in per_table
        ]).astype(np.float64)

        return cls(_counts_to_frequencies(counts), bits=bits)

    def probabilities(self) -> np.ndarray:
        """Float probabilities implied by the integer frequency tables."""
        return self.frequencies / TOTAL_FREQUENCY

    def entropy_bits_per_symbol(self) -> np.ndarray:
        """Shannon entropy of each table, in bits/symbol."""
        probabilities = self.probabilities()
        return -np.sum(probabilities * np.log2(probabilities), axis=1)

    def expected_bits(self, symbols: np.ndarray, table_index: np.ndarray) -> float:
        """Ideal code length for `symbols` under this model, in bits.

        This is the information-theoretic target the arithmetic coder
        approaches from above - not a size it can beat.
        """
        probabilities = self.probabilities()
        return float(-np.sum(np.log2(probabilities[table_index, symbols])))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": ENTROPY_MODEL_NAME,
            "version": ENTROPY_MODEL_VERSION,
            "bits": self.bits,
            "num_symbols": self.num_symbols,
            "num_tables": self.num_tables,
            "total_frequency": TOTAL_FREQUENCY,
            "min_frequency": MIN_FREQUENCY,
            "smoothing_count": SMOOTHING_COUNT,
            "frequencies": self.frequencies.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmpiricalEntropyModel":
        if data.get("version") != ENTROPY_MODEL_VERSION:
            raise ValueError(
                f"Unsupported entropy model version {data.get('version')!r}; "
                f"this build understands version {ENTROPY_MODEL_VERSION}"
            )
        return cls(np.array(data["frequencies"], dtype=np.int64), bits=int(data["bits"]))

    def model_id(self) -> bytes:
        """8-byte identity of this model, stored in the .nvc header.

        Lets a decoder detect that it loaded a different calibration than the
        one used to encode, instead of silently emitting garbage symbols.
        """
        digest = hashlib.sha256()
        digest.update(json.dumps({
            "name": ENTROPY_MODEL_NAME,
            "version": ENTROPY_MODEL_VERSION,
            "bits": self.bits,
            "frequencies": self.frequencies.tolist(),
        }, sort_keys=True).encode("utf-8"))
        return digest.digest()[:8]


def _counts_to_frequencies(counts: np.ndarray) -> np.ndarray:
    """Turn raw counts into integer frequencies summing to TOTAL_FREQUENCY.

    Applies Laplace smoothing, floors every entry at MIN_FREQUENCY, then
    repairs the rounding residual so each row sums exactly - the arithmetic
    coder requires an exact total, not an approximate one.
    """
    smoothed = counts.astype(np.float64) + SMOOTHING_COUNT
    probabilities = smoothed / smoothed.sum(axis=1, keepdims=True)

    frequencies = np.maximum(
        np.floor(probabilities * TOTAL_FREQUENCY).astype(np.int64), MIN_FREQUENCY
    )

    for row in frequencies:
        residual = TOTAL_FREQUENCY - int(row.sum())
        order = np.argsort(-row)

        if residual > 0:
            # Hand the leftover counts to the most probable symbols, where
            # they cost the least relative accuracy.
            for offset in range(residual):
                row[order[offset % len(order)]] += 1
        elif residual < 0:
            # Reclaim from the largest bins first, never dropping below the
            # floor. Flooring rare symbols up to MIN_FREQUENCY can push the
            # row well over TOTAL_FREQUENCY on a very skewed distribution, so
            # this must take as much as each bin can spare rather than one
            # unit per bin.
            deficit = -residual
            for index in order:
                if deficit == 0:
                    break
                available = int(row[index]) - MIN_FREQUENCY
                if available <= 0:
                    continue
                taken = min(available, deficit)
                row[index] -= taken
                deficit -= taken
            if deficit != 0:
                raise ValueError(
                    f"Cannot normalize frequency table: {len(row)} symbols each need at "
                    f"least {MIN_FREQUENCY}, which exceeds TOTAL_FREQUENCY={TOTAL_FREQUENCY}"
                )

    return frequencies


def empirical_entropy(symbols: np.ndarray, num_symbols: int) -> float:
    """Order-0 empirical entropy of a symbol sequence, in bits/symbol.

    Measured directly from `symbols` with no smoothing - this describes the
    data itself, independent of any model fitted to other data.
    """
    counts = np.bincount(np.asarray(symbols).ravel(), minlength=num_symbols)
    total = counts.sum()
    if total == 0:
        raise ValueError("empirical_entropy: no symbols provided")
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def symbol_distribution_summary(symbols: np.ndarray, num_symbols: int) -> dict[str, Any]:
    """Frequency/probability/entropy summary of an observed symbol sequence."""
    flat = np.asarray(symbols).ravel()
    counts = np.bincount(flat, minlength=num_symbols)
    total = int(counts.sum())
    entropy = empirical_entropy(flat, num_symbols)

    return {
        "num_symbols_possible": num_symbols,
        "num_unique_symbols_observed": int((counts > 0).sum()),
        "total_symbols": total,
        "counts": counts.tolist(),
        "probabilities": (counts / total).tolist(),
        "empirical_entropy_bits_per_symbol": entropy,
        "fixed_width_bits_per_symbol": float(math.log2(num_symbols)),
        "entropy_headroom_bits_per_symbol": float(math.log2(num_symbols)) - entropy,
    }
