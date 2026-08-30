"""Synthetic-data regression test for the core claim in parallel_entropy_poc.py:
splitting one combined arithmetic-coded stream into 64 independent
per-channel streams round-trips bit-exact, for any valid entropy model -
not just the one real DAVIS frame the PoC's __main__ happens to use.

Kept separate from the main tests/ suite (see TESTING.md) since this is
exploratory hardware-architecture work, not src/nvc or scripts/*.py, but
follows the same synthetic-only, no-GPU, no-network philosophy so it stays
fast and doesn't depend on real checkpoints/calibrations being on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parallel_entropy_poc import _decode_nvc_hw, _nvc_hw_per_channel_streams  # noqa: E402

from nvc.compression.entropy_model import EmpiricalEntropyModel, _counts_to_frequencies  # noqa: E402


def _make_entropy_model(num_channels: int, bits: int, seed: int) -> EmpiricalEntropyModel:
    rng = np.random.default_rng(seed)
    num_symbols = 2 ** bits
    counts = rng.integers(1, 200, size=(num_channels, num_symbols)).astype(np.float64)
    return EmpiricalEntropyModel(_counts_to_frequencies(counts), bits=bits)


def test_per_channel_split_round_trips_bit_exact():
    channels, height, width, bits = 16, 4, 4, 8
    model = _make_entropy_model(channels, bits, seed=0)
    rng = np.random.default_rng(1)
    symbols = rng.integers(0, 2 ** bits, size=channels * height * width).astype(np.int64)

    combined, per_channel_payloads, _ = _nvc_hw_per_channel_streams(symbols, model, channels, height, width)
    decoded = _decode_nvc_hw(combined, model, channels, height, width)

    assert np.array_equal(decoded, symbols)
    assert len(per_channel_payloads) == channels


def test_per_channel_split_works_at_every_project_bit_depth():
    # 4/6/8-bit are the three operating points this whole codec is
    # evaluated at (see MILESTONE_8_RESULTS.md) - the split must hold at
    # all three, not just 8-bit.
    channels, height, width = 8, 4, 4
    for bits in (4, 6, 8):
        model = _make_entropy_model(channels, bits, seed=bits)
        rng = np.random.default_rng(bits + 100)
        symbols = rng.integers(0, 2 ** bits, size=channels * height * width).astype(np.int64)

        combined, _, _ = _nvc_hw_per_channel_streams(symbols, model, channels, height, width)
        decoded = _decode_nvc_hw(combined, model, channels, height, width)

        assert np.array_equal(decoded, symbols), f"round-trip failed at {bits}-bit"


def test_per_channel_split_overhead_is_small():
    # The one real cost of independence: each of the 64 streams pays its
    # own flush + a 2-byte length prefix. For a realistic-sized payload
    # this should stay a small fraction of the total, not dominate it.
    channels, height, width, bits = 64, 16, 16, 8
    model = _make_entropy_model(channels, bits, seed=2)
    rng = np.random.default_rng(3)
    # Skewed (not uniform) symbols, like real calibrated latents - a
    # uniform distribution would make every stream incompressible and
    # exaggerate the relative overhead unrealistically.
    symbols = np.clip(rng.normal(128, 20, size=channels * height * width), 0, 255).astype(np.int64)

    combined, per_channel_payloads, _ = _nvc_hw_per_channel_streams(symbols, model, channels, height, width)
    payload_only_bytes = sum(len(p) for p in per_channel_payloads)
    length_prefix_bytes = 2 * channels

    assert length_prefix_bytes / len(combined) < 0.05  # prefixes stay under 5% of the total
    assert payload_only_bytes > 0
