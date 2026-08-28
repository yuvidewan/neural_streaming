"""Benchmark for the arithmetic coder (encode_symbols/decode_symbols).

Standalone from Milestone 7's RD benchmark harness - this measures the
coder in isolation, not the full codec, so a Python-vs-C migration can be
timed apples-to-apples before/after without any other pipeline stage
(model forward pass, quantization, file I/O) blurring the number.

Data is synthetic but shaped exactly like real production usage: one
frame's worth of symbols (64 channels x 16x16 = 16,384 symbols, matching
BaselineAutoencoder's default latent shape) with a realistic per-channel
skewed distribution (Gaussian-ish, not uniform noise), built the same way
codec.py builds it - via EmpiricalEntropyModel + channel_table_index.

Usage:
    python scripts\\benchmark_range_coder.py [--reps 30] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from nvc.compression.codec import channel_table_index
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.compression.range_coder import decode_symbols, encode_symbols

LATENT_CHANNELS = 64
LATENT_H = 16
LATENT_W = 16
BITS = 8
NUM_SYMBOLS_PER_FRAME = LATENT_CHANNELS * LATENT_H * LATENT_W  # 16,384 - one frame


def _build_frame(seed: int) -> tuple[np.ndarray, np.ndarray, EmpiricalEntropyModel]:
    """One frame's worth of realistic (Gaussian-ish, per-channel) symbols."""
    rng = np.random.default_rng(seed)
    num_symbols = 2 ** BITS
    center = num_symbols / 2
    # Each channel gets its own mean/spread, like real trained latents do -
    # calibration is built from a larger synthetic calibration set, then the
    # single frame benchmarked is drawn fresh from the same distributions.
    means = rng.uniform(center - 40, center + 40, size=LATENT_CHANNELS)
    stds = rng.uniform(8, 30, size=LATENT_CHANNELS)

    calibration = np.stack([
        np.clip(rng.normal(m, s, size=4000), 0, num_symbols - 1).astype(np.int64)
        for m, s in zip(means, stds)
    ])  # [channels, 4000]
    model = EmpiricalEntropyModel.from_symbols(
        calibration.T[:, :, None], bits=BITS, num_tables=LATENT_CHANNELS
    )

    frame = np.stack([
        np.clip(rng.normal(m, s, size=LATENT_H * LATENT_W), 0, num_symbols - 1).astype(np.int64)
        for m, s in zip(means, stds)
    ]).reshape(-1)  # C-major, matches channel_table_index's ordering
    table_index = channel_table_index(LATENT_CHANNELS, LATENT_H, LATENT_W)
    return frame, table_index, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=30, help="Timed repetitions (default 30).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--label", default="python",
        help="Tag recorded in the output JSON, e.g. 'python' or 'c' (default: python).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Where to write the JSON results (default: outputs/benchmarks/range_coder_<label>.json)",
    )
    args = parser.parse_args()

    symbols, table_index, model = _build_frame(args.seed)
    print(f"Frame: {NUM_SYMBOLS_PER_FRAME} symbols ({LATENT_CHANNELS} channels x "
          f"{LATENT_H}x{LATENT_W}), {BITS}-bit, {model.num_tables} per-channel tables.")

    # Warmup (not timed) - JIT-free here (CPython), but keeps first-call
    # effects like disk-cache/branch-predictor warmup out of the numbers.
    payload = encode_symbols(symbols, model.cumulative, table_index)
    _ = decode_symbols(payload, len(symbols), model.cumulative, table_index)
    print(f"Payload size: {len(payload)} bytes "
          f"({len(payload) * 8 / NUM_SYMBOLS_PER_FRAME:.3f} bits/symbol)")

    encode_times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        payload = encode_symbols(symbols, model.cumulative, table_index)
        encode_times.append(time.perf_counter() - t0)

    decode_times = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        decoded = decode_symbols(payload, len(symbols), model.cumulative, table_index)
        decode_times.append(time.perf_counter() - t0)

    assert np.array_equal(decoded, symbols), "round-trip mismatch - coder is not lossless!"

    def summarize(times: list[float]) -> dict:
        ms = [t * 1000 for t in times]
        return {
            "mean_ms": statistics.mean(ms),
            "median_ms": statistics.median(ms),
            "min_ms": min(ms),
            "max_ms": max(ms),
            "stdev_ms": statistics.stdev(ms) if len(ms) > 1 else 0.0,
            "per_symbol_us": statistics.mean(ms) * 1000 / NUM_SYMBOLS_PER_FRAME,
        }

    enc_stats = summarize(encode_times)
    dec_stats = summarize(decode_times)

    print(f"\n[{args.label}] encode: {enc_stats['mean_ms']:.3f} ms/frame "
          f"(median {enc_stats['median_ms']:.3f}, {enc_stats['per_symbol_us']:.4f} us/symbol)")
    print(f"[{args.label}] decode: {dec_stats['mean_ms']:.3f} ms/frame "
          f"(median {dec_stats['median_ms']:.3f}, {dec_stats['per_symbol_us']:.4f} us/symbol)")
    print(f"[{args.label}] round-trip verified lossless over {NUM_SYMBOLS_PER_FRAME} symbols.")

    out_path = args.out or Path("outputs/benchmarks") / f"range_coder_{args.label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "label": args.label,
        "seed": args.seed,
        "reps": args.reps,
        "num_symbols_per_frame": NUM_SYMBOLS_PER_FRAME,
        "latent_shape": [LATENT_CHANNELS, LATENT_H, LATENT_W],
        "bits": BITS,
        "payload_bytes": len(payload),
        "bits_per_symbol": len(payload) * 8 / NUM_SYMBOLS_PER_FRAME,
        "encode": enc_stats,
        "decode": dec_stats,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
