"""Proof-of-concept for the accelerator's core architectural bet: splitting
the .nvc payload from ONE combined arithmetic-coded stream (today's format)
into 64 independent per-channel streams (one per latent channel, "NVC-HW"),
so N hardware lanes can encode/decode different channels simultaneously.

This does not simulate clock cycles or synthesize anything - it proves the
functional claim a hardware redesign absolutely must get right before any
RTL is worth writing: that splitting the stream this way still round-trips
bit-exact, using the project's own real, already-correct encode_symbols/
decode_symbols primitives (not a reimplementation), against a real trained
checkpoint's real calibration and a real DAVIS frame. It also measures the
one real cost of this change (per-stream flush overhead) so the trade-off
is quantified, not assumed.

Usage (from the project root, with .venv activated):

    python hardware\\parallel_entropy_poc.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nvc.compression.calibration import load_calibration
from nvc.compression.codec import channel_table_index, latent_to_symbols
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.compression.quantization import QuantizationParams
from nvc.compression.range_coder import decode_symbols, encode_symbols
from nvc.data.image_io import read_image_as_tensor
from nvc.evaluation.sequences import discover_sequences
from nvc.training import load_model_from_checkpoint

CHECKPOINT = Path("outputs/qat_combined/checkpoints_qat_noise/best.pt")
CALIBRATION = Path("outputs/calibration/qat_combined_noise.json")
MANIFEST = Path("data/processed/manifest.json")

# Physical hardware lane counts to project a closed-form latency estimate
# for. Every latent channel has exactly LATENT_H*LATENT_W symbols (spatial
# size is fixed by the trained model), so the 64 channels split perfectly
# evenly across any of these - no load-balancing logic needed in hardware.
LANE_COUNTS = [1, 8, 16, 32, 64]

# From reading src/nvc/compression/_native/range_coder.c: the per-symbol
# core update is `(span * cum) >> 16` - a multiply and a FIXED shift, no
# divider, because TOTAL_FREQUENCY = 1 << 16 by construction (see
# entropy_model.py). Encode is one such update; decode additionally has to
# search the cumulative table for the symbol whose range contains the
# scaled value. These are order-of-magnitude, not measured-in-silicon,
# cycle counts for a straightforward pipelined datapath - stated as
# assumptions, not results (see ARCHITECTURE.md, "Open risks").
ASSUMED_CYCLES_PER_SYMBOL_ENCODE = 2
ASSUMED_CYCLES_PER_SYMBOL_DECODE_BITS = {
    8: 8,  # binary search over <=256 entries: ceil(log2(256))
    6: 6,  # <=64 entries
    4: 4,  # <=16 entries
}
ASSUMED_CLOCK_HZ = 400_000_000  # a conservative, unremarkable FPGA fabric clock


def _load_real_symbols() -> tuple[np.ndarray, EmpiricalEntropyModel, int, int, int]:
    """Real checkpoint, real calibration, real DAVIS test frame - the exact
    quantities the actual .nvc encode path would produce, not synthetic
    stand-ins."""
    device = torch.device("cpu")
    model, _ = load_model_from_checkpoint(CHECKPOINT, device=device)

    sequences = discover_sequences(MANIFEST, split="test")
    frame_path = sequences[0].frame_paths[0]
    frame = read_image_as_tensor(frame_path).unsqueeze(0).to(device)

    with torch.no_grad():
        latent = model.encode(frame)

    calib_doc = load_calibration(CALIBRATION)
    params = QuantizationParams.from_dict(calib_doc["quantization"])
    entropy_model = EmpiricalEntropyModel.from_dict(calib_doc["entropy_model"])
    bits = calib_doc["quantization"]["bits"]

    symbols = latent_to_symbols(latent, params)  # flat, C-major, length C*H*W
    channels, height, width = latent.shape[1], latent.shape[2], latent.shape[3]
    print(f"Real frame: {frame_path.name}, latent {channels}x{height}x{width}, "
          f"{bits}-bit, {len(symbols)} symbols")
    return symbols, entropy_model, channels, height, width


def _baseline_single_stream(symbols: np.ndarray, model: EmpiricalEntropyModel,
                             channels: int, height: int, width: int) -> tuple[bytes, float]:
    """Today's format: ONE combined arithmetic-coded stream over all
    symbols, switching cumulative tables per symbol via table_index. This
    is inherently serial - decode_symbols has one (low, range) state
    shared across all 16,384 symbols."""
    table_index = channel_table_index(channels, height, width)
    encode_symbols(symbols, model.cumulative, table_index)  # warmup: first ctypes/DLL
    # call on Windows pays a one-off page-fault/binding cost unrelated to the
    # algorithm - benchmark_range_coder.py excludes it from its own timing
    # the same way, for the same reason.
    t0 = time.perf_counter()
    payload = encode_symbols(symbols, model.cumulative, table_index)
    elapsed = time.perf_counter() - t0
    return payload, elapsed


def _nvc_hw_per_channel_streams(
    symbols: np.ndarray, model: EmpiricalEntropyModel, channels: int, height: int, width: int,
) -> tuple[bytes, list[bytes], float]:
    """Proposed format: 64 independent per-channel streams, each with its
    own (low, range) state - mutually independent, so N hardware lanes can
    run different channels' streams concurrently with zero cross-channel
    dependency. Length-prefixed (2 bytes, big-endian) and concatenated into
    one combined payload so the rest of the .nvc container is untouched.
    """
    symbols_per_channel = height * width
    per_channel_symbols = symbols.reshape(channels, symbols_per_channel)
    zero_table_index = np.zeros(symbols_per_channel, dtype=np.int64)
    encode_symbols(per_channel_symbols[0], model.cumulative[0:1], zero_table_index)  # warmup, see above

    payloads = []
    t0 = time.perf_counter()
    for c in range(channels):
        single_table_cumulative = model.cumulative[c : c + 1]
        payload_c = encode_symbols(per_channel_symbols[c], single_table_cumulative, zero_table_index)
        if len(payload_c) > 0xFFFF:
            raise RuntimeError(f"channel {c} payload {len(payload_c)}B exceeds 16-bit length prefix")
        payloads.append(payload_c)
    elapsed = time.perf_counter() - t0

    combined = bytearray()
    for payload_c in payloads:
        combined += len(payload_c).to_bytes(2, "big")
        combined += payload_c
    return bytes(combined), payloads, elapsed


def _decode_nvc_hw(
    combined: bytes, model: EmpiricalEntropyModel, channels: int, height: int, width: int,
) -> np.ndarray:
    symbols_per_channel = height * width
    zero_table_index = np.zeros(symbols_per_channel, dtype=np.int64)

    decoded_channels = []
    offset = 0
    for c in range(channels):
        length = int.from_bytes(combined[offset : offset + 2], "big")
        offset += 2
        payload_c = combined[offset : offset + length]
        offset += length
        single_table_cumulative = model.cumulative[c : c + 1]
        decoded_channels.append(
            decode_symbols(payload_c, symbols_per_channel, single_table_cumulative, zero_table_index)
        )
    return np.concatenate(decoded_channels)


def main() -> int:
    symbols, model, channels, height, width = _load_real_symbols()
    symbols_per_channel = height * width
    bits = int(np.log2(model.frequencies.shape[1]))

    print("\n=== Functional correctness: does the redesigned format round-trip? ===")
    baseline_payload, baseline_encode_s = _baseline_single_stream(symbols, model, channels, height, width)
    hw_payload, per_channel_payloads, hw_encode_s = _nvc_hw_per_channel_streams(
        symbols, model, channels, height, width
    )
    decoded = _decode_nvc_hw(hw_payload, model, channels, height, width)

    bit_exact = np.array_equal(decoded, symbols)
    print(f"NVC-HW round-trip bit-exact vs. original symbols: {bit_exact}")
    if not bit_exact:
        print("FAIL: the parallel-stream redesign does not preserve correctness.", file=sys.stderr)
        return 1

    print("\n=== Bitrate cost of independence (the one real overhead of this design) ===")
    baseline_bytes = len(baseline_payload)
    hw_bytes = len(hw_payload)
    length_prefix_bytes = 2 * channels
    payload_only_bytes = hw_bytes - length_prefix_bytes
    print(f"Baseline (1 combined stream):        {baseline_bytes:,} bytes")
    print(f"NVC-HW (64 independent streams):     {hw_bytes:,} bytes "
          f"({payload_only_bytes:,} payload + {length_prefix_bytes:,} length prefixes)")
    overhead = hw_bytes - baseline_bytes
    print(f"Overhead: {overhead:+,} bytes ({100 * overhead / baseline_bytes:+.3f}% of baseline size)")
    print(f"  ({channels} independent stream flushes, ~1-2 bytes of unavoidable arithmetic-coder "
          f"tail per stream, + {length_prefix_bytes} bytes of length prefixes)")

    print("\n=== Software reference timing (this dev CPU, C backend - NOT a hardware cycle count) ===")
    print(f"Baseline encode (1 call, {len(symbols)} symbols):        {baseline_encode_s * 1000:.3f} ms")
    print(f"NVC-HW encode ({channels} calls, {symbols_per_channel} symbols each): {hw_encode_s * 1000:.3f} ms")
    print("(Per-call Python/ctypes overhead dominates the per-channel-call version here - this "
          "is exactly the overhead real hardware lanes eliminate by not being 64 separate "
          "Python function calls. Not a projection of hardware speed.)")

    print("\n=== Closed-form hardware latency projection (assumptions stated in this file's header) ===")
    print(f"{'Lanes':>6s} {'channels/lane':>14s} {'encode us/frame':>16s} {'decode us/frame':>16s}")
    decode_cycles = ASSUMED_CYCLES_PER_SYMBOL_DECODE_BITS.get(bits, 8)
    for lanes in LANE_COUNTS:
        rounds = -(-channels // lanes)  # ceil
        encode_cycles_total = rounds * symbols_per_channel * ASSUMED_CYCLES_PER_SYMBOL_ENCODE
        decode_cycles_total = rounds * symbols_per_channel * decode_cycles
        encode_us = 1e6 * encode_cycles_total / ASSUMED_CLOCK_HZ
        decode_us = 1e6 * decode_cycles_total / ASSUMED_CLOCK_HZ
        print(f"{lanes:>6d} {channels / lanes:>14.1f} {encode_us:>16.2f} {decode_us:>16.2f}")

    print("\nAll numbers above are either measured against real project artifacts (correctness, "
          "byte overhead) or explicitly labeled assumptions (cycle counts, clock) - see "
          "ARCHITECTURE.md for how these are used.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
