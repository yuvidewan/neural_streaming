"""Benchmark the .nvc codec on real DAVIS test frames.

For each supplied calibration (one per bit depth) this encodes and decodes
every evaluated frame and reports actual, measured bitrate - not theoretical
storage. It also:

- verifies entropy coding is exactly lossless (decoded symbols == encoded),
- verifies the .nvc path reconstructs bit-identically to Milestone 5's
  direct quantize-then-decode path, so no distortion is attributable to
  entropy coding,
- compares three representations of the same frame (raw uint8 RGB,
  fixed-width quantized latent, entropy-coded .nvc),
- breaks down coding efficiency against the model's own expected cost.

Frames are taken in the test loader's fixed (unshuffled) order - never
selected by how well they happen to compress.

Outputs:
    outputs/metrics/codec_benchmark.json
    outputs/metrics/codec_benchmark.csv

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\benchmark_codec.py `
        --checkpoint outputs\\checkpoints\\best.pt `
        --calibration outputs\\calibration\\latent_quantization.json `
                      outputs\\calibration\\latent_quantization_6bit.json `
                      outputs\\calibration\\latent_quantization_4bit.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from nvc.compression import (
    EmpiricalEntropyModel,
    QuantizationParams,
    channel_table_index,
    count_clipped,
    decode_latent,
    empirical_entropy,
    encode_latent,
    load_calibration,
    symbols_to_latent,
)
from nvc.data.loaders import create_test_loader
from nvc.data.validation import DatasetValidationError
from nvc.evaluation.basic_metrics import psnr_from_mse
from nvc.training import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

_UINT8_BITS = 8


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure real .nvc bitrate and quality on the DAVIS test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--calibration", type=Path, nargs="+", required=True,
        help="One calibration file per bit depth to benchmark.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit to the first N test frames (default: the whole test split).",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--metrics-dir", type=Path, default=defaults.metrics_dir)
    return parser


def _benchmark_one_configuration(
    model, frames: list[torch.Tensor], calibration: dict, device: torch.device
) -> dict:
    params = QuantizationParams.from_dict(calibration["quantization"]).to(device)
    entropy_model = EmpiricalEntropyModel.from_dict(calibration["entropy_model"])
    bits = params.bits

    squared_error = 0.0
    pixel_count = 0
    payload_bits: list[int] = []
    total_bits: list[int] = []
    compression_ratios: list[float] = []
    all_symbols: list[np.ndarray] = []
    model_expected_bits_total = 0.0
    clipped_total = 0
    clipped_values = 0
    header_bits = 0
    encode_seconds = 0.0
    decode_seconds = 0.0

    for frame in frames:
        with torch.no_grad():
            latent = model.encode(frame)

        clipping = count_clipped(latent, params)
        clipped_total += clipping["clipped_total"]
        clipped_values += clipping["total_values"]

        start = time.perf_counter()
        encoded = encode_latent(
            latent, params=params, entropy_model=entropy_model,
            image_shape=tuple(frame.shape[1:]),
        )
        encode_seconds += time.perf_counter() - start

        start = time.perf_counter()
        decoded_latent, header, decoded_symbols = decode_latent(
            encoded.data, entropy_model=entropy_model
        )
        decode_seconds += time.perf_counter() - start

        # Entropy coding must be exactly lossless.
        if not np.array_equal(encoded.symbols, decoded_symbols):
            raise AssertionError(
                f"Entropy coding was not lossless: "
                f"{int(np.sum(encoded.symbols != decoded_symbols))} symbols differ"
            )

        # The .nvc path must reproduce Milestone 5's direct quantize->decode
        # path exactly, proving no distortion comes from entropy coding.
        direct_latent = symbols_to_latent(
            encoded.symbols, tuple(latent.shape[1:]), params.to("cpu")
        )
        if not torch.equal(direct_latent, decoded_latent):
            raise AssertionError(
                ".nvc dequantized latent differs from the direct quantized latent"
            )

        with torch.no_grad():
            reconstruction = model.decode(decoded_latent.to(device))

        squared_error += torch.sum((reconstruction - frame) ** 2).item()
        pixel_count += frame.numel()

        payload_bits.append(encoded.payload_bits)
        total_bits.append(encoded.total_bits)
        header_bits = encoded.header_bits
        raw_bits = frame.numel() * _UINT8_BITS
        compression_ratios.append(raw_bits / encoded.total_bits)

        all_symbols.append(encoded.symbols)
        table_index = channel_table_index(*latent.shape[1:])
        model_expected_bits_total += entropy_model.expected_bits(encoded.symbols, table_index)

    symbols = np.concatenate(all_symbols)
    symbol_count = len(all_symbols[0])
    num_frames = len(frames)
    pixels_per_frame = frames[0].shape[2] * frames[0].shape[3]

    aggregate_mse = squared_error / pixel_count
    payload_bits_array = np.array(payload_bits, dtype=np.float64)
    total_bits_array = np.array(total_bits, dtype=np.float64)

    fixed_width_bits = symbol_count * bits
    mean_payload_bits = float(payload_bits_array.mean())
    mean_model_expected_bits = model_expected_bits_total / num_frames

    return {
        "bits": bits,
        "mode": params.mode,
        "frames": num_frames,
        "symbols_per_frame": symbol_count,

        "psnr_db": psnr_from_mse(aggregate_mse).item(),
        "mse": aggregate_mse,

        "mean_payload_bytes": mean_payload_bits / 8,
        "mean_total_bytes": float(total_bits_array.mean()) / 8,
        "min_total_bytes": float(total_bits_array.min()) / 8,
        "max_total_bytes": float(total_bits_array.max()) / 8,
        "header_bytes": header_bits / 8,

        "mean_payload_bpp": mean_payload_bits / pixels_per_frame,
        "mean_total_bpp": float(total_bits_array.mean()) / pixels_per_frame,
        "min_total_bpp": float(total_bits_array.min()) / pixels_per_frame,
        "max_total_bpp": float(total_bits_array.max()) / pixels_per_frame,

        "mean_compression_ratio": float(np.mean(compression_ratios)),
        "min_compression_ratio": float(np.min(compression_ratios)),
        "max_compression_ratio": float(np.max(compression_ratios)),

        "mean_payload_bits_per_symbol": mean_payload_bits / symbol_count,
        "mean_total_bits_per_symbol": float(total_bits_array.mean()) / symbol_count,
        "fixed_width_bits_per_symbol": float(bits),
        "test_empirical_entropy_bits_per_symbol": empirical_entropy(symbols, 2 ** bits),
        "model_expected_bits_per_symbol": mean_model_expected_bits / symbol_count,
        "coder_overhead_vs_model_bits_per_symbol": (
            (mean_payload_bits - mean_model_expected_bits) / symbol_count
        ),

        "clipped_percent": 100.0 * clipped_total / clipped_values,
        "fixed_width_latent_bytes": fixed_width_bits / 8,
        "encode_seconds_per_frame": encode_seconds / num_frames,
        "decode_seconds_per_frame": decode_seconds / num_frames,
        "entropy_coding_lossless": True,
        "matches_direct_quantized_path": True,
    }


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device=device)

    try:
        # batch_size=1: the codec operates on one frame at a time.
        test_loader = create_test_loader(
            args.manifest, batch_size=1, crop_size=defaults.random_crop_size
        )
    except DatasetValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    frames = []
    for index, batch in enumerate(test_loader):
        if args.max_frames is not None and index >= args.max_frames:
            break
        frames.append(batch.to(device))
    if not frames:
        print("[ERROR] No test frames were loaded.", file=sys.stderr)
        return 1

    print(f"Benchmarking {len(frames)} test frames across "
          f"{len(args.calibration)} configuration(s)...")

    rows = []
    for path in args.calibration:
        try:
            calibration = load_calibration(path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        print(f"  {path.name} ...", flush=True)
        row = _benchmark_one_configuration(model, frames, calibration, device)
        row["calibration_file"] = str(path)
        rows.append(row)

    rows.sort(key=lambda entry: -entry["bits"])

    # Three-way comparison at the primary operating point (highest bit depth).
    primary = rows[0]
    frame_pixels = frames[0].shape[2] * frames[0].shape[3]
    raw_bytes = frames[0].numel()
    comparison = {
        "A_raw_uint8_rgb": {
            "bytes": raw_bytes,
            "bits": raw_bytes * _UINT8_BITS,
            "bpp": raw_bytes * _UINT8_BITS / frame_pixels,
            "compression_ratio": 1.0,
            "psnr_db": None,  # the lossless source; PSNR against itself is undefined
            "note": "uncompressed source representation",
        },
        "B_fixed_width_quantized_latent": {
            "bytes": primary["fixed_width_latent_bytes"],
            "bits": primary["fixed_width_latent_bytes"] * 8,
            "bpp": primary["fixed_width_latent_bytes"] * 8 / frame_pixels,
            "compression_ratio": raw_bytes / primary["fixed_width_latent_bytes"],
            "psnr_db": primary["psnr_db"],
            "note": (
                f"{primary['bits']} bits/symbol, no entropy coding, no header; "
                "theoretical tensor storage only"
            ),
        },
        "C_entropy_coded_nvc": {
            "bytes": primary["mean_total_bytes"],
            "bits": primary["mean_total_bytes"] * 8,
            "bpp": primary["mean_total_bpp"],
            "compression_ratio": primary["mean_compression_ratio"],
            "psnr_db": primary["psnr_db"],
            "note": "actual measured .nvc file, header included",
        },
    }

    results = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "split": "test",
        "frames_evaluated": len(frames),
        "seed": args.seed,
        "frame_selection": "first N frames in fixed unshuffled test-loader order",
        "metric_note": (
            "PSNR is derived from MSE aggregated over every pixel of the evaluated "
            "frames. Bitrate figures are measured from real .nvc files, not estimated."
        ),
        "three_way_comparison": comparison,
        "configurations": rows,
    }

    metrics_dir = args.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "codec_benchmark.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_fields = [
        "bits", "mode", "psnr_db", "mean_total_bpp", "mean_payload_bpp",
        "mean_compression_ratio", "mean_total_bytes", "mean_payload_bits_per_symbol",
        "test_empirical_entropy_bits_per_symbol", "model_expected_bits_per_symbol",
        "fixed_width_bits_per_symbol", "clipped_percent",
    ]
    csv_path = metrics_dir / "codec_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 100)
    print(f"Codec Benchmark - DAVIS test split ({len(frames)} frames, real .nvc files)")
    print("=" * 100)
    header = (
        f"{'Bits':>5}{'Mode':>13}{'PSNR (dB)':>11}{'Total BPP':>11}{'Payload BPP':>13}"
        f"{'Ratio':>8}{'Bytes':>10}{'Pay b/sym':>11}{'Clip %':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['bits']:>5}{row['mode']:>13}{row['psnr_db']:>11.3f}"
            f"{row['mean_total_bpp']:>11.4f}{row['mean_payload_bpp']:>13.4f}"
            f"{row['mean_compression_ratio']:>8.2f}{row['mean_total_bytes']:>10.0f}"
            f"{row['mean_payload_bits_per_symbol']:>11.4f}{row['clipped_percent']:>8.3f}"
        )
    print("-" * len(header))
    print()
    print("Entropy coding efficiency (bits/symbol):")
    efficiency_header = (
        f"{'Bits':>5}{'Fixed width':>14}{'Test entropy':>15}{'Model expected':>17}"
        f"{'Actual payload':>17}{'Coder overhead':>17}"
    )
    print(efficiency_header)
    print("-" * len(efficiency_header))
    for row in rows:
        print(
            f"{row['bits']:>5}{row['fixed_width_bits_per_symbol']:>14.4f}"
            f"{row['test_empirical_entropy_bits_per_symbol']:>15.4f}"
            f"{row['model_expected_bits_per_symbol']:>17.4f}"
            f"{row['mean_payload_bits_per_symbol']:>17.4f}"
            f"{row['coder_overhead_vs_model_bits_per_symbol']:>+17.4f}"
        )
    print("-" * len(efficiency_header))
    print()
    print(f"Three-way comparison of one frame at the {primary['bits']}-bit operating point:")
    for name, entry in comparison.items():
        psnr_text = "     -   " if entry["psnr_db"] is None else f"{entry['psnr_db']:.3f} dB"
        print(
            f"  {name:<32} {entry['bytes']:>10,.0f} bytes  "
            f"{entry['bpp']:>8.4f} BPP  {entry['compression_ratio']:>7.2f}x  {psnr_text}"
        )
    print()
    print("Entropy coding verified lossless and identical to the direct quantized path.")
    print("=" * 100)
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
