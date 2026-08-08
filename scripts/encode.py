"""Encode a single image into a .nvc compressed file.

    image -> encoder -> latent -> fixed quantizer -> symbols
          -> arithmetic coder -> .nvc

Quantization parameters and the entropy model come from a calibration file
produced by scripts/calibrate_quantizer.py - nothing is calibrated here, so
encoding is deterministic and the decoder can reproduce it exactly.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\encode.py `
        --checkpoint outputs\\checkpoints\\best.pt `
        --input data\\frames\\test\\bmx-bumps_000001.png `
        --output outputs\\compressed\\frame.nvc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from nvc.compression import (
    EmpiricalEntropyModel,
    NVCWriter,
    QuantizationParams,
    encode_frame,
    load_calibration,
)
from nvc.data.image_io import read_image_as_tensor
from nvc.data.validation import DatasetValidationError
from nvc.training import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode an image into a .nvc compressed file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Image file to encode.")
    parser.add_argument("--output", type=Path, required=True, help="Destination .nvc path.")
    parser.add_argument(
        "--calibration", type=Path,
        default=defaults.checkpoint_dir.parent / "calibration" / "latent_quantization.json",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.input.is_file():
        print(f"[ERROR] Input image not found: {args.input}", file=sys.stderr)
        return 1

    try:
        calibration = load_calibration(args.calibration)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)

    params = QuantizationParams.from_dict(calibration["quantization"]).to(device)
    entropy_model = EmpiricalEntropyModel.from_dict(calibration["entropy_model"])

    try:
        frame = read_image_as_tensor(args.input).to(device)
    except DatasetValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    try:
        result = encode_frame(model, frame, params=params, entropy_model=entropy_model)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    written = NVCWriter.write(args.output, result.header, result.data[result.header.header_size:])

    original_bytes = frame.shape[1] * frame.shape[2] * frame.shape[0]
    pixels = result.header.image_width * result.header.image_height

    print("=" * 62)
    print("Encode")
    print("=" * 62)
    print(f"Input:                {args.input}")
    print(f"Output:               {args.output}")
    print(f"Quantization:         {result.header.quantization_bits}-bit "
          f"{result.header.quantization_mode}")
    print(f"Latent:               "
          f"{result.header.latent_channels}x{result.header.latent_height}"
          f"x{result.header.latent_width}  ({result.header.symbol_count:,} symbols)")
    print()
    print(f"Raw uint8 RGB source: {original_bytes:,} bytes")
    print(f".nvc total:           {written:,} bytes "
          f"(header {result.header.header_size:,} + payload {len(result.data) - result.header.header_size:,})")
    print(f"Compression ratio:    {original_bytes / written:.2f}x  (vs raw uint8 RGB)")
    print()
    print(f"Payload-only BPP:     {result.bits_per_pixel(payload_only=True):.4f}")
    print(f"Total-file BPP:       {result.bits_per_pixel():.4f}")
    print(f"Payload bits/symbol:  {result.bits_per_symbol():.4f}")
    print(f"Header overhead:      {100.0 * result.header_bits / result.total_bits:.2f}% of the file")
    print(f"Pixels:               {pixels:,}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
