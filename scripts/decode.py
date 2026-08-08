"""Decode a .nvc compressed file back into an image.

    .nvc -> arithmetic decoder -> symbols -> dequantizer
         -> approximate latent -> decoder -> reconstructed image

Quantization parameters are read from the file's own header; the entropy
model comes from the calibration file and is checked against the model id
recorded at encode time, so a mismatched calibration fails loudly instead of
producing garbage.

Pass --reference to also report PSNR against the original frame.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\decode.py `
        --checkpoint outputs\\checkpoints\\best.pt `
        --input outputs\\compressed\\frame.nvc `
        --output outputs\\reconstructed\\frame.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from nvc.compression import EmpiricalEntropyModel, NVCFormatError, decode_frame, load_calibration
from nvc.data.image_io import read_image_as_tensor, write_tensor_as_image
from nvc.data.validation import DatasetValidationError
from nvc.evaluation.basic_metrics import mse, psnr
from nvc.training import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode a .nvc compressed file back into an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Source .nvc file.")
    parser.add_argument("--output", type=Path, required=True, help="Destination image path.")
    parser.add_argument(
        "--calibration", type=Path,
        default=defaults.checkpoint_dir.parent / "calibration" / "latent_quantization.json",
    )
    parser.add_argument(
        "--reference", type=Path, default=None,
        help="Original image, to report reconstruction PSNR against.",
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
        print(f"[ERROR] .nvc file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        calibration = load_calibration(args.calibration)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    entropy_model = EmpiricalEntropyModel.from_dict(calibration["entropy_model"])

    data = args.input.read_bytes()
    try:
        reconstruction, header = decode_frame(model, data, entropy_model=entropy_model)
    except NVCFormatError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    write_tensor_as_image(reconstruction, args.output)

    print("=" * 62)
    print("Decode")
    print("=" * 62)
    print(f"Input:                {args.input} ({len(data):,} bytes)")
    print(f"Output:               {args.output}")
    print(f"Format version:       {header.format_version}")
    print(f"Quantization:         {header.quantization_bits}-bit {header.quantization_mode}")
    print(f"Image:                {header.image_channels}x{header.image_height}x{header.image_width}")
    print(f"Latent:               "
          f"{header.latent_channels}x{header.latent_height}x{header.latent_width}")
    print(f"Symbols decoded:      {header.symbol_count:,}")
    print(f"Entropy model id:     {header.entropy_model_id.hex()}")

    if args.reference is not None:
        if not args.reference.is_file():
            print(f"[ERROR] Reference image not found: {args.reference}", file=sys.stderr)
            return 1
        try:
            reference = read_image_as_tensor(args.reference).unsqueeze(0).to(reconstruction.device)
        except DatasetValidationError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        if reference.shape != reconstruction.shape:
            print(
                f"[ERROR] Reference shape {tuple(reference.shape)} does not match "
                f"decoded shape {tuple(reconstruction.shape)}",
                file=sys.stderr,
            )
            return 1
        print()
        print(f"MSE vs reference:     {mse(reconstruction, reference).item():.6f}")
        print(f"PSNR vs reference:    {psnr(reconstruction, reference).item():.2f} dB")

    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
