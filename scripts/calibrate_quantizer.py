"""Derive fixed quantization parameters and an entropy model from the
TRAINING split, and write them to a shared calibration file.

This is the step that turns Milestone 5's per-batch dynamic quantization
into something a real codec can use: both encoder and decoder load the same
file, so the decoder never has to guess parameters it was not given.

Validation and test frames are never touched here - the loader is explicitly
a train-split loader.

Outputs:
    outputs/calibration/latent_quantization.json
    outputs/metrics/symbol_distribution.json
    outputs/visualizations/symbol_distribution.png

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\calibrate_quantizer.py --checkpoint outputs\\checkpoints\\best.pt
    python scripts\\calibrate_quantizer.py --checkpoint outputs\\checkpoints\\best.pt --bits 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from nvc.compression import (
    CALIBRATION_METHOD,
    EmpiricalEntropyModel,
    calibrate_quantization_params,
    channel_table_index,
    collect_calibration_latents,
    count_clipped,
    latent_to_symbols,
    save_calibration,
    symbol_distribution_summary,
)
from nvc.compression.calibration import DEFAULT_LOWER_PERCENTILE, DEFAULT_UPPER_PERCENTILE
from nvc.data.loaders import create_train_loader
from nvc.data.validation import DatasetValidationError
from nvc.training import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate fixed per-channel quantization parameters and an empirical "
            "entropy model from the training split."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json"
    )
    parser.add_argument("--bits", type=int, default=defaults.quantization_bits)
    parser.add_argument(
        "--mode", choices=["global", "per_channel"], default=defaults.quantization_mode
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--max-batches", type=int, default=50,
        help="Training batches to calibrate on (50 x batch_size frames by default).",
    )
    parser.add_argument("--lower-percentile", type=float, default=DEFAULT_LOWER_PERCENTILE)
    parser.add_argument("--upper-percentile", type=float, default=DEFAULT_UPPER_PERCENTILE)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument(
        "--output", type=Path,
        default=defaults.checkpoint_dir.parent / "calibration" / "latent_quantization.json",
    )
    parser.add_argument("--metrics-dir", type=Path, default=defaults.metrics_dir)
    parser.add_argument("--visualizations-dir", type=Path, default=defaults.visualizations_dir)
    return parser


def _plot_symbol_distribution(summary: dict, bits: int, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probabilities = np.array(summary["probabilities"])
    symbols = np.arange(len(probabilities))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(symbols, probabilities, width=1.0)
    axes[0].set_title(f"{bits}-bit symbol distribution (calibration set)")
    axes[0].set_xlabel("Symbol")
    axes[0].set_ylabel("Probability")

    axes[1].bar(symbols, probabilities, width=1.0)
    axes[1].set_yscale("log")
    axes[1].set_title("Symbol distribution (log probability)")
    axes[1].set_xlabel("Symbol")
    axes[1].set_ylabel("Probability (log)")

    fig.suptitle(
        f"Empirical entropy {summary['empirical_entropy_bits_per_symbol']:.4f} bits/symbol "
        f"vs {summary['fixed_width_bits_per_symbol']:.1f} bits/symbol fixed width"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.manifest.is_file():
        print(f"[ERROR] Manifest not found: {args.manifest}", file=sys.stderr)
        return 1
    if not 1 <= args.bits <= 16:
        parser.error("--bits must be between 1 and 16")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device=device)

    try:
        # TRAIN split only - calibration must never see val/test data.
        train_loader = create_train_loader(
            args.manifest, batch_size=args.batch_size, seed=args.seed,
            crop_size=defaults.random_crop_size,
        )
    except DatasetValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"Collecting calibration latents from the TRAIN split (max {args.max_batches} batches)...")
    latents = collect_calibration_latents(model, train_loader, device, max_batches=args.max_batches)
    num_frames = latents.shape[0]

    params = calibrate_quantization_params(
        latents, bits=args.bits, mode=args.mode,
        lower_percentile=args.lower_percentile, upper_percentile=args.upper_percentile,
    )

    clipping = count_clipped(latents, params)

    # Quantize the calibration latents with the fixed grid to fit the model.
    quantizer_params = params.to(latents.device)
    symbol_frames = np.stack([
        latent_to_symbols(latents[i : i + 1], quantizer_params).reshape(latents.shape[1:])
        for i in range(num_frames)
    ])

    entropy_model = EmpiricalEntropyModel.from_symbols(
        symbol_frames, bits=args.bits, num_tables=latents.shape[1]
    )
    summary = symbol_distribution_summary(symbol_frames, 2 ** args.bits)

    table_index = channel_table_index(*latents.shape[1:])
    per_channel_entropy = entropy_model.entropy_bits_per_symbol()
    model_expected_bits = entropy_model.expected_bits(
        symbol_frames.reshape(num_frames, -1)[0], table_index
    )

    metadata = {
        "method": CALIBRATION_METHOD,
        "lower_percentile": args.lower_percentile,
        "upper_percentile": args.upper_percentile,
        "bits": args.bits,
        "mode": args.mode,
        "latent_channels": int(latents.shape[1]),
        "latent_height": int(latents.shape[2]),
        "latent_width": int(latents.shape[3]),
        "calibration_frames": int(num_frames),
        "calibration_split": "train",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "seed": args.seed,
        "clipping": clipping,
        "entropy_model_id": entropy_model.model_id().hex(),
    }

    save_calibration(
        args.output, params=params,
        entropy_model_data=entropy_model.to_dict(), metadata=metadata,
    )

    # Bit depth in the filename so calibrating several depths does not have
    # each run silently clobber the previous run's artifacts.
    metrics_path = args.metrics_dir / f"symbol_distribution_{args.bits}bit.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({
        "calibration_metadata": metadata,
        "aggregate_symbol_distribution": summary,
        "per_channel_model_entropy_bits_per_symbol": per_channel_entropy.tolist(),
        "mean_per_channel_model_entropy": float(per_channel_entropy.mean()),
        "model_expected_bits_for_one_frame": model_expected_bits,
    }, indent=2), encoding="utf-8")

    _plot_symbol_distribution(
        summary, args.bits, args.visualizations_dir / f"symbol_distribution_{args.bits}bit.png"
    )

    print("=" * 68)
    print("Quantization Calibration")
    print("=" * 68)
    print(f"Checkpoint:            {args.checkpoint} (epoch {checkpoint['epoch']})")
    print(f"Split used:            train  ({num_frames} frames)")
    print(f"Method:                {CALIBRATION_METHOD} "
          f"[{args.lower_percentile}, {args.upper_percentile}]")
    print(f"Bit depth / mode:      {args.bits}-bit / {args.mode}")
    print(f"Latent shape:          {tuple(latents.shape[1:])}")
    print()
    print(f"Clipped by percentile: {clipping['clipped_total']:,} / {clipping['total_values']:,} "
          f"values ({clipping['clipped_percent']:.4f}%)")
    print(f"  below range:         {clipping['clipped_low']:,}")
    print(f"  above range:         {clipping['clipped_high']:,}")
    print()
    print(f"Unique symbols seen:   {summary['num_unique_symbols_observed']} "
          f"of {summary['num_symbols_possible']} possible")
    print(f"Fixed-width cost:      {summary['fixed_width_bits_per_symbol']:.4f} bits/symbol")
    print(f"Empirical entropy:     {summary['empirical_entropy_bits_per_symbol']:.4f} bits/symbol "
          f"(aggregate, order-0)")
    print(f"Per-channel model:     {per_channel_entropy.mean():.4f} bits/symbol (mean over channels)")
    print(f"  -> headroom vs fixed width: "
          f"{summary['fixed_width_bits_per_symbol'] - per_channel_entropy.mean():.4f} bits/symbol")
    print()
    print(f"Entropy model id:      {entropy_model.model_id().hex()}")
    print("=" * 68)
    print(f"Calibration written to: {args.output}")
    print(f"Symbol stats written to: {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
