"""End-to-end latent quantization experiment.

Measures what quantizing the latent actually costs in reconstruction
quality, across bit widths and scaling modes:

    frame -> encoder -> float latent -> quantize -> integer latent
          -> dequantize -> approximate latent -> decoder -> reconstruction

The float32 row is the unquantized baseline (encoder -> decoder directly),
so every quantized row can be read as a delta against it.

Metric aggregation: squared error is summed over every pixel of the whole
evaluated split and divided once at the end, then PSNR is derived from that
aggregate MSE. This is a dataset-level figure - it is deliberately NOT the
mean of per-batch PSNR values, which would weight small batches equally
with full ones and is not comparable across different batch sizes.

Nothing here is entropy-coded and no bitstream is produced. See the storage
note in `src/nvc/compression/storage_analysis.py`.

Outputs:
    outputs/metrics/quantization_results.json
    outputs/metrics/quantization_results.csv
    outputs/visualizations/quantization_comparison.png

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\quantization_experiment.py --checkpoint outputs\\checkpoints\\best.pt
    python scripts\\quantization_experiment.py --checkpoint outputs\\checkpoints\\best.pt --max-batches 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from nvc.compression import UniformQuantizer, latent_storage_analysis
from nvc.data.loaders import create_test_loader
from nvc.data.validation import DatasetValidationError
from nvc.evaluation.basic_metrics import psnr_from_mse
from nvc.training import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

# (label, bits, mode). bits=None marks the unquantized float32 baseline.
_CONFIGURATIONS: list[tuple[str, int | None, str | None]] = [
    ("Float32 baseline", None, None),
    ("Global", 8, "global"),
    ("Per-channel", 8, "per_channel"),
    ("Global", 6, "global"),
    ("Per-channel", 6, "per_channel"),
    ("Global", 4, "global"),
    ("Per-channel", 4, "per_channel"),
]

_COMPARISON_SAMPLES = 4


class _ErrorAccumulator:
    """Running totals for one configuration across the evaluated split."""

    def __init__(self) -> None:
        self.image_squared_error = 0.0
        self.image_elements = 0
        self.latent_squared_error = 0.0
        self.latent_absolute_error = 0.0
        self.latent_elements = 0
        self.latent_max_abs_error = 0.0

    def update_image(self, reconstruction: torch.Tensor, target: torch.Tensor) -> None:
        self.image_squared_error += torch.sum((reconstruction - target) ** 2).item()
        self.image_elements += target.numel()

    def update_latent(self, dequantized: torch.Tensor, original: torch.Tensor) -> None:
        difference = dequantized - original
        self.latent_squared_error += torch.sum(difference ** 2).item()
        self.latent_absolute_error += torch.sum(torch.abs(difference)).item()
        self.latent_elements += original.numel()
        self.latent_max_abs_error = max(
            self.latent_max_abs_error, torch.max(torch.abs(difference)).item()
        )

    def summary(self) -> dict:
        image_mse = self.image_squared_error / self.image_elements
        result = {
            "mse": image_mse,
            "psnr_db": psnr_from_mse(image_mse).item(),
        }
        if self.latent_elements:
            result.update({
                "latent_mse": self.latent_squared_error / self.latent_elements,
                "latent_mae": self.latent_absolute_error / self.latent_elements,
                "latent_max_abs_error": self.latent_max_abs_error,
            })
        else:
            # Float32 baseline: the latent is untouched, so error is exactly 0.
            result.update({"latent_mse": 0.0, "latent_mae": 0.0, "latent_max_abs_error": 0.0})
        return result


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure reconstruction quality when the latent is uniformly "
            "quantized at several bit widths and scaling modes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json"
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Limit the experiment to this many test batches (default: the whole test split).",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--metrics-dir", type=Path, default=defaults.metrics_dir)
    parser.add_argument("--visualizations-dir", type=Path, default=defaults.visualizations_dir)
    return parser


def _save_comparison_grid(
    originals: torch.Tensor, reconstructions: dict[str, torch.Tensor], output_path: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    column_titles = ["Original"] + list(reconstructions)
    num_rows = originals.shape[0]
    num_columns = len(column_titles)

    fig, axes = plt.subplots(
        num_rows, num_columns, figsize=(1.7 * num_columns, 1.85 * num_rows), squeeze=False
    )
    for row in range(num_rows):
        images = [originals[row]] + [reconstructions[key][row] for key in reconstructions]
        for column, image in enumerate(images):
            axes[row][column].imshow(image.permute(1, 2, 0).clamp(0, 1).numpy())
            axes[row][column].axis("off")
            if row == 0:
                axes[row][column].set_title(column_titles[column], fontsize=8)

    fig.suptitle("Latent quantization: reconstruction comparison", fontsize=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.manifest.is_file():
        print(
            f"[ERROR] Manifest not found: {args.manifest}. Run scripts\\prepare_dataset.py first.",
            file=sys.stderr,
        )
        return 1

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device=device)

    try:
        test_loader = create_test_loader(
            args.manifest, batch_size=args.batch_size, crop_size=defaults.random_crop_size,
        )
    except DatasetValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    quantizers = {
        _key(label, bits): (None if bits is None else UniformQuantizer(bits, mode))
        for label, bits, mode in _CONFIGURATIONS
    }
    accumulators = {key: _ErrorAccumulator() for key in quantizers}

    comparison_originals: torch.Tensor | None = None
    comparison_reconstructions: dict[str, torch.Tensor] = {}

    total_frames = 0
    for index, batch in enumerate(test_loader):
        if args.max_batches is not None and index >= args.max_batches:
            break
        batch = batch.to(device)
        total_frames += batch.shape[0]

        # Encode once per batch; every configuration reuses this latent.
        latent = model.encode(batch)

        for key, quantizer in quantizers.items():
            if quantizer is None:
                effective_latent = latent
            else:
                effective_latent, _ = quantizer.quantize_dequantize(latent)
                accumulators[key].update_latent(effective_latent, latent)

            reconstruction = model.decode(effective_latent)
            accumulators[key].update_image(reconstruction, batch)

            if index == 0:
                comparison_reconstructions[key] = reconstruction[:_COMPARISON_SAMPLES].cpu()

        if index == 0:
            comparison_originals = batch[:_COMPARISON_SAMPLES].cpu()

    if total_frames == 0:
        print("[ERROR] No test batches were evaluated.", file=sys.stderr)
        return 1

    latent_shape = tuple(latent.shape[1:])
    image_shape = tuple(batch.shape[1:])

    rows = []
    for label, bits, mode in _CONFIGURATIONS:
        key = _key(label, bits)
        summary = accumulators[key].summary()
        rows.append({
            "configuration": label,
            "bits": 32 if bits is None else bits,
            "mode": mode or "none (unquantized float32)",
            **summary,
        })

    results = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "split": "test",
        "frames_evaluated": total_frames,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "image_shape_per_sample": list(image_shape),
        "latent_shape_per_sample": list(latent_shape),
        "metric_note": (
            "MSE is aggregated over every pixel of the evaluated split, then PSNR "
            "is derived from that single aggregate MSE (not a mean of per-batch PSNR). "
            "Latent errors are aggregated the same way over all latent elements."
        ),
        "quantization_note": (
            "Scale/zero_point are calibrated per batch from the tensor being quantized. "
            "No entropy coding is applied and no bitstream is produced; these results "
            "isolate the distortion introduced by the quantization grid alone."
        ),
        "results": rows,
        "raw_storage_analysis": latent_storage_analysis(image_shape, latent_shape),
    }

    metrics_dir = args.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "quantization_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_path = metrics_dir / "quantization_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    comparison_path = args.visualizations_dir / "quantization_comparison.png"
    if comparison_originals is not None:
        _save_comparison_grid(comparison_originals, comparison_reconstructions, comparison_path)

    baseline_psnr = rows[0]["psnr_db"]
    print("=" * 88)
    print(f"Latent Quantization Experiment - DAVIS test split ({total_frames} frames)")
    print("=" * 88)
    header = (
        f"{'Configuration':<18}{'Bits':>5}{'PSNR (dB)':>12}{'dPSNR':>9}"
        f"{'MSE':>12}{'Latent MSE':>13}{'Latent MAE':>12}{'Max |err|':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        delta = row["psnr_db"] - baseline_psnr
        delta_text = "  -  " if row["bits"] == 32 else f"{delta:+.2f}"
        print(
            f"{row['configuration']:<18}{row['bits']:>5}{row['psnr_db']:>12.3f}{delta_text:>9}"
            f"{row['mse']:>12.6f}{row['latent_mse']:>13.6f}{row['latent_mae']:>12.6f}"
            f"{row['latent_max_abs_error']:>11.4f}"
        )
    print("-" * len(header))
    print("Latent MSE/MAE are LATENT-space errors; MSE/PSNR are IMAGE-space. Not the same thing.")
    print("No entropy coding applied - these are not codec bitrates or compression ratios.")
    print("=" * 88)
    print(f"JSON:       {json_path}")
    print(f"CSV:        {csv_path}")
    print(f"Comparison: {comparison_path}")
    return 0


def _key(label: str, bits: int | None) -> str:
    return "float32" if bits is None else f"{label.lower().replace('-', '_')}_{bits}bit"


if __name__ == "__main__":
    sys.exit(main())
