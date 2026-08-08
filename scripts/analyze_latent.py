"""Latent-space analysis CLI for a trained BaselineAutoencoder.

Runs only the encoder over the DAVIS test split, then reports descriptive
statistics of the learned latent representation and saves distribution
visualizations. Nothing is trained, quantized, or compressed here - this is
the measurement step that informs the quantizer design.

Outputs:
    outputs/metrics/latent_statistics.json
    outputs/visualizations/latent_histogram.png
    outputs/visualizations/latent_channel_statistics.png
    outputs/visualizations/latent_heatmap.png

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\analyze_latent.py --checkpoint outputs\\checkpoints\\best.pt
    python scripts\\analyze_latent.py --checkpoint outputs\\checkpoints\\best.pt --max-batches 10

Run `python scripts\\analyze_latent.py --help` for the full option list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from nvc.compression import latent_storage_analysis
from nvc.data.loaders import create_test_loader
from nvc.data.validation import DatasetValidationError
from nvc.evaluation.latent_analysis import extract_latents, latent_statistics
from nvc.training import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

# The latent has 64 channels; drawing all of them makes an unreadable wall
# of thumbnails. The heatmap shows this many channels (the first N by index,
# not hand-picked) and says so in the figure title.
_HEATMAP_CHANNELS = 16


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the latent representation produced by a trained "
            "BaselineAutoencoder over the test split."
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
        help="Limit analysis to this many test batches (default: the whole test split).",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--metrics-dir", type=Path, default=defaults.metrics_dir)
    parser.add_argument("--visualizations-dir", type=Path, default=defaults.visualizations_dir)
    return parser


def _plot_histogram(latents: torch.Tensor, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = latents.flatten().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(values, bins=200)
    axes[0].set_title("Latent value distribution")
    axes[0].set_xlabel("Latent value")
    axes[0].set_ylabel("Count")

    axes[1].hist(values, bins=200)
    axes[1].set_yscale("log")
    axes[1].set_title("Latent value distribution (log count)")
    axes[1].set_xlabel("Latent value")
    axes[1].set_ylabel("Count (log)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_channel_statistics(stats: dict, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_channel = stats["per_channel"]
    channels = range(len(per_channel["mean"]))

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].errorbar(list(channels), per_channel["mean"], yerr=per_channel["std"], fmt="o", markersize=3)
    axes[0].set_title("Per-channel mean +/- standard deviation")
    axes[0].set_ylabel("Value")

    axes[1].plot(list(channels), per_channel["min"], marker=".", linestyle="none", label="min")
    axes[1].plot(list(channels), per_channel["max"], marker=".", linestyle="none", label="max")
    axes[1].set_title("Per-channel observed range (drives per-channel quantization scales)")
    axes[1].set_xlabel("Latent channel index")
    axes[1].set_ylabel("Value")
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_heatmap(latents: torch.Tensor, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = latents[0]
    num_channels = min(_HEATMAP_CHANNELS, sample.shape[0])
    columns = 4
    rows = (num_channels + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.2 * rows))
    for index, ax in enumerate(axes.flat):
        if index < num_channels:
            ax.imshow(sample[index].numpy())
            ax.set_title(f"ch {index}", fontsize=8)
        ax.axis("off")

    fig.suptitle(
        f"Latent channels 0-{num_channels - 1} of {sample.shape[0]} (test sample 0)"
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

    print(f"Encoding test split (checkpoint epoch {checkpoint['epoch']})...")
    latents = extract_latents(model, test_loader, device, max_batches=args.max_batches)
    stats = latent_statistics(latents)

    image_shape = (3, defaults.frame_height, defaults.frame_width)
    latent_shape = tuple(stats["latent_shape_per_sample"])
    storage = latent_storage_analysis(image_shape, latent_shape)

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "split": "test",
        "statistics": stats,
        "raw_storage_analysis": storage,
    }

    metrics_path = args.metrics_dir / "latent_statistics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _plot_histogram(latents, args.visualizations_dir / "latent_histogram.png")
    _plot_channel_statistics(stats, args.visualizations_dir / "latent_channel_statistics.png")
    _plot_heatmap(latents, args.visualizations_dir / "latent_heatmap.png")

    global_stats = stats["global"]
    print("=" * 62)
    print("Latent Statistics (DAVIS test split)")
    print("=" * 62)
    print(f"Frames analyzed:      {stats['num_samples']}")
    print(f"Latent shape/sample:  {latent_shape}  ({stats['elements_per_sample']:,} values)")
    print(f"Total latent values:  {stats['total_elements']:,}")
    print()
    print(f"  min:                {global_stats['min']:.6f}")
    print(f"  max:                {global_stats['max']:.6f}")
    print(f"  mean:               {global_stats['mean']:.6f}")
    print(f"  std:                {global_stats['std']:.6f}")
    print(f"  median:             {global_stats['median']:.6f}")
    print(f"  exactly zero:       {global_stats['percent_exactly_zero']:.4f}%")
    print(f"  near zero (<{stats['near_zero_threshold']}):  {global_stats['percent_near_zero']:.4f}%")
    print()
    per_channel_std = stats["per_channel"]["std"]
    per_channel_range = [
        mx - mn for mx, mn in zip(stats["per_channel"]["max"], stats["per_channel"]["min"])
    ]
    print(f"  per-channel std:    min {min(per_channel_std):.4f} / max {max(per_channel_std):.4f}")
    print(
        f"  per-channel range:  min {min(per_channel_range):.4f} / max {max(per_channel_range):.4f} "
        f"(spread {max(per_channel_range) / max(min(per_channel_range), 1e-12):.1f}x)"
    )
    print()
    print("Theoretical RAW tensor storage (NOT a compression ratio - no entropy coding):")
    for name, entry in storage["latent"].items():
        print(
            f"  latent {name:<8} {entry['total_bits']:>9,} bits "
            f"({entry['total_bytes']:>9,.0f} bytes)  "
            f"raw size ratio vs uint8 RGB frame: {entry['raw_size_ratio_vs_uint8_image']:.2f}x"
        )
    print("=" * 62)
    print(f"Statistics written to: {metrics_path}")
    print(f"Visualizations written to: {args.visualizations_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
