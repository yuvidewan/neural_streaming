"""Plot training curves (train MSE, val MSE, val PSNR) from the history
JSON written by scripts/train_autoencoder.py.

Only plots the epochs actually present in the history file - if it holds
just one or two epochs (e.g. from a CPU smoke test), the plot shows exactly
that; nothing here fabricates additional data points.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\plot_training_history.py
    python scripts\\plot_training_history.py --history outputs\\checkpoints\\history.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nvc.utils.config import load_default_config


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot epoch vs. training MSE, validation MSE, and validation PSNR "
            "from a training history JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--history", type=Path, default=defaults.checkpoint_dir / "history.json",
        help="Path to the history JSON written by scripts/train_autoencoder.py.",
    )
    parser.add_argument(
        "--output", type=Path, default=defaults.visualizations_dir / "training_curves.png",
        help="Path to save the plotted curves.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    if not args.history.is_file():
        print(
            f"[ERROR] History file not found: {args.history}. Run scripts\\train_autoencoder.py first.",
            file=sys.stderr,
        )
        return 1

    history = json.loads(args.history.read_text(encoding="utf-8"))
    if not history:
        print(f"[ERROR] History file {args.history} is empty.", file=sys.stderr)
        return 1
    if len(history) < 2:
        print(f"[WARN] Only {len(history)} epoch(s) of history - this is a single point, not a curve.")

    epochs = [record["epoch"] for record in history]
    train_loss = [record["train_loss"] for record in history]
    val_loss = [record["val_loss"] for record in history]
    val_psnr = [record["val_psnr"] for record in history]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_loss, marker="o")
    axes[0].set_title("Training MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")

    axes[1].plot(epochs, val_loss, marker="o", color="tab:orange")
    axes[1].set_title("Validation MSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")

    axes[2].plot(epochs, val_psnr, marker="o", color="tab:green")
    axes[2].set_title("Validation PSNR")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("PSNR (dB)")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    print(f"Saved training curves ({len(history)} epoch(s)): {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
