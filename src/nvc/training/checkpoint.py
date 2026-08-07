"""Checkpoint save/load and resume-state restoration for training runs.

A checkpoint carries everything needed to resume training exactly: model
and optimizer state, the last completed epoch, the full metric history, and
the model's architecture config (so the model can be rebuilt without
knowing the original CLI arguments, e.g. in scripts/reconstruct.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, Any]],
    model_config: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "model_config": model_config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=False: these are checkpoints this project wrote itself
    # (not untrusted third-party files), and they carry plain history/config
    # dicts alongside the tensors, not just a bare state_dict.
    return torch.load(path, map_location=map_location, weights_only=False)


def resume_training_state(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Restore model/optimizer weights from a checkpoint in place.

    Returns (next_epoch, history) so the caller continues epoch numbering
    from where training left off rather than restarting at 1. Raises
    RuntimeError (from load_state_dict) with PyTorch's own shape-mismatch
    message if the checkpoint's architecture doesn't match model/optimizer.
    """
    checkpoint = load_checkpoint(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"] + 1, checkpoint["history"]
