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

from nvc.models import BaselineAutoencoder


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, Any]],
    model_config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """`extra` (Milestone 9A): an optional, generic escape hatch for
    milestone-specific training-time state that must NOT live in
    `model_state_dict`/`model_config` - e.g. a rate estimator's own
    parameters (`nvc.training.rate_estimator.RateEstimator`), which are
    training-only infrastructure, never used at inference, and must never
    change what `BaselineAutoencoder(**model_config)` + `load_state_dict`
    produces on the ordinary inference path.

    Omitted (the default, `None`): the saved dict has no `"extra"` key at
    all - byte-for-byte the same checkpoint this function has always
    produced. `load_model_from_checkpoint`/`resume_training_state` below
    only ever read their own known keys, so old checkpoints (no `extra`)
    and new ones (with it) both load through them unchanged either way -
    an `extra` key is nothing either function looks for.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "history": history,
        "model_config": model_config,
    }
    if extra is not None:
        document["extra"] = extra
    torch.save(document, path)


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


def load_model_from_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
    eval_mode: bool = True,
) -> tuple[BaselineAutoencoder, dict[str, Any]]:
    """Rebuild a trained BaselineAutoencoder from a checkpoint.

    The architecture comes from the checkpoint's own saved `model_config`,
    so callers never have to re-specify the training-time arguments (e.g.
    --latent-channels). Returns (model, checkpoint) so callers can also read
    `epoch`/`history` without loading the file twice.

    This is the single model-loading path shared by every inference-side
    consumer (reconstruction, latent analysis, quantization experiments) -
    it is deliberately not reimplemented per script.
    """
    checkpoint = load_checkpoint(path, map_location=device)
    model = BaselineAutoencoder(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model = model.to(device)
    if eval_mode:
        model.eval()
    return model, checkpoint


def resume_model_only(
    path: str | Path,
    *,
    model: torch.nn.Module,
    map_location: str | torch.device | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Restore model weights + epoch/history, but NOT optimizer state.

    Milestone 9C. `resume_training_state` below restores the optimizer too,
    which is correct when continuing the *same* run but impossible when the
    optimizer's parameter list has changed since the checkpoint was written.
    That is exactly the case when starting rate training (`--rate-enabled`)
    from any pre-M9 checkpoint: the optimizer then also owns the rate
    estimator's `loc`/`log_scale`, so it holds 18 parameters where the
    checkpoint's saved state has 16, and `optimizer.load_state_dict` raises
    `ValueError: loaded state dict contains a parameter group that doesn't
    match the size of optimizer's group`.

    Restoring the weights and starting the optimizer fresh is also the
    experimentally correct choice for a lambda sweep: every arm then begins
    from byte-identical model weights AND an identical (empty) optimizer
    state, so the only difference between runs is lambda itself. Carrying
    over Adam moments accumulated under a pure-distortion objective would
    make the first steps of each arm depend on a history none of them share
    with their own loss function.

    Returns (next_epoch, history), same as `resume_training_state`.
    """
    checkpoint = load_checkpoint(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint["epoch"] + 1, checkpoint["history"]


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

    Raises ValueError (from `optimizer.load_state_dict`) if the optimizer's
    parameter list has changed since the checkpoint was written - see
    `resume_model_only` above for when that happens and what to use instead.
    """
    checkpoint = load_checkpoint(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"] + 1, checkpoint["history"]
