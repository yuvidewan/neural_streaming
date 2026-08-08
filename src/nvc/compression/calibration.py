"""Fixed quantization calibration from the training split.

WHY THIS EXISTS
---------------
Milestone 5 calibrated scale/zero_point from whichever tensor was being
quantized. That is fine for measuring grid distortion, but it is not a
codec: the decoder receives only a bitstream and cannot re-derive
parameters it never saw. A real codec needs parameters that are fixed ahead
of time, identical on both sides, and derived without touching evaluation
data.

This module derives them once from a subset of the TRAINING split and
writes them to a calibration file that both encoder and decoder load.

CALIBRATION METHOD
------------------
Default: per-channel percentile ranges at (0.1, 99.9).

Percentiles rather than absolute min/max because the Milestone 5 analysis
found a sharply peaked latent distribution with long tails (range about
[-16.5, 13.7] while the standard deviation is only 2.24). Letting a handful
of extreme values define the grid would stretch every quantization step to
cover outliers that almost never occur, wasting most of the grid on empty
space. Clipping the outermost 0.2% instead buys a materially finer step for
the other 99.8%, at the cost of clamping those few values - a trade this
project measures rather than assumes (see `count_clipped`, and the clipping
figures reported by scripts/calibrate_quantizer.py).

Percentiles are configurable. Setting them to (0.0, 100.0) reproduces plain
min/max calibration exactly, so that option is retained rather than removed.

The percentile choice is NOT tuned against the validation or test split -
it is a fixed, documented default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from nvc.compression.quantization import QuantizationParams

CALIBRATION_METHOD = "per_channel_percentile"
DEFAULT_LOWER_PERCENTILE = 0.1
DEFAULT_UPPER_PERCENTILE = 99.9

# Matches the degenerate-range handling in quantization.py: a channel whose
# calibrated range collapses to zero width is widened rather than producing
# scale = 0.
_DEGENERATE_RANGE_HALF_WIDTH = 0.5


@torch.no_grad()
def collect_calibration_latents(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> torch.Tensor:
    """Encode calibration frames, returning latents as one CPU tensor.

    `loader` MUST be a training-split loader. Nothing here enforces that -
    it is the caller's contract, and scripts/calibrate_quantizer.py builds a
    train loader explicitly.
    """
    model.eval()
    latents: list[torch.Tensor] = []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        latents.append(model.encode(batch.to(device)).cpu())

    if not latents:
        raise ValueError("collect_calibration_latents: the data loader produced no batches")
    return torch.cat(latents, dim=0)


def calibrate_quantization_params(
    latents: torch.Tensor,
    *,
    bits: int,
    mode: str = "per_channel",
    lower_percentile: float = DEFAULT_LOWER_PERCENTILE,
    upper_percentile: float = DEFAULT_UPPER_PERCENTILE,
) -> QuantizationParams:
    """Derive fixed affine quantization parameters from calibration latents.

    Uses the same affine formulation as UniformQuantizer (see
    quantization.py); only the source of [x_min, x_max] differs - here it is
    a percentile of the calibration set rather than the tensor being coded.
    """
    if latents.dim() != 4:
        raise ValueError(
            f"Expected a 4D [N, C, H, W] calibration tensor, got shape {tuple(latents.shape)}"
        )
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError(
            "Require 0 <= lower_percentile < upper_percentile <= 100, got "
            f"({lower_percentile}, {upper_percentile})"
        )

    num_channels = latents.shape[1]
    if mode == "global":
        samples = [latents.flatten()]
    elif mode == "per_channel":
        samples = list(latents.permute(1, 0, 2, 3).reshape(num_channels, -1))
    else:
        raise ValueError(f"mode must be 'global' or 'per_channel', got {mode!r}")

    lower_quantile = lower_percentile / 100.0
    upper_quantile = upper_percentile / 100.0

    minimums, maximums = [], []
    for values in samples:
        values = values.to(torch.float32)
        minimums.append(torch.quantile(values, lower_quantile).item())
        maximums.append(torch.quantile(values, upper_quantile).item())

    x_min = torch.tensor(minimums, dtype=torch.float32).reshape(1, -1, 1, 1)
    x_max = torch.tensor(maximums, dtype=torch.float32).reshape(1, -1, 1, 1)

    degenerate = x_max == x_min
    if degenerate.any():
        x_min = torch.where(degenerate, x_min - _DEGENERATE_RANGE_HALF_WIDTH, x_min)
        x_max = torch.where(degenerate, x_max + _DEGENERATE_RANGE_HALF_WIDTH, x_max)

    q_min, q_max = 0, 2 ** bits - 1
    scale = (x_max - x_min) / (q_max - q_min)
    zero_point = q_min - torch.round(x_min / scale)

    return QuantizationParams(scale=scale, zero_point=zero_point, bits=bits, mode=mode)


def save_calibration(
    path: str | Path,
    *,
    params: QuantizationParams,
    entropy_model_data: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Write quantization parameters and the entropy model to one JSON file.

    Encoder and decoder both load this file; it is the shared codec
    definition, and is NOT embedded in every .nvc payload (the .nvc header
    instead carries an entropy-model id so a mismatch is detected).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "calibration_metadata": metadata,
        "quantization": params.to_dict(),
        "entropy_model": entropy_model_data,
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def load_calibration(path: str | Path) -> dict[str, Any]:
    """Read a calibration file written by `save_calibration`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibration file not found: {path}. Run scripts/calibrate_quantizer.py first."
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    for key in ("quantization", "entropy_model", "calibration_metadata"):
        if key not in document:
            raise ValueError(f"Calibration file {path} is missing required section '{key}'")
    return document
