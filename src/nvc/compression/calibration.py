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

from nvc.compression.quantization import QuantizationParams, _compand

CALIBRATION_METHOD = "per_channel_percentile"
DEFAULT_LOWER_PERCENTILE = 0.1
DEFAULT_UPPER_PERCENTILE = 99.9

# torch.quantile refuses any input above 2**24 (16,777,216) elements
# (confirmed on PyTorch 2.13.0 - see OPTIMIZATION_ANALYSIS.md, B1). At the
# default 64x16x16 latent this is 1,024 calibration frames in `global` mode
# (one quantile call over the whole flattened set) or 65,536 frames in
# `per_channel` mode (one call per channel, so only that channel's H*W*N
# values count). Not firing today only because calibrate_quantizer.py's
# default --max-batches keeps runs under both limits - this is a latent
# crash waiting for anyone who raises it, not a hypothetical.
_MAX_QUANTILE_ELEMENTS = 10_000_000

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
    bits_per_channel: tuple[int, ...] | None = None,
    companding_gamma: float | None = None,
) -> QuantizationParams:
    """Derive fixed affine quantization parameters from calibration latents.

    Uses the same affine formulation as UniformQuantizer (see
    quantization.py); only the source of [x_min, x_max] differs - here it is
    a percentile of the calibration set rather than the tensor being coded.

    `bits_per_channel` (per-channel bit allocation, OPTIMIZATION_ANALYSIS.md
    Q3 - see `allocate_bits_per_channel` to derive it) and `companding_gamma`
    (non-uniform bins, Q2) are both optional; `None` (the default for both)
    reproduces today's exact uniform-grid behavior. Percentiles are always
    computed the same way (0.1/99.9 by default) - only what the resulting
    range is DIVIDED BY (per-channel levels instead of the table-wide count)
    or computed FROM (the companded latent instead of the raw one) changes.
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
    if bits_per_channel is not None and mode != "per_channel":
        raise ValueError("bits_per_channel is only valid in 'per_channel' mode")

    num_channels = latents.shape[1]
    if bits_per_channel is not None and len(bits_per_channel) != num_channels:
        raise ValueError(
            f"bits_per_channel has {len(bits_per_channel)} entries but latents "
            f"have {num_channels} channels"
        )

    transformed = _compand(latents, companding_gamma) if companding_gamma is not None else latents

    if mode == "global":
        samples = [transformed.flatten()]
    elif mode == "per_channel":
        samples = list(transformed.permute(1, 0, 2, 3).reshape(num_channels, -1))
    else:
        raise ValueError(f"mode must be 'global' or 'per_channel', got {mode!r}")

    lower_quantile = lower_percentile / 100.0
    upper_quantile = upper_percentile / 100.0

    minimums, maximums = [], []
    for values in samples:
        values = values.to(torch.float32)
        minimums.append(_quantile_safe(values, lower_quantile))
        maximums.append(_quantile_safe(values, upper_quantile))

    x_min = torch.tensor(minimums, dtype=torch.float32).reshape(1, -1, 1, 1)
    x_max = torch.tensor(maximums, dtype=torch.float32).reshape(1, -1, 1, 1)

    degenerate = x_max == x_min
    if degenerate.any():
        x_min = torch.where(degenerate, x_min - _DEGENERATE_RANGE_HALF_WIDTH, x_min)
        x_max = torch.where(degenerate, x_max + _DEGENERATE_RANGE_HALF_WIDTH, x_max)

    if bits_per_channel is None:
        q_min, q_max = 0, 2 ** bits - 1
        scale = (x_max - x_min) / (q_max - q_min)
    else:
        # Each channel's own level count, not the table-wide one - the
        # entire point of per-channel bit allocation (see this module's
        # and quantization.py's docstrings).
        levels = torch.tensor(
            [2 ** b - 1 for b in bits_per_channel], dtype=torch.float32,
        ).reshape(1, -1, 1, 1)
        q_min = 0
        scale = (x_max - x_min) / levels
    zero_point = q_min - torch.round(x_min / scale)

    return QuantizationParams(
        scale=scale, zero_point=zero_point, bits=bits, mode=mode,
        bits_per_channel=bits_per_channel, companding_gamma=companding_gamma,
    )


def allocate_bits_per_channel(
    latents: torch.Tensor,
    *,
    average_bits: int,
    min_bits: int = 2,
    max_bits: int | None = None,
) -> tuple[int, ...]:
    """Classical (reverse) water-filling bit allocation from per-channel
    variance, at a fixed average-bit budget - the textbook solution to
    "spend more levels on high-information channels, fewer on flat ones"
    for a set of roughly-Gaussian-like sources (see calibration.py's own
    docstring: the latent is "sharply peaked ... with long tails", close
    enough to this family for the classical result to be a reasonable,
    principled starting point rather than an ad hoc heuristic):

        bits_c = average_bits + 0.5 * log2(var_c / geometric_mean(var))

    rounded to the nearest integer and clamped to [min_bits, max_bits]
    (`max_bits` defaults to `average_bits` - a channel is never given MORE
    levels than the table's own default depth, only fewer, so the entropy
    table sizing this project already uses is never exceeded). Rounding
    and clamping both perturb the realized average away from
    `average_bits`; the excess/deficit is then greedily redistributed one
    bit at a time (adding to the currently-lowest-bit channel, or removing
    from the currently-highest) so the realized average matches the
    requested budget as closely as integer bits allow - this is a real
    reallocation at a fixed total budget, not a free increase.
    """
    if latents.dim() != 4:
        raise ValueError(
            f"Expected a 4D [N, C, H, W] calibration tensor, got shape {tuple(latents.shape)}"
        )
    if not 1 <= min_bits:
        raise ValueError(f"min_bits must be >= 1, got {min_bits}")
    if max_bits is None:
        max_bits = average_bits
    if not min_bits <= average_bits <= max_bits:
        raise ValueError(
            f"Require min_bits <= average_bits <= max_bits, got "
            f"({min_bits}, {average_bits}, {max_bits})"
        )

    num_channels = latents.shape[1]
    variance = latents.permute(1, 0, 2, 3).reshape(num_channels, -1).to(torch.float64).var(dim=1)
    # A perfectly constant channel has variance 0 - floor it so log2 stays
    # finite; that channel gets pushed toward min_bits anyway, correctly,
    # since it is (by definition) the least informative one.
    variance = torch.clamp(variance, min=1e-12)
    log_variance = torch.log2(variance)
    water_level = float(log_variance.mean())

    raw_bits = average_bits + 0.5 * (log_variance - water_level)
    bits_per_channel = [
        int(min(max(round(float(b)), min_bits), max_bits)) for b in raw_bits
    ]

    target_total = average_bits * num_channels
    _rebalance_to_target_total(bits_per_channel, target_total, min_bits, max_bits)

    return tuple(bits_per_channel)


def _rebalance_to_target_total(
    bits_per_channel: list[int], target_total: int, min_bits: int, max_bits: int,
) -> None:
    """In-place: nudge individual channels by +/-1 bit until the list sums
    to exactly `target_total`, always taking from the currently-highest (or
    giving to the currently-lowest) channel first - keeps the allocation as
    close as possible to the variance-driven ranking `allocate_bits_per_channel`
    computed, rather than an arbitrary tie-break order."""
    total = sum(bits_per_channel)
    while total > target_total:
        index = max(range(len(bits_per_channel)), key=lambda i: bits_per_channel[i])
        if bits_per_channel[index] <= min_bits:
            break  # every channel is already at the floor - cannot reduce further
        bits_per_channel[index] -= 1
        total -= 1
    while total < target_total:
        index = min(range(len(bits_per_channel)), key=lambda i: bits_per_channel[i])
        if bits_per_channel[index] >= max_bits:
            break  # every channel is already at the ceiling - cannot raise further
        bits_per_channel[index] += 1
        total += 1


def _quantile_safe(values: torch.Tensor, q: float) -> float:
    """`torch.quantile`, but subsampled below its hard 2**24-element limit
    instead of crashing above it.

    Below the limit: identical to calling `torch.quantile` directly - no
    behavior change for every calibration run that already works today.
    Above it: a deterministic random subsample of exactly
    `_MAX_QUANTILE_ELEMENTS` values, using the ambient PyTorch RNG (so it
    follows the same seeding rule as everything else in this pipeline -
    `seed_everything`, called once by the caller, is sufficient for
    reproducibility; no private generator is kept here, deliberately,
    matching `QuantizationNoise.apply`'s same choice). Percentile estimation
    from a 10-million-value random sample is statistically indistinguishable
    from using the full set for this project's purposes - calibration is
    already an approximation (a fixed grid derived from a training subset),
    not an exact computation.
    """
    if values.numel() > _MAX_QUANTILE_ELEMENTS:
        indices = torch.randperm(values.numel(), device=values.device)[:_MAX_QUANTILE_ELEMENTS]
        values = values[indices]
    return torch.quantile(values, q).item()


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
