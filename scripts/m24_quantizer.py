"""M24 - the quantizer, measured against bmshj2018's at FROZEN WEIGHTS.

The CompressAI anchor (`outputs/benchmarks/compressai_anchor/README.md`) found
that `bmshj2018-factorized` needs about HALF our all-intra bits, and that it is
our design minus our quantizer: the same GDN transform at stride 16, the same
Vimeo-90K, no context model and no hyperprior. What it does differently is

  * fixed-step rounding of an UNBOUNDED integer alphabet,
  * a learned per-channel density,
  * one model per lambda, trained for rate.

Ours instead fits a per-channel affine grid to the 0.1/99.9 percentiles at a
FIXED depth of 3, 4 or 5 bits. Three consequences, and all three are in the
quantizer rather than the transform:

  1 the step is tied to each channel's RANGE, so a channel carrying little
    information still gets 2**b levels across its own spread - which is exactly
    the "cost decoupled from content" the latent-shape diagnostic measured
    (Spearman +0.215, a channel at 1/20 of peak energy paying ~68% of full
    depth);
  2 the level count is fixed rather than whatever the distribution needs;
  3 everything past the percentile bounds is CLIPPED, a distortion source
    bmshj2018 does not have at all.

THE QUESTION THIS ANSWERS, AND THE ONE IT DOES NOT
--------------------------------------------------
The anchor's +90.9% mixes two causes: the quantizer, and weights trained for
rate at a given lambda. They need different work - the first is a coding change,
the second needs Vimeo and a retrain - so the project should know which it is
buying before paying for either.

This module measures the quantizer ALONE, on the latents the current trained
transform already produces. Nothing is retrained. So it answers "how much of the
gap is reachable without touching the weights", and it cannot answer "how much
more would rate-aware training add" - that is Phase B, and it needs a GPU and
the dataset.

Sweeping the step is what makes a frozen-weight comparison fair: our scheme
traces its rate-distortion curve by changing the bit depth, bmshj2018's by
changing the step, and both are then curves over the same latents scored the
same way.

THREE ARMS, SO THE TWO CAUSES SEPARATE
--------------------------------------
  grid_histogram     our actual intra codec: percentile grid at fixed depth, a
                     per-channel histogram fitted on TRAIN.
  grid_laplace       the same grid, with a per-channel discretised Laplace
                     instead of the histogram. Isolates the DENSITY MODEL.
  unit_step_laplace  rounding at a swept step with the same kind of density.
                     Isolates the GRID RULE, which is the part the anchor points
                     at.

`grid_histogram` -> `grid_laplace` is the density's contribution;
`grid_laplace` -> `unit_step_laplace` is the grid rule's. Reporting the two
separately is the point: the anchor named both differences at once, and the fix
for each is different work.

Rate is ideal bits under a TRAIN-fitted density, which holds the entropy model
comparable across arms instead of mixing in a particular coder's overhead. The
real-bytes check belongs with a deployment, not with a diagnosis - and the coder
has been within 0.011% of ideal every time this project has measured it.
Distortion is decoded-pixel PSNR through `benchmark_parity`'s scorer, so these
curves sit on the same axes as every other number in the project.

TRAIN fits every density and every grid. VAL-B reports. TEST is never opened.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.codec import latent_to_symbols, symbols_to_latent

ARMS = ("grid_histogram", "grid_laplace", "unit_step_laplace")
DEFAULT_GRID_BITS = (3, 4, 5, 6, 7)
# Multipliers on the deployed 4-bit median step, chosen to bracket the grid arm's
# quality range from both sides so the two curves overlap and a BD-rate between
# them is interpolation rather than extrapolation.
DEFAULT_STEP_MULTIPLIERS = (0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
LOWER_PERCENTILE = 0.1
UPPER_PERCENTILE = 99.9


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------------
# Densities, both fitted on TRAIN only
# --------------------------------------------------------------------------------------------


def fit_histogram(symbols: np.ndarray, *, alphabet: int, channels: int) -> np.ndarray:
    """Per-channel add-one histogram over a fixed alphabet. [channels, alphabet].

    Add-one rather than unsmoothed because a held-out symbol a channel never
    produced on TRAIN must not cost infinity - the same zero-probability safety
    every table in this codec carries.
    """
    flat = symbols.reshape(symbols.shape[0], channels, -1)
    counts = np.ones((channels, alphabet), dtype=np.float64)
    for channel in range(channels):
        counts[channel] += np.bincount(flat[:, channel, :].reshape(-1),
                                       minlength=alphabet)[:alphabet]
    return counts / counts.sum(axis=1, keepdims=True)


def fit_laplace(values: np.ndarray, *, channels: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel Laplace by maximum likelihood: median and mean absolute deviation.

    Laplace rather than Gaussian for the reason `calibration.py` already records
    about this project's latents - sharply peaked with long tails - and the
    reason `rate_estimator.py` gives for its own choice. The ML estimates are
    the median and the mean absolute deviation from it, not the mean and the
    standard deviation.
    """
    flat = np.asarray(values).reshape(values.shape[0], channels, -1).transpose(1, 0, 2)
    flat = flat.reshape(channels, -1)
    location = np.median(flat, axis=1)
    scale = np.abs(flat - location[:, None]).mean(axis=1)
    # A degenerate channel (every value identical) would give scale 0 and an
    # infinite log-probability; floor it at half a step's worth of spread.
    return location, np.maximum(scale, 1e-6)


def _laplace_cdf(x: np.ndarray, location: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Laplace CDF, with BOTH branches clipped before the exponential.

    `np.where` evaluates both arms whatever the condition selects, so clipping
    only the upper one still overflows `exp` on a far-left tail value and can
    hand back inf or nan from the arm that was never going to be chosen.
    """
    centred = x - location
    left = 0.5 * np.exp(np.clip(centred, None, 0.0) / scale)
    right = 1.0 - 0.5 * np.exp(-np.clip(centred, 0.0, None) / scale)
    return np.where(centred < 0.0, left, right)


def discretised_laplace_bits(symbols: np.ndarray, location: np.ndarray, scale: np.ndarray, *,
                             channels: int, floor: float = 1e-12) -> float:
    """Total ideal bits for integer `symbols` under a per-channel Laplace.

    The probability MASS of the unit-width bin centred on each integer, which is
    what a real coder would need - a density value is not a probability, and
    using one would understate the rate.
    """
    flat = np.asarray(symbols).reshape(symbols.shape[0], channels, -1).transpose(1, 0, 2)
    flat = flat.reshape(channels, -1).astype(np.float64)
    total = 0.0
    for channel in range(channels):
        upper = _laplace_cdf(flat[channel] + 0.5, location[channel], scale[channel])
        lower = _laplace_cdf(flat[channel] - 0.5, location[channel], scale[channel])
        total += float(-np.log2(np.maximum(upper - lower, floor)).sum())
    return total


def histogram_bits(symbols: np.ndarray, probabilities: np.ndarray, *,
                   channels: int) -> float:
    flat = symbols.reshape(symbols.shape[0], channels, -1)
    total = 0.0
    for channel in range(channels):
        picked = probabilities[channel][flat[:, channel, :].reshape(-1)]
        total += float(-np.log2(np.maximum(picked, 1e-300)).sum())
    return total


# --------------------------------------------------------------------------------------------
# The two grid rules
# --------------------------------------------------------------------------------------------


def deployed_grid(train_latents: torch.Tensor, *, bits: int):
    """Our intra quantizer: per-channel affine, fitted to TRAIN percentiles."""
    return calibrate_quantization_params(
        train_latents, bits=bits, mode="per_channel",
        lower_percentile=LOWER_PERCENTILE, upper_percentile=UPPER_PERCENTILE)


def grid_symbols(latents: torch.Tensor, params) -> np.ndarray:
    """[N, C, H, W] -> [N, C*H*W] symbols, frame by frame as the coder does."""
    return np.stack([latent_to_symbols(latents[index:index + 1], params)
                     for index in range(latents.shape[0])])


def grid_reconstruct(symbols: np.ndarray, params, shape) -> torch.Tensor:
    return torch.cat([symbols_to_latent(symbols[index], shape, params)
                      for index in range(symbols.shape[0])], dim=0)


def reference_step(train_latents: torch.Tensor, *, bits: int = 4) -> float:
    """The median per-channel step the deployed grid uses at `bits`.

    The unit-step sweep is expressed as multiples of this so the two arms are
    guaranteed to overlap in quality, which is what makes a BD-rate between them
    an interpolation.
    """
    params = deployed_grid(train_latents, bits=bits)
    return float(params.scale.flatten().median())


def unit_step_symbols(latents: torch.Tensor, *, step: float) -> np.ndarray:
    """bmshj2018's rule: round to integers at a fixed step, nothing clipped.

    No percentile range and no level cap, so a channel costs what its own
    distribution costs rather than what its range implies.
    """
    rounded = torch.round(latents / step)
    return rounded.reshape(latents.shape[0], -1).cpu().numpy().astype(np.int64)


def unit_step_reconstruct(symbols: np.ndarray, *, step: float, shape) -> torch.Tensor:
    values = torch.from_numpy(np.asarray(symbols, dtype=np.float32)) * step
    return values.reshape(symbols.shape[0], *shape)


# --------------------------------------------------------------------------------------------
# One operating point
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def measure_point(model, bp, arm: str, *, train_latents: torch.Tensor,
                  val_latents: torch.Tensor, val_frames: torch.Tensor, setting: float,
                  device, batch: int = 8) -> dict[str, Any]:
    """Rate and distortion for one arm at one operating point.

    Rate: ideal bits per pixel under a TRAIN-fitted density. Distortion:
    decoded-pixel PSNR and MS-SSIM through `benchmark_parity`'s scorer, so the
    numbers share axes with every other measurement in the project.
    """
    channels = train_latents.shape[1]
    shape = tuple(train_latents.shape[1:])

    if arm.startswith("grid"):
        bits = int(setting)
        params = deployed_grid(train_latents, bits=bits)
        train_symbols = grid_symbols(train_latents, params)
        val_symbols = grid_symbols(val_latents, params)
        reconstructed = grid_reconstruct(val_symbols, params, shape)
        alphabet = 2 ** bits
        if arm == "grid_histogram":
            probabilities = fit_histogram(train_symbols, alphabet=alphabet,
                                          channels=channels)
            total_bits = histogram_bits(val_symbols, probabilities, channels=channels)
            density = "per-channel add-one histogram"
        else:
            location, scale = fit_laplace(train_symbols, channels=channels)
            total_bits = discretised_laplace_bits(val_symbols, location, scale,
                                                  channels=channels)
            density = "per-channel discretised Laplace"
        clipped = float(np.mean((val_symbols == 0) | (val_symbols == alphabet - 1)))
        extra = {"bits": bits, "alphabet": alphabet,
                 "step_median": float(params.scale.flatten().median()),
                 "at_alphabet_edge_fraction": clipped}
    else:
        step = float(setting)
        train_symbols = unit_step_symbols(train_latents, step=step)
        val_symbols = unit_step_symbols(val_latents, step=step)
        reconstructed = unit_step_reconstruct(val_symbols, step=step, shape=shape)
        location, scale = fit_laplace(train_symbols, channels=channels)
        total_bits = discretised_laplace_bits(val_symbols, location, scale,
                                              channels=channels)
        density = "per-channel discretised Laplace"
        span = int(np.abs(val_symbols).max())
        extra = {"step": step, "symbol_span": span,
                 "effective_bits": math.log2(2 * span + 1) if span else 0.0,
                 "distinct_symbols": int(np.unique(val_symbols).size)}

    decoded = []
    for start in range(0, reconstructed.shape[0], batch):
        chunk = reconstructed[start:start + batch].to(device)
        decoded.append(model.decode(chunk).detach().cpu())
    decoded = torch.cat(decoded, dim=0)
    scores = bp.score_frames(bp.to_uint8_frames(decoded), val_frames, device=device)

    pixels = int(val_frames.shape[0] * val_frames.shape[2] * val_frames.shape[3])
    return {
        "arm": arm, "setting": setting, "density": density,
        "total_ideal_bits": total_bits,
        "bpp": total_bits / pixels,
        "bits_per_symbol": total_bits / val_symbols.size,
        "psnr": float(np.mean(scores["psnr"])),
        "msssim": float(np.mean(scores["msssim"])),
        "symbols": int(val_symbols.size), "pixels": pixels,
        **extra,
    }
