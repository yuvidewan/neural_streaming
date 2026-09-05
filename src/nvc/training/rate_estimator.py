"""Differentiable per-channel Laplace rate proxy for training (Milestone 9A).

TRAINING vs. DEPLOYMENT - read this before touching either
-------------------------------------------------------------
    TRAINING (this module):
        A differentiable, per-channel, LEARNED Laplace density evaluated on
        the (possibly QAT-noised) latent, used ONLY to produce a
        training-time rate gradient. It has never seen real symbol
        statistics beyond whatever gradient descent pushes it toward.

    DEPLOYMENT (unchanged, not touched by this module):
        `nvc.compression.entropy_model.EmpiricalEntropyModel` - a static,
        non-differentiable, per-channel INTEGER histogram calibrated
        post-hoc from real symbol counts - plus the real arithmetic/range
        coder. This module is never imported by, invoked by, or wired into
        the `.nvc` encode/decode path. It exists purely to give the
        training loss a rate signal; the actual coded bits still come
        entirely from the existing calibration + entropy coder, exactly as
        before M9.

WHY LAPLACE, WHY PER-CHANNEL, WHY THIS BIN WIDTH
-------------------------------------------------
Matches the STRUCTURE of the deployed model as closely as a differentiable
stand-in can:
  - One distribution per latent channel, mirroring EmpiricalEntropyModel's
    one-table-per-channel design (`nvc.compression.calibration`'s own
    docstring: channels have visibly different distributions).
  - Laplace, not Gaussian: `calibration.py` documents this project's real
    latent as "a sharply peaked latent distribution with long tails" - the
    textbook case for Laplace over Gaussian.
  - The bin width used to turn a continuous density into a probability MASS
    is the *same* `scale` tensor `QuantizationNoise` already loads from an
    existing calibration file - not a second, independently invented bin
    width. When both QAT and rate training are active, the caller passes
    `quantization_noise.scale` directly (see `RateEstimator.__init__`'s
    `bin_width` argument) - see scripts/train_autoencoder.py.

THE MATHEMATICS
----------------
For a per-channel Laplace(loc, scale) and a bin of width `w` centered on a
latent value `z`, the probability MASS assigned to that bin is the exact
CDF difference:

    P(z) = F(z + w/2) - F(z - w/2)

which is what `R = -sum(log2(P(symbol)))` needs - not the density itself.
Writing the Laplace CDF as an offset from its own center,

    F(loc + t) = 0.5 + sign(t) * (1 - exp(-|t| / scale)) / 2

lets the bin-probability difference be computed as G(t_upper) - G(t_lower)
with G(t) = sign(t) * (1 - exp(-|t|/scale)) / 2 - the "0.5 +" cancels
ALGEBRAICALLY before any floating-point subtraction happens, rather than
computing two numbers near 1.0 (or near 0.0) and subtracting them (see
`_half_cdf_offset` below; this is the numerically stable form the M9A brief
requires - `torch.expm1` gives `1 - exp(-x)` full precision for small x,
and for large x the exponential term underflows cleanly to 0 with no
cancellation either way). `torch.clamp(..., min=_MIN_PROBABILITY)` before
`-log2` guarantees a finite rate for every input, mirroring
`entropy_model.py`'s own zero-probability safety philosophy (Laplace
smoothing there; an epsilon floor here, same purpose).

A note on `torch.sign`'s zero gradient: `_half_cdf_offset` multiplies by
`sign(t)` explicitly AND depends on `|t|` (whose own gradient is `sign(t)`),
so the two `sign(t)` factors combine to `sign(t)**2 == 1` almost everywhere
under autograd - the true Laplace density is recovered exactly as the
gradient w.r.t. `t`, not zeroed out. (The single point `t == 0` gets
gradient 0 instead of the analytic `1/(2*scale)` - a measure-zero
discrepancy, not a practical concern.)

NORMALIZATION
--------------
`forward()` returns MEAN BITS-PER-PIXEL over the batch, normalized by the
INPUT IMAGE's pixel count (`height * width`, no channel factor) - this
matches `nvc.evaluation.rd_benchmark`'s `aggregate_bpp` convention exactly
(`sum(bytes) * 8 / sum(width * height)`), so this training-time number is
unit-comparable to the real measured payload BPP reported by
`benchmark_rd.py`, even though the two are not expected to match exactly
(see this module's own docstring section above).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from nvc.compression.calibration import load_calibration
from nvc.compression.quantization import QuantizationParams

# Floor applied to the estimated bin probability before -log2, so the rate
# is finite for every input - including latent values far in the tail of a
# poorly-fit density, where the true bin probability may underflow to 0.
_MIN_PROBABILITY = 1e-9


def _half_cdf_offset(t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """F(loc + t) - 0.5 for a Laplace(loc, scale), as a function of the
    offset `t` from `loc` - see this module's docstring for the derivation
    and why this form avoids subtracting two near-equal CDF values."""
    return torch.sign(t) * (-torch.expm1(-t.abs() / scale)) * 0.5


class RateEstimator(nn.Module):
    """Learned per-channel Laplace rate proxy, evaluated on a latent tensor.

    A real `nn.Module` (unlike its sibling `QuantizationNoise`) because it
    genuinely has learnable parameters (`loc`, `log_scale`) that must
    receive gradients and appear in the training optimizer's parameter
    list. It is never attached to `BaselineAutoencoder` and never enters
    that model's `state_dict()` - inference never instantiates or requires
    this class (see `checkpoint.py`'s `extra` field for how a rate-trained
    checkpoint still loads through the ordinary inference path).

    SCALE TRACKING (post-M9F fix for MILESTONE_9_PLAN.md Section 9F.5)
    ---------------------------------------------------------------------
    Section 9F.5 measured, on the real deployed codec, why a rate term can
    stop improving anything past a certain lambda: the real quantizer
    RECALIBRATES its grid to each model's own latent range
    (`calibrate_quantization_params`), so a model that simply shrinks its
    latent uniformly pays almost nothing extra on the real coded bitrate -
    the grid shrinks right along with it. But this class's `bin_width` used
    to be frozen once, at construction, from a calibration file computed
    before training started. Against a FIXED bin width, that same uniform
    shrinkage reads as a large rate reduction, so the training signal
    rewarded exactly the one strategy the real codec does not reward.

    `track_scale=True` closes that gap: `update_bin_width()` (called once
    per training step, after scoring, from `train_one_epoch_with_rate`)
    re-derives `bin_width` toward the CURRENT batch's own dynamic range -
    using the same min/max-over-levels formula
    `UniformQuantizer.compute_params` already implements for exactly this
    ("dynamic" per-tensor calibration) - EMA-smoothed across steps for
    stability. This makes the proxy scale-aware, matching the deployed
    pipeline's own behavior, instead of scale-sensitive where the deployed
    pipeline is scale-invariant.

    Default is `track_scale=False`: every existing M9A/M9C/M9C.1 tested
    behavior (a frozen bin width) is unchanged unless explicitly opted into.
    """

    def __init__(
        self,
        bin_width: torch.Tensor,
        *,
        bits: int,
        mode: str,
        track_scale: bool = False,
        scale_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        if not torch.is_floating_point(bin_width):
            raise TypeError(f"bin_width must be a floating-point tensor, got dtype {bin_width.dtype}")
        if not torch.isfinite(bin_width).all() or (bin_width <= 0).any():
            raise ValueError("bin_width must be finite and strictly positive everywhere")
        if bin_width.dim() != 4 or bin_width.shape[0] != 1 or bin_width.shape[2:] != (1, 1):
            raise ValueError(
                f"bin_width must broadcast against a [B, C, H, W] latent, i.e. have "
                f"shape [1, C, 1, 1] (or [1, 1, 1, 1]), got {tuple(bin_width.shape)}"
            )
        if not 1 <= bits <= 16:
            raise ValueError(f"bits must be in [1, 16], got {bits}")
        if not 0.0 <= scale_momentum < 1.0:
            raise ValueError(f"scale_momentum must be in [0, 1), got {scale_momentum}")

        # A buffer, not a Parameter - moves with .to(device) and is saved in
        # this module's own state_dict, but is never touched by the
        # optimizer (it is either frozen, or updated in-place by
        # update_bin_width's own EMA rule, never by a gradient step).
        self.register_buffer("bin_width", bin_width.detach().clone())
        self.bits = bits
        self.mode = mode
        self.track_scale = track_scale
        self.scale_momentum = scale_momentum

        num_channels = bin_width.shape[1]
        # log_scale, not scale directly: exp(log_scale) guarantees a
        # strictly positive Laplace scale for every possible parameter
        # value, with no separate clamp needed. Initialized to log(1) = 0
        # (scale = 1) and loc = 0 - arbitrary but harmless starting points;
        # both are learned from the very first optimizer step.
        self.loc = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    @classmethod
    def from_calibration(
        cls,
        path: str | Path,
        *,
        bits: int | None = None,
        mode: str | None = None,
        track_scale: bool = False,
        scale_momentum: float = 0.99,
    ) -> "RateEstimator":
        """Build a STANDALONE rate estimator (rate training without QAT)
        from a calibration file - same validation as
        `QuantizationNoise.from_calibration`, deliberately kept in sync
        with it since both must reject the same kinds of mismatches.

        When QAT IS also enabled for the same training run, do NOT use this
        - construct `RateEstimator(quantization_noise.scale, bits=..., mode=...)`
        directly instead, so the two components share the literal same
        bin-width tensor rather than two independently loaded copies of it
        (see this module's docstring, "why this bin width").
        """
        document = load_calibration(path)
        params = QuantizationParams.from_dict(document["quantization"])
        if bits is not None and bits != params.bits:
            raise ValueError(
                f"Calibration {path} was computed for {params.bits}-bit quantization, "
                f"but bits={bits} was requested. Generate a calibration at the bit "
                f"depth you intend to train for, or omit `bits` to use the file's own."
            )
        if mode is not None and mode != params.mode:
            raise ValueError(
                f"Calibration {path} was computed in {params.mode!r} mode, but "
                f"mode={mode!r} was requested. The rate estimator's bin width must "
                "match the quantization mode actually used by the codec - generate a "
                "calibration in that mode, or omit `mode` to use the file's own."
            )
        if document["calibration_metadata"].get("calibration_split") != "train":
            raise ValueError(
                f"Calibration {path} was not computed from the 'train' split "
                f"(calibration_split={document['calibration_metadata'].get('calibration_split')!r}). "
                "The rate estimator's bin width must come from train-split calibration only."
            )
        return cls(
            params.scale, bits=params.bits, mode=params.mode,
            track_scale=track_scale, scale_momentum=scale_momentum,
        )

    @torch.no_grad()
    def update_bin_width(self, z: torch.Tensor) -> None:
        """Nudge `bin_width` toward the CURRENT latent's own dynamic range,
        EMA-smoothed. No-op unless `track_scale=True` was passed at
        construction (see this class's docstring, "SCALE TRACKING").

        Uses the same min/max-over-levels formula
        `UniformQuantizer.compute_params` already implements for "dynamic"
        (per-tensor, calibration-free) quantization - not a new estimation
        method, a reuse of one this project already has, tested, and trusts.
        `z` should be the same latent just scored by `rate_bits`/`forward`
        (the noised latent, if QAT is active) - call this AFTER scoring, so
        a step's own rate is measured against the bin width it was told
        about, not one updated mid-step.
        """
        if not self.track_scale:
            return
        if z.dim() != 4:
            raise ValueError(f"Expected a 4D [B, C, H, W] latent tensor, got shape {tuple(z.shape)}")

        q_levels = 2 ** self.bits - 1
        if self.mode == "global":
            instantaneous = (z.amax() - z.amin()) / q_levels
            instantaneous = instantaneous.reshape(1, 1, 1, 1).expand_as(self.bin_width)
        else:
            instantaneous = (z.amax(dim=(0, 2, 3)) - z.amin(dim=(0, 2, 3))) / q_levels
            instantaneous = instantaneous.reshape(1, -1, 1, 1)
        # A degenerate (zero-width) batch range would otherwise EMA the bin
        # width toward 0 and eventually make rate_bits() rely entirely on
        # the epsilon floor - clamped away, mirroring quantization.py's own
        # degenerate-range handling in spirit (never let a grid collapse).
        instantaneous = torch.clamp(instantaneous, min=1e-6).to(
            device=self.bin_width.device, dtype=self.bin_width.dtype
        )

        self.bin_width.mul_(self.scale_momentum).add_(instantaneous, alpha=1.0 - self.scale_momentum)

    def rate_bits(self, z: torch.Tensor) -> torch.Tensor:
        """Per-element estimated code length in bits, same shape as `z`.

        Vectorized over the full [B, C, H, W] tensor - no Python loop over
        channels or symbols.
        """
        if z.dim() != 4:
            raise ValueError(f"Expected a 4D [B, C, H, W] latent tensor, got shape {tuple(z.shape)}")

        bin_width = self.bin_width.to(device=z.device, dtype=z.dtype)
        loc = self.loc.to(device=z.device, dtype=z.dtype)
        scale = torch.exp(self.log_scale).to(device=z.device, dtype=z.dtype)

        half = bin_width * 0.5
        upper_offset = z + half - loc
        lower_offset = z - half - loc

        probability = _half_cdf_offset(upper_offset, scale) - _half_cdf_offset(lower_offset, scale)
        probability = torch.clamp(probability, min=_MIN_PROBABILITY)
        return -torch.log2(probability)

    def forward(self, z: torch.Tensor, image_pixels: int) -> torch.Tensor:
        """Mean bits-per-pixel over the batch. `image_pixels` is the INPUT
        IMAGE's height*width (e.g. 256*256 = 65536) - the caller's actual
        batch shape, not assumed or hardcoded here, so this stays correct
        under any crop size."""
        if image_pixels <= 0:
            raise ValueError(f"image_pixels must be positive, got {image_pixels}")
        bits = self.rate_bits(z)
        total_bits_per_sample = bits.sum(dim=(1, 2, 3))
        return (total_bits_per_sample / image_pixels).mean()

    def to_dict(self) -> dict[str, Any]:
        """Summary for training-history logging - never used to rebuild state."""
        return {
            "bits": self.bits,
            "mode": self.mode,
            "num_channels": int(self.bin_width.shape[1]),
            "track_scale": self.track_scale,
            "scale_momentum": self.scale_momentum if self.track_scale else None,
        }
