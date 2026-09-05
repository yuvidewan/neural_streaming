"""Uniform scalar (affine) quantization of latent tensors.

Deliberately written out longhand rather than delegating to
`torch.ao.quantization` - the whole point of this milestone is to measure
and reason about the quantization mathematics, which a library that hides
the scale/zero-point derivation would obscure.

THE MATHEMATICS
---------------
For a float tensor `x` and a bit width `b`, the representable integer grid is

    q_min = 0
    q_max = 2**b - 1

Given an observed value range [x_min, x_max] (how it is observed is what
distinguishes the two modes below):

    scale      = (x_max - x_min) / (q_max - q_min)
    zero_point = q_min - round(x_min / scale)

Quantize (float -> integer):

    q = clamp(round(x / scale) + zero_point, q_min, q_max)

Dequantize (integer -> approximate float):

    x_hat = (q - zero_point) * scale

`zero_point` is deliberately NOT clamped into [q_min, q_max]. It is stored
as separate metadata, not packed into the same integer field as `q`, so
clamping it would be a pointless self-inflicted constraint - and it would
break ranges that do not straddle zero (e.g. a channel whose values all sit
near 5.0, where the correct zero_point is a large negative number).
Leaving it unclamped means x_hat reproduces such ranges correctly, and
also makes exact 0.0 exactly representable whenever 0 lies inside
[x_min, x_max].

DEGENERATE RANGE
----------------
If x_max == x_min (a constant tensor or constant channel), the range has
zero width and `scale` would be 0, giving a division by zero. The range is
therefore symmetrically widened to [c - 0.5, c + 0.5] around the constant
`c`, which yields a well-defined scale and reconstructs `c` to within one
quantization step instead of producing NaN or inf.

CALIBRATION SCOPE (an honest caveat)
------------------------------------
Scale and zero_point here are derived from the very tensor being quantized
("dynamic" quantization). That is the right choice for *measuring* how much
distortion the quantization grid itself introduces, which is what this
milestone is for. A real codec would have to transmit these parameters as
side information, or derive them once from a calibration set and freeze
them; neither is implemented yet, and the storage figures reported by
`storage_analysis.py` explicitly exclude that metadata cost.

PER-CHANNEL BIT ALLOCATION (optional, OPTIMIZATION_ANALYSIS.md Q3)
--------------------------------------------------------------------
`QuantizationParams.bits_per_channel`, when set, lets each channel use FEWER
levels than the table's own `bits` (e.g. table bits=8 but a low-information
channel allocated only 4) - a genuine distortion/rate trade at the SAME
average bit budget, reallocating precision from flat channels (extra
resolution buys them little) to high-variance ones (extra resolution buys
them more), the classical water-filling bit-allocation result
(`bits_c = bits_avg + 0.5*log2(var_c / geometric_mean(var))`; see
`calibration.allocate_bits_per_channel`).

Deliberately does NOT touch the entropy table, the arithmetic coder, or the
`.nvc` format: the table stays sized at the ordinary `bits` (2**bits
symbols) for every channel, and a channel allocated fewer bits simply never
produces symbol values above its own smaller `2**bits_c - 1` - the EXISTING
Laplace-smoothed entropy model already prices a symbol that never occurs
correctly, with no format change needed. A decoder needs no extra
information either: dequantization is `(q - zero_point) * scale`, exactly
as today, regardless of how many levels the encoder chose to use.

NON-UNIFORM QUANTIZATION / COMPANDING (optional, OPTIMIZATION_ANALYSIS.md Q2)
--------------------------------------------------------------------------------
`QuantizationParams.companding_gamma`, when set, applies a power-law
transform `y = sign(x) * |x|^gamma` before the ordinary uniform grid (and
its exact inverse `x = sign(y) * |y|^(1/gamma)` after dequantizing).
`calibration.py`'s own docstring already documents this project's latent as
"a sharply peaked distribution with long tails" - equal-width bins waste
most of their resolution on the near-empty tails. A `gamma < 1` compresses
large |x| together (coarser bins far from zero) while expanding small |x|
apart (finer bins near zero, where almost all the density is) - the
standard mu-law-style companding idea, chosen over an iterative Lloyd-Max
fit for simplicity: one parameter, monotonic, and it targets exactly the
peaked-with-long-tails shape already measured, not a distribution shape
that would need iterative fitting to characterize.

Like `bits_per_channel`, this needs no `.nvc` format change: `scale`/
`zero_point` are computed IN THE COMPANDED DOMAIN and stored exactly as
before; a decoder that has the calibration file (which now also carries
`companding_gamma`) applies the same inverse transform, with the same
per-channel `scale`/`zero_point` fields the format already carries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

QUANTIZATION_MODES = ("global", "per_channel")

# Widening applied to a zero-width range so `scale` never becomes 0.
_DEGENERATE_RANGE_HALF_WIDTH = 0.5


def _compand(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """y = sign(x) * |x|^gamma - see this module's docstring, "NON-UNIFORM
    QUANTIZATION". Exactly the identity when gamma == 1.0."""
    return torch.sign(x) * x.abs().pow(gamma)


def _expand(y: torch.Tensor, gamma: float) -> torch.Tensor:
    """Exact inverse of `_compand`: x = sign(y) * |y|^(1/gamma)."""
    return torch.sign(y) * y.abs().pow(1.0 / gamma)


@dataclass(frozen=True)
class QuantizationParams:
    """Affine quantization parameters plus the grid they describe.

    `scale` and `zero_point` are broadcastable against the tensor they were
    derived from: shape [1, 1, 1, 1] in global mode, [1, C, 1, 1] in
    per-channel mode.

    `bits_per_channel` and `companding_gamma` are both optional and default
    to `None` (today's exact behavior - a single uniform grid, no
    transform). See this module's docstring for what each does and why
    neither needs a `.nvc` format change.
    """

    scale: torch.Tensor
    zero_point: torch.Tensor
    bits: int
    mode: str
    bits_per_channel: tuple[int, ...] | None = None
    companding_gamma: float | None = None

    def __post_init__(self) -> None:
        if self.bits_per_channel is not None:
            if self.mode != "per_channel":
                raise ValueError("bits_per_channel is only valid in 'per_channel' mode")
            if len(self.bits_per_channel) != self.scale.shape[1]:
                raise ValueError(
                    f"bits_per_channel has {len(self.bits_per_channel)} entries but scale "
                    f"has {self.scale.shape[1]} channels"
                )
            for channel_bits in self.bits_per_channel:
                if not 1 <= channel_bits <= self.bits:
                    raise ValueError(
                        f"each bits_per_channel value must be in [1, {self.bits}] "
                        f"(the table's own bit depth), got {channel_bits}"
                    )
        if self.companding_gamma is not None:
            if not math.isfinite(self.companding_gamma) or self.companding_gamma <= 0:
                raise ValueError(
                    f"companding_gamma must be finite and > 0, got {self.companding_gamma}"
                )

    @property
    def q_min(self) -> int:
        return 0

    @property
    def q_max(self) -> int:
        """The TABLE's max code, i.e. the entropy model's alphabet ceiling -
        always `2**bits - 1` regardless of `bits_per_channel`. Use
        `effective_q_max` for the per-channel clamp bound actually applied
        during quantization."""
        return 2 ** self.bits - 1

    @property
    def effective_q_max(self) -> "int | torch.Tensor":
        """The clamp bound `quantize()` actually applies: `q_max` broadcast
        as today, unless `bits_per_channel` narrows it per channel."""
        if self.bits_per_channel is None:
            return self.q_max
        levels = torch.tensor(
            [2 ** b - 1 for b in self.bits_per_channel],
            dtype=self.scale.dtype, device=self.scale.device,
        )
        return levels.reshape(1, -1, 1, 1)

    @property
    def num_levels(self) -> int:
        return 2 ** self.bits

    def to(self, device: str | torch.device) -> "QuantizationParams":
        """Move scale/zero_point onto `device` (they must match the latent)."""
        return QuantizationParams(
            scale=self.scale.to(device),
            zero_point=self.zero_point.to(device),
            bits=self.bits,
            mode=self.mode,
            bits_per_channel=self.bits_per_channel,
            companding_gamma=self.companding_gamma,
        )

    def to_dict(self) -> dict:
        """JSON-serializable form, for writing a calibration file."""
        return {
            "bits": self.bits,
            "mode": self.mode,
            "scale": self.scale.flatten().tolist(),
            "zero_point": self.zero_point.flatten().tolist(),
            "bits_per_channel": (
                list(self.bits_per_channel) if self.bits_per_channel is not None else None
            ),
            "companding_gamma": self.companding_gamma,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuantizationParams":
        """Rebuild from `to_dict()` output (the decoder's entry point).

        `bits_per_channel`/`companding_gamma` are read with `.get(..., None)`
        so calibration files written before either feature existed (every
        M7/M8/M9 calibration on disk) still load unchanged, with both
        correctly defaulting to "off".
        """
        mode = data["mode"]
        if mode not in QUANTIZATION_MODES:
            raise ValueError(f"mode must be one of {QUANTIZATION_MODES}, got {mode!r}")

        scale = torch.tensor(data["scale"], dtype=torch.float32).reshape(1, -1, 1, 1)
        zero_point = torch.tensor(data["zero_point"], dtype=torch.float32).reshape(1, -1, 1, 1)
        if scale.shape != zero_point.shape:
            raise ValueError("scale and zero_point must have the same number of entries")
        if mode == "global" and scale.numel() != 1:
            raise ValueError(f"global mode expects exactly 1 scale, got {scale.numel()}")
        if (scale <= 0).any():
            raise ValueError("all scale values must be strictly positive")

        bits_per_channel = data.get("bits_per_channel")
        return cls(
            scale=scale, zero_point=zero_point, bits=int(data["bits"]), mode=mode,
            bits_per_channel=tuple(bits_per_channel) if bits_per_channel is not None else None,
            companding_gamma=data.get("companding_gamma"),
        )


class UniformQuantizer:
    """Uniform scalar affine quantizer for [B, C, H, W] latent tensors.

    Two modes, which this milestone measures against each other rather than
    assuming a winner:

    - "global":      one (scale, zero_point) pair for the entire tensor.
                     Cheapest metadata; a single channel with an unusually
                     wide range stretches the grid for every other channel.
    - "per_channel": an independent (scale, zero_point) pair per latent
                     channel, computed over the (B, H, W) axes. Adapts to
                     channels with very different dynamic ranges, at the
                     cost of C times as many parameters to carry.
    """

    def __init__(self, bits: int, mode: str = "global", *, companding_gamma: float | None = None) -> None:
        if not isinstance(bits, int) or isinstance(bits, bool):
            raise TypeError(f"bits must be an int, got {type(bits).__name__}")
        if not 1 <= bits <= 16:
            raise ValueError(f"bits must be in [1, 16], got {bits}")
        if mode not in QUANTIZATION_MODES:
            raise ValueError(f"mode must be one of {QUANTIZATION_MODES}, got {mode!r}")
        if companding_gamma is not None and (not math.isfinite(companding_gamma) or companding_gamma <= 0):
            raise ValueError(f"companding_gamma must be finite and > 0, got {companding_gamma}")

        self.bits = bits
        self.mode = mode
        self.companding_gamma = companding_gamma

    def compute_params(self, x: torch.Tensor) -> QuantizationParams:
        """Derive (scale, zero_point) from the observed range of `x`."""
        _validate_latent(x)
        x_transformed = _compand(x, self.companding_gamma) if self.companding_gamma is not None else x

        if self.mode == "global":
            x_min = x_transformed.min().reshape(1, 1, 1, 1)
            x_max = x_transformed.max().reshape(1, 1, 1, 1)
        else:
            # amin/amax over batch, height, width -> one value per channel.
            x_min = x_transformed.amin(dim=(0, 2, 3)).reshape(1, -1, 1, 1)
            x_max = x_transformed.amax(dim=(0, 2, 3)).reshape(1, -1, 1, 1)

        # Widen any zero-width range so scale is strictly positive.
        degenerate = x_max == x_min
        if degenerate.any():
            x_min = torch.where(degenerate, x_min - _DEGENERATE_RANGE_HALF_WIDTH, x_min)
            x_max = torch.where(degenerate, x_max + _DEGENERATE_RANGE_HALF_WIDTH, x_max)

        q_min, q_max = 0, 2 ** self.bits - 1
        scale = (x_max - x_min) / (q_max - q_min)
        zero_point = q_min - torch.round(x_min / scale)

        return QuantizationParams(
            scale=scale, zero_point=zero_point, bits=self.bits, mode=self.mode,
            companding_gamma=self.companding_gamma,
        )

    def quantize(
        self, x: torch.Tensor, params: QuantizationParams | None = None
    ) -> tuple[torch.Tensor, QuantizationParams]:
        """float -> integer grid. Returns (integer tensor, params used).

        Pass `params` to reuse a previously derived grid (e.g. to quantize
        several tensors identically); omit it to calibrate on `x` itself.
        `params` (not `self`) is authoritative for companding/bit-allocation,
        exactly like it already is for scale/zero_point/bits/mode - a
        caller can quantize with parameters this instance did not compute.
        """
        _validate_latent(x)
        if params is None:
            params = self.compute_params(x)

        x_transformed = (
            _compand(x, params.companding_gamma) if params.companding_gamma is not None else x
        )
        q = torch.round(x_transformed / params.scale) + params.zero_point
        q = torch.clamp(q, min=params.q_min)
        q_max = params.effective_q_max
        # torch.clamp refuses a scalar min mixed with a tensor max (only
        # matching types are allowed) - q_min is always the scalar 0, so
        # branch on q_max's type instead of forcing q_min into a tensor.
        q = torch.clamp(q, max=q_max) if isinstance(q_max, int) else torch.minimum(q, q_max)
        return q.to(torch.int32), params

    def dequantize(self, q: torch.Tensor, params: QuantizationParams) -> torch.Tensor:
        """integer grid -> approximate float."""
        x_hat = (q.to(params.scale.dtype) - params.zero_point) * params.scale
        if params.companding_gamma is not None:
            x_hat = _expand(x_hat, params.companding_gamma)
        return x_hat

    def quantize_dequantize(
        self, x: torch.Tensor, params: QuantizationParams | None = None
    ) -> tuple[torch.Tensor, QuantizationParams]:
        """Full round trip: float -> integer -> approximate float.

        This is what the end-to-end experiment feeds back into the decoder.
        """
        q, params = self.quantize(x, params)
        return self.dequantize(q, params), params


def count_clipped(x: torch.Tensor, params: QuantizationParams) -> dict[str, float]:
    """How many values fall outside a fixed quantization grid and get clamped.

    Only meaningful for externally calibrated (fixed) params: when the grid
    is derived from `x` itself nothing can fall outside it. With percentile
    calibration some clipping is expected and intentional - this quantifies
    it so the trade-off is measured rather than assumed.

    Uses `effective_q_max` (not the table-wide `q_max`), so a channel given
    fewer levels via `bits_per_channel` is correctly scored against its own
    smaller ceiling, and companded `x` if `params.companding_gamma` is set -
    both match exactly what `UniformQuantizer.quantize` actually applies.
    """
    x_transformed = _compand(x, params.companding_gamma) if params.companding_gamma is not None else x
    unclamped = torch.round(x_transformed / params.scale) + params.zero_point
    below = (unclamped < params.q_min).sum().item()
    above = (unclamped > params.effective_q_max).sum().item()
    total = x.numel()
    return {
        "clipped_low": below,
        "clipped_high": above,
        "clipped_total": below + above,
        "total_values": total,
        "clipped_percent": 100.0 * (below + above) / total,
    }


def quantization_error(original: torch.Tensor, dequantized: torch.Tensor) -> dict[str, float]:
    """Error between a latent and its dequantized reconstruction.

    This is LATENT-space error, not image-space error - the two are
    reported separately throughout this milestone and must not be conflated.
    """
    difference = dequantized - original
    return {
        "latent_mse": torch.mean(difference ** 2).item(),
        "latent_mae": torch.mean(torch.abs(difference)).item(),
        "latent_max_abs_error": torch.max(torch.abs(difference)).item(),
    }


def _validate_latent(x: torch.Tensor) -> None:
    if not torch.is_floating_point(x):
        raise TypeError(f"Quantizer expects a floating-point tensor, got dtype {x.dtype}")
    if x.dim() != 4:
        raise ValueError(
            f"Quantizer expects a 4D [B, C, H, W] latent tensor, got shape {tuple(x.shape)}"
        )
    if not torch.isfinite(x).all():
        raise ValueError("Quantizer input contains NaN or inf values")
