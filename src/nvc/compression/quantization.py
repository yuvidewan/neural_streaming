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
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

QUANTIZATION_MODES = ("global", "per_channel")

# Widening applied to a zero-width range so `scale` never becomes 0.
_DEGENERATE_RANGE_HALF_WIDTH = 0.5


@dataclass(frozen=True)
class QuantizationParams:
    """Affine quantization parameters plus the grid they describe.

    `scale` and `zero_point` are broadcastable against the tensor they were
    derived from: shape [1, 1, 1, 1] in global mode, [1, C, 1, 1] in
    per-channel mode.
    """

    scale: torch.Tensor
    zero_point: torch.Tensor
    bits: int
    mode: str

    @property
    def q_min(self) -> int:
        return 0

    @property
    def q_max(self) -> int:
        return 2 ** self.bits - 1

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
        )

    def to_dict(self) -> dict:
        """JSON-serializable form, for writing a calibration file."""
        return {
            "bits": self.bits,
            "mode": self.mode,
            "scale": self.scale.flatten().tolist(),
            "zero_point": self.zero_point.flatten().tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuantizationParams":
        """Rebuild from `to_dict()` output (the decoder's entry point)."""
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

        return cls(scale=scale, zero_point=zero_point, bits=int(data["bits"]), mode=mode)


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

    def __init__(self, bits: int, mode: str = "global") -> None:
        if not isinstance(bits, int) or isinstance(bits, bool):
            raise TypeError(f"bits must be an int, got {type(bits).__name__}")
        if not 1 <= bits <= 16:
            raise ValueError(f"bits must be in [1, 16], got {bits}")
        if mode not in QUANTIZATION_MODES:
            raise ValueError(f"mode must be one of {QUANTIZATION_MODES}, got {mode!r}")

        self.bits = bits
        self.mode = mode

    def compute_params(self, x: torch.Tensor) -> QuantizationParams:
        """Derive (scale, zero_point) from the observed range of `x`."""
        _validate_latent(x)

        if self.mode == "global":
            x_min = x.min().reshape(1, 1, 1, 1)
            x_max = x.max().reshape(1, 1, 1, 1)
        else:
            # amin/amax over batch, height, width -> one value per channel.
            x_min = x.amin(dim=(0, 2, 3)).reshape(1, -1, 1, 1)
            x_max = x.amax(dim=(0, 2, 3)).reshape(1, -1, 1, 1)

        # Widen any zero-width range so scale is strictly positive.
        degenerate = x_max == x_min
        if degenerate.any():
            x_min = torch.where(degenerate, x_min - _DEGENERATE_RANGE_HALF_WIDTH, x_min)
            x_max = torch.where(degenerate, x_max + _DEGENERATE_RANGE_HALF_WIDTH, x_max)

        q_min, q_max = 0, 2 ** self.bits - 1
        scale = (x_max - x_min) / (q_max - q_min)
        zero_point = q_min - torch.round(x_min / scale)

        return QuantizationParams(
            scale=scale, zero_point=zero_point, bits=self.bits, mode=self.mode
        )

    def quantize(
        self, x: torch.Tensor, params: QuantizationParams | None = None
    ) -> tuple[torch.Tensor, QuantizationParams]:
        """float -> integer grid. Returns (integer tensor, params used).

        Pass `params` to reuse a previously derived grid (e.g. to quantize
        several tensors identically); omit it to calibrate on `x` itself.
        """
        _validate_latent(x)
        if params is None:
            params = self.compute_params(x)

        q = torch.round(x / params.scale) + params.zero_point
        q = torch.clamp(q, params.q_min, params.q_max)
        return q.to(torch.int32), params

    def dequantize(self, q: torch.Tensor, params: QuantizationParams) -> torch.Tensor:
        """integer grid -> approximate float."""
        return (q.to(params.scale.dtype) - params.zero_point) * params.scale

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
    """
    unclamped = torch.round(x / params.scale) + params.zero_point
    below = (unclamped < params.q_min).sum().item()
    above = (unclamped > params.q_max).sum().item()
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
