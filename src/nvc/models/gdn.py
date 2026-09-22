"""Generalized Divisive Normalization (Balle, Laparra and Simoncelli 2016),
and its approximate inverse IGDN.

WHY THIS EXISTS
---------------
The Stage 0 scoreboard put the deployed codec at +450.7% BD-rate against
libx264, and located the gap in the transform rather than in the entropy
coder: the 593k-parameter Conv/ReLU autoencoder sits near its own
reconstruction ceiling (4-bit to 5-bit latents buy +0.36 dB for 43% more
bits). GDN is the first of the two things the compression literature
changed about that transform, and it is the cheaper one.

WHAT IT DOES
------------
For an activation x with C channels, GDN divides each channel by a norm
pooled across ALL channels at the same spatial position:

    y_i = x_i / sqrt(beta_i + sum_j gamma_ij * x_j^2)

IGDN multiplies by the same square root instead of dividing. beta is a
per-channel floor (C values) and gamma is a full channel-mixing matrix
(C x C values), both learned.

Unlike ReLU it is smooth, invertible in the region it is used, and - the
part that matters for compression - it is a *joint* nonlinearity: it lets
one channel suppress another, which is how it gaussianizes the latent.
A ReLU network has to spend capacity representing that statistical
dependence; GDN removes it from the signal, so the entropy coder sees a
distribution closer to what its per-channel factorized model assumes.
That is why every learned image codec since 2016 uses it and why it is
the first lever in Stage 1 of PARITY_ROADMAP.md.

THE REPARAMETERIZATION, AND WHY IT IS NOT OPTIONAL
--------------------------------------------------
beta and gamma must stay non-negative or the square root takes a negative
argument and training produces NaN. Clamping them directly kills the
gradient the moment a value touches the bound and the parameter can never
come back. The standard fix (used by the reference implementation and by
CompressAI) stores the SQUARE ROOTS of the parameters and applies a lower
bound whose backward pass is asymmetric:

  - forward:  max(x, bound), so the value is always legal;
  - backward: pass the gradient through when x is above the bound, OR when
    the gradient would push x back up. Only the one direction that would
    drive the parameter further below the bound is zeroed.

That is `LowerBound` below. A parameter pinned at the floor therefore
still has a route back, which a plain `torch.clamp` does not give it.

`_PEDESTAL` (2**-36) is added inside the square and subtracted after it,
which keeps the derivative of the square root finite when a parameter sits
exactly at zero - the reference implementation's trick, kept here because
the alternative is a silent NaN several hours into a training run.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# Reference-implementation constants. _REPARAM_OFFSET is the offset the
# parameters are stored with; the pedestal is its square.
_REPARAM_OFFSET = 2.0**-18
_PEDESTAL = _REPARAM_OFFSET**2
_BETA_MINIMUM = 1e-6


class LowerBound(torch.autograd.Function):
    """max(x, bound) with a gradient that survives the bound.

    The gradient is passed through where x >= bound (the bound is not
    active) or where the incoming gradient is negative (gradient descent
    would *increase* x, moving it back into the legal region). It is zeroed
    only where both x is below the bound and the update would push it
    further below - the one case where the clamp is genuinely blocking.
    """

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, bound: float) -> torch.Tensor:
        bound_tensor = torch.full_like(tensor, bound)
        ctx.save_for_backward(tensor, bound_tensor)
        return torch.max(tensor, bound_tensor)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        tensor, bound_tensor = ctx.saved_tensors
        pass_through = (tensor >= bound_tensor) | (grad_output < 0)
        return grad_output * pass_through.to(grad_output.dtype), None


class GDN(nn.Module):
    """Generalized Divisive Normalization over a [B, C, H, W] tensor.

    Args:
        channels: C. beta has this many entries, gamma has C x C.
        inverse: False for GDN (divide by the norm, used in the analysis
            transform), True for IGDN (multiply by it, used in synthesis).
        beta_minimum: floor on beta. The default keeps the argument of the
            square root strictly positive even where gamma has collapsed.
        gamma_initial: the diagonal of gamma at initialization. The
            off-diagonal starts at zero, so the layer begins life as a
            near-identity per-channel normalization and only learns
            cross-channel coupling if the data rewards it.
    """

    def __init__(
        self,
        channels: int,
        *,
        inverse: bool = False,
        beta_minimum: float = _BETA_MINIMUM,
        gamma_initial: float = 0.1,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"GDN needs at least one channel, got {channels}")
        if beta_minimum < 0:
            raise ValueError(f"beta_minimum must be non-negative, got {beta_minimum}")
        if gamma_initial < 0:
            raise ValueError(f"gamma_initial must be non-negative, got {gamma_initial}")

        self.channels = channels
        self.inverse = inverse
        self.gamma_initial = gamma_initial

        # The bounds live in the reparameterized (square-root) space, which
        # is where LowerBound is applied - hence the square roots here.
        self.beta_bound = float((beta_minimum + _PEDESTAL) ** 0.5)
        self.gamma_bound = float(_REPARAM_OFFSET)

        beta = torch.ones(channels)
        self.beta_reparam = nn.Parameter(torch.sqrt(beta + _PEDESTAL))

        gamma = gamma_initial * torch.eye(channels)
        self.gamma_reparam = nn.Parameter(torch.sqrt(gamma + _PEDESTAL))

    def beta(self) -> torch.Tensor:
        """The bounded, de-reparameterized beta actually used in forward()."""
        bounded = LowerBound.apply(self.beta_reparam, self.beta_bound)
        return bounded**2 - _PEDESTAL

    def gamma(self) -> torch.Tensor:
        """The bounded, de-reparameterized gamma actually used in forward()."""
        bounded = LowerBound.apply(self.gamma_reparam, self.gamma_bound)
        return bounded**2 - _PEDESTAL

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"GDN expects a [B, {self.channels}, H, W] tensor, got shape {tuple(x.shape)}"
            )

        # The sum over j of gamma_ij * x_j^2 is exactly a 1x1 convolution of
        # the squared activations with gamma as its kernel, with beta as the
        # bias - one cuDNN call instead of an explicit broadcast-and-sum.
        norm = F.conv2d(
            x * x,
            self.gamma().reshape(self.channels, self.channels, 1, 1),
            self.beta(),
        )
        norm = torch.sqrt(norm)

        return x * norm if self.inverse else x / norm

    def extra_repr(self) -> str:
        return f"channels={self.channels}, inverse={self.inverse}"


class IGDN(GDN):
    """GDN with inverse=True, for readability at the call site."""

    def __init__(self, channels: int, **kwargs: Any) -> None:
        kwargs.pop("inverse", None)
        super().__init__(channels, inverse=True, **kwargs)
