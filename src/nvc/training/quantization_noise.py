"""Differentiable uniform quantization-noise relaxation (Milestone 8A).

WHAT THIS IS FOR
-----------------
Milestone 7 measured that NVC's 8-bit and 6-bit operating points are almost
indistinguishable (+0.07 dB PSNR for 35% more bits) while quality drops off
faster going into 4-bit - the learned representation is not especially
robust to coarse quantization. The real quantizer (`round(x / scale)`, see
`nvc.compression.quantization`) has zero gradient almost everywhere, so the
network never gets a training signal that says "this direction of the
latent is about to be destroyed by quantization." Standard fix, and the one
this module implements: during training, replace the real quantization step
with a differentiable stand-in, uniform additive noise matching the real
quantizer's step size, so backprop can push the encoder toward regions of
latent space that survive it.

THE MATHEMATICS
----------------
For a quantization step (`scale`) delta, the real quantizer's rounding error
`round(x / delta) * delta - x` is, for a reasonably smooth input
distribution relative to the grid, well approximated by
Uniform(-delta/2, delta/2) noise (this is the standard high-resolution
quantization-noise approximation, and the basis for every uniform-noise QAT
relaxation used in the learned-compression literature - see e.g. Balle et
al. 2017/2018). So training time only adds:

    z_tilde = z + noise,     noise ~ Uniform(-scale/2, scale/2)
    x_hat   = decoder(z_tilde)
    loss    = MSE(x_hat, x)

`scale` is exactly the `QuantizationParams.scale` a real `UniformQuantizer`
would use at the bit depth being trained for (see "WHERE SCALE COMES FROM"
below) - not an arbitrary noise magnitude picked independently of the actual
quantization grid.

This is purely a *training-time surrogate*. Inference always goes through
the real `UniformQuantizer.quantize`/`dequantize` (or the full `.nvc`
codec path); nothing here is invoked at inference, and this module has no
learnable parameters of its own - `scale` is a frozen constant for the
whole training run.

WHERE SCALE COMES FROM
------------------------
`QuantizationNoise` never estimates its own scale (e.g. from the current
minibatch) and never touches DAVIS. It is built once, before training
starts, from an existing calibration artifact produced by the project's own
`scripts/calibrate_quantizer.py` against Vimeo TRAIN frames only - the same
mechanism Milestone 7 already uses to freeze quantization parameters for
the codec. `from_calibration()` below just loads that JSON file's
`QuantizationParams` and wraps it. This was chosen over a dynamic
per-batch estimate because a per-batch scale would (a) keep moving as the
encoder's output distribution shifts during training, undermining the very
robustness objective this milestone is testing, and (b) make the run
non-reproducible in the ordinary sense (the noise distribution would depend
on batch order/composition, not just the seed). A single frozen artifact
is stable, reproducible, and auditable: exactly which calibration file
produced the noise scale for a given checkpoint can be recorded in that
checkpoint's training history.
"""

from __future__ import annotations

from pathlib import Path

import torch

from nvc.compression.calibration import load_calibration
from nvc.compression.quantization import QuantizationParams


class QuantizationNoise:
    """Frozen, differentiable Uniform(-scale/2, scale/2) latent perturbation.

    Deliberately NOT an `nn.Module`: it has no learnable parameters and must
    never appear in a model's `state_dict()` (a QAT checkpoint must remain
    loadable by the ordinary `load_model_from_checkpoint` inference path,
    which never re-supplies this object - see `BaselineAutoencoder`). `scale`
    is moved to the input tensor's device on every call instead of relying on
    `nn.Module.to()`, since it is not a registered buffer.
    """

    def __init__(self, scale: torch.Tensor, *, bits: int, mode: str) -> None:
        if not torch.is_floating_point(scale):
            raise TypeError(f"scale must be a floating-point tensor, got dtype {scale.dtype}")
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("scale must be finite and strictly positive everywhere")
        if scale.dim() != 4 or scale.shape[0] != 1 or scale.shape[2:] != (1, 1):
            raise ValueError(
                f"scale must broadcast against a [B, C, H, W] latent, i.e. have "
                f"shape [1, C, 1, 1] (or [1, 1, 1, 1]), got {tuple(scale.shape)}"
            )
        if not 1 <= bits <= 16:
            raise ValueError(f"bits must be in [1, 16], got {bits}")

        # Frozen: detached and cloned so no autograd history or aliasing
        # into a caller's tensor can sneak in, and never updated afterward.
        self.scale = scale.detach().clone()
        self.bits = bits
        self.mode = mode

    @classmethod
    def from_calibration(
        cls, path: str | Path, *, bits: int | None = None, mode: str | None = None
    ) -> "QuantizationNoise":
        """Build from a calibration file written by scripts/calibrate_quantizer.py.

        `bits`/`mode` default to the calibration file's own recorded values;
        pass them explicitly only to assert the file matches what you intend
        to train with (mismatches raise rather than silently using the wrong
        step size, or silently training with global noise while the codec
        elsewhere uses per-channel quantization).
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
                f"mode={mode!r} was requested. The training noise must match the "
                "quantization mode actually used by the codec - generate a "
                "calibration in that mode, or omit `mode` to use the file's own."
            )
        if document["calibration_metadata"].get("calibration_split") != "train":
            raise ValueError(
                f"Calibration {path} was not computed from the 'train' split "
                f"(calibration_split={document['calibration_metadata'].get('calibration_split')!r}). "
                "Quantization-noise scale for training must come from train-split "
                "calibration only."
            )
        return cls(params.scale, bits=params.bits, mode=params.mode)

    def apply(self, z: torch.Tensor) -> torch.Tensor:
        """z_tilde = z + Uniform(-scale/2, scale/2), broadcast per the scale's shape.

        Uses the ambient PyTorch RNG (`torch.rand_like`), so reproducibility
        follows the same seeding rules as every other stochastic op in the
        training loop (see `nvc.utils.seed.seed_everything`) - no private
        generator is kept, deliberately, so `seed_everything` alone is
        sufficient to reproduce a run.
        """
        if z.dim() != 4:
            raise ValueError(f"Expected a 4D [B, C, H, W] latent tensor, got shape {tuple(z.shape)}")
        scale = self.scale.to(device=z.device, dtype=z.dtype)
        noise = (torch.rand_like(z) - 0.5) * scale
        return z + noise

    def to_dict(self) -> dict:
        """Summary for training-history logging - never used to rebuild state."""
        return {"bits": self.bits, "mode": self.mode}
