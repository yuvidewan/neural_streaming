"""MS-SSIM, the perceptual quality metric used alongside PSNR.

PSNR (see basic_metrics.py) is a pure pixel-error measure and is a poor
predictor of how a human perceives quality - it systematically favors blur
over the kinds of error viewers actually notice. Every serious codec
comparison in this field reports a structural/perceptual metric alongside
it, which is why Milestone 7 adds MS-SSIM before making any claim about
this codec relative to H.264/H.265.

The underlying implementation is `pytorch-msssim` rather than a
hand-rolled one: MS-SSIM has enough fiddly detail (Gaussian window sizing,
the five-scale weight vector from Wang et al. 2003, per-scale downsampling)
that a from-scratch version would be a subtle-bug farm with no upside. This
module is a thin, validating wrapper over it.

This module does not touch, wrap, or alter MSE/PSNR - those keep their
existing behavior exactly.
"""

from __future__ import annotations

import torch

# pytorch-msssim downsamples 4 times internally, so each spatial dimension
# must exceed 2**4 * (win_size - 1) = 160 for the default 11-pixel window.
MIN_SPATIAL_SIZE = 161

_PIXEL_RANGE_TOLERANCE = 1e-4


class MetricInputError(ValueError):
    """Raised when a metric is given tensors it cannot meaningfully score.

    A distinct, typed error so a caller can tell "you handed me the wrong
    kind of tensor" apart from a genuine numerical problem - and so the
    underlying library's bare AssertionError never leaks out.
    """


def msssim(
    prediction: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0
) -> torch.Tensor:
    """Multi-Scale Structural Similarity, averaged over the batch.

    Expects two matching [B, C, H, W] float tensors in [0, 1] - the same
    convention FrameDataset and the codec use throughout this project.
    Returns a scalar tensor in roughly [0, 1], where 1.0 means identical.

    Raises MetricInputError (never a bare assertion) for wrong rank,
    mismatched shapes, non-float dtype, out-of-range values, or spatial
    dimensions too small for MS-SSIM's five scales.
    """
    _validate_pair(prediction, target, data_range=data_range)

    # Imported lazily so that merely importing nvc.evaluation does not hard
    # depend on pytorch-msssim being installed - only calling msssim() does.
    try:
        from pytorch_msssim import ms_ssim as _ms_ssim
    except ImportError as exc:  # pragma: no cover - depends on install state
        raise MetricInputError(
            "MS-SSIM requires the 'pytorch-msssim' package, which is not installed. "
            "Install it with: pip install pytorch-msssim  (it is listed in "
            "requirements.txt)."
        ) from exc

    return _ms_ssim(prediction, target, data_range=data_range, size_average=True)


def _validate_pair(
    prediction: torch.Tensor, target: torch.Tensor, *, data_range: float
) -> None:
    for name, tensor in (("prediction", prediction), ("target", target)):
        if not isinstance(tensor, torch.Tensor):
            raise MetricInputError(
                f"{name} must be a torch.Tensor, got {type(tensor).__name__}"
            )
        if tensor.dim() != 4:
            raise MetricInputError(
                f"{name} must be a 4D [B, C, H, W] tensor, got shape {tuple(tensor.shape)}"
            )
        if not torch.is_floating_point(tensor):
            raise MetricInputError(
                f"{name} must be a floating-point tensor, got dtype {tensor.dtype}"
            )
        if not torch.isfinite(tensor).all():
            raise MetricInputError(f"{name} contains NaN or inf values")

    if prediction.shape != target.shape:
        raise MetricInputError(
            f"prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )

    height, width = prediction.shape[-2:]
    if min(height, width) < MIN_SPATIAL_SIZE:
        raise MetricInputError(
            f"MS-SSIM needs both spatial dimensions >= {MIN_SPATIAL_SIZE} (it downsamples "
            f"4 times), got {height}x{width}. Use PSNR/MSE for smaller images."
        )

    low, high = -_PIXEL_RANGE_TOLERANCE, data_range + _PIXEL_RANGE_TOLERANCE
    for name, tensor in (("prediction", prediction), ("target", target)):
        minimum, maximum = tensor.min().item(), tensor.max().item()
        if minimum < low or maximum > high:
            raise MetricInputError(
                f"{name} values must lie in [0, {data_range}] "
                f"(got [{minimum:.4f}, {maximum:.4f}]). Clamp or rescale before scoring."
            )
