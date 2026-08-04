"""Reusable dataset/frame validation checks.

Shared by FrameDataset (which runs these on every loaded sample) and by
tests, so the same rules are exercised whether a check happens for real
during data loading or is triggered deliberately with a synthetic bad
tensor in a unit test.
"""

from __future__ import annotations

import torch

_PIXEL_RANGE_TOLERANCE = 1e-5


class DatasetValidationError(Exception):
    """Raised for any dataset/manifest/frame problem, with a specific,
    actionable message - never a bare assertion or a raw AttributeError
    from downstream code touching a None/malformed value.
    """


def validate_frame_tensor(
    tensor: torch.Tensor,
    *,
    source: str = "<tensor>",
    expected_channels: int = 3,
    expected_height: int | None = None,
    expected_width: int | None = None,
) -> None:
    """Check a loaded frame tensor's dtype, shape, and pixel range.

    Expects a [C, H, W] float32 tensor already scaled to [0, 1] (what
    FrameDataset produces before any transform is applied). Raises
    DatasetValidationError on the first problem found.
    """
    if tensor.dtype != torch.float32:
        raise DatasetValidationError(f"{source}: expected dtype float32, got {tensor.dtype}")
    if tensor.ndim != 3:
        raise DatasetValidationError(
            f"{source}: expected a 3D [C, H, W] tensor, got shape {tuple(tensor.shape)}"
        )

    channels, height, width = tensor.shape
    if channels != expected_channels:
        raise DatasetValidationError(
            f"{source}: expected {expected_channels} channels, got {channels}"
        )
    if expected_height is not None and height != expected_height:
        raise DatasetValidationError(
            f"{source}: expected height {expected_height}, got {height}"
        )
    if expected_width is not None and width != expected_width:
        raise DatasetValidationError(
            f"{source}: expected width {expected_width}, got {width}"
        )

    min_value = tensor.min().item()
    max_value = tensor.max().item()
    if min_value < -_PIXEL_RANGE_TOLERANCE or max_value > 1 + _PIXEL_RANGE_TOLERANCE:
        raise DatasetValidationError(
            f"{source}: pixel values out of expected [0, 1] range "
            f"(min={min_value:.4f}, max={max_value:.4f})"
        )
