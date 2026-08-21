"""Tests for the MS-SSIM metric added in Milestone 7.

Uses small synthetic tensors only - no dataset, no checkpoint, no CUDA.
Every tensor is at least MIN_SPATIAL_SIZE on both axes because MS-SSIM
downsamples four times; the too-small case is tested explicitly.
"""

from __future__ import annotations

import pytest
import torch

from nvc.evaluation.basic_metrics import mse, psnr
from nvc.evaluation.perceptual_metrics import MIN_SPATIAL_SIZE, MetricInputError, msssim

# Smallest size MS-SSIM accepts; keeps these tests fast.
SIZE = MIN_SPATIAL_SIZE


def _image(batch: int = 1, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 3, SIZE, SIZE, generator=generator)


# --- Core behavior ---


def test_identical_images_score_one():
    image = _image()

    assert msssim(image, image).item() == pytest.approx(1.0, abs=1e-5)


def test_identical_batch_scores_one():
    images = _image(batch=4)

    assert msssim(images, images).item() == pytest.approx(1.0, abs=1e-5)


def test_output_is_a_scalar_tensor():
    image = _image()

    score = msssim(image, image)

    assert isinstance(score, torch.Tensor)
    assert score.dim() == 0


def test_score_lies_within_the_expected_range():
    reference = _image(seed=1)
    noisy = (reference + 0.2 * torch.randn_like(reference)).clamp(0, 1)

    score = msssim(noisy, reference).item()

    assert -1.0 <= score <= 1.0


def test_degraded_image_scores_below_identical():
    reference = _image(seed=2)
    noisy = (reference + 0.1 * torch.randn_like(reference)).clamp(0, 1)

    assert msssim(noisy, reference).item() < msssim(reference, reference).item()


def test_more_degradation_scores_lower():
    reference = _image(seed=3)
    torch.manual_seed(0)
    slight = (reference + 0.02 * torch.randn_like(reference)).clamp(0, 1)
    severe = (reference + 0.30 * torch.randn_like(reference)).clamp(0, 1)

    assert msssim(severe, reference).item() < msssim(slight, reference).item()


def test_is_deterministic():
    reference = _image(seed=4)
    noisy = (reference + 0.05 * torch.randn_like(reference)).clamp(0, 1)

    assert msssim(noisy, reference).item() == msssim(noisy, reference).item()


def test_batch_score_is_the_mean_of_individual_scores():
    reference = _image(batch=3, seed=5)
    torch.manual_seed(1)
    noisy = (reference + 0.08 * torch.randn_like(reference)).clamp(0, 1)

    batched = msssim(noisy, reference).item()
    individual = [
        msssim(noisy[i : i + 1], reference[i : i + 1]).item() for i in range(3)
    ]

    assert batched == pytest.approx(sum(individual) / len(individual), abs=1e-5)


# --- Input validation ---


def test_rejects_non_tensor_input():
    with pytest.raises(MetricInputError):
        msssim([[0.0]], _image())


@pytest.mark.parametrize("shape", [(3, SIZE, SIZE), (SIZE, SIZE), (1, 1, 3, SIZE, SIZE)])
def test_rejects_wrong_rank(shape):
    tensor = torch.rand(*shape)
    with pytest.raises(MetricInputError, match="4D"):
        msssim(tensor, tensor)


def test_rejects_mismatched_shapes():
    with pytest.raises(MetricInputError, match="same shape"):
        msssim(_image(batch=1), _image(batch=2))


def test_rejects_integer_dtype():
    tensor = torch.zeros(1, 3, SIZE, SIZE, dtype=torch.uint8)
    with pytest.raises(MetricInputError, match="floating-point"):
        msssim(tensor, tensor)


def test_rejects_values_above_the_data_range():
    tensor = torch.full((1, 3, SIZE, SIZE), 1.5)
    with pytest.raises(MetricInputError, match=r"\[0, 1"):
        msssim(tensor, tensor)


def test_rejects_negative_values():
    tensor = torch.full((1, 3, SIZE, SIZE), -0.5)
    with pytest.raises(MetricInputError, match=r"\[0, 1"):
        msssim(tensor, tensor)


def test_rejects_non_finite_values():
    tensor = _image()
    tensor[0, 0, 0, 0] = float("nan")
    with pytest.raises(MetricInputError, match="NaN"):
        msssim(tensor, tensor)


def test_rejects_images_too_small_for_five_scales():
    # MS-SSIM downsamples 4 times, so it genuinely cannot score tiny images;
    # this must be a clear typed error, not the library's bare AssertionError.
    small = torch.rand(1, 3, MIN_SPATIAL_SIZE - 1, MIN_SPATIAL_SIZE - 1)

    with pytest.raises(MetricInputError, match="MS-SSIM needs"):
        msssim(small, small)


def test_accepts_exactly_the_minimum_size():
    image = torch.rand(1, 3, MIN_SPATIAL_SIZE, MIN_SPATIAL_SIZE)

    assert msssim(image, image).item() == pytest.approx(1.0, abs=1e-5)


# --- Existing metrics must be untouched by this addition ---


def test_mse_and_psnr_behavior_is_unchanged():
    image = _image(seed=6)

    assert mse(image, image).item() == 0.0
    assert psnr(image, image).item() == float("inf")
