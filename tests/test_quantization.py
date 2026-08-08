"""Tests for Milestone 5: uniform scalar quantization, latent analysis,
raw storage arithmetic, and baseline-model integrity.

All tests use tiny synthetic tensors and stay fast. The integrity tests
specifically guard the property that quantization is a pure, read-only
operation on the trained model - Milestone 4's checkpoint and weights must
survive this milestone untouched.
"""

from __future__ import annotations

import math

import pytest
import torch

from nvc.compression import (
    QUANTIZATION_MODES,
    UniformQuantizer,
    latent_storage_analysis,
    quantization_error,
    tensor_bit_cost,
)
from nvc.evaluation.basic_metrics import psnr_from_mse
from nvc.evaluation.latent_analysis import extract_latents, latent_statistics
from nvc.models import BaselineAutoencoder

BIT_WIDTHS = (8, 6, 4)


# --- Construction / validation ---


@pytest.mark.parametrize("bits", BIT_WIDTHS)
@pytest.mark.parametrize("mode", QUANTIZATION_MODES)
def test_quantizer_constructs_for_supported_configurations(bits, mode):
    quantizer = UniformQuantizer(bits, mode)
    assert quantizer.bits == bits
    assert quantizer.mode == mode


def test_quantizer_rejects_unknown_mode():
    with pytest.raises(ValueError):
        UniformQuantizer(8, "not_a_mode")


@pytest.mark.parametrize("bits", [0, -1, 17])
def test_quantizer_rejects_out_of_range_bits(bits):
    with pytest.raises(ValueError):
        UniformQuantizer(bits)


def test_quantizer_rejects_non_integer_bits():
    with pytest.raises(TypeError):
        UniformQuantizer(8.0)


def test_quantizer_rejects_non_4d_tensor():
    quantizer = UniformQuantizer(8)
    with pytest.raises(ValueError):
        quantizer.quantize(torch.rand(3, 8, 8))


def test_quantizer_rejects_integer_input():
    quantizer = UniformQuantizer(8)
    with pytest.raises(TypeError):
        quantizer.quantize(torch.zeros(1, 2, 4, 4, dtype=torch.int32))


def test_quantizer_rejects_non_finite_input():
    quantizer = UniformQuantizer(8)
    x = torch.zeros(1, 1, 2, 2)
    x[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        quantizer.quantize(x)


# --- Known-value quantization / dequantization ---


def test_known_value_quantization_unit_range_8bit():
    # x spans exactly [0, 1] -> scale = 1/255, zero_point = 0, so the
    # endpoints must land on the extreme integer levels.
    quantizer = UniformQuantizer(8, "global")
    x = torch.tensor([[[[0.0, 0.5, 1.0]]]])

    q, params = quantizer.quantize(x)

    assert params.scale.item() == pytest.approx(1.0 / 255.0)
    assert params.zero_point.item() == 0.0
    assert q.flatten().tolist() == [0, 127, 255]


def test_known_value_dequantization_inverts_quantization():
    quantizer = UniformQuantizer(8, "global")
    x = torch.tensor([[[[0.0, 0.5, 1.0]]]])

    q, params = quantizer.quantize(x)
    x_hat = quantizer.dequantize(q, params)

    assert x_hat.flatten().tolist() == pytest.approx([0.0, 127 / 255, 1.0], abs=1e-6)


@pytest.mark.parametrize("bits", BIT_WIDTHS)
def test_grid_endpoints_are_hit_exactly_for_a_cleanly_divided_range(bits):
    # [0, 1] divides evenly onto the integer grid, so the endpoints must
    # land exactly on q_min and q_max.
    quantizer = UniformQuantizer(bits, "global")
    x = torch.linspace(0.0, 1.0, 64).reshape(1, 1, 8, 8)

    q, params = quantizer.quantize(x)

    assert params.num_levels == 2 ** bits
    assert q.min().item() == params.q_min
    assert q.max().item() == params.q_max


@pytest.mark.parametrize("bits", BIT_WIDTHS)
def test_grid_is_near_fully_utilized_for_an_arbitrary_range(bits):
    # For an arbitrary range an endpoint can sit exactly halfway between two
    # levels (here x_min/scale == -76.5), and round-half-to-even resolves the
    # tie one level inward. That costs at most one level at each end - the
    # accuracy guarantee that actually matters is the half-step error bound
    # asserted in test_round_trip_error_bounded_by_half_a_quantization_step.
    quantizer = UniformQuantizer(bits, "global")
    x = torch.linspace(-3.0, 7.0, 64).reshape(1, 1, 8, 8)

    q, params = quantizer.quantize(x)

    assert q.min().item() <= params.q_min + 1
    assert q.max().item() >= params.q_max - 1


@pytest.mark.parametrize("bits", BIT_WIDTHS)
def test_quantized_output_never_leaves_valid_range(bits):
    quantizer = UniformQuantizer(bits, "global")
    x = torch.randn(4, 3, 8, 8) * 50.0

    q, params = quantizer.quantize(x)

    assert q.min().item() >= params.q_min
    assert q.max().item() <= params.q_max


def test_quantize_clips_values_outside_a_reused_grid():
    # Calibrate on a narrow tensor, then quantize a much wider one with the
    # same params: everything outside the calibrated range must clip to the
    # grid endpoints rather than overflow.
    quantizer = UniformQuantizer(8, "global")
    calibration = torch.zeros(1, 1, 2, 2)
    calibration[0, 0, 0, 0] = -1.0
    calibration[0, 0, 1, 1] = 1.0
    params = quantizer.compute_params(calibration)

    wide = torch.tensor([[[[-100.0, 0.0, 100.0]]]])
    q, _ = quantizer.quantize(wide, params)

    assert q.min().item() == params.q_min
    assert q.max().item() == params.q_max


def test_quantized_tensor_has_integer_dtype():
    quantizer = UniformQuantizer(8)
    q, _ = quantizer.quantize(torch.rand(1, 2, 4, 4))

    assert q.dtype == torch.int32
    assert torch.equal(q, q.to(torch.float32).round().to(torch.int32))


def test_dequantized_tensor_is_float_with_original_shape():
    quantizer = UniformQuantizer(8)
    x = torch.rand(2, 3, 4, 4)

    x_hat, _ = quantizer.quantize_dequantize(x)

    assert x_hat.dtype == x.dtype
    assert x_hat.shape == x.shape


# --- Edge cases: constants, zero, negatives ---


@pytest.mark.parametrize("constant", [0.0, 0.5, 5.0, -3.0])
@pytest.mark.parametrize("mode", QUANTIZATION_MODES)
def test_constant_tensor_round_trips_without_nan(constant, mode):
    # A constant tensor has a zero-width range, which would divide by zero
    # if the degenerate case were not widened.
    quantizer = UniformQuantizer(8, mode)
    x = torch.full((2, 3, 4, 4), constant)

    x_hat, params = quantizer.quantize_dequantize(x)

    assert torch.isfinite(x_hat).all()
    assert torch.isfinite(params.scale).all()
    assert (params.scale > 0).all()
    assert x_hat.flatten()[0].item() == pytest.approx(constant, abs=0.01)


def test_all_zero_tensor_reconstructs_exactly():
    quantizer = UniformQuantizer(8)
    x = torch.zeros(1, 2, 4, 4)

    x_hat, _ = quantizer.quantize_dequantize(x)

    assert torch.equal(x_hat, x)


def test_exact_zero_is_representable_within_a_signed_range():
    # Affine quantization places an integer level exactly on 0.0 whenever 0
    # lies inside the observed range - a property worth locking down, since
    # latents are roughly zero-centered.
    quantizer = UniformQuantizer(8, "global")
    x = torch.tensor([[[[-1.0, 0.0, 1.0]]]])

    x_hat, _ = quantizer.quantize_dequantize(x)

    assert x_hat.flatten()[1].item() == 0.0


def test_negative_values_survive_round_trip():
    quantizer = UniformQuantizer(8, "global")
    x = torch.linspace(-5.0, -1.0, 16).reshape(1, 1, 4, 4)

    x_hat, _ = quantizer.quantize_dequantize(x)

    assert (x_hat < 0).all()
    assert torch.allclose(x_hat, x, atol=0.02)


def test_far_from_zero_range_reconstructs_accurately():
    # Guards the unclamped zero_point: a range nowhere near 0 (all ~1000)
    # would collapse if zero_point were clamped into [q_min, q_max].
    quantizer = UniformQuantizer(8, "global")
    x = torch.linspace(1000.0, 1001.0, 16).reshape(1, 1, 4, 4)

    x_hat, _ = quantizer.quantize_dequantize(x)

    assert torch.allclose(x_hat, x, atol=0.01)


# --- Round-trip accuracy and bit-width ordering ---


@pytest.mark.parametrize("mode", QUANTIZATION_MODES)
def test_round_trip_error_decreases_with_more_bits(mode):
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, 8)

    errors = []
    for bits in (4, 6, 8):
        x_hat, _ = UniformQuantizer(bits, mode).quantize_dequantize(x)
        errors.append(quantization_error(x, x_hat)["latent_mse"])

    assert errors[0] > errors[1] > errors[2]


@pytest.mark.parametrize("bits", BIT_WIDTHS)
def test_round_trip_error_bounded_by_half_a_quantization_step(bits):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8)
    quantizer = UniformQuantizer(bits, "global")

    x_hat, params = quantizer.quantize_dequantize(x)

    # Rounding to the nearest grid point can never err by more than half a
    # step (plus a small tolerance for float arithmetic).
    max_error = (x_hat - x).abs().max().item()
    assert max_error <= params.scale.item() / 2 + 1e-5


def test_per_channel_beats_global_when_channel_ranges_differ():
    # Channel 0 spans [-100, 100] and channel 1 spans [-0.01, 0.01]. One
    # global scale must cover both, destroying channel 1; per-channel scales
    # adapt. This is the case per-channel mode exists for.
    x = torch.zeros(1, 2, 4, 4)
    x[:, 0] = torch.linspace(-100, 100, 16).reshape(4, 4)
    x[:, 1] = torch.linspace(-0.01, 0.01, 16).reshape(4, 4)

    global_hat, _ = UniformQuantizer(8, "global").quantize_dequantize(x)
    per_channel_hat, _ = UniformQuantizer(8, "per_channel").quantize_dequantize(x)

    global_error = quantization_error(x, global_hat)["latent_mse"]
    per_channel_error = quantization_error(x, per_channel_hat)["latent_mse"]
    assert per_channel_error < global_error


def test_global_and_per_channel_agree_when_all_channels_share_a_range():
    # With identical per-channel ranges there is nothing for per-channel
    # scaling to exploit, so both modes should coincide.
    x = torch.linspace(-1.0, 1.0, 32).reshape(1, 2, 4, 4)
    x[:, 1] = x[:, 0]

    global_hat, _ = UniformQuantizer(8, "global").quantize_dequantize(x)
    per_channel_hat, _ = UniformQuantizer(8, "per_channel").quantize_dequantize(x)

    assert torch.allclose(global_hat, per_channel_hat, atol=1e-6)


# --- Parameter shapes ---


def test_global_mode_produces_scalar_parameters():
    params = UniformQuantizer(8, "global").compute_params(torch.rand(2, 5, 4, 4))

    assert params.scale.shape == (1, 1, 1, 1)
    assert params.zero_point.shape == (1, 1, 1, 1)


def test_per_channel_mode_produces_one_parameter_per_channel():
    params = UniformQuantizer(8, "per_channel").compute_params(torch.rand(2, 5, 4, 4))

    assert params.scale.shape == (1, 5, 1, 1)
    assert params.zero_point.shape == (1, 5, 1, 1)


def test_zero_point_is_integer_valued():
    params = UniformQuantizer(8, "per_channel").compute_params(torch.randn(2, 3, 4, 4))

    assert torch.equal(params.zero_point, torch.round(params.zero_point))


# --- Error metrics ---


def test_quantization_error_is_zero_for_identical_tensors():
    x = torch.rand(1, 2, 4, 4)
    errors = quantization_error(x, x)

    assert errors["latent_mse"] == 0.0
    assert errors["latent_mae"] == 0.0
    assert errors["latent_max_abs_error"] == 0.0


def test_quantization_error_reports_known_difference():
    original = torch.zeros(1, 1, 2, 2)
    dequantized = torch.full((1, 1, 2, 2), 0.5)

    errors = quantization_error(original, dequantized)

    assert errors["latent_mse"] == pytest.approx(0.25)
    assert errors["latent_mae"] == pytest.approx(0.5)
    assert errors["latent_max_abs_error"] == pytest.approx(0.5)


def test_psnr_from_mse_matches_direct_formula():
    assert psnr_from_mse(0.25).item() == pytest.approx(10.0 * math.log10(1 / 0.25))
    assert psnr_from_mse(0.0).item() == float("inf")


# --- Raw storage arithmetic ---


def test_tensor_bit_cost_counts_values_and_bits():
    cost = tensor_bit_cost((64, 16, 16), 8)

    assert cost["num_values"] == 16384
    assert cost["total_bits"] == 16384 * 8
    assert cost["total_bytes"] == 16384


def test_latent_storage_analysis_matches_hand_calculation():
    analysis = latent_storage_analysis((3, 256, 256), (64, 16, 16))

    assert analysis["image_raw_uint8"]["total_bits"] == 3 * 256 * 256 * 8
    assert analysis["latent"]["float32"]["total_bits"] == 16384 * 32
    assert analysis["latent"]["8_bit"]["total_bits"] == 16384 * 8
    assert analysis["latent"]["4_bit"]["total_bits"] == 16384 * 4
    # 1,572,864 raw image bits / 131,072 8-bit latent bits = 12x
    assert analysis["latent"]["8_bit"]["raw_size_ratio_vs_uint8_image"] == pytest.approx(12.0)


def test_storage_analysis_labels_itself_as_raw_not_compression():
    analysis = latent_storage_analysis((3, 256, 256), (64, 16, 16))

    # The caveat must travel with the numbers, so nothing downstream can
    # quote them as a compression ratio by accident.
    assert "not a compression ratio" in analysis["note"].lower()
    assert "entropy coding" in analysis["note"].lower()
    for entry in analysis["latent"].values():
        assert "raw_size_ratio_vs_uint8_image" in entry
        assert "compression_ratio" not in entry


# --- Latent analysis ---


def _tiny_model_and_loader():
    model = BaselineAutoencoder(in_channels=3, latent_channels=4, base_channels=8)
    frames = [torch.rand(3, 32, 32) for _ in range(6)]
    loader = torch.utils.data.DataLoader(frames, batch_size=2)
    return model, loader


def test_extract_latents_returns_cpu_tensor_of_expected_shape():
    model, loader = _tiny_model_and_loader()

    latents = extract_latents(model, loader, torch.device("cpu"))

    assert latents.shape == (6, 4, 2, 2)
    assert latents.device.type == "cpu"


def test_extract_latents_respects_max_batches():
    model, loader = _tiny_model_and_loader()

    latents = extract_latents(model, loader, torch.device("cpu"), max_batches=2)

    assert latents.shape[0] == 4


def test_extract_latents_does_not_run_the_decoder():
    # Latent spatial size must be the encoder's output (32/16 = 2), not the
    # full 32x32 input - proof the decoder was never involved.
    model, loader = _tiny_model_and_loader()

    latents = extract_latents(model, loader, torch.device("cpu"))

    assert latents.shape[2:] == (2, 2)


def test_latent_statistics_reports_expected_fields_and_values():
    latents = torch.zeros(4, 3, 2, 2)
    latents[:, 0] = 1.0
    latents[:, 1] = -1.0

    stats = latent_statistics(latents)

    assert stats["num_samples"] == 4
    assert stats["latent_shape_per_sample"] == [3, 2, 2]
    assert stats["elements_per_sample"] == 12
    assert stats["global"]["min"] == pytest.approx(-1.0)
    assert stats["global"]["max"] == pytest.approx(1.0)
    assert stats["global"]["mean"] == pytest.approx(0.0)
    # Channel 2 is all zeros: one third of every sample.
    assert stats["global"]["percent_exactly_zero"] == pytest.approx(100 / 3)
    assert len(stats["per_channel"]["mean"]) == 3


def test_latent_statistics_rejects_wrong_dimensionality():
    with pytest.raises(ValueError):
        latent_statistics(torch.rand(3, 8, 8))


# --- Model integrity: quantization must not disturb the baseline model ---


def test_quantization_does_not_modify_model_weights():
    model = BaselineAutoencoder(in_channels=3, latent_channels=4, base_channels=8)
    before = [p.clone() for p in model.parameters()]
    x = torch.rand(2, 3, 32, 32)

    with torch.no_grad():
        latent = model.encode(x)
        dequantized, _ = UniformQuantizer(4, "per_channel").quantize_dequantize(latent)
        model.decode(dequantized)

    for original, current in zip(before, model.parameters()):
        assert torch.equal(original, current)


def test_quantization_does_not_mutate_the_input_latent():
    model = BaselineAutoencoder(in_channels=3, latent_channels=4, base_channels=8)
    with torch.no_grad():
        latent = model.encode(torch.rand(2, 3, 32, 32))
    latent_copy = latent.clone()

    UniformQuantizer(4, "global").quantize_dequantize(latent)

    assert torch.equal(latent, latent_copy)


def test_unquantized_path_is_unchanged_by_the_quantization_import():
    # The float32 baseline path must still be bit-identical to a plain
    # forward pass now that quantization exists alongside it.
    torch.manual_seed(0)
    model = BaselineAutoencoder(in_channels=3, latent_channels=4, base_channels=8)
    model.eval()
    x = torch.rand(2, 3, 32, 32)

    with torch.no_grad():
        direct = model(x)
        staged = model.decode(model.encode(x))

    assert torch.equal(direct, staged)
