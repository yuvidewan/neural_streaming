"""Tests for the M10I conditional residual codec itself.

The claim under test is that the residual representation is CONDITIONED on the
warped reference. That is easy to assert in prose and easy to get wrong in code -
a module can take `z_ref` as an argument and quietly ignore it, and every
downstream number would still look plausible. So the tests here check the
conditioning is load-bearing: same residual + different reference must produce a
different coded tensor, and gradients must actually flow back through the
reference path.

They also pin the zero-init identity property, which is what makes the whole
M10H-vs-M10I comparison an ablation rather than a comparison of two unrelated
codecs.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _codec(seed: int = 0, latent_channels: int = 8, hidden_channels: int = 16):
    m10i = _load_script("m10i_conditional_residual")
    torch.manual_seed(seed)
    return m10i, m10i.build_codec({"latent_channels": latent_channels,
                                   "hidden_channels": hidden_channels})


def _randomise(codec, seed: int = 1):
    """Give the zero-initialised output layers real weights, so tests that are
    about conditioning are not trivially satisfied by the identity."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for branch in (codec.analysis, codec.synthesis):
            for layer in branch:
                if isinstance(layer, torch.nn.Conv2d):
                    layer.weight.normal_(0.0, 0.05, generator=generator)
                    layer.bias.normal_(0.0, 0.01, generator=generator)
    return codec


# --- shape and identity ------------------------------------------------------


def test_analysis_and_synthesis_preserve_the_latent_shape():
    _, codec = _codec()
    residual = torch.randn(2, 8, 16, 16)
    reference = torch.randn(2, 8, 16, 16)

    coded = codec.encode_residual(residual, reference)
    restored = codec.decode_residual(coded, reference)

    assert coded.shape == residual.shape
    assert restored.shape == residual.shape


def test_the_codec_is_exactly_the_identity_at_initialisation():
    """Zero-init on the last conv of each branch means M10I starts as M10H, bit
    for bit. Without this the comparison is not an ablation."""
    _, codec = _codec()
    residual = torch.randn(3, 8, 16, 16)
    reference = torch.randn(3, 8, 16, 16)

    coded = codec.encode_residual(residual, reference)

    assert torch.equal(coded, residual)
    assert torch.equal(codec.decode_residual(coded, reference), residual)


def test_a_trained_codec_is_no_longer_the_identity():
    _, codec = _codec()
    _randomise(codec)
    residual = torch.randn(2, 8, 16, 16)
    reference = torch.randn(2, 8, 16, 16)

    assert not torch.equal(codec.encode_residual(residual, reference), residual)


# --- conditioning is load-bearing --------------------------------------------


def test_the_same_residual_with_a_different_reference_codes_differently():
    """The central property. If the analysis transform ignored `z_ref` these two
    would be identical, and 'conditional' would be a label rather than a
    mechanism."""
    _, codec = _codec()
    _randomise(codec)
    residual = torch.randn(1, 8, 16, 16)
    reference_a = torch.randn(1, 8, 16, 16)
    reference_b = torch.randn(1, 8, 16, 16)

    with torch.no_grad():
        coded_a = codec.encode_residual(residual, reference_a)
        coded_b = codec.encode_residual(residual, reference_b)

    assert not torch.allclose(coded_a, coded_b)


def test_synthesis_is_conditioned_too_not_just_analysis():
    _, codec = _codec()
    _randomise(codec)
    coded = torch.randn(1, 8, 16, 16)
    reference_a = torch.randn(1, 8, 16, 16)
    reference_b = torch.randn(1, 8, 16, 16)

    assert not torch.allclose(codec.decode_residual(coded, reference_a),
                              codec.decode_residual(coded, reference_b))


def test_gradients_flow_through_the_reference_path():
    """Conditioning must be trainable, not just present in the forward pass."""
    _, codec = _codec()
    _randomise(codec)
    residual = torch.randn(1, 8, 16, 16)
    reference = torch.randn(1, 8, 16, 16, requires_grad=True)

    codec.encode_residual(residual, reference).pow(2).mean().backward()

    assert reference.grad is not None
    assert float(reference.grad.abs().sum()) > 0


def test_every_conditional_parameter_receives_gradient():
    _, codec = _codec()
    _randomise(codec)
    residual = torch.randn(2, 8, 16, 16)
    reference = torch.randn(2, 8, 16, 16)

    coded = codec.encode_residual(residual, reference)
    codec.decode_residual(coded, reference).pow(2).mean().backward()

    dead = [name for name, parameter in codec.named_parameters()
            if parameter.grad is None or float(parameter.grad.abs().sum()) == 0]
    assert not dead, f"no gradient reached: {dead}"


def test_inference_is_deterministic():
    _, codec = _codec()
    _randomise(codec)
    codec.eval()
    residual = torch.randn(1, 8, 16, 16)
    reference = torch.randn(1, 8, 16, 16)

    with torch.no_grad():
        first = codec.encode_residual(residual, reference)
        second = codec.encode_residual(residual, reference)

    assert torch.equal(first, second)


def test_conditioning_is_spatially_local_not_global_pooling():
    """A 3x3 convolutional stack has a bounded receptive field, so changing the
    reference in one corner must not alter the coded residual in the opposite
    one. This distinguishes real per-position conditioning from a global
    summary that happens to change everything."""
    _, codec = _codec(latent_channels=4, hidden_channels=8)
    _randomise(codec)
    residual = torch.randn(1, 4, 16, 16)
    reference = torch.randn(1, 4, 16, 16)
    perturbed = reference.clone()
    perturbed[:, :, 0, 0] += 5.0

    with torch.no_grad():
        base = codec.encode_residual(residual, reference)
        changed = codec.encode_residual(residual, perturbed)
    difference = (base - changed).abs()

    assert float(difference[:, :, 0, 0].sum()) > 0, "the perturbed position must change"
    assert float(difference[:, :, -1, -1].sum()) == 0, "a distant position must not"


# --- checkpointing -----------------------------------------------------------


def test_a_saved_codec_round_trips(tmp_path):
    m10i, codec = _codec()
    _randomise(codec)
    path = tmp_path / "codec.pt"
    torch.save({"codec_state_dict": codec.state_dict(),
                "codec_config": codec.config_dict()}, path)

    restored, _ = m10i.load_codec(path)
    residual = torch.randn(1, 8, 16, 16)
    reference = torch.randn(1, 8, 16, 16)

    assert restored.config_dict() == codec.config_dict()
    with torch.no_grad():
        assert torch.equal(restored.encode_residual(residual, reference),
                           codec.encode_residual(residual, reference))


# --- training behaviour -------------------------------------------------------


def test_training_produces_finite_losses_and_live_gradients():
    """A miniature version of the real loop: frozen decoder, QAT noise on the
    coded tensor, L = D + lambda*R."""
    from nvc.compression.calibration import calibrate_quantization_params
    from nvc.models.autoencoder import BaselineAutoencoder
    from nvc.training.quantization_noise import QuantizationNoise
    from nvc.training.rate_estimator import RateEstimator

    m10i, codec = _codec(latent_channels=4, hidden_channels=8)
    _randomise(codec)
    torch.manual_seed(0)
    model = BaselineAutoencoder(in_channels=3, latent_channels=4, base_channels=8).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    frames = torch.rand(2, 3, 32, 32)
    z_t = model.encode(frames)
    z_ref = z_t + 0.05 * torch.randn_like(z_t)
    params = calibrate_quantization_params(z_t - z_ref, bits=4, mode="per_channel")
    noise = QuantizationNoise(params.scale.clone(), bits=4, mode="per_channel")
    estimator = RateEstimator(params.scale.clone(), bits=4, mode="per_channel",
                              track_scale=True, scale_momentum=0.99)

    coded = codec.encode_residual(z_t - z_ref, z_ref)
    reconstruction = model.decode(z_ref + codec.decode_residual(noise.apply(coded), z_ref))
    distortion = F.mse_loss(reconstruction, frames)
    rate = estimator(coded, 32 * 32)
    loss = distortion + 3.0e-4 * rate
    loss.backward()

    assert math.isfinite(float(loss))
    assert math.isfinite(float(distortion)) and math.isfinite(float(rate))
    assert float(rate) > 0
    total = sum(float(p.grad.abs().sum()) for p in codec.parameters() if p.grad is not None)
    assert total > 0, "no gradient reached the conditional codec"
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in model.parameters()), (
        "the frozen encoder/decoder must not accumulate gradient")


# --- checkpoint selection (M10G convention) -----------------------------------


def test_checkpoint_selection_is_deterministic_and_validation_only():
    """M10I selects with the M10G rule; these two properties are what stop the
    selection from quietly peeking at the benchmark."""
    convention = _load_script("m10g_evaluation_convention")
    history = [{"epoch": 1, "val_loss": 5e-4, "val_psnr": 30.0, "rate_enabled": True},
               {"epoch": 2, "val_loss": 4e-4, "val_psnr": 31.0, "rate_enabled": True},
               {"epoch": 3, "val_loss": 4.6e-4, "val_psnr": 30.2, "rate_enabled": True}]

    first = convention.select_checkpoint(history, objective_key="val_loss")
    second = convention.select_checkpoint(list(reversed(history)), objective_key="val_loss")

    assert first["selected_epoch"] == second["selected_epoch"] == 2
    assert first["selection_domain"] == "validation"
    assert first["final_epoch"] == 3, "the final checkpoint is retained as secondary"


def test_a_test_metric_in_the_training_history_is_refused():
    convention = _load_script("m10g_evaluation_convention")
    history = [{"epoch": 1, "val_loss": 5e-4, "rate_enabled": True},
               {"epoch": 2, "val_loss": 4e-4, "rate_enabled": True, "test_psnr": 31.0}]

    with pytest.raises(convention.TestMetricLeakError):
        convention.select_checkpoint(history, objective_key="val_loss")


def test_the_declared_rate_points_are_data_driven_not_the_old_default():
    m10i = _load_script("m10i_conditional_residual")

    assert m10i.RATE_POINTS == (5, 4, 3)
    assert m10i.TRAIN_BITS == 4, "one model, trained at the middle point"
    assert m10i.FROZEN_LAMBDA == pytest.approx(3.0e-4)
    source = Path("scripts/m10i_conditional_residual.py").read_text(encoding="utf-8")
    assert "vertical" in source.lower(), "the rate-point choice must record its evidence"
