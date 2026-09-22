"""Tests for the Stage 1 transform: GDN/IGDN and the residual
analysis/synthesis autoencoder (PARITY_ROADMAP.md Stage 1).

Everything here runs on CPU with tiny channel counts and 32x32 frames, so
the suite stays fast. The one exception is the parameter-budget test,
which builds the real default configuration - that costs a few hundred
milliseconds and is the whole point of Stage 1, so it is worth it.

The properties under test are the ones a codec actually depends on:
the [0, 1] output range, the exact stride-16 shape contract, encoding a
frame giving the same latent whether it arrives alone or in a batch, and
GDN's parameters staying in the region where the square root is real.
"""

from __future__ import annotations

import pytest
import torch

from nvc.models import (
    AnalysisTransform,
    BaselineAutoencoder,
    GDN,
    IGDN,
    ResidualBlock,
    ResidualGDNAutoencoder,
    SynthesisTransform,
)
from nvc.models.gdn import LowerBound
from nvc.training.quantization_noise import QuantizationNoise

# Small enough to be instant, large enough that a 32x32 frame gives a
# non-degenerate 2x2 latent.
_TINY_KW = {"in_channels": 3, "latent_channels": 4, "base_channels": 8, "residual_blocks": 1}


# --- GDN: the normalization itself ------------------------------------------------


def test_gdn_preserves_shape():
    x = torch.rand(2, 6, 8, 8)

    assert GDN(6)(x).shape == x.shape


def test_gdn_with_zero_gamma_is_the_identity():
    """beta initializes to 1 and gamma to gamma_initial * I, so with
    gamma_initial=0 the divisor is sqrt(1) everywhere and the layer must
    pass its input through untouched. This pins the parameterization: if
    the pedestal were added or subtracted in the wrong place, the divisor
    would drift off 1 and this would fail."""
    x = torch.randn(2, 5, 4, 4)

    y = GDN(5, gamma_initial=0.0)(x)

    assert torch.allclose(y, x, atol=1e-6)


def test_gdn_suppresses_large_activations_more_than_small_ones():
    """The defining behavior: the divisor grows with the magnitude of the
    activations, so GDN is contractive and compresses dynamic range."""
    gdn = GDN(3, gamma_initial=0.5)
    small = torch.full((1, 3, 2, 2), 0.1)
    large = torch.full((1, 3, 2, 2), 10.0)

    small_gain = (gdn(small) / small).mean().item()
    large_gain = (gdn(large) / large).mean().item()

    assert large_gain < small_gain
    assert large_gain < 1.0


def test_igdn_expands_where_gdn_contracts():
    x = torch.full((1, 3, 2, 2), 4.0)

    assert GDN(3, gamma_initial=0.5)(x).abs().max() < x.abs().max()
    assert IGDN(3, gamma_initial=0.5)(x).abs().max() > x.abs().max()


def test_igdn_ignores_an_explicit_inverse_argument():
    """IGDN is GDN with inverse pinned on; passing inverse=False must not
    silently produce a forward GDN under an inverse name."""
    assert IGDN(3, inverse=False).inverse is True


def test_gdn_parameters_stay_non_negative_after_a_hostile_update():
    """A real training run can drive a parameter negative. The bounded
    accessors must still hand forward() a legal (non-negative) beta and
    gamma, or the square root returns NaN."""
    gdn = GDN(4)
    with torch.no_grad():
        gdn.beta_reparam.fill_(-5.0)
        gdn.gamma_reparam.fill_(-5.0)

    assert (gdn.beta() >= 0).all()
    assert (gdn.gamma() >= 0).all()
    assert torch.isfinite(gdn(torch.randn(1, 4, 4, 4))).all()


def test_gdn_is_finite_on_zeros_and_on_large_inputs():
    gdn = GDN(4)

    assert torch.isfinite(gdn(torch.zeros(1, 4, 4, 4))).all()
    assert torch.isfinite(gdn(torch.full((1, 4, 4, 4), 1e4))).all()


def test_gdn_gradients_reach_beta_and_gamma():
    gdn = GDN(4)

    gdn(torch.randn(2, 4, 4, 4)).square().mean().backward()

    assert gdn.beta_reparam.grad is not None and gdn.beta_reparam.grad.abs().sum() > 0
    assert gdn.gamma_reparam.grad is not None and gdn.gamma_reparam.grad.abs().sum() > 0


def test_gdn_rejects_a_wrong_shaped_input():
    with pytest.raises(ValueError, match="B, 4, H, W"):
        GDN(4)(torch.rand(2, 3, 8, 8))

    with pytest.raises(ValueError, match="B, 4, H, W"):
        GDN(4)(torch.rand(4, 8, 8))


@pytest.mark.parametrize(
    "kwargs",
    [{"channels": 0}, {"channels": 4, "beta_minimum": -1.0}, {"channels": 4, "gamma_initial": -1.0}],
)
def test_gdn_rejects_illegal_construction(kwargs):
    with pytest.raises(ValueError):
        GDN(**kwargs)


# --- LowerBound: the asymmetric gradient that keeps a pinned parameter alive -------


def test_lower_bound_clamps_the_value():
    x = torch.tensor([-1.0, 0.0, 0.5, 2.0])

    assert torch.equal(LowerBound.apply(x, 0.5), torch.tensor([0.5, 0.5, 0.5, 2.0]))


def test_lower_bound_passes_gradient_above_the_bound_and_when_it_points_back():
    """Three regimes in one check:
      - x above the bound: gradient passes regardless of sign;
      - x below the bound with a negative gradient (descent would raise x,
        back toward the legal region): gradient passes;
      - x below the bound with a positive gradient (descent would push it
        further below): gradient is blocked.
    Getting the last two the wrong way round is the classic bug - it makes
    a parameter that touches the floor stay there forever."""
    x = torch.tensor([2.0, 0.1, 0.1], requires_grad=True)
    incoming = torch.tensor([1.0, -1.0, 1.0])

    LowerBound.apply(x, 0.5).backward(incoming)

    assert torch.equal(x.grad, torch.tensor([1.0, -1.0, 0.0]))


# --- ResidualBlock ----------------------------------------------------------------


def test_residual_block_starts_as_the_identity():
    """The second convolution is zero-initialized, so adding depth cannot
    move the starting point."""
    block = ResidualBlock(6)
    x = torch.randn(2, 6, 4, 4)

    assert torch.allclose(block(x), x, atol=1e-6)


def test_residual_block_stops_being_the_identity_once_trained():
    """Negative control for the test above: the identity must come from
    the zero initialization, not from the block being unable to do
    anything at all."""
    block = ResidualBlock(6)
    with torch.no_grad():
        block.conv2.weight.normal_(0.0, 0.5)
    x = torch.randn(2, 6, 4, 4)

    assert not torch.allclose(block(x), x, atol=1e-6)


def test_residual_block_preserves_shape():
    x = torch.rand(2, 6, 8, 8)

    assert ResidualBlock(6)(x).shape == x.shape


# --- Analysis / synthesis shape contract ------------------------------------------


@pytest.mark.parametrize("size", [32, 48, 64])
def test_analysis_downsamples_by_exactly_sixteen(size):
    analysis = AnalysisTransform(**_TINY_KW)

    z = analysis(torch.rand(2, 3, size, size))

    assert z.shape == (2, 4, size // 16, size // 16)


def test_synthesis_upsamples_by_exactly_sixteen():
    synthesis = SynthesisTransform(
        out_channels=3, latent_channels=4, base_channels=8, residual_blocks=1
    )

    y = synthesis(torch.rand(2, 4, 2, 2))

    assert y.shape == (2, 3, 32, 32)


def test_analysis_rejects_sizes_that_are_not_divisible_by_sixteen():
    with pytest.raises(ValueError, match="divisible by 16"):
        AnalysisTransform(**_TINY_KW)(torch.rand(1, 3, 40, 32))


def test_analysis_rejects_the_wrong_channel_count():
    with pytest.raises(ValueError, match=r"B, 3, H, W"):
        AnalysisTransform(**_TINY_KW)(torch.rand(1, 1, 32, 32))


def test_synthesis_rejects_the_wrong_latent_channel_count():
    synthesis = SynthesisTransform(
        out_channels=3, latent_channels=4, base_channels=8, residual_blocks=1
    )

    with pytest.raises(ValueError, match=r"B, 4, H, W"):
        synthesis(torch.rand(1, 3, 2, 2))


# --- The autoencoder --------------------------------------------------------------


def test_round_trip_shape_and_output_range():
    model = ResidualGDNAutoencoder(**_TINY_KW).eval()
    x = torch.rand(2, 3, 32, 32)

    with torch.no_grad():
        y = model(x)

    assert y.shape == x.shape
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_encoding_a_frame_does_not_depend_on_the_rest_of_the_batch():
    """The contract that rules out BatchNorm: a frame's latent must not be
    a function of what else was in the batch, or the encoder and the
    decoder can disagree about what the bitstream means.

    Checked to a tight tolerance rather than bit-exactly, and the
    difference is not this model's arithmetic: PyTorch's 3x3 and 1x1
    convolution kernels (the residual blocks, and the 1x1 GDN uses for its
    channel mixing) pick a different code path at batch 4 than at batch 1,
    which moves the last couple of mantissa bits. Measured at ~7e-9 here,
    against a quantizer step of order 0.1 - about seven orders of
    magnitude below one symbol - and the codec encodes a frame at a time
    anyway. Real batch leakage would land far above this bound."""
    torch.manual_seed(0)
    model = ResidualGDNAutoencoder(**_TINY_KW).eval()
    batch = torch.rand(4, 3, 32, 32)

    with torch.no_grad():
        together = model.encode(batch)
        alone = model.encode(batch[2:3])

    assert torch.allclose(alone, together[2:3], rtol=0, atol=1e-6)


def test_encoding_is_deterministic():
    """Bit-exact repeatability at a fixed batch shape, which is the
    property the codec actually needs: encode and decode must agree on the
    latent down to the last bit. See the batch-composition test above for
    the one place that only holds numerically."""
    model = ResidualGDNAutoencoder(**_TINY_KW).eval()
    x = torch.rand(1, 3, 32, 32)

    with torch.no_grad():
        assert torch.equal(model.encode(x), model.encode(x))

    batch = torch.rand(4, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(model.encode(batch), model.encode(batch))


def test_config_dict_rebuilds_an_identical_model():
    torch.manual_seed(0)
    model = ResidualGDNAutoencoder(**_TINY_KW).eval()

    rebuilt = ResidualGDNAutoencoder(**model.config_dict()).eval()
    rebuilt.load_state_dict(model.state_dict())

    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(model(x), rebuilt(x))


def test_quantization_noise_is_not_part_of_the_config_or_the_state_dict():
    """Same contract as BaselineAutoencoder: quantization noise is a
    training method, not an architectural choice, so a checkpoint rebuilt
    from config_dict() must never silently re-attach it."""
    noise = QuantizationNoise(torch.full((1, 4, 1, 1), 0.1), bits=4, mode="per_channel")
    model = ResidualGDNAutoencoder(**_TINY_KW, quantization_noise=noise)

    assert "quantization_noise" not in model.config_dict()
    assert not any("quantization_noise" in key for key in model.state_dict())


def test_quantization_noise_fires_in_training_mode_only():
    torch.manual_seed(0)
    noise = QuantizationNoise(torch.full((1, 4, 1, 1), 2.0), bits=4, mode="per_channel")
    model = ResidualGDNAutoencoder(**_TINY_KW, quantization_noise=noise)
    x = torch.rand(1, 3, 32, 32)

    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x), model(x))

    model.train()
    with torch.no_grad():
        assert not torch.equal(model(x), model(x))


def test_every_parameter_receives_a_gradient():
    """A dead branch - a residual stack never reached, a GDN whose output
    is discarded - would show up here and nowhere else until a training
    run silently wasted its capacity."""
    model = ResidualGDNAutoencoder(**_TINY_KW)
    x = torch.rand(2, 3, 32, 32)

    torch.nn.functional.mse_loss(model(x), x).backward()

    starved = [
        name for name, parameter in model.named_parameters()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all()
    ]
    assert not starved, f"parameters with no finite gradient: {starved}"


def test_the_model_can_actually_learn():
    """A smoke test against the failure GDN is prone to: a divisor that
    collapses toward zero and takes the loss to NaN a few steps in. Twenty
    steps of Adam on one fixed batch must reduce the reconstruction error
    and keep every number finite."""
    torch.manual_seed(0)
    model = ResidualGDNAutoencoder(**_TINY_KW).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(2, 3, 32, 32)

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(value)) for value in losses)
    assert losses[-1] < losses[0]


def test_the_interface_matches_the_baseline_autoencoder():
    """Existing training, checkpoint and evaluation code is written
    against BaselineAutoencoder's surface; Stage 1 swaps the architecture,
    not the call sites."""
    baseline = BaselineAutoencoder(in_channels=3, latent_channels=4, base_channels=8)
    stage1 = ResidualGDNAutoencoder(**_TINY_KW)

    for name in ("encode", "decode", "forward", "num_parameters", "config_dict"):
        assert callable(getattr(stage1, name))
        assert callable(getattr(baseline, name))


def test_the_default_configuration_meets_the_stage_1_capacity_target():
    """PARITY_ROADMAP.md Stage 1 asks for ~8-12M parameters at 192-256
    channels, against the baseline's 593k. This is the one number the
    whole stage is built around, so it is pinned rather than left to
    whatever the defaults happen to produce."""
    model = ResidualGDNAutoencoder()

    assert model.latent_channels == 192 and model.base_channels == 192
    assert 8_000_000 <= model.num_parameters() <= 12_000_000
    assert model.num_parameters() > 10 * BaselineAutoencoder().num_parameters()
