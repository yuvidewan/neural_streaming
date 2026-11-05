"""The P1 pilot's model and rate are sound before any number from it is trusted."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _p1():
    spec = importlib.util.spec_from_file_location("p1_pframe", Path("scripts") / "p1_pframe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_laplace_mass_over_the_integers_sums_to_one():
    p1 = _p1()
    symbols = torch.arange(-2000, 2001, dtype=torch.float64)
    for scale in (0.11, 0.7, 3.0, 25.0):
        bits = p1.laplace_bits(symbols, torch.tensor(scale, dtype=torch.float64))
        # The 1e-9 likelihood floor adds at most 4001e-9 of mass in the far tails.
        assert abs((2 ** -bits).sum().item() - 1) < 5e-6


def test_laplace_bits_are_symmetric_and_grow_away_from_zero():
    p1 = _p1()
    v = torch.tensor([0.0, 1.0, -1.0, 4.0, -4.0])
    bits = p1.laplace_bits(v, torch.tensor(1.5))
    assert torch.allclose(bits[1], bits[2]) and torch.allclose(bits[3], bits[4])
    assert bits[0] < bits[1] < bits[3]


def test_a_warm_started_coder_is_the_intra_codec_plus_a_prior():
    from nvc.models.residual_transform import ResidualGDNAutoencoder
    p1 = _p1()
    torch.manual_seed(0)
    intra = ResidualGDNAutoencoder(latent_channels=8, base_channels=16).eval()
    model = p1.ConditionalPFrameCoder.from_intra(intra, context_channels=8).eval()
    frame, warped = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    context = model.context(warped)
    latent = model.encoder(torch.cat([frame, context], dim=1))
    assert torch.allclose(latent, intra.encode(frame), atol=1e-6)
    result = model(frame, warped)
    decoded = result["symbols"] + model.prior(context).chunk(2, dim=1)[0]
    assert torch.allclose(result["reconstruction"], intra.decode(decoded).clamp(0, 1), atol=1e-6)
    assert result["bits"].shape == (2,)
    assert result["symbols"].shape == (2, 8, 4, 4)
    rebuilt = p1.ConditionalPFrameCoder(**model.config_dict())
    rebuilt.load_state_dict(model.state_dict())


def test_eval_symbols_are_integers_and_training_is_differentiable():
    p1 = _p1()
    torch.manual_seed(0)
    model = p1.ConditionalPFrameCoder(latent_channels=8, base_channels=16, context_channels=8)
    frame, warped = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    model.eval()
    symbols = model(frame, warped)["symbols"]
    assert torch.equal(symbols, symbols.round())
    model.train()
    result = model(frame, warped)
    (result["bits"].sum() + result["reconstruction"].sum()).backward()
    assert model.encoder.net[0].weight.grad is not None
    assert model.prior[0].weight.grad is not None


def test_static_bits_charges_the_train_histogram():
    p1 = _p1()
    histogram = torch.zeros(1, 2 * p1.SYMBOL_OFFSET, dtype=torch.float64)
    histogram[0, p1.SYMBOL_OFFSET] = 3
    histogram[0, p1.SYMBOL_OFFSET + 1] = 1
    zeros = torch.zeros(1, 1, 2, 2)
    ones = torch.ones(1, 1, 2, 2)
    assert p1.static_bits(zeros, histogram).item() < p1.static_bits(ones, histogram).item()
    expected = -4 * torch.log2(torch.tensor((3 + 1 / 1024) / 5, dtype=torch.float64)).item()
    assert abs(p1.static_bits(zeros, histogram).item() - expected) < 1e-9
