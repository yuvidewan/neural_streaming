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


def _mc():
    spec = importlib.util.spec_from_file_location(
        "m10h_motion_compensation", Path("scripts") / "m10h_motion_compensation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batched_motion_and_warp_match_the_deployed_search_per_image():
    p1, mc = _p1(), _mc()
    torch.manual_seed(0)
    reference = torch.rand(3, 3, 48, 64)
    # A shifted copy plus noise, so the search has a real answer to find.
    current = torch.roll(reference, shifts=(3, -5), dims=(2, 3)) + 0.01 * torch.rand(3, 3, 48, 64)
    current[1] = reference[1]          # an all-zero-motion image exercises the tie-break
    motion = p1.batched_block_motion(reference, current, search_range=6)
    warped = p1.batched_warp(reference, motion)
    for i in range(3):
        expected = mc.estimate_block_motion(reference[i:i + 1], current[i:i + 1], search_range=6)
        assert torch.equal(motion[i], expected)
        assert torch.equal(warped[i:i + 1], mc.warp_blocks(reference[i:i + 1], expected))
    assert motion[1].abs().sum() == 0


def test_vimeo_clips_share_one_crop_and_flip_across_the_clip(tmp_path):
    import cv2
    import numpy as np
    p1 = _p1()
    clip = tmp_path / "00001" / "0001"
    clip.mkdir(parents=True)
    for i in range(1, 8):
        image = np.full((64, 96, 3), i * 30, dtype=np.uint8)
        cv2.imwrite(str(clip / f"im{i}.png"), image)
    (tmp_path / "00001" / "0002").mkdir()          # incomplete: no frames
    dirs = p1.complete_clip_dirs(tmp_path)
    assert dirs == [clip]
    sample = p1.VimeoClips(dirs, length=4, crop=32)[0]
    assert sample.shape == (4, 3, 32, 32) and sample.dtype == torch.uint8
    values = sample[:, 0, 0, 0].tolist()
    assert values == sorted(values) and len(set(values)) == 4   # consecutive frames, in order


def test_unrolled_step_chains_reconstructions_and_trains_the_coder():
    from nvc.models.residual_transform import ResidualGDNAutoencoder
    p1 = _p1()
    torch.manual_seed(0)
    intra = ResidualGDNAutoencoder(latent_channels=8, base_channels=16).eval()
    model = p1.ConditionalPFrameCoder.from_intra(intra, context_channels=8).train()
    clip = torch.rand(2, 3, 3, 32, 32)
    mse, bits = p1.unrolled_step(model, intra, clip)
    (mse + 1e-3 * bits).backward()
    assert model.encoder.net[0].weight.grad is not None
    assert all(p.grad is None for p in intra.parameters())


def test_per_frame_backward_gives_the_whole_chain_gradient():
    from nvc.models.residual_transform import ResidualGDNAutoencoder
    p1 = _p1()
    torch.manual_seed(0)
    intra = ResidualGDNAutoencoder(latent_channels=8, base_channels=16).eval()
    model = p1.ConditionalPFrameCoder.from_intra(intra, context_channels=8).eval()
    clip = torch.rand(2, 4, 3, 32, 32)
    mse, bits = p1.unrolled_step(model, intra, clip)
    (mse + 0.01 * bits / (32 * 32)).backward()
    whole = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    model.zero_grad(set_to_none=True)
    p1.unrolled_step(model, intra, clip, 0.01)
    split = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(whole) == len(split)
    assert all(torch.allclose(a, b, rtol=1e-4, atol=1e-7) for a, b in zip(whole, split))
