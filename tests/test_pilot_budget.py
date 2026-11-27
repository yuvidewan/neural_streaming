"""docs/PLAN_budget_pilot.md, "The budget, enforced in code": the pilot's decoder must stay at or
below 10,000 multiply-adds per pixel for an I-frame and a P-frame, and its encoder at or below
60,000, counted by PyTorch's FLOP counter at 1920x1088 with MAC = FLOP / 2. A pilot whose
decoder is over budget is not a valid result, so this test guards every training run."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from nvc.pilot.model import (CONFIG_PATH, PilotCodec, decoder_macs_per_pixel,
                             encoder_macs_per_pixel, load_config, pad_to)

DECODER_BUDGET = 10_000
ENCODER_BUDGET = 60_000


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return PilotCodec(load_config(CONFIG_PATH)).eval()


@pytest.mark.parametrize("intra", [True, False], ids=["I", "P"])
def test_decoder_within_budget(model, intra):
    macs = decoder_macs_per_pixel(model, 1088, 1920, intra=intra)
    assert 5_000 < macs <= DECODER_BUDGET, macs     # the lower bound catches a broken counter


@pytest.mark.parametrize("intra", [True, False], ids=["I", "P"])
def test_encoder_within_budget(model, intra):
    assert encoder_macs_per_pixel(model, 1088, 1920, intra=intra) <= ENCODER_BUDGET


def test_the_counter_counts_a_known_layer():
    from nvc.pilot.model import count_macs
    conv = torch.nn.Conv2d(8, 16, 3, padding=1)
    assert count_macs(lambda: conv(torch.zeros(1, 8, 10, 10))) == 8 * 16 * 9 * 100


def test_training_forward_backward(model):
    model.train()
    clip = torch.rand(2, 2, 3, 64, 64)
    levels = torch.tensor([0, 63])
    features, total = None, 0.0
    for t in range(2):
        x_hat, features, bits = model(clip[:, t], features, levels)
        assert x_hat.shape == clip[:, t].shape and bits.shape == (2,)
        assert torch.isfinite(bits).all() and (bits > 0).all()
        total = total + ((x_hat - clip[:, t]) ** 2).mean() + (bits / 4096).mean() * 1e-3
    total.backward()
    assert model.syn_out[0].weight.grad is not None and model.log_q.grad is not None
    model.eval()
    model.zero_grad(set_to_none=True)


def test_real_range_coder_round_trip(model):
    """encode -> real range coder -> decode gives the encoder's reconstruction exactly, with
    an odd frame size padded to 64; ideal and real sizes agree to within a few percent even
    for random weights."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import pilot_budget as pb
    torch.manual_seed(1)
    frames = torch.rand(3, 3, 72, 100)
    recon, bits, real, records = pb.code_video(model, frames, 40, torch.device("cpu"),
                                               pb.RangeCoder())
    assert recon.shape == frames.shape
    assert all(r["decode_exact"] for r in records)
    assert [r["type"] for r in records] == ["I", "P", "P"]
    assert 0.9 < real * 8 / bits < 1.1


def test_pad_to_64():
    assert pad_to(torch.zeros(1, 3, 1080, 1920)).shape[-2:] == (1088, 1920)
    assert pad_to(torch.zeros(1, 3, 128, 64)).shape[-2:] == (128, 64)
