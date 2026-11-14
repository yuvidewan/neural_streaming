import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p1_pframe as p1  # noqa: E402
import p3_hyperprior as p3  # noqa: E402


def _models():
    torch.manual_seed(0)
    base = p1.ConditionalPFrameCoder(latent_channels=8, base_channels=16, context_channels=4)
    return base.eval(), p3.HyperpriorPFrameCoder(base, hyper_channels=4, hidden=8).eval()


def test_step_zero_is_the_base_coder_plus_z():
    base, hyper = _models()
    frame, warped = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        plain = base(frame, warped)
        side = hyper(frame, warped)
    assert torch.equal(plain["reconstruction"], side["reconstruction"])
    assert torch.allclose(plain["bits"], side["y_bits"])
    assert torch.all(side["z_bits"] > 0)
    assert torch.allclose(side["bits"], side["y_bits"] + side["z_bits"])


def test_side_information_reaches_the_prior_after_training_starts():
    base, hyper = _models()
    hyper.train()
    frame, warped = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    result = hyper(frame, warped)
    (result["bits"].mean() + (result["reconstruction"] - frame).pow(2).mean()).backward()
    assert hyper.h_s[-1].weight.grad.abs().sum() > 0
    assert hyper.z_log_scale.grad is not None
