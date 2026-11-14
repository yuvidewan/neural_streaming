import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p1_pframe as p1  # noqa: E402
import p4_wide_latent as p4  # noqa: E402


def _base():
    torch.manual_seed(0)
    # The real base width (192) is assumed by the new branches.
    return p1.ConditionalPFrameCoder(latent_channels=8, base_channels=192,
                                     context_channels=4).eval()


@pytest.mark.parametrize("kind", ["plain", "a1", "a2"])
def test_step_zero_reconstructs_exactly_like_the_base(kind):
    base = _base()
    wide = p4.WideLatentPFrameCoder(base, kind).eval()
    frame, warped = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        if kind != "plain":
            wide.calibrate(frame, warped)
        plain = base(frame, warped)
        result = wide(frame, warped)
    assert torch.equal(plain["reconstruction"], result["reconstruction"])
    if kind == "plain":
        assert torch.allclose(plain["bits"], result["bits"])
    else:
        assert torch.all(result["bits"] > plain["bits"])


@pytest.mark.parametrize("kind", ["a1", "a2"])
def test_new_branch_gets_gradient_and_starts_near_target_std(kind):
    wide = p4.WideLatentPFrameCoder(_base(), kind)
    frame, warped = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    wide.calibrate(frame, warped)
    wide.train()
    result = wide(frame, warped)
    ((result["reconstruction"] - frame).pow(2).mean() + 1e-4 * result["bits"].mean()).backward()
    assert wide.extra_syn.weight.grad.abs().sum() > 0
    with torch.no_grad():
        h = wide.base.encoder.net[:-1](torch.cat([frame, wide.base.context(warped)], 1))
        assert abs(float(wide.extra_enc(h).std()) - p4.TARGET_INIT_STD) < 0.05


def test_no_rounding_flag_changes_only_the_decoded_latent():
    wide = p4.WideLatentPFrameCoder(_base(), "a2").eval()
    frame, warped = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        a = wide(frame, warped)
        b = wide(frame, warped, rounding=False)
    assert torch.allclose(a["bits"], b["bits"])
    assert not torch.equal(a["reconstruction"], b["reconstruction"])
