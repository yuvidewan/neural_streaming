import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p1_gate3 as g3  # noqa: E402
import p1_pframe as p1  # noqa: E402
import p4_gate3 as p4g  # noqa: E402
import p4_wide_latent as p4  # noqa: E402


def _model():
    torch.manual_seed(0)
    base = p1.ConditionalPFrameCoder(latent_channels=8, base_channels=192, context_channels=4)
    wide = p4.WideLatentPFrameCoder(base, "a2")
    frame, warped = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    wide.calibrate(frame, warped)
    with torch.no_grad():   # make the detail branch matter
        wide.extra_syn.weight.normal_(0, 0.02)
        wide.extra_prior[-1].weight.normal_(0, 0.02)
    return wide.eval(), frame, warped


def test_coded_frame_matches_the_model_and_decodes_exactly():
    model, frame, warped = _model()
    tables = g3.cumulative_tables()
    with torch.no_grad():
        payload, reconstruction, ideal = p4g.encode_frame(model, frame, warped, tables)
        expected = model(frame, warped)
        decoded = p4g.decode_frame(model, payload, warped, tables)
    assert torch.allclose(reconstruction, expected["reconstruction"], atol=1e-6)
    assert torch.equal(decoded, reconstruction)
    assert abs(ideal - float(expected["bits"])) / float(expected["bits"]) < 0.05
    assert len(payload) * 8 < 1.1 * ideal + 64


def test_intra_frame_codes_against_a_gray_reference():
    model, frame, _ = _model()
    tables = g3.cumulative_tables()
    with torch.no_grad():
        payload, reconstruction, _ = p4g.encode_frame(model, frame, p4g.gray(frame), tables)
        decoded = p4g.decode_frame(model, payload, torch.full_like(frame, p4.GRAY), tables)
        expected = p4.code_intra(model, frame)["reconstruction"]
    assert torch.equal(decoded, reconstruction)
    assert torch.allclose(reconstruction, expected, atol=1e-6)
