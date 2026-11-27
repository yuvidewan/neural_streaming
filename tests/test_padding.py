"""docs/PLAN_fair_eval.md proof 2: frames whose sides do not divide by 16 code and decode
exactly, through the real codec (wide I-frames, A2 P-frames, bit-aware motion).

Needs the trained weights and CUDA; skipped otherwise. Sizes are width x height as in the plan:
72x64, 70x98 and a portrait 1080x1920."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import p4_gate3 as pg  # noqa: E402


def test_padded_size_rule():
    assert pg.padded_size(1080, 1920) == (1088, 1920)
    assert pg.padded_size(1072, 1920) == (1072, 1920)
    assert pg.padded_size(64, 72) == (64, 80)
    assert pg.padded_size(98, 70, 32) == (128, 96)


def test_pad_frames_repeats_edges_and_leaves_aligned_frames_alone():
    frames = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)
    assert pg.pad_frames(torch.zeros(1, 3, 32, 48)).shape == (1, 3, 32, 48)
    out = pg.pad_frames(frames)
    assert out.shape == (2, 3, 16, 16)
    assert torch.equal(out[..., :4, :6], frames)
    assert torch.equal(out[..., 10, 3], frames[..., 3, 3])      # bottom rows repeat the last row
    assert torch.equal(out[..., 2, 12], frames[..., 2, 5])      # right columns repeat the last column


weights = ROOT / "outputs" / "p4_wide_latent" / "vimeo"
needs_codec = pytest.mark.skipif(not torch.cuda.is_available() or not weights.is_dir(),
                                 reason="needs CUDA and the trained weights")


@pytest.fixture(scope="module")
def coders():
    sys.path.insert(0, str(ROOT / "outputs" / "p12_audit2"))
    import os
    old = os.getcwd()
    os.chdir(ROOT)
    try:
        import bitstream as bs
        yield bs, bs.Coders("1e-2", torch.device("cuda"))
    finally:
        os.chdir(old)


def smooth_frames(count: int, height: int, width: int) -> torch.Tensor:
    yy = torch.linspace(0, 1, height).view(height, 1)
    xx = torch.linspace(0, 1, width).view(1, width)
    out = []
    for t in range(count):
        r = 0.5 + 0.4 * torch.sin(6 * xx + 0.3 * t) * torch.cos(5 * yy)
        g = 0.5 + 0.3 * torch.cos(4 * xx - 7 * yy + 0.2 * t)
        b = 0.4 + 0.3 * yy + 0.1 * torch.sin(9 * xx + t)
        out.append(torch.stack([r, g, b]).clamp(0, 1))
    return torch.stack(out)


@needs_codec
@pytest.mark.parametrize("width,height,count", [(72, 64, 4), (70, 98, 4), (1080, 1920, 3)])
def test_odd_sizes_round_trip_exactly(coders, tmp_path, width, height, count):
    bs, c = coders
    frames = smooth_frames(count, height, width)
    path = tmp_path / f"{width}x{height}.nvct"
    encoded = pg.encode_sequence(bs.mc, bs.m21, c.intra, c.icoder, c.pcoder, frames, path,
                                 c.rig, c.tables, gop=30, block_size=c.args.block_size,
                                 search_range=c.args.search_range, identity=c.identity,
                                 motion_lambda=0.25)
    assert encoded["reconstructions"].shape == frames.shape
    header = bs.mc.TemporalStreamReader(path).header
    assert (header.image_width, header.image_height) == (width, height)
    decoded = c.decode(path)
    assert torch.equal(decoded.cpu(), encoded["reconstructions"])
    mse = torch.mean((decoded.cpu().clamp(0, 1) - frames) ** 2)
    assert 10 * torch.log10(1 / mse) > 25           # it actually codes the picture
