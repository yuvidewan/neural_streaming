"""scripts/motion_pred.py: predicted motion coding and the bit-aware search (docs/PLAN_motion_rate.md)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mp = _load("motion_pred")
mc = _load("m10h_motion_compensation")


def _random_field(rng, by, bx, r=16):
    return rng.integers(-r, r + 1, size=(2, by, bx)).astype(np.int64)


def test_vectorised_predictor_matches_sequential():
    rng = np.random.default_rng(0)
    for by, bx in ((1, 1), (1, 5), (5, 1), (4, 7), (16, 16)):
        field = _random_field(rng, by, bx)
        pred = mp.predict_field(field)
        for y in range(by):
            for x in range(bx):
                assert tuple(pred[:, y, x]) == mp._predict_at(field, y, x, bx), (by, bx, y, x)


def test_round_trip_is_exact():
    rng = np.random.default_rng(1)
    for by, bx in ((1, 1), (3, 9), (16, 16), (67, 120)):
        field = torch.from_numpy(_random_field(rng, by, bx))
        payload, scales = mp.encode_field(field)
        decoded, decoded_scales = mp.decode_field(payload, (by, bx))
        assert torch.equal(decoded, field) and decoded_scales == scales


def test_smooth_field_is_cheap():
    smooth = torch.full((2, 16, 16), 3, dtype=torch.long)
    noisy = torch.from_numpy(_random_field(np.random.default_rng(2), 16, 16))
    assert len(mp.encode_field(smooth)[0]) < len(mp.encode_field(noisy)[0]) / 10


def test_tables_are_valid():
    cumulative, bits = mp.tables()
    assert cumulative.shape == (mp.SCALE_LEVELS, 2 * mp.MVD_RADIUS + 2)
    assert (np.diff(cumulative, axis=1) >= 1).all()
    assert (cumulative[:, -1] == mp.FREQUENCY_TOTAL).all()
    assert np.isfinite(bits).all()


def test_search_extremes():
    torch.manual_seed(0)
    ref = torch.rand(1, 3, 64, 64)
    cur = torch.roll(ref, (2, -3), (2, 3))
    cost = mc.block_sad_costs(ref, cur, search_range=4)
    # lambda_m = 0 keeps the lowest SAD everywhere (the true shift away from the edges)
    free = mp.search_rate_aware(cost, search_range=4, lambda_m=0.0, scale_indices=(30, 30))
    assert tuple(free[:, 1, 1].tolist()) == (-2, 3)
    # a huge lambda_m makes every mvd zero, so the whole field is the first block's (0, 0)
    stiff = mp.search_rate_aware(cost, search_range=4, lambda_m=1e9, scale_indices=(30, 30))
    assert (stiff == 0).all()


def test_block_sad_costs_match_loop_choice():
    torch.manual_seed(3)
    ref = torch.rand(1, 3, 48, 64)
    cur = torch.roll(ref, (1, 2), (2, 3)) + 0.01 * torch.rand(1, 3, 48, 64)
    cost = mc.block_sad_costs(ref, cur, search_range=3)
    assert cost.shape == (49, 3, 4)
    field = mc._estimate_block_motion_loop(ref, cur, search_range=3)
    span = 7
    index = (field[0] + 3) * span + (field[1] + 3)
    chosen = cost.gather(0, index.unsqueeze(0))[0]
    assert torch.equal(chosen, cost.amin(dim=0))
