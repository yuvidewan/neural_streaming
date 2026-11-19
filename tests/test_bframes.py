"""scripts/bframes.py: frame order, two-sided prediction and B payloads (docs/PLAN_bframes.md)."""

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


bf = _load("bframes")
mp = _load("motion_pred")


def test_order_round_trip_and_causality():
    for count in (1, 2, 5, 30, 31, 61, 96):
        for gop in (1, 4, 30):
            for m in (1, 2, 3, 4):
                order = bf.coding_order(count, gop, m)
                assert sorted(t for _, t, _ in order) == list(range(count)), (count, gop, m)
                assert bf.display_order([k for k, _, _ in order]) == order, (count, gop, m)
                done = set()
                for kind, t, refs in order:
                    assert all(r in done for r in refs)
                    assert all(r // gop == t // gop for r in refs)       # closed GOP
                    if kind == "B":
                        assert refs[0] < t < refs[1]
                    done.add(t)


def test_m1_is_all_p():
    order = bf.coding_order(61, 30, 1)
    assert [k for k, _, _ in order] == (["I"] + ["P"] * 29) * 2 + ["I"]
    assert all(refs == ((t - 1,) if k == "P" else ()) for k, t, refs in order)


def test_m4_hierarchy():
    order = bf.coding_order(9, 30, 4)
    assert order[:5] == [("I", 0, ()), ("P", 4, (0,)), ("B", 2, (0, 4)), ("B", 1, (0, 2)),
                         ("B", 3, (2, 4))]


def test_payload_round_trip():
    rng = np.random.default_rng(0)
    for by, bx in ((1, 1), (4, 7), (30, 53)):
        modes = torch.from_numpy(rng.integers(0, 3, size=(by, bx)))
        fwd = torch.from_numpy(rng.integers(-16, 17, size=(2, by, bx)))
        bwd = torch.from_numpy(rng.integers(-16, 17, size=(2, by, bx)))
        fwd, bwd = bf.clear_unused(fwd, bwd, modes)
        payload, _, _ = bf.encode_payload(fwd, bwd, modes)
        f, b, m = bf.decode_payload(payload, (by, bx))
        assert torch.equal(f, fwd) and torch.equal(b, bwd) and torch.equal(m, modes)


def test_unused_vectors_cost_nothing():
    rng = np.random.default_rng(1)
    modes = torch.zeros(10, 10, dtype=torch.long)                 # all forward
    fwd = torch.from_numpy(rng.integers(-16, 17, size=(2, 10, 10)))
    bwd = torch.from_numpy(rng.integers(-16, 17, size=(2, 10, 10)))
    fwd2, bwd2 = bf.clear_unused(fwd, bwd, modes)
    assert torch.equal(fwd2, fwd)
    assert (bwd2.numpy() - mp.predict_field(bwd2.numpy()) == 0).all()


def test_combine_and_modes():
    torch.manual_seed(0)
    cur = torch.rand(1, 3, 32, 32)
    noise = 0.3 * torch.rand(1, 3, 32, 32)
    fwd, bwd = cur.clone(), cur + noise
    fwd[:, :, :16] += noise[:, :, :16]                           # top half: both equally noisy
    zero = torch.zeros(2, 2, dtype=torch.float64)
    modes = bf.choose_modes(cur, fwd, bwd, 16, 0.0, zero, zero)
    assert (modes[1] == bf.FORWARD).all()                        # bottom: forward is exact
    pred = bf.combine(fwd, bwd, modes, 16)
    assert torch.equal(pred[:, :, 16:], cur[:, :, 16:])
