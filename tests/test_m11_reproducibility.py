"""Permanent regression: the temporal codec is reproducible ACROSS PROCESSES.

M10L found that two runs of the same codec, same checkpoint, same data, emitted
different bytes, because `calibrate_grids` called `model.decode` outside
`deterministic_kernels()`. Commit 61dd8434 added the guard and tested that it
is active. These tests check the property that actually matters - two
independent Python processes produce identical calibration, z_ref, residual
symbols, stream bytes and reconstruction.

Measured on a GPU when this test was written: with the pre-fix module, two
processes disagreed on 9 of the 15 fingerprinted fields (residual grid off by
5e-5 relative, stream bytes by 1-2); with the fix, all 15 agree. On a CPU-only
machine the drift cannot occur, so the cross-process test passes trivially
there - it still guards the seeding and ordering that reproducibility also
depends on, and it is a real check on any machine with CUDA.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Small enough for the suite, large enough to contain I- and P-frames, real
# block motion and two full GOPs.
SMALL = ["--source", "synthetic", "--size", "64", "--frames", "8",
         "--calibration-frames", "12", "--gop", "4", "--search-range", "8",
         "--decode-repeats", "2"]


def _child(tmp_path: Path, name: str, *extra: str) -> dict:
    out = tmp_path / f"{name}.json"
    completed = subprocess.run(
        [sys.executable, "scripts/m11_reproducibility.py", "--child", "--out", str(out),
         *SMALL, *extra],
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr[-3000:]
    return json.loads(out.read_text(encoding="utf-8"))


def test_two_independent_processes_produce_identical_codec_output(tmp_path):
    """The permanent regression. Identical calibration, z_ref, residual symbols,
    stream bytes and reconstruction, from two separate interpreters."""
    repro = _load_script("m11_reproducibility")

    first = _child(tmp_path, "a")
    second = _child(tmp_path, "b")
    comparison = repro.compare(first, second)

    assert comparison["fields_compared"] >= 19
    assert comparison["mismatched_fields"] == [], comparison
    assert comparison["residual_scale_max_abs_diff"] == 0.0
    assert all(delta == 0 for delta in comparison["stream_byte_diff"].values())
    assert first["closed_loop"]["p_frames"] > 0, "the clip must contain P-frames"
    # M11's own path is part of the permanent regression: probabilities,
    # frequency tables and payload identical across processes, and decodable.
    for field in ("m11_probabilities", "m11_frequencies", "m11_payload"):
        assert first["entropy"][field] == second["entropy"][field], field
    assert first["entropy"]["m11_round_trip"] and second["entropy"]["m11_round_trip"]


def test_the_fingerprint_is_not_vacuous(tmp_path):
    """"Identical" only means something if a real difference would show up. A
    different seed changes the model and the frames, and every output field has
    to move with it."""
    repro = _load_script("m11_reproducibility")

    comparison = repro.compare(_child(tmp_path, "a"), _child(tmp_path, "b", "--seed", "7"))

    for field in ("calibration.residual_scale", "closed_loop.z_ref",
                  "closed_loop.residual_symbols", "stream.sha256", "reconstruction"):
        assert field in comparison["mismatched_fields"], field


def test_compare_reports_every_differing_field():
    repro = _load_script("m11_reproducibility")
    base = {"calibration": {"residual_scale": "a", "x": 1}, "stream": {
        "sha256": "s", "container_bytes": 10, "residual_bytes": 6, "motion_bytes": 2},
        "reconstruction": "r", "raw": {"residual_scale": [1.0, 2.0]},
        "device": "cuda", "decode": {"unguarded_max_abs_diff": 1e-4}}
    other = json.loads(json.dumps(base))
    other["stream"]["motion_bytes"] = 3
    other["reconstruction"] = "q"
    other["raw"]["residual_scale"] = [1.0, 2.5]
    other["device"] = "cpu"
    other["decode"]["unguarded_max_abs_diff"] = 9.0

    comparison = repro.compare(base, other)

    assert comparison["mismatched_fields"] == ["reconstruction", "stream.motion_bytes"], \
        "run metadata and within-process decode spread are not codec output"
    assert comparison["stream_byte_diff"]["motion_bytes"] == 1
    assert comparison["residual_scale_max_abs_diff"] == pytest.approx(0.5)
    assert not comparison["identical"]


def test_the_guard_makes_decode_self_consistent():
    """The root cause in isolation: with the guard, repeated decodes of one
    latent are bit-identical on any device."""
    repro = _load_script("m11_reproducibility")
    mc = _load_script("m10h_motion_compensation")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, _, clip = repro.synthetic_setup(device, seed=0, size=64, frames=2)

    with torch.no_grad():
        latent = model.encode(clip.load_frames()[0:1].to(device))
    spread = repro.decode_self_consistency(mc, model, latent, repeats=3)

    assert spread["guarded_max_abs_diff"] == 0.0


def test_synthetic_sequences_contain_real_motion():
    """The regression is only meaningful if the block matcher has work to do."""
    repro = _load_script("m11_reproducibility")
    frames = repro.synthetic_frames(3, 64, seed=0)

    assert frames.shape == (3, 3, 64, 64)
    assert not torch.equal(frames[0], frames[1])
    assert float(frames.min()) >= 0.0 and float(frames.max()) <= 1.0
