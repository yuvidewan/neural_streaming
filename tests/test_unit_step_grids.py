"""Tests for `scripts/unit_step_grids.py` - the refit chain's calibration under
the quantizer a checkpoint was trained for.

Two properties carry the Phase B video refit. A percentile checkpoint (run4, the
deployed transform) must get exactly the calibration it always got, or every
existing refit is silently re-fitted on a different grid. A unit-step checkpoint
must get params that dequantize to `round(x / step) * step` in both params
blocks, at the one alphabet width the `.nvct` header can carry.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import torch

from nvc.compression.quantization import UniformQuantizer
from nvc.evaluation.sequences import BenchmarkSequence
from nvc.models.autoencoder import BaselineAutoencoder

from helpers import make_sequence


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TINY = {"in_channels": 3, "latent_channels": 4, "base_channels": 8}
CALIBRATION = dict(block_size=16, search_range=4, gop_size=2)


def _model(seed: int = 0) -> BaselineAutoencoder:
    torch.manual_seed(seed)
    model = BaselineAutoencoder(**TINY)
    model.eval()
    return model


def _sequences(tmp_path, *, count=2, frames_per_sequence=4, size=64):
    sequences = []
    for index in range(count):
        directory = make_sequence(
            tmp_path / f"seq{index}", num_frames=frames_per_sequence, width=size, height=size)
        sequences.append(BenchmarkSequence(
            dataset="synthetic", sequence_id=f"seq{index}", split="train",
            frame_paths=tuple(sorted(directory.iterdir())), width=size, height=size))
    return sequences


def _checkpoint(tmp_path, name: str, extra: dict | None) -> Path:
    path = tmp_path / f"{name}.pt"
    payload = {"model_state_dict": {}}
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)
    return path


def _unit_step_width(grids, mc, model, sequences, checkpoint) -> int:
    """The stream width the wrapper fits, read off its own refusal."""
    with pytest.raises(ValueError, match="Pass that as the rate point") as refused:
        grids.calibrate_grids(mc, model, sequences, checkpoint=checkpoint, bits=10,
                              **CALIBRATION)
    return int(re.search(r"stream is (\d+) bits wide", str(refused.value)).group(1))


def test_checkpoint_quantizer_reads_the_training_record(tmp_path):
    grids = _load_script("unit_step_grids")
    assert grids.checkpoint_quantizer(_checkpoint(tmp_path, "run4", None)) == {
        "quantizer": "percentile"}
    assert grids.checkpoint_quantizer(_checkpoint(tmp_path, "qat", {"qat_bits": 4})) == {
        "quantizer": "percentile"}
    assert grids.checkpoint_quantizer(_checkpoint(
        tmp_path, "phase_b", {"quantizer": "unit_step", "unit_step": 0.5})) == {
        "quantizer": "unit_step", "step": 0.5}


def test_percentile_checkpoint_gets_exactly_the_old_calibration(tmp_path):
    grids = _load_script("unit_step_grids")
    mc = _load_script("m10h_motion_compensation")
    model, sequences = _model(), _sequences(tmp_path)
    checkpoint = _checkpoint(tmp_path, "run4", None)

    ours = grids.calibrate_grids(mc, model, sequences, checkpoint=checkpoint, bits=4,
                                 **CALIBRATION)
    theirs = mc.calibrate_grids(model, sequences, bits=4, **CALIBRATION)

    for block in ("intra_params", "residual_params"):
        assert torch.equal(ours[block].scale, theirs[block].scale)
        assert torch.equal(ours[block].zero_point, theirs[block].zero_point)
        assert ours[block].bits == theirs[block].bits == 4
    for table in ("intra_entropy_model", "residual_entropy_model", "motion_entropy_model"):
        assert ours[table].model_id() == theirs[table].model_id()
    assert ours["provenance"] == theirs["provenance"]
    assert grids.cache_identity(checkpoint) == {}


def test_unit_step_params_round_to_the_trained_step_at_one_shared_width(tmp_path):
    grids = _load_script("unit_step_grids")
    mc = _load_script("m10h_motion_compensation")
    model, sequences = _model(), _sequences(tmp_path)
    step = 0.05
    checkpoint = _checkpoint(tmp_path, "phase_b", {"quantizer": "unit_step", "unit_step": step})
    width = _unit_step_width(grids, mc, model, sequences, checkpoint)

    calibration = grids.calibrate_grids(mc, model, sequences, checkpoint=checkpoint,
                                        bits=width, **CALIBRATION)

    provenance = calibration["provenance"]
    assert provenance["quantizer"] == "unit_step" and provenance["unit_step"] == step
    assert max(provenance["intra_bits_needed"], provenance["residual_bits_needed"]) == width
    for block in ("intra_params", "residual_params"):
        params = calibration[block]
        assert params.bits == width and params.mode == "per_channel"
        assert torch.all(params.scale == step)
    for table in ("intra_entropy_model", "residual_entropy_model"):
        assert calibration[table].bits == width

    # the intra block dequantizes to exactly round(x / step) * step on TRAIN latents
    with torch.no_grad():
        latent = model.encode(sequences[0].load_frames()[:1])
    params = calibration["intra_params"]
    quantizer = UniformQuantizer(params.bits, params.mode)
    symbols, _ = quantizer.quantize(latent, params)
    restored = quantizer.dequantize(symbols, params)
    assert torch.equal(restored, torch.round(latent / step) * step)

    assert grids.cache_identity(checkpoint)["unit_step_grids_source"]


def test_unit_step_refuses_a_rate_point_that_is_not_its_width(tmp_path):
    grids = _load_script("unit_step_grids")
    mc = _load_script("m10h_motion_compensation")
    model, sequences = _model(), _sequences(tmp_path)
    checkpoint = _checkpoint(tmp_path, "phase_b", {"quantizer": "unit_step", "unit_step": 0.05})
    width = _unit_step_width(grids, mc, model, sequences, checkpoint)

    with pytest.raises(ValueError, match="needs a wider alphabet"):
        grids.calibrate_grids(mc, model, sequences, checkpoint=checkpoint, bits=width - 1,
                              **CALIBRATION)


def test_the_percentile_primitive_is_restored_even_after_a_refusal(tmp_path):
    grids = _load_script("unit_step_grids")
    mc = _load_script("m10h_motion_compensation")
    original = mc.calibrate_quantization_params
    checkpoint = _checkpoint(tmp_path, "phase_b", {"quantizer": "unit_step", "unit_step": 0.05})

    _unit_step_width(grids, mc, _model(), _sequences(tmp_path), checkpoint)

    assert mc.calibrate_quantization_params is original


def test_unit_step_grids_leaves_the_m22_identity_files_alone():
    """`m10h_motion_compensation.py` is in `m22_residual.code_identity()`; the
    wrapper exists so the unit step never needs an edit there."""
    source = Path("scripts/m10h_motion_compensation.py").read_text(encoding="utf-8")
    assert "unit_step" not in source
