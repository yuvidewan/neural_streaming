"""M13 Phases F/H - bitstream compatibility for the recalibrated arm through
the REAL `.nvct` v2 container (not just the single-frame functions
`test_m13_recalibration.py` already covers).

Mirrors `test_m11_deployment.py`'s TINY closed-loop setup - a real
`TemporalStreamWriter`/`TemporalStreamReader` round trip, real motion
estimation, real GOP handling - but drives it through
`scripts/m13_closed_loop.py` with both the unchanged `m11_op` arm and the
new `m13_recal` arm, since those are what M13 actually adds.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.codec import latent_to_symbols
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.models.autoencoder import BaselineAutoencoder


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TINY = {"in_channels": 3, "latent_channels": 4, "base_channels": 8}


def _autoencoder(seed: int = 0):
    torch.manual_seed(seed)
    model = BaselineAutoencoder(**TINY).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _moving_frames(count: int = 12, size: int = 64, *, seed: int = 0, step: int = 2):
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 3, size, size, generator=generator)
    return torch.cat([torch.roll(base, shifts=(i * step, i * step), dims=(2, 3))
                      for i in range(count)], dim=0)


def _calibration(mc, model, frames, *, bits=4, search_range=8, block_size=16):
    with torch.no_grad():
        latents = torch.cat([model.encode(frames[i:i + 1]) for i in range(frames.shape[0])])
    intra_params = calibrate_quantization_params(latents, bits=bits, mode="per_channel")
    intra_symbols = np.stack([
        latent_to_symbols(latents[i:i + 1], intra_params).reshape(latents.shape[1], -1)
        for i in range(latents.shape[0])])
    intra_model = EmpiricalEntropyModel.from_symbols(
        intra_symbols, bits=bits, num_tables=latents.shape[1])
    residuals = latents[1:] - latents[:-1]
    residual_params = calibrate_quantization_params(residuals, bits=bits, mode="per_channel")
    blocks = (frames.shape[2] // block_size) * (frames.shape[3] // block_size)
    motion_model = EmpiricalEntropyModel.from_symbols(
        np.stack([np.stack([np.arange(blocks) % (2 * search_range + 1),
                            np.arange(blocks) % (2 * search_range + 1)])]),
        bits=mc.motion_alphabet_bits(search_range), num_tables=2)
    references = [latents[i].numpy() for i in range(1, latents.shape[0])]
    symbols = [latent_to_symbols(latents[i:i + 1] - latents[i - 1:i],
                                 residual_params).reshape(latents.shape[1:])
               for i in range(1, latents.shape[0])]
    return {"intra_params": intra_params, "intra_entropy_model": intra_model,
            "residual_params": residual_params, "motion_entropy_model": motion_model,
            "references": references, "symbols": symbols}


@pytest.fixture(scope="module")
def mc():
    return _load_script("m10h_motion_compensation")


@pytest.fixture(scope="module")
def ma():
    return _load_script("m11_ar_entropy")


@pytest.fixture(scope="module")
def mk():
    return _load_script("m10k_learned_entropy")


@pytest.fixture(scope="module")
def ml():
    return _load_script("m10l_shared_codebook")


@pytest.fixture(scope="module")
def m13():
    return _load_script("m13_recalibration")


@pytest.fixture(scope="module")
def cl():
    return _load_script("m13_closed_loop")


CHANNELS, BITS = 4, 4
ZERO = torch.full((CHANNELS,), 8)


def _rig(mc, mk, ml, ma, m13, *, bits=BITS, frames=12, seed=0):
    cl = _load_script("m13_closed_loop")
    model = _autoencoder(seed)
    clip = _moving_frames(frames, seed=seed)
    # Calibration always comes from a fixed, longer clip - independent of how
    # short `clip` (the sequence actually under test) is, exactly like
    # test_m11_deployment.py's `_setup` - a 1-frame test clip has no residuals
    # of its own to calibrate a residual quantizer from.
    calibration = _calibration(mc, model, _moving_frames(12, seed=seed), bits=bits)

    torch.manual_seed(3)
    m10k = mk.build_model({"latent_channels": CHANNELS, "alphabet": 2 ** bits,
                           "hidden": 8}).eval()
    samples = ml.sample_training_distributions(
        m10k, calibration["references"], device=torch.device("cpu"), max_rows=2000)
    assign_codebook = ml.fit_codebook(samples, 16, bits=bits)

    m11_model = ma.from_m10k(m10k, group_size=2).eval()
    with torch.no_grad():
        m11_model.features[0].weight[:, 1:] = 0.4

    train_symbols = np.stack([s.reshape(-1) for s in calibration["symbols"]])
    train_k = m13.m11_g16_prototype_indices(
        m11_model, assign_codebook, np.stack(calibration["symbols"]),
        np.stack(calibration["references"]), ZERO, device=torch.device("cpu"))
    frequencies, strength, _, _ = m13.fit_recalibrated_frequencies(
        assign_codebook, train_symbols.reshape(-1), train_k.reshape(-1),
        train_symbols.reshape(-1), train_k.reshape(-1),  # VAL-A == TRAIN here: rig is tiny/synthetic
        alphabet=2 ** bits)
    coding_codebook = m13.build_recalibrated_codebook(assign_codebook, frequencies)

    arms = cl.build_arms(ma, m13, m11_model, assign_codebook, coding_codebook, ZERO,
                         m10k_identity=b"\x9a" * 8, calibration_signature="test", bits=bits)
    return model, clip, calibration, arms


def _encode(mc, ma, m13, model, frames, arms, tmp_path, calibration, *, bits=BITS, gop_size=4):
    cl = _load_script("m13_closed_loop")
    paths = {arm: tmp_path / f"{arm}.nvct" for arm in arms}
    result = cl.encode_multi(
        mc, ma, m13, model, frames, arms, paths,
        intra_params=calibration["intra_params"],
        intra_entropy_model=calibration["intra_entropy_model"],
        residual_params=calibration["residual_params"],
        motion_entropy_model=calibration["motion_entropy_model"],
        bits=bits, gop_size=gop_size, block_size=16, search_range=8)
    return result, paths


def _decode(mc, ma, m13, model, path, arm, spec, calibration, *, bits=BITS):
    cl = _load_script("m13_closed_loop")
    return cl.decode_sequence(mc, ma, m13, model, path, arm, spec,
                              intra_entropy_model=calibration["intra_entropy_model"],
                              motion_entropy_model=calibration["motion_entropy_model"], bits=bits)


# --- old-stream / new-stream round trips ------------------------------------------------


def test_old_arm_m11_op_round_trips_unchanged(mc, ma, mk, ml, m13, cl, tmp_path):
    model, frames, calibration, arms = _rig(mc, mk, ml, ma, m13)
    result, paths = _encode(mc, ma, m13, model, frames, arms, tmp_path, calibration)
    decoded, symbols, _ = _decode(mc, ma, m13, model, paths["m11_op"], "m11_op",
                                  arms["m11_op"], calibration)

    for coded, back in zip(result["symbols"], symbols):
        assert np.array_equal(coded.reshape(-1), back.reshape(-1))
    assert torch.equal(decoded.cpu(), result["reconstructions"])


def test_new_arm_m13_recal_round_trips(mc, ma, mk, ml, m13, cl, tmp_path):
    model, frames, calibration, arms = _rig(mc, mk, ml, ma, m13)
    result, paths = _encode(mc, ma, m13, model, frames, arms, tmp_path, calibration)
    decoded, symbols, timings = _decode(mc, ma, m13, model, paths["m13_recal"], "m13_recal",
                                        arms["m13_recal"], calibration)

    for coded, back in zip(result["symbols"], symbols):
        assert np.array_equal(coded.reshape(-1), back.reshape(-1))
    assert torch.equal(decoded.cpu(), result["reconstructions"])
    assert set(timings) == {"network", "tables", "coder"}


def test_m11_op_and_m13_recal_decode_to_identical_symbols_and_reconstruction(
        mc, ma, mk, ml, m13, cl, tmp_path):
    model, frames, calibration, arms = _rig(mc, mk, ml, ma, m13)
    result, paths = _encode(mc, ma, m13, model, frames, arms, tmp_path, calibration)
    decoded_old, symbols_old, _ = _decode(mc, ma, m13, model, paths["m11_op"], "m11_op",
                                          arms["m11_op"], calibration)
    decoded_new, symbols_new, _ = _decode(mc, ma, m13, model, paths["m13_recal"], "m13_recal",
                                          arms["m13_recal"], calibration)

    for a, b in zip(symbols_old, symbols_new):
        assert np.array_equal(a.reshape(-1), b.reshape(-1))
    assert torch.equal(decoded_old.cpu(), decoded_new.cpu())
    # Only the residual byte count may differ.
    assert result["arms"]["m11_op"]["motion_bytes"] == result["arms"]["m13_recal"]["motion_bytes"]
    assert result["arms"]["m11_op"]["i_frame_residual_bytes"] \
        == result["arms"]["m13_recal"]["i_frame_residual_bytes"]


# --- distinct identities, provenance rejection --------------------------------------------


def test_m11_op_and_m13_recal_have_distinct_stream_identities(mc, ma, mk, ml, m13, cl, tmp_path):
    model, frames, calibration, arms = _rig(mc, mk, ml, ma, m13)
    assert arms["m11_op"]["identity"] != arms["m13_recal"]["identity"]
    assert len(arms["m11_op"]["identity"]) == len(arms["m13_recal"]["identity"]) == 8


def test_decoding_a_recalibrated_stream_with_the_old_spec_is_rejected(mc, ma, mk, ml, m13, cl,
                                                                       tmp_path):
    model, frames, calibration, arms = _rig(mc, mk, ml, ma, m13)
    _, paths = _encode(mc, ma, m13, model, frames, arms, tmp_path, calibration)

    mismatched_spec = dict(arms["m13_recal"], identity=arms["m11_op"]["identity"])
    with pytest.raises(Exception, match="entropy model mismatch") as raised:
        _decode(mc, ma, m13, model, paths["m13_recal"], "m13_recal", mismatched_spec, calibration)
    assert type(raised.value).__name__ == "TemporalFormatError"


def test_decoding_an_old_stream_with_the_recalibrated_spec_is_rejected(mc, ma, mk, ml, m13, cl,
                                                                        tmp_path):
    model, frames, calibration, arms = _rig(mc, mk, ml, ma, m13)
    _, paths = _encode(mc, ma, m13, model, frames, arms, tmp_path, calibration)

    mismatched_spec = dict(arms["m11_op"], identity=arms["m13_recal"]["identity"])
    with pytest.raises(Exception, match="entropy model mismatch") as raised:
        _decode(mc, ma, m13, model, paths["m11_op"], "m11_op", mismatched_spec, calibration)
    assert type(raised.value).__name__ == "TemporalFormatError"


# --- .nvct v2 unchanged -------------------------------------------------------------------


def test_the_container_format_is_still_nvct_v2(mc, ma, mk, ml, m13, cl, tmp_path):
    model, frames, calibration, arms = _rig(mc, mk, ml, ma, m13)
    _, paths = _encode(mc, ma, m13, model, frames, arms, tmp_path, calibration)

    for arm in ("m11_op", "m13_recal"):
        header = mc.TemporalStreamReader(paths[arm]).header
        raw = paths[arm].read_bytes()
        assert raw[:4] == mc.TEMPORAL_MAGIC
        assert header.format_version == mc.TEMPORAL_FORMAT_VERSION == 2


def test_only_the_residual_entropy_model_id_differs_between_headers(mc, ma, mk, ml, m13, cl,
                                                                     tmp_path):
    model, frames, calibration, arms = _rig(mc, mk, ml, ma, m13)
    _, paths = _encode(mc, ma, m13, model, frames, arms, tmp_path, calibration)
    old_header = mc.TemporalStreamReader(paths["m11_op"]).header
    new_header = mc.TemporalStreamReader(paths["m13_recal"]).header

    assert old_header.residual_entropy_model_id != new_header.residual_entropy_model_id
    for field in ("format_version", "gop_size", "quantization_bits", "quantization_mode",
                  "image_width", "image_height", "image_channels", "latent_channels",
                  "latent_height", "latent_width", "frame_count", "block_size", "search_range",
                  "motion_bits", "intra_entropy_model_id", "motion_entropy_model_id"):
        assert getattr(old_header, field) == getattr(new_header, field), field


# --- short sequences / GOP boundaries, resumable-decoder compatibility --------------------


@pytest.mark.parametrize("frame_count,gop", [(1, 4), (2, 4), (5, 4), (9, 4)])
def test_short_sequences_and_gop_boundaries(mc, ma, mk, ml, m13, cl, tmp_path, frame_count, gop):
    model, clip, calibration, arms = _rig(mc, mk, ml, ma, m13, frames=frame_count)
    result, paths = _encode(mc, ma, m13, model, clip, arms, tmp_path, calibration, gop_size=gop)
    decoded, symbols, _ = _decode(mc, ma, m13, model, paths["m13_recal"], "m13_recal",
                                  arms["m13_recal"], calibration)

    assert len(symbols) == len(result["symbols"])
    assert torch.equal(decoded.cpu(), result["reconstructions"])
    assert decoded.shape[0] == frame_count


def test_repeated_encodes_of_the_recalibrated_arm_are_byte_identical(mc, ma, mk, ml, m13, cl,
                                                                      tmp_path):
    model, frames, calibration, arms = _rig(mc, mk, ml, ma, m13)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, first = _encode(mc, ma, m13, model, frames, arms, tmp_path / "a", calibration)
    _, second = _encode(mc, ma, m13, model, frames, arms, tmp_path / "b", calibration)
    assert first["m13_recal"].read_bytes() == second["m13_recal"].read_bytes()
