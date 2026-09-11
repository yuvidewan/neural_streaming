"""M14 - bitstream compatibility for the motion-table recalibration through
the REAL `.nvct` v2 container, mirroring test_m13_closed_loop.py's rig.

Only "motion" is exercised here (Phase B rejected "intra" as weak before it
ever reached coded validation, per the milestone's own pre-declared gate -
see test_m14_recalibration.py's `test_intra_gate_result_was_rejected_...`
for the recorded verdict) - so there is exactly one new identity/config to
prove compatible, not several.
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
BITS = 4


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


def _calibration(mc, model, frames, *, bits=BITS, search_range=8, block_size=16):
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
def m14():
    return _load_script("m14_recalibration")


@pytest.fixture(scope="module")
def cl14():
    return _load_script("m14_closed_loop")


ZERO = torch.full((4,), 8)


def _residual_arm(mc, mk, ml, ma, m13, calibration, *, bits=BITS, seed=0):
    torch.manual_seed(seed + 3)
    m10k = mk.build_model({"latent_channels": 4, "alphabet": 2 ** bits, "hidden": 8}).eval()
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
        train_symbols.reshape(-1), train_k.reshape(-1), alphabet=2 ** bits)
    coding_codebook = m13.build_recalibrated_codebook(assign_codebook, frequencies)
    identity = ma.model_identity(m11_model, m10k_identity=b"\x9a" * 8,
                                 calibration_signature="test", bits=bits, codebook=coding_codebook)
    return {"m13_recal": {"model": m11_model, "zero": ZERO, "assign_codebook": assign_codebook,
                          "coding_codebook": coding_codebook, "identity": identity}}


def _rig(mc, mk, ml, ma, m13, m14, *, frames=12, seed=0):
    model = _autoencoder(seed)
    clip = _moving_frames(frames, seed=seed)
    calibration = _calibration(mc, model, _moving_frames(12, seed=seed), bits=BITS)
    arms = _residual_arm(mc, mk, ml, ma, m13, calibration, seed=seed)

    # a broader, synthetic "TRAIN" motion sample - a handful of independent
    # short clips, standing in for m14's per-sequence-capped TRAIN collection.
    class _FakeSequence:
        def __init__(self, tensor):
            self._tensor = tensor
            self.sequence_id = "fake"
            self.frame_count = tensor.shape[0]

        def load_frames(self):
            return self._tensor

    train_sequences = [_FakeSequence(_moving_frames(6, seed=seed + i)) for i in range(3)]
    train_motion = m14.collect_motion_symbols(mc, model, train_sequences, block_size=16,
                                              search_range=8, gop_size=100, max_frames=10 ** 9,
                                              reference_mode="mc", device=torch.device("cpu"))
    new_motion = m14.fit_empirical(train_motion, bits=mc.motion_alphabet_bits(8), num_tables=2)
    return model, clip, calibration, arms, new_motion


def _encode_decode(mc, ma, m13, model, frames, arms, tmp_path, calibration, *, motion_model,
                   gop=4, path_name="stream.nvct"):
    """Round-trips ONE clip through `encode_multi`/`decode_sequence` directly
    - like test_m13_closed_loop.py's own `_encode`/`_decode` helpers - rather
    than `run_sequences_for_arms`, which also computes MS-SSIM (needs >=161px
    frames; this rig's tiny 64px synthetic clips are for speed, not for
    exercising the perceptual-metric path)."""
    cl = _load_script("m13_closed_loop")
    paths = {"m13_recal": tmp_path / path_name}
    result = cl.encode_multi(mc, ma, m13, model, frames, arms, paths,
                             intra_params=calibration["intra_params"],
                             intra_entropy_model=calibration["intra_entropy_model"],
                             residual_params=calibration["residual_params"],
                             motion_entropy_model=motion_model, bits=BITS, gop_size=gop,
                             block_size=16, search_range=8)
    decoded, symbols, timings = cl.decode_sequence(
        mc, ma, m13, model, paths["m13_recal"], "m13_recal", arms["m13_recal"],
        intra_entropy_model=calibration["intra_entropy_model"], motion_entropy_model=motion_model,
        bits=BITS)
    return result, decoded, symbols, timings


# --- 9/11/12: identity distinctness, old/new round trips ------------------------------


def test_baseline_and_recalibrated_motion_have_distinct_identities(mc, mk, ml, ma, m13, m14, cl14):
    model, frames, calibration, arms, new_motion = _rig(mc, mk, ml, ma, m13, m14)
    assert new_motion.model_id() != calibration["motion_entropy_model"].model_id()
    assert len(new_motion.model_id()) == len(calibration["motion_entropy_model"].model_id()) == 8


def test_baseline_stream_round_trips_unchanged(mc, mk, ml, ma, m13, m14, cl14, tmp_path):
    model, frames, calibration, arms, new_motion = _rig(mc, mk, ml, ma, m13, m14)
    result, decoded, symbols, _ = _encode_decode(
        mc, ma, m13, model, frames, arms, tmp_path, calibration,
        motion_model=calibration["motion_entropy_model"])
    assert torch.equal(decoded.cpu(), result["reconstructions"])
    for coded, back in zip(result["symbols"], symbols):
        assert np.array_equal(coded.reshape(-1), back.reshape(-1))


def test_recalibrated_motion_stream_round_trips(mc, mk, ml, ma, m13, m14, cl14, tmp_path):
    model, frames, calibration, arms, new_motion = _rig(mc, mk, ml, ma, m13, m14)
    result, decoded, symbols, timings = _encode_decode(
        mc, ma, m13, model, frames, arms, tmp_path, calibration, motion_model=new_motion)
    assert torch.equal(decoded.cpu(), result["reconstructions"])
    for coded, back in zip(result["symbols"], symbols):
        assert np.array_equal(coded.reshape(-1), back.reshape(-1))
    assert set(timings) == {"network", "tables", "coder"}


def test_baseline_and_recalibrated_produce_identical_symbols_and_reconstruction(
        mc, mk, ml, ma, m13, m14, cl14, tmp_path):
    model, frames, calibration, arms, new_motion = _rig(mc, mk, ml, ma, m13, m14)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    base_result, base_decoded, base_symbols, _ = _encode_decode(
        mc, ma, m13, model, frames, arms, tmp_path / "a", calibration,
        motion_model=calibration["motion_entropy_model"])
    new_result, new_decoded, new_symbols, _ = _encode_decode(
        mc, ma, m13, model, frames, arms, tmp_path / "b", calibration, motion_model=new_motion)

    # Residual symbols and the final reconstruction are UNCHANGED - only
    # which motion table the coder used differs.
    for a, b in zip(base_symbols, new_symbols):
        assert np.array_equal(a.reshape(-1), b.reshape(-1))
    assert torch.equal(base_decoded.cpu(), new_decoded.cpu())
    assert base_result["arms"]["m13_recal"]["i_frame_residual_bytes"] \
        == new_result["arms"]["m13_recal"]["i_frame_residual_bytes"]
    assert base_result["arms"]["m13_recal"]["p_frame_residual_bytes"] \
        == new_result["arms"]["m13_recal"]["p_frame_residual_bytes"]


# --- 10: provenance mismatch rejection -------------------------------------------------


def test_decoding_with_the_wrong_motion_table_is_now_rejected(mc, mk, ml, ma, m13, m14, cl14,
                                                               tmp_path):
    # A real gap found DURING M14 (not present in M13, which never varied
    # motion_entropy_model across calls): scripts/m13_closed_loop.decode_sequence
    # used to check only residual_entropy_model_id, never
    # intra_/motion_entropy_model_id, so decoding a recalibrated-motion
    # stream with the DEPLOYED table (or vice versa) would have silently
    # produced a corrupted reconstruction instead of an error. Fixed in
    # scripts/m13_closed_loop.py's decode_sequence (all three .nvct v2
    # identities are now checked); this test proves the fix, not just
    # documents the gap.
    cl = _load_script("m13_closed_loop")
    model, frames, calibration, arms, new_motion = _rig(mc, mk, ml, ma, m13, m14)
    paths = {"m13_recal": tmp_path / "recalibrated_motion.nvct"}
    cl.encode_multi(mc, ma, m13, model, frames, arms, paths,
                    intra_params=calibration["intra_params"],
                    intra_entropy_model=calibration["intra_entropy_model"],
                    residual_params=calibration["residual_params"],
                    motion_entropy_model=new_motion, bits=BITS, gop_size=4, block_size=16,
                    search_range=8)

    with pytest.raises(mc.TemporalFormatError, match="motion entropy model mismatch"):
        cl.decode_sequence(mc, ma, m13, model, paths["m13_recal"], "m13_recal", arms["m13_recal"],
                           intra_entropy_model=calibration["intra_entropy_model"],
                           motion_entropy_model=calibration["motion_entropy_model"],  # WRONG table
                           bits=BITS)

    # And decoding with the CORRECT (matching) table still works.
    decoded, _, _ = cl.decode_sequence(mc, ma, m13, model, paths["m13_recal"], "m13_recal",
                                       arms["m13_recal"],
                                       intra_entropy_model=calibration["intra_entropy_model"],
                                       motion_entropy_model=new_motion, bits=BITS)
    assert decoded.shape[0] == frames.shape[0]


def test_motion_entropy_model_id_is_one_of_the_three_nvct_v2_identity_fields():
    source = Path("scripts/m10h_motion_compensation.py").read_text(encoding="utf-8")
    assert "motion_entropy_model_id" in source
    assert "residual_entropy_model_id" in source
    assert "intra_entropy_model_id" in source
