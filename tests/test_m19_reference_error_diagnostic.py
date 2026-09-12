"""M19 - tests for the reference-error shape diagnostic. Reuses the
established tiny-rig pattern from test_m13-m18's own test files.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.codec import latent_to_symbols
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.data.image_io import write_tensor_as_image
from nvc.evaluation.sequences import BenchmarkSequence
from nvc.models.autoencoder import BaselineAutoencoder


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
def m19():
    return _load_script("m19_reference_error_diagnostic")


TINY = {"in_channels": 3, "latent_channels": 4, "base_channels": 8}
BITS = 4
GOP = 4


def _autoencoder(seed: int = 0):
    torch.manual_seed(seed)
    model = BaselineAutoencoder(**TINY).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _write_sequence(tmp_path: Path, name: str, count: int, *, size: int = 64,
                    seed: int = 0, step: int = 2) -> BenchmarkSequence:
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand(3, size, size, generator=generator)
    directory = tmp_path / name
    directory.mkdir()
    paths = []
    for i in range(count):
        frame = torch.roll(base, shifts=(i * step, i * step), dims=(1, 2))
        path = directory / f"frame_{i:04d}.png"
        write_tensor_as_image(frame, path)
        paths.append(path)
    return BenchmarkSequence(dataset="fake", sequence_id=name, split="train",
                             frame_paths=tuple(paths), width=size, height=size)


def _calibration(model, frames, *, bits=BITS):
    with torch.no_grad():
        latents = torch.cat([model.encode(frames[i:i + 1]) for i in range(frames.shape[0])])
    intra_params = calibrate_quantization_params(latents, bits=bits, mode="per_channel")
    intra_symbols = np.stack([
        latent_to_symbols(latents[i:i + 1], intra_params).reshape(latents.shape[1], -1)
        for i in range(latents.shape[0])])
    intra_entropy_model = EmpiricalEntropyModel.from_symbols(
        intra_symbols, bits=bits, num_tables=latents.shape[1])
    residuals = latents[1:] - latents[:-1]
    residual_params = calibrate_quantization_params(residuals, bits=bits, mode="per_channel")
    return intra_params, intra_entropy_model, residual_params


def _residual_arm(mc, mk, ml, ma, m13, model, frames, *, bits=BITS, seed=0):
    intra_params, intra_entropy_model, residual_params = _calibration(model, frames, bits=bits)
    with torch.no_grad():
        latents = torch.cat([model.encode(frames[i:i + 1]) for i in range(frames.shape[0])])
    references = [latents[i].numpy() for i in range(1, latents.shape[0])]
    symbols_for_k = [latent_to_symbols(latents[i:i + 1] - latents[i - 1:i], residual_params)
                     .reshape(latents.shape[1:]) for i in range(1, latents.shape[0])]

    torch.manual_seed(seed + 3)
    m10k = mk.build_model({"latent_channels": 4, "alphabet": 2 ** bits, "hidden": 8}).eval()
    samples = ml.sample_training_distributions(m10k, references, device=torch.device("cpu"),
                                                max_rows=2000)
    assign_codebook = ml.fit_codebook(samples, 16, bits=bits)
    model11 = ma.from_m10k(m10k, group_size=2).eval()
    with torch.no_grad():
        model11.features[0].weight[:, 1:] = 0.4
    zero = torch.full((4,), 2 ** (bits - 1))

    train_symbols = np.stack([s.reshape(-1) for s in symbols_for_k])
    gate = _load_script("m12_spatial_offline_gate")
    train_k = gate.m11_g16_prototype_indices(model11, assign_codebook, np.stack(symbols_for_k),
                                             np.stack(references), zero, device=torch.device("cpu"))
    frequencies, strength, _, _ = m13.fit_recalibrated_frequencies(
        assign_codebook, train_symbols.reshape(-1), train_k.reshape(-1),
        train_symbols.reshape(-1), train_k.reshape(-1), alphabet=2 ** bits)
    coding_codebook = m13.build_recalibrated_codebook(assign_codebook, frequencies)
    return (model11, assign_codebook, coding_codebook, zero, intra_params, intra_entropy_model,
           residual_params)


@pytest.fixture
def rig(tmp_path, mc, mk, ml, ma, m13):
    model = _autoencoder(seed=1)
    calib_frames = torch.cat([
        _write_sequence(tmp_path, f"calib{i}", 8, seed=10 + i).load_frames() for i in range(3)])
    arm = _residual_arm(mc, mk, ml, ma, m13, model, calib_frames)
    sequence = _write_sequence(tmp_path, "diag", 13, seed=1)
    return model, sequence, arm


def _run(mc, ma, m13, mk, m19, model, sequence, arm, seed=42):
    model11, assign_codebook, coding_codebook, zero, intra_params, intra_entropy_model, residual_params = arm
    return m19.diagnose_reference_error(
        mc, ma, m13, mk, model, model11, assign_codebook, coding_codebook, sequence, bits=BITS,
        intra_params=intra_params, intra_entropy_model=intra_entropy_model,
        residual_params=residual_params, zero=zero, gop_size=GOP, block_size=16,
        search_range=16, device=torch.device("cpu"), seed=seed)


# --- 1/2/3/4: exact real/oracle reconstruction, latent/pixel error computation ---------


def test_real_chain_matches_encode_multi_exactly(mc, ma, m13, mk, m19, rig):
    cl = _load_script("m13_closed_loop")
    model, sequence, (model11, assign_codebook, coding_codebook, zero, intra_params,
                      intra_entropy_model, residual_params) = rig
    frames = sequence.load_frames()

    blocks = (frames.shape[2] // 16) * (frames.shape[3] // 16)
    motion_model = EmpiricalEntropyModel.from_symbols(
        np.stack([np.stack([np.arange(blocks) % 33, np.arange(blocks) % 33])]),
        bits=mc.motion_alphabet_bits(16), num_tables=2)
    identity = ma.model_identity(model11, m10k_identity=b"\x9a" * 8,
                                 calibration_signature="test", bits=BITS, codebook=coding_codebook)
    arms = {"m13_recal": {"model": model11, "zero": zero, "assign_codebook": assign_codebook,
                          "coding_codebook": coding_codebook, "identity": identity}}
    paths = {"m13_recal": Path(sequence.frame_paths[0]).parent / "scratch.nvct"}
    result = cl.encode_multi(mc, ma, m13, model, frames, arms, paths,
                             intra_params=intra_params, intra_entropy_model=intra_entropy_model,
                             residual_params=residual_params, motion_entropy_model=motion_model,
                             bits=BITS, gop_size=GOP, block_size=16, search_range=16)
    paths["m13_recal"].unlink(missing_ok=True)

    rows = _run(mc, ma, m13, mk, m19, model, sequence, rig[2])
    assert sum(r["real_bytes"] for r in rows) == result["arms"]["m13_recal"]["p_frame_residual_bytes"]


def test_pixel_and_latent_error_are_never_conflated(mc, ma, m13, mk, m19, rig):
    model, sequence, arm = rig
    rows = _run(mc, ma, m13, mk, m19, model, sequence, arm)
    assert len(rows) > 0
    for r in rows:
        assert set(r["pixel_error"]) == {"mean", "mae", "rms", "variance"}
        assert set(r["latent_error"]) == {"mean", "mae", "rms", "variance"}
        # pixel and latent error live in DIFFERENT domains (different tensor
        # sizes/scales) - their RMS values must not be required (or expected)
        # to match; this test only pins that both are computed and populated.
        assert r["pixel_error"]["rms"] >= 0.0
        assert r["latent_error"]["rms"] >= 0.0


# --- error_magnitude_per_symbol alignment (the bug caught and fixed pre-run) -----------


def test_error_magnitude_per_symbol_is_channel_specific_not_averaged(mc, ma, m13, mk, m19, rig):
    model, sequence, arm = rig
    rows = _run(mc, ma, m13, mk, m19, model, sequence, arm)
    channels = 4
    for r in rows:
        mags = np.asarray(r["error_magnitude_per_symbol"])
        churn = np.asarray(r["assignment_changed_per_symbol"])
        assert mags.shape == churn.shape
        per_channel = mags.reshape(channels, -1)
        # A real per-(channel,position) magnitude varies across channels for
        # the SAME spatial position - a channel-averaged-then-tiled bug would
        # make every channel's row identical. Assert they are NOT all equal
        # (this rig's synthetic per-channel latent scales differ, so a
        # genuine per-channel signal must show up).
        assert not np.allclose(per_channel[0], per_channel[1]) or channels == 1


# --- 5/6: deterministic channel/spatial analysis ----------------------------------------


def test_channel_arrays_have_consistent_shape_and_are_deterministic(mc, ma, m13, mk, m19, rig):
    model, sequence, arm = rig
    rows_a = _run(mc, ma, m13, mk, m19, model, sequence, arm)
    rows_b = _run(mc, ma, m13, mk, m19, model, sequence, arm)
    for a, b in zip(rows_a, rows_b):
        assert np.array_equal(a["channel_mae"], b["channel_mae"])
        assert np.array_equal(a["channel_churn"], b["channel_churn"])
        assert len(a["channel_mae"]) == 4  # TINY latent_channels
        assert 0.0 <= a["assignment_changed_fraction"] <= 1.0


def test_spatial_metrics_are_finite_and_bounded(mc, ma, m13, mk, m19, rig):
    model, sequence, arm = rig
    rows = _run(mc, ma, m13, mk, m19, model, sequence, arm)
    for r in rows:
        assert -1.0 <= r["autocorr_h"] <= 1.0 + 1e-6
        assert -1.0 <= r["autocorr_v"] <= 1.0 + 1e-6
        assert r["low_freq_energy"] >= 0.0 and r["high_freq_energy"] >= 0.0
        assert set(r["spatial_region_error"]) == {"flat", "texture", "edge"}
        assert r["boundary_error"] >= 0.0 and r["interior_error"] >= 0.0


# --- 7/8: codebook assignment accounting, G16 code-length accounting -------------------


def test_codebook_and_code_length_accounting_are_self_consistent(mc, ma, m13, mk, m19, rig):
    model, sequence, arm = rig
    rows = _run(mc, ma, m13, mk, m19, model, sequence, arm)
    for r in rows:
        delta = np.asarray(r["delta_code_len_per_symbol"])
        real_len = np.asarray(r["code_len_real_per_symbol"])
        oracle_len = np.asarray(r["code_len_oracle_per_symbol"])
        assert np.allclose(delta, real_len - oracle_len)
        assert (real_len >= 0).all() and (oracle_len >= 0).all()
        churn = np.asarray(r["assignment_changed_per_symbol"])
        assert set(np.unique(churn)).issubset({0, 1})


# --- 9: GOP-position accounting ----------------------------------------------------------


def test_gop_position_tagging(mc, ma, m13, mk, m19, rig):
    model, sequence, arm = rig
    rows = _run(mc, ma, m13, mk, m19, model, sequence, arm)
    boundary_indices = [r["index"] for r in rows if r["is_boundary"]]
    assert boundary_indices == [1, 5, 9]
    for r in rows:
        assert r["gop_position"] == r["index"] % GOP


# --- 10: no TEST access -------------------------------------------------------------------


def test_diagnose_reference_error_has_no_test_channel(m19):
    parameters = list(inspect.signature(m19.diagnose_reference_error).parameters)
    assert not any("test" in p.lower() for p in parameters)


def test_m19_script_never_feeds_test_sequences_into_a_fitting_call():
    source = Path("scripts/m19_reference_error_diagnostic.py").read_text(encoding="utf-8")
    fitting_calls = ("fit_recalibrated_frequencies(", "m11_g16_prototype_indices(",
                    "diagnose_reference_error(", "load_or_collect(")
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if any(call in line for call in fitting_calls):
            window = "\n".join(lines[index:index + 6])
            assert "test_sequences" not in window and 'split="test"' not in window


# --- 11: reproducibility -------------------------------------------------------------------


def test_shuffled_control_is_deterministic_for_a_fixed_seed(mc, ma, m13, mk, m19, rig):
    model, sequence, arm = rig
    rows_a = _run(mc, ma, m13, mk, m19, model, sequence, arm, seed=7)
    rows_b = _run(mc, ma, m13, mk, m19, model, sequence, arm, seed=7)
    assert [r["shuffled_bytes"] for r in rows_a] == [r["shuffled_bytes"] for r in rows_b]


def test_shuffled_control_differs_from_a_different_seed(mc, ma, m13, mk, m19, rig):
    model, sequence, arm = rig
    rows_a = _run(mc, ma, m13, mk, m19, model, sequence, arm, seed=7)
    rows_b = _run(mc, ma, m13, mk, m19, model, sequence, arm, seed=8)
    # Compare the CONTINUOUS ideal-bits estimate, not the rounded byte count:
    # this rig's tiny (~25-byte) payloads can round to the same integer byte
    # count even when the underlying permutation - confirmed genuinely
    # different for these two seeds - and its entropy cost differ.
    assert [r["shuffled_ideal_bits"] for r in rows_a] != [r["shuffled_ideal_bits"] for r in rows_b]


# --- 12: production-source compatibility --------------------------------------------------


def test_m19_scripts_do_not_modify_any_production_source():
    import subprocess
    result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1])
    modified = [line for line in result.stdout.splitlines()
               if line.startswith(" M") or line.startswith("M ")]
    modified_production = [line for line in modified if "CHANGELOG.md" not in line]
    assert modified_production == [], f"unexpected production modifications: {modified_production}"


def test_m19_introduces_no_new_container_format():
    source = Path("scripts/m19_reference_error_diagnostic.py").read_text(encoding="utf-8")
    assert "nvct_v3" not in source.lower() and "format_version = 3" not in source.lower()
    assert "TemporalStreamWriter" not in source
