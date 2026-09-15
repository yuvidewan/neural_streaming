"""M17 - tests for the real-vs-oracle residual diagnostic through the
ACTUAL deployed M11-G16 + M13 pipeline. Reuses the established tiny-rig
pattern from test_m13/14/15/16's own test files (BaselineAutoencoder TINY,
on-disk synthetic sequences, an inline M11-G16 + codebook + M13 recalibration
setup) rather than the real checkpoint, so these run in seconds and catch
wiring bugs before any GPU-hours are spent on the real diagnostic.
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
def m17():
    return _load_script("m17_residual_diagnostic")


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
    """A tiny but REAL M11-G16 + K=512(->16 here) + M13-recalibrated arm,
    mirroring test_m14/15/16_closed_loop's own established rig."""
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


# --- 1: exact real-reference chain (reuses M16's own proven pattern) ------------------


def test_diagnose_residual_oracle_total_bytes_match_encode_multi_total(mc, ma, m13, mk, rig):
    cl = _load_script("m13_closed_loop")
    m17 = _load_script("m17_residual_diagnostic")
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
    paths = {"m13_recal": Path(sequence.frame_paths[0]).parent / "scratch2.nvct"}
    result = cl.encode_multi(mc, ma, m13, model, frames, arms, paths,
                             intra_params=intra_params, intra_entropy_model=intra_entropy_model,
                             residual_params=residual_params, motion_entropy_model=motion_model,
                             bits=BITS, gop_size=GOP, block_size=16, search_range=16)
    paths["m13_recal"].unlink(missing_ok=True)

    rows = m17.diagnose_residual_oracle(
        mc, ma, m13, mk, model, model11, assign_codebook, coding_codebook, sequence, bits=BITS,
        intra_params=intra_params, intra_entropy_model=intra_entropy_model,
        residual_params=residual_params, zero=zero, gop_size=GOP, block_size=16,
        search_range=16, device=torch.device("cpu"))

    assert sum(r["A_real_bytes"] for r in rows) == result["arms"]["m13_recal"]["p_frame_residual_bytes"]
    assert len(rows) == result["arms"]["m13_recal"]["p_frames"]


# --- 2: oracle path isolation (B/C never feed back into the real chain) ---------------


def test_oracle_variants_never_affect_the_real_chain(mc, ma, m13, mk, rig):
    m17 = _load_script("m17_residual_diagnostic")
    model, sequence, (model11, assign_codebook, coding_codebook, zero, intra_params,
                      intra_entropy_model, residual_params) = rig
    rows_first = m17.diagnose_residual_oracle(
        mc, ma, m13, mk, model, model11, assign_codebook, coding_codebook, sequence, bits=BITS,
        intra_params=intra_params, intra_entropy_model=intra_entropy_model,
        residual_params=residual_params, zero=zero, gop_size=GOP, block_size=16,
        search_range=16, device=torch.device("cpu"))
    # Re-running must reproduce identical A_real bytes - if B/C leaked into
    # the real chain's state, a second run would silently diverge.
    rows_second = m17.diagnose_residual_oracle(
        mc, ma, m13, mk, model, model11, assign_codebook, coding_codebook, sequence, bits=BITS,
        intra_params=intra_params, intra_entropy_model=intra_entropy_model,
        residual_params=residual_params, zero=zero, gop_size=GOP, block_size=16,
        search_range=16, device=torch.device("cpu"))
    assert [r["A_real_bytes"] for r in rows_first] == [r["A_real_bytes"] for r in rows_second]


# --- Phase F: oracle-variant payloads round-trip through decode_frame_recalibrated -----


def test_oracle_variant_payloads_round_trip_when_given_the_matching_reference(mc, ma, m13, mk, rig):
    # This is what Phase F's "coded validation gate" actually checks: that
    # B/C's reported byte counts are genuine, correctly-decodable arithmetic
    # coding, not a computation artifact - given the SAME reference a
    # hypothetical oracle-aware decoder would need (which no real decoder
    # ever has - see m17_coded_validation_gate.py's own module docstring for
    # why this is a corroboration, never a claim of deployability).
    m17 = _load_script("m17_residual_diagnostic")
    model, sequence, (model11, assign_codebook, coding_codebook, zero, intra_params,
                      intra_entropy_model, residual_params) = rig
    rows = m17.diagnose_residual_oracle(
        mc, ma, m13, mk, model, model11, assign_codebook, coding_codebook, sequence, bits=BITS,
        intra_params=intra_params, intra_entropy_model=intra_entropy_model,
        residual_params=residual_params, zero=zero, gop_size=GOP, block_size=16,
        search_range=16, device=torch.device("cpu"), keep_payloads_and_references=True)
    assert len(rows) > 0
    latent_shape = tuple(rows[0]["C_full_oracle_reference"].shape[1:])
    for r in rows:
        for name in ("A_real", "B_oracle_ref_real_motion", "C_full_oracle"):
            decoded = m13.decode_frame_recalibrated(
                model11, assign_codebook, coding_codebook, r[f"{name}_payload"],
                r[f"{name}_reference"], zero, bits=BITS, shape=latent_shape)
            assert np.array_equal(decoded, r[f"{name}_symbols"])
            assert len(r[f"{name}_payload"]) == r[f"{name}_bytes"]


def test_keep_payloads_flag_defaults_off_and_does_not_change_reported_bytes(mc, ma, m13, mk, rig):
    # The diagnostic flag must be strictly additive - Phase B/D/E's own
    # numbers (computed with the flag off) must be unaffected by it.
    m17 = _load_script("m17_residual_diagnostic")
    model, sequence, (model11, assign_codebook, coding_codebook, zero, intra_params,
                      intra_entropy_model, residual_params) = rig
    kwargs = dict(bits=BITS, intra_params=intra_params, intra_entropy_model=intra_entropy_model,
                 residual_params=residual_params, zero=zero, gop_size=GOP, block_size=16,
                 search_range=16, device=torch.device("cpu"))
    rows_off = m17.diagnose_residual_oracle(mc, ma, m13, mk, model, model11, assign_codebook,
                                            coding_codebook, sequence, **kwargs)
    rows_on = m17.diagnose_residual_oracle(mc, ma, m13, mk, model, model11, assign_codebook,
                                           coding_codebook, sequence, keep_payloads_and_references=True,
                                           **kwargs)
    assert "A_real_payload" not in rows_off[0]
    assert "A_real_payload" in rows_on[0]
    assert [r["A_real_bytes"] for r in rows_off] == [r["A_real_bytes"] for r in rows_on]


# --- 3: no TEST access during fitting --------------------------------------------------


def test_diagnose_residual_oracle_has_no_test_channel(m17):
    parameters = list(inspect.signature(m17.diagnose_residual_oracle).parameters)
    assert not any("test" in p.lower() for p in parameters)


def test_m17_script_never_feeds_test_sequences_into_a_fitting_call():
    source = Path("scripts/m17_residual_diagnostic.py").read_text(encoding="utf-8")
    fitting_calls = ("fit_recalibrated_frequencies(", "m11_g16_prototype_indices(",
                    "diagnose_residual_oracle(", "load_or_collect(")
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if any(call in line for call in fitting_calls):
            window = "\n".join(lines[index:index + 6])
            assert "test_sequences" not in window and "split=\"test\"" not in window, (
                f"line {index + 1} calls a fitting function near TEST access")


# --- 4/5/6/7/8: deterministic repeated diagnostics, symbol/context/assignment/code-length ---


def test_diagnose_residual_oracle_is_fully_deterministic(mc, ma, m13, mk, rig):
    m17 = _load_script("m17_residual_diagnostic")
    model, sequence, (model11, assign_codebook, coding_codebook, zero, intra_params,
                      intra_entropy_model, residual_params) = rig
    kwargs = dict(bits=BITS, intra_params=intra_params, intra_entropy_model=intra_entropy_model,
                 residual_params=residual_params, zero=zero, gop_size=GOP, block_size=16,
                 search_range=16, device=torch.device("cpu"))
    rows_a = m17.diagnose_residual_oracle(mc, ma, m13, mk, model, model11, assign_codebook,
                                          coding_codebook, sequence, **kwargs)
    rows_b = m17.diagnose_residual_oracle(mc, ma, m13, mk, model, model11, assign_codebook,
                                          coding_codebook, sequence, **kwargs)
    for a, b in zip(rows_a, rows_b):
        assert a["A_real_bytes"] == b["A_real_bytes"]
        assert a["C_full_oracle_bytes"] == b["C_full_oracle_bytes"]
        assert np.array_equal(a["A_real_symbols"], b["A_real_symbols"])
        assert np.array_equal(a["A_real_table_index"], b["A_real_table_index"])
        assert a["A_real_ideal_bits"] == b["A_real_ideal_bits"]


def test_variant_b_isolates_reference_from_motion(mc, ma, m13, mk, rig):
    # B uses A's real motion vectors but the oracle reference pixels; C
    # additionally re-estimates motion against the oracle reference. A/B/C
    # must therefore generally differ in the fields the milestone asks to
    # decompose, and every row must carry all three variants' data.
    m17 = _load_script("m17_residual_diagnostic")
    model, sequence, (model11, assign_codebook, coding_codebook, zero, intra_params,
                      intra_entropy_model, residual_params) = rig
    rows = m17.diagnose_residual_oracle(
        mc, ma, m13, mk, model, model11, assign_codebook, coding_codebook, sequence, bits=BITS,
        intra_params=intra_params, intra_entropy_model=intra_entropy_model,
        residual_params=residual_params, zero=zero, gop_size=GOP, block_size=16,
        search_range=16, device=torch.device("cpu"))
    assert len(rows) > 0
    for r in rows:
        for name in ("A_real", "B_oracle_ref_real_motion", "C_full_oracle"):
            assert r[f"{name}_bytes"] > 0
            assert r[f"{name}_ideal_bits"] >= 0
        assert 0.0 <= r["fraction_assignments_changed_A_vs_B"] <= 1.0
        assert 0.0 <= r["fraction_assignments_changed_B_vs_C"] <= 1.0
        assert 0.0 <= r["fraction_symbols_changed_A_vs_C"] <= 1.0


# --- 12: GOP-position accounting --------------------------------------------------------


def test_gop_position_tagging_matches_m16_convention(mc, ma, m13, mk, rig):
    m17 = _load_script("m17_residual_diagnostic")
    model, sequence, (model11, assign_codebook, coding_codebook, zero, intra_params,
                      intra_entropy_model, residual_params) = rig
    rows = m17.diagnose_residual_oracle(
        mc, ma, m13, mk, model, model11, assign_codebook, coding_codebook, sequence, bits=BITS,
        intra_params=intra_params, intra_entropy_model=intra_entropy_model,
        residual_params=residual_params, zero=zero, gop_size=GOP, block_size=16,
        search_range=16, device=torch.device("cpu"))
    boundary_indices = [r["index"] for r in rows if r["is_boundary"]]
    assert boundary_indices == [1, 5, 9]  # gop=4 -> I at 0,4,8,12
    for r in rows:
        assert r["gop_position"] == r["index"] % GOP


# --- 9/10: no production-source modification, existing-stream compatibility -----------


def test_m17_scripts_do_not_modify_any_production_source():
    import subprocess
    result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1])
    # Scoped to src/nvc/ deliberately. An earlier version failed on ANY modified
    # tracked file, so an unrelated in-progress edit (a README change, say) looked
    # identical to this milestone touching production code.
    modified = [line for line in result.stdout.splitlines()
               if line[:2].strip() in {"M", "A", "D", "R"}]
    modified_production = [line for line in modified if "src/nvc/" in line.replace("\\", "/")]
    assert modified_production == [], f"unexpected production modifications: {modified_production}"


def test_m17_introduces_no_new_container_format():
    source = Path("scripts/m17_residual_diagnostic.py").read_text(encoding="utf-8")
    assert "nvct_v3" not in source.lower() and "format_version = 3" not in source.lower()
    assert "TemporalStreamWriter" not in source and "TemporalStreamHeader" not in source
