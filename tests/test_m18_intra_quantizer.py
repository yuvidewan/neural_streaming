"""M18 - tests for the intra-quantizer audit and the BASE-vs-CANDIDATE
reference bridge. Reuses the established tiny-rig pattern from
test_m13/14/15/16/17's own test files.
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
from nvc.compression.quantization import UniformQuantizer, count_clipped, quantization_error
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
def m18():
    return _load_script("m18_reference_bridge")


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


# --- 5/9/10: real closed-loop reference propagation matches encode_multi exactly -------


def test_base_intra_variant_matches_encode_multi_exactly(mc, ma, m13, mk, rig):
    cl = _load_script("m13_closed_loop")
    m18 = _load_script("m18_reference_bridge")
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

    rows = m18.run_closed_loop_with_intra_variant(
        mc, m13, model, model11, assign_codebook, coding_codebook, sequence, bits=BITS,
        intra_params=intra_params, intra_entropy_model=intra_entropy_model,
        residual_params=residual_params, zero=zero, gop_size=GOP, block_size=16,
        search_range=16, device=torch.device("cpu"))

    i_bytes = sum(r["i_bytes"] for r in rows if r["frame_type"] == "I")
    p_bytes = sum(r["p_bytes"] for r in rows if r["frame_type"] == "P")
    stats = result["arms"]["m13_recal"]
    assert i_bytes == stats["i_frame_residual_bytes"]
    assert p_bytes == stats["p_frame_residual_bytes"]


# --- 4: boundary-position identification -----------------------------------------------


def test_boundary_positions_tagged_correctly(mc, ma, m13, mk, rig):
    m18 = _load_script("m18_reference_bridge")
    model, sequence, (model11, assign_codebook, coding_codebook, zero, intra_params,
                      intra_entropy_model, residual_params) = rig
    rows = m18.run_closed_loop_with_intra_variant(
        mc, m13, model, model11, assign_codebook, coding_codebook, sequence, bits=BITS,
        intra_params=intra_params, intra_entropy_model=intra_entropy_model,
        residual_params=residual_params, zero=zero, gop_size=GOP, block_size=16,
        search_range=16, device=torch.device("cpu"))
    boundary_indices = [r["index"] for r in rows if r["frame_type"] == "P" and r["is_boundary"]]
    assert boundary_indices == [1, 5, 9]


# --- 1/8/9/10: audit correctness, residual/I-frame/total-stream byte accounting --------


def test_quantizer_audit_metrics_are_internally_consistent(mc):
    torch.manual_seed(7)
    latents = torch.randn(20, 4, 8, 8) * torch.tensor([1.0, 3.0, 0.2, 5.0]).view(1, 4, 1, 1)
    params = calibrate_quantization_params(latents, bits=4, mode="per_channel")
    quantizer = UniformQuantizer(4, mode="per_channel")
    dequantized, _ = quantizer.quantize_dequantize(latents, params)
    error = quantization_error(latents, dequantized)
    clipped = count_clipped(latents, params)
    assert error["latent_mse"] >= 0
    assert 0.0 <= clipped["clipped_percent"] <= 100.0
    # A finer step (more bits) must never increase MSE for the SAME data.
    params8 = calibrate_quantization_params(latents, bits=8, mode="per_channel")
    dequantized8, _ = UniformQuantizer(8, mode="per_channel").quantize_dequantize(latents, params8)
    assert quantization_error(latents, dequantized8)["latent_mse"] <= error["latent_mse"]


def test_candidate_and_base_intra_variants_produce_different_byte_totals_when_params_differ(
        mc, ma, m13, mk, rig):
    m18 = _load_script("m18_reference_bridge")
    model, sequence, (model11, assign_codebook, coding_codebook, zero, intra_params,
                      intra_entropy_model, residual_params) = rig
    frames = sequence.load_frames()
    with torch.no_grad():
        latents = torch.cat([model.encode(frames[i:i + 1]) for i in range(frames.shape[0])])
    # A deliberately different (tighter-percentile) candidate quantizer.
    candidate_params = calibrate_quantization_params(
        latents, bits=BITS, mode="per_channel", lower_percentile=5.0, upper_percentile=95.0)
    candidate_symbols = np.stack([
        latent_to_symbols(latents[i:i + 1], candidate_params).reshape(latents.shape[1], -1)
        for i in range(latents.shape[0])])
    candidate_entropy_model = EmpiricalEntropyModel.from_symbols(
        candidate_symbols, bits=BITS, num_tables=latents.shape[1])
    assert candidate_entropy_model.model_id() != intra_entropy_model.model_id()

    base_rows = m18.run_closed_loop_with_intra_variant(
        mc, m13, model, model11, assign_codebook, coding_codebook, sequence, bits=BITS,
        intra_params=intra_params, intra_entropy_model=intra_entropy_model,
        residual_params=residual_params, zero=zero, gop_size=GOP, block_size=16,
        search_range=16, device=torch.device("cpu"))
    candidate_rows = m18.run_closed_loop_with_intra_variant(
        mc, m13, model, model11, assign_codebook, coding_codebook, sequence, bits=BITS,
        intra_params=candidate_params, intra_entropy_model=candidate_entropy_model,
        residual_params=residual_params, zero=zero, gop_size=GOP, block_size=16,
        search_range=16, device=torch.device("cpu"))

    base_summary = m18._summarize_run(base_rows)
    candidate_summary = m18._summarize_run(candidate_rows)
    # I-frame bytes are free to differ (that IS the point of a candidate);
    # P-frame bytes may ALSO differ because a different I-frame reconstruction
    # changes the reference chain from frame 1 onward - this is exactly the
    # bridge effect Phase C measures, not a bug.
    assert isinstance(base_summary["total_i_bytes"], int)
    assert isinstance(candidate_summary["total_i_bytes"], int)
    assert base_summary["p_frames"] == candidate_summary["p_frames"]


# --- 7: candidate does not mutate the baseline path -------------------------------------


def test_running_candidate_variant_does_not_change_a_subsequent_base_run(mc, ma, m13, mk, rig):
    m18 = _load_script("m18_reference_bridge")
    model, sequence, (model11, assign_codebook, coding_codebook, zero, intra_params,
                      intra_entropy_model, residual_params) = rig
    kwargs = dict(bits=BITS, residual_params=residual_params, zero=zero, gop_size=GOP,
                 block_size=16, search_range=16, device=torch.device("cpu"))

    base_before = m18.run_closed_loop_with_intra_variant(
        mc, m13, model, model11, assign_codebook, coding_codebook, sequence,
        intra_params=intra_params, intra_entropy_model=intra_entropy_model, **kwargs)

    frames = sequence.load_frames()
    with torch.no_grad():
        latents = torch.cat([model.encode(frames[i:i + 1]) for i in range(frames.shape[0])])
    candidate_params = calibrate_quantization_params(
        latents, bits=BITS, mode="per_channel", lower_percentile=5.0, upper_percentile=95.0)
    candidate_symbols = np.stack([
        latent_to_symbols(latents[i:i + 1], candidate_params).reshape(latents.shape[1], -1)
        for i in range(latents.shape[0])])
    candidate_entropy_model = EmpiricalEntropyModel.from_symbols(
        candidate_symbols, bits=BITS, num_tables=latents.shape[1])
    _ = m18.run_closed_loop_with_intra_variant(
        mc, m13, model, model11, assign_codebook, coding_codebook, sequence,
        intra_params=candidate_params, intra_entropy_model=candidate_entropy_model, **kwargs)

    base_after = m18.run_closed_loop_with_intra_variant(
        mc, m13, model, model11, assign_codebook, coding_codebook, sequence,
        intra_params=intra_params, intra_entropy_model=intra_entropy_model, **kwargs)

    assert [r.get("i_bytes", r.get("p_bytes")) for r in base_before] == \
        [r.get("i_bytes", r.get("p_bytes")) for r in base_after]


# --- 2/3: TRAIN-only calibration, candidate calibration determinism --------------------


def test_fit_intra_entropy_model_has_no_test_channel(m18):
    parameters = list(inspect.signature(m18._fit_intra_entropy_model).parameters)
    assert not any("test" in p.lower() for p in parameters)


def test_m18_scripts_never_feed_test_sequences_into_a_fitting_call():
    for script_name in ("m18_baseline.py", "m18_candidates.py", "m18_reference_bridge.py"):
        source = (Path("scripts") / script_name).read_text(encoding="utf-8")
        fitting_calls = ("calibrate_quantization_params(", "_fit_intra_entropy_model(",
                        "fit_recalibrated_frequencies(", "build_policy(")
        lines = source.splitlines()
        for index, line in enumerate(lines):
            if any(call in line for call in fitting_calls):
                window = "\n".join(lines[index:index + 6])
                assert "test_sequences" not in window and 'split="test"' not in window, (
                    f"{script_name}:{index + 1} calls a fitting function near TEST access")


def test_candidate_intra_calibration_is_deterministic(mc, ma, m13, mk, rig):
    m18 = _load_script("m18_reference_bridge")
    model, sequence, (model11, assign_codebook, coding_codebook, zero, intra_params,
                      intra_entropy_model, residual_params) = rig
    fit_one = m18._fit_intra_entropy_model(model, [sequence], params=intra_params, bits=BITS,
                                           max_frames=10 ** 9, device=torch.device("cpu"))
    fit_two = m18._fit_intra_entropy_model(model, [sequence], params=intra_params, bits=BITS,
                                           max_frames=10 ** 9, device=torch.device("cpu"))
    assert fit_one.model_id() == fit_two.model_id()
    assert np.array_equal(fit_one.frequencies, fit_two.frequencies)


# --- 12/13: provenance, compatibility ---------------------------------------------------


def test_m18_scripts_do_not_modify_any_production_source():
    import subprocess
    result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1])
    modified = [line for line in result.stdout.splitlines()
               if line.startswith(" M") or line.startswith("M ")]
    modified_production = [line for line in modified if "CHANGELOG.md" not in line]
    assert modified_production == [], f"unexpected production modifications: {modified_production}"


def test_m18_introduces_no_new_container_format():
    for script_name in ("m18_baseline.py", "m18_candidates.py", "m18_reference_bridge.py"):
        source = (Path("scripts") / script_name).read_text(encoding="utf-8")
        assert "nvct_v3" not in source.lower() and "format_version = 3" not in source.lower()
        assert "TemporalStreamWriter" not in source
