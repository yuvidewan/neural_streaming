"""M15 - tests for the calibration-policy allocation functions (Phase B) and
their compatibility with the existing deployed coding path (Phase K/L).

Two layers, matching M14's own split: (1) the allocation functions
themselves are pure and need no model/GPU/real frames - tested directly
against synthetic `BenchmarkSequence` objects; (2) the "does a policy's
fitted table thread through coding safely" question is answered by round
-tripping real (tiny, on-disk) sequences through the UNMODIFIED
`m13_closed_loop.encode_multi`/`decode_sequence`, mirroring
test_m14_closed_loop.py's rig.
"""

from __future__ import annotations

import dataclasses
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
def m15cal():
    return _load_script("m15_calibration_policy")


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


def _fake_sequences(counts: list[int], *, prefix: str = "seq") -> list[BenchmarkSequence]:
    """Synthetic TRAIN population for the pure allocation-function tests -
    no real files needed, since these functions only read `.frame_count`
    (a property of `len(frame_paths)`) and `.sequence_id`, and truncate
    `frame_paths` via `dataclasses.replace` without touching disk."""
    return [
        BenchmarkSequence(dataset="fake", sequence_id=f"{prefix}{i:02d}", split="train",
                          frame_paths=tuple(Path(f"{prefix}{i:02d}_{f}.png") for f in range(count)),
                          width=64, height=64)
        for i, count in enumerate(counts)
    ]


# --- 1/2: deterministic frame selection, exact total budget ------------------------------


def test_sequential_allocation_is_deterministic_and_hits_the_exact_budget(m15cal):
    sequences = _fake_sequences([30, 40, 50, 20])
    first = m15cal.sequential_allocation(sequences, 55)
    second = m15cal.sequential_allocation(sequences, 55)
    assert [s.frame_paths for s in first] == [s.frame_paths for s in second]
    assert sum(s.frame_count for s in first) == 55


def test_uniform_allocation_is_deterministic_and_hits_the_exact_budget(m15cal):
    sequences = _fake_sequences([30, 40, 50, 20, 25])
    first = m15cal.uniform_allocation(sequences, 47)
    second = m15cal.uniform_allocation(sequences, 47)
    assert [s.frame_paths for s in first] == [s.frame_paths for s in second]
    assert sum(s.frame_count for s in first) == 47


def test_shuffled_uniform_allocation_is_deterministic_for_a_fixed_seed(m15cal):
    sequences = _fake_sequences([30, 40, 50, 20, 25, 60])
    first = m15cal.shuffled_uniform_allocation(sequences, 100, seed=7)
    second = m15cal.shuffled_uniform_allocation(sequences, 100, seed=7)
    assert [(s.sequence_id, s.frame_count) for s in first] == \
        [(s.sequence_id, s.frame_count) for s in second]
    assert sum(s.frame_count for s in first) == 100


# --- 3: sequence coverage ------------------------------------------------------------------


def test_sequential_allocation_covers_only_the_sequences_it_needs(m15cal):
    # Budget exhausted inside the 3rd sequence - the 4th and 5th are never touched.
    sequences = _fake_sequences([30, 40, 50, 20, 25])
    allocated = m15cal.sequential_allocation(sequences, 100)
    assert [s.sequence_id for s in allocated] == ["seq00", "seq01", "seq02"]
    assert allocated[-1].frame_count == 30  # 100 - 30 - 40 = 30, truncated mid-sequence


def test_uniform_allocation_covers_every_sequence(m15cal):
    sequences = _fake_sequences([30, 40, 50, 20, 25])
    allocated = m15cal.uniform_allocation(sequences, 100)
    assert {s.sequence_id for s in allocated} == {s.sequence_id for s in sequences}


def test_coverage_statistics_report_full_vs_narrow_coverage(m15cal):
    sequences = _fake_sequences([30, 40, 50, 20, 25])
    narrow = m15cal.sequential_allocation(sequences, 55)  # seq00 full (30) + seq01 partial (25)
    broad = m15cal.uniform_allocation(sequences, 55)  # 55/5 = 11 flat, zero remainder
    narrow_stats = m15cal.coverage_statistics(narrow, sequences)
    broad_stats = m15cal.coverage_statistics(broad, sequences)
    assert narrow_stats["sequences_represented"] < broad_stats["sequences_represented"]
    assert broad_stats["percent_sequences_covered"] == 100.0
    assert narrow_stats["total_frames_selected"] == broad_stats["total_frames_selected"] == 55
    assert broad_stats["stdev_frames_per_represented_sequence"] < \
        narrow_stats["stdev_frames_per_represented_sequence"] or narrow_stats["sequences_represented"] <= 1


# --- 4/5: uniform allocation, remainder allocation -----------------------------------------


def test_uniform_allocation_distributes_the_remainder_deterministically(m15cal):
    # 47 frames / 5 sequences = base 9, remainder 2 -> first 2 sequences get 10.
    sequences = _fake_sequences([30, 40, 50, 20, 25])
    allocated = m15cal.uniform_allocation(sequences, 47)
    counts = {s.sequence_id: s.frame_count for s in allocated}
    assert counts["seq00"] == 10 and counts["seq01"] == 10
    assert counts["seq02"] == counts["seq03"] == counts["seq04"] == 9


def test_uniform_allocation_with_zero_remainder_is_perfectly_flat(m15cal):
    sequences = _fake_sequences([30, 40, 50, 20])  # 4 sequences
    allocated = m15cal.uniform_allocation(sequences, 40)  # 40 / 4 = 10, remainder 0
    assert all(s.frame_count == 10 for s in allocated)


def test_broad_flat_allocation_matches_the_m14_recipe(m15cal):
    sequences = _fake_sequences([30, 40, 50, 20, 5])
    allocated = m15cal.broad_flat_allocation(sequences, 8)
    counts = {s.sequence_id: s.frame_count for s in allocated}
    assert counts["seq00"] == counts["seq01"] == counts["seq02"] == counts["seq03"] == 8
    assert counts["seq04"] == 5  # capped at what the sequence actually has


# --- 6: current sequential-policy compatibility --------------------------------------------


def test_build_policy_a_reproduces_sequential_allocation_exactly(m15cal):
    sequences = _fake_sequences([30, 40, 50, 20, 25, 60, 70, 15])
    via_dispatch = m15cal.build_policy("A_sequential_400", sequences, total_budget=200)
    direct = m15cal.sequential_allocation(sequences, 200)
    assert [s.frame_paths for s in via_dispatch] == [s.frame_paths for s in direct]


def test_build_policy_defaults_match_the_milestones_own_definitions(m15cal):
    # No override -> A/B/D use exactly 400, C uses exactly 8/sequence (576
    # for the real 72-sequence TRAIN population) - the milestone's own
    # policy definitions, not just whatever a test happens to pass.
    sequences = _fake_sequences([500, 500, 500])  # plenty of headroom for the real defaults
    assert sum(s.frame_count for s in m15cal.build_policy("A_sequential_400", sequences)) == 400
    assert sum(s.frame_count for s in m15cal.build_policy("B_uniform_400", sequences)) == 400
    assert sum(s.frame_count for s in m15cal.build_policy("D_shuffled_uniform_400", sequences)) == 400
    assert sum(s.frame_count for s in m15cal.build_policy("C_broad_576", sequences)) == 8 * 3


def test_frame_paths_are_always_a_prefix_never_a_different_selection(m15cal):
    # Every policy must select a PREFIX of each sequence's own frames, never
    # skip around within it - calibrate_grids' own walk only ever consumes
    # front-to-back, so a policy that did anything else would not be
    # comparable to it or reproducible via the same collectors.
    sequences = _fake_sequences([30, 40, 50, 20, 25, 60, 70, 15])
    for name in m15cal.POLICY_NAMES:
        for allocated in m15cal.build_policy(name, sequences, total_budget=100):
            original = next(s for s in sequences if s.sequence_id == allocated.sequence_id)
            assert allocated.frame_paths == original.frame_paths[:allocated.frame_count]


# --- 7/8: TRAIN-only, no TEST channel --------------------------------------------------


def test_policy_functions_have_no_test_channel(m15cal):
    for fn in (m15cal.sequential_allocation, m15cal.uniform_allocation,
              m15cal.broad_flat_allocation, m15cal.shuffled_uniform_allocation,
              m15cal.build_policy):
        parameters = list(inspect.signature(fn).parameters)
        assert not any("test" in p.lower() for p in parameters)


@pytest.mark.parametrize("script_name", ["m15_offline_gate.py", "m15_coded_validation.py",
                                         "m15_davis_benchmark.py"])
def test_scripts_never_feed_test_sequences_into_a_fitting_or_policy_call(script_name):
    # Mirrors M13/M14's own equivalent guard.
    source = Path("scripts") / script_name
    text = source.read_text(encoding="utf-8")
    fitting_calls = ("collect_intra_symbols(", "collect_motion_symbols(", "fit_empirical(",
                    "build_policy(", "m15cal.build_policy(")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if any(call in line for call in fitting_calls):
            window = "\n".join(lines[index:index + 4])
            assert "test_sequences" not in window, (
                f"{script_name}:{index + 1} calls a fitting/policy function near "
                f"'test_sequences' - TEST must never reach the fit")


# --- 9: cross-process reproducibility (in-process determinism, M14's own convention) -------


def test_build_policy_is_deterministic_across_independent_calls(m15cal):
    sequences = _fake_sequences([25, 33, 41, 18, 29, 37])
    for name in m15cal.POLICY_NAMES:
        first = m15cal.build_policy(name, sequences, seed=42, total_budget=96)
        second = m15cal.build_policy(name, sequences, seed=42, total_budget=96)
        assert [(s.sequence_id, s.frame_paths) for s in first] == \
            [(s.sequence_id, s.frame_paths) for s in second]


# --- integration rig: real (tiny, on-disk) sequences through the real coder ----------------


TINY = {"in_channels": 3, "latent_channels": 4, "base_channels": 8}
BITS = 4


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


def _policy_motion_table(m15cal, mc, m14, model, train_sequences, policy_name, *, seed=42,
                         total_budget=40):
    seqs = m15cal.build_policy(policy_name, train_sequences, seed=seed, total_budget=total_budget)
    symbols = m14.collect_motion_symbols(mc, model, seqs, block_size=16, search_range=8,
                                         gop_size=100, max_frames=10 ** 9, reference_mode="mc",
                                         device=torch.device("cpu"))
    return m14.fit_empirical(symbols, bits=mc.motion_alphabet_bits(8), num_tables=2)


def _rig(tmp_path, mc, mk, ml, ma, m13, m14, m15cal, *, frames=12, seed=0):
    model = _autoencoder(seed)
    clip_sequence = _write_sequence(tmp_path, "clip", frames, seed=seed)
    clip = clip_sequence.load_frames()
    calibration = _calibration(mc, model, clip, bits=BITS)
    arms = _residual_arm(mc, mk, ml, ma, m13, calibration, seed=seed)

    # A synthetic TRAIN population with deliberately UNEVEN sequence lengths,
    # so Policy A (sequential) and Policy B (uniform) allocate visibly
    # differently - the whole point of what is being tested here.
    train_sequences = [_write_sequence(tmp_path, f"train{i}", n, seed=seed + i)
                       for i, n in enumerate([10, 20, 10, 20, 10, 20])]
    policy_a_motion = _policy_motion_table(m15cal, mc, m14, model, train_sequences,
                                           "A_sequential_400", seed=seed)
    policy_b_motion = _policy_motion_table(m15cal, mc, m14, model, train_sequences,
                                           "B_uniform_400", seed=seed)
    return model, clip, calibration, arms, policy_a_motion, policy_b_motion, train_sequences


def _encode_decode(mc, ma, m13, model, frames, arms, tmp_path, calibration, *, motion_model,
                   gop=4, path_name="stream.nvct"):
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


# --- 13: calibration identity changes -------------------------------------------------


def test_different_policies_produce_distinct_motion_identities(tmp_path, mc, mk, ml, ma, m13,
                                                                m14, m15cal):
    model, frames, calibration, arms, policy_a_motion, policy_b_motion, _ = _rig(
        tmp_path, mc, mk, ml, ma, m13, m14, m15cal)
    assert policy_a_motion.model_id() != policy_b_motion.model_id()
    assert len(policy_a_motion.model_id()) == len(policy_b_motion.model_id()) == 8


# --- 10/11/12/15/16: fixed symbols/motion/reconstruction, old/new stream compatibility -----


def test_policy_a_stream_round_trips_unchanged(tmp_path, mc, mk, ml, ma, m13, m14, m15cal):
    model, frames, calibration, arms, policy_a_motion, _, _ = _rig(
        tmp_path, mc, mk, ml, ma, m13, m14, m15cal)
    result, decoded, symbols, _ = _encode_decode(
        mc, ma, m13, model, frames, arms, tmp_path, calibration, motion_model=policy_a_motion)
    assert torch.equal(decoded.cpu(), result["reconstructions"])
    for coded, back in zip(result["symbols"], symbols):
        assert np.array_equal(coded.reshape(-1), back.reshape(-1))


def test_policy_b_stream_round_trips_correctly(tmp_path, mc, mk, ml, ma, m13, m14, m15cal):
    model, frames, calibration, arms, _, policy_b_motion, _ = _rig(
        tmp_path, mc, mk, ml, ma, m13, m14, m15cal)
    result, decoded, symbols, timings = _encode_decode(
        mc, ma, m13, model, frames, arms, tmp_path, calibration, motion_model=policy_b_motion)
    assert torch.equal(decoded.cpu(), result["reconstructions"])
    for coded, back in zip(result["symbols"], symbols):
        assert np.array_equal(coded.reshape(-1), back.reshape(-1))
    assert set(timings) == {"network", "tables", "coder"}


def test_switching_the_calibration_policy_changes_neither_symbols_nor_reconstruction(
        tmp_path, mc, mk, ml, ma, m13, m14, m15cal):
    model, frames, calibration, arms, policy_a_motion, policy_b_motion, _ = _rig(
        tmp_path, mc, mk, ml, ma, m13, m14, m15cal)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a_result, a_decoded, a_symbols, _ = _encode_decode(
        mc, ma, m13, model, frames, arms, tmp_path / "a", calibration, motion_model=policy_a_motion)
    b_result, b_decoded, b_symbols, _ = _encode_decode(
        mc, ma, m13, model, frames, arms, tmp_path / "b", calibration, motion_model=policy_b_motion)

    # Residual symbols and the final reconstruction are UNCHANGED - only
    # which calibration policy fitted the motion table differs.
    for a, b in zip(a_symbols, b_symbols):
        assert np.array_equal(a.reshape(-1), b.reshape(-1))
    assert torch.equal(a_decoded.cpu(), b_decoded.cpu())
    assert a_result["arms"]["m13_recal"]["i_frame_residual_bytes"] \
        == b_result["arms"]["m13_recal"]["i_frame_residual_bytes"]
    assert a_result["arms"]["m13_recal"]["p_frame_residual_bytes"] \
        == b_result["arms"]["m13_recal"]["p_frame_residual_bytes"]


# --- 14: provenance mismatch rejection, exercised with an M15-policy-derived table ---------


def test_decoding_a_policy_b_stream_with_the_policy_a_table_is_rejected(
        tmp_path, mc, mk, ml, ma, m13, m14, m15cal):
    cl = _load_script("m13_closed_loop")
    model, frames, calibration, arms, policy_a_motion, policy_b_motion, _ = _rig(
        tmp_path, mc, mk, ml, ma, m13, m14, m15cal)
    paths = {"m13_recal": tmp_path / "policy_b.nvct"}
    cl.encode_multi(mc, ma, m13, model, frames, arms, paths,
                    intra_params=calibration["intra_params"],
                    intra_entropy_model=calibration["intra_entropy_model"],
                    residual_params=calibration["residual_params"],
                    motion_entropy_model=policy_b_motion, bits=BITS, gop_size=4, block_size=16,
                    search_range=8)

    with pytest.raises(mc.TemporalFormatError, match="motion entropy model mismatch"):
        cl.decode_sequence(mc, ma, m13, model, paths["m13_recal"], "m13_recal", arms["m13_recal"],
                           intra_entropy_model=calibration["intra_entropy_model"],
                           motion_entropy_model=policy_a_motion,  # WRONG policy's table
                           bits=BITS)

    decoded, _, _ = cl.decode_sequence(mc, ma, m13, model, paths["m13_recal"], "m13_recal",
                                       arms["m13_recal"],
                                       intra_entropy_model=calibration["intra_entropy_model"],
                                       motion_entropy_model=policy_b_motion, bits=BITS)
    assert decoded.shape[0] == frames.shape[0]


# --- .nvct v2 unchanged, no new container version ------------------------------------------


def test_m15_introduces_no_new_container_format():
    for script_name in ("m15_offline_gate.py", "m15_coded_validation.py", "m15_davis_benchmark.py"):
        source = (Path("scripts") / script_name).read_text(encoding="utf-8")
        assert "TEMPORAL_FORMAT_VERSION" not in source or "= 2" in source
        assert "nvct_v3" not in source.lower() and "format_version = 3" not in source.lower()
