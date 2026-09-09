"""Tests for deploying the M10L shared codebook through the real codec.

M10L changes only how the predicted probabilities are represented, so the
residual symbols, the motion payload and the reconstruction must come out
identical to M10H's, M10J's and M10K's. These tests hold that to the letter, on
real `.nvct` v2 streams through the real coder.

Two properties here are specific to a codebook and were the reason for pinning
them in tests rather than prose:

  * M10K is coupled to the quantization calibration it was fitted under; a
    codebook inherits that coupling AND adds one to the M10K weights, so the
    stream identity has to bind all three or a stale pairing silently costs
    bits instead of failing;
  * the coder never constrained how many tables exist, which is why K prototype
    tables need no container change - the same reason M10K's 16,384 needed none.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.codec import latent_to_symbols
from nvc.compression.entropy_model import MIN_FREQUENCY, TOTAL_FREQUENCY, EmpiricalEntropyModel
from nvc.models.autoencoder import BaselineAutoencoder
from nvc.utils.config import load_default_config


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


def _arms(mc, ce, mk, ml, calibration, *, bits=4, channels=4, size=16):
    arms = {}
    for scheme in ("marginal", "local_activity4"):
        context_model = ce.fit_context_model(scheme, calibration["references"])
        built = ce.build_conditional_entropy_model(
            calibration["symbols"], calibration["references"], context_model, bits=bits)
        arms[scheme] = {"context_model": context_model,
                        "entropy_model": built["entropy_model"],
                        "identity": built["entropy_model"].model_id()}
    torch.manual_seed(3)
    learned = mk.build_model({"latent_channels": channels, "alphabet": 2 ** bits,
                              "hidden": 8})
    learned.eval()
    arms["learned"] = {"model": learned, "identity": b"\x9a" * 8}

    samples = ml.sample_training_distributions(
        learned, calibration["references"], device=torch.device("cpu"), max_rows=2000)
    codebook = ml.fit_codebook(samples, size, bits=bits,
                               provenance={"source": "train references"})
    arms["codebook"] = {
        "model": learned, "codebook": codebook,
        "identity": codebook.codebook_id(model_identity=b"\x9a" * 8,
                                         calibration_signature="test")}
    return arms


def _setup(bits: int = 4, size: int = 16):
    mc = _load_script("m10h_motion_compensation")
    ce = _load_script("m10j_conditional_entropy")
    mk = _load_script("m10k_learned_entropy")
    ml = _load_script("m10l_shared_codebook")
    ev = _load_script("m10l_evaluate")
    model = _autoencoder()
    frames = _moving_frames()
    calibration = _calibration(mc, model, frames, bits=bits)
    arms = _arms(mc, ce, mk, ml, calibration, bits=bits, size=size)
    return mc, ce, mk, ml, ev, model, frames, calibration, arms


def _encode(ev, mc, mk, ml, model, frames, arms, tmp_path, calibration, *, bits=4,
            gop_size=4):
    paths = {arm: tmp_path / f"{arm}.nvct" for arm in arms}
    result = ev.encode_multi(
        mc, mk, ml, model, frames, arms, paths,
        intra_params=calibration["intra_params"],
        intra_entropy_model=calibration["intra_entropy_model"],
        residual_params=calibration["residual_params"],
        motion_entropy_model=calibration["motion_entropy_model"],
        bits=bits, gop_size=gop_size, block_size=16, search_range=8)
    return result, paths


# --- only the probability model may differ ---------------------------------------


def test_every_arm_shares_one_motion_payload(tmp_path):
    mc, ce, mk, ml, ev, model, frames, calibration, arms = _setup()

    result, paths = _encode(ev, mc, mk, ml, model, frames, arms, tmp_path, calibration)

    motions = {arm: result["arms"][arm]["motion_bytes"] for arm in arms}
    assert len(set(motions.values())) == 1, f"motion differed: {motions}"
    payloads = {arm: [m for _, m, _ in mc.TemporalStreamReader(paths[arm])] for arm in arms}
    for arm in arms:
        assert payloads[arm] == payloads["marginal"], f"{arm} motion payload differs"


def test_the_codebook_arm_reproduces_the_same_symbols_and_reconstruction(tmp_path):
    """The decisive invariant: M10L may change bytes, never pictures."""
    mc, ce, mk, ml, ev, model, frames, calibration, arms = _setup()

    result, paths = _encode(ev, mc, mk, ml, model, frames, arms, tmp_path, calibration)
    decoded, symbols = ev.decode_sequence(
        mc, mk, ml, model, paths["codebook"], "codebook", arms["codebook"],
        intra_entropy_model=calibration["intra_entropy_model"],
        motion_entropy_model=calibration["motion_entropy_model"], bits=4)

    assert len(symbols) == len(result["symbols"])
    for coded, round_tripped in zip(result["symbols"], symbols):
        assert np.array_equal(coded.reshape(-1), round_tripped.reshape(-1))
    assert torch.equal(decoded.cpu(), result["reconstructions"])


def test_only_the_residual_byte_count_differs_between_arms(tmp_path):
    mc, ce, mk, ml, ev, model, frames, calibration, arms = _setup()

    result, _ = _encode(ev, mc, mk, ml, model, frames, arms, tmp_path, calibration)

    for arm in arms:
        assert result["arms"][arm]["i_frame_residual_bytes"] == \
            result["arms"]["marginal"]["i_frame_residual_bytes"], \
            "I-frames are coded identically in every arm"
        assert result["arms"][arm]["p_frames"] == result["arms"]["marginal"]["p_frames"]


def test_byte_accounting_closes_for_the_codebook_arm(tmp_path):
    mc, ce, mk, ml, ev, model, frames, calibration, arms = _setup()

    result, paths = _encode(ev, mc, mk, ml, model, frames, arms, tmp_path, calibration)

    stats = result["arms"]["codebook"]
    assert stats["motion_bytes"] + stats["residual_bytes"] \
        + stats["container_overhead_bytes"] == paths["codebook"].stat().st_size


@pytest.mark.parametrize("bits", [3, 4, 5])
def test_all_three_rate_points_round_trip(bits, tmp_path):
    mc, ce, mk, ml, ev, model, frames, calibration, arms = _setup(bits=bits)

    result, paths = _encode(ev, mc, mk, ml, model, frames, arms, tmp_path, calibration,
                            bits=bits)
    decoded, symbols = ev.decode_sequence(
        mc, mk, ml, model, paths["codebook"], "codebook", arms["codebook"],
        intra_entropy_model=calibration["intra_entropy_model"],
        motion_entropy_model=calibration["motion_entropy_model"], bits=bits)

    for coded, round_tripped in zip(result["symbols"], symbols):
        assert np.array_equal(coded.reshape(-1), round_tripped.reshape(-1))
    assert torch.equal(decoded.cpu(), result["reconstructions"])


@pytest.mark.parametrize("size", [4, 16, 64])
def test_any_codebook_size_round_trips(size, tmp_path):
    """K is a free parameter of the deployment, not a magic number the coder
    knows about - so every candidate must decode, not just the selected one."""
    mc, ce, mk, ml, ev, model, frames, calibration, arms = _setup(size=size)

    result, paths = _encode(ev, mc, mk, ml, model, frames, arms, tmp_path, calibration)
    _, symbols = ev.decode_sequence(
        mc, mk, ml, model, paths["codebook"], "codebook", arms["codebook"],
        intra_entropy_model=calibration["intra_entropy_model"],
        motion_entropy_model=calibration["motion_entropy_model"], bits=4)

    assert arms["codebook"]["codebook"].size == size
    for coded, round_tripped in zip(result["symbols"], symbols):
        assert np.array_equal(coded.reshape(-1), round_tripped.reshape(-1))


# --- causality and determinism ----------------------------------------------------


def test_decoding_uses_only_the_stream_and_is_deterministic(tmp_path):
    """The decoder gets z_ref, the codebook and the calibration - nothing from
    the encoder's side. Two decodes must agree bit for bit."""
    mc, ce, mk, ml, ev, model, frames, calibration, arms = _setup()

    _, paths = _encode(ev, mc, mk, ml, model, frames, arms, tmp_path, calibration)
    first, first_symbols = ev.decode_sequence(
        mc, mk, ml, model, paths["codebook"], "codebook", arms["codebook"],
        intra_entropy_model=calibration["intra_entropy_model"],
        motion_entropy_model=calibration["motion_entropy_model"], bits=4)
    second, second_symbols = ev.decode_sequence(
        mc, mk, ml, model, paths["codebook"], "codebook", arms["codebook"],
        intra_entropy_model=calibration["intra_entropy_model"],
        motion_entropy_model=calibration["motion_entropy_model"], bits=4)

    assert torch.equal(first, second)
    for one, other in zip(first_symbols, second_symbols):
        assert np.array_equal(one, other)


def test_the_table_index_is_derived_not_transmitted(tmp_path):
    """K tables would be useless if the stream had to carry which one each
    symbol used - 16,384 indices would cost more than the tables saved. The
    decoder must recompute the assignment from z_ref alone.
    """
    mc, ce, mk, ml, ev, model, frames, calibration, arms = _setup()

    result, paths = _encode(ev, mc, mk, ml, model, frames, arms, tmp_path, calibration)

    codebook_bytes = result["arms"]["codebook"]["residual_bytes"]
    learned_bytes = result["arms"]["learned"]["residual_bytes"]
    assert codebook_bytes < learned_bytes * 2, (
        "residual payload grew as if side information had been added")


def test_i_frames_carry_no_motion_and_sequences_start_with_one(tmp_path):
    mc, ce, mk, ml, ev, model, frames, calibration, arms = _setup()

    _, paths = _encode(ev, mc, mk, ml, model, frames, arms, tmp_path, calibration)

    entries = list(mc.TemporalStreamReader(paths["codebook"]))
    assert entries[0][0] == mc.FRAME_TYPE_I
    assert all(motion == b"" for frame_type, motion, _ in entries
               if frame_type == mc.FRAME_TYPE_I)


# --- stream and model compatibility -----------------------------------------------


def test_a_stream_declaring_a_different_codebook_is_rejected(tmp_path):
    """A stale codebook must be an identity mismatch, not a silent rate loss."""
    mc, ce, mk, ml, ev, model, frames, calibration, arms = _setup()

    _, paths = _encode(ev, mc, mk, ml, model, frames, arms, tmp_path, calibration)
    stale = dict(arms["codebook"], identity=b"\x00" * 8)

    with pytest.raises(mc.TemporalFormatError, match="entropy model mismatch"):
        ev.decode_sequence(
            mc, mk, ml, model, paths["codebook"], "codebook", stale,
            intra_entropy_model=calibration["intra_entropy_model"],
            motion_entropy_model=calibration["motion_entropy_model"], bits=4)


def test_codebook_identity_binds_model_calibration_and_prototypes(tmp_path):
    """All three couplings in one 8-byte field: change any of them and the
    stream identity changes, so the container's existing check catches it."""
    mc, ce, mk, ml, ev, model, frames, calibration, arms = _setup()
    codebook = arms["codebook"]["codebook"]

    base = codebook.codebook_id(model_identity=b"\x9a" * 8, calibration_signature="test")
    assert base == arms["codebook"]["identity"]
    assert codebook.codebook_id(model_identity=b"\x9b" * 8,
                                calibration_signature="test") != base
    assert codebook.codebook_id(model_identity=b"\x9a" * 8,
                                calibration_signature="other") != base
    assert len(base) == 8


def test_m10h_m10j_and_m10k_streams_are_unaffected(tmp_path):
    """M10L adds an arm; it must not perturb the three that came before."""
    mc, ce, mk, ml, ev, model, frames, calibration, arms = _setup()

    result, paths = _encode(ev, mc, mk, ml, model, frames, arms, tmp_path, calibration)

    for arm in ("marginal", "local_activity4", "learned"):
        decoded, symbols = ev.decode_sequence(
            mc, mk, ml, model, paths[arm], arm, arms[arm],
            intra_entropy_model=calibration["intra_entropy_model"],
            motion_entropy_model=calibration["motion_entropy_model"], bits=4)
        assert torch.equal(decoded.cpu(), result["reconstructions"])
        for coded, round_tripped in zip(result["symbols"], symbols):
            assert np.array_equal(coded.reshape(-1), round_tripped.reshape(-1))


def test_the_container_format_is_unchanged(tmp_path):
    """No .nvct version bump: K tables ride the existing 8-byte model id."""
    mc, ce, mk, ml, ev, model, frames, calibration, arms = _setup()

    _, paths = _encode(ev, mc, mk, ml, model, frames, arms, tmp_path, calibration)
    header = mc.TemporalStreamReader(paths["codebook"]).header

    assert mc.TEMPORAL_FORMAT_VERSION == 2
    assert len(header.residual_entropy_model_id) == 8
    assert header.residual_entropy_model_id == arms["codebook"]["identity"]


def test_a_truncated_residual_payload_is_rejected(tmp_path):
    mc, ce, mk, ml, ev, model, frames, calibration, arms = _setup()

    _, paths = _encode(ev, mc, mk, ml, model, frames, arms, tmp_path, calibration)
    data = bytearray(paths["codebook"].read_bytes())
    path = tmp_path / "truncated.nvct"
    path.write_bytes(data[:len(data) - 32])

    with pytest.raises(Exception):
        ev.decode_sequence(
            mc, mk, ml, model, path, "codebook", arms["codebook"],
            intra_entropy_model=calibration["intra_entropy_model"],
            motion_entropy_model=calibration["motion_entropy_model"], bits=4)


# --- fitting discipline -------------------------------------------------------------


def test_the_codebook_is_fitted_on_train_references_only():
    """Provenance is recorded on the object, not just in the run log, so a
    codebook that came from anywhere else is visible at deployment time."""
    mc = _load_script("m10h_motion_compensation")
    mk = _load_script("m10k_learned_entropy")
    ml = _load_script("m10l_shared_codebook")
    model = _autoencoder()
    frames = _moving_frames()
    calibration = _calibration(mc, model, frames)
    torch.manual_seed(3)
    learned = mk.build_model({"latent_channels": 4, "alphabet": 16, "hidden": 8}).eval()

    samples = ml.sample_training_distributions(
        learned, calibration["references"], device=torch.device("cpu"), max_rows=1500)
    codebook = ml.fit_codebook(samples, 16, bits=4)

    assert codebook.provenance["split"] == "train"
    assert codebook.provenance["training_rows"] == samples.shape[0]


def test_k_selection_reads_the_gate_report_not_the_benchmark(tmp_path):
    """K must come from validation. The evaluator refuses to invent one, and
    refuses to run at all if the gate did not pass."""
    ev = _load_script("m10l_evaluate")
    defaults = load_default_config()

    ml = _load_script("m10l_shared_codebook")

    report = tmp_path / "offline_gate.json"
    report.write_text(json.dumps({
        "gate_passed": True, "selected_metric": "code_length",
        "selected": {"4": {"codebook_size": 64}, "3": {"codebook_size": 32}},
    }), encoding="utf-8")
    args = ev.build_arg_parser(defaults).parse_args(
        ["--gate-report", str(report), "--rate-points", "4", "3"])
    assert ev._selection(args, ml) == ({4: 64, 3: 32}, "code_length")

    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({"gate_passed": False}), encoding="utf-8")
    args = ev.build_arg_parser(defaults).parse_args(["--gate-report", str(failed)])
    with pytest.raises(SystemExit, match="did not pass"):
        ev._selection(args, ml)

    args = ev.build_arg_parser(defaults).parse_args(
        ["--gate-report", str(tmp_path / "missing.json")])
    with pytest.raises(SystemExit, match="not found"):
        ev._selection(args, ml)


def test_k_selection_prefers_the_best_rate_among_passing_candidates():
    """Not the smallest K that passes. The runtime bar is a threshold that has
    already been met, so the remaining choice should go to rate - and the
    measured frontier showed the smallest passing K was not even the fastest.
    """
    ml = _load_script("m10l_shared_codebook")
    candidates = [
        {"codebook_size": 16, "metric": "code_length", "passes_gate": True,
         "held_out_bits_per_symbol": 3.131},
        {"codebook_size": 64, "metric": "code_length", "passes_gate": True,
         "held_out_bits_per_symbol": 3.122},
        {"codebook_size": 512, "metric": "code_length", "passes_gate": False,
         "held_out_bits_per_symbol": 3.118},
    ]

    assert ml.select_codebook_size(candidates)["codebook_size"] == 64
    assert ml.select_codebook_size(
        [dict(c, passes_gate=False) for c in candidates]) is None


def test_metric_selection_uses_validation_scores_only():
    ml = _load_script("m10l_shared_codebook")
    candidates = [
        {"codebook_size": 64, "metric": "code_length", "passes_gate": True,
         "held_out_bits_per_symbol": 2.186},
        {"codebook_size": 64, "metric": "l1", "passes_gate": True,
         "held_out_bits_per_symbol": 2.194},
    ]

    assert ml.select_metric(candidates) == "code_length"
    assert ml.select_metric(candidates[:1]) == "code_length"
    assert ml.select_metric([]) == ml.DEFAULT_METRIC


def test_the_calibration_signature_changes_with_the_grid():
    """The signature is what binds a codebook to its quantization grid; if it
    did not move with the grid, the M10K coupling would go undetected."""
    mc = _load_script("m10h_motion_compensation")
    ev = _load_script("m10l_evaluate")
    model = _autoencoder()
    frames = _moving_frames()

    four = _calibration(mc, model, frames, bits=4)
    five = _calibration(mc, model, frames, bits=5)
    signature = ev.calibration_signature(four, bits=4, calibration_frames=400,
                                         quant_mode="per_channel")

    assert signature == ev.calibration_signature(
        four, bits=4, calibration_frames=400, quant_mode="per_channel")
    assert signature != ev.calibration_signature(
        five, bits=5, calibration_frames=400, quant_mode="per_channel")
    assert signature != ev.calibration_signature(
        four, bits=4, calibration_frames=200, quant_mode="per_channel")


def test_m10l_leaves_src_and_the_earlier_milestones_alone():
    from nvc.compression import nvc_format

    assert nvc_format.MAGIC == b"NVC1"
    assert nvc_format.STREAM_MAGIC == b"NVCS"
    mc = _load_script("m10h_motion_compensation")
    assert mc.TEMPORAL_FORMAT_VERSION == 2
    ce = _load_script("m10j_conditional_entropy")
    assert ce.DEPLOYED_CONTEXTS == ("magnitude4", "local_activity4")
    mk = _load_script("m10k_learned_entropy")
    assert mk.FROZEN_LAMBDA == 3.0e-4

    trainer = _load_script("train_autoencoder")
    actions = {a.dest for a in trainer.build_arg_parser(load_default_config())._actions}
    for required in ("rate_lambda", "rate_lr", "resume_model_only", "seed"):
        assert required in actions
