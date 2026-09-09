"""Tests for deploying the M10K learned entropy model through the real codec.

M10K's claim is that only the probability model changes. These tests hold that
to the letter: the residual symbols, the motion payload and the reconstruction
must come out identical to M10H's and M10J's, and the only permitted difference
is the entropy-coded residual byte count.

One property here is specific to a learned model and was found by running it:
the model is fitted to symbols produced under a particular quantization
calibration, and handing it symbols from a *different* calibration silently
costs bits rather than failing. That coupling is tested explicitly so it cannot
be forgotten at deployment time.
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


def _arms(mc, ce, mk, calibration, *, bits=4, channels=4):
    arms = {}
    for scheme in ("marginal", "local_activity4"):
        context_model = ce.fit_context_model(scheme, calibration["references"])
        built = ce.build_conditional_entropy_model(
            calibration["symbols"], calibration["references"], context_model, bits=bits)
        arms[scheme] = {"context_model": context_model,
                        "entropy_model": built["entropy_model"],
                        "identity": built["entropy_model"].model_id()}
    torch.manual_seed(3)
    learned = mk.build_model({"latent_channels": channels, "alphabet": 2 ** bits, "hidden": 8})
    learned.eval()
    arms["learned"] = {"model": learned, "identity": b"\x9a" * 8}
    return arms


def _setup(bits: int = 4):
    mc = _load_script("m10h_motion_compensation")
    ce = _load_script("m10j_conditional_entropy")
    mk = _load_script("m10k_learned_entropy")
    ev = _load_script("m10k_evaluate")
    model = _autoencoder()
    frames = _moving_frames()
    calibration = _calibration(mc, model, frames, bits=bits)
    arms = _arms(mc, ce, mk, calibration, bits=bits)
    return mc, ce, mk, ev, model, frames, calibration, arms


def _encode(ev, mc, mk, model, frames, arms, tmp_path, calibration, *, bits=4, gop_size=4):
    paths = {arm: tmp_path / f"{arm}.nvct" for arm in arms}
    result = ev.encode_multi(
        mc, mk, model, frames, arms, paths,
        intra_params=calibration["intra_params"],
        intra_entropy_model=calibration["intra_entropy_model"],
        residual_params=calibration["residual_params"],
        motion_entropy_model=calibration["motion_entropy_model"],
        bits=bits, gop_size=gop_size, block_size=16, search_range=8)
    return result, paths


# --- only the probability model may differ ---------------------------------------


def test_every_arm_shares_one_motion_payload(tmp_path):
    mc, ce, mk, ev, model, frames, calibration, arms = _setup()

    result, paths = _encode(ev, mc, mk, model, frames, arms, tmp_path, calibration)

    motions = {arm: result["arms"][arm]["motion_bytes"] for arm in arms}
    assert len(set(motions.values())) == 1, f"motion differed: {motions}"
    payloads = {arm: [m for _, m, _ in mc.TemporalStreamReader(paths[arm])] for arm in arms}
    for arm in arms:
        assert payloads[arm] == payloads["marginal"], f"{arm} motion payload differs"


def test_the_learned_arm_reproduces_the_same_symbols_and_reconstruction(tmp_path):
    mc, ce, mk, ev, model, frames, calibration, arms = _setup()

    result, paths = _encode(ev, mc, mk, model, frames, arms, tmp_path, calibration)

    for arm in arms:
        decoded, symbols = ev.decode_sequence(
            mc, mk, model, paths[arm], arm, arms[arm],
            intra_entropy_model=calibration["intra_entropy_model"],
            motion_entropy_model=calibration["motion_entropy_model"], bits=4)
        assert torch.equal(decoded.cpu(), result["reconstructions"]), f"{arm} reconstruction"
        for produced, returned in zip(result["symbols"], symbols):
            assert np.array_equal(produced.reshape(-1), returned.reshape(-1)), f"{arm} symbols"


def test_only_the_residual_byte_count_differs_between_arms(tmp_path):
    mc, ce, mk, ev, model, frames, calibration, arms = _setup()

    result, _ = _encode(ev, mc, mk, model, frames, arms, tmp_path, calibration)

    stats = {arm: result["arms"][arm] for arm in arms}
    # I-frames are coded by the intra path, untouched by the residual model.
    assert len({s["i_frame_residual_bytes"] for s in stats.values()}) == 1
    assert len({s["motion_bytes"] for s in stats.values()}) == 1
    assert len({s["container_overhead_bytes"] for s in stats.values()}) == 1
    # ...and the P-frame residual is the one thing that may move.
    assert len({s["p_frame_residual_bytes"] for s in stats.values()}) > 1


def test_byte_accounting_closes_for_the_learned_arm(tmp_path):
    mc, ce, mk, ev, model, frames, calibration, arms = _setup()

    result, paths = _encode(ev, mc, mk, model, frames, arms, tmp_path, calibration)

    stats = result["arms"]["learned"]
    on_disk = paths["learned"].stat().st_size
    assert stats["container_bytes"] == on_disk
    assert (stats["motion_bytes"] + stats["residual_bytes"]
            + stats["container_overhead_bytes"]) == on_disk


@pytest.mark.parametrize("bits", [5, 4, 3])
def test_all_three_rate_points_round_trip(bits, tmp_path):
    mc, ce, mk, ev, model, frames, calibration, arms = _setup(bits=bits)

    result, paths = _encode(ev, mc, mk, model, frames, arms, tmp_path, calibration, bits=bits)

    decoded, _ = ev.decode_sequence(
        mc, mk, model, paths["learned"], "learned", arms["learned"],
        intra_entropy_model=calibration["intra_entropy_model"],
        motion_entropy_model=calibration["motion_entropy_model"], bits=bits)
    assert torch.equal(decoded.cpu(), result["reconstructions"])


# --- causality and determinism ----------------------------------------------------


def test_decoding_is_deterministic_and_uses_only_the_stream(tmp_path):
    mc, ce, mk, ev, model, frames, calibration, arms = _setup()
    _, paths = _encode(ev, mc, mk, model, frames, arms, tmp_path, calibration)

    def decode():
        return ev.decode_sequence(
            mc, mk, model, paths["learned"], "learned", arms["learned"],
            intra_entropy_model=calibration["intra_entropy_model"],
            motion_entropy_model=calibration["motion_entropy_model"], bits=4)[0]

    first = decode()
    del frames
    assert torch.equal(first, decode())


def test_motion_is_estimated_only_against_the_reconstruction(tmp_path, monkeypatch):
    mc, ce, mk, ev, model, frames, calibration, arms = _setup()
    seen = []
    original = mc.estimate_block_motion

    def recording(reference, current, **kwargs):
        seen.append((reference.detach().clone(), current.detach().clone()))
        return original(reference, current, **kwargs)

    monkeypatch.setattr(mc, "estimate_block_motion", recording)
    _encode(ev, mc, mk, model, frames, arms, tmp_path, calibration)

    assert seen
    for reference, current in seen:
        for index in range(frames.shape[0]):
            assert not torch.equal(reference, frames[index:index + 1])
        assert any(torch.equal(current, frames[i:i + 1]) for i in range(frames.shape[0]))


def test_i_frames_carry_no_motion_and_sequences_start_with_one(tmp_path):
    mc, ce, mk, ev, model, frames, calibration, arms = _setup()

    _, paths = _encode(ev, mc, mk, model, frames, arms, tmp_path, calibration)

    for arm in arms:
        frame_type, motion, _ = next(iter(mc.TemporalStreamReader(paths[arm])))
        assert frame_type == mc.FRAME_TYPE_I and motion == b""


def test_sequence_boundaries_reset_reference_state(tmp_path):
    mc, ce, mk, ev, model, _, _, _ = _setup()
    for index, seed in enumerate((0, 7)):
        frames = _moving_frames(8, seed=seed)
        calibration = _calibration(mc, model, frames)
        arms = _arms(mc, _load_script("m10j_conditional_entropy"), mk, calibration)
        directory = tmp_path / f"seq{index}"
        directory.mkdir()
        _, paths = _encode(ev, mc, mk, model, frames, arms, directory, calibration)
        for arm in arms:
            frame_type, motion, _ = next(iter(mc.TemporalStreamReader(paths[arm])))
            assert frame_type == mc.FRAME_TYPE_I and motion == b""


# --- stream and model compatibility -----------------------------------------------


def test_a_stream_declaring_a_different_model_is_rejected(tmp_path):
    mc, ce, mk, ev, model, frames, calibration, arms = _setup()
    _, paths = _encode(ev, mc, mk, model, frames, arms, tmp_path, calibration)

    wrong = dict(arms["learned"])
    wrong["identity"] = b"\x00" * 8
    with pytest.raises(mc.TemporalFormatError, match="residual entropy model mismatch"):
        ev.decode_sequence(
            mc, mk, model, paths["learned"], "learned", wrong,
            intra_entropy_model=calibration["intra_entropy_model"],
            motion_entropy_model=calibration["motion_entropy_model"], bits=4)


def test_m10h_and_m10j_streams_are_unaffected(tmp_path):
    """M10K existing must not change how the earlier arms decode."""
    mc, ce, mk, ev, model, frames, calibration, arms = _setup()

    result, paths = _encode(ev, mc, mk, model, frames, arms, tmp_path, calibration)

    for arm in ("marginal", "local_activity4"):
        decoded, _ = ev.decode_sequence(
            mc, mk, model, paths[arm], arm, arms[arm],
            intra_entropy_model=calibration["intra_entropy_model"],
            motion_entropy_model=calibration["motion_entropy_model"], bits=4)
        assert torch.equal(decoded.cpu(), result["reconstructions"])
        assert mc.TemporalStreamReader(paths[arm]).header.format_version == 2


def test_a_truncated_residual_payload_is_rejected(tmp_path):
    mc, ce, mk, ev, model, frames, calibration, arms = _setup()
    _, paths = _encode(ev, mc, mk, model, frames, arms, tmp_path, calibration)
    path = paths["learned"]
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 60])

    with pytest.raises(mc.TemporalFormatError, match="Truncated"):
        ev.decode_sequence(
            mc, mk, model, path, "learned", arms["learned"],
            intra_entropy_model=calibration["intra_entropy_model"],
            motion_entropy_model=calibration["motion_entropy_model"], bits=4)


def test_the_learned_model_is_coupled_to_the_calibration_it_was_fitted_under():
    """Found by running it: a learned model scored against symbols from a
    DIFFERENT quantization grid silently costs bits instead of failing. The
    deployment therefore has to pin the calibration, and this test records why.
    """
    mc = _load_script("m10h_motion_compensation")
    mk = _load_script("m10k_learned_entropy")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    references = [rng.normal(size=(4, 8, 8)).astype(np.float32) for _ in range(40)]
    # Symbols under grid A: driven by the sign of the reference.
    symbols_a = [np.where(r > 0, 3, 11).astype(np.int64) for r in references]
    # Symbols under grid B: a different mapping, as a coarser grid would give.
    symbols_b = [np.where(r > 0, 5, 9).astype(np.int64) for r in references]

    model = mk.build_model({"latent_channels": 4, "alphabet": 16, "hidden": 8})
    mk.train_entropy_model(model, (symbols_a[:32], references[:32]),
                           (symbols_a[32:], references[32:]),
                           epochs=8, batch_size=4, learning_rate=1e-2,
                           device=torch.device("cpu"))

    matched = mk.cross_entropy_bits(model, symbols_a[32:], references[32:],
                                    device=torch.device("cpu"))
    mismatched = mk.cross_entropy_bits(model, symbols_b[32:], references[32:],
                                       device=torch.device("cpu"))

    assert mismatched > matched, (
        "a mismatched calibration must cost bits - if it did not, the model was "
        "not actually using the symbol distribution it was fitted to")


def test_m10k_leaves_src_and_the_earlier_milestones_alone():
    from nvc.compression import nvc_format

    assert nvc_format.MAGIC == b"NVC1"
    assert nvc_format.STREAM_MAGIC == b"NVCS"
    mc = _load_script("m10h_motion_compensation")
    assert mc.TEMPORAL_FORMAT_VERSION == 2
    ce = _load_script("m10j_conditional_entropy")
    assert ce.DEPLOYED_CONTEXTS == ("magnitude4", "local_activity4")

    trainer = _load_script("train_autoencoder")
    actions = {a.dest for a in trainer.build_arg_parser(load_default_config())._actions}
    for required in ("rate_lambda", "rate_lr", "resume_model_only", "seed"):
        assert required in actions
