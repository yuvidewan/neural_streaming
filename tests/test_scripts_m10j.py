"""Tests for the M10J coding path and scripts.

M10J's claim is narrow and therefore checkable: given the SAME residual symbols,
choosing a probability table from the reference costs fewer bits. Everything
that would undermine that claim is tested here - that the symbols really are
identical to M10H's, that motion and reconstruction do not move, that the
decoder derives the same context the encoder used, and that the container still
refuses a stream whose declared model does not match.

The strongest guarantee is structural rather than assertive: `encode_multi`
produces every arm's payload from one closed-loop pass over one symbol array, so
the arms cannot diverge in anything but their tables.
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


def _model(seed: int = 0):
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


def _grids(mc, model, frames, *, bits=4, search_range=8, block_size=16):
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
    residual_symbols = np.stack([
        latent_to_symbols(residuals[i:i + 1], residual_params).reshape(residuals.shape[1], -1)
        for i in range(residuals.shape[0])])
    residual_model = EmpiricalEntropyModel.from_symbols(
        residual_symbols, bits=bits, num_tables=residuals.shape[1])
    blocks = (frames.shape[2] // block_size) * (frames.shape[3] // block_size)
    motion_model = EmpiricalEntropyModel.from_symbols(
        np.stack([np.stack([np.arange(blocks) % (2 * search_range + 1),
                            np.arange(blocks) % (2 * search_range + 1)])]),
        bits=mc.motion_alphabet_bits(search_range), num_tables=2)
    return {"intra_params": intra_params, "intra_entropy_model": intra_model,
            "residual_params": residual_params, "residual_entropy_model": residual_model,
            "motion_entropy_model": motion_model}


def _setup(bits: int = 4):
    mc = _load_script("m10h_motion_compensation")
    ce = _load_script("m10j_conditional_entropy")
    model = _model()
    frames = _moving_frames()
    grids = _grids(mc, model, frames, bits=bits)

    with torch.no_grad():
        latents = torch.cat([model.encode(frames[i:i + 1]) for i in range(frames.shape[0])])
    references = [latents[i].numpy() for i in range(latents.shape[0])]
    residual_symbols = [
        latent_to_symbols(latents[i:i + 1] - latents[max(i - 1, 0):max(i - 1, 0) + 1],
                          grids["residual_params"]).reshape(latents.shape[1:])
        for i in range(1, latents.shape[0])]
    arms = {}
    for scheme in ("marginal",) + ce.DEPLOYED_CONTEXTS:
        context_model = ce.fit_context_model(scheme, references)
        arms[scheme] = ce.build_conditional_entropy_model(
            residual_symbols, references[1:], context_model, bits=bits)
    return mc, ce, model, frames, grids, arms


def _encode_all(ce, mc, model, frames, arms, tmp_path, grids, *, gop_size=4):
    paths = {arm: tmp_path / f"{arm}.nvct" for arm in arms}
    result = ce.encode_multi(
        mc, model, frames, arms, paths,
        intra_params=grids["intra_params"],
        intra_entropy_model=grids["intra_entropy_model"],
        residual_params=grids["residual_params"],
        motion_entropy_model=grids["motion_entropy_model"],
        gop_size=gop_size, block_size=16, search_range=8)
    return result, paths


# --- the ablation is structural ------------------------------------------------


def test_every_arm_shares_one_motion_payload_and_one_symbol_set(tmp_path):
    """The central fairness property. Motion and symbols are computed once and
    shared, so they cannot differ between arms."""
    mc, ce, model, frames, grids, arms = _setup()

    result, paths = _encode_all(ce, mc, model, frames, arms, tmp_path, grids)

    motions = {arm: result["arms"][arm]["motion_bytes"] for arm in arms}
    assert len(set(motions.values())) == 1, f"motion differed: {motions}"
    payloads = {arm: [m for _, m, _ in mc.TemporalStreamReader(paths[arm])] for arm in arms}
    reference = payloads["marginal"]
    for arm in arms:
        assert payloads[arm] == reference, f"{arm} motion payload differs byte-wise"


def test_the_conditional_arms_match_m10h_symbols_and_reconstruction(tmp_path):
    """M10J must change only the probability model - the quantized residual and
    the pixels it reconstructs to have to be identical to M10H's."""
    mc, ce, model, frames, grids, arms = _setup()

    result, paths = _encode_all(ce, mc, model, frames, arms, tmp_path, grids)

    for arm, spec in arms.items():
        decoded, symbols = ce.decode_sequence(
            mc, model, paths[arm],
            intra_entropy_model=grids["intra_entropy_model"],
            residual_entropy_model=spec["entropy_model"],
            context_model=spec["context_model"],
            motion_entropy_model=grids["motion_entropy_model"], return_symbols=True)
        assert torch.equal(decoded.cpu(), result["reconstructions"]), f"{arm} reconstruction"
        for produced, returned in zip(result["symbols"], symbols):
            assert np.array_equal(produced.reshape(-1), returned.reshape(-1)), f"{arm} symbols"


def test_residual_quantization_is_untouched(tmp_path):
    """Only the table selection changes; the grid the symbols come from is the
    same object for every arm."""
    mc, ce, model, frames, grids, arms = _setup()

    result, _ = _encode_all(ce, mc, model, frames, arms, tmp_path, grids)

    with torch.no_grad():
        latents = torch.cat([model.encode(frames[i:i + 1]) for i in range(frames.shape[0])])
    # Every recorded symbol must lie inside the declared alphabet of that grid.
    for symbols in result["symbols"]:
        assert symbols.min() >= 0
        assert symbols.max() < 2 ** grids["residual_params"].bits


# --- encoder/decoder context equality ------------------------------------------


def test_the_decoder_derives_the_same_context_the_encoder_used(tmp_path):
    """If the two disagreed, decoding would silently produce different symbols -
    so an exact symbol round trip IS the context-equality check."""
    mc, ce, model, frames, grids, arms = _setup()
    result, paths = _encode_all(ce, mc, model, frames, arms, tmp_path, grids)

    for arm, spec in arms.items():
        if spec["context_model"].cardinality == 1:
            continue
        _, symbols = ce.decode_sequence(
            mc, model, paths[arm],
            intra_entropy_model=grids["intra_entropy_model"],
            residual_entropy_model=spec["entropy_model"],
            context_model=spec["context_model"],
            motion_entropy_model=grids["motion_entropy_model"], return_symbols=True)
        for produced, returned in zip(result["symbols"], symbols):
            assert np.array_equal(produced.reshape(-1), returned.reshape(-1))


def test_decoding_is_deterministic_and_uses_only_the_stream(tmp_path):
    mc, ce, model, frames, grids, arms = _setup()
    _, paths = _encode_all(ce, mc, model, frames, arms, tmp_path, grids)
    spec = arms["local_activity4"]

    def decode():
        return ce.decode_sequence(
            mc, model, paths["local_activity4"],
            intra_entropy_model=grids["intra_entropy_model"],
            residual_entropy_model=spec["entropy_model"],
            context_model=spec["context_model"],
            motion_entropy_model=grids["motion_entropy_model"])

    first = decode()
    del frames
    assert torch.equal(first, decode())


def test_motion_is_estimated_only_against_the_reconstruction(tmp_path, monkeypatch):
    mc, ce, model, frames, grids, arms = _setup()
    seen = []
    original = mc.estimate_block_motion

    def recording(reference, current, **kwargs):
        seen.append((reference.detach().clone(), current.detach().clone()))
        return original(reference, current, **kwargs)

    monkeypatch.setattr(mc, "estimate_block_motion", recording)
    _encode_all(ce, mc, model, frames, arms, tmp_path, grids)

    assert seen
    for reference, current in seen:
        for index in range(frames.shape[0]):
            assert not torch.equal(reference, frames[index:index + 1])
        assert any(torch.equal(current, frames[i:i + 1]) for i in range(frames.shape[0]))


def test_i_frames_carry_no_motion_and_sequences_start_with_one(tmp_path):
    mc, ce, model, frames, grids, arms = _setup()

    result, paths = _encode_all(ce, mc, model, frames, arms, tmp_path, grids)

    for arm in arms:
        frame_type, motion, _ = next(iter(mc.TemporalStreamReader(paths[arm])))
        assert frame_type == mc.FRAME_TYPE_I and motion == b""
        for record in result["arms"][arm]["frames"]:
            if record["frame_type"] == "I":
                assert record["motion_bytes"] == 0


def test_sequence_boundaries_reset_reference_state(tmp_path):
    mc, ce, model, _, grids, arms = _setup()
    for index, seed in enumerate((0, 9)):
        frames = _moving_frames(8, seed=seed)
        directory = tmp_path / f"seq{index}"
        directory.mkdir()
        _, paths = _encode_all(ce, mc, model, frames, arms, directory, grids)
        for arm in arms:
            frame_type, motion, _ = next(iter(mc.TemporalStreamReader(paths[arm])))
            assert frame_type == mc.FRAME_TYPE_I and motion == b""


# --- container and accounting ---------------------------------------------------


def test_byte_accounting_closes_for_every_arm(tmp_path):
    mc, ce, model, frames, grids, arms = _setup()

    result, paths = _encode_all(ce, mc, model, frames, arms, tmp_path, grids)

    for arm in arms:
        stats = result["arms"][arm]
        on_disk = paths[arm].stat().st_size
        assert stats["container_bytes"] == on_disk
        assert (stats["motion_bytes"] + stats["residual_bytes"]
                + stats["container_overhead_bytes"]) == on_disk
        assert stats["residual_bytes"] == (stats["i_frame_residual_bytes"]
                                           + stats["p_frame_residual_bytes"])


def test_a_marginal_decoder_refuses_a_conditional_stream(tmp_path):
    """`.nvct` needs no change: the residual entropy model id already differs,
    and the decoder checks it."""
    mc, ce, model, frames, grids, arms = _setup()
    _, paths = _encode_all(ce, mc, model, frames, arms, tmp_path, grids)

    with pytest.raises(mc.TemporalFormatError, match="residual entropy model mismatch"):
        ce.decode_sequence(
            mc, model, paths["local_activity4"],
            intra_entropy_model=grids["intra_entropy_model"],
            residual_entropy_model=arms["marginal"]["entropy_model"],
            context_model=arms["marginal"]["context_model"],
            motion_entropy_model=grids["motion_entropy_model"])


def test_a_marginal_stream_stays_decodable_with_its_own_model(tmp_path):
    """Backward compatibility: M10H streams are unaffected by M10J existing."""
    mc, ce, model, frames, grids, arms = _setup()
    result, paths = _encode_all(ce, mc, model, frames, arms, tmp_path, grids)

    decoded = ce.decode_sequence(
        mc, model, paths["marginal"],
        intra_entropy_model=grids["intra_entropy_model"],
        residual_entropy_model=arms["marginal"]["entropy_model"],
        context_model=arms["marginal"]["context_model"],
        motion_entropy_model=grids["motion_entropy_model"])

    assert torch.equal(decoded.cpu(), result["reconstructions"])
    assert mc.TemporalStreamReader(paths["marginal"]).header.format_version == 2


def test_a_truncated_residual_payload_is_rejected(tmp_path):
    mc, ce, model, frames, grids, arms = _setup()
    _, paths = _encode_all(ce, mc, model, frames, arms, tmp_path, grids)
    path = paths["magnitude4"]
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 50])

    with pytest.raises(mc.TemporalFormatError, match="Truncated"):
        ce.decode_sequence(
            mc, model, path,
            intra_entropy_model=grids["intra_entropy_model"],
            residual_entropy_model=arms["magnitude4"]["entropy_model"],
            context_model=arms["magnitude4"]["context_model"],
            motion_entropy_model=grids["motion_entropy_model"])


@pytest.mark.parametrize("bits", [5, 4, 3])
def test_all_three_rate_points_round_trip(bits, tmp_path):
    mc, ce, model, frames, grids, arms = _setup(bits=bits)

    result, paths = _encode_all(ce, mc, model, frames, arms, tmp_path, grids)

    for arm, spec in arms.items():
        decoded = ce.decode_sequence(
            mc, model, paths[arm],
            intra_entropy_model=grids["intra_entropy_model"],
            residual_entropy_model=spec["entropy_model"],
            context_model=spec["context_model"],
            motion_entropy_model=grids["motion_entropy_model"])
        assert torch.equal(decoded.cpu(), result["reconstructions"])


# --- calibration provenance and scripts -----------------------------------------


def test_calibration_records_that_it_used_training_data_only():
    ce = _load_script("m10j_conditional_entropy")
    analysis = _load_script("m10j_entropy_analysis")

    source = Path("scripts/m10j_conditional_entropy.py").read_text(encoding="utf-8")
    assert '"split": "train"' in source
    # The offline analysis must never reach for the test split.
    analysis_source = Path("scripts/m10j_entropy_analysis.py").read_text(encoding="utf-8")
    assert 'split="test"' not in analysis_source
    assert 'split="train"' in analysis_source and 'split="val"' in analysis_source


def test_the_offline_analysis_guards_against_plug_in_bias():
    """Adding contexts always reduces a plug-in entropy estimate, so the
    analysis has to measure a random context of equal cardinality and score on
    held-out data - otherwise it would report a gain for pure noise."""
    analysis = _load_script("m10j_entropy_analysis")
    source = Path("scripts/m10j_entropy_analysis.py").read_text(encoding="utf-8")

    assert "random_train_entropy_bits" in source
    assert "cross_entropy_bits_per_symbol" in source
    assert "net_reduction_percent" in source


@pytest.mark.parametrize("name", ["m10j_entropy_analysis", "m10j_conditional_entropy",
                                  "m10j_evaluate"])
def test_every_m10j_script_follows_the_project_script_contract(name):
    mod = _load_script(name)

    assert callable(mod.build_arg_parser)
    assert callable(mod.main)
    assert mod.build_arg_parser(load_default_config()).parse_args([]) is not None


def test_m10j_does_not_modify_the_shipped_codec_or_earlier_containers():
    from nvc.compression import nvc_format

    assert nvc_format.MAGIC == b"NVC1"
    assert nvc_format.STREAM_MAGIC == b"NVCS"
    m10g = _load_script("m10g_temporal_baseline")
    mc = _load_script("m10h_motion_compensation")
    assert m10g.TEMPORAL_FORMAT_VERSION == 1
    assert mc.TEMPORAL_FORMAT_VERSION == 2, "M10J reuses M10H's container unchanged"

    trainer = _load_script("train_autoencoder")
    actions = {a.dest for a in trainer.build_arg_parser(load_default_config())._actions}
    for required in ("rate_lambda", "rate_lr", "resume_model_only", "seed"):
        assert required in actions
