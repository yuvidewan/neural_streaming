"""Tests for deploying the M11 channel-autoregressive entropy model through the codec.

M11 may change only how residual symbols are entropy-coded. So on real `.nvct`
v2 streams, through the real coder, the residual symbols, the motion payload and
the reconstruction must come out identical to M10H/M10J/M10K/M10L's - and a
decoder given the wrong model, calibration, bit depth or context definition must
be stopped rather than left to decode garbage.
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


def _arms(calibration, *, bits=4, channels=4):
    ce = _load_script("m10j_conditional_entropy")
    mk = _load_script("m10k_learned_entropy")
    ml = _load_script("m10l_shared_codebook")
    ma = _load_script("m11_ar_entropy")
    cx = _load_script("m11_causal_context")
    arms = {}
    for scheme in ("marginal", "local_activity4"):
        context_model = ce.fit_context_model(scheme, calibration["references"])
        built = ce.build_conditional_entropy_model(
            calibration["symbols"], calibration["references"], context_model, bits=bits)
        arms[scheme] = {"context_model": context_model, "entropy_model": built["entropy_model"],
                        "identity": built["entropy_model"].model_id()}
    torch.manual_seed(3)
    m10k = mk.build_model({"latent_channels": channels, "alphabet": 2 ** bits,
                           "hidden": 8}).eval()
    arms["learned"] = {"model": m10k, "identity": b"\x9a" * 8}
    samples = ml.sample_training_distributions(
        m10k, calibration["references"], device=torch.device("cpu"), max_rows=2000)
    m10l = ml.fit_codebook(samples, 16, bits=bits)
    arms["codebook"] = {"model": m10k, "codebook": m10l,
                        "identity": m10l.codebook_id(model_identity=b"\x9a" * 8,
                                                     calibration_signature="test")}
    zero = torch.from_numpy(cx.zero_symbols(calibration["residual_params"], channels))
    for arm, group, use_codebook in (("m11", 1, False), ("m11_op", 2, True)):
        model = ma.from_m10k(m10k, group_size=group).eval()
        with torch.no_grad():
            model.features[0].weight[:, 1:] = 0.4       # context that actually matters
        codebook = ml.fit_codebook(samples, 16, bits=bits) if use_codebook else None
        arms[arm] = {"model": model, "zero": zero, "codebook": codebook,
                     "identity": ma.model_identity(model, m10k_identity=b"\x9a" * 8,
                                                   calibration_signature="test", bits=bits,
                                                   codebook=codebook)}
    return arms


def _setup(bits: int = 4, frames: int = 12):
    mc = _load_script("m10h_motion_compensation")
    model = _autoencoder()
    clip = _moving_frames(frames)
    calibration = _calibration(mc, model, _moving_frames(12), bits=bits)
    return mc, model, clip, calibration, _arms(calibration, bits=bits)


def _encode(model, frames, arms, tmp_path, calibration, *, bits=4, gop_size=4):
    mc = _load_script("m10h_motion_compensation")
    ev = _load_script("m11_evaluate")
    paths = {arm: tmp_path / f"{arm}.nvct" for arm in arms}
    result = ev.encode_multi(
        mc, _load_script("m10k_learned_entropy"), _load_script("m10l_shared_codebook"),
        _load_script("m11_ar_entropy"), model, frames, arms, paths,
        intra_params=calibration["intra_params"],
        intra_entropy_model=calibration["intra_entropy_model"],
        residual_params=calibration["residual_params"],
        motion_entropy_model=calibration["motion_entropy_model"],
        bits=bits, gop_size=gop_size, block_size=16, search_range=8)
    return result, paths


def _decode(model, path, arm, spec, calibration, *, bits=4):
    mc = _load_script("m10h_motion_compensation")
    ev = _load_script("m11_evaluate")
    return ev.decode_sequence(
        mc, _load_script("m10k_learned_entropy"), _load_script("m10l_shared_codebook"),
        _load_script("m11_ar_entropy"), model, path, arm, spec,
        intra_entropy_model=calibration["intra_entropy_model"],
        motion_entropy_model=calibration["motion_entropy_model"], bits=bits)


# --- only the entropy model may differ ---------------------------------------------------


def test_every_arm_shares_one_motion_payload(tmp_path):
    mc, model, frames, calibration, arms = _setup()
    result, paths = _encode(model, frames, arms, tmp_path, calibration)

    assert len({result["arms"][a]["motion_bytes"] for a in arms}) == 1
    payloads = {a: [m for _, m, _ in mc.TemporalStreamReader(paths[a])] for a in arms}
    for arm in arms:
        assert payloads[arm] == payloads["marginal"], arm


@pytest.mark.parametrize("arm", ["m11", "m11_op"])
def test_m11_arms_reproduce_the_symbols_and_reconstruction(tmp_path, arm):
    """The decisive invariant: M11 may change bytes, never pictures."""
    mc, model, frames, calibration, arms = _setup()
    result, paths = _encode(model, frames, arms, tmp_path, calibration)
    decoded, symbols, timings = _decode(model, paths[arm], arm, arms[arm], calibration)

    assert len(symbols) == len(result["symbols"]) > 0
    for coded, back in zip(result["symbols"], symbols):
        assert np.array_equal(coded.reshape(-1), back.reshape(-1))
    assert torch.equal(decoded.cpu(), result["reconstructions"])
    assert set(timings) == {"network", "tables", "coder"}


def test_only_the_residual_byte_count_differs(tmp_path):
    mc, model, frames, calibration, arms = _setup()
    result, _ = _encode(model, frames, arms, tmp_path, calibration)
    for arm in arms:
        assert result["arms"][arm]["i_frame_residual_bytes"] == \
            result["arms"]["marginal"]["i_frame_residual_bytes"]
        assert result["arms"][arm]["p_frames"] == result["arms"]["marginal"]["p_frames"]


def test_byte_accounting_closes_for_the_m11_arms(tmp_path):
    mc, model, frames, calibration, arms = _setup()
    result, paths = _encode(model, frames, arms, tmp_path, calibration)
    for arm in ("m11", "m11_op"):
        stats = result["arms"][arm]
        assert stats["motion_bytes"] + stats["residual_bytes"] \
            + stats["container_overhead_bytes"] == paths[arm].stat().st_size


@pytest.mark.parametrize("bits", [3, 4, 5])
def test_all_three_rate_points_round_trip(tmp_path, bits):
    mc, model, frames, calibration, arms = _setup(bits=bits)
    result, paths = _encode(model, frames, arms, tmp_path, calibration, bits=bits)
    for arm in ("m11", "m11_op"):
        decoded, symbols, _ = _decode(model, paths[arm], arm, arms[arm], calibration, bits=bits)
        for coded, back in zip(result["symbols"], symbols):
            assert np.array_equal(coded.reshape(-1), back.reshape(-1))
        assert torch.equal(decoded.cpu(), result["reconstructions"])


def test_m10h_m10j_m10k_and_m10l_are_unaffected(tmp_path):
    mc, model, frames, calibration, arms = _setup()
    result, paths = _encode(model, frames, arms, tmp_path, calibration)
    for arm in ("marginal", "local_activity4", "learned", "codebook"):
        decoded, symbols, _ = _decode(model, paths[arm], arm, arms[arm], calibration)
        assert torch.equal(decoded.cpu(), result["reconstructions"]), arm


# --- short sequences and boundaries ---------------------------------------------------------


@pytest.mark.parametrize("frames,gop", [(1, 4), (2, 4), (5, 4), (9, 4)])
def test_short_sequences_and_gop_boundaries(tmp_path, frames, gop):
    """One frame (I only, no residual symbols at all), one P-frame, and clips
    ending just after a GOP boundary."""
    mc, model, clip, calibration, arms = _setup(frames=frames)
    result, paths = _encode(model, clip, arms, tmp_path, calibration, gop_size=gop)
    decoded, symbols, _ = _decode(model, paths["m11_op"], "m11_op", arms["m11_op"], calibration)

    assert len(symbols) == len(result["symbols"])
    assert torch.equal(decoded.cpu(), result["reconstructions"])
    assert decoded.shape[0] == frames


def test_repeated_encodes_produce_byte_identical_streams(tmp_path):
    mc, model, frames, calibration, arms = _setup()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, first = _encode(model, frames, arms, tmp_path / "a", calibration)
    _, second = _encode(model, frames, arms, tmp_path / "b", calibration)
    for arm in ("m11", "m11_op"):
        assert first[arm].read_bytes() == second[arm].read_bytes(), arm


# --- stream identity, provenance and the container ------------------------------------------


def test_a_stream_declaring_a_different_model_is_rejected(tmp_path):
    mc, model, frames, calibration, arms = _setup()
    _, paths = _encode(model, frames, arms, tmp_path, calibration)
    stale = dict(arms["m11"], identity=b"\x00" * 8)
    # _decode loads its own copy of the M10H module, so the exception CLASS is a
    # different object from `mc.TemporalFormatError`; match it by name instead.
    with pytest.raises(Exception, match="entropy model mismatch") as raised:
        _decode(model, paths["m11"], "m11", stale, calibration)
    assert type(raised.value).__name__ == "TemporalFormatError"


def test_the_four_temporal_entropy_models_and_m11_have_distinct_identities():
    mc, model, frames, calibration, arms = _setup()
    identities = [arms[a]["identity"] for a in arms]
    assert len(set(identities)) == len(identities)
    assert all(len(i) == 8 for i in identities), "must fit .nvct v2's 8-byte field"


def _checkpoint(ma, *, bits=4, group=1, signature="cal", m10k=b"\x01" * 8):
    return {"calibration_signature": signature, "bits": bits,
            "model_config": {"group_size": group},
            "context_definition_id": ma.context_definition_id(group),
            "m10k_identity": m10k.hex()}


def test_provenance_accepts_the_matching_model():
    ma = _load_script("m11_ar_entropy")
    ev = _load_script("m11_evaluate")
    ev.check_provenance(_checkpoint(ma), signature="cal", bits=4, group_size=1,
                        context_definition_id=ma.context_definition_id(1),
                        m10k_identity=b"\x01" * 8)


@pytest.mark.parametrize("mismatch", ["calibration", "bits", "group", "context", "m10k"])
def test_provenance_rejects_every_kind_of_mismatch(mismatch):
    """Wrong calibration, wrong bit depth, wrong grouping, wrong context
    definition, wrong base model: each is a hard stop, never a fallback."""
    ma = _load_script("m11_ar_entropy")
    ev = _load_script("m11_evaluate")
    checkpoint = _checkpoint(ma)
    expected = dict(signature="cal", bits=4, group_size=1,
                    context_definition_id=ma.context_definition_id(1),
                    m10k_identity=b"\x01" * 8)
    if mismatch == "calibration":
        expected["signature"] = "other"
    elif mismatch == "bits":
        expected["bits"] = 5
    elif mismatch == "group":
        checkpoint["model_config"]["group_size"] = 8
    elif mismatch == "context":
        checkpoint["context_definition_id"] = ma.context_definition_id(8)
    else:
        expected["m10k_identity"] = b"\x02" * 8
    with pytest.raises(ev.ProvenanceError, match="provenance mismatch"):
        ev.check_provenance(checkpoint, **expected)


def test_the_container_format_is_unchanged(tmp_path):
    mc, model, frames, calibration, arms = _setup()
    _, paths = _encode(model, frames, arms, tmp_path, calibration)
    header = mc.TemporalStreamReader(paths["m11"]).header
    assert mc.TEMPORAL_FORMAT_VERSION == 2
    assert header.residual_entropy_model_id == arms["m11"]["identity"]


def test_the_operating_point_is_read_from_the_training_report(tmp_path):
    ev = _load_script("m11_evaluate")
    report = tmp_path / "m11_training.json"
    report.write_text('{"operating_point_group_size": 16}', encoding="utf-8")
    args = ev.build_arg_parser(load_default_config()).parse_args(
        ["--training-report", str(report)])
    assert ev._operating_point(args) == 16
    args = ev.build_arg_parser(load_default_config()).parse_args(
        ["--training-report", str(tmp_path / "missing.json")])
    with pytest.raises(SystemExit, match="not found"):
        ev._operating_point(args)


def test_m11_leaves_src_and_the_earlier_milestones_alone():
    from nvc.compression import nvc_format

    assert nvc_format.MAGIC == b"NVC1"
    assert nvc_format.STREAM_MAGIC == b"NVCS"
    assert _load_script("m10h_motion_compensation").TEMPORAL_FORMAT_VERSION == 2
    assert _load_script("m10j_conditional_entropy").DEPLOYED_CONTEXTS == (
        "magnitude4", "local_activity4")
    assert _load_script("m10k_learned_entropy").FROZEN_LAMBDA == 3.0e-4
    assert _load_script("m10l_shared_codebook").CANDIDATE_K == (16, 32, 64, 128, 256, 512)
