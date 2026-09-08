"""Tests for the M10I coding path and evaluation scripts.

The experiment's validity rests on one structural claim: M10H and M10I differ
ONLY in the residual model. If the conditional path perturbed motion estimation
in any way - a different reconstruction feeding the estimator, a different
search, a different quantization of the vectors - then a change in total bitrate
could not be attributed to conditioning at all.

So the load-bearing test here is that both paths, run on the same frames with the
same calibration, emit **byte-identical motion payloads**. The rest cover the
causal invariants and the container's refusal to decode a malformed or
mismatched stream.
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


def _moving_frames(count: int = 8, size: int = 64, *, seed: int = 0, step: int = 2):
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 3, size, size, generator=generator)
    return torch.cat([torch.roll(base, shifts=(index * step, index * step), dims=(2, 3))
                      for index in range(count)], dim=0)


def _grids(mc, model, frames, *, bits: int = 8, search_range: int = 8, block_size: int = 16):
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


def _setup(seed: int = 0):
    mc = _load_script("m10h_motion_compensation")
    m10i = _load_script("m10i_conditional_residual")
    model = _model(seed)
    frames = _moving_frames()
    grids = _grids(mc, model, frames)
    codec = m10i.build_codec({"latent_channels": TINY["latent_channels"], "hidden_channels": 8})
    codec.eval()
    return mc, m10i, model, frames, grids, codec


def _encode_conditional(m10i, mc, model, codec, frames, path, grids, **kwargs):
    return m10i.encode_sequence_conditional(
        mc, model, codec, frames, path,
        intra_params=grids["intra_params"], intra_entropy_model=grids["intra_entropy_model"],
        residual_params=grids["residual_params"],
        residual_entropy_model=grids["residual_entropy_model"],
        motion_entropy_model=grids["motion_entropy_model"],
        gop_size=kwargs.get("gop_size", 4), block_size=16, search_range=8)


def _decode_conditional(m10i, mc, model, codec, path, grids, **kwargs):
    return m10i.decode_sequence_conditional(
        mc, model, codec, path,
        intra_entropy_model=grids["intra_entropy_model"],
        residual_entropy_model=grids["residual_entropy_model"],
        motion_entropy_model=grids["motion_entropy_model"], **kwargs)


# --- the ablation must not be confounded -------------------------------------


def test_motion_payload_is_byte_identical_between_the_marginal_and_conditional_paths(tmp_path):
    """The load-bearing test for the whole experiment. If motion differed, a
    change in total bitrate could not be attributed to the residual model."""
    mc, m10i, model, frames, grids, codec = _setup()

    baseline = mc.encode_sequence(
        model, frames, tmp_path / "m10h.nvct",
        intra_params=grids["intra_params"], intra_entropy_model=grids["intra_entropy_model"],
        residual_params=grids["residual_params"],
        residual_entropy_model=grids["residual_entropy_model"],
        motion_entropy_model=grids["motion_entropy_model"],
        mode="mc", gop_size=4, block_size=16, search_range=8)
    conditional = _encode_conditional(
        m10i, mc, model, codec, frames, tmp_path / "m10i.nvct", grids)

    assert conditional["motion_bytes"] == baseline["motion_bytes"]
    assert [f["motion_bytes"] for f in conditional["frames"]] == \
           [f["motion_bytes"] for f in baseline["frames"]]

    baseline_motion = [m for _, m, _ in mc.TemporalStreamReader(tmp_path / "m10h.nvct")]
    conditional_motion = [m for _, m, _ in mc.TemporalStreamReader(tmp_path / "m10i.nvct")]
    assert conditional_motion == baseline_motion, "motion payloads must be byte-identical"


def test_at_initialisation_the_conditional_path_reproduces_the_marginal_one(tmp_path):
    """The codec is the identity at init, so M10I must produce exactly M10H's
    stream. That is what makes the trained comparison an ablation."""
    mc, m10i, model, frames, grids, codec = _setup()

    baseline = mc.encode_sequence(
        model, frames, tmp_path / "m10h.nvct",
        intra_params=grids["intra_params"], intra_entropy_model=grids["intra_entropy_model"],
        residual_params=grids["residual_params"],
        residual_entropy_model=grids["residual_entropy_model"],
        motion_entropy_model=grids["motion_entropy_model"],
        mode="mc", gop_size=4, block_size=16, search_range=8)
    conditional = _encode_conditional(
        m10i, mc, model, codec, frames, tmp_path / "m10i.nvct", grids)

    assert conditional["residual_bytes"] == baseline["residual_bytes"]
    assert (tmp_path / "m10i.nvct").read_bytes() == (tmp_path / "m10h.nvct").read_bytes()


def test_a_trained_codec_changes_the_residual_but_not_the_motion(tmp_path):
    mc, m10i, model, frames, grids, codec = _setup()
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for branch in (codec.analysis, codec.synthesis):
            branch[-1].weight.normal_(0.0, 0.02, generator=generator)

    baseline = mc.encode_sequence(
        model, frames, tmp_path / "m10h.nvct",
        intra_params=grids["intra_params"], intra_entropy_model=grids["intra_entropy_model"],
        residual_params=grids["residual_params"],
        residual_entropy_model=grids["residual_entropy_model"],
        motion_entropy_model=grids["motion_entropy_model"],
        mode="mc", gop_size=4, block_size=16, search_range=8)
    conditional = _encode_conditional(
        m10i, mc, model, codec, frames, tmp_path / "m10i.nvct", grids)

    assert conditional["motion_bytes"] == baseline["motion_bytes"]
    assert conditional["p_frame_residual_bytes"] != baseline["p_frame_residual_bytes"]
    assert conditional["i_frame_residual_bytes"] == baseline["i_frame_residual_bytes"], \
        "the I-frame path is untouched by the residual model"


# --- causality ----------------------------------------------------------------


def test_encoder_and_decoder_reach_the_same_reference_state(tmp_path):
    mc, m10i, model, frames, grids, codec = _setup()
    path = tmp_path / "seq.nvct"

    encoded = _encode_conditional(m10i, mc, model, codec, frames, path, grids)
    _, decoded_latents = _decode_conditional(
        m10i, mc, model, codec, path, grids, return_latents=True)

    assert torch.equal(encoded["encoder_latents"], decoded_latents)


def test_the_decoder_reproduces_the_encoders_reconstruction(tmp_path):
    mc, m10i, model, frames, grids, codec = _setup()
    path = tmp_path / "seq.nvct"

    encoded = _encode_conditional(m10i, mc, model, codec, frames, path, grids)
    decoded = _decode_conditional(m10i, mc, model, codec, path, grids)

    assert torch.equal(encoded["encoder_reconstructions"], decoded)


def test_decoding_is_deterministic_and_uses_only_the_stream(tmp_path):
    mc, m10i, model, frames, grids, codec = _setup()
    path = tmp_path / "seq.nvct"
    _encode_conditional(m10i, mc, model, codec, frames, path, grids)

    first = _decode_conditional(m10i, mc, model, codec, path, grids)
    del frames
    second = _decode_conditional(m10i, mc, model, codec, path, grids)

    assert torch.equal(first, second)


def test_motion_is_estimated_against_the_reconstruction_never_an_original(tmp_path, monkeypatch):
    mc, m10i, model, frames, grids, codec = _setup()
    seen = []
    original = mc.estimate_block_motion

    def recording(reference, current, **kwargs):
        seen.append((reference.detach().clone(), current.detach().clone()))
        return original(reference, current, **kwargs)

    monkeypatch.setattr(mc, "estimate_block_motion", recording)
    _encode_conditional(m10i, mc, model, codec, frames, tmp_path / "seq.nvct", grids)

    assert seen
    for reference, current in seen:
        for index in range(frames.shape[0]):
            assert not torch.equal(reference, frames[index:index + 1]), (
                "motion was estimated against an original frame the decoder lacks")
        assert any(torch.equal(current, frames[i:i + 1]) for i in range(frames.shape[0]))


def test_no_future_frame_is_accessed(tmp_path, monkeypatch):
    mc, m10i, model, frames, grids, codec = _setup()
    order = []
    original = mc.estimate_block_motion

    def recording(reference, current, **kwargs):
        matches = [i for i in range(frames.shape[0]) if torch.equal(current, frames[i:i + 1])]
        order.append(matches[0] if matches else -1)
        return original(reference, current, **kwargs)

    monkeypatch.setattr(mc, "estimate_block_motion", recording)
    _encode_conditional(m10i, mc, model, codec, frames, tmp_path / "seq.nvct", grids)

    assert order == sorted(order) and len(set(order)) == len(order)


def test_a_p_frame_cannot_be_coded_without_a_reference(tmp_path, monkeypatch):
    mc, m10i, model, frames, grids, codec = _setup()
    monkeypatch.setattr(mc, "gop_frame_types", lambda count, gop: [mc.FRAME_TYPE_P] * count)

    with pytest.raises((mc.CausalityViolationError, mc.TemporalFormatError)):
        _encode_conditional(m10i, mc, model, codec, frames, tmp_path / "bad.nvct", grids)


def test_i_frames_carry_no_motion_and_every_sequence_starts_with_one(tmp_path):
    mc, m10i, model, frames, grids, codec = _setup()
    path = tmp_path / "seq.nvct"
    encoded = _encode_conditional(m10i, mc, model, codec, frames, path, grids)

    frame_type, motion, _ = next(iter(mc.TemporalStreamReader(path)))
    assert frame_type == mc.FRAME_TYPE_I
    assert motion == b""
    for record in encoded["frames"]:
        if record["frame_type"] == "I":
            assert record["motion_bytes"] == 0


def test_sequence_boundaries_reset_reference_state(tmp_path):
    mc, m10i, model, _, _, codec = _setup()
    for index, seed in enumerate((0, 5)):
        frames = _moving_frames(6, seed=seed)
        grids = _grids(mc, model, frames)
        path = tmp_path / f"seq{index}.nvct"
        _encode_conditional(m10i, mc, model, codec, frames, path, grids)
        frame_type, motion, _ = next(iter(mc.TemporalStreamReader(path)))
        assert frame_type == mc.FRAME_TYPE_I and motion == b""


# --- bitstream ----------------------------------------------------------------


def test_the_residual_bitstream_round_trips(tmp_path):
    mc, m10i, model, frames, grids, codec = _setup()
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        codec.analysis[-1].weight.normal_(0.0, 0.02, generator=generator)
    path = tmp_path / "seq.nvct"

    encoded = _encode_conditional(m10i, mc, model, codec, frames, path, grids)
    decoded = _decode_conditional(m10i, mc, model, codec, path, grids)

    assert decoded.shape == frames.shape
    assert torch.equal(encoded["encoder_reconstructions"], decoded)


def test_total_byte_accounting_closes(tmp_path):
    mc, m10i, model, frames, grids, codec = _setup()
    path = tmp_path / "seq.nvct"

    encoded = _encode_conditional(m10i, mc, model, codec, frames, path, grids)

    on_disk = path.stat().st_size
    assert encoded["container_bytes"] == on_disk
    assert (encoded["motion_bytes"] + encoded["residual_bytes"]
            + encoded["container_overhead_bytes"]) == on_disk
    assert encoded["residual_bytes"] == (encoded["i_frame_residual_bytes"]
                                         + encoded["p_frame_residual_bytes"])


def test_a_truncated_residual_payload_is_rejected(tmp_path):
    mc, m10i, model, frames, grids, codec = _setup()
    path = tmp_path / "seq.nvct"
    _encode_conditional(m10i, mc, model, codec, frames, path, grids)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 50])

    with pytest.raises(mc.TemporalFormatError, match="Truncated"):
        _decode_conditional(m10i, mc, model, codec, path, grids)


def test_a_mismatched_residual_entropy_model_is_rejected(tmp_path):
    """An M10H decoder handed an M10I stream must fail loudly: the two arms
    share a reference mode but not a residual entropy model."""
    mc, m10i, model, frames, grids, codec = _setup()
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        codec.analysis[-1].weight.normal_(0.0, 0.05, generator=generator)
    path = tmp_path / "seq.nvct"
    _encode_conditional(m10i, mc, model, codec, frames, path, grids)

    other = _grids(mc, model, _moving_frames(8, seed=42, step=5))
    with pytest.raises(mc.TemporalFormatError, match="entropy model mismatch"):
        _decode_conditional(m10i, mc, model, codec, path,
                            {**grids, "residual_entropy_model": other["residual_entropy_model"]})


# --- scripts ------------------------------------------------------------------


@pytest.mark.parametrize("name", ["m10i_conditional_residual", "m10i_evaluate"])
def test_every_m10i_script_follows_the_project_script_contract(name):
    mod = _load_script(name)

    assert callable(mod.build_arg_parser)
    assert callable(mod.main)
    assert mod.build_arg_parser(load_default_config()).parse_args([]) is not None


def test_the_evaluator_holds_motion_constant_and_declares_three_rate_points():
    evaluate = _load_script("m10i_evaluate")

    assert evaluate.RATE_POINTS == (5, 4, 3)
    assert evaluate.FROZEN_LAMBDA == pytest.approx(3.0e-4)
    source = Path("scripts/m10i_evaluate.py").read_text(encoding="utf-8")
    assert "motion_identical" in source, "the ablation must verify motion is unchanged"
    assert "CONFOUNDED" in source, "a motion difference must be flagged, not absorbed"


def test_the_watched_sequences_are_the_ones_earlier_milestones_flagged():
    evaluate = _load_script("m10i_evaluate")

    assert set(evaluate.WATCH) == {"bmx-bumps", "drone", "cat-girl",
                                   "drift-chicane", "gold-fish", "schoolgirls"}


def test_m10i_does_not_modify_the_shipped_codec_or_earlier_containers():
    from nvc.compression import nvc_format

    assert nvc_format.MAGIC == b"NVC1"
    assert nvc_format.STREAM_MAGIC == b"NVCS"

    m10g = _load_script("m10g_temporal_baseline")
    mc = _load_script("m10h_motion_compensation")
    assert m10g.TEMPORAL_FORMAT_VERSION == 1
    assert mc.TEMPORAL_FORMAT_VERSION == 2, "M10I reuses M10H's container unchanged"

    trainer = _load_script("train_autoencoder")
    actions = {a.dest for a in trainer.build_arg_parser(load_default_config())._actions}
    for required in ("rate_lambda", "rate_lr", "resume_model_only", "seed"):
        assert required in actions
