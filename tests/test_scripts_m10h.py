"""Tests for the M10H scripts and the `.nvct` version 2 container.

Version 2 adds a per-frame MOTION payload alongside the residual one, with
explicit lengths for both. That separation is what makes motion bytes
attributable rather than folded invisibly into the residual, so the tests here
lean on it: truncation in either payload must be caught independently, and an
I-frame must be structurally incapable of carrying motion.

Also checks that M10G's version-1 reader correctly REFUSES a version-2 stream -
that is the version field doing its job, not a regression.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.utils.config import load_default_config


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _header(mc, **overrides):
    fields = dict(
        gop_size=10, quantization_bits=8, quantization_mode="per_channel",
        image_width=256, image_height=256, image_channels=3,
        latent_channels=64, latent_height=16, latent_width=16, frame_count=3,
        num_intra_quantization_params=64, num_residual_quantization_params=64,
        block_size=16, search_range=16, motion_bits=6, reference_mode="mc",
        intra_entropy_model_id=b"\xaa" * 8, residual_entropy_model_id=b"\xbb" * 8,
        motion_entropy_model_id=b"\xcc" * 8)
    fields.update(overrides)
    return mc.TemporalStreamHeader(**fields)


def _params(channels=64):
    return calibrate_quantization_params(
        torch.randn(4, channels, 4, 4), bits=8, mode="per_channel")


def _write_stream(mc, tmp_path, frames=(("I", b"", b"abc"), ("P", b"mm", b"defg"))):
    header = _header(mc, frame_count=len(frames), latent_channels=8,
                     num_intra_quantization_params=8, num_residual_quantization_params=8)
    params = _params(8)
    path = tmp_path / "seq.nvct"
    with mc.TemporalStreamWriter(path, header, params, params) as writer:
        for kind, motion, residual in frames:
            writer.append_frame(
                mc.FRAME_TYPE_I if kind == "I" else mc.FRAME_TYPE_P, motion, residual)
    return path


# --- header ------------------------------------------------------------------


def test_header_round_trips_and_is_the_documented_size():
    mc = _load_script("m10h_motion_compensation")
    header = _header(mc)

    packed = header.pack()
    assert len(packed) == mc.TEMPORAL_HEADER_SIZE == 56

    restored = mc.TemporalStreamHeader.unpack(packed)
    assert restored.format_version == 2
    assert restored.gop_size == 10
    assert restored.block_size == 16 and restored.search_range == 16
    assert restored.motion_bits == 6
    assert restored.reference_mode == "mc"
    assert restored.image_shape == (3, 256, 256)
    assert restored.latent_shape == (64, 16, 16)
    assert restored.block_grid == (16, 16)
    assert restored.motion_entropy_model_id == b"\xcc" * 8


def test_the_block_grid_matches_the_latent_grid_at_this_geometry():
    """16x16 blocks on a 256x256 frame give exactly one motion vector per
    latent position, which is why that block size was chosen."""
    mc = _load_script("m10h_motion_compensation")

    header = _header(mc)

    assert header.block_grid == (header.latent_height, header.latent_width)


def test_bad_magic_and_unknown_version_are_rejected():
    mc = _load_script("m10h_motion_compensation")
    packed = _header(mc).pack()

    with pytest.raises(mc.TemporalFormatError, match="magic"):
        mc.TemporalStreamHeader.unpack(b"XXXX" + packed[4:])

    corrupt = bytearray(packed)
    corrupt[4] = 99
    with pytest.raises(mc.TemporalFormatError, match="version"):
        mc.TemporalStreamHeader.unpack(bytes(corrupt))


def test_a_corrupt_reference_mode_code_is_rejected():
    """Corrupt motion metadata: an unknown reference mode changes how every
    P-frame would be interpreted, so it must fail rather than guess."""
    mc = _load_script("m10h_motion_compensation")
    corrupt = bytearray(_header(mc).pack())
    corrupt[31] = 9  # reference_mode byte

    with pytest.raises(mc.TemporalFormatError, match="reference mode"):
        mc.TemporalStreamHeader.unpack(bytes(corrupt))


def test_a_truncated_header_is_rejected():
    mc = _load_script("m10h_motion_compensation")

    with pytest.raises(mc.TemporalFormatError, match="Truncated"):
        mc.TemporalStreamHeader.unpack(b"NVCT\x02\x0a")


def test_m10g_version_one_reader_refuses_a_version_two_stream(tmp_path):
    """The version field doing its job - not a regression."""
    mc = _load_script("m10h_motion_compensation")
    m10g = _load_script("m10g_temporal_baseline")
    path = _write_stream(mc, tmp_path)

    with pytest.raises(m10g.TemporalFormatError, match="version"):
        m10g.TemporalStreamReader(path)


# --- frame records -----------------------------------------------------------


def test_motion_and_residual_payloads_are_separately_addressable(tmp_path):
    mc = _load_script("m10h_motion_compensation")
    path = _write_stream(mc, tmp_path)

    records = list(mc.TemporalStreamReader(path))

    assert records[0] == (mc.FRAME_TYPE_I, b"", b"abc")
    assert records[1] == (mc.FRAME_TYPE_P, b"mm", b"defg")


def test_an_i_frame_may_not_carry_motion(tmp_path):
    mc = _load_script("m10h_motion_compensation")
    header = _header(mc, frame_count=1, latent_channels=8,
                     num_intra_quantization_params=8, num_residual_quantization_params=8)
    params = _params(8)

    writer = mc.TemporalStreamWriter(tmp_path / "bad.nvct", header, params, params)
    try:
        with pytest.raises(mc.TemporalFormatError, match="no motion payload"):
            writer.append_frame(mc.FRAME_TYPE_I, b"motion", b"residual")
    finally:
        writer._file.close()


def test_a_stream_may_not_open_with_a_p_frame(tmp_path):
    mc = _load_script("m10h_motion_compensation")
    header = _header(mc, frame_count=1, latent_channels=8,
                     num_intra_quantization_params=8, num_residual_quantization_params=8)
    params = _params(8)

    writer = mc.TemporalStreamWriter(tmp_path / "bad.nvct", header, params, params)
    try:
        with pytest.raises(mc.TemporalFormatError, match="must be an I-frame"):
            writer.append_frame(mc.FRAME_TYPE_P, b"m", b"r")
    finally:
        writer._file.close()


def test_a_truncated_motion_payload_is_rejected(tmp_path):
    """Motion and residual truncation must be caught independently - a reader
    that only validated the total length would decode a short motion field as
    if it were complete."""
    mc = _load_script("m10h_motion_compensation")
    path = _write_stream(mc, tmp_path, frames=(("I", b"", b"abc"), ("P", b"m" * 64, b"r" * 8)))
    data = path.read_bytes()
    # Drop everything after the P-frame record header plus a few motion bytes.
    path.write_bytes(data[: len(data) - 60])

    with pytest.raises(mc.TemporalFormatError, match="Truncated motion payload"):
        list(mc.TemporalStreamReader(path))


def test_a_truncated_residual_payload_is_rejected(tmp_path):
    mc = _load_script("m10h_motion_compensation")
    path = _write_stream(mc, tmp_path, frames=(("I", b"", b"abc"), ("P", b"mm", b"r" * 64)))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 40])

    with pytest.raises(mc.TemporalFormatError, match="Truncated residual payload"):
        list(mc.TemporalStreamReader(path))


def test_trailing_data_is_rejected(tmp_path):
    mc = _load_script("m10h_motion_compensation")
    path = _write_stream(mc, tmp_path)
    path.write_bytes(path.read_bytes() + b"garbage")

    with pytest.raises(mc.TemporalFormatError, match="Trailing"):
        list(mc.TemporalStreamReader(path))


def test_an_unknown_frame_type_is_rejected(tmp_path):
    mc = _load_script("m10h_motion_compensation")
    path = _write_stream(mc, tmp_path)
    reader = mc.TemporalStreamReader(path)
    offset = (mc.TEMPORAL_HEADER_SIZE + reader.header.num_intra_quantization_params * 8
              + reader.header.num_residual_quantization_params * 8)
    data = bytearray(path.read_bytes())
    data[offset] = 7
    path.write_bytes(bytes(data))

    with pytest.raises(mc.TemporalFormatError, match="frame_type"):
        list(mc.TemporalStreamReader(path))


def test_writing_fewer_frames_than_declared_is_rejected(tmp_path):
    mc = _load_script("m10h_motion_compensation")
    header = _header(mc, frame_count=3, latent_channels=8,
                     num_intra_quantization_params=8, num_residual_quantization_params=8)
    params = _params(8)

    writer = mc.TemporalStreamWriter(tmp_path / "short.nvct", header, params, params)
    writer.append_frame(mc.FRAME_TYPE_I, b"", b"x")
    with pytest.raises(mc.TemporalFormatError, match="only 1 were written"):
        writer.close()


def test_a_mismatched_motion_entropy_model_is_rejected(tmp_path):
    mc = _load_script("m10h_motion_compensation")
    torch.manual_seed(0)
    from nvc.models.autoencoder import BaselineAutoencoder
    net = BaselineAutoencoder(in_channels=3, latent_channels=4, base_channels=8).eval()
    frames = torch.rand(6, 3, 64, 64)

    import tests.test_motion_compensation as helper  # reuse the grid builder
    grids = helper._grids(mc, net, frames)
    path = tmp_path / "seq.nvct"
    helper._encode(mc, net, frames, path, grids, gop_size=3)

    wrong = EmpiricalEntropyModel.from_symbols(
        np.stack([np.stack([np.arange(16) % 17, (np.arange(16) + 3) % 17])]),
        bits=mc.motion_alphabet_bits(8), num_tables=2)
    with pytest.raises(mc.TemporalFormatError, match="motion entropy model mismatch"):
        mc.decode_sequence(
            net, path,
            intra_entropy_model=grids["intra_entropy_model"],
            residual_entropy_model=grids["residual_entropy_model"],
            motion_entropy_model=wrong)


# --- scripts -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["m10h_motion_compensation", "m10h_evaluate"])
def test_every_m10h_script_follows_the_project_script_contract(name):
    mod = _load_script(name)

    assert callable(mod.build_arg_parser)
    assert callable(mod.main)
    assert mod.build_arg_parser(load_default_config()).parse_args([]) is not None


def test_the_evaluator_uses_two_rate_points_that_separate_in_quality():
    """8/6-bit was measured and rejected: it moves rate without moving quality,
    leaving no shared quality range for a BD-rate integration."""
    evaluate = _load_script("m10h_evaluate")

    assert evaluate.RATE_POINTS == (8, 4)
    assert len(set(evaluate.RATE_POINTS)) == 2
    source = Path("scripts/m10h_evaluate.py").read_text(encoding="utf-8")
    assert "6-bit was measured and REJECTED" in source


def test_the_evaluator_compares_all_three_coded_paths_plus_a_quarantined_oracle():
    evaluate = _load_script("m10h_evaluate")
    mc = _load_script("m10h_motion_compensation")

    labels = [a[0] for a in evaluate.ARMS]
    assert labels == ["intra", "prev", "mc", "oracle"]
    assert dict((a[0], a[2]) for a in evaluate.ARMS)["intra"] == 1, "intra is the coder at GOP=1"
    accounted = {a[0]: mc.is_rate_accounted(a[1]) for a in evaluate.ARMS}
    assert accounted == {"intra": True, "prev": True, "mc": True, "oracle": False}


def test_the_frozen_operating_point_and_checkpoint_convention_are_used():
    evaluate = _load_script("m10h_evaluate")

    assert evaluate.FROZEN_LAMBDA == pytest.approx(3.0e-4)
    assert "best.pt" in str(evaluate.DEFAULT_CHECKPOINT)
    assert "lambda_3.0e-04" in str(evaluate.DEFAULT_CHECKPOINT)


def test_the_motion_sensitive_sequences_are_the_ones_m10g_flagged():
    evaluate = _load_script("m10h_evaluate")

    assert set(evaluate.WATCH) == {"bmx-bumps", "drift-chicane", "schoolgirls", "gold-fish"}


def test_m10h_does_not_modify_the_shipped_codec_or_the_m10g_container():
    from nvc.compression import nvc_format

    assert nvc_format.MAGIC == b"NVC1"
    assert nvc_format.STREAM_MAGIC == b"NVCS"

    m10g = _load_script("m10g_temporal_baseline")
    mc = _load_script("m10h_motion_compensation")
    assert m10g.TEMPORAL_FORMAT_VERSION == 1, "M10G's container is frozen"
    assert mc.TEMPORAL_FORMAT_VERSION == 2
    assert m10g.TEMPORAL_MAGIC == mc.TEMPORAL_MAGIC == b"NVCT"

    trainer = _load_script("train_autoencoder")
    actions = {a.dest for a in trainer.build_arg_parser(load_default_config())._actions}
    for required in ("rate_lambda", "rate_lr", "resume_model_only", "seed"):
        assert required in actions
