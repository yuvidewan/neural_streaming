"""docs/PLAN_audit2_fixes.md, fix 1, proof 1: .nvct v2 streams may omit the two quantization
blocks (header counts 0), which is what wide-I-frame streams now do. Streams with blocks read
exactly as before, and an old-I-frame decode of a block-less stream refuses to guess."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from nvc.compression.quantization import QuantizationParams


def _load_script(name: str):
    sys.path.insert(0, "scripts")
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mc():
    return _load_script("m10h_motion_compensation")


def _header(mc, count: int):
    return mc.TemporalStreamHeader(
        gop_size=30, quantization_bits=5, quantization_mode="per_channel", image_width=64,
        image_height=64, image_channels=3, latent_channels=64, latent_height=4, latent_width=4,
        frame_count=2, num_intra_quantization_params=count,
        num_residual_quantization_params=count, block_size=16, search_range=16, motion_bits=5,
        reference_mode="mc", intra_entropy_model_id=b"intra___",
        residual_entropy_model_id=b"coder___", motion_entropy_model_id=b"motion__")


def _params():
    return QuantizationParams(scale=torch.linspace(0.1, 1.0, 64).reshape(1, 64, 1, 1),
                              zero_point=torch.zeros(1, 64, 1, 1), bits=5, mode="per_channel")


def _write(mc, path, header, blocks):
    with mc.TemporalStreamWriter(path, header, blocks, blocks) as writer:
        writer.append_frame(mc.FRAME_TYPE_I, b"", b"\x01\x02\x03")
        writer.append_frame(mc.FRAME_TYPE_P, b"\x07", b"\x04\x05")


def test_block_less_stream_round_trips_and_is_1024_bytes_smaller(mc, tmp_path):
    with_blocks, without = tmp_path / "a.nvct", tmp_path / "b.nvct"
    _write(mc, with_blocks, _header(mc, 64), _params())
    _write(mc, without, _header(mc, 0), None)
    old, new = with_blocks.read_bytes(), without.read_bytes()
    assert len(old) - len(new) == 1024
    # Same bytes apart from the two count fields (offsets 24-27) and the blocks themselves.
    size = mc.TEMPORAL_HEADER_SIZE
    assert new[:24] == old[:24] and new[28:size] == old[28:size]
    assert new[24:28] == b"\0\0\0\0"
    assert new[size:] == old[size + 1024:]
    reader = mc.TemporalStreamReader(without)
    assert reader.intra_params is None and reader.residual_params is None
    assert list(reader) == [(mc.FRAME_TYPE_I, b"", b"\x01\x02\x03"),
                            (mc.FRAME_TYPE_P, b"\x07", b"\x04\x05")]


def test_stream_with_blocks_reads_as_before(mc, tmp_path):
    path = tmp_path / "a.nvct"
    _write(mc, path, _header(mc, 64), _params())
    reader = mc.TemporalStreamReader(path)
    assert torch.allclose(reader.intra_params.scale.flatten(), _params().scale.flatten())
    assert len(list(reader)) == 2


def test_writer_refuses_counts_that_disagree_with_the_blocks(mc, tmp_path):
    with pytest.raises(mc.TemporalFormatError):
        mc.TemporalStreamWriter(tmp_path / "x.nvct", _header(mc, 0), _params(), _params())
    with pytest.raises(mc.TemporalFormatError):
        mc.TemporalStreamWriter(tmp_path / "y.nvct", _header(mc, 64), None, None)


def test_old_i_frame_decode_of_a_block_less_stream_raises(mc, tmp_path):
    g = _load_script("p4_gate3")
    path = tmp_path / "b.nvct"
    _write(mc, path, _header(mc, 0), None)
    rig = {"intra_entropy_model": type("E", (), {"model_id": lambda self: b"intra___"})(),
           "motion_entropy_model": type("E", (), {"model_id": lambda self: b"motion__"})()}
    pcoder = torch.nn.Linear(1, 1)
    with pytest.raises(mc.TemporalFormatError, match="no intra quantization block"):
        g.decode_sequence(mc, None, None, None, pcoder, path, rig, None, identity=b"coder___")
