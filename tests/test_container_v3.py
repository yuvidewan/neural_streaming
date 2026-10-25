"""`.nvct` v3 - the side-information record, and the proof it changed nothing.

M23 measured a transmitted scale map worth +1.42/+1.76/+1.77% of real coded
bytes, decoded from its own payload. v2 had nowhere to put it. v3 adds a fourth
per-frame length and ten header bytes, and nothing else.

The load-bearing tests here are the compatibility ones:

  * `test_a_v2_stream_is_byte_identical_to_the_pre_change_writer` - built by
    running the previous implementation out of git and comparing bytes. Every
    `.nvct` v2 stream the project has written stays valid, and that is the
    condition for touching a frozen format at all;
  * `test_the_v2_constants_keep_their_v2_values` - `m10g_temporal_baseline` and
    `m10h_motion_compensation` parse the layout with them directly;
  * `test_iteration_still_yields_three_tuples_on_both_versions` - no existing
    reader may change shape;
  * the fail-closed group - a stream must not be able to acquire side
    information its header does not declare.
"""

from __future__ import annotations

import io
import subprocess
import types
from pathlib import Path

import pytest
import torch

from nvc.compression.quantization import QuantizationParams
from nvc.video import container as C

ROOT = Path(__file__).resolve().parents[1]

FRAMES_V2 = [(0, b"", b"AAA"), (1, b"mm", b"BBBB"), (1, b"nn", b"CC")]
FRAMES_V3 = [(0, b"", b"AAA", b"s1"), (1, b"mm", b"BBBB", b"s22"),
             (1, b"nn", b"CC", b"s333")]


def _header(version: int = 2, *, latent_height: int = 4, latent_width: int = 4, **extra):
    return C.TemporalStreamHeader(
        gop_size=10, quantization_bits=4, quantization_mode="per_channel",
        image_width=64, image_height=64, image_channels=3,
        latent_channels=4, latent_height=latent_height, latent_width=latent_width,
        frame_count=3, num_intra_quantization_params=4,
        num_residual_quantization_params=4, block_size=16, search_range=16,
        motion_bits=6, reference_mode="mc", intra_entropy_model_id=b"\x01" * 8,
        residual_entropy_model_id=b"\x02" * 8, motion_entropy_model_id=b"\x03" * 8,
        format_version=version, **extra)


def _params():
    return QuantizationParams(scale=torch.ones(1, 4, 1, 1),
                              zero_point=torch.zeros(1, 4, 1, 1), bits=4,
                              mode="per_channel")


def _v3_header():
    return _header(3, side_bins=8, side_stride=2, side_entropy_model_id=b"\x09" * 8)


@pytest.fixture(scope="module")
def previous_container():
    """The implementation as of HEAD, loaded from git.

    Compatibility claimed against a remembered layout is not compatibility. This
    builds the actual previous module and compares bytes with it.
    """
    source = subprocess.run(
        ["git", "show", "HEAD:src/nvc/video/container.py"],
        capture_output=True, text=True, check=True, cwd=ROOT).stdout
    module = types.ModuleType("previous_container")
    exec(compile(source, "previous_container.py", "exec"), module.__dict__)
    return module


# ----------------------------------------------------------------------------------------
# Backward compatibility - the condition for touching a frozen format
# ----------------------------------------------------------------------------------------


def test_a_v2_stream_is_byte_identical_to_the_pre_change_writer(previous_container):
    old_header = previous_container.TemporalStreamHeader(
        gop_size=10, quantization_bits=4, quantization_mode="per_channel",
        image_width=64, image_height=64, image_channels=3, latent_channels=4,
        latent_height=4, latent_width=4, frame_count=3,
        num_intra_quantization_params=4, num_residual_quantization_params=4,
        block_size=16, search_range=16, motion_bits=6, reference_mode="mc",
        intra_entropy_model_id=b"\x01" * 8, residual_entropy_model_id=b"\x02" * 8,
        motion_entropy_model_id=b"\x03" * 8)

    before = previous_container.stream_to_bytes(old_header, _params(), _params(), FRAMES_V2)
    after = C.stream_to_bytes(_header(), _params(), _params(), FRAMES_V2)

    assert after == before


def test_the_previous_reader_reads_the_new_writers_v2_output(previous_container):
    stream = C.stream_to_bytes(_header(), _params(), _params(), FRAMES_V2)

    assert list(previous_container.TemporalStreamReader(stream)) == FRAMES_V2


def test_the_new_reader_reads_the_previous_writers_output(previous_container):
    old_header = previous_container.TemporalStreamHeader(
        gop_size=10, quantization_bits=4, quantization_mode="per_channel",
        image_width=64, image_height=64, image_channels=3, latent_channels=4,
        latent_height=4, latent_width=4, frame_count=3,
        num_intra_quantization_params=4, num_residual_quantization_params=4,
        block_size=16, search_range=16, motion_bits=6, reference_mode="mc",
        intra_entropy_model_id=b"\x01" * 8, residual_entropy_model_id=b"\x02" * 8,
        motion_entropy_model_id=b"\x03" * 8)
    stream = previous_container.stream_to_bytes(old_header, _params(), _params(), FRAMES_V2)

    assert list(C.TemporalStreamReader(stream)) == FRAMES_V2


def test_the_v2_constants_keep_their_v2_values():
    """`m10g_temporal_baseline` and `m10h_motion_compensation` import these and
    parse the byte layout with them. Repointing either at v3 would silently
    misread every v2 stream those scripts touch."""
    assert C.TEMPORAL_FORMAT_VERSION == 2
    assert C.TEMPORAL_HEADER_SIZE == 56
    assert C._FRAME_RECORD_STRUCT.format == "<BII"
    assert C._FRAME_RECORD_STRUCT.size == 9
    # v3's are separate
    assert C.TEMPORAL_HEADER_SIZE_V3 == 66
    assert C._FRAME_RECORD_STRUCT_V3.size == 13
    assert C.SUPPORTED_FORMAT_VERSIONS == (2, 3)


def test_the_writer_still_defaults_to_v2():
    """A v3 stream is only produced by a caller that asks for one."""
    assert _header().format_version == 2
    assert C.TemporalStreamReader(
        C.stream_to_bytes(_header(), _params(), _params(), FRAMES_V2)
    ).header.format_version == 2


def test_iteration_still_yields_three_tuples_on_both_versions():
    v2 = C.stream_to_bytes(_header(), _params(), _params(), FRAMES_V2)
    v3 = C.stream_to_bytes(_v3_header(), _params(), _params(), FRAMES_V3)

    assert list(C.TemporalStreamReader(v2)) == FRAMES_V2
    assert list(C.TemporalStreamReader(v3)) == [f[:3] for f in FRAMES_V3]
    for stream in (v2, v3):
        assert all(len(frame) == 3 for frame in C.TemporalStreamReader(stream))


def test_append_frame_still_takes_three_positional_arguments():
    buffer = io.BytesIO()
    writer = C.TemporalStreamWriter(buffer, _header(), _params(), _params())
    for frame_type, motion, residual in FRAMES_V2:
        writer.append_frame(frame_type, motion, residual)
    writer.close()

    assert list(C.TemporalStreamReader(buffer.getvalue())) == FRAMES_V2


# ----------------------------------------------------------------------------------------
# v3 itself
# ----------------------------------------------------------------------------------------


def test_a_v3_stream_round_trips_its_side_payloads():
    stream = C.stream_to_bytes(_v3_header(), _params(), _params(), FRAMES_V3)

    assert list(C.TemporalStreamReader(stream).iter_frames_with_side()) == FRAMES_V3


def test_a_v2_stream_reports_empty_side_payloads():
    """So v3-aware code can read either version through one path."""
    stream = C.stream_to_bytes(_header(), _params(), _params(), FRAMES_V2)

    rows = list(C.TemporalStreamReader(stream).iter_frames_with_side())

    assert [row[3] for row in rows] == [b"", b"", b""]
    assert [row[:3] for row in rows] == FRAMES_V2


def test_the_side_header_fields_survive_a_round_trip():
    stream = C.stream_to_bytes(_v3_header(), _params(), _params(), FRAMES_V3)

    header = C.TemporalStreamReader(stream).header

    assert header.format_version == 3
    assert header.side_bins == 8
    assert header.side_stride == 2
    assert header.side_entropy_model_id == b"\x09" * 8
    assert header.carries_side_information is True
    assert header.header_size == C.TEMPORAL_HEADER_SIZE_V3


def test_v3_costs_exactly_ten_header_bytes_and_four_per_frame():
    """The whole size of the change, asserted so it cannot grow unnoticed."""
    v2 = C.stream_to_bytes(_header(), _params(), _params(), FRAMES_V2)
    bare_v3 = C.stream_to_bytes(_header(3), _params(), _params(), FRAMES_V2)

    assert len(bare_v3) - len(v2) == 10 + 4 * len(FRAMES_V2)


def test_the_side_grid_ceil_divides_like_the_pooling_that_produces_it():
    """`m23_phaseb_gate.coarsen` uses ceil_mode=True, so a 5x5 latent at stride 2
    gives a 3x3 map. A floor here would have the decoder expect 2x2."""
    header = _header(3, latent_height=5, latent_width=5, side_bins=8, side_stride=2)

    assert header.side_grid == (3, 3)
    assert _header(3, latent_height=16, latent_width=16, side_bins=8,
                   side_stride=2).side_grid == (8, 8)
    assert _header(3, latent_height=16, latent_width=16, side_bins=8,
                   side_stride=1).side_grid == (16, 16)


def test_a_v3_stream_may_declare_no_side_channel():
    """side_bins = 0 is how v3 says it carries none, and it must still read."""
    stream = C.stream_to_bytes(_header(3), _params(), _params(), FRAMES_V2)
    reader = C.TemporalStreamReader(stream)

    assert reader.header.carries_side_information is False
    assert list(reader) == FRAMES_V2


# ----------------------------------------------------------------------------------------
# Fail-closed: a stream cannot acquire side information its header does not declare
# ----------------------------------------------------------------------------------------


def test_a_v2_header_cannot_describe_a_side_channel():
    with pytest.raises(C.TemporalFormatError, match="v2 header cannot describe"):
        _header(2, side_bins=8, side_stride=2)


def test_an_unknown_format_version_is_refused_on_write_and_on_read():
    with pytest.raises(C.TemporalFormatError, match="Unsupported .nvct format version 4"):
        _header(4)
    stream = bytearray(C.stream_to_bytes(_header(), _params(), _params(), FRAMES_V2))
    stream[4] = 9
    with pytest.raises(C.TemporalFormatError, match="Unsupported .nvct format version 9"):
        C.TemporalStreamHeader.unpack(bytes(stream))


@pytest.mark.parametrize("bins,stride,match", [
    (6, 2, "power of two"),
    (8, 0, "side_stride must be >= 1"),
])
def test_malformed_side_parameters_are_refused(bins, stride, match):
    with pytest.raises(C.TemporalFormatError, match=match):
        _header(3, side_bins=bins, side_stride=stride)


def test_a_side_payload_on_a_v2_stream_is_refused():
    with pytest.raises(C.TemporalFormatError, match="format version 2, which has no"):
        C.stream_to_bytes(_header(), _params(), _params(),
                          [(0, b"", b"A", b"x"), (1, b"m", b"B"), (1, b"n", b"C")])


def test_a_side_payload_without_a_declared_side_channel_is_refused():
    """A v3 header is not a licence to write side bytes; side_bins must say so."""
    with pytest.raises(C.TemporalFormatError, match="declares side_bins=0"):
        C.stream_to_bytes(_header(3), _params(), _params(),
                          [(0, b"", b"A", b"x"), (1, b"m", b"B"), (1, b"n", b"C")])


def test_a_truncated_side_payload_is_refused():
    stream = C.stream_to_bytes(_v3_header(), _params(), _params(), FRAMES_V3)

    with pytest.raises(C.TemporalFormatError, match="Truncated side payload on frame 2"):
        list(C.TemporalStreamReader(stream[:-2]))


def test_trailing_data_after_a_v3_stream_is_refused():
    stream = C.stream_to_bytes(_v3_header(), _params(), _params(), FRAMES_V3)

    with pytest.raises(C.TemporalFormatError, match="Trailing data"):
        list(C.TemporalStreamReader(stream + b"\x00"))


def test_a_truncated_v3_header_is_refused():
    stream = C.stream_to_bytes(_v3_header(), _params(), _params(), FRAMES_V3)

    with pytest.raises(C.TemporalFormatError, match="Truncated .nvct v3 header"):
        C.TemporalStreamHeader.unpack(stream[:60])


def test_a_frame_declaring_side_bytes_against_a_zero_bins_header_is_refused():
    """Hand-built: a v3 stream whose header says side_bins=0 but whose frame
    record declares a side length. The reader must refuse rather than hand back
    bytes nothing describes."""
    stream = bytearray(C.stream_to_bytes(_v3_header(), _params(), _params(), FRAMES_V3))
    stream[C.TEMPORAL_HEADER_SIZE] = 0          # side_bins -> 0, frames unchanged

    with pytest.raises(C.TemporalFormatError, match="declares side_bins=0"):
        list(C.TemporalStreamReader(bytes(stream)))


def test_a_frame_tuple_of_the_wrong_length_is_refused():
    with pytest.raises(C.TemporalFormatError, match="must be \\(type, motion, residual\\)"):
        C.stream_to_bytes(_v3_header(), _params(), _params(), [(0, b"", b"A", b"x", b"y")])


def test_side_grid_is_refused_on_a_stream_without_a_side_channel():
    with pytest.raises(C.TemporalFormatError, match="declares no side channel"):
        _header(2).side_grid


def test_the_v2_invariants_still_hold_on_v3():
    """The first frame must be an I-frame and an I-frame carries no motion -
    adding a payload must not have loosened either."""
    with pytest.raises(C.TemporalFormatError, match="first frame of a stream must be"):
        C.stream_to_bytes(_v3_header(), _params(), _params(),
                          [(1, b"m", b"A", b"x"), (1, b"m", b"B", b"y"),
                           (1, b"n", b"C", b"z")])
    with pytest.raises(C.TemporalFormatError, match="I-frame must carry no motion"):
        C.stream_to_bytes(_v3_header(), _params(), _params(),
                          [(0, b"m", b"A", b"x"), (1, b"m", b"B", b"y"),
                           (1, b"n", b"C", b"z")])
