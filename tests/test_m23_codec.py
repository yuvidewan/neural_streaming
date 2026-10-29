"""M23 Phase B2 step 2 - tests for `scripts/m23_codec.py` and its validation driver.

The load-bearing tests:

  * `test_the_encoder_decodes_its_own_map_before_using_it` - the encoder must not
    condition on anything the decoder lacks, and it proves that by round-tripping
    its own side payload rather than trusting it;
  * `test_the_decoder_takes_its_context_only_from_the_stream` - if the decode
    path could reach the encoder's map, the measurement would describe a codec
    that cannot exist;
  * `test_the_v2_path_must_byte_match_m21s_deployed_arm` - a reimplemented I/O
    path could change the bytes for reasons unrelated to the scale map;
  * `test_the_stride_is_selected_on_val_a_not_val_b` - VAL-A and VAL-B disagree
    here, and taking VAL-B's answer would be choosing a hyperparameter on the
    split that reports it.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from nvc.compression.entropy_model import EmpiricalEntropyModel

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def codec():
    return _load_script("m23_codec")


@pytest.fixture(scope="module")
def driver():
    return _load_script("m23_deploy_validation")


@pytest.fixture(scope="module")
def pb():
    return _load_script("m23_phaseb_gate")


def _tables(rows: int, alphabet: int, seed: int = 0) -> EmpiricalEntropyModel:
    cv = _load_script("m23_coded_validation")
    mk = _load_script("m10k_learned_entropy")
    rng = np.random.default_rng(seed)
    probabilities = rng.random((rows, alphabet)) + 0.5
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return cv.coder_tables(mk, probabilities, bits=int(np.log2(alphabet)))


def _side(codec, *, bins=8, stride=2, groups=16, alphabet=16):
    return codec.SideSpecification(
        bins=bins, stride=stride, edges=np.linspace(0.1, 0.9, bins - 1),
        map_table=_tables(1, bins, seed=1), child_table=_tables(groups * bins, alphabet,
                                                                seed=2),
        groups=groups, identity=b"\x07" * 8)


# ----------------------------------------------------------------------------------------
# Decodability
# ----------------------------------------------------------------------------------------


def test_the_encoder_decodes_its_own_map_before_using_it(codec):
    """The encoder codes the map, decodes it back from its own payload, and
    refuses to continue if the two differ. Conditioning on the un-round-tripped
    map would describe a codec no decoder could reproduce."""
    source = inspect.getsource(codec.encode_sequence)

    encode_at = source.index("side_payload = encode_symbols(")
    decode_at = source.index("recovered = decode_symbols(")
    check_at = source.index("if not np.array_equal(recovered, flat_map):")
    context_at = source.index("context = context_from_map(")
    assert encode_at < decode_at < check_at < context_at
    assert "undecodable" in source
    assert "context_from_map(pb, recovered" in source


def test_the_decoder_takes_its_context_only_from_the_stream(codec):
    """`side` supplies frozen TABLES, exactly like the intra and motion entropy
    models; the map itself must come from the frame record."""
    source = inspect.getsource(codec.decode_sequence)

    assert "binned = decode_symbols(side_payload" in source
    assert "context = context_from_map(pb, binned.reshape" in source
    # nothing in the decoder may read an encoder-side map
    assert "quantize_scale_map" not in source
    assert "scale_map_signal" not in source


def test_the_decoder_refuses_a_side_stream_without_its_tables(codec):
    source = inspect.getsource(codec.decode_sequence)

    assert "stream declares a side channel but no SideSpecification" in source
    assert "the given specification has bins=" in source


def test_the_side_payload_is_read_before_the_residual_is_decoded(codec):
    """Transmitted, not causal: the context is known for every position before
    the first symbol group, which is what lets the residual decode group-wise
    with no interleaving."""
    source = inspect.getsource(codec.decode_residual_with_context)

    assert "already known for every position" in source
    assert "context[start * plane:stop * plane]" in source


# ----------------------------------------------------------------------------------------
# Equivalence with the deployed arm
# ----------------------------------------------------------------------------------------


def test_the_v2_path_must_byte_match_m21s_deployed_arm(driver):
    """And the run must STOP if it does not - a byte difference would make every
    number below attributable to reimplemented I/O rather than the map."""
    source = inspect.getsource(driver.main)

    assert "prove_equivalence(" in source
    assert 'if not proof["byte_identical"]:' in source
    assert "refusing to measure" in source
    assert "return 1" in source
    proof = inspect.getsource(driver.prove_equivalence)
    assert "encode_sequence_refined(" in proof
    assert "side=None" in proof


def test_the_v2_path_takes_the_deployed_residual_coder(codec):
    """With no side channel the residual must go through M13's own function, not
    a joint-table lookalike."""
    source = inspect.getsource(codec.encode_sequence)

    assert "if side is None:" in source
    assert "m13.encode_frame_recalibrated(" in source
    decode = inspect.getsource(codec.decode_sequence)
    assert "m13.decode_frame_recalibrated(" in decode


def test_the_production_container_is_used_not_m10hs_copy(codec):
    """The point of this module: v3 lives in the production container, and the
    measurement path had been using m10h's parallel copy."""
    source = inspect.getsource(codec)

    assert "from nvc.video.container import" in source
    assert "mc.TemporalStreamWriter" not in source
    assert "mc.TemporalStreamReader" not in source


# ----------------------------------------------------------------------------------------
# The side specification
# ----------------------------------------------------------------------------------------


def test_the_joint_table_count_must_match_prototypes_times_contexts(codec):
    """A mismatch here would index the wrong table and silently corrupt a
    stream, so it is refused at construction."""
    with pytest.raises(ValueError, match="needs 128"):
        codec.SideSpecification(
            bins=8, stride=2, edges=np.zeros(7), map_table=_tables(1, 8),
            child_table=_tables(64, 16), groups=16, identity=b"\x00" * 8)


def test_bins_must_be_a_power_of_two(codec):
    with pytest.raises(ValueError, match="power of two"):
        codec.SideSpecification(
            bins=6, stride=2, edges=np.zeros(5), map_table=_tables(1, 8),
            child_table=_tables(96, 16), groups=16, identity=b"\x00" * 8)


def test_the_coarse_grid_ceil_divides_like_the_pooling_and_the_header(codec):
    """`coarsen` uses ceil_mode=True and the production header's `side_grid`
    ceil-divides; a floor anywhere here would make the decoder read the wrong
    number of map symbols."""
    side = _side(codec, stride=2)

    assert side.coarse_grid(16, 16) == (8, 8)
    assert side.coarse_grid(5, 5) == (3, 3)
    assert _side(codec, stride=1).coarse_grid(16, 16) == (16, 16)
    assert _side(codec, stride=4).coarse_grid(16, 16) == (4, 4)


def test_the_map_bit_width_follows_the_bin_count(codec):
    assert _side(codec, bins=8).map_bit_width == 3
    assert _side(codec, bins=16).map_bit_width == 4


def test_the_joint_index_is_prototype_times_bins_plus_context(codec):
    """Encoder and decoder must build the same index, so the expression is
    pinned in both."""
    encode = inspect.getsource(codec.encode_residual_with_context)
    decode = inspect.getsource(codec.decode_residual_with_context)

    assert "* side.bins + context" in encode
    assert "* side.bins" in decode and "+ context[start * plane:stop * plane]" in decode


# ----------------------------------------------------------------------------------------
# A real round trip of the residual coder, through the real range coder
# ----------------------------------------------------------------------------------------


def test_the_context_actually_steers_the_residual_coder(codec, pb):
    """The negative control: coding against one context and decoding against
    another must corrupt the symbols. If it does not, the joint tables are not
    doing anything and the whole arm measures nothing."""
    from nvc.compression.range_coder import decode_symbols, encode_symbols

    rng = np.random.default_rng(0)
    channels, height, width, bins, groups = 4, 8, 8, 8, 16
    symbols = rng.integers(0, 16, size=channels * height * width)
    prototype = rng.integers(0, groups, size=symbols.size)
    side = _side(codec, bins=bins, groups=groups)
    context = rng.integers(0, bins, size=symbols.size)
    other = (context + 1) % bins

    payload = encode_symbols(symbols, side.child_table.cumulative,
                             prototype * bins + context)
    right = decode_symbols(payload, symbols.size, side.child_table.cumulative,
                           prototype * bins + context)
    wrong = decode_symbols(payload, symbols.size, side.child_table.cumulative,
                           prototype * bins + other)

    assert np.array_equal(right, symbols)
    assert not np.array_equal(wrong, symbols)


def test_the_map_is_built_from_the_dequantized_residual(codec):
    """The decoder only ever holds the dequantized residual, so a map built from
    the continuous one would describe something never transmitted."""
    source = inspect.getsource(codec.encode_sequence)

    assert "residual = symbols_to_latent(symbols, latent_shape" in source
    assert "binned = quantize_scale_map(pb, residual, side)" in source
    assert "DEQUANTIZED residual" in source


def test_the_map_construction_calls_the_gate_rather_than_reimplementing_it(codec):
    """The codec must code the map the gate measured, or the realisation ratio
    compares two different things."""
    source = inspect.getsource(codec.quantize_scale_map)

    assert "pb.coarsen(" in source
    assert "pb.scale_map_signal(" in source
    assert "pb.to_bins(" in source


# ----------------------------------------------------------------------------------------
# Protocol: which split chose the stride
# ----------------------------------------------------------------------------------------


def test_the_stride_is_selected_on_val_a_not_val_b(driver):
    """B1b's VAL-A picked stride 1 at every rate point; VAL-B prefers stride 2 at
    4- and 3-bit. Taking VAL-B's answer would be selecting a hyperparameter on
    the split that reports it."""
    assert driver.SELECTED_STRIDE == 1
    assert driver.SELECTED_BINS == 8
    source = inspect.getsource(driver)
    assert "VAL-A" in source
    assert "selecting it on VAL-B would be choosing a" in source
    main = inspect.getsource(driver.main)
    assert '"stride_selected_on": "VAL-A"' in main
    assert '"stride_val_b_disagreement"' in main


def test_the_disagreement_is_recorded_as_a_variance_warning(driver):
    """If TEST lands below VAL-B's figure, the reason should already be written
    down rather than discovered afterwards."""
    source = inspect.getsource(driver)

    assert "variance warning" in source
    assert "rather than a surprise" in source


def test_the_baseline_is_the_deployed_coding_codebook(driver):
    """Not a re-recalibration of it: the candidate must differ from the shipped
    codec in exactly one way."""
    source = inspect.getsource(driver.fit_side_specification)

    assert 'spec["coding_codebook"].probabilities' in source
    assert "DEPLOYED" in source


def test_bd_rate_reuses_the_classical_curves_already_measured_on_this_split(driver):
    """Re-measuring x264/x265 would introduce a second scale; the parity report
    already has them on this exact split and convention."""
    source = inspect.getsource(driver.main)

    assert "parity_report" in source
    assert "bp.bd_rate(" in source
    assert "directly comparable" in source


def test_round_trip_failure_fails_the_run(driver):
    source = inspect.getsource(driver.main)

    assert '"all_round_trips_exact"' in source
    assert 'return 0 if report["all_round_trips_exact"] else 1' in source
