"""M23 Phase B2 step 1 - tests for `scripts/m23_coded_validation.py`.

The load-bearing tests:

  * `test_the_residual_is_coded_against_the_DECODED_map` - the whole claim of
    this step. If the encoder's copy of the map leaks into the residual's table
    indices, the measurement describes something no decoder can reproduce;
  * `test_the_map_payload_is_counted_in_the_total` - a transmitted map that is
    not counted makes the gain free and the number meaningless;
  * `test_the_child_table_matches_the_distribution_child_bits_scores` - Phase B1
    measured a specific distribution; if the coder codes a different one, the
    realisation ratio compares two unrelated things;
  * `test_a_failed_round_trip_fails_the_run` - a byte count from a stream that
    does not decode is worthless.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cv():
    return _load_script("m23_coded_validation")


@pytest.fixture(scope="module")
def cx():
    return _load_script("m11_causal_context")


@pytest.fixture(scope="module")
def m10k():
    return _load_script("m10k_learned_entropy")


@pytest.fixture(scope="module")
def pb():
    return _load_script("m23_phaseb_gate")


def _uniformish(rows: int, alphabet: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probabilities = rng.random((rows, alphabet)) + 0.5
    return probabilities / probabilities.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------------------
# Decodability - the claim this step exists to make
# ----------------------------------------------------------------------------------------


def test_the_residual_is_coded_against_the_DECODED_map(cv):
    """The map is encoded, decoded back from its own payload, and only the decoded
    values may reach the residual's table indices. Using the encoder's copy would
    measure a codec that cannot exist."""
    source = inspect.getsource(cv.code_candidate_frame)

    decode_at = source.index("decoded_map = decode_symbols(")
    context_at = source.index("context = broadcast(")
    joint_at = source.index("joint = k * bins + context")
    assert decode_at < context_at < joint_at
    assert "broadcast(decoded_map.reshape(" in source
    # the encoder's copy must not be what the context is built from
    assert "broadcast(binned_map" not in source
    assert "never `binned_map`" in source


def test_the_map_round_trip_is_checked_separately_from_the_residual(cv):
    """If the map does not decode exactly, the context differs between encoder and
    decoder and the residual's exactness is luck. Both are asserted."""
    source = inspect.getsource(cv.code_candidate_frame)

    assert "map_exact = bool(np.array_equal(decoded_map, flat_map))" in source
    assert '"exact": bool(np.array_equal(decoded, symbols)) and map_exact' in source


def test_a_failed_round_trip_fails_the_run(cv):
    """A byte count from a stream that does not decode is worthless, so the exit
    status has to depend on exactness."""
    source = inspect.getsource(cv.main)

    assert '"all_round_trips_exact"' in source
    assert 'return 0 if report["all_round_trips_exact"] else 1' in source


# ----------------------------------------------------------------------------------------
# The charge
# ----------------------------------------------------------------------------------------


def test_the_map_payload_is_counted_in_the_total(cv):
    source = inspect.getsource(cv.code_candidate_frame)

    assert '"total_bytes": len(payload) + len(map_payload)' in source
    baseline = inspect.getsource(cv.code_baseline_frame)
    assert '"map_bytes": 0' in baseline, "the baseline transmits nothing"


def test_the_control_pays_the_same_charge_as_the_candidate(cv):
    """The control transmits a permuted map of identical size. If it were charged
    less, the net gain would be overstated."""
    source = inspect.getsource(cv.main)

    assert 'for arm, source in (("candidate", val_binned), ("control", permuted_binned))' \
        in source
    assert "code_candidate_frame(" in source
    # both arms go through the SAME coding function, so the charge cannot diverge
    assert source.count("code_candidate_frame(") == 1


def test_the_map_table_is_fitted_on_train_only(cv):
    source = inspect.getsource(cv.map_coder_table)

    assert "train_binned" in source
    assert "fitted on TRAIN only" in source


# ----------------------------------------------------------------------------------------
# The coded distribution is the one Phase B1 measured
# ----------------------------------------------------------------------------------------


def test_the_child_table_matches_the_distribution_child_bits_scores(cv, cx):
    """`child_bits` computes the smoothed probability per symbol to avoid building
    a huge array; the coder needs the table. Both must be the same distribution,
    or the realisation ratio compares two unrelated numbers."""
    rng = np.random.default_rng(0)
    groups, card, alphabet = 5, 4, 8
    child_counts = rng.integers(0, 7, size=(groups * card, alphabet)).astype(np.float64)
    parent = _uniformish(groups, alphabet, seed=1)
    strength = 3.0

    table = cv.smoothed_child_probabilities(child_counts, parent, card=card,
                                            strength=strength)

    assert table.shape == (groups * card, alphabet)
    assert np.allclose(table.sum(axis=1), 1.0)
    # score a handful of symbols both ways and require agreement
    symbols = rng.integers(0, alphabet, size=200)
    groups_index = rng.integers(0, groups, size=200)
    context = rng.integers(0, card, size=200)
    theirs = cx.child_bits(child_counts, parent, symbols, groups_index, context, card,
                           strength)
    joint = groups_index * card + context
    ours = float(-np.log2(table[joint, symbols]).sum())
    assert ours == pytest.approx(theirs, rel=1e-9)


def test_the_child_table_falls_back_to_the_parent_for_an_unseen_context(cv):
    """An unseen joint context must inherit its parent row exactly, which is what
    makes a 4096-table model safe on held-out data."""
    groups, card, alphabet = 3, 4, 8
    child_counts = np.zeros((groups * card, alphabet))
    parent = _uniformish(groups, alphabet, seed=2)

    table = cv.smoothed_child_probabilities(child_counts, parent, card=card, strength=10.0)

    for group in range(groups):
        for context in range(card):
            assert np.allclose(table[group * card + context], parent[group])


def test_the_coder_tables_are_valid_and_deterministic(cv, m10k):
    """Every row must total exactly 65536 with no zero entry, or the range coder
    refuses - and the conversion must be reproducible so encoder and decoder
    agree."""
    probabilities = _uniformish(40, 16, seed=3)

    first = cv.coder_tables(m10k, probabilities, bits=4)
    second = cv.coder_tables(m10k, probabilities, bits=4)

    assert first.frequencies.shape == (40, 16)
    assert (first.frequencies.sum(axis=1) == 1 << 16).all()
    assert (first.frequencies >= 1).all()
    assert np.array_equal(first.frequencies, second.frequencies)
    assert first.cumulative.shape == (40, 17)


def test_the_tables_go_through_m10ks_own_conversion(cv):
    """Not a second float-to-integer scheme invented here: encoder and decoder
    agreement across machines rests on that one deterministic converter."""
    source = inspect.getsource(cv.coder_tables)

    assert "m10k.probabilities_to_frequencies(probabilities)" in source


# ----------------------------------------------------------------------------------------
# A real round trip through the real coder
# ----------------------------------------------------------------------------------------


def test_a_real_candidate_frame_round_trips_and_pays_for_its_map(cv, m10k, pb):
    """End to end on synthetic data, through the actual range coder: the frame
    decodes exactly, and the reported total includes the map's bytes."""
    rng = np.random.default_rng(4)
    channels, height, width, bins, groups = 4, 8, 8, 8, 16
    alphabet = 16
    symbols = rng.integers(0, alphabet, size=channels * height * width)
    k = rng.integers(0, groups, size=symbols.size)
    binned = rng.integers(0, bins, size=(height // 2, width // 2))

    child_table = cv.coder_tables(m10k, _uniformish(groups * bins, alphabet, seed=5), bits=4)
    map_table = cv.map_coder_table(binned[None, ...], bins=bins)

    row = cv.code_candidate_frame(
        symbols, k, binned, child_table=child_table, map_table=map_table, bins=bins,
        channels=channels, height=height, width=width,
        broadcast=pb.broadcast_to_symbol_positions)

    assert row["exact"] is True
    assert row["map_exact"] is True
    assert row["map_bytes"] > 0
    assert row["total_bytes"] == row["residual_bytes"] + row["map_bytes"]


def test_a_real_baseline_frame_round_trips_and_transmits_nothing(cv, m10k):
    rng = np.random.default_rng(6)
    symbols = rng.integers(0, 16, size=512)
    k = rng.integers(0, 32, size=symbols.size)
    table = cv.coder_tables(m10k, _uniformish(32, 16, seed=7), bits=4)

    row = cv.code_baseline_frame(symbols, k, table)

    assert row["exact"] is True
    assert row["map_bytes"] == 0
    assert row["total_bytes"] == row["residual_bytes"]


def test_a_mismatched_map_would_break_the_residual_decode(cv, m10k, pb):
    """The negative control for decodability: if the decoder were handed a
    DIFFERENT map from the encoder's, the table indices diverge and the symbols
    come back wrong. This is what coding against the decoded map prevents."""
    rng = np.random.default_rng(8)
    channels, height, width, bins, groups = 4, 8, 8, 8, 16
    symbols = rng.integers(0, 16, size=channels * height * width)
    k = rng.integers(0, groups, size=symbols.size)
    binned = rng.integers(0, bins, size=(height // 2, width // 2))
    wrong = (binned + 1) % bins

    child_table = cv.coder_tables(m10k, _uniformish(groups * bins, 16, seed=9), bits=4)
    broadcast = pb.broadcast_to_symbol_positions

    def indices(source):
        context = broadcast(source[None, ...], channels=channels, height=height,
                            width=width).reshape(-1)
        return k * bins + context

    from nvc.compression.range_coder import decode_symbols, encode_symbols
    payload = encode_symbols(symbols, child_table.cumulative, indices(binned))
    decoded = decode_symbols(payload, symbols.size, child_table.cumulative, indices(wrong))

    assert not np.array_equal(decoded, symbols), (
        "a different map must corrupt the decode; if it does not, the context is not "
        "actually steering the coder and this gate measures nothing")


# ----------------------------------------------------------------------------------------
# Reporting and scope
# ----------------------------------------------------------------------------------------


def test_realisation_is_measured_against_b1s_own_prediction(cv):
    """The point of the step: ideal bits are not bytes, so the ratio has to be
    reported rather than assumed to be 1."""
    source = inspect.getsource(cv.main)

    assert '"b1_ideal_bits_gain_percent"' in source
    assert '"realisation_ratio"' in source
    assert 'r["arm"] == "scale_map"' in source


def test_the_container_change_is_named_as_out_of_scope(cv):
    """`.nvct` v2 cannot carry a third payload. Saying so in the report is what
    keeps this from reading as a shippable result."""
    source = inspect.getsource(cv)

    assert "out_of_scope" in source
    assert "format version" in source
    assert "No `src/nvc/` module is modified" in source


def test_the_control_is_credited_by_m11s_rule(cv):
    source = inspect.getsource(cv.main)

    assert "gate_a.net_of_control(" in source
    assert "gate_a.control_interpretation(" in source


def test_pre_registration_is_recorded(cv):
    source = inspect.getsource(cv.main)

    assert '"pre_registered"' in source
    for key in ("arms", "baseline", "primary_metric", "net_of_control_rule",
                "round_trip_required"):
        assert f'"{key}"' in source
    assert cv.ARMS == ("baseline", "candidate", "control")
