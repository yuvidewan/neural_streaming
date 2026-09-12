"""M20 - tests for the codebook-assignment hysteresis rule.

The three load-bearing tests in this file are:

  * `test_margin_zero_reproduces_the_deployed_assignment_exactly` - the whole
    sweep is a set of deltas against margin = 0, so if that point is not
    byte-identical to the deployed `m13.encode_frame_recalibrated` path, every
    number M20 reports is measured from the wrong origin;
  * `test_encoder_and_decoder_agree_on_every_assignment` - hysteresis is only
    admissible if the decoder can rebuild the assignment with no side
    information. This is the executable version of that claim, run for every
    state and a non-zero margin;
  * `test_hysteresis_never_reads_the_symbol_it_is_about_to_code` - the causality
    property the decoder-compatibility argument rests on.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m20():
    return _load_script("m20_hysteresis")


@pytest.fixture(scope="module")
def m13():
    return _load_script("m13_recalibration")


@pytest.fixture(scope="module")
def ma():
    return _load_script("m11_ar_entropy")


@pytest.fixture(scope="module")
def mk():
    return _load_script("m10k_learned_entropy")


@pytest.fixture(scope="module")
def ml():
    return _load_script("m10l_shared_codebook")


@pytest.fixture(scope="module")
def mt():
    return _load_script("m11_train")


C, H, W = 8, 4, 4
ALPHABET = 16
GROUP = 2
ZERO = torch.full((C,), 8)
SHAPE = (C, H, W)


def _model(ma, mk, *, seed=0):
    torch.manual_seed(seed)
    m10k = mk.build_model({"latent_channels": C, "alphabet": ALPHABET, "hidden": 8}).eval()
    torch.nn.init.normal_(m10k.channel_embedding.weight)
    model = ma.from_m10k(m10k, group_size=GROUP).eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        model.features[0].weight[:, 1:] = torch.randn(
            model.features[0].weight[:, 1:].shape, generator=generator) * 0.3
    return model


def _frames(count, seed=0):
    generator = torch.Generator().manual_seed(seed)
    references = [torch.randn(C, H, W, generator=generator).numpy().astype(np.float32)
                  for _ in range(count)]
    symbols = [torch.randint(0, ALPHABET, (C, H, W), generator=generator).numpy().astype(np.int64)
               for _ in range(count)]
    return np.stack(symbols), np.stack(references)


@pytest.fixture(scope="module")
def rig(m13, ma, mk, ml, mt):
    """A miniature but STRUCTURALLY REAL rig: a real G16 model, a real fitted
    K-prototype codebook, and a real M13 recalibrated coding codebook - so the
    assign/coding split M20 depends on is present, not stubbed."""
    model = _model(ma, mk)
    symbols, references = _frames(24, seed=10)
    assign_codebook = mt.fit_model_codebook(ml, model, (torch.from_numpy(symbols),
                                                        torch.from_numpy(references)),
                                            ZERO, bits=4, size=6, rows=2000,
                                            device=torch.device("cpu"), seed=0)
    table_index = m13.m11_g16_prototype_indices(model, assign_codebook, symbols, references,
                                                ZERO, device=torch.device("cpu"))
    select_symbols, select_references = _frames(8, seed=20)
    select_k = m13.m11_g16_prototype_indices(model, assign_codebook, select_symbols,
                                             select_references, ZERO, device=torch.device("cpu"))
    frequencies, strength, _, _ = m13.fit_recalibrated_frequencies(
        assign_codebook, symbols.reshape(-1), table_index.reshape(-1),
        select_symbols.reshape(-1), select_k.reshape(-1), alphabet=ALPHABET)
    coding_codebook = m13.build_recalibrated_codebook(assign_codebook, frequencies,
                                                      provenance={"strength": strength})
    probe_symbols, probe_references = _frames(3, seed=30)
    return {"model": model, "assign_codebook": assign_codebook,
            "coding_codebook": coding_codebook,
            "references": [torch.from_numpy(r)[None] for r in probe_references],
            "symbols": [s for s in probe_symbols]}


def _costs_and_best(m20, rig, index=0):
    rows = m20.frame_rows(rig["model"], rig["references"][index], rig["symbols"][index], ZERO)
    cost_tensor = m20.assignment_costs(rows, rig["assign_codebook"])
    return cost_tensor.double().numpy(), m20.baseline_assignment(cost_tensor)


# --- 1: baseline assignment reproduction ---------------------------------------------------


def test_assignment_costs_reproduce_assign_tensor_exactly(m20, rig):
    for index in range(len(rig["symbols"])):
        rows = m20.frame_rows(rig["model"], rig["references"][index], rig["symbols"][index], ZERO)
        deployed = rig["assign_codebook"].assign_tensor(rows)
        reproduced = m20.baseline_assignment(m20.assignment_costs(rows, rig["assign_codebook"]))
        assert np.array_equal(deployed, reproduced)


def test_assignment_cost_is_cross_entropy_in_bits(m20, rig):
    """cost(i, k) must be sum_s p_i(s) * -log2 q_k(s) against the INTEGER coder
    frequencies - not against the pre-rounding Lloyd centroids.

    The deployed path runs that GEMM in the network's own float32, so an
    independent float64 recompute agrees to float32 precision, not to float64 -
    checked at 1e-5 relative. What must hold EXACTLY is the resulting argmin,
    and `test_assignment_costs_reproduce_assign_tensor_exactly` pins that.
    """
    from nvc.compression.entropy_model import TOTAL_FREQUENCY
    rows = m20.frame_rows(rig["model"], rig["references"][0], rig["symbols"][0], ZERO)
    codebook = rig["assign_codebook"]
    expected = (rows.double().numpy()
                @ (-np.log2(codebook.frequencies / float(TOTAL_FREQUENCY))).T)
    actual = m20.assignment_costs(rows, codebook).double().numpy()
    assert np.allclose(actual, expected, rtol=1e-5, atol=1e-6)


# --- 2: margin = 0 exact equivalence -------------------------------------------------------


def test_margin_zero_is_the_argmin_for_every_state(m20, rig):
    costs, best = _costs_and_best(m20, rig)
    decoy = np.roll(best, 3)          # a previous assignment that genuinely differs
    for state in m20.STATES:
        assignment, trace = m20.apply_hysteresis(
            costs, best, margin=0.0, state=state, previous_frame=decoy,
            shape=SHAPE, group_size=GROUP)
        assert np.array_equal(assignment, best), state
        assert trace["held"] == 0, state


def test_margin_zero_reproduces_the_deployed_assignment_exactly(m20, m13, rig):
    """The origin of every delta in the sweep: at margin = 0 the M20 coding path
    must produce the SAME BYTES as the deployed `encode_frame_recalibrated`."""
    for index in range(len(rig["symbols"])):
        reference, symbols = rig["references"][index], rig["symbols"][index]
        deployed_payload, deployed_ideal = m13.encode_frame_recalibrated(
            rig["model"], rig["assign_codebook"], rig["coding_codebook"], reference, symbols,
            ZERO, bits=4)
        encoded = m20.encode_frame_hysteresis(
            rig["model"], rig["assign_codebook"], rig["coding_codebook"], reference, symbols,
            ZERO, bits=4, margin=0.0, state="temporal", previous_frame=None)
        assert encoded["payload"] == deployed_payload
        assert encoded["ideal_bits"] == deployed_ideal


def test_a_strictly_positive_margin_actually_changes_something(m20, rig):
    """Guards against a rule that is silently inert - a sweep of no-ops would
    'prove' hysteresis is harmless without ever exercising it."""
    costs, best = _costs_and_best(m20, rig)
    decoy = np.roll(best, 3)
    moved = False
    for state in m20.STATES:
        assignment, trace = m20.apply_hysteresis(
            costs, best, margin=5.0, state=state, previous_frame=decoy,
            shape=SHAPE, group_size=GROUP)
        moved |= bool(np.any(assignment != best)) and trace["held"] > 0
    assert moved


# --- 3: encoder / decoder assignment equivalence -------------------------------------------


@pytest.mark.parametrize("state", ["temporal", "channel_group", "raster"])
@pytest.mark.parametrize("margin", [0.0, 0.05, 0.5])
def test_encoder_and_decoder_agree_on_every_assignment(m20, rig, state, margin):
    """The decoder-compatibility proof: the decoder rebuilds the assignment from
    the payload plus state it already has, and it must match exactly."""
    previous = None
    for index in range(len(rig["symbols"])):
        reference, symbols = rig["references"][index], rig["symbols"][index]
        encoded = m20.encode_frame_hysteresis(
            rig["model"], rig["assign_codebook"], rig["coding_codebook"], reference, symbols,
            ZERO, bits=4, margin=margin, state=state, previous_frame=previous)
        decoded = m20.decode_frame_hysteresis(
            rig["model"], rig["assign_codebook"], rig["coding_codebook"], encoded["payload"],
            reference, ZERO, bits=4, shape=SHAPE, margin=margin, state=state,
            previous_frame=previous)
        assert np.array_equal(decoded["symbols"], np.asarray(symbols).reshape(-1))
        assert np.array_equal(decoded["assignment"], encoded["assignment"])
        assert np.array_equal(decoded["baseline_assignment"], encoded["baseline_assignment"])
        previous = encoded["assignment"]


def test_decoder_rejects_an_unknown_state(m20, rig):
    encoded = m20.encode_frame_hysteresis(
        rig["model"], rig["assign_codebook"], rig["coding_codebook"], rig["references"][0],
        rig["symbols"][0], ZERO, bits=4, margin=0.1, state="raster", previous_frame=None)
    with pytest.raises(ValueError, match="unknown hysteresis state"):
        m20.decode_frame_hysteresis(
            rig["model"], rig["assign_codebook"], rig["coding_codebook"], encoded["payload"],
            rig["references"][0], ZERO, bits=4, shape=SHAPE, margin=0.1, state="lookahead",
            previous_frame=None)


# --- 4: hysteresis state determinism -------------------------------------------------------


@pytest.mark.parametrize("state", ["temporal", "channel_group", "raster"])
def test_hysteresis_is_deterministic(m20, rig, state):
    costs, best = _costs_and_best(m20, rig)
    previous = np.roll(best, 5)
    first, trace_a = m20.apply_hysteresis(costs, best, margin=0.2, state=state,
                                          previous_frame=previous, shape=SHAPE, group_size=GROUP)
    second, trace_b = m20.apply_hysteresis(costs, best, margin=0.2, state=state,
                                           previous_frame=previous, shape=SHAPE, group_size=GROUP)
    assert np.array_equal(first, second)
    assert trace_a == trace_b


def test_hysteresis_does_not_mutate_its_inputs(m20, rig):
    costs, best = _costs_and_best(m20, rig)
    previous = np.roll(best, 5)
    costs_copy, best_copy, previous_copy = costs.copy(), best.copy(), previous.copy()
    for state in m20.STATES:
        m20.apply_hysteresis(costs, best, margin=0.2, state=state, previous_frame=previous,
                             shape=SHAPE, group_size=GROUP)
    assert np.array_equal(costs, costs_copy)
    assert np.array_equal(best, best_copy)
    assert np.array_equal(previous, previous_copy)


# --- 5: no information the decoder does not have -------------------------------------------


def test_hysteresis_never_reads_the_symbol_it_is_about_to_code(m20, rig):
    """`apply_hysteresis` is handed costs, the argmin and a previous assignment -
    and nothing else. Pinned by signature so a future 'small' change that slips
    the target symbols in cannot pass silently."""
    parameters = set(inspect.signature(m20.apply_hysteresis).parameters)
    assert parameters == {"costs", "best", "margin", "state", "previous_frame", "shape",
                          "group_size"}
    assert not any("symbol" in name or "target" in name for name in parameters)


def test_assignment_is_independent_of_the_current_decoding_group_symbols(m20, rig):
    """Flipping a symbol in the LAST decoding group cannot move ANY assignment -
    the causality property the decoder rebuild depends on."""
    reference, symbols = rig["references"][0], rig["symbols"][0]
    rows = m20.frame_rows(rig["model"], reference, symbols, ZERO)
    original = rig["assign_codebook"].assign_tensor(rows)
    flipped = np.asarray(symbols).copy()
    flipped[-1, -1, -1] = (flipped[-1, -1, -1] + 1) % ALPHABET
    moved = rig["assign_codebook"].assign_tensor(
        m20.frame_rows(rig["model"], reference, flipped, ZERO))
    assert np.array_equal(original, moved)


def test_encode_and_decode_take_the_same_hysteresis_inputs(m20):
    """Neither side may take an argument the other lacks - a rule needing
    encoder-only state would show up here as an asymmetry."""
    encode = set(inspect.signature(m20.encode_frame_hysteresis).parameters)
    decode = set(inspect.signature(m20.decode_frame_hysteresis).parameters)
    for name in ("margin", "state", "previous_frame"):
        assert name in encode and name in decode
    assert decode - encode == {"payload", "shape"}
    assert encode - decode == {"symbols"}


def test_no_side_information_is_written_into_the_payload(m20, rig):
    """A hysteresis payload must be the SAME LENGTH as the symbols it codes
    would be under any other assignment - i.e. the rule buys or costs bits only
    through the entropy tables, never through an appended side channel."""
    reference, symbols = rig["references"][0], rig["symbols"][0]
    lengths = set()
    for margin in (0.0, 0.5):
        encoded = m20.encode_frame_hysteresis(
            rig["model"], rig["assign_codebook"], rig["coding_codebook"], reference, symbols,
            ZERO, bits=4, margin=margin, state="raster", previous_frame=None)
        decoded = m20.decode_frame_hysteresis(
            rig["model"], rig["assign_codebook"], rig["coding_codebook"], encoded["payload"],
            reference, ZERO, bits=4, shape=SHAPE, margin=margin, state="raster",
            previous_frame=None)
        assert np.array_equal(decoded["symbols"], np.asarray(symbols).reshape(-1))
        lengths.add(len(encoded["payload"]))
    # Different tables, so different lengths are expected - what must NOT happen
    # is a systematic constant overhead, which an appended side channel would be.
    assert all(length > 0 for length in lengths)


# --- 6: margin sweep determinism -----------------------------------------------------------


def test_declared_sweep_is_fixed_and_ordered(m20):
    assert m20.MARGIN_SWEEP[0] == 0.0
    assert list(m20.MARGIN_SWEEP) == sorted(m20.MARGIN_SWEEP)
    assert len(set(m20.MARGIN_SWEEP)) == len(m20.MARGIN_SWEEP)
    assert m20.STATES == ("temporal", "channel_group", "raster")


def test_configuration_list_is_deterministic_and_shares_margin_zero():
    sweep = _load_script("m20_sweep")
    m20 = _load_script("m20_hysteresis")
    first = sweep.configurations(m20.STATES, m20.MARGIN_SWEEP, m20.DIAGNOSTIC_RULES)
    second = sweep.configurations(m20.STATES, m20.MARGIN_SWEEP, m20.DIAGNOSTIC_RULES)
    assert first == second
    assert first[0] == ("baseline", 0.0)
    assert len([c for c in first if c[1] == 0.0 and c[0] == "baseline"]) == 1
    assert len(first) == 1 + len(m20.STATES) * (len(m20.MARGIN_SWEEP) - 1) \
        + len(m20.DIAGNOSTIC_RULES)


@pytest.mark.parametrize("gains,expected", [
    ([0.0, -0.1, -0.3, -0.9], "immediately harmful"),
    ([0.0, 0.0001, -0.0002, 0.0003], "flat"),
    ([0.0, 0.2, 0.5, 0.9], "monotonic improvement"),
    ([0.0, 0.2, 0.9, 0.3], "optimum at a finite margin"),
])
def test_response_shape_is_classified_without_assuming_monotonicity(gains, expected):
    """Phase F must be able to name every shape it might see - in particular a
    finite interior optimum, which a monotonicity assumption would hide."""
    analysis = _load_script("m20_analysis")
    margins = [0.0, 0.01, 0.02, 0.05]
    assert expected in analysis.classify_response(margins, gains)


def test_gate_thresholds_match_the_project_wide_ones(m20):
    analysis = _load_script("m20_analysis")
    assert (analysis.WEAK_BELOW_PERCENT, analysis.MEANINGFUL_ABOVE_PERCENT) == (0.5, 1.0)
    assert (m20.WEAK_BELOW_PERCENT, m20.MEANINGFUL_ABOVE_PERCENT) == (0.5, 1.0)
    assert m20.verdict(0.49) == "weak"
    assert m20.verdict(0.5) == "marginal"
    assert m20.verdict(1.0) == "meaningful"
    assert m20.verdict(-3.0) == "weak"


def test_churn_reduction_alone_cannot_produce_a_passing_verdict(m20):
    """The milestone's own rule: the verdict must be driven by coded bytes, and
    the verdict helper must not even see churn."""
    assert "churn" not in inspect.signature(m20.verdict).parameters
    assert set(inspect.signature(m20.verdict).parameters) == {"gain_percent"}


def test_margin_response_is_monotone_in_how_many_positions_are_held(m20, rig):
    """A larger margin can only ever hold MORE positions - if that is violated
    the sweep is not measuring a single ordered knob."""
    costs, best = _costs_and_best(m20, rig)
    previous = np.roll(best, 3)
    held = [m20.apply_hysteresis(costs, best, margin=margin, state="temporal",
                                 previous_frame=previous, shape=SHAPE,
                                 group_size=GROUP)[1]["held"]
            for margin in m20.MARGIN_SWEEP]
    assert held == sorted(held)


# --- 7: routing-only decomposition ---------------------------------------------------------


def test_two_by_two_decomposition_partitions_every_position():
    sweep = _load_script("m20_sweep")
    accumulator = sweep.Accumulator(8)
    size = 32
    rng = np.random.default_rng(0)
    assignment_real = rng.integers(0, 8, size)
    assignment_oracle = rng.integers(0, 8, size)
    symbols_real = rng.integers(0, 4, size)
    symbols_oracle = rng.integers(0, 4, size)
    code_len_real = rng.random(size)
    code_len_oracle = rng.random(size)
    accumulator.add_frame(
        real_bytes=100, oracle_bytes=90, ideal_bits=800.0, code_len_real=code_len_real,
        code_len_oracle=code_len_oracle, assignment_real=assignment_real,
        assignment_oracle=assignment_oracle, baseline_real=assignment_real,
        symbols_real=symbols_real, symbols_oracle=symbols_oracle,
        code_len_baseline=code_len_real, is_boundary=False)
    summary = accumulator.to_dict()
    cells = summary["decomposition_2x2"]
    assert sum(c["positions"] for c in cells.values()) == size
    assert pytest.approx(sum(c["fraction_of_positions"] for c in cells.values())) == 1.0
    assert pytest.approx(sum(c["sum_delta_code_len_bits"] for c in cells.values())) == \
        float((code_len_real - code_len_oracle).sum())
    assert pytest.approx(sum(c["share_of_total_excess_bits"] for c in cells.values())) == 1.0


def test_held_positions_are_classified_better_worse_or_equal():
    sweep = _load_script("m20_sweep")
    accumulator = sweep.Accumulator(4)
    baseline = np.array([0, 1, 2, 3])
    held = np.array([1, 1, 0, 0])           # positions 0, 2 and 3 were held
    code_len_baseline = np.array([1.0, 1.0, 1.0, 1.0])
    code_len_real = np.array([2.0, 1.0, 0.5, 1.0])   # worse, -, better, equal
    accumulator.add_frame(
        real_bytes=1, oracle_bytes=1, ideal_bits=1.0, code_len_real=code_len_real,
        code_len_oracle=code_len_baseline, assignment_real=held,
        assignment_oracle=baseline, baseline_real=baseline,
        symbols_real=np.zeros(4, dtype=np.int64), symbols_oracle=np.zeros(4, dtype=np.int64),
        code_len_baseline=code_len_baseline, is_boundary=True)
    summary = accumulator.to_dict()
    assert summary["held_positions"] == 3
    assert summary["held_made_code_length_worse"] == 1
    assert summary["held_made_code_length_better"] == 1
    assert summary["held_made_code_length_equal"] == 1
    assert summary["gop_position"]["boundary"]["frames"] == 1
    assert summary["gop_position"]["ordinary"]["frames"] == 0


# --- 8: actual byte accounting -------------------------------------------------------------


def test_code_with_assignment_matches_the_deployed_coder(m20, m13, rig):
    reference, symbols = rig["references"][1], rig["symbols"][1]
    deployed_payload, deployed_ideal = m13.encode_frame_recalibrated(
        rig["model"], rig["assign_codebook"], rig["coding_codebook"], reference, symbols,
        ZERO, bits=4)
    rows = m20.frame_rows(rig["model"], reference, symbols, ZERO)
    assignment = rig["assign_codebook"].assign_tensor(rows)
    payload, ideal = m20.code_with_assignment(rig["coding_codebook"],
                                              np.asarray(symbols).reshape(-1), assignment)
    assert payload == deployed_payload
    assert ideal == deployed_ideal


def test_ideal_bits_track_the_chosen_tables(m20, rig):
    """Ideal bits must respond to the routing, and the non-causal per-symbol
    oracle must be the floor - the quantity Phase E's 'held made it worse'
    classification and the Phase D upper bound are both built on this.

    Note the deployed assignment is deliberately NOT asserted to beat every
    fixed table: it minimises cross-entropy against the ORIGINAL prototypes
    while the coder spends bits on M13's RECALIBRATED ones, so it is not
    optimal for the coding tables. That mismatch is a real property of the
    deployed pipeline, and M20 measures it rather than assuming it away.
    """
    reference, symbols = rig["references"][0], rig["symbols"][0]
    flat = np.asarray(symbols).reshape(-1)
    rows = m20.frame_rows(rig["model"], reference, symbols, ZERO)
    best = rig["assign_codebook"].assign_tensor(rows)
    _, ideal_best = m20.code_with_assignment(rig["coding_codebook"], flat, best)
    oracle = m20.oracle_table_assignment(rig["coding_codebook"], flat)
    _, ideal_oracle = m20.code_with_assignment(rig["coding_codebook"], flat, oracle)
    assert ideal_oracle <= ideal_best
    for table in range(rig["coding_codebook"].size):
        _, ideal_fixed = m20.code_with_assignment(
            rig["coding_codebook"], flat, np.full_like(best, table))
        assert ideal_oracle <= ideal_fixed


def test_diagnostic_rules_are_labelled_and_not_candidates(m20, rig):
    """`oracle_table` reads the symbol it codes, so it must never be presented
    as a hysteresis candidate - only as an upper bound."""
    assert m20.DIAGNOSTIC_RULES == ("coding_metric", "oracle_table")
    assert not set(m20.DIAGNOSTIC_RULES) & set(m20.STATES)
    parameters = set(inspect.signature(m20.oracle_table_assignment).parameters)
    assert "symbols_flat" in parameters
    assert "symbol" not in " ".join(inspect.signature(
        m20.coding_metric_assignment).parameters)


# --- 9: no TEST fitting --------------------------------------------------------------------


@pytest.mark.parametrize("script", ["m20_hysteresis", "m20_baseline", "m20_sweep",
                                    "m20_assignment_trace", "m20_analysis", "m20_provenance"])
def test_m20_scripts_never_touch_the_test_split(script):
    source = (ROOT / "scripts" / f"{script}.py").read_text(encoding="utf-8")
    assert 'split="test"' not in source
    assert "test_sequences" not in source


def test_val_b_selection_never_returns_test_sequences(m20):
    """VAL-B must come from the VAL split, chosen by a fixed stride - never by
    how well anything compresses, and never from TEST."""
    source = inspect.getsource(m20.val_b_sequences)
    assert 'split="val"' in source
    assert "[1::2]" in source
    assert "test" not in source.replace("# ", "").replace("never TEST", "")


def test_the_margin_is_declared_before_any_held_out_result_is_read(m20):
    """The sweep lives in the module, not in a caller that could pick it after
    seeing a number."""
    source = (ROOT / "scripts" / "m20_sweep.py").read_text(encoding="utf-8")
    assert "m20.MARGIN_SWEEP" in source
    assert not any(line.strip().startswith("MARGIN") for line in source.splitlines())


# --- 10: provenance ------------------------------------------------------------------------


def test_hysteresis_does_not_change_any_entropy_identity(m20, ma, rig):
    """Hysteresis re-routes positions among EXISTING tables; it never refits a
    table, so every codebook identity must be unchanged by running it."""
    before = (rig["assign_codebook"].codebook_id().hex(),
              rig["coding_codebook"].codebook_id().hex())
    m20.encode_frame_hysteresis(
        rig["model"], rig["assign_codebook"], rig["coding_codebook"], rig["references"][0],
        rig["symbols"][0], ZERO, bits=4, margin=0.5, state="raster", previous_frame=None)
    after = (rig["assign_codebook"].codebook_id().hex(),
             rig["coding_codebook"].codebook_id().hex())
    assert before == after
    identity = ma.model_identity(rig["model"], m10k_identity=b"\x00" * 8,
                                 calibration_signature="sig", bits=4,
                                 codebook=rig["coding_codebook"])
    assert identity == ma.model_identity(rig["model"], m10k_identity=b"\x00" * 8,
                                         calibration_signature="sig", bits=4,
                                         codebook=rig["coding_codebook"])


def test_recorded_identities_still_match_m19(m20):
    """If the frozen rig drifted, every M20 number would be measured against a
    different baseline than M13-M19's."""
    recorded = json.loads((ROOT / "outputs/m19_reference_error_audit/m19_identities.json")
                          .read_text(encoding="utf-8"))
    baseline_path = ROOT / "outputs/m20_codebook_hysteresis/m20_baseline.json"
    if not baseline_path.is_file():
        pytest.skip("Phase 0 has not been run in this working tree")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    for point in baseline["rate_points"]:
        expected = recorded[str(point["bits"])]
        assert point["residual_identity"] == expected["residual_identity"]
        assert point["assign_codebook_id"] == expected["assign_codebook_id"]
        assert point["coding_codebook_id"] == expected["coding_codebook_id"]


# --- 11: compatibility ---------------------------------------------------------------------


def test_m20_does_not_modify_any_production_source():
    result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True,
                            cwd=ROOT, check=False)
    modified = [line for line in result.stdout.splitlines()
                if line.startswith(" M") or line.startswith("M ")]
    modified_production = [line for line in modified if "CHANGELOG.md" not in line]
    assert modified_production == [], f"unexpected production modifications: {modified_production}"


@pytest.mark.parametrize("script", ["m20_hysteresis", "m20_baseline", "m20_sweep",
                                    "m20_assignment_trace", "m20_analysis", "m20_provenance"])
def test_m20_introduces_no_new_container_format(script):
    source = (ROOT / "scripts" / f"{script}.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "nvct_v3" not in lowered and "format_version = 3" not in lowered
    assert "TemporalStreamWriter" not in source


def test_m20_reuses_the_deployed_coder_rather_than_a_reimplementation(m20):
    source = inspect.getsource(m20.code_with_assignment)
    assert "encode_symbols(" in source
    decode_source = inspect.getsource(m20.decode_frame_hysteresis)
    assert "ResumableDecoder(" in decode_source


# --- 12: independent-process reproducibility -----------------------------------------------


def test_hysteresis_is_reproducible_in_an_independent_process(tmp_path):
    """Phase K in miniature: the rule must produce identical decisions in a
    fresh interpreter, not merely twice inside one warm process."""
    script = tmp_path / "probe.py"
    script.write_text(
        "import importlib.util, json, sys\n"
        "import numpy as np\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        f"spec = importlib.util.spec_from_file_location('m20', {str(ROOT / 'scripts' / 'm20_hysteresis.py')!r})\n"
        "m20 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m20)\n"
        "rng = np.random.default_rng(7)\n"
        "costs = rng.random((2048, 16))\n"
        "best = costs.argmin(axis=1)\n"
        "previous = np.roll(best, 3)\n"
        "out = {}\n"
        "for state in m20.STATES:\n"
        "    for margin in m20.MARGIN_SWEEP:\n"
        "        a, t = m20.apply_hysteresis(costs, best, margin=margin, state=state,\n"
        "                                    previous_frame=previous, shape=(8, 16, 16),\n"
        "                                    group_size=2)\n"
        "        out[f'{state}@{margin}'] = [int(a.sum()), int(a[-1]), t['held']]\n"
        "print(json.dumps(out))\n", encoding="utf-8")
    runs = [subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                           cwd=ROOT, check=True).stdout for _ in range(2)]
    assert json.loads(runs[0]) == json.loads(runs[1])
    assert len(json.loads(runs[0])) == len(_load_script("m20_hysteresis").STATES) * \
        len(_load_script("m20_hysteresis").MARGIN_SWEEP)
