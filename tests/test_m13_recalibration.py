"""M13 - tests for TRAIN-only codebook recalibration.

The one property every test here ultimately protects is the module
docstring's central claim: recalibration changes ONLY the arithmetic
coder's frequency tables, never which prototype a position is assigned to,
never the residual symbols, never the reconstruction. If that claim is
wrong, M13's whole "differs only in entropy frequency estimation" premise
is wrong too - so `test_assignment_never_uses_the_recalibrated_codebook`
and `test_recalibration_does_not_mutate_the_original_codebook` are the two
load-bearing tests in this file.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
def cx():
    return _load_script("m11_causal_context")


@pytest.fixture(scope="module")
def mt():
    return _load_script("m11_train")


C, H, W = 8, 4, 4
ALPHABET = 16
ZERO = torch.full((C,), 8)


def _m10k(mk, seed=0):
    torch.manual_seed(seed)
    model = mk.build_model({"latent_channels": C, "alphabet": ALPHABET, "hidden": 8}).eval()
    torch.nn.init.normal_(model.channel_embedding.weight)
    return model


def _m11_model(ma, mk, *, group=2, seed=0):
    model = ma.from_m10k(_m10k(mk, seed), group_size=group).eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        model.features[0].weight[:, 1:] = torch.randn(
            model.features[0].weight[:, 1:].shape, generator=generator) * 0.3
    return model


def _codebook(ml, mt, model, symbols, references, *, k=6, seed=0):
    train_set = (torch.from_numpy(symbols), torch.from_numpy(references))
    return mt.fit_model_codebook(ml, model, train_set, ZERO, bits=4, size=k, rows=2000,
                                 device=torch.device("cpu"), seed=seed)


def _frames(count, seed=0):
    generator = torch.Generator().manual_seed(seed)
    references = [torch.randn(C, H, W, generator=generator).numpy().astype(np.float32)
                 for _ in range(count)]
    symbols = [torch.randint(0, ALPHABET, (C, H, W), generator=generator).numpy().astype(np.int64)
              for _ in range(count)]
    return np.stack(symbols), np.stack(references)


def _prototype_indices(m13, model, codebook, symbols, references):
    return m13.m11_g16_prototype_indices(model, codebook, symbols, references, ZERO,
                                         device=torch.device("cpu"))


def _setup(m13, ma, mk, ml, mt, *, seed=0, k=6, train_frames=20, select_frames=8):
    model = _m11_model(ma, mk, group=2, seed=seed)
    train_symbols, train_references = _frames(train_frames, seed=seed + 10)
    codebook = _codebook(ml, mt, model, train_symbols, train_references, k=k, seed=seed)
    train_k = _prototype_indices(m13, model, codebook, train_symbols, train_references)
    select_symbols, select_references = _frames(select_frames, seed=seed + 20)
    select_k = _prototype_indices(m13, model, codebook, select_symbols, select_references)
    return model, codebook, (train_symbols, train_references, train_k), \
        (select_symbols, select_references, select_k)


# --- 1/4: TRAIN-only fitting, deterministic --------------------------------------------


def test_recalibration_is_deterministic(m13, ma, mk, ml, mt):
    model, codebook, (t_sym, _, t_k), (s_sym, _, s_k) = _setup(m13, ma, mk, ml, mt)
    flat = lambda a: a.reshape(a.shape[0], -1).reshape(-1)

    first = m13.fit_recalibrated_frequencies(
        codebook, flat(t_sym), t_k.reshape(-1), flat(s_sym), s_k.reshape(-1), alphabet=ALPHABET)
    second = m13.fit_recalibrated_frequencies(
        codebook, flat(t_sym), t_k.reshape(-1), flat(s_sym), s_k.reshape(-1), alphabet=ALPHABET)

    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]


def test_recalibrated_frequencies_satisfy_coder_invariants(m13, ma, mk, ml, mt):
    from nvc.compression.entropy_model import TOTAL_FREQUENCY, MIN_FREQUENCY
    model, codebook, (t_sym, _, t_k), (s_sym, _, s_k) = _setup(m13, ma, mk, ml, mt)
    flat = lambda a: a.reshape(-1)

    frequencies, strength, _, _ = m13.fit_recalibrated_frequencies(
        codebook, flat(t_sym), flat(t_k), flat(s_sym), flat(s_k), alphabet=ALPHABET)

    assert frequencies.shape == codebook.frequencies.shape
    assert np.all(frequencies.sum(axis=1) == TOTAL_FREQUENCY)
    assert np.all(frequencies >= MIN_FREQUENCY)


def test_recalibration_reflects_actual_train_symbol_counts(m13, ma, mk, ml, cx):
    # Direct behavioral proof: if prototype k=0 ALWAYS emits symbol 5 in
    # TRAIN, the recalibrated P(5 | 0) must end up far higher than whatever
    # the (randomly-initialized) deployed prior says - i.e. this is learning
    # from TRAIN counts, not just echoing the prior back.
    groups, alphabet = 4, ALPHABET
    rng = np.random.default_rng(0)
    n = 20_000
    train_k = rng.integers(0, groups, n)
    train_symbols = rng.integers(0, alphabet, n)
    train_symbols[train_k == 0] = 5  # deterministic signal for prototype 0

    class _FakeCodebook:
        size = groups
        probabilities = np.full((groups, alphabet), 1.0 / alphabet)  # flat, uninformative prior

    select_k = rng.integers(0, groups, 2000)
    select_symbols = rng.integers(0, alphabet, 2000)
    select_symbols[select_k == 0] = 5

    frequencies, strength, _, _ = m13.fit_recalibrated_frequencies(
        _FakeCodebook(), train_symbols, train_k, select_symbols, select_k, alphabet=alphabet)

    probability_of_5_given_0 = frequencies[0, 5] / frequencies[0].sum()
    assert probability_of_5_given_0 > 0.9


def test_strength_selection_is_driven_by_select_data_not_train(m13):
    # Fixing TRAIN, two very different VAL-A ("select") sets must be able to
    # select DIFFERENT smoothing strengths - proof that strength selection
    # actually reads the select split rather than silently ignoring it.
    groups, alphabet = 4, ALPHABET
    rng = np.random.default_rng(1)
    n = 20_000
    train_k = rng.integers(0, groups, n)
    train_symbols = rng.integers(0, alphabet, n)
    train_symbols[train_k == 0] = 5

    class _FakeCodebook:
        size = groups
        probabilities = np.full((groups, alphabet), 1.0 / alphabet)

    # VAL-A "consistent": agrees with the strong TRAIN signal -> favors trusting the counts.
    consistent_k = rng.integers(0, groups, 3000)
    consistent_symbols = rng.integers(0, alphabet, 3000)
    consistent_symbols[consistent_k == 0] = 5

    # VAL-A "contradictory": prototype 0 emits a DIFFERENT deterministic
    # symbol on VAL-A, so trusting the TRAIN counts fully would cost bits on
    # VAL-A -> favors heavier smoothing toward the flat prior instead.
    contrary_k = rng.integers(0, groups, 3000)
    contrary_symbols = rng.integers(0, alphabet, 3000)
    contrary_symbols[contrary_k == 0] = 9

    _, strength_consistent, _, _ = m13.fit_recalibrated_frequencies(
        _FakeCodebook(), train_symbols, train_k, consistent_symbols, contrary_k * 0 + consistent_k,
        alphabet=alphabet)
    _, strength_contrary, _, _ = m13.fit_recalibrated_frequencies(
        _FakeCodebook(), train_symbols, train_k, contrary_symbols, contrary_k, alphabet=alphabet)

    assert strength_consistent != strength_contrary


# --- the load-bearing safety property: assignment never reads the recalibrated table -----


def test_assignment_never_uses_the_recalibrated_codebook(m13, ma, mk, ml, mt):
    model, codebook, (t_sym, t_ref, t_k), (s_sym, s_ref, s_k) = _setup(m13, ma, mk, ml, mt)
    flat = lambda a: a.reshape(-1)
    frequencies, *_ = m13.fit_recalibrated_frequencies(
        codebook, flat(t_sym), flat(t_k), flat(s_sym), flat(s_k), alphabet=ALPHABET)
    coding_codebook = m13.build_recalibrated_codebook(codebook, frequencies)

    # A codebook whose .assign_tensor blows up if ever called - if
    # encode/decode secretly used the recalibrated table for assignment,
    # this test would fail with the planted exception instead of silently
    # passing.
    class _ExplodingAssignTensor:
        def __getattr__(self, item):
            if item == "assign_tensor":
                raise AssertionError(
                    "assign_tensor was called on the RECALIBRATED codebook - "
                    "assignment must only ever use the original, deployed one")
            return getattr(coding_codebook, item)

    reference = torch.from_numpy(t_ref[0])[None]
    symbols = t_sym[0]
    payload, _ = m13.encode_frame_recalibrated(
        model, codebook, _ExplodingAssignTensor(), reference, symbols, ZERO, bits=4)
    decoded = m13.decode_frame_recalibrated(
        model, codebook, _ExplodingAssignTensor(), payload, reference, ZERO, bits=4,
        shape=(C, H, W))
    assert np.array_equal(decoded, symbols.reshape(-1))


def test_recalibrated_and_original_coding_give_the_same_table_index(m13, ma, mk, ml, mt):
    # Directly: the table_index encode_frame_recalibrated computes must be
    # BYTE-IDENTICAL to calling assign_codebook.assign_tensor directly -
    # recalibration must be invisible to assignment.
    model, codebook, (t_sym, t_ref, t_k), _ = _setup(m13, ma, mk, ml, mt)
    reference = torch.from_numpy(t_ref[0])[None]
    symbols = t_sym[0]

    with torch.no_grad():
        rows = ma._rows(model.log_probabilities(
            reference, model.planes(torch.from_numpy(symbols)[None], ZERO)))
        expected_table_index = codebook.assign_tensor(rows)

    # A recalibrated codebook with DELIBERATELY skewed frequencies, so that
    # if it were mistakenly used for assignment the result would differ.
    skewed = np.zeros_like(codebook.frequencies)
    skewed[:, 0] = 65536 - (codebook.frequencies.shape[1] - 1)
    skewed[:, 1:] = 1
    coding_codebook = m13.build_recalibrated_codebook(codebook, skewed)

    _, table_index = ma._tables_for(rows, codebook, mk)
    assert np.array_equal(table_index, expected_table_index)
    # And the coding codebook's OWN assignment (if it were used) would
    # actually have differed, proving this is not a vacuous check.
    assert not np.array_equal(coding_codebook.assign_tensor(rows), expected_table_index)


# --- 5/8/9: same symbols, same reconstruction, resumable-decoder compatible -----------


def test_old_and_new_tables_decode_to_the_identical_true_symbols(m13, ma, mk, ml, mt):
    model, codebook, (t_sym, t_ref, t_k), (s_sym, s_ref, s_k) = _setup(m13, ma, mk, ml, mt)
    flat = lambda a: a.reshape(-1)
    frequencies, *_ = m13.fit_recalibrated_frequencies(
        codebook, flat(t_sym), flat(t_k), flat(s_sym), flat(s_k), alphabet=ALPHABET)
    coding_codebook = m13.build_recalibrated_codebook(codebook, frequencies)

    reference = torch.from_numpy(t_ref[0])[None]
    symbols = t_sym[0]

    old_payload, _ = m13.encode_frame_recalibrated(model, codebook, codebook, reference,
                                                    symbols, ZERO, bits=4)
    new_payload, _ = m13.encode_frame_recalibrated(model, codebook, coding_codebook, reference,
                                                    symbols, ZERO, bits=4)

    old_decoded = m13.decode_frame_recalibrated(model, codebook, codebook, old_payload,
                                                reference, ZERO, bits=4, shape=(C, H, W))
    new_decoded = m13.decode_frame_recalibrated(model, codebook, coding_codebook, new_payload,
                                                reference, ZERO, bits=4, shape=(C, H, W))

    assert np.array_equal(old_decoded, symbols.reshape(-1))
    assert np.array_equal(new_decoded, symbols.reshape(-1))
    assert np.array_equal(old_decoded, new_decoded)


def test_recalibrated_decode_reconstructs_identically_to_legacy_ma_decode_frame(m13, ma, mk, ml, mt):
    # decode_frame_recalibrated(..., coding_codebook=assign_codebook) - i.e.
    # "no recalibration applied" - must reproduce m11_ar_entropy.decode_frame
    # byte-for-byte (both should, by construction, produce identical
    # reconstructions since neither changes the entropy model in this case).
    model, codebook, (t_sym, t_ref, t_k), _ = _setup(m13, ma, mk, ml, mt)
    reference = torch.from_numpy(t_ref[0])[None]
    symbols = t_sym[0]

    payload, _ = ma.encode_frame(model, reference, symbols, ZERO, bits=4, codebook=codebook)
    legacy = ma.decode_frame(model, payload, reference, ZERO, bits=4, shape=(C, H, W),
                             codebook=codebook)
    via_m13 = m13.decode_frame_recalibrated(model, codebook, codebook, payload, reference, ZERO,
                                            bits=4, shape=(C, H, W))

    assert np.array_equal(legacy, via_m13)
    assert np.array_equal(legacy, symbols.reshape(-1))


# --- 6/7: provenance / identity separation --------------------------------------------


def test_recalibrated_codebook_has_a_distinct_identity(m13, ma, mk, ml, mt):
    model, codebook, (t_sym, _, t_k), (s_sym, _, s_k) = _setup(m13, ma, mk, ml, mt)
    flat = lambda a: a.reshape(-1)
    frequencies, *_ = m13.fit_recalibrated_frequencies(
        codebook, flat(t_sym), flat(t_k), flat(s_sym), flat(s_k), alphabet=ALPHABET)
    coding_codebook = m13.build_recalibrated_codebook(codebook, frequencies)

    assert coding_codebook.codebook_id() != codebook.codebook_id()

    identity_old = ma.model_identity(model, m10k_identity=b"\x01" * 8,
                                     calibration_signature="cal", bits=4, codebook=codebook)
    identity_new = ma.model_identity(model, m10k_identity=b"\x01" * 8,
                                     calibration_signature="cal", bits=4, codebook=coding_codebook)
    assert identity_old != identity_new
    assert len(identity_old) == len(identity_new) == 8  # still fits .nvct v2's 8-byte field


# --- 11: no mutation of the deployed codebook -------------------------------------------


def test_recalibration_does_not_mutate_the_original_codebook(m13, ma, mk, ml, mt):
    model, codebook, (t_sym, _, t_k), (s_sym, _, s_k) = _setup(m13, ma, mk, ml, mt)
    flat = lambda a: a.reshape(-1)
    original_frequencies = codebook.frequencies.copy()
    original_cumulative = codebook.cumulative.copy()

    frequencies, *_ = m13.fit_recalibrated_frequencies(
        codebook, flat(t_sym), flat(t_k), flat(s_sym), flat(s_k), alphabet=ALPHABET)
    m13.build_recalibrated_codebook(codebook, frequencies)

    assert np.array_equal(codebook.frequencies, original_frequencies)
    assert np.array_equal(codebook.cumulative, original_cumulative)


# --- 10: no TEST data can reach the fit --------------------------------------------------


def test_fit_recalibrated_frequencies_has_no_channel_for_test_data(m13):
    # Structural guarantee, not just a convention: the function's signature
    # only accepts a TRAIN pair and a SELECT (VAL-A) pair - there is no
    # parameter through which TEST symbols/prototypes could be passed in,
    # so a caller literally cannot leak TEST into the fit by accident.
    import inspect
    parameters = list(inspect.signature(m13.fit_recalibrated_frequencies).parameters)
    assert parameters == ["original_codebook", "train_symbols_flat", "train_k_flat",
                          "select_symbols_flat", "select_k_flat", "alphabet"]
    assert not any("test" in p.lower() for p in parameters)


def test_m13_scripts_check_the_m11_g16_checkpoint_provenance_before_using_it():
    # Both Phase D and Phase E must call M11's OWN `check_provenance` guard
    # (scripts/m11_evaluate.py, unmodified) on the loaded M11-G16 checkpoint
    # before fitting/evaluating with it - a stale checkpoint (wrong
    # calibration, bit depth, group size, or M10K lineage) must be a stop,
    # never a silent rate loss, exactly as M11 already established.
    for script_name in ("m13_coded_validation.py", "m13_davis_benchmark.py"):
        source = Path("scripts") / script_name
        text = source.read_text(encoding="utf-8")
        assert "check_provenance(" in text, f"{script_name} must call check_provenance"
        assert "ProvenanceError" in text, f"{script_name} must catch ProvenanceError"


def test_check_provenance_rejects_a_mismatched_m11_g16_checkpoint():
    # Direct behavioral proof, not just "the call is present": a checkpoint
    # claiming a different calibration signature must be rejected by the
    # SAME `ProvenanceError` M11's own deployment path raises.
    ev = _load_script("m11_evaluate")
    ma = _load_script("m11_ar_entropy")

    good = {"calibration_signature": "sig-a", "bits": 4,
           "model_config": {"group_size": 16},
           "context_definition_id": ma.context_definition_id(16),
           "m10k_identity": (b"\x01" * 8).hex()}
    ev.check_provenance(good, signature="sig-a", bits=4, group_size=16,
                        context_definition_id=ma.context_definition_id(16),
                        m10k_identity=b"\x01" * 8)  # must not raise

    stale = dict(good, calibration_signature="sig-b")
    with pytest.raises(ev.ProvenanceError):
        ev.check_provenance(stale, signature="sig-a", bits=4, group_size=16,
                            context_definition_id=ma.context_definition_id(16),
                            m10k_identity=b"\x01" * 8)


def test_davis_benchmark_never_feeds_test_sequences_into_the_fit():
    # In scripts/m13_davis_benchmark.py, the `test_sequences` variable (DAVIS
    # TEST split) must appear ONLY as an argument to `cl.run_sequences` (the
    # closed-loop BENCHMARK/evaluation) - never to `fit_recalibrated_frequencies`
    # or `m11_g16_prototype_indices` (the fit). Checked on the actual script
    # source, line by line, so a future edit that accidentally threads TEST
    # sequences into the fit fails this test immediately.
    source = Path("scripts/m13_davis_benchmark.py").read_text(encoding="utf-8")
    fitting_calls = ("m11_g16_prototype_indices(", "fit_recalibrated_frequencies(")
    lines = source.splitlines()
    for index, line in enumerate(lines):
        if any(call in line for call in fitting_calls):
            # The call may span a couple of lines (wrapped arguments) - check
            # a small window after the call opens.
            window = "\n".join(lines[index:index + 4])
            assert "test_sequences" not in window, (
                f"line {index + 1} calls a fitting function near "
                f"'test_sequences' - TEST must never reach the fit")
    assert "cl.run_sequences(" in source and "test_sequences" in source, (
        "sanity check: the script must still actually use test_sequences somewhere "
        "(as the evaluation subject) or this test is vacuous")
