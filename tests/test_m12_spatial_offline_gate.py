"""M12 Phase B - tests for the spatial-context offline gate's NEW code.

Phase B reuses M11's causal-context primitives (scripts/m11_causal_context.py)
and estimator machinery UNCHANGED - their causality proofs, leak detection,
permutation-control behavior, smoothing and split discipline are already
covered by tests/test_m11_causal_context.py, and this file does not repeat
that coverage.

What IS new in M12 Phase B is `m12_spatial_offline_gate.m11_g16_prototype_indices`
- swapping the offline gate's "parent" grouping variable from M10L's codebook
index to the ACTUAL deployed M11-G16 model's own codebook index, computed with
the true previously-decoded symbols exactly as the encoder does - plus the
gate's assembly of M11's recalibration/context/control estimators around that
new parent. Those are what these tests exercise, along with the split-isolation
discipline (TRAIN/VAL-A/VAL-B/TEST) the gate depends on but which M11 never
had a standalone test for.
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
def gate():
    return _load_script("m12_spatial_offline_gate")


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
    # M11 models take (reference, planes) - use M11's OWN codebook-fitting
    # helper (m11_train.fit_model_codebook), not M10K/M10L's
    # sample_training_distributions (which calls log_probabilities(reference)
    # alone and does not exist for an M11 model).
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


# --- m11_g16_prototype_indices: the new parent-swap function ------------------


def test_prototype_indices_are_within_the_codebook_size(gate, ma, mk, ml, mt):
    model = _m11_model(ma, mk, group=2)
    symbols, references = _frames(5, seed=1)
    codebook = _codebook(ml, mt, model, symbols, references, k=6)

    indices = gate.m11_g16_prototype_indices(model, codebook, symbols, references, ZERO,
                                             device=torch.device("cpu"))

    assert indices.shape == (5, C * H * W)
    assert indices.min() >= 0
    assert indices.max() < codebook.size


def test_prototype_indices_match_the_single_frame_assignment_path(gate, ma, mk, ml, mt):
    # `m11_g16_prototype_indices` batches frames together for speed; each row
    # must still equal what `frame_table_index`-style single-frame assignment
    # (the deployed per-frame decode path) would give for that frame alone.
    model = _m11_model(ma, mk, group=2)
    symbols, references = _frames(4, seed=2)
    codebook = _codebook(ml, mt, model, symbols, references, k=6)

    batched = gate.m11_g16_prototype_indices(model, codebook, symbols, references, ZERO,
                                             device=torch.device("cpu"), batch_size=4)

    with torch.no_grad():
        for i in range(len(symbols)):
            single = gate.m11_g16_prototype_indices(
                model, codebook, symbols[i:i + 1], references[i:i + 1], ZERO,
                device=torch.device("cpu"), batch_size=1)
            assert np.array_equal(batched[i], single[0])


def test_prototype_indices_follow_c_major_flat_order(gate, ma, mk, ml, mt):
    # Row i of the returned [N, C*H*W] array must correspond to flat symbol
    # index i = c*H*W + y*W + x - the SAME order `symbols.reshape(-1)` uses -
    # or the gate would silently pair contexts/symbols/prototypes from
    # different positions.
    model = _m11_model(ma, mk, group=2)
    symbols, references = _frames(1, seed=3)
    codebook = _codebook(ml, mt, model, symbols, references, k=6)

    indices = gate.m11_g16_prototype_indices(model, codebook, symbols, references, ZERO,
                                             device=torch.device("cpu"))[0]

    with torch.no_grad():
        tensor_ref = torch.from_numpy(references[:1])
        tensor_sym = torch.from_numpy(symbols[:1])
        log_probabilities = model.log_probabilities(tensor_ref, model.planes(tensor_sym, ZERO))
        probabilities = log_probabilities.exp()[0]  # [C, A, H, W]
        expected_rows = probabilities.permute(0, 2, 3, 1).reshape(-1, ALPHABET)
        expected = codebook.assign_tensor(expected_rows)

    assert np.array_equal(indices, expected)


def test_prototype_indices_are_deterministic(gate, ma, mk, ml, mt):
    model = _m11_model(ma, mk, group=2)
    symbols, references = _frames(3, seed=4)
    codebook = _codebook(ml, mt, model, symbols, references, k=6)

    first = gate.m11_g16_prototype_indices(model, codebook, symbols, references, ZERO,
                                           device=torch.device("cpu"))
    second = gate.m11_g16_prototype_indices(model, codebook, symbols, references, ZERO,
                                            device=torch.device("cpu"))

    assert np.array_equal(first, second)


def test_prototype_indices_use_the_true_symbols_not_a_placeholder(gate, ma, mk, ml, mt):
    # The context planes must actually depend on the SUPPLIED symbols (the
    # true, previously-decoded ones) - not silently fall back to all-zero /
    # placeholder context, which would make every frame look context-free.
    model = _m11_model(ma, mk, group=2)
    symbols_a, references = _frames(2, seed=5)
    rng = np.random.default_rng(6)
    symbols_b = rng.integers(0, ALPHABET, symbols_a.shape).astype(np.int64)
    codebook = _codebook(ml, mt, model, symbols_a, references, k=6)

    indices_a = gate.m11_g16_prototype_indices(model, codebook, symbols_a, references, ZERO,
                                               device=torch.device("cpu"))
    indices_b = gate.m11_g16_prototype_indices(model, codebook, symbols_b, references, ZERO,
                                               device=torch.device("cpu"))

    # Channel 0 has no context at all (nothing decoded yet), so it must be
    # unaffected by which symbol array is "true" - later channels, whose
    # planes depend on the (different) earlier symbols, generally will differ.
    plane = H * W
    assert np.array_equal(indices_a[:, :plane], indices_b[:, :plane])
    assert not np.array_equal(indices_a[:, plane:], indices_b[:, plane:])


# --- gate arithmetic: recalibration / context / control, against the new parent -


def test_an_informative_spatial_context_beats_its_permuted_control_under_the_new_parent(cx):
    # Reproduces the gate's core estimator (fit_parent -> smoothed_parent ->
    # fit_child -> child_bits, with a whole-split-permuted control) on tiny
    # synthetic data where the PARENT is a coarse group (standing in for an
    # M11-G16 prototype k) and the CONTEXT is constructed to be genuinely
    # informative beyond that parent - the real arm must then show a clear
    # positive net gain, and the permuted control must be ~0, exactly the
    # pattern the real M12 gate run must also show to be trustworthy.
    rng = np.random.default_rng(0)
    alphabet, groups, card = 8, 3, 4
    n = 40_000
    train_groups = rng.integers(0, groups, n)
    train_context = rng.integers(0, card, n)
    # Symbol is a deterministic function of (group, context) plus small noise -
    # informative beyond the group-only marginal.
    train_symbols = (train_groups + 2 * train_context + rng.integers(0, 2, n)) % alphabet

    val_groups = rng.integers(0, groups, n)
    val_context = rng.integers(0, card, n)
    val_symbols = (val_groups + 2 * val_context + rng.integers(0, 2, n)) % alphabet

    uniform_prior = np.full((groups, alphabet), 1.0 / alphabet)
    parent_counts = cx.fit_parent(train_symbols, train_groups, groups, alphabet)
    parent = cx.smoothed_parent(parent_counts, uniform_prior, 1.0)
    parent_bits = cx.parent_bits(parent, val_symbols, val_groups) / val_symbols.size

    child_counts = cx.fit_child(train_symbols, train_groups, train_context, card, groups, alphabet)
    real_bits = cx.child_bits(child_counts, parent, val_symbols, val_groups, val_context, card,
                              1.0) / val_symbols.size

    control_rng = np.random.default_rng(1)
    train_context_shuffled = cx.permuted_context(train_context, control_rng)
    val_context_shuffled = cx.permuted_context(val_context, control_rng)
    control_child = cx.fit_child(train_symbols, train_groups, train_context_shuffled, card,
                                 groups, alphabet)
    control_bits = cx.child_bits(control_child, parent, val_symbols, val_groups,
                                 val_context_shuffled, card, 1.0) / val_symbols.size

    real_gain = (parent_bits - real_bits) / parent_bits * 100.0
    control_gain = (parent_bits - control_bits) / parent_bits * 100.0

    assert real_gain > 1.0          # the planted signal is real and sizeable
    assert abs(control_gain) < 0.2  # the shuffled control finds ~nothing
    assert real_gain - max(control_gain, 0.0) > 1.0  # net gain survives netting


def test_an_uninformative_context_shows_no_net_gain_under_the_new_parent(cx):
    # The negative control: when the context carries NO information beyond
    # the parent group, both the real and control arms must land near zero -
    # otherwise the estimator itself would manufacture a spurious "gain",
    # exactly the plug-in bias M11's random control exists to catch.
    rng = np.random.default_rng(2)
    alphabet, groups, card = 8, 3, 4
    n = 40_000
    train_groups = rng.integers(0, groups, n)
    train_context = rng.integers(0, card, n)  # independent of the symbol
    train_symbols = (train_groups + rng.integers(0, alphabet, n)) % alphabet

    val_groups = rng.integers(0, groups, n)
    val_context = rng.integers(0, card, n)
    val_symbols = (val_groups + rng.integers(0, alphabet, n)) % alphabet

    uniform_prior = np.full((groups, alphabet), 1.0 / alphabet)
    parent_counts = cx.fit_parent(train_symbols, train_groups, groups, alphabet)
    parent = cx.smoothed_parent(parent_counts, uniform_prior, 1.0)
    parent_bits = cx.parent_bits(parent, val_symbols, val_groups) / val_symbols.size

    child_counts = cx.fit_child(train_symbols, train_groups, train_context, card, groups, alphabet)
    real_bits = cx.child_bits(child_counts, parent, val_symbols, val_groups, val_context, card,
                              1.0) / val_symbols.size
    real_gain = (parent_bits - real_bits) / parent_bits * 100.0

    assert abs(real_gain) < 0.2


# --- split isolation: TRAIN / VAL-A / VAL-B / TEST -----------------------------


def test_the_gate_never_requests_the_test_split():
    # Structural guard: the gate must read TRAIN and VAL frames only, through
    # m11_data.load_or_collect, which itself only ever calls
    # discover_sequences(..., split="train") and split="val" - "test" must
    # not appear anywhere in the gate script's source.
    source = Path("scripts/m12_spatial_offline_gate.py").read_text(encoding="utf-8")
    assert 'split="test"' not in source
    assert "discover_sequences" not in source  # only m11_data.load_or_collect calls it


def test_val_a_and_val_b_are_disjoint_and_split_by_sequence(gate, ma):
    md = _load_script("m11_data")
    sequence_index = np.array([0, 0, 0, 1, 1, 2, 2, 2, 3])
    data = {"val_sequence_index": sequence_index}

    select_mask, report_mask = md.split_validation(data)

    assert not np.any(select_mask & report_mask)   # disjoint
    assert np.all(select_mask | report_mask)        # covers every frame
    # every frame of a given sequence lands on the SAME side of the split
    for sequence_id in np.unique(sequence_index):
        rows = sequence_index == sequence_id
        assert len(set(select_mask[rows].tolist())) == 1
