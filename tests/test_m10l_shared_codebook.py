"""Tests for the M10L shared entropy-table codebook.

A codebook that approximates M10K's per-position distributions is only usable as
a codec component if two things hold exactly, not approximately:

  * the codebook path can BE M10K - with one table per position mapped to
    itself, the emitted bytes must be byte-identical, or the "penalty vs M10K"
    it reports is measuring the wrong thing;
  * encoder and decoder derive the SAME table_index from the same z_ref with no
    side information, which makes the argmin tie-break and the one-off
    quantization of prototypes correctness requirements rather than details.

Most of what follows is about those two, plus the degenerate cases a clustering
step can hit in production (empty clusters, K larger than the number of distinct
distributions, adversarial distributions).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nvc.compression.entropy_model import MIN_FREQUENCY, TOTAL_FREQUENCY
from nvc.compression.range_coder import decode_symbols, encode_symbols


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ml():
    return _load_script("m10l_shared_codebook")


@pytest.fixture(scope="module")
def mk():
    return _load_script("m10k_learned_entropy")


def _distributions(rows: int, alphabet: int, seed: int = 0) -> np.ndarray:
    """Rows of plausible predicted distributions: peaked, in varied places."""
    rng = np.random.RandomState(seed)
    logits = rng.randn(rows, alphabet) * 2.0
    exponentiated = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _model(ml_module, channels: int = 4, alphabet: int = 16, hidden: int = 8, seed: int = 0):
    mk_module = _load_script("m10k_learned_entropy")
    torch.manual_seed(seed)
    model = mk_module.build_model({"latent_channels": channels, "alphabet": alphabet,
                                   "hidden": hidden})
    model.eval()
    return model


def _reference(channels: int = 4, size: int = 8, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, channels, size, size, generator=generator)


# --- 1. K = 16,384 equivalence with M10K ---------------------------------------


def test_identity_codebook_reproduces_m10k_exactly(ml, mk):
    """The zero-loss reference: same frequencies, same table_index, same bytes.

    This is the milestone's STOP condition. If a codebook holding each position's
    own distribution does not reproduce M10K, then every "penalty vs M10K" number
    measured through the codebook path is measuring plumbing, not clustering.
    """
    model = _model(ml, alphabet=16)
    reference = _reference()
    symbols = np.arange(4 * 8 * 8, dtype=np.int64) % 16

    entropy_model, m10k_index = mk.frame_entropy_model(model, reference, bits=4)
    probabilities = ml.frame_probabilities(model, reference).double().numpy()
    codebook, identity_index = ml.identity_codebook(probabilities, bits=4)

    assert np.array_equal(codebook.frequencies, entropy_model.frequencies)
    assert np.array_equal(identity_index, m10k_index)
    assert encode_symbols(symbols, codebook.cumulative, identity_index) == \
        encode_symbols(symbols, entropy_model.cumulative, m10k_index)


def test_identity_codebook_search_also_reproduces_m10k_bytes(ml, mk):
    """Not just the mapping - the nearest-prototype SEARCH must land on M10K too.

    Index ties are allowed (two identical distributions may both resolve to the
    lower index); identical tables give identical bytes either way, so the
    payload is what has to match.
    """
    model = _model(ml, alphabet=16)
    reference = _reference(seed=3)
    symbols = np.arange(4 * 8 * 8, dtype=np.int64) % 16

    entropy_model, m10k_index = mk.frame_entropy_model(model, reference, bits=4)
    probabilities = ml.frame_probabilities(model, reference).double().numpy()
    codebook, _ = ml.identity_codebook(probabilities, bits=4)

    searched = codebook.assign(probabilities)
    assert encode_symbols(symbols, codebook.cumulative, searched) == \
        encode_symbols(symbols, entropy_model.cumulative, m10k_index)


def test_identity_codebook_round_trips_through_the_existing_coder(ml):
    probabilities = _distributions(256, 16)
    codebook, table_index = ml.identity_codebook(probabilities, bits=4)
    symbols = (np.arange(256) * 7 % 16).astype(np.int64)

    payload = encode_symbols(symbols, codebook.cumulative, table_index)
    decoded = decode_symbols(payload, symbols.size, codebook.cumulative, table_index)

    assert np.array_equal(decoded, symbols)


# --- 2-3. deterministic construction and assignment ----------------------------


def test_codebook_construction_is_deterministic(ml):
    samples = _distributions(2000, 16, seed=1)
    first = ml.fit_codebook(samples, 32, bits=4, seed=42)
    second = ml.fit_codebook(samples, 32, bits=4, seed=42)

    assert np.array_equal(first.frequencies, second.frequencies)
    assert first.codebook_id() == second.codebook_id()


def test_a_different_seed_may_differ_but_stays_valid(ml):
    samples = _distributions(2000, 16, seed=1)
    codebook = ml.fit_codebook(samples, 32, bits=4, seed=7)

    assert codebook.frequencies.sum(axis=1).tolist() == [TOTAL_FREQUENCY] * 32
    assert (codebook.frequencies >= MIN_FREQUENCY).all()


def test_table_assignment_is_deterministic(ml):
    samples = _distributions(2000, 16, seed=2)
    codebook = ml.fit_codebook(samples, 16, bits=4)
    probe = _distributions(500, 16, seed=99)

    assert np.array_equal(codebook.assign(probe), codebook.assign(probe))


def test_assignment_ties_resolve_to_the_lowest_index(ml):
    """Encoder and decoder must agree on a tie, so the rule is pinned by test.

    Two identical prototypes make every row a tie; the answer must be index 0
    on any backend, which is why the reduction is over indices, not values.
    """
    uniform = np.full((2, 8), 1.0 / 8)
    codebook = ml.SharedCodebook(
        np.full((2, 8), TOTAL_FREQUENCY // 8, dtype=np.int64), bits=3)

    assert codebook.assign(uniform).tolist() == [0, 0]
    costs = np.array([[1.0, 1.0, 1.0], [2.0, 0.5, 0.5]])
    assert ml.argmin_lowest_index(costs).tolist() == [0, 1]


def test_numpy_and_torch_assignment_paths_agree(ml):
    """The deployed path assigns on the device; the reference path assigns in
    numpy. They must not disagree, or a stream encoded on one would not decode
    on the other."""
    samples = _distributions(1500, 16, seed=4)
    codebook = ml.fit_codebook(samples, 32, bits=4)
    probe = _distributions(400, 16, seed=5)

    numpy_index = codebook.assign(probe)
    torch_index = codebook.assign_tensor(torch.from_numpy(probe).double())

    assert np.array_equal(numpy_index, torch_index)


# --- 4-5. the coder's two hard invariants --------------------------------------


@pytest.mark.parametrize("bits,size", [(3, 8), (4, 16), (4, 64), (5, 32)])
def test_every_prototype_sums_to_exactly_total_frequency(ml, bits, size):
    samples = _distributions(1000, 2 ** bits, seed=bits)
    codebook = ml.fit_codebook(samples, size, bits=bits)

    assert codebook.frequencies.sum(axis=1).tolist() == [TOTAL_FREQUENCY] * size


@pytest.mark.parametrize("bits,size", [(3, 8), (4, 16), (4, 64), (5, 32)])
def test_no_prototype_has_a_zero_frequency(ml, bits, size):
    samples = _distributions(1000, 2 ** bits, seed=bits)
    codebook = ml.fit_codebook(samples, size, bits=bits)

    assert int(codebook.frequencies.min()) >= MIN_FREQUENCY


def test_a_codebook_violating_the_coder_invariants_cannot_be_built(ml):
    with pytest.raises(ValueError, match="sum to exactly"):
        ml.SharedCodebook(np.ones((4, 16), dtype=np.int64), bits=4)
    broken = np.full((2, 16), TOTAL_FREQUENCY // 16, dtype=np.int64)
    broken[0, 0], broken[0, 1] = 0, broken[0, 1] + broken[0, 0]
    with pytest.raises(ValueError):
        ml.SharedCodebook(broken, bits=4)


# --- 6. coder round trip -------------------------------------------------------


@pytest.mark.parametrize("size", [4, 16, 64])
def test_shared_codebook_round_trips_through_the_existing_coder(ml, size):
    samples = _distributions(2000, 16, seed=6)
    codebook = ml.fit_codebook(samples, size, bits=4)
    probe = _distributions(1024, 16, seed=7)
    table_index = codebook.assign(probe)
    symbols = (np.arange(1024) * 11 % 16).astype(np.int64)

    payload = encode_symbols(symbols, codebook.cumulative, table_index)
    decoded = decode_symbols(payload, symbols.size, codebook.cumulative, table_index)

    assert np.array_equal(decoded, symbols)


def test_the_coder_needs_no_modification_for_a_codebook(ml):
    """K tables is the same mechanism as 64 (M10H) or 16,384 (M10K): the coder
    never constrained the table count, which is why M10L needs no format change."""
    samples = _distributions(500, 8, seed=8)
    for size in (1, 3, 17, 129):
        codebook = ml.fit_codebook(samples, size, bits=3)
        probe = _distributions(64, 8, seed=size)
        table_index = codebook.assign(probe)
        symbols = (np.arange(64) % 8).astype(np.int64)
        payload = encode_symbols(symbols, codebook.cumulative, table_index)
        assert np.array_equal(
            decode_symbols(payload, 64, codebook.cumulative, table_index), symbols)


# --- 7. metric behaviour -------------------------------------------------------


def test_kl_and_cross_entropy_give_the_same_assignment(ml):
    """They differ by H(p), a per-row constant, so the argmin cannot differ.

    The module claims this in prose; here it is measured, because if it were
    false the gate's "code length" metric and its reported KL would be scoring
    different codebooks.
    """
    prototypes = _distributions(24, 16, seed=10)
    probe = _distributions(300, 16, seed=11)

    code_length = ml.prototype_costs(probe, prototypes, metric="code_length")
    divergence = ml.prototype_costs(probe, prototypes, metric="kl")

    assert np.array_equal(ml.argmin_lowest_index(code_length),
                          ml.argmin_lowest_index(divergence))
    assert (divergence >= -1e-9).all(), "KL divergence cannot be negative"


def test_l1_is_a_different_assignment(ml):
    """Kept as a real alternative, not a synonym - the gate selects between them
    on validation rather than assuming one is right."""
    prototypes = _distributions(16, 16, seed=12)
    probe = _distributions(300, 16, seed=13)

    l1 = ml.argmin_lowest_index(ml.prototype_costs(probe, prototypes, metric="l1"))
    code = ml.argmin_lowest_index(
        ml.prototype_costs(probe, prototypes, metric="code_length"))

    assert not np.array_equal(l1, code)


def test_an_unknown_metric_is_rejected(ml):
    with pytest.raises(ValueError, match="unknown metric"):
        ml.prototype_costs(_distributions(4, 8), _distributions(2, 8), metric="euclidean")


# --- 8. degenerate and adversarial inputs --------------------------------------


def test_k_larger_than_the_number_of_distinct_distributions(ml):
    """Three distinct rows, K=32. Must produce a valid codebook, not raise."""
    samples = np.repeat(_distributions(3, 16, seed=14), 40, axis=0)
    codebook = ml.fit_codebook(samples, 32, bits=4)

    assert codebook.size == 32
    assert codebook.frequencies.sum(axis=1).tolist() == [TOTAL_FREQUENCY] * 32
    assert codebook.provenance["occupied_clusters"] <= 32


def test_k_larger_than_the_sample_count(ml):
    samples = _distributions(5, 16, seed=15)
    codebook = ml.fit_codebook(samples, 16, bits=4)

    assert codebook.size == 16
    assert (codebook.frequencies >= MIN_FREQUENCY).all()


def test_empty_clusters_are_reseeded_not_dropped(ml):
    """Identical rows collapse most clusters; K must still be K afterwards."""
    samples = np.repeat(_distributions(2, 8, seed=16), 100, axis=0)
    codebook = ml.fit_codebook(samples, 24, bits=3)

    assert codebook.frequencies.shape == (24, 8)
    assert codebook.frequencies.sum(axis=1).tolist() == [TOTAL_FREQUENCY] * 24


@pytest.mark.parametrize("kind", ["uniform", "peaked", "random"])
def test_adversarial_distributions_still_produce_a_valid_codebook(ml, kind):
    """The three shapes that break naive float->int conversion: a uniform row
    where every entry rounds identically, a row where one symbol takes almost
    all the mass and the rest underflow, and unstructured noise."""
    rng = np.random.RandomState(17)
    if kind == "uniform":
        samples = np.full((200, 16), 1.0 / 16)
    elif kind == "peaked":
        samples = np.full((200, 16), 1e-9)
        samples[np.arange(200), rng.randint(0, 16, 200)] = 1.0
        samples = samples / samples.sum(axis=1, keepdims=True)
    else:
        samples = rng.random_sample((200, 16))
        samples = samples / samples.sum(axis=1, keepdims=True)

    codebook = ml.fit_codebook(samples, 8, bits=4)
    table_index = codebook.assign(samples)

    assert codebook.frequencies.sum(axis=1).tolist() == [TOTAL_FREQUENCY] * 8
    assert int(codebook.frequencies.min()) >= MIN_FREQUENCY
    symbols = (np.arange(200) % 16).astype(np.int64)
    payload = encode_symbols(symbols, codebook.cumulative, table_index)
    assert np.array_equal(
        decode_symbols(payload, 200, codebook.cumulative, table_index), symbols)


def test_fitting_rejects_impossible_inputs(ml):
    with pytest.raises(ValueError, match="codebook size must be"):
        ml.fit_codebook(_distributions(10, 16), 0, bits=4)
    with pytest.raises(ValueError, match="does not match"):
        ml.fit_codebook(_distributions(10, 16), 4, bits=3)
    with pytest.raises(ValueError, match="rows, alphabet"):
        ml.fit_codebook(np.zeros((2, 2, 2)), 4, bits=1)


# --- 9. reproducibility and serialisation --------------------------------------


def test_a_codebook_survives_a_round_trip_through_json(ml):
    """The decoder loads the codebook from disk; the tables it reads must be the
    integers the encoder used, not a re-derivation that could round differently."""
    samples = _distributions(1500, 16, seed=18)
    codebook = ml.fit_codebook(samples, 32, bits=4)

    restored = ml.SharedCodebook.from_dict(json.loads(json.dumps(codebook.to_dict())))

    assert np.array_equal(restored.frequencies, codebook.frequencies)
    assert np.array_equal(restored.cumulative, codebook.cumulative)
    assert restored.codebook_id() == codebook.codebook_id()


def test_an_unknown_codebook_version_is_rejected(ml):
    samples = _distributions(200, 8, seed=19)
    payload = ml.fit_codebook(samples, 4, bits=3).to_dict()
    payload["version"] = 99

    with pytest.raises(ValueError, match="unsupported codebook version"):
        ml.SharedCodebook.from_dict(payload)


def test_repeated_fits_across_processes_agree(ml):
    """Reproducibility across runs, not just within one: the fit is re-run from
    a freshly imported module."""
    samples = _distributions(800, 8, seed=20)
    first = ml.fit_codebook(samples, 16, bits=3, seed=5)
    second = _load_script("m10l_shared_codebook").fit_codebook(
        samples, 16, bits=3, seed=5)

    assert np.array_equal(first.frequencies, second.frequencies)


# --- 10. identity and provenance -----------------------------------------------


def test_identity_changes_with_model_calibration_and_size(ml):
    """A stale codebook/model/calibration pairing must be a different stream, so
    the container's existing identity check rejects it instead of the decoder
    silently paying more bits - the failure mode M10K measured at 14.8%."""
    samples = _distributions(1000, 16, seed=21)
    codebook = ml.fit_codebook(samples, 16, bits=4)
    base = codebook.codebook_id(model_identity=b"\x01" * 8, calibration_signature="a")

    assert codebook.codebook_id(model_identity=b"\x02" * 8,
                                calibration_signature="a") != base
    assert codebook.codebook_id(model_identity=b"\x01" * 8,
                                calibration_signature="b") != base
    assert ml.fit_codebook(samples, 32, bits=4).codebook_id(
        model_identity=b"\x01" * 8, calibration_signature="a") != base
    assert len(base) == 8, "must fit the .nvct v2 8-byte field without a format change"


def test_identity_changes_when_a_single_frequency_changes(ml):
    samples = _distributions(500, 8, seed=22)
    codebook = ml.fit_codebook(samples, 8, bits=3)
    mutated = codebook.frequencies.copy()
    mutated[0, 0] += 1
    mutated[0, 1] -= 1

    assert ml.SharedCodebook(mutated, bits=3).codebook_id() != codebook.codebook_id()


def test_fit_provenance_records_train_only_origin(ml):
    samples = _distributions(600, 8, seed=23)
    codebook = ml.fit_codebook(samples, 8, bits=3, provenance={"source": "train"})

    assert codebook.provenance["split"] == "train"
    assert codebook.provenance["training_rows"] == 600
    assert codebook.provenance["metric"] == "code_length"
    assert codebook.provenance["iterations"] >= 1


# --- 11. the deployed per-frame path -------------------------------------------


def test_frame_table_index_has_one_entry_per_symbol_in_coder_order(ml):
    model = _model(ml, channels=4, alphabet=16)
    reference = _reference(channels=4, size=8)
    codebook = ml.fit_codebook(_distributions(500, 16, seed=24), 16, bits=4)

    table_index = ml.frame_table_index(model, codebook, reference)

    assert table_index.shape == (4 * 8 * 8,)
    assert table_index.min() >= 0 and table_index.max() < codebook.size


def test_frame_probabilities_match_the_m10k_table_order(ml, mk):
    """Row i must be the distribution for flat symbol i, the same order M10K's
    `frame_entropy_model` uses - otherwise every symbol would be coded against
    the wrong position's model."""
    model = _model(ml, channels=4, alphabet=16)
    reference = _reference(channels=4, size=8, seed=25)

    entropy_model, _ = mk.frame_entropy_model(model, reference, bits=4)
    probabilities = ml.frame_probabilities(model, reference).double().numpy()

    assert np.array_equal(mk.probabilities_to_frequencies(probabilities),
                          entropy_model.frequencies)


def test_the_same_reference_always_yields_the_same_table_index(ml):
    model = _model(ml, channels=4, alphabet=16)
    reference = _reference(channels=4, size=8, seed=26)
    codebook = ml.fit_codebook(_distributions(500, 16, seed=27), 32, bits=4)

    first = ml.frame_table_index(model, codebook, reference)
    second = ml.frame_table_index(model, codebook, reference)

    assert np.array_equal(first, second)


def test_unique_tables_used_reports_actual_occupancy(ml):
    assert ml.unique_tables_used(np.array([0, 0, 1, 1, 5])) == 3
    assert ml.unique_tables_used(np.zeros(100, dtype=np.int64)) == 1


# --- 12. rate behaviour --------------------------------------------------------


def test_a_larger_codebook_does_not_score_worse_on_its_own_training_rows(ml):
    """Sanity on the fit itself: more prototypes must not increase the EXPECTED
    code length of the data they were fitted to.

    The quantity has to be the clustering objective - sum_i p_i . -log2 q_k(i),
    the bits a symbol drawn from p_i costs under its prototype. Scoring an
    arbitrary symbol sequence instead would measure that sequence's luck, not
    the codebook.
    """
    samples = _distributions(3000, 16, seed=28)

    costs = []
    for size in (4, 16, 64):
        codebook = ml.fit_codebook(samples, size, bits=4)
        table_index = codebook.assign(samples)
        expected = -np.log2(codebook.probabilities[table_index])
        costs.append(float((samples * expected).sum()))

    assert costs[1] <= costs[0] + 1e-6
    assert costs[2] <= costs[1] + 1e-6


def test_expected_bits_matches_a_direct_log2_sum(ml):
    codebook = ml.fit_codebook(_distributions(400, 8, seed=29), 8, bits=3)
    probe = _distributions(120, 8, seed=30)
    table_index = codebook.assign(probe)
    symbols = (np.arange(120) % 8).astype(np.int64)

    direct = float(-np.log2(codebook.probabilities[table_index, symbols]).sum())

    assert codebook.expected_bits(symbols, table_index) == pytest.approx(direct)


# --- 13. memory ----------------------------------------------------------------


def test_a_shared_codebook_is_orders_of_magnitude_smaller_than_m10k(ml):
    """The point of the exercise, stated as a test: 16,384 tables of 5-bit
    symbols need ~4 MB of cumulative array per frame; K=64 needs kilobytes,
    built once."""
    samples = _distributions(2000, 32, seed=31)
    codebook = ml.fit_codebook(samples, 64, bits=5)
    m10k_like = ml.SharedCodebook(
        np.full((16384, 32), TOTAL_FREQUENCY // 32, dtype=np.int64), bits=5)

    assert codebook.table_memory_bytes() * 100 < m10k_like.table_memory_bytes()


def test_sampling_training_distributions_is_bounded_and_deterministic(ml):
    model = _model(ml, channels=4, alphabet=16)
    references = [np.asarray(_reference(channels=4, size=8, seed=s)[0]) for s in range(4)]

    first = ml.sample_training_distributions(
        model, references, device=torch.device("cpu"), max_rows=200)
    second = ml.sample_training_distributions(
        model, references, device=torch.device("cpu"), max_rows=200)

    assert first.shape[0] <= 200 and first.shape[1] == 16
    assert np.array_equal(first, second)
    # float32 softmax, so rows sum to 1 only to single-precision accuracy.
    assert first.sum(axis=1) == pytest.approx(np.ones(first.shape[0]), abs=1e-6)


def test_sampling_rejects_empty_input(ml):
    model = _model(ml)
    with pytest.raises(ValueError, match="no reference frames"):
        ml.sample_training_distributions(model, [], device=torch.device("cpu"),
                                         max_rows=10)


def test_the_device_tie_break_is_bit_identical_to_the_reference_one(ml):
    """The device path takes a fast route (plain argmin) and only falls back to
    the explicit lowest-index reduction on rows that actually tie. That is an
    optimisation, so it has to be proved equivalent - including on inputs made
    entirely of ties, where the fast route would otherwise be free to disagree.
    """
    rng = np.random.RandomState(33)
    cases = [
        rng.random_sample((500, 64)),                       # generic, no ties
        np.zeros((50, 16)),                                 # every row all-ties
        np.tile(np.array([[1.0, 0.0, 0.0, 1.0]]), (40, 1)),  # ties at index 1 and 2
        np.repeat(rng.random_sample((30, 1)), 8, axis=1),   # each row constant
    ]
    for costs in cases:
        expected = ml.argmin_lowest_index(costs)
        actual = ml._torch_argmin_lowest_index(torch.from_numpy(costs)).numpy()
        assert np.array_equal(actual, expected)


def test_each_metric_uses_its_own_cost_minimising_centroid(ml):
    """Lloyd's algorithm only descends if the update minimises the SAME cost the
    assignment uses: the mean under KL/code length, the component-wise median
    under L1. Using the mean for both would handicap the L1 arm and make the
    metric comparison meaningless, so both objectives are checked to improve
    over their own initialisation.
    """
    samples = _distributions(1200, 8, seed=41)

    for metric in ("code_length", "l1"):
        codebook = ml.fit_codebook(samples, 8, bits=3, metric=metric)
        seeded = ml._seed_prototypes(samples, 8, metric=metric, seed=ml.DEFAULT_SEED)

        def objective(prototypes):
            costs = ml.prototype_costs(samples, prototypes, metric=metric)
            return float(costs[np.arange(len(samples)),
                               ml.argmin_lowest_index(costs)].sum())

        assert objective(codebook.probabilities) < objective(seeded), metric
