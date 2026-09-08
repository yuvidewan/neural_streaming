"""Tests for the M10J reference-conditioned entropy model.

Two things make this experiment meaningful, and both are easy to get subtly
wrong:

  * the context must be computable by the DECODER, from `z_ref` alone, before it
    touches the residual payload - otherwise the bits saved are bits the decoder
    cannot actually spend;
  * the tables must be fitted on TRAINING data only, or the measured gain is
    just memorised test statistics.

The tests below check both directly, along with the arithmetic coder's
requirements on the conditional tables (exact frequency totals, valid CDFs) and
the deterministic fallback that keeps sparse contexts from becoming unstable
tiny histograms.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.compression.range_coder import decode_symbols, encode_symbols


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _references(count: int = 12, channels: int = 4, size: int = 8, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(channels, size, size, generator=generator).numpy()
            for _ in range(count)]


def _symbols(count: int = 12, channels: int = 4, size: int = 8, bits: int = 4, seed: int = 1):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 2 ** bits, size=(channels, size, size)) for _ in range(count)]


# --- the context function ------------------------------------------------------


@pytest.mark.parametrize("scheme", ["magnitude4", "local_activity4"])
def test_context_generation_is_deterministic(scheme):
    ce = _load_script("m10j_conditional_entropy")
    references = _references()
    model = ce.fit_context_model(scheme, references)

    first = model.contexts(references[0])
    second = model.contexts(references[0])

    assert np.array_equal(first, second)


@pytest.mark.parametrize("scheme", ["magnitude4", "local_activity4"])
def test_contexts_stay_inside_the_declared_cardinality(scheme):
    ce = _load_script("m10j_conditional_entropy")
    references = _references()
    model = ce.fit_context_model(scheme, references)

    # Include an extreme reference the thresholds were never fitted on.
    contexts = model.contexts(references[0] * 1000.0)

    assert contexts.min() >= 0
    assert contexts.max() < model.cardinality


@pytest.mark.parametrize("scheme", ["magnitude4", "local_activity4"])
def test_the_context_depends_only_on_the_reference(scheme):
    """It is a pure function of z_ref: the residual, the current frame and the
    original previous frame are all absent from its signature and its inputs."""
    import inspect

    ce = _load_script("m10j_conditional_entropy")
    model = ce.fit_context_model(scheme, _references())

    parameters = list(inspect.signature(model.contexts).parameters)
    assert parameters == ["reference"], f"contexts() takes {parameters}"

    reference = _references(1, seed=7)[0]
    assert np.array_equal(model.contexts(reference), model.contexts(reference.copy()))


def test_a_different_reference_can_select_a_different_table():
    """If every reference mapped to the same context the model would be the
    marginal one wearing a different name."""
    ce = _load_script("m10j_conditional_entropy")
    references = _references()
    model = ce.fit_context_model("magnitude4", references)

    small = model.contexts(np.zeros_like(references[0]))
    large = model.contexts(np.full_like(references[0], 100.0))

    assert small.max() < large.min(), "magnitude context must separate small from large"
    channels, height, width = references[0].shape
    assert not np.array_equal(model.table_index(channels, height, width, small),
                              model.table_index(channels, height, width, large))


def test_the_marginal_scheme_is_a_single_context():
    ce = _load_script("m10j_conditional_entropy")
    model = ce.fit_context_model("marginal", _references())

    contexts = model.contexts(_references(1, seed=3)[0])

    assert model.cardinality == 1
    assert contexts.max() == 0


def test_table_index_matches_the_coders_symbol_order():
    """Symbols are flattened C-major; the table index must follow the same
    order or every symbol would be coded with the wrong channel's table."""
    ce = _load_script("m10j_conditional_entropy")
    model = ce.fit_context_model("magnitude4", _references())
    channels, height, width = 4, 2, 2
    contexts = np.arange(channels * height * width).reshape(channels, height, width) % 4

    table = model.table_index(channels, height, width, contexts)

    assert table.shape == (channels * height * width,)
    assert table.min() >= 0 and table.max() < channels * model.cardinality
    # First height*width entries belong to channel 0, and so on.
    assert all(table[i] // model.cardinality == i // (height * width) for i in range(table.size))


def test_an_unknown_context_scheme_is_rejected():
    ce = _load_script("m10j_conditional_entropy")

    with pytest.raises(ValueError, match="Unknown context scheme"):
        ce.ReferenceContextModel("not_a_scheme", None, 4)


def test_context_metadata_round_trips_and_a_malformed_one_is_rejected():
    ce = _load_script("m10j_conditional_entropy")
    model = ce.fit_context_model("local_activity4", _references())

    restored = ce.ReferenceContextModel.from_dict(model.to_dict())
    reference = _references(1, seed=11)[0]

    assert restored.scheme == model.scheme and restored.cardinality == model.cardinality
    assert np.array_equal(restored.contexts(reference), model.contexts(reference))
    with pytest.raises(ValueError, match="Unknown context scheme"):
        ce.ReferenceContextModel.from_dict({"scheme": "bogus", "cardinality": 4,
                                            "thresholds": None})


# --- table construction --------------------------------------------------------


def test_tables_are_built_from_training_symbols_only():
    ce = _load_script("m10j_conditional_entropy")
    references, symbols = _references(), _symbols()
    context_model = ce.fit_context_model("magnitude4", references)

    built = ce.build_conditional_entropy_model(symbols, references, context_model, bits=4)

    provenance = built["provenance"]
    assert provenance["split"] == "train"
    assert provenance["training_frames"] == len(symbols)
    assert provenance["training_symbols"] == sum(s.size for s in symbols)
    assert provenance["tables"] == symbols[0].shape[0] * context_model.cardinality


def test_the_conditional_model_has_one_table_per_channel_and_context():
    ce = _load_script("m10j_conditional_entropy")
    references, symbols = _references(), _symbols()

    marginal = ce.build_conditional_entropy_model(
        symbols, references, ce.fit_context_model("marginal", references), bits=4)
    conditional = ce.build_conditional_entropy_model(
        symbols, references, ce.fit_context_model("magnitude4", references), bits=4)

    channels = symbols[0].shape[0]
    assert marginal["entropy_model"].num_tables == channels
    assert conditional["entropy_model"].num_tables == channels * 4


def test_sparse_contexts_fall_back_to_the_marginal_distribution():
    """A context seen a handful of times would otherwise become an unstable
    histogram that costs more bits than the table it replaced."""
    ce = _load_script("m10j_conditional_entropy")
    references, symbols = _references(count=3), _symbols(count=3)
    context_model = ce.fit_context_model("magnitude4", references)

    built = ce.build_conditional_entropy_model(symbols, references, context_model, bits=4)

    # 3 frames x 8x8 = 192 symbols per channel, far below the 1000 threshold, so
    # every context must have fallen back.
    assert built["provenance"]["fallback_tables"] == 4 * context_model.cardinality
    frequencies = built["entropy_model"].frequencies.reshape(4, context_model.cardinality, -1)
    for channel in range(4):
        for context in range(1, context_model.cardinality):
            assert np.array_equal(frequencies[channel, 0], frequencies[channel, context])


def test_conditional_tables_satisfy_the_arithmetic_coders_requirements():
    ce = _load_script("m10j_conditional_entropy")
    references, symbols = _references(count=40), _symbols(count=40)
    built = ce.build_conditional_entropy_model(
        symbols, references, ce.fit_context_model("magnitude4", references), bits=4)
    model = built["entropy_model"]

    assert (model.frequencies.sum(axis=1) == 65536).all(), "each table must total exactly 65536"
    assert (model.frequencies > 0).all(), "a zero-frequency symbol would be uncodable"
    cumulative = model.cumulative
    assert (np.diff(cumulative, axis=1) > 0).all(), "CDF must be strictly increasing"
    assert (cumulative[:, 0] == 0).all() and (cumulative[:, -1] == 65536).all()


def test_the_conditional_model_id_differs_from_the_marginal_one():
    """This is what makes the container's existing mismatch check catch a
    marginal decoder handed a conditional stream - no format change needed."""
    ce = _load_script("m10j_conditional_entropy")
    references, symbols = _references(count=40), _symbols(count=40)

    marginal = ce.build_conditional_entropy_model(
        symbols, references, ce.fit_context_model("marginal", references), bits=4)
    conditional = ce.build_conditional_entropy_model(
        symbols, references, ce.fit_context_model("magnitude4", references), bits=4)

    assert marginal["entropy_model"].model_id() != conditional["entropy_model"].model_id()


# --- coding --------------------------------------------------------------------


@pytest.mark.parametrize("bits", [5, 4, 3])
def test_conditional_symbols_round_trip_exactly_at_every_rate_point(bits):
    ce = _load_script("m10j_conditional_entropy")
    references = _references(count=40)
    symbols = _symbols(count=40, bits=bits)
    context_model = ce.fit_context_model("local_activity4", references)
    built = ce.build_conditional_entropy_model(symbols, references, context_model, bits=bits)
    shape = symbols[0].shape
    contexts = context_model.contexts(references[0])

    payload = ce.encode_residual_symbols(
        symbols[0], contexts, shape=shape,
        entropy_model=built["entropy_model"], context_model=context_model)
    decoded = ce.decode_residual_symbols(
        payload, contexts, shape=shape,
        entropy_model=built["entropy_model"], context_model=context_model)

    assert np.array_equal(decoded, symbols[0].reshape(-1))


def test_decoding_with_the_wrong_context_does_not_silently_succeed():
    """The context is not side information the decoder may guess at - using a
    different one must corrupt the symbols, which is why encoder/decoder context
    equality is a hard requirement rather than an optimisation."""
    ce = _load_script("m10j_conditional_entropy")
    # Enough samples that no context falls back to the marginal table - with
    # fallbacks every table in a channel is identical and the context genuinely
    # would not matter, which is the fallback working, not a bug.
    references = _references(count=40, size=16)
    symbols = _symbols(count=40, size=16)
    context_model = ce.fit_context_model("magnitude4", references)
    built = ce.build_conditional_entropy_model(symbols, references, context_model, bits=4)
    assert built["provenance"]["fallback_tables"] == 0
    shape = symbols[0].shape

    right = context_model.contexts(references[0])
    wrong = context_model.contexts(references[1])
    payload = ce.encode_residual_symbols(
        symbols[0], right, shape=shape,
        entropy_model=built["entropy_model"], context_model=context_model)
    decoded = ce.decode_residual_symbols(
        payload, wrong, shape=shape,
        entropy_model=built["entropy_model"], context_model=context_model)

    assert not np.array_equal(decoded, symbols[0].reshape(-1))


def test_ideal_code_length_matches_the_emitted_payload():
    """The arithmetic coder should land within a fraction of a percent of the
    Shannon cost of its own tables; a large gap would mean the coder, not the
    model, is the bottleneck."""
    ce = _load_script("m10j_conditional_entropy")
    references = _references(count=60, size=16)
    symbols = _symbols(count=60, size=16)
    context_model = ce.fit_context_model("magnitude4", references)
    built = ce.build_conditional_entropy_model(symbols, references, context_model, bits=4)
    shape = symbols[0].shape
    contexts = context_model.contexts(references[0])

    payload = ce.encode_residual_symbols(
        symbols[0], contexts, shape=shape,
        entropy_model=built["entropy_model"], context_model=context_model)
    ideal = ce._ideal_code_bits(
        symbols[0], contexts, shape=shape,
        entropy_model=built["entropy_model"], context_model=context_model)

    assert abs(len(payload) * 8 - ideal) / ideal < 0.02
