"""Tests for the M10K learned conditional entropy model.

A learned probability model is only usable as a codec component if it is
*exactly* reproducible: the decoder rebuilds the frequency tables by running the
same network on the same reference, and a single differing integer in any table
desynchronises the arithmetic coder for the rest of the frame. So most of these
tests are about determinism and the float-to-integer conversion, not about
accuracy.

The other thing under test is causality: the predictor's only input is `z_ref`,
which the decoder reconstructs before it touches the residual payload.
"""

from __future__ import annotations

import importlib.util
import math
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


def _model(channels: int = 4, alphabet: int = 16, hidden: int = 8, seed: int = 0):
    mk = _load_script("m10k_learned_entropy")
    torch.manual_seed(seed)
    model = mk.build_model({"latent_channels": channels, "alphabet": alphabet,
                            "hidden": hidden})
    model.eval()
    return mk, model


def _reference(batch: int = 1, channels: int = 4, size: int = 8, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, channels, size, size, generator=generator)


# --- the model -----------------------------------------------------------------


def test_the_model_predicts_a_distribution_at_every_position():
    mk, model = _model()
    reference = _reference()

    logits = model(reference)
    log_probabilities = model.log_probabilities(reference)

    assert logits.shape == (1, 4, 16, 8, 8), "one alphabet-sized vector per position"
    probabilities = log_probabilities.exp()
    assert torch.allclose(probabilities.sum(dim=2), torch.ones(1, 4, 8, 8), atol=1e-5)


def test_the_model_is_small():
    """The question is whether learned modelling helps at all, not how big a
    network can be trained - the brief caps this at well under 100k."""
    mk, model = _model(channels=64, alphabet=32, hidden=32)

    parameters = sum(p.numel() for p in model.parameters())

    assert parameters < 100_000, f"{parameters:,} parameters is over the M10K budget"


def test_the_prediction_depends_on_the_reference():
    """If it ignored z_ref it would be a marginal model with extra steps."""
    mk, model = _model()

    with torch.no_grad():
        a = model.log_probabilities(_reference(seed=1))
        b = model.log_probabilities(_reference(seed=2))

    assert not torch.allclose(a, b)


def test_channel_identity_reaches_the_prediction():
    """M10H and M10J get channel identity from having one table per channel; the
    learned model needs an explicit embedding to have the same information."""
    mk, model = _model()
    with torch.no_grad():
        model.channel_embedding.weight.normal_(0.0, 1.0)
        # Same content in every channel - any difference must come from the embedding.
        reference = _reference(channels=1).repeat(1, 4, 1, 1)
        log_probabilities = model.log_probabilities(reference)

    assert not torch.allclose(log_probabilities[0, 0], log_probabilities[0, 1])


def test_inference_is_deterministic():
    mk, model = _model()
    reference = _reference()

    with torch.no_grad():
        first = model.log_probabilities(reference)
        second = model.log_probabilities(reference)

    assert torch.equal(first, second)


def test_the_predictor_takes_only_the_reference():
    """Causality, structurally: there is no parameter through which the residual
    symbol, the current frame or a future frame could be supplied."""
    import inspect

    mk, model = _model()

    assert list(inspect.signature(model.forward).parameters) == ["reference"]
    assert list(inspect.signature(model.log_probabilities).parameters) == ["reference"]


# --- float probabilities -> coder frequencies -----------------------------------


@pytest.mark.parametrize("alphabet", [32, 16, 8])
def test_frequencies_total_exactly_and_never_hit_zero(alphabet):
    mk = _load_script("m10k_learned_entropy")
    rng = np.random.default_rng(0)
    probabilities = rng.dirichlet(np.ones(alphabet) * 0.3, size=500)

    frequencies = mk.probabilities_to_frequencies(probabilities)

    assert (frequencies.sum(axis=1) == TOTAL_FREQUENCY).all()
    assert (frequencies >= MIN_FREQUENCY).all(), "a zero frequency makes a symbol uncodable"


def test_an_extremely_peaked_distribution_still_codes_every_symbol():
    """Underflow is the dangerous case: a probability that rounds to zero would
    make its symbol impossible to code, and the encoder would crash mid-frame."""
    mk = _load_script("m10k_learned_entropy")
    probabilities = np.full((10, 16), 1e-12)
    probabilities[:, 0] = 1.0 - 15e-12

    frequencies = mk.probabilities_to_frequencies(probabilities)

    assert (frequencies >= MIN_FREQUENCY).all()
    assert (frequencies.sum(axis=1) == TOTAL_FREQUENCY).all()


def test_the_conversion_is_deterministic():
    """Encoder and decoder run this independently; the same probabilities must
    give byte-identical tables or the coder desynchronises."""
    mk = _load_script("m10k_learned_entropy")
    rng = np.random.default_rng(1)
    probabilities = rng.dirichlet(np.ones(16), size=200)

    assert np.array_equal(mk.probabilities_to_frequencies(probabilities),
                          mk.probabilities_to_frequencies(probabilities.copy()))


def test_ties_are_broken_deterministically():
    """A uniform distribution makes every entry equal, so the residual has to be
    distributed by a fixed rule rather than by whatever order argsort happens to
    return."""
    mk = _load_script("m10k_learned_entropy")
    uniform = np.full((4, 10), 0.1)

    first = mk.probabilities_to_frequencies(uniform)
    second = mk.probabilities_to_frequencies(uniform)

    assert np.array_equal(first, second)
    assert (first.sum(axis=1) == TOTAL_FREQUENCY).all()


def test_the_conversion_preserves_the_ordering_of_probabilities():
    mk = _load_script("m10k_learned_entropy")
    probabilities = np.array([[0.5, 0.3, 0.15, 0.05]])

    frequencies = mk.probabilities_to_frequencies(probabilities)[0]

    assert frequencies[0] > frequencies[1] > frequencies[2] > frequencies[3]


def test_a_malformed_probability_array_is_rejected():
    mk = _load_script("m10k_learned_entropy")

    with pytest.raises(ValueError, match=r"\[tables, alphabet\]"):
        mk.probabilities_to_frequencies(np.ones((2, 3, 4)))


# --- the per-frame table + the real coder ---------------------------------------


@pytest.mark.parametrize("bits", [5, 4, 3])
def test_a_per_frame_model_round_trips_through_the_real_coder(bits):
    """The audit finding in practice: one table per position, fed to the
    project's existing arithmetic coder unchanged."""
    mk, model = _model(channels=4, alphabet=2 ** bits, hidden=8)
    reference = _reference(channels=4, size=8)
    rng = np.random.default_rng(0)
    symbols = rng.integers(0, 2 ** bits, 4 * 8 * 8)

    entropy_model, table = mk.frame_entropy_model(model, reference, bits=bits)
    payload = encode_symbols(symbols, entropy_model.cumulative, table)
    decoded = decode_symbols(payload, symbols.size, entropy_model.cumulative, table)

    assert entropy_model.num_tables == 4 * 8 * 8, "one table per symbol position"
    assert np.array_equal(table, np.arange(symbols.size))
    assert np.array_equal(decoded, symbols)


def test_encoder_and_decoder_build_identical_tables():
    """The decoder rebuilds the tables from z_ref; they must match bit for bit."""
    mk, model = _model()
    reference = _reference()

    encoder_model, encoder_table = mk.frame_entropy_model(model, reference, bits=4)
    decoder_model, decoder_table = mk.frame_entropy_model(model, reference, bits=4)

    assert np.array_equal(encoder_model.frequencies, decoder_model.frequencies)
    assert np.array_equal(encoder_table, decoder_table)


def test_a_different_reference_produces_different_tables():
    mk, model = _model()
    with torch.no_grad():
        model.head.weight.normal_(0.0, 0.5)

    a, _ = mk.frame_entropy_model(model, _reference(seed=1), bits=4)
    b, _ = mk.frame_entropy_model(model, _reference(seed=2), bits=4)

    assert not np.array_equal(a.frequencies, b.frequencies)


def test_the_frame_model_identity_changes_with_the_reference():
    """Model identity is what the container checks; it must actually vary."""
    mk, model = _model()
    with torch.no_grad():
        model.head.weight.normal_(0.0, 0.5)

    a, _ = mk.frame_entropy_model(model, _reference(seed=1), bits=4)
    b, _ = mk.frame_entropy_model(model, _reference(seed=2), bits=4)

    assert a.model_id() != b.model_id()


# --- training ------------------------------------------------------------------


def test_training_reduces_the_rate_objective_and_selects_on_validation():
    mk = _load_script("m10k_learned_entropy")
    convention = _load_script("m10g_evaluation_convention")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    channels, size, alphabet = 4, 8, 16
    # A learnable relationship: the symbol is driven by the sign of the reference.
    references = [rng.normal(size=(channels, size, size)).astype(np.float32) for _ in range(24)]
    symbols = [np.where(r > 0, 3, 11).astype(np.int64) for r in references]
    model = mk.build_model({"latent_channels": channels, "alphabet": alphabet, "hidden": 8})

    history = mk.train_entropy_model(
        model, (symbols[:16], references[:16]), (symbols[16:], references[16:]),
        epochs=8, batch_size=4, learning_rate=1e-2, device=torch.device("cpu"))

    assert all(math.isfinite(r["val_nll_bits"]) for r in history)
    assert history[-1]["train_nll_bits"] < history[0]["train_nll_bits"], "no learning happened"
    assert history[-1]["val_nll_bits"] < math.log2(alphabet), "worse than a uniform model"
    selection = convention.select_checkpoint(history, objective_key="val_loss")
    assert selection["selection_domain"] == "validation"


def test_the_best_validation_weights_are_the_ones_left_in_the_model():
    """M10I's defect: reporting a selected epoch while keeping the final weights."""
    mk = _load_script("m10k_learned_entropy")
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    references = [rng.normal(size=(4, 8, 8)).astype(np.float32) for _ in range(16)]
    symbols = [np.where(r > 0, 2, 9).astype(np.int64) for r in references]
    model = mk.build_model({"latent_channels": 4, "alphabet": 16, "hidden": 8})

    history = mk.train_entropy_model(
        model, (symbols[:12], references[:12]), (symbols[12:], references[12:]),
        epochs=6, batch_size=4, learning_rate=1e-2, device=torch.device("cpu"))
    best = min(r["val_nll_bits"] for r in history)
    measured = mk.cross_entropy_bits(model, symbols[12:], references[12:],
                                     device=torch.device("cpu"))

    assert measured == pytest.approx(best, abs=1e-4), (
        "the model left in memory is not the best-validation one")


def test_a_saved_model_round_trips(tmp_path):
    mk, model = _model()
    with torch.no_grad():
        model.head.weight.normal_(0.0, 0.3)
    path = tmp_path / "entropy.pt"
    torch.save({"model_state_dict": model.state_dict(),
                "model_config": model.config_dict()}, path)

    restored, _ = mk.load_entropy_model(path)
    reference = _reference(seed=5)

    assert restored.config_dict() == model.config_dict()
    with torch.no_grad():
        assert torch.equal(restored.log_probabilities(reference),
                           model.log_probabilities(reference))


def test_the_gate_declares_its_rate_points_and_pure_rate_objective():
    mk = _load_script("m10k_learned_entropy")
    source = Path("scripts/m10k_learned_entropy.py").read_text(encoding="utf-8")

    assert mk.RATE_POINTS == (5, 4, 3)
    assert mk.FROZEN_LAMBDA == pytest.approx(3.0e-4)
    # No reconstruction loss anywhere in the training objective.
    assert "mse" not in source.lower().split("training\n----------")[1][:800]


@pytest.mark.parametrize("name", ["m10k_learned_entropy", "m10k_evaluate"])
def test_every_m10k_script_follows_the_project_script_contract(name):
    from nvc.utils.config import load_default_config

    mod = _load_script(name)

    assert callable(mod.build_arg_parser)
    assert callable(mod.main)
    assert mod.build_arg_parser(load_default_config()).parse_args([]) is not None


def test_m10k_does_not_modify_the_shipped_codec_or_earlier_containers():
    from nvc.compression import nvc_format
    from nvc.utils.config import load_default_config

    assert nvc_format.MAGIC == b"NVC1"
    assert nvc_format.STREAM_MAGIC == b"NVCS"
    m10g = _load_script("m10g_temporal_baseline")
    mc = _load_script("m10h_motion_compensation")
    assert m10g.TEMPORAL_FORMAT_VERSION == 1
    assert mc.TEMPORAL_FORMAT_VERSION == 2, "M10K reuses M10H's container unchanged"

    trainer = _load_script("train_autoencoder")
    actions = {a.dest for a in trainer.build_arg_parser(load_default_config())._actions}
    for required in ("rate_lambda", "rate_lr", "resume_model_only", "seed"):
        assert required in actions
