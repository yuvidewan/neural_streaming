"""Tests for the M11 channel-autoregressive entropy model.

An autoregressive entropy model fails silently when it fails: a context that
reads one symbol too far, or an encoder that computes a probability with
different arithmetic than the decoder, still produces a plausible-looking
stream - one no decoder can read. So most of these tests are about causality
and encoder/decoder agreement, checked directly rather than inferred from a
round trip that might pass by luck.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from nvc.compression.entropy_model import MIN_FREQUENCY, TOTAL_FREQUENCY


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ma():
    return _load_script("m11_ar_entropy")


@pytest.fixture(scope="module")
def mk():
    return _load_script("m10k_learned_entropy")


C, H, W = 8, 6, 6
ZERO = torch.full((C,), 8)


def _m10k(mk, seed=0, alphabet=16):
    torch.manual_seed(seed)
    model = mk.build_model({"latent_channels": C, "alphabet": alphabet, "hidden": 8}).eval()
    torch.nn.init.normal_(model.channel_embedding.weight)
    return model


def _model(ma, mk, *, group=1, seed=0, alphabet=16, live=True):
    """An M11 model whose context weights are non-zero, so context actually matters."""
    model = ma.from_m10k(_m10k(mk, seed, alphabet), group_size=group).eval()
    if live:
        generator = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            model.features[0].weight[:, 1:] = torch.randn(
                model.features[0].weight[:, 1:].shape, generator=generator) * 0.5
    return model


def _inputs(seed=0, alphabet=16):
    generator = torch.Generator().manual_seed(seed)
    reference = torch.randn(1, C, H, W, generator=generator)
    symbols = torch.randint(0, alphabet, (C, H, W), generator=generator).numpy()
    return reference, symbols


# --- warm start and ablation --------------------------------------------------------


def test_warm_start_reproduces_m10k_exactly(ma, mk):
    """At step 0 M11 IS M10K, so any gain is attributable to training from there."""
    m10k = _m10k(mk)
    model = ma.from_m10k(m10k).eval()
    reference, symbols = _inputs()
    planes = model.planes(torch.from_numpy(symbols)[None], ZERO)

    with torch.no_grad():
        assert torch.equal(model.log_probabilities(reference, planes),
                           m10k.log_probabilities(reference))


def test_the_no_context_ablation_ignores_the_planes(ma, mk):
    model = ma.from_m10k(_m10k(mk), use_context=False).eval()
    with torch.no_grad():
        model.features[0].weight[:, 1:] = 1.0            # would matter if planes were used
    reference, symbols = _inputs()
    planes = model.planes(torch.from_numpy(symbols)[None], ZERO)

    with torch.no_grad():
        assert torch.equal(model(reference, planes), model(reference, torch.zeros_like(planes)))


def test_parameter_count_is_well_under_budget(ma, mk):
    """Brief target: < 100k. The real configuration (64 channels, 32 hidden)."""
    real = ma.ChannelContextEntropyModel(64, 32, 32)
    parameters = sum(p.numel() for p in real.parameters())
    assert parameters < 20_000, parameters


# --- causality ------------------------------------------------------------------------


@pytest.mark.parametrize("group", [1, 2, 4, 8])
def test_planes_use_only_channels_before_the_current_group(ma, group):
    """Perturb every channel from the current group's start onward; the planes
    for that channel must not move. That is exactly what the decoder lacks."""
    symbols = torch.randint(0, 16, (1, C, H, W), generator=torch.Generator().manual_seed(3))
    base = ma.context_planes(symbols, ZERO, group)
    starts = ma.group_starts(C, group)
    for channel in range(C):
        perturbed = symbols.clone()
        perturbed[:, int(starts[channel]):] = torch.randint(
            0, 16, perturbed[:, int(starts[channel]):].shape)
        assert torch.equal(ma.context_planes(perturbed, ZERO, group)[:, channel],
                           base[:, channel]), f"channel {channel} leaks"


@pytest.mark.parametrize("group", [1, 2, 4])
def test_model_outputs_are_causal_end_to_end(ma, mk, group):
    """Not just the planes - the network output for a channel must not depend
    on its own group or anything later."""
    model = _model(ma, mk, group=group)
    reference, symbols = _inputs(1)
    starts = ma.group_starts(C, group)
    tensor = torch.from_numpy(symbols)[None]
    with torch.no_grad():
        base = model.log_probabilities(reference, model.planes(tensor, ZERO))
        for channel in (0, 3, C - 1):
            perturbed = tensor.clone()
            perturbed[:, int(starts[channel]):] = (perturbed[:, int(starts[channel]):] + 5) % 16
            again = model.log_probabilities(reference, model.planes(perturbed, ZERO))
            assert torch.equal(again[:, channel], base[:, channel])


def test_the_causality_test_would_catch_a_leak(ma):
    """A plane that reads the CURRENT channel must fail the same check."""
    symbols = torch.randint(0, 16, (1, C, H, W), generator=torch.Generator().manual_seed(4))
    leaky = lambda s: (s - ZERO.view(1, -1, 1, 1)).abs().float()
    perturbed = symbols.clone()
    perturbed[:, 3:] = (perturbed[:, 3:] + 1) % 16
    assert not torch.equal(leaky(perturbed)[:, 3], leaky(symbols)[:, 3])


def test_group_one_planes_are_previous_channel_and_running_mean(ma):
    symbols = torch.full((1, 4, 1, 1), 8)
    symbols[0, 0] = 11                                   # |3| in channel 0 only
    planes = ma.context_planes(symbols, torch.full((4,), 8), 1)[0, :, :, 0, 0]

    assert planes[:, 0].tolist() == pytest.approx([0.0, 1.0, 0.0, 0.0])        # prev
    assert planes[:, 1].tolist() == pytest.approx([0.0, 1.0, 0.5, 1 / 3])      # activity
    assert planes[:, 2].tolist() == [0.0, 1.0, 1.0, 1.0]                       # available


def test_the_first_group_has_no_context(ma):
    symbols = torch.randint(0, 16, (1, C, H, W))
    for group in (1, 2, 4, 8):
        planes = ma.context_planes(symbols, ZERO, group)
        assert torch.count_nonzero(planes[:, :group]) == 0, group


def test_one_group_spanning_every_channel_is_the_no_context_model(ma):
    symbols = torch.randint(0, 16, (1, C, H, W))
    assert torch.count_nonzero(ma.context_planes(symbols, ZERO, C)) == 0


def test_group_size_must_divide_the_channel_count(ma):
    with pytest.raises(ValueError, match="must divide"):
        ma.group_starts(C, 3)
    with pytest.raises(ValueError, match="must divide"):
        ma.ChannelContextEntropyModel(C, 16, 8, group_size=5)


# --- probabilities and tables ------------------------------------------------------------


def test_probabilities_are_normalised_at_every_position(ma, mk):
    model = _model(ma, mk)
    reference, symbols = _inputs()
    with torch.no_grad():
        log_probabilities = model.log_probabilities(
            reference, model.planes(torch.from_numpy(symbols)[None], ZERO))
    assert torch.allclose(log_probabilities.exp().sum(dim=2), torch.ones(1, C, H, W), atol=1e-5)


def test_frequency_tables_sum_exactly_and_have_no_zero(ma, mk):
    model = _model(ma, mk)
    reference, symbols = _inputs()
    with torch.no_grad():
        rows = ma._rows(model.log_probabilities(
            reference, model.planes(torch.from_numpy(symbols)[None], ZERO)))
    cumulative, _ = ma._tables_for(rows, None, mk)
    frequencies = np.diff(cumulative, axis=1)

    assert (cumulative[:, -1] == TOTAL_FREQUENCY).all()
    assert int(frequencies.min()) >= MIN_FREQUENCY


def test_probability_generation_is_deterministic(ma, mk):
    model = _model(ma, mk)
    reference, symbols = _inputs()
    planes = model.planes(torch.from_numpy(symbols)[None], ZERO)
    with torch.no_grad():
        assert torch.equal(model(reference, planes), model(reference, planes))


def test_a_sample_is_independent_of_the_other_samples_in_the_batch(ma, mk):
    """The decoder runs the full batch with LATER channels still unknown; the
    encoder runs it with every channel known. That is only safe if a channel's
    output ignores every other channel's planes - checked, not assumed."""
    model = _model(ma, mk)
    reference, symbols = _inputs()
    planes = model.planes(torch.from_numpy(symbols)[None], ZERO)
    with torch.no_grad():
        base = model(reference, planes)
        scrambled = planes.clone()
        scrambled[:, 4:] = torch.rand_like(scrambled[:, 4:])
        assert torch.equal(model(reference, scrambled)[:, :4], base[:, :4])


@pytest.mark.parametrize("group", [1, 2, 4, 8])
def test_encoder_and_decoder_derive_identical_tables(ma, mk, group):
    """The decoder must build, group by group, exactly the tables the encoder
    built in one pass. Compared table by table, not just through a round trip."""
    model = _model(ma, mk, group=group)
    reference, symbols = _inputs(2)
    tensor = torch.from_numpy(symbols)[None]
    with torch.no_grad():
        encoder_rows = ma._rows(model.log_probabilities(reference, model.planes(tensor, ZERO)))
        decoded = ZERO.view(C, 1, 1).expand(C, H, W).clone()
        for start in range(0, C, group):
            rows = ma._rows(model.log_probabilities(reference, model.planes(decoded[None], ZERO))
                            [:, start:start + group])
            assert torch.equal(rows, encoder_rows[start * H * W:(start + group) * H * W])
            decoded[start:start + group] = tensor[0, start:start + group]


# --- coding ----------------------------------------------------------------------------------


@pytest.mark.parametrize("group", [1, 2, 4, 8])
@pytest.mark.parametrize("use_codebook", [False, True])
def test_round_trip_through_the_existing_coder(ma, mk, group, use_codebook):
    ml = _load_script("m10l_shared_codebook")
    model = _model(ma, mk, group=group)
    reference, symbols = _inputs(5)
    codebook = (ml.fit_codebook(np.random.default_rng(0).dirichlet(np.ones(16), 400), 16, bits=4)
                if use_codebook else None)
    with torch.no_grad():
        payload, ideal = ma.encode_frame(model, reference, symbols, ZERO, bits=4,
                                         codebook=codebook)
        decoded = ma.decode_frame(model, payload, reference, ZERO, bits=4, shape=(C, H, W),
                                  codebook=codebook)
    assert np.array_equal(decoded, symbols.reshape(-1))
    assert len(payload) * 8 >= ideal - 1, "the coder cannot beat the ideal code length"


@pytest.mark.parametrize("pattern", ["all_zero_symbol", "all_max", "all_min", "alternating"])
def test_adversarial_symbol_frames_round_trip(ma, mk, pattern):
    model = _model(ma, mk)
    reference, _ = _inputs()
    if pattern == "all_zero_symbol":
        symbols = np.full((C, H, W), 8)
    elif pattern == "all_max":
        symbols = np.full((C, H, W), 15)
    elif pattern == "all_min":
        symbols = np.zeros((C, H, W), dtype=np.int64)
    else:
        symbols = (np.indices((C, H, W)).sum(axis=0) % 2) * 15
    with torch.no_grad():
        payload, _ = ma.encode_frame(model, reference, symbols, ZERO, bits=4)
        decoded = ma.decode_frame(model, payload, reference, ZERO, bits=4, shape=(C, H, W))
    assert np.array_equal(decoded, symbols.reshape(-1))


@pytest.mark.parametrize("bits", [3, 4, 5])
def test_every_rate_point_round_trips(ma, mk, bits):
    alphabet = 2 ** bits
    model = _model(ma, mk, alphabet=alphabet)
    reference, symbols = _inputs(6, alphabet)
    zero = torch.full((C,), alphabet // 2)
    with torch.no_grad():
        payload, _ = ma.encode_frame(model, reference, symbols, zero, bits=bits)
        decoded = ma.decode_frame(model, payload, reference, zero, bits=bits, shape=(C, H, W))
    assert np.array_equal(decoded, symbols.reshape(-1))


def test_repeated_encodes_are_byte_identical(ma, mk):
    model = _model(ma, mk)
    reference, symbols = _inputs(7)
    with torch.no_grad():
        first, _ = ma.encode_frame(model, reference, symbols, ZERO, bits=4)
        second, _ = ma.encode_frame(model, reference, symbols, ZERO, bits=4)
    assert first == second


def test_a_decoder_with_the_wrong_model_does_not_recover_the_symbols(ma, mk):
    """What the identity check exists to prevent: the wrong model does not raise,
    it silently decodes garbage."""
    reference, symbols = _inputs(8)
    with torch.no_grad():
        payload, _ = ma.encode_frame(_model(ma, mk, seed=0), reference, symbols, ZERO, bits=4)
        wrong = ma.decode_frame(_model(ma, mk, seed=9), payload, reference, ZERO, bits=4,
                                shape=(C, H, W))
    assert not np.array_equal(wrong, symbols.reshape(-1))


def test_timings_are_reported_per_stage(ma, mk):
    model = _model(ma, mk, group=2)
    reference, symbols = _inputs()
    encode, decode = {}, {}
    with torch.no_grad():
        payload, _ = ma.encode_frame(model, reference, symbols, ZERO, bits=4, timings=encode)
        ma.decode_frame(model, payload, reference, ZERO, bits=4, shape=(C, H, W), timings=decode)
    assert set(encode) == set(decode) == {"network", "tables", "coder"}


# --- identity ------------------------------------------------------------------------------------


def test_identity_binds_weights_context_calibration_bits_and_codebook(ma, mk):
    ml = _load_script("m10l_shared_codebook")
    model = _model(ma, mk)
    kwargs = dict(m10k_identity=b"\x01" * 8, calibration_signature="cal", bits=4)
    base = ma.model_identity(model, **kwargs)

    assert len(base) == 8, "must fit the .nvct v2 8-byte field"
    assert ma.model_identity(model, **kwargs) == base
    assert ma.model_identity(_model(ma, mk, seed=1), **kwargs) != base              # weights
    assert ma.model_identity(_model(ma, mk, group=2), **kwargs) != base             # context
    assert ma.model_identity(model, **dict(kwargs, calibration_signature="x")) != base
    assert ma.model_identity(model, **dict(kwargs, bits=5)) != base
    assert ma.model_identity(model, **dict(kwargs, m10k_identity=b"\x02" * 8)) != base
    codebook = ml.fit_codebook(np.random.default_rng(0).dirichlet(np.ones(16), 200), 8, bits=4)
    assert ma.model_identity(model, **kwargs, codebook=codebook) != base


def test_context_definition_identity_depends_on_group_size(ma):
    assert ma.context_definition_id(1) != ma.context_definition_id(8)
    assert ma.context_definition_id(4) == ma.context_definition_id(4)


def test_a_checkpoint_round_trips_through_disk(ma, mk, tmp_path):
    model = _model(ma, mk, group=4)
    path = tmp_path / "m11.pt"
    torch.save({"model_state_dict": model.state_dict(), "model_config": model.config_dict()},
               path)
    restored, _ = ma.load_model(path)
    reference, symbols = _inputs()
    planes = model.planes(torch.from_numpy(symbols)[None], ZERO)
    with torch.no_grad():
        assert torch.equal(restored(reference, planes), model(reference, planes))
    assert restored.group_size == 4


# --- training ---------------------------------------------------------------------------------


def test_training_learns_channel_context_that_exists(ma, mk):
    """Data where a channel's residual is busy exactly where earlier channels
    were busy. The context model must beat the ablation on held-out frames."""
    rng = np.random.default_rng(0)

    def frames(count):
        out = []
        for _ in range(count):
            mask = rng.random((H, W)) < 0.3
            frame = np.where(mask[None], rng.integers(0, 16, size=(C, H, W)), 8)
            out.append(frame)
        return torch.from_numpy(np.stack(out)).long()

    train_set = (frames(64), torch.zeros(64, C, H, W))
    held_set = (frames(24), torch.zeros(24, C, H, W))
    results = {}
    for use_context in (True, False):
        model = ma.from_m10k(_m10k(mk), use_context=use_context)
        ma.train(model, train_set, held_set, ZERO, epochs=8, batch_size=8,
                 learning_rate=1e-2, device=torch.device("cpu"), log=lambda m: None)
        with torch.no_grad():
            results[use_context] = ma.nll_bits(model, *held_set, ZERO,
                                               device=torch.device("cpu"))
    assert results[True] < 0.9 * results[False]


def test_training_keeps_the_warm_start_when_nothing_improves_it(ma, mk):
    """Epoch 0 is a legitimate winner: if every update hurts the selection
    split, the returned weights are the untouched warm start."""
    model = ma.from_m10k(_m10k(mk))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    symbols = torch.randint(0, 16, (8, C, H, W))
    references = torch.randn(8, C, H, W)
    history = ma.train(model, (symbols, references), (symbols[:2], references[:2]), ZERO,
                       epochs=2, batch_size=4, learning_rate=10.0,
                       device=torch.device("cpu"), log=lambda m: None)
    best = min(history, key=lambda h: (h["val_loss"], h["epoch"]))
    if best["epoch"] == 0:
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
