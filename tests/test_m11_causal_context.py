"""Tests for M11's causal residual contexts and the estimators that score them.

An autoregressive entropy model is only decodable if the probability for symbol
i uses nothing the decoder has not already decoded. Getting that wrong does not
crash anything - it produces a model that looks better offline and a stream no
decoder can read - so causality is tested directly, including against contexts
that are deliberately leaky, which the checker must catch.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cx():
    return _load_script("m11_causal_context")


def _frame(seed=0, channels=6, size=8, alphabet=16):
    return np.random.default_rng(seed).integers(0, alphabet, size=(channels, size, size))


ZERO = np.full(6, 8)


# --- 1. scan order -----------------------------------------------------------------


def test_coding_order_is_c_major_raster_as_the_coder_consumes_it(cx):
    """The whole causal argument rests on this, so it is pinned against the real
    codec's flattening rather than restated."""
    import torch
    from nvc.compression.codec import latent_to_symbols
    from nvc.compression.quantization import QuantizationParams

    channels, height, width = 3, 4, 5
    latent = torch.arange(channels * height * width, dtype=torch.float32).reshape(
        1, channels, height, width)
    params = QuantizationParams(scale=torch.ones(1, 1, 1, 1), zero_point=torch.zeros(1, 1, 1, 1),
                                bits=8, mode="global")
    flat = latent_to_symbols(latent, params)

    for c, y, x in ((0, 0, 0), (0, 1, 0), (1, 0, 0), (2, 3, 4), (1, 2, 3)):
        assert flat[cx.coding_index(c, y, x, height, width)] == latent[0, c, y, x]


# --- 2. no future-symbol access -----------------------------------------------------


@pytest.mark.parametrize("name", [
    "left", "up", "left_up", "left_up_upleft", "neighbourhood", "left_raw",
    "prev_channel", "channel_activity", "neighbourhood_x_channel", "left_up_x_channel"])
def test_every_candidate_context_is_causal(cx, name):
    function = cx.CONTEXTS[name][0]
    for seed in range(3):
        result = cx.check_causality(function, _frame(seed), ZERO, 16, probes=64, seed=seed)
        assert result["causal"], f"{name} leaks at {result['leaking_positions'][:5]}"


@pytest.mark.parametrize("leak", ["right", "down", "self", "next_channel", "down_right"])
def test_the_causality_checker_catches_planted_leaks(cx, leak):
    """A checker that never fails proves nothing. Each of these reads one symbol
    the decoder cannot have yet, and each must be caught."""
    offsets = {"right": dict(dx=-1), "down": dict(dy=-1), "next_channel": dict(dc=-1),
               "down_right": dict(dy=-1, dx=-1)}
    if leak == "self":
        function = lambda s, z, a: cx.magnitudes(s, z)
    else:
        function = lambda s, z, a: cx._shift(cx.magnitudes(s, z), **offsets[leak])

    assert not cx.check_causality(function, _frame(), ZERO, 16, probes=64)["causal"]


def test_earlier_channels_are_available_even_spatially_ahead(cx):
    """C-major order decodes a whole channel before the next begins, so a
    position in channel c-1 that is right of / below the current one is past,
    not future. A context reading it must pass the checker."""
    ahead_in_previous_channel = lambda s, z, a: cx._shift(cx.magnitudes(s, z), dc=1, dy=-1,
                                                          dx=-1)
    assert cx.check_causality(ahead_in_previous_channel, _frame(), ZERO, 16,
                              probes=64)["causal"]


# --- 3-5. neighbour correctness -------------------------------------------------------


def test_left_up_and_upleft_read_the_right_neighbours(cx):
    frame = _frame(4)
    m = cx.magnitudes(frame, ZERO)
    left, up = cx.ctx_left(frame, ZERO, 16), cx.ctx_up(frame, ZERO, 16)
    triple = cx.ctx_left_up_upleft(frame, ZERO, 16)
    levels = cx.NEIGHBOUR_LEVELS

    for c, y, x in ((2, 3, 5), (0, 1, 1), (5, 7, 7)):
        assert left[c, y, x] == m[c, y, x - 1]
        assert up[c, y, x] == m[c, y - 1, x]
        assert triple[c, y, x] == (m[c, y, x - 1] * levels + m[c, y - 1, x]) * levels \
            + m[c, y - 1, x - 1]


def test_neighbourhood_includes_the_up_right_neighbour(cx):
    """The row above is fully decoded, so up-right is available and used."""
    frame = np.full((1, 4, 4), 8)
    frame[0, 0, 2] = 15                            # only the up-right of (1, 1) is busy
    context = cx.ctx_neighbourhood(frame, np.array([8]), 16)

    assert context[0, 1, 1] > 0
    assert context[0, 1, 0] == 0, "up-right of (1, 0) is (0, 1), which is quiet"


def test_magnitude_is_measured_from_each_channels_own_zero_symbol(cx):
    frame = np.array([[[5]], [[9]]])
    assert cx.magnitudes(frame, np.array([5, 7])).reshape(-1).tolist() == [0, 2]
    assert cx.magnitudes(np.array([[[0]]]), np.array([12]))[0, 0, 0] == cx.MAGNITUDE_CLIP


# --- 22 / 24. boundaries ----------------------------------------------------------------


def test_the_first_symbol_has_no_context_at_all(cx):
    frame = _frame(5)
    for name in cx.CONTEXTS:
        function = cx.CONTEXTS[name][0]
        on_first = function(frame, ZERO, 16)[0, 0, 0]
        altered = frame.copy()
        altered.reshape(-1)[1:] = (altered.reshape(-1)[1:] + 3) % 16
        assert function(altered, ZERO, 16)[0, 0, 0] == on_first, name


def test_frame_edges_use_an_explicit_not_available_level(cx):
    """A missing neighbour is not a zero residual; conflating them would teach the
    model that edges are quiet."""
    frame = np.full((2, 3, 3), 8)
    left, up = cx.ctx_left(frame, np.array([8, 8]), 16), cx.ctx_up(frame, np.array([8, 8]), 16)

    assert (left[:, :, 0] == cx.NOT_AVAILABLE).all()
    assert (up[:, 0, :] == cx.NOT_AVAILABLE).all()
    assert (left[:, :, 1:] == 0).all(), "a genuinely zero residual is level 0, not NA"


def test_channel_boundary_first_channel_has_no_channel_context(cx):
    frame = _frame(6)
    assert (cx.ctx_prev_channel(frame, ZERO, 16)[0] == cx.NOT_AVAILABLE).all()
    assert (cx.ctx_channel_activity(frame, ZERO, 16)[0] == cx.CHANNEL_LEVELS - 1).all()


def test_spatial_context_does_not_wrap_across_channels(cx):
    """(c, 0, 0) must not see (c-1, H-1, W-1) as its 'left' neighbour, even though
    it is the previous symbol in flat order."""
    frame = np.full((2, 2, 2), 8)
    frame[0, 1, 1] = 15
    assert cx.ctx_left(frame, np.array([8, 8]), 16)[1, 0, 0] == cx.NOT_AVAILABLE


def test_channel_activity_is_the_running_mean_over_earlier_channels(cx):
    frame = np.full((4, 1, 1), 8)
    frame[0, 0, 0] = 11                             # magnitude 3 in channel 0 only
    activity = cx._prev_channel_activity(cx.magnitudes(frame, np.full(4, 8)))

    assert np.isnan(activity[0, 0, 0])
    assert activity[1:, 0, 0].tolist() == pytest.approx([3.0, 1.5, 1.0])


def test_contexts_stay_inside_their_declared_cardinality(cx):
    for alphabet in (8, 16, 32):
        frame = np.random.default_rng(alphabet).integers(0, alphabet, size=(6, 8, 8))
        zero = np.full(6, alphabet // 2)
        for name in cx.CONTEXTS:
            values, card = cx.compute_context(name, frame, zero, alphabet)
            assert values.min() >= 0 and values.max() < card, name


# --- 6. random control -----------------------------------------------------------------


def test_the_permutation_control_keeps_the_histogram_and_breaks_alignment(cx):
    context = cx.ctx_left_up(_frame(7), ZERO, 16)
    permuted = cx.permuted_context(context, np.random.default_rng(0))

    assert np.array_equal(np.bincount(context.reshape(-1)),
                          np.bincount(permuted.reshape(-1))), "same cardinality and sparsity"
    assert not np.array_equal(context, permuted)


def test_the_control_is_deterministic_for_a_seed(cx):
    context = cx.ctx_left(_frame(8), ZERO, 16)
    first = cx.permuted_context(context, np.random.default_rng(3))
    second = cx.permuted_context(context, np.random.default_rng(3))
    assert np.array_equal(first, second)


# --- estimators ---------------------------------------------------------------------------


def test_an_informative_context_beats_its_control_on_held_out_data(cx):
    """End-to-end estimator check on data where the answer is known: the symbol
    equals its left neighbour most of the time. The real context must win on
    held-out rows and the permuted control must not."""
    rng = np.random.default_rng(1)

    def sample(frames):
        out = []
        for _ in range(frames):
            frame = np.full((2, 8, 8), 8)
            for c in range(2):
                for y in range(8):
                    for x in range(8):
                        frame[c, y, x] = (frame[c, y, x - 1] if x and rng.random() < 0.8
                                          else rng.integers(0, 16))
            out.append(frame)
        return np.stack(out)

    train, held = sample(40), sample(20)
    zero = np.full(2, 8)
    groups = np.zeros(train.size, dtype=np.int64)
    held_groups = np.zeros(held.size, dtype=np.int64)
    parent = cx.smoothed_parent(cx.fit_parent(train.reshape(-1), groups, 1, 16),
                                np.full((1, 16), 1 / 16), 1.0)
    baseline = cx.parent_bits(parent, held.reshape(-1), held_groups)

    def score(train_ctx, held_ctx, card):
        child = cx.fit_child(train.reshape(-1), groups, train_ctx.reshape(-1), card, 1, 16)
        return cx.child_bits(child, parent, held.reshape(-1), held_groups,
                             held_ctx.reshape(-1), card, 16.0)

    real = score(np.stack([cx.ctx_left_raw(f, zero, 16) for f in train]),
                 np.stack([cx.ctx_left_raw(f, zero, 16) for f in held]), 17)
    control_rng = np.random.default_rng(2)
    control = score(
        np.stack([cx.permuted_context(cx.ctx_left_raw(f, zero, 16), control_rng) for f in train]),
        np.stack([cx.permuted_context(cx.ctx_left_raw(f, zero, 16), control_rng) for f in held]),
        17)

    assert real < 0.8 * baseline
    assert control > baseline * 0.995, "a permuted context must not look informative"


def test_hierarchical_smoothing_falls_back_to_the_parent_for_unseen_contexts(cx):
    parent = np.array([[0.7, 0.1, 0.1, 0.1]])
    child = np.zeros((3, 4))                         # context 2 never seen in training
    bits = cx.child_bits(child, parent, np.array([0]), np.array([0]), np.array([2]), 3, 50.0)
    assert bits == pytest.approx(-np.log2(0.7))


def test_smoothing_strength_is_selected_by_the_supplied_score(cx):
    strength, bits = cx.select_strength(lambda s: abs(np.log2(s) - 4.0), grid=(1, 4, 16, 64))
    assert strength == 16 and bits == 0.0


def test_context_definition_identity_distinguishes_every_context(cx):
    identities = {cx.context_definition_id(name) for name in cx.CONTEXTS}
    assert len(identities) == len(cx.CONTEXTS)
    assert cx.context_definition_id("left") == cx.context_definition_id("left")


def test_zero_symbol_matches_what_the_quantizer_emits_for_a_zero_residual(cx):
    import torch
    from nvc.compression.calibration import calibrate_quantization_params
    from nvc.compression.codec import latent_to_symbols

    residuals = torch.randn(20, 3, 4, 4) * torch.tensor([0.5, 1.0, 2.0]).view(1, 3, 1, 1) + 0.1
    params = calibrate_quantization_params(residuals, bits=4, mode="per_channel")
    zero = cx.zero_symbols(params, 3)

    expected = latent_to_symbols(torch.zeros(1, 3, 1, 1), params)
    assert zero.tolist() == expected.tolist()


def test_a_per_frame_shuffle_leaks_frame_activity_and_a_global_one_does_not(cx):
    """Found by running the gate: shuffling a context WITHIN each frame keeps that
    frame's context histogram, i.e. how busy the frame is overall - information
    gathered partly from future positions. On data where frames differ in
    activity, that per-frame 'control' looks informative. A shuffle across the
    whole split does not, which is why the gate uses it."""
    rng = np.random.default_rng(11)
    zero = np.full(2, 8)

    def frames(count):
        out = []
        for index in range(count):
            spread = 0 if index % 2 == 0 else 7          # quiet frames and busy frames
            out.append(8 + rng.integers(-spread, spread + 1, size=(2, 8, 8)))
        return np.stack(out)

    train, held = frames(60), frames(30)
    groups = np.zeros(train.size, dtype=np.int64)
    held_groups = np.zeros(held.size, dtype=np.int64)
    parent = cx.smoothed_parent(cx.fit_parent(train.reshape(-1), groups, 1, 16),
                                np.full((1, 16), 1 / 16), 1.0)
    baseline = cx.parent_bits(parent, held.reshape(-1), held_groups)

    def control_bits(train_ctx, held_ctx):
        child = cx.fit_child(train.reshape(-1), groups, train_ctx.reshape(-1), 5, 1, 16)
        return cx.child_bits(child, parent, held.reshape(-1), held_groups,
                             held_ctx.reshape(-1), 5, 16.0)

    t_ctx = np.stack([cx.ctx_left(f, zero, 16) for f in train])
    h_ctx = np.stack([cx.ctx_left(f, zero, 16) for f in held])
    per_frame = control_bits(np.stack([cx.permuted_context(c, rng) for c in t_ctx]),
                             np.stack([cx.permuted_context(c, rng) for c in h_ctx]))
    whole_split = control_bits(cx.permuted_context(t_ctx, rng), cx.permuted_context(h_ctx, rng))

    assert per_frame < 0.97 * baseline, "the per-frame shuffle carries frame activity"
    assert whole_split > 0.995 * baseline, "a whole-split shuffle carries nothing"
