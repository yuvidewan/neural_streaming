"""M24 Phase A - tests for `scripts/m24_quantizer.py` and its gate.

The load-bearing tests:

  * `test_the_arms_differ_in_exactly_one_thing_each` - the whole value of the
    gate is that it attributes the gap to the density OR the grid rule. If two
    things move between neighbouring arms, the decomposition means nothing;
  * `test_the_laplace_cdf_does_not_overflow_on_a_far_tail` - `np.where` evaluates
    both branches, so an unclipped exponential returns inf from the arm that was
    never going to be selected;
  * `test_bits_are_a_probability_mass_not_a_density` - a density value is not a
    probability, and using one would understate every rate in the report;
  * `test_every_density_is_fitted_on_train_only`.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m24():
    return _load_script("m24_quantizer")


@pytest.fixture(scope="module")
def gate():
    return _load_script("m24_offline_gate")


@pytest.fixture(scope="module")
def latents():
    """Channels with very different energies - the situation the latent-shape
    diagnostic found, and the one the grid rule handles badly."""
    torch.manual_seed(0)
    energies = torch.tensor([4.0, 2.0, 1.0, 0.2])
    return torch.randn(40, 4, 8, 8) * energies[None, :, None, None]


# ----------------------------------------------------------------------------------------
# The decomposition only means something if each step moves one thing
# ----------------------------------------------------------------------------------------


def test_the_arms_differ_in_exactly_one_thing_each(m24):
    assert m24.ARMS == ("grid_histogram", "grid_laplace", "unit_step_laplace")
    source = inspect.getsource(m24.measure_point)

    # the two grid arms share the grid and differ only in the density
    assert 'if arm == "grid_histogram":' in source
    assert "fit_histogram(" in source and "fit_laplace(" in source
    # the two Laplace arms share the density and differ only in the rule
    grid_block = source[source.index('if arm.startswith("grid")'):source.index("else:\n        step")]
    assert "deployed_grid(" in grid_block
    assert "unit_step_symbols(" not in grid_block


def test_the_grid_arm_is_the_deployed_quantizer_not_a_lookalike(m24):
    """If the baseline is not literally what ships, the gate measures a strawman."""
    source = inspect.getsource(m24.deployed_grid)

    assert "calibrate_quantization_params(" in source
    assert 'mode="per_channel"' in source
    assert "lower_percentile=LOWER_PERCENTILE" in source
    assert (m24.LOWER_PERCENTILE, m24.UPPER_PERCENTILE) == (0.1, 99.9)
    symbols = inspect.getsource(m24.grid_symbols)
    assert "latent_to_symbols(" in symbols


def test_the_unit_step_arm_clips_nothing(m24, latents):
    """The grid's percentile bounds are a distortion source bmshj2018 does not
    have; the comparison is only honest if the unit-step arm really is unbounded."""
    # the CODE must not clip; the docstring is allowed to say "nothing clipped",
    # which an earlier version of this assertion matched on and failed
    code = "".join(line for line in inspect.getsource(m24.unit_step_symbols).splitlines()
                   if not line.strip().startswith(('"""', '#', '*')))
    assert "np.clip" not in code and ".clamp" not in code

    symbols = m24.unit_step_symbols(latents, step=0.1)
    # the extreme values survive as extreme symbols rather than saturating
    assert abs(symbols).max() > 100
    assert np.unique(symbols).size > 200


# ----------------------------------------------------------------------------------------
# Rate accounting
# ----------------------------------------------------------------------------------------


def test_the_laplace_cdf_does_not_overflow_on_a_far_tail(m24):
    """`np.where` evaluates BOTH branches. Clipping only the upper one still
    overflows `exp` on a far-left value and hands back inf or nan from the arm
    that was never selected."""
    location = np.array([0.0])
    scale = np.array([0.5])
    far = np.array([-10_000.0, 10_000.0, 0.0])

    with np.errstate(over="raise", invalid="raise"):
        out = m24._laplace_cdf(far, location, scale)

    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[1] == pytest.approx(1.0, abs=1e-12)
    assert out[2] == pytest.approx(0.5)


def test_the_laplace_cdf_is_monotone_and_bounded(m24):
    location, scale = np.array([0.3]), np.array([1.5])
    x = np.linspace(-20.0, 20.0, 401)

    out = m24._laplace_cdf(x, location, scale)

    assert np.all(np.diff(out) >= -1e-12)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_bits_are_a_probability_mass_not_a_density(m24):
    """A Laplace DENSITY can exceed 1 for a small scale, which would make
    -log2 negative and understate the rate. The code must integrate the unit bin."""
    source = inspect.getsource(m24.discretised_laplace_bits)
    assert "probability MASS" in source
    assert "_laplace_cdf(flat[channel] + 0.5" in source
    assert "_laplace_cdf(flat[channel] - 0.5" in source

    # a very peaked channel: density at the mode is >> 1, mass is <= 1, so bits >= 0
    symbols = np.zeros((4, 1, 9), dtype=np.int64)
    bits = m24.discretised_laplace_bits(symbols, np.array([0.0]), np.array([0.01]),
                                        channels=1)
    assert bits >= 0.0


def test_a_coarser_step_costs_fewer_bits(m24, latents):
    previous = None
    for step in (0.25, 0.5, 1.0, 2.0, 4.0):
        symbols = m24.unit_step_symbols(latents, step=step)
        location, scale = m24.fit_laplace(symbols, channels=4)
        bits = m24.discretised_laplace_bits(symbols, location, scale, channels=4) \
            / symbols.size
        if previous is not None:
            assert bits < previous, f"step {step} did not cost less than the finer one"
        previous = bits


def test_the_histogram_is_smoothed_so_an_unseen_symbol_is_not_infinite(m24):
    train = np.zeros((4, 2, 9), dtype=np.int64)          # only symbol 0 ever seen
    probabilities = m24.fit_histogram(train, alphabet=8, channels=2)

    assert (probabilities > 0).all()
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(2))
    unseen = np.full((1, 2, 9), 7, dtype=np.int64)
    assert math.isfinite(m24.histogram_bits(unseen, probabilities, channels=2))


def test_laplace_is_fitted_by_median_and_mean_absolute_deviation(m24):
    """The ML estimates for a Laplace, not the mean and standard deviation - the
    wrong pair would misfit the long tails this project's latents have."""
    source = inspect.getsource(m24.fit_laplace)
    assert "np.median(" in source
    assert "maximum likelihood" in source

    values = np.array([[[-2.0, -1.0, 0.0, 1.0, 50.0]]])    # one heavy outlier
    location, scale = m24.fit_laplace(values, channels=1)

    assert location[0] == pytest.approx(0.0)               # median, not the mean (9.6)
    assert scale[0] == pytest.approx(np.mean([2.0, 1.0, 0.0, 1.0, 50.0]))


def test_a_degenerate_channel_does_not_produce_infinite_bits(m24):
    """A channel whose values are all identical fits scale 0, and an unfloored
    scale would divide by zero."""
    values = np.zeros((4, 1, 9))

    _, scale = m24.fit_laplace(values, channels=1)

    assert scale[0] > 0.0
    bits = m24.discretised_laplace_bits(np.zeros((4, 1, 9), dtype=np.int64),
                                        np.array([0.0]), scale, channels=1)
    assert math.isfinite(bits)


# ----------------------------------------------------------------------------------------
# The grid rule's defect, measured directly
# ----------------------------------------------------------------------------------------


def test_the_grid_gives_a_low_energy_channel_the_same_depth_as_a_loud_one(m24, latents):
    """The defect the whole milestone is about: the step is tied to the channel's
    RANGE at a fixed level count, so a channel carrying little information still
    spends the full depth on its own narrow spread."""
    params = m24.deployed_grid(latents, bits=4)
    steps = params.scale.flatten()

    # 20x energy difference between the loudest and quietest channel
    assert float(steps[0]) / float(steps[3]) > 10
    # yet both are coded on exactly 16 levels
    symbols = m24.grid_symbols(latents, params)
    assert symbols.max() <= 15 and symbols.min() >= 0


def test_the_reference_step_brackets_the_sweep(m24, latents):
    """The step multipliers are relative to the deployed 4-bit step so the two
    curves overlap and a BD-rate between them interpolates rather than
    extrapolates."""
    reference = m24.reference_step(latents, bits=4)
    params = m24.deployed_grid(latents, bits=4)

    assert reference == pytest.approx(float(params.scale.flatten().median()))
    assert min(m24.DEFAULT_STEP_MULTIPLIERS) < 1.0 < max(m24.DEFAULT_STEP_MULTIPLIERS)


def test_reconstruction_round_trips_through_each_rule(m24, latents):
    shape = tuple(latents.shape[1:])
    params = m24.deployed_grid(latents, bits=5)

    grid = m24.grid_reconstruct(m24.grid_symbols(latents, params), params, shape)
    unit = m24.unit_step_reconstruct(m24.unit_step_symbols(latents, step=0.5),
                                     step=0.5, shape=shape)

    assert grid.shape == latents.shape
    assert unit.shape == latents.shape
    # both are close to the original, and finer settings are closer
    coarse = m24.unit_step_reconstruct(m24.unit_step_symbols(latents, step=2.0),
                                       step=2.0, shape=shape)
    assert float((unit - latents).abs().mean()) < float((coarse - latents).abs().mean())


# ----------------------------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------------------------


def test_every_density_is_fitted_on_train_only(gate):
    source = inspect.getsource(gate.main)

    assert "train_latents=train_latents" in source
    assert "val_latents=val_latents" in source
    measure = inspect.getsource(_load_script("m24_quantizer").measure_point)
    # the fits take TRAIN symbols; the bits are scored on VAL symbols
    assert "fit_histogram(train_symbols" in measure
    assert "fit_laplace(train_symbols" in measure
    assert "histogram_bits(val_symbols" in measure


def test_test_split_is_never_opened(gate):
    source = inspect.getsource(gate)

    assert 'split="test"' not in source
    assert "TEST is never opened" in source
    assert 'split="train"' in source
    assert "val_b_sequences(" in source


def test_the_gate_states_what_it_cannot_answer(gate):
    """A frozen-weight measurement cannot price rate-aware training, and a reader
    who misses that would read this as the whole CompressAI gap."""
    source = inspect.getsource(gate)

    assert "cannot_answer" in source
    assert "Phase B" in source
    assert "needs Vimeo and a GPU" in source


def test_the_decomposition_is_reported_as_two_separate_causes(gate):
    source = inspect.getsource(gate.main)

    assert "density model (histogram -> Laplace)" in source
    assert "grid rule (percentile depth -> unit step)" in source
    assert "grid_laplace_vs_grid_histogram" in source
    assert "unit_step_laplace_vs_grid_laplace" in source


def test_rate_is_ideal_bits_and_the_report_says_so(m24):
    """Not coded bytes: holding the entropy model comparable is the point, and
    the coder has been within 0.011% of ideal every time it was measured."""
    source = inspect.getsource(m24)

    assert "ideal bits" in source
    assert "0.011%" in source


# ----------------------------------------------------------------------------------------
# The ceiling: the finding this phase actually produced
# ----------------------------------------------------------------------------------------


def test_the_percentile_grid_saturates_with_bit_depth_and_unit_step_does_not(m24, latents):
    """The headline. The grid clips everything past its percentile bounds, and no
    number of levels INSIDE the range repairs a clipped value - so its distortion
    approaches a floor set by the clipped tails. Unit-step rounding has no bounds
    and keeps improving.

    This is why the quantizer has to be replaced rather than widened, and it is
    measured here on the error in the latent itself so the test does not need a
    decoder.
    """
    def grid_error(bits: int) -> float:
        params = m24.deployed_grid(latents, bits=bits)
        shape = tuple(latents.shape[1:])
        rebuilt = m24.grid_reconstruct(m24.grid_symbols(latents, params), params, shape)
        return float(((rebuilt - latents) ** 2).mean())

    def unit_error(step: float) -> float:
        shape = tuple(latents.shape[1:])
        rebuilt = m24.unit_step_reconstruct(
            m24.unit_step_symbols(latents, step=step), step=step, shape=shape)
        return float(((rebuilt - latents) ** 2).mean())

    coarse, fine, finest = grid_error(5), grid_error(8), grid_error(11)
    # doubling the depth three more times buys almost nothing: the remaining error
    # is clipping, which more levels cannot touch
    assert finest > 0.0
    assert finest / fine > 0.5, "the grid should be near its clipping floor by 8 bits"
    assert fine < coarse, "more depth must still help before the floor is reached"

    # unit step keeps paying off across the same span, because nothing is clipped
    ratios = [unit_error(step) for step in (1.0, 0.25, 0.0625)]
    assert ratios[1] < ratios[0] / 3
    assert ratios[2] < ratios[1] / 3


def test_the_clipped_fraction_is_reported_so_the_ceiling_is_visible(m24):
    """`at_alphabet_edge_fraction` is how a reader sees the clipping without
    having to infer it from a saturating curve."""
    source = inspect.getsource(m24.measure_point)

    assert "at_alphabet_edge_fraction" in source
    assert "symbol_span" in source, "the unit-step arm reports its range instead"
