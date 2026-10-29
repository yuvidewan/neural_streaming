"""M23 - tests for `scripts/m23_hyperprior.py` and `scripts/m23_offline_gate.py`.

The load-bearing tests here are the ones that stop the gate from measuring
something other than what it claims:

  * `test_a_hyper_arm_cannot_be_evaluated_without_being_charged` - an unpriced
    side channel would make every hyper arm look better for free, which is the
    single way this gate could produce a wrong answer;
  * `test_the_side_cost_is_in_the_training_loss` - without it h_a is free to emit
    an arbitrarily expensive hyper-latent;
  * `test_each_arm_sees_only_what_it_claims` - the arms differ by conditioning
    and nothing else, or the comparison is not apples-to-apples;
  * `test_the_control_is_a_whole_split_derangement` - a per-frame shuffle leaks
    frame-level activity and manufactured a spurious ~0.1% gain in M12.
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
def m23():
    return _load_script("m23_hyperprior")


@pytest.fixture(scope="module")
def gate():
    return _load_script("m23_offline_gate")


SMALL = {"latent_channels": 4, "alphabet": 8, "hidden": 6,
         "hyper_channels": 3, "hyper_hidden": 6}


def _model(m23, arm):
    return m23.build_model(arm, dict(SMALL))


def _inputs(arm, *, frames=2, channels=4, side=8):
    reference = torch.randn(frames, channels, side, side)
    signal = torch.randn(frames, channels, side, side)
    return reference, signal, (frames, channels, side, side)


# ----------------------------------------------------------------------------------------
# The arms differ by conditioning and nothing else
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("arm,reference,hyper", [
    ("zref_only", True, False),
    ("hyper_only", False, True),
    ("hyper_plus_zref", True, True),
])
def test_each_arm_sees_only_what_it_claims(m23, arm, reference, hyper):
    model = _model(m23, arm)

    assert model.predictor.uses_reference is reference
    assert model.uses_hyper is hyper
    # the modules themselves, not just the flags - a present-but-unused reference
    # stack would put parameters in one arm that the others do not have
    assert hasattr(model.predictor, "features") is reference
    assert hasattr(model.predictor, "hyper_synthesis") is hyper
    assert hasattr(model, "analysis") is hyper
    assert hasattr(model, "rate_prior") is hyper


def test_the_zref_path_is_m10ks_so_the_baseline_is_m10ks_family(m23):
    """`zref_only` is the number the hyperprior has to beat. If its architecture
    drifts from M10K's, the gate is comparing against something the project has
    never deployed."""
    m10k = _load_script("m10k_learned_entropy")
    theirs = m10k.build_model({"latent_channels": 4, "alphabet": 8, "hidden": 6})
    ours = _model(m23, "zref_only")

    assert ours.parameter_count() == sum(p.numel() for p in theirs.parameters())
    reference, _, shape = _inputs("zref_only")
    assert (ours.predictor.log_probabilities(reference=reference, shape=shape).shape
            == theirs.log_probabilities(reference).shape)


@pytest.mark.parametrize("arm", ["zref_only", "hyper_only", "hyper_plus_zref"])
def test_log_probabilities_are_a_distribution_over_the_alphabet(m23, arm):
    model = _model(m23, arm)
    reference, signal, shape = _inputs(arm)
    log_probabilities, _, _ = m23._forward_arm(
        model, signal=signal if model.uses_hyper else None,
        reference=reference if model.predictor.uses_reference else None,
        shape=shape, hyper_bits=8, noisy=False)

    assert log_probabilities.shape == (shape[0], shape[1], SMALL["alphabet"], shape[2], shape[3])
    totals = log_probabilities.exp().sum(dim=2)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)


def test_an_arm_refuses_to_run_without_its_own_conditioning(m23):
    reference, signal, shape = _inputs("x")

    with pytest.raises(ValueError, match="needs a reference"):
        _model(m23, "zref_only").predictor.forward(reference=None, shape=shape)
    with pytest.raises(ValueError, match="needs a hyper-latent"):
        _model(m23, "hyper_only").predictor.forward(hyper=None, shape=shape)
    with pytest.raises(ValueError, match="needs an explicit output shape"):
        _model(m23, "hyper_only").predictor.forward(hyper=torch.randn(2, 3, 2, 2))
    with pytest.raises(ValueError, match="unknown arm"):
        m23.build_model("telepathy", dict(SMALL))


# ----------------------------------------------------------------------------------------
# The side channel is charged - the one way this gate could be wrong
# ----------------------------------------------------------------------------------------


def test_a_hyper_arm_cannot_be_evaluated_without_being_charged(m23):
    """An unpriced hyper-latent makes every hyper arm look better for free. The
    evaluator must refuse rather than silently charge zero."""
    model = _model(m23, "hyper_only")
    symbols = torch.zeros(2, 4, 64, dtype=torch.long)
    signals = torch.randn(2, 4, 8, 8)

    with pytest.raises(ValueError, match="side channel can be charged"):
        m23.evaluate_arm(model, symbols, None, signals, device=torch.device("cpu"),
                         hyper_bits=8, hyper_entropy_model=None)


def test_a_non_transmitting_arm_is_charged_nothing(m23):
    model = _model(m23, "zref_only")
    symbols = torch.zeros(2, 4, 64, dtype=torch.long)
    references = torch.randn(2, 4, 8, 8)

    result = m23.evaluate_arm(model, symbols, references, None, device=torch.device("cpu"),
                              hyper_bits=8, hyper_entropy_model=None)

    assert result["side_bits_per_symbol"] == 0.0
    assert result["hyper_values_transmitted"] == 0
    assert result["net_bits_per_symbol"] == pytest.approx(result["y_bits_per_symbol"])


def test_the_net_cost_is_y_plus_side_and_shares_y_s_denominator(m23):
    """Arms are only comparable if the side cost is expressed per y-symbol: a
    hyper arm has fewer hyper values than y symbols, and dividing by its own
    count would understate the charge."""
    model = _model(m23, "hyper_only")
    symbols = torch.randint(0, SMALL["alphabet"], (3, 4, 64))
    signals = torch.randn(3, 4, 8, 8)
    train, _ = m23.collect_hyper_symbols(model, signals, device=torch.device("cpu"),
                                         hyper_bits=8)
    table = m23.fit_hyper_entropy_model(train, bits=8,
                                        hyper_channels=SMALL["hyper_channels"])

    result = m23.evaluate_arm(model, symbols, None, signals, device=torch.device("cpu"),
                              hyper_bits=8, hyper_entropy_model=table)

    assert result["symbols_scored"] == 3 * 4 * 64
    assert result["hyper_values_transmitted"] < result["symbols_scored"]
    assert result["side_bits_per_symbol"] == pytest.approx(
        result["side_bits_total"] / result["symbols_scored"])
    assert result["net_bits_per_symbol"] == pytest.approx(
        result["y_bits_per_symbol"] + result["side_bits_per_symbol"])


def test_the_side_cost_is_in_the_training_loss(m23):
    """Without the side term in the loss, h_a is free to emit an arbitrarily
    expensive hyper-latent and the measured 'gain' is an artefact."""
    source = inspect.getsource(m23.train_arm)

    assert "rate_prior.bits" in source
    assert "y_bits + side_bits" in source

    # and behaviourally: the prior's parameters must actually receive gradients
    model = _model(m23, "hyper_only")
    symbols = torch.randint(0, SMALL["alphabet"], (2, 4, 64))
    signals = torch.randn(2, 4, 8, 8)
    batch = (symbols, None, signals)
    m23.train_arm(model, batch, batch, epochs=1, batch_size=2, learning_rate=1e-3,
                  device=torch.device("cpu"), hyper_bits=8, log=lambda m: None)
    assert model.rate_prior.log_scale.grad is not None
    assert torch.any(model.rate_prior.log_scale.grad != 0)


def test_the_reported_side_cost_comes_from_the_real_table_not_the_proxy(m23):
    """The Laplace prior is a training stand-in. Reporting its estimate as the
    side cost would be reporting a density, not bits the coder would pay."""
    source = inspect.getsource(m23.evaluate_arm)

    assert "hyper_latent_bits(" in source
    assert "side_proxy_bits_total" in source, "the proxy must be reported SEPARATELY"
    model = _model(m23, "hyper_only")
    symbols = torch.randint(0, SMALL["alphabet"], (2, 4, 64))
    signals = torch.randn(2, 4, 8, 8)
    train, _ = m23.collect_hyper_symbols(model, signals, device=torch.device("cpu"),
                                         hyper_bits=8)
    table = m23.fit_hyper_entropy_model(train, bits=8,
                                        hyper_channels=SMALL["hyper_channels"])
    result = m23.evaluate_arm(model, symbols, None, signals, device=torch.device("cpu"),
                              hyper_bits=8, hyper_entropy_model=table)

    assert result["side_bits_total"] > 0
    assert result["side_bits_total"] != result["side_proxy_bits_total"]


# ----------------------------------------------------------------------------------------
# The hyper-latent's quantization and its real bit cost
# ----------------------------------------------------------------------------------------


def test_quantization_is_integer_rounding_at_unit_step(m23):
    z = torch.tensor([[[[-1.4, -0.6, 0.0, 0.5, 2.49]]]])
    symbols, dequantized, fraction = m23.quantize_hyper_latent(z, bits=8)

    offset = m23.hyper_symbol_offset(8)
    assert dequantized.flatten().tolist() == [-1.0, -1.0, 0.0, 0.0, 2.0]
    assert (symbols.flatten() - offset).tolist() == [-1, -1, 0, 0, 2]
    assert fraction == 0.0


def test_a_hyper_latent_that_does_not_fit_its_alphabet_is_reported_not_hidden(m23):
    """A learned scale that overflows the alphabet silently changes what is being
    measured, so the clamped fraction is a reported number."""
    z = torch.full((1, 1, 2, 2), 10_000.0)
    _, dequantized, fraction = m23.quantize_hyper_latent(z, bits=8)

    assert fraction == 1.0
    assert float(dequantized.max()) == m23.hyper_symbol_offset(8) - 1


def test_the_real_side_cost_is_minus_log2_of_the_table_probability(m23):
    symbols = np.zeros((4, 2, 2, 2), dtype=np.int64) + 10
    symbols[:, 1] = 200
    table = m23.fit_hyper_entropy_model(symbols, bits=8, hyper_channels=2)

    bits = m23.hyper_latent_bits(table, symbols, hyper_channels=2)

    probabilities = table.probabilities()
    expected = -(4 * 4) * math.log2(probabilities[0][10]) \
               - (4 * 4) * math.log2(probabilities[1][200])
    assert bits == pytest.approx(expected, rel=1e-9)


def test_a_predictable_hyper_channel_is_cheaper_than_a_random_one(m23):
    """The accounting has to reward a cheap side channel, or the gate cannot tell
    an affordable hyperprior from an expensive one."""
    constant = np.full((8, 2, 2, 2), 128, dtype=np.int64)
    varied = constant.copy()
    varied[:, 1] = np.random.default_rng(0).integers(0, 256, size=varied[:, 1].shape)
    table_constant = m23.fit_hyper_entropy_model(constant, bits=8, hyper_channels=2)
    table_varied = m23.fit_hyper_entropy_model(varied, bits=8, hyper_channels=2)

    cheap = m23.hyper_latent_bits(table_constant, constant, hyper_channels=2)
    dear = m23.hyper_latent_bits(table_varied, varied, hyper_channels=2)

    assert cheap < dear


def test_laplace_smoothing_puts_a_FLOOR_on_the_side_cost(m23):
    """Worth pinning, because it is a cost the gate reports and not a bug.

    `EmpiricalEntropyModel.from_symbols` gives every symbol in the alphabet at
    least one count - the zero-probability safety every table in this codec
    relies on. On the hyper-latent's 2**8 alphabet that means even a PERFECTLY
    predictable channel pays -log2((n+1)/(n+256)) per value, because 255 unused
    symbols still hold probability mass. A constant channel with 32 observations
    costs ~3.1 bits/value, not ~0.

    The consequence for the milestone: the side channel's cost is driven by the
    hyper alphabet as much as by the hyper-latent's real entropy, so a cheaper
    `--hyper-bits` is a lever on it, and `side_share_of_net` is the number to
    watch rather than assumed negligible.
    """
    observations = 32
    constant = np.full((observations, 1, 1, 1), 128, dtype=np.int64)
    table = m23.fit_hyper_entropy_model(constant, bits=8, hyper_channels=1)

    per_value = m23.hyper_latent_bits(table, constant, hyper_channels=1) / observations
    floor = -math.log2((observations + 1) / (observations + 2 ** 8))

    # rel=1e-3, not exact: the table renormalises its Laplace counts to integer
    # frequencies summing to TOTAL_FREQUENCY (1<<16) for the range coder, so the
    # realised probability is the rounded one, not the exact ratio.
    assert per_value == pytest.approx(floor, rel=1e-3)
    assert per_value > 3.0
    # and a smaller alphabet is genuinely cheaper, which is why it is an option
    small = m23.fit_hyper_entropy_model(
        np.full((observations, 1, 1, 1), 8, dtype=np.int64), bits=4, hyper_channels=1)
    small_per_value = m23.hyper_latent_bits(
        small, np.full((observations, 1, 1, 1), 8, dtype=np.int64),
        hyper_channels=1) / observations
    assert small_per_value < per_value


def test_the_hyper_table_is_fitted_on_the_table_axis(m23):
    """`from_symbols` takes axis 1 as the table axis. Pre-transposing it, as an
    earlier draft of this module did, silently fits one table per POSITION."""
    symbols = np.zeros((3, 4, 2, 2), dtype=np.int64)
    for channel in range(4):
        symbols[:, channel] = 20 * channel

    table = m23.fit_hyper_entropy_model(symbols, bits=8, hyper_channels=4)

    assert table.frequencies.shape[0] == 4
    probabilities = table.probabilities()
    for channel in range(4):
        assert probabilities[channel].argmax() == 20 * channel


# ----------------------------------------------------------------------------------------
# The control
# ----------------------------------------------------------------------------------------


def test_the_control_is_a_whole_split_derangement(m23):
    order = m23.permuted_hyper_signal(400, seed=7)

    assert sorted(order.tolist()) == list(range(400))
    assert not np.any(order == np.arange(400)), "a fixed point leaks its own frame's signal"


def test_the_control_is_reproducible_and_seed_dependent(m23):
    assert np.array_equal(m23.permuted_hyper_signal(50, seed=1),
                          m23.permuted_hyper_signal(50, seed=1))
    assert not np.array_equal(m23.permuted_hyper_signal(50, seed=1),
                              m23.permuted_hyper_signal(50, seed=2))


def test_the_control_permutes_frames_not_positions_within_a_frame(m23):
    """A within-frame shuffle would leave the frame's own activity in place,
    which is exactly the leak M12 found. The control returns a frame ORDER."""
    order = m23.permuted_hyper_signal(12, seed=3)

    assert order.shape == (12,)
    assert order.dtype.kind in "iu"


def test_the_gate_runs_its_control_as_a_refitted_arm(gate):
    """A control evaluated with the uncontrolled model's weights measures the
    permutation's effect on a model fitted to the true signal, not the signal's
    contribution. It has to be trained under the permutation too."""
    source = inspect.getsource(gate.run_arm)

    assert "if control and model.uses_hyper" in source
    assert 'batch_for("train"' in source and 'batch_for("val_a"' in source
    # the permutation is applied inside batch_for, so it reaches every split
    assert source.index("if control and model.uses_hyper") < source.index('batch_for("train"')


# ----------------------------------------------------------------------------------------
# Geometry, signals, and pre-registration
# ----------------------------------------------------------------------------------------


def test_a_non_square_latent_is_refused_rather_than_guessed(m23):
    target = torch.zeros(1, 4, 12, dtype=torch.long)

    with pytest.raises(ValueError, match="cannot infer the latent geometry"):
        m23._spatial(target, None, None)
    assert m23._spatial(target, torch.zeros(1, 4, 3, 4), None) == (3, 4)
    assert m23._spatial(torch.zeros(1, 4, 16, dtype=torch.long), None, None) == (4, 4)


def test_h_a_is_given_the_decoders_residual_not_the_continuous_one(m23):
    """The conservative choice, and it has to stay conservative: handing h_a the
    continuous residual would give it information no real encoder path promises
    to reuse, biasing the gate UP."""
    from nvc.compression.calibration import calibrate_quantization_params
    from nvc.compression.codec import latent_to_symbols, symbols_to_latent

    torch.manual_seed(0)
    latent = torch.randn(1, 4, 4, 4)
    params = calibrate_quantization_params(latent, bits=4, mode="per_channel")
    symbols = latent_to_symbols(latent, params)
    expected = symbols_to_latent(symbols, (4, 4, 4), params)

    got = m23.dequantized_residual(symbols.reshape(1, 4, -1), params, shape=(4, 4, 4))

    assert torch.allclose(got, expected)
    assert not torch.allclose(got, latent), "the signal must be the DEQUANTIZED residual"


def test_the_gate_pre_registers_its_arms_bars_and_baselines(m23, gate):
    assert m23.ARMS == ("zref_only", "hyper_only", "hyper_plus_zref")
    assert m23.HYPER_ARMS == ("hyper_only", "hyper_plus_zref")
    assert (m23.WEAK_BELOW_PERCENT, m23.MEANINGFUL_ABOVE_PERCENT) == (0.5, 1.0)
    assert gate.CHANNEL_BASELINE == {"intra": "deployed_static_table",
                                     "residual": "zref_only"}


def test_the_intra_channel_has_no_zref_arm_because_there_is_no_zref(gate):
    """An intra frame has no reference. Offering `zref_only` there would be
    conditioning on a tensor that does not exist at decode time."""
    assert "zref_only" not in gate.CHANNEL_ARMS["intra"]
    assert gate.CHANNEL_ARMS["intra"] == ("hyper_only",)
    assert "zref_only" == gate.CHANNEL_BASELINE["residual"]


@pytest.mark.parametrize("gain,expected", [
    (0.0, "weak"), (0.49, "weak"), (0.5, "marginal"), (0.99, "marginal"),
    (1.0, "meaningful"), (12.0, "meaningful"), (-3.0, "weak"),
])
def test_verdict_uses_the_projects_pre_registered_bars(m23, gain, expected):
    assert m23.verdict(gain) == expected


def test_the_gate_states_what_it_cannot_say(gate):
    """The residual baseline is M10K's family, not the deployed stack. A reader
    who misses that would read a gain here as a gain over the shipped codec."""
    source = inspect.getsource(gate.main)

    assert "limitations" in source
    assert "NOT the deployed stack" in source
    assert "LOWER bound" in source
    context = inspect.getsource(gate.deployed_stack_context)
    assert "never subtracted" in context


# ----------------------------------------------------------------------------------------
# Training behaviour
# ----------------------------------------------------------------------------------------


def test_training_restores_the_best_validation_weights(m23):
    """The M10G convention deploys the best-validation model. Reporting a
    selected epoch while measuring the final weights is the defect M10I hit."""
    torch.manual_seed(0)
    model = _model(m23, "zref_only")
    symbols = torch.randint(0, SMALL["alphabet"], (4, 4, 64))
    references = torch.randn(4, 4, 8, 8)
    batch = (symbols, references, None)

    history = m23.train_arm(model, batch, batch, epochs=4, batch_size=2,
                            learning_rate=1e-2, device=torch.device("cpu"),
                            hyper_bits=8, log=lambda m: None)

    best = min(history, key=lambda h: (h["val_loss"], h["epoch"]))
    result = m23.evaluate_arm(model, symbols, references, None,
                              device=torch.device("cpu"), hyper_bits=8)
    assert result["net_bits_per_symbol"] == pytest.approx(best["val_loss"], rel=1e-4)


def test_training_reduces_bits_on_data_it_can_fit(m23):
    torch.manual_seed(0)
    model = _model(m23, "hyper_plus_zref")
    # one symbol value per channel: trivially learnable, so a failure here is the
    # training loop rather than the difficulty of the data
    symbols = torch.zeros(4, 4, 64, dtype=torch.long)
    for channel in range(4):
        symbols[:, channel] = channel
    references = torch.randn(4, 4, 8, 8)
    signals = torch.randn(4, 4, 8, 8)
    batch = (symbols, references, signals)

    history = m23.train_arm(model, batch, batch, epochs=6, batch_size=2,
                            learning_rate=1e-2, device=torch.device("cpu"),
                            hyper_bits=8, log=lambda m: None)

    assert history[-1]["val_a_bits"] < history[0]["val_a_bits"]
    assert history[-1]["val_a_bits"] < math.log2(SMALL["alphabet"])


def test_evaluation_is_deterministic(m23):
    torch.manual_seed(0)
    model = _model(m23, "hyper_plus_zref")
    symbols = torch.randint(0, SMALL["alphabet"], (3, 4, 64))
    references = torch.randn(3, 4, 8, 8)
    signals = torch.randn(3, 4, 8, 8)
    train, _ = m23.collect_hyper_symbols(model, signals, device=torch.device("cpu"),
                                         hyper_bits=8)
    table = m23.fit_hyper_entropy_model(train, bits=8,
                                        hyper_channels=SMALL["hyper_channels"])

    first = m23.evaluate_arm(model, symbols, references, signals,
                             device=torch.device("cpu"), hyper_bits=8,
                             hyper_entropy_model=table)
    second = m23.evaluate_arm(model, symbols, references, signals,
                              device=torch.device("cpu"), hyper_bits=8,
                              hyper_entropy_model=table)

    assert first["net_bits_per_symbol"] == second["net_bits_per_symbol"]
    assert first["side_bits_total"] == second["side_bits_total"]


# ----------------------------------------------------------------------------------------
# How the control is credited - M11's rule, not a plain subtraction
# ----------------------------------------------------------------------------------------


def test_a_negative_control_is_not_added_back(gate):
    """The defect this replaced: a plain `gain - control` credits an arm for its
    control being BAD. The 5-bit residual control lost 8.53%, which a plain
    subtraction turned into "+10.10% net" from a +1.57% measurement.

    M11 settled this already, in `m11_offline_gate.py`: "A negative control (the
    cost of a useless context) is NOT added back". A control that loses is a
    harmed model, not a null one.
    """
    assert gate.net_of_control(1.5734, -8.5301) == pytest.approx(1.5734)
    assert gate.net_of_control(6.6870, 6.0248) == pytest.approx(0.6622, abs=1e-4)
    assert gate.net_of_control(2.0, 0.0) == pytest.approx(2.0)
    # never inflates, never ignores a positive control
    for gain in (-3.0, 0.0, 1.0, 9.0):
        for control in (-9.0, -0.05, 0.0, 0.4, 12.0):
            net = gate.net_of_control(gain, control)
            assert net <= gain + 1e-12
            if control > 0:
                assert net == pytest.approx(gain - control)


def test_the_rule_matches_m11s_implementation(gate):
    """Not a new convention invented for M23 - the same expression M11 uses, so
    every gate in the project credits its control identically."""
    m11 = _load_script("m11_offline_gate")
    source = inspect.getsource(m11)

    assert 'max(result["control_gain_percent"], 0.0)' in source
    assert "max(control_percent, 0.0)" in inspect.getsource(gate.net_of_control)
    assert gate.CONTROL_TOLERANCE_PERCENT == m11.CONTROL_TOLERANCE_PERCENT


def test_the_control_is_interpreted_not_just_subtracted(gate):
    """A reader has to be able to tell the three cases apart: a control that
    gains (spurious), one that loses (confirms content specificity), and one that
    is ~zero (clean)."""
    assert "subtracted" in gate.control_interpretation(5.0)
    assert "actively misleads" in gate.control_interpretation(-8.5)
    assert "not credited" in gate.control_interpretation(-8.5)
    assert "approximately zero" in gate.control_interpretation(0.0)
    assert "approximately zero" in gate.control_interpretation(
        gate.CONTROL_TOLERANCE_PERCENT)
