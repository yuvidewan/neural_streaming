"""M23 Phase B1 - tests for `scripts/m23_phaseb_gate.py`.

The load-bearing tests:

  * `test_the_transmitted_context_is_charged_in_every_arm_including_the_control`
    - a transmitted context that is not charged is not a hyperprior, it is a
    free oracle, and every number would be meaningless;
  * `test_the_baseline_is_the_full_stack_not_m10ks_family` - the entire point of
    Phase B is that Phase A's baseline was too weak;
  * `test_the_context_is_shared_by_every_channel_at_a_position` - the map is
    spatial, and a per-channel context would be a different, much more
    expensive thing;
  * `test_no_causality_check_is_made_because_the_context_is_transmitted` - the
    one place this gate must deliberately differ from M10J/M11/M12.
"""

from __future__ import annotations

import importlib.util
import inspect
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
def pb():
    return _load_script("m23_phaseb_gate")


# ----------------------------------------------------------------------------------------
# The charge
# ----------------------------------------------------------------------------------------


def test_the_transmitted_context_is_charged_in_every_arm_including_the_control(pb):
    """The control transmits exactly as much as the real arm. Charging only the
    real arm would flatter the control and understate the net gain; charging
    neither would make both free."""
    source = inspect.getsource(pb.main)

    assert "side_b_per_symbol" in source
    assert '"net_bits_val_b": bits_b + side_b_per_symbol' in source
    # the loop that builds that net covers real AND control
    assert 'for label, train_c, val_c in (("real", t_ctx, v_ctx),' in source
    # and the smoothing strength is selected on a charged objective too
    assert "+ side_a_per_symbol" in source


def test_the_charge_is_the_real_table_cost_of_the_map_that_is_transmitted(pb):
    """Charged for the coarse binned map the arm consumes - not the full
    hyper-latent it may have been derived from, and not a density estimate."""
    rng = np.random.default_rng(0)
    train = rng.integers(0, 8, size=(10, 4, 4)).astype(np.int64)

    bits = pb.context_map_bits(train, train, bins=8)

    # uniform over 8 symbols is 3 bits/value; Laplace smoothing moves it slightly
    assert bits / train.size == pytest.approx(3.0, abs=0.05)
    # a skewed map is cheaper, which is what an entropy coder is for
    skewed = np.zeros_like(train)
    skewed[0, 0, 0] = 7
    assert pb.context_map_bits(skewed, skewed, bins=8) < bits


def test_the_pre_charge_gain_is_reported_so_the_charge_can_decide_the_verdict(pb):
    """The charge has to be able to turn a gain into a loss, or it is decoration."""
    source = inspect.getsource(pb.main)

    assert '"symbol_only_gain_percent"' in source
    assert "before the charge" in source


def test_the_bin_count_must_be_a_power_of_two(pb):
    """The map is coded on an EmpiricalEntropyModel alphabet, which is 2**bits."""
    values = torch.randn(4, 1, 8, 8)

    assert pb.train_bin_edges(values, bins=8).size == 7
    with pytest.raises(ValueError, match="power of two"):
        pb.train_bin_edges(values, bins=6)


# ----------------------------------------------------------------------------------------
# The baseline is the thing Phase A could not use
# ----------------------------------------------------------------------------------------


def test_the_baseline_is_the_full_stack_not_m10ks_family(pb):
    """Phase A's baseline was `zref_only`. If Phase B measured against that too,
    it would answer nothing."""
    source = inspect.getsource(pb.main)

    assert 'baselines["m11_g16_recalibrated"]' in source
    assert 'baseline = baselines["m11_g16_recalibrated"]' in source
    # `zref_only` may be NAMED in the prose that explains why Phase B exists, but
    # it must never be a baseline this gate measures against
    assert 'baselines["zref_only"]' not in source
    assert "baseline = baselines[\"zref_only\"]" not in source
    assert "m11_g16_recalibrated (the full deployed stack" in inspect.getsource(pb)


def test_both_of_m12s_baselines_are_reported(pb):
    """The deployed codebook AND its recalibration, so the recalibration is never
    mistaken for a context gain."""
    source = inspect.getsource(pb.main)

    assert 'baselines["m11_g16_deployed"]' in source
    assert "cx.fit_parent" in source
    assert "cx.smoothed_parent" in source
    assert "cx.select_strength" in source


def test_it_reuses_m12s_prototype_indices_rather_than_reimplementing_them(pb):
    """A second implementation of the assignment would make the two milestones'
    numbers incomparable, which is the whole value of reusing the gate."""
    source = inspect.getsource(pb.main)

    assert "m12.m11_g16_prototype_indices" in source
    assert "m12_spatial_offline_gate" in source


# ----------------------------------------------------------------------------------------
# The context's shape
# ----------------------------------------------------------------------------------------


def test_the_context_is_shared_by_every_channel_at_a_position(pb):
    binned = np.arange(2 * 4 * 4).reshape(2, 4, 4) % 8

    context = pb.broadcast_to_symbol_positions(binned, channels=3, height=8, width=8)

    assert context.shape == (2, 3 * 8 * 8)
    per_channel = context.reshape(2, 3, 8, 8)
    assert np.array_equal(per_channel[:, 0], per_channel[:, 1])
    assert np.array_equal(per_channel[:, 0], per_channel[:, 2])


def test_the_coarse_map_upsamples_the_way_a_decoder_would(pb):
    """Nearest, because the decoder rebuilds the map from the coarse values it was
    sent - a smoother reconstruction would use information never transmitted."""
    binned = np.array([[[0, 7]]], dtype=np.int64)       # 1 frame, 1x2

    context = pb.broadcast_to_symbol_positions(binned, channels=1, height=2, width=4)

    assert context.reshape(1, 1, 2, 4)[0, 0].tolist() == [[0, 0, 7, 7], [0, 0, 7, 7]]


def test_coarsening_reduces_the_values_transmitted(pb):
    values = torch.randn(3, 1, 16, 16)

    assert pb.coarsen(values, stride=1).shape == (3, 1, 16, 16)
    assert pb.coarsen(values, stride=2).shape == (3, 1, 8, 8)
    assert pb.coarsen(values, stride=4).shape == (3, 1, 4, 4)


def test_quantile_bins_are_populated_and_fitted_on_train_only(pb):
    torch.manual_seed(0)
    train = torch.randn(20, 1, 8, 8).abs()
    edges = pb.train_bin_edges(train, bins=8)
    binned = pb.to_bins(train, edges)

    counts = np.bincount(binned.reshape(-1), minlength=8)
    assert (counts > 0).all()
    assert counts.max() / counts.min() < 1.2, "quantile bins should be near-equal on TRAIN"
    # a shifted VAL distribution is binned by TRAIN's edges, not refitted
    shifted = pb.to_bins(train + 10.0, edges)
    assert shifted.min() == 7, "every shifted value lands in the top TRAIN bin"


def test_the_binning_documents_that_it_is_the_pessimistic_choice_on_cost(pb):
    """Quantile edges make the map's marginal uniform, which is the most
    expensive distribution it could have. That is deliberate and has to stay
    written down, because a cheaper binning would change the verdict."""
    source = inspect.getsource(pb.train_bin_edges)

    assert "UNIFORM" in source
    assert "pessimistic" in source


# ----------------------------------------------------------------------------------------
# Transmitted, not causal - the deliberate difference from every prior context gate
# ----------------------------------------------------------------------------------------


def test_no_causality_check_is_made_because_the_context_is_transmitted(pb):
    """M10J/M11/M12 all check causality, because their contexts are functions of
    already-decoded symbols. This one is sent in the bitstream: it is available
    before any symbol is decoded, so a causality check would be meaningless -
    and the charge is what replaces it. Both halves are asserted so neither can
    be dropped without a failure."""
    source = inspect.getsource(pb)

    # the prose explains why the check is absent, so what must be absent is the
    # CALL - any `cx.check_causality(...)` on this gate's context
    assert "cx.check_causality(" not in source
    assert "check_causality(" not in inspect.getsource(pb.main)
    assert "transmitted_not_causal" in source
    assert "differs_from_prior_context_gates" in source
    assert "worthless unless" in source


def test_the_scale_map_arm_is_the_mechanism_with_nothing_learned(pb):
    """This arm separates "does a transmitted scale map help" from "did our
    analysis transform learn a good one"."""
    residual = torch.tensor([[[[-2.0, 2.0]], [[-4.0, 0.0]]]])    # 1 frame, 2 ch, 1x2

    signal = pb.scale_map_signal(residual)

    assert signal.shape == (1, 1, 1, 2)
    assert signal[0, 0, 0, 0] == pytest.approx(3.0)      # mean(|-2|, |-4|)
    assert signal[0, 0, 0, 1] == pytest.approx(1.0)      # mean(|2|, |0|)


def test_the_learned_arm_quantizes_before_averaging(pb):
    """The decoder only ever has the quantized hyper-latent, so averaging the
    continuous values first would use precision that was never sent."""
    source = inspect.getsource(pb.learned_signal)

    assert source.index("quantize_hyper_latent") < source.index("mean(dim=1")
    assert "only ever has the quantized values" in source


# ----------------------------------------------------------------------------------------
# Control crediting and pre-registration
# ----------------------------------------------------------------------------------------


def test_the_control_is_credited_by_m11s_rule(pb):
    source = inspect.getsource(pb.main)

    assert "gate_a.net_of_control(" in source
    assert "gate_a.control_interpretation(" in source


def test_the_control_permutes_each_split_separately(pb):
    """Permuting VAL-A and VAL-B together would move context between the split
    that selects the smoothing strength and the split that reports."""
    source = inspect.getsource(pb.main)

    assert "v_perm[select_mask] = cx.permuted_context(v_ctx[select_mask], rng)" in source
    assert "v_perm[report_mask] = cx.permuted_context(v_ctx[report_mask], rng)" in source


def test_pre_registration_is_recorded_in_the_report(pb):
    source = inspect.getsource(pb.main)

    assert '"pre_registered"' in source
    for key in ("arms", "baseline", "weak_below_percent", "meaningful_above_percent",
                "control_tolerance_percent", "net_of_control_rule", "split_discipline"):
        assert f'"{key}"' in source
    assert pb.ARMS == ("scale_map", "learned")
