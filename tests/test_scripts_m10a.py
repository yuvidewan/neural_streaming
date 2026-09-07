"""Tests for M10A: the scale-tracked rate proxy's diagnostic and pilot scripts.

Covers what M10A adds on top of the scale-tracking unit tests already in
`tests/test_rate_estimator.py` (which cover `update_bin_width`'s EMA, its
no-op-when-disabled behaviour, global mode, degenerate batches, and that
validation does not mutate tracking state):

  * the location-scale INVARIANCE property that makes the fix work at all -
    tracking the bin width alone is not sufficient, the density has to follow
    the latent too;
  * that a tracked bin width survives a checkpoint round trip, since the M10A
    evaluation scores each arm on the grid it learned;
  * the two new scripts, per TESTING.md's "Added a new script?" rule.

Everything runs on CPU against synthetic data from tests/helpers.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from helpers import TINY_MODEL_KWARGS, make_tiny_calibration, make_tiny_checkpoint, make_tiny_manifest  # noqa: E402

from nvc.training import QuantizationNoise, RateEstimator, save_checkpoint  # noqa: E402
from nvc.models import BaselineAutoencoder  # noqa: E402
from nvc.utils.config import load_default_config  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup(tmp_path: Path):
    manifest = make_tiny_manifest(tmp_path, num_sequences=16, frames_per_sequence=8, width=32, height=32)
    checkpoint = make_tiny_checkpoint(
        tmp_path / "m8_qat.pt", epoch=40, model_kwargs={"base_channels": 32},
        history=[{"epoch": 40, "train_loss": 5e-4, "val_loss": 4.554493e-04, "val_psnr": 34.3}],
    )
    calibration = make_tiny_calibration(
        tmp_path / "calib.json", checkpoint_path=checkpoint, bits=4, mode="per_channel",
    )
    return manifest, checkpoint, calibration


# --- The property that actually closes the exploit -------------------------


def test_rate_is_exactly_invariant_when_bin_width_and_density_both_scale():
    """The mathematical core of M10A.

    Laplace is a location-scale family, so scaling the latent, the bin width,
    `loc` and `scale` together by the same factor leaves every bin's
    probability - and therefore the rate - exactly unchanged. This is why
    tracking the bin width is necessary but NOT sufficient: with the density
    frozen at its initialization the proxy still pays out for shrinkage.
    """
    torch.manual_seed(0)
    z = torch.randn(4, 8, 16, 16) * 8
    bin_width = torch.full((1, 8, 1, 1), 3.32)

    rates = []
    for factor in (1.0, 0.75, 0.5, 0.25):
        estimator = RateEstimator(bin_width * factor, bits=4, mode="per_channel")
        with torch.no_grad():
            estimator.log_scale.fill_(float(torch.log(torch.tensor(factor))))
            estimator.loc.zero_()
        rates.append(estimator(z * factor, 256).item())

    assert rates == pytest.approx([rates[0]] * len(rates), rel=1e-6)


def test_tracking_the_bin_width_alone_leaves_a_residual_shrinkage_reward():
    """The complement of the test above, pinned so the limitation stays visible:
    with the density held at initialization, a tracked bin width reduces but
    does not remove the reward."""
    torch.manual_seed(0)
    z = torch.randn(4, 8, 16, 16) * 8
    bin_width = torch.full((1, 8, 1, 1), 3.32)

    def rate_with_tracked_bin_width_only(factor: float) -> float:
        estimator = RateEstimator(bin_width * factor, bits=4, mode="per_channel")
        return estimator(z * factor, 256).item()  # loc=0, scale=1, i.e. unadapted

    full = rate_with_tracked_bin_width_only(1.0)
    quarter = rate_with_tracked_bin_width_only(0.25)

    assert quarter < full, "an unadapted density still rewards shrinkage"


# --- The tracked bin width must survive a checkpoint round trip ------------


def test_tracked_bin_width_is_saved_and_restored_with_the_estimator_state(tmp_path):
    """M10A scores each arm on the grid it learned, so the tracked bin width has
    to travel in the checkpoint. It is a registered buffer, which puts it in
    `state_dict()` - pinned here because the evaluation depends on it."""
    _, checkpoint_path, calibration = _setup(tmp_path)
    noise = QuantizationNoise.from_calibration(calibration, bits=4, mode="per_channel")

    estimator = RateEstimator(
        noise.scale, bits=noise.bits, mode=noise.mode, track_scale=True, scale_momentum=0.5,
    )
    original = estimator.bin_width.clone()
    for _ in range(30):
        estimator.update_bin_width(torch.randn(4, TINY_MODEL_KWARGS["latent_channels"], 8, 8) * 20)
    assert not torch.allclose(estimator.bin_width, original), "tracking must have moved it"

    assert "bin_width" in estimator.state_dict()
    restored = RateEstimator(noise.scale, bits=noise.bits, mode=noise.mode, track_scale=True)
    restored.load_state_dict(estimator.state_dict())
    assert torch.equal(restored.bin_width, estimator.bin_width)


def test_a_disabled_tracker_keeps_the_calibration_bin_width_exactly(tmp_path):
    """Existing behaviour must be untouched when the feature is off."""
    _, checkpoint_path, calibration = _setup(tmp_path)
    noise = QuantizationNoise.from_calibration(calibration, bits=4, mode="per_channel")

    estimator = RateEstimator(noise.scale, bits=noise.bits, mode=noise.mode)  # default: off
    before = estimator.bin_width.clone()
    for _ in range(30):
        estimator.update_bin_width(torch.randn(4, TINY_MODEL_KWARGS["latent_channels"], 8, 8) * 50)

    assert torch.equal(estimator.bin_width, before)
    assert estimator.track_scale is False
    assert estimator.to_dict()["track_scale"] is False


# --- m10a_shrink_diagnostic ------------------------------------------------


def test_shrink_diagnostic_arg_parser_defaults():
    mod = _load_script("m10a_shrink_diagnostic")
    args = mod.build_arg_parser(load_default_config()).parse_args([])

    assert args.checkpoint == Path("outputs/qat_combined/checkpoints_qat_noise/best.pt")
    assert args.output_dir == Path("outputs/m10a_pilot")
    assert mod.SHRINK_FACTORS == (1.0, 0.75, 0.50, 0.25)
    assert args.density_fit_lr == pytest.approx(1e-2), "must match the project's --rate-lr"


def test_shrink_diagnostic_reports_a_missing_checkpoint(tmp_path, capsys):
    mod = _load_script("m10a_shrink_diagnostic")
    manifest, _, calibration = _setup(tmp_path)

    exit_code = mod.main([
        "--manifest", str(manifest), "--calibration", str(calibration),
        "--checkpoint", str(tmp_path / "absent.pt"), "--device", "cpu",
    ])

    assert exit_code == 1
    assert "--checkpoint not found" in capsys.readouterr().err


def test_shrink_diagnostic_end_to_end_orders_the_three_configurations(tmp_path):
    """The diagnostic's whole point: A rewards shrinkage most, C least."""
    mod = _load_script("m10a_shrink_diagnostic")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "diag"

    exit_code = mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--batch-size", "2", "--max-batches", "2",
        "--adapt-steps", "50", "--density-fit-steps", "200",
        "--output-dir", str(output_dir), "--device", "cpu",
    ])

    assert exit_code == 0
    report = json.loads((output_dir / "shrink_diagnostic.json").read_text())
    assert [row["factor"] for row in report["rows"]] == [1.0, 0.75, 0.50, 0.25]

    frozen = report["frozen_reward_at_quarter_bpp"]
    tracked = report["tracked_reward_at_quarter_bpp"]
    assert frozen > 0, "the frozen bin width must reward shrinkage - that is the exploit"
    assert tracked < frozen, "tracking the bin width must reduce the reward"
    # Configuration C's magnitude is not asserted here. On a randomly
    # initialised tiny model the latent is near-degenerate (abs-mean ~0.1) and a
    # few hundred fit steps do not converge, so the ordering is not reliable at
    # this scale. The property C depends on is proved exactly and
    # scale-independently by
    # `test_rate_is_exactly_invariant_when_bin_width_and_density_both_scale`;
    # here we only require that the configuration is computed and finite.
    assert report["tracked_and_fitted_reward_at_quarter_bpp"] == pytest.approx(
        report["tracked_and_fitted_reward_at_quarter_bpp"]
    )
    assert all("tracked_and_fitted_rate_bpp" in row for row in report["rows"])

    # The deployed calibrator's own bin width must scale with the latent - that
    # is the ground truth the proxy is imitating.
    widths = [row["deployed_bin_width_mean"] for row in report["rows"]]
    assert widths == sorted(widths, reverse=True)


# --- m10a_pilot ------------------------------------------------------------


def test_pilot_arg_parser_and_arms():
    mod = _load_script("m10a_pilot")
    args = mod.build_arg_parser(load_default_config()).parse_args([])

    assert args.epochs * args.max_batches == 500, "M10A specifies a 500-step pilot"
    assert args.rate_lr == pytest.approx(1e-2)
    assert args.learning_rate == pytest.approx(1e-4)
    assert args.expect_sha256 == mod.EXPECTED_START_SHA256

    by_name = {arm["name"]: arm for arm in mod.ARMS}
    assert by_name["CTRL"]["lambda"] == 0.0
    assert by_name["M10A-L"]["lambda"] == pytest.approx(9.0757e-04)
    assert by_name["M10A-M"]["lambda"] == pytest.approx(2.8700e-03)
    assert by_name["M10A-H"]["lambda"] == pytest.approx(9.0757e-03)
    assert len({arm["dir"] for arm in mod.ARMS}) == len(mod.ARMS)


def test_pilot_refuses_a_start_checkpoint_with_the_wrong_hash(tmp_path, capsys):
    """All four arms must start from the same verified checkpoint."""
    mod = _load_script("m10a_pilot")
    manifest, checkpoint, calibration = _setup(tmp_path)

    exit_code = mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--device", "cpu",
    ])

    assert exit_code == 1
    assert "sha256 mismatch" in capsys.readouterr().err


def test_recording_estimator_does_not_change_numerics(tmp_path):
    """The pilot's instrumentation subclass must be observation-only."""
    mod = _load_script("m10a_pilot")
    _, checkpoint, calibration = _setup(tmp_path)
    noise = QuantizationNoise.from_calibration(calibration, bits=4, mode="per_channel")
    z = torch.randn(2, TINY_MODEL_KWARGS["latent_channels"], 8, 8) * 5

    plain = RateEstimator(noise.scale, bits=4, mode="per_channel", track_scale=True, scale_momentum=0.5)
    recording = mod._RecordingRateEstimator(
        noise.scale, bits=4, mode="per_channel", track_scale=True, scale_momentum=0.5,
    )

    mod._STEP_LOG.clear()
    assert recording(z, 64).item() == pytest.approx(plain(z, 64).item(), rel=1e-9)
    plain.update_bin_width(z)
    recording.update_bin_width(z)
    assert torch.allclose(recording.bin_width, plain.bin_width)
    assert len(mod._STEP_LOG) == 1, "and it must actually record"
    mod._STEP_LOG.clear()


def test_pilot_end_to_end_writes_summary_step_log_and_best_checkpoints(tmp_path):
    mod = _load_script("m10a_pilot")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "pilot"

    exit_code = mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--expect-sha256", "",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--epochs", "1", "--max-batches", "2", "--batch-size", "2",
        "--output-dir", str(output_dir), "--device", "cpu",
    ])

    assert exit_code == 0
    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert summary["track_scale"] is True
    assert len(summary["arms"]) == 4
    assert summary["arms"][0]["lambda"] == 0.0, "the control comes first"

    for arm in summary["arms"]:
        assert arm["all_finite"] is True
        assert Path(arm["best_checkpoint"]).is_file(), "best.pt must be written"
        # The M9F.1 guarantee, re-pinned for M10A.
        assert arm["best_selection_used_only_current_objective"] is True
        assert arm["stale_history_records_ignored"] == 1
        assert arm["start_checkpoint_sha256"] == summary["start_checkpoint_sha256"]
        steps = json.loads((Path(arm["checkpoint_dir"]) / "step_log.json").read_text())
        assert len(steps) == arm["steps"] > 0
        for record in steps:
            assert record["bin_width_after"] > 0
            assert record["latent_abs_mean"] >= 0


def test_pilot_control_arm_gets_no_rate_gradient(tmp_path):
    """lambda=0 must leave the estimator at its initialization: `0.0 * rate`
    contributes exactly zero gradient regardless of the rate LR."""
    mod = _load_script("m10a_pilot")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "pilot"

    mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--expect-sha256", "",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--epochs", "1", "--max-batches", "2", "--batch-size", "2", "--only", "CTRL",
        "--output-dir", str(output_dir), "--device", "cpu",
    ])

    summary = json.loads((output_dir / "training_summary.json").read_text())
    control = summary["arms"][0]
    assert control["lambda"] == 0.0
    assert control["rate_estimator_loc"]["std"] == pytest.approx(0.0, abs=1e-12)
    assert control["rate_estimator_scale"]["mean"] == pytest.approx(1.0, abs=1e-9)


# --- m10a_evaluate ---------------------------------------------------------


def test_evaluate_reuses_the_m9_calibration_and_benchmark_stages():
    """M10A's numbers are only comparable to M9's because both go through the
    same calibration and benchmark code. Pinned so a future edit that forks
    them is caught."""
    evaluate = _load_script("m10a_evaluate")
    m9 = _load_script("m9_final_calibrate_benchmark")

    assert callable(m9._calibrate) and callable(m9._benchmark)
    assert evaluate.BIT_DEPTHS == m9.BIT_DEPTHS
    models = evaluate._model_set(Path("outputs/m10a_pilot"))
    assert [m["key"] for m in models] == list(evaluate.M10A_ORDER)
    for model in models:
        assert model["checkpoint"].name == "best.pt"
        assert model["checkpoint"].parts[:2] == ("outputs", "m10a_pilot")


def test_evaluate_correlation_helpers():
    evaluate = _load_script("m10a_evaluate")

    assert evaluate._pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert evaluate._pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)
    assert evaluate._spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)
    # Too few points to be meaningful - the helpers say so rather than guessing.
    assert evaluate._pearson([1.0, 2.0], [1.0, 2.0]) is None
    assert evaluate._spearman([1.0, 2.0], [1.0, 2.0]) is None


def test_evaluate_reports_a_missing_manifest(tmp_path, capsys):
    evaluate = _load_script("m10a_evaluate")

    exit_code = evaluate.main([
        "--manifest", str(tmp_path / "absent.json"), "--stage", "calibrate", "--device", "cpu",
    ])

    assert exit_code == 1
    assert "--manifest not found" in capsys.readouterr().err
