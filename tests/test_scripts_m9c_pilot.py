"""Tests for the Milestone 9C scripts: m9c_rd_diagnostic and m9c_lambda_pilot.

Per this project's testing standard (TESTING.md, "Added a new script?"), each
covers `build_arg_parser` defaults, a missing-required-file error path, and a
happy-path `main(argv)` run against synthetic data that checks the exit code
AND the files actually written.

Everything runs on CPU against tiny synthetic data from tests/helpers.py - no
GPU, no DAVIS, no real checkpoint. The real-data runs these scripts perform
for the milestone itself are recorded in MILESTONE_9_PLAN.md, not re-executed
here.
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

from nvc.utils.config import load_default_config  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli_checkpoint(path: Path, *, epoch: int = 1) -> Path:
    """Built at the CLI's default base_channels=32 so --resume can load it -
    see tests/test_m9c_resume_model_only.py's own helper for why."""
    return make_tiny_checkpoint(path, epoch=epoch, model_kwargs={"base_channels": 32})


def _tiny_setup(tmp_path: Path):
    manifest = make_tiny_manifest(tmp_path, num_sequences=8, frames_per_sequence=4, width=32, height=32)
    checkpoint = _cli_checkpoint(tmp_path / "m8_qat.pt", epoch=40)
    calibration = make_tiny_calibration(
        tmp_path / "calib.json", checkpoint_path=checkpoint, bits=4, mode="per_channel",
    )
    return manifest, checkpoint, calibration


# --- m9c_rd_diagnostic -----------------------------------------------------


def test_diagnostic_arg_parser_defaults_point_at_the_m8_qat_artifacts():
    mod = _load_script("m9c_rd_diagnostic")
    args = mod.build_arg_parser(load_default_config()).parse_args([])

    assert args.checkpoint == Path("outputs/qat_combined/checkpoints_qat_noise/best.pt")
    assert args.calibration == Path("outputs/calibration/vimeo_epoch17_4bit.json")
    assert args.output_dir == Path("outputs/m9c_lambda_pilot")
    assert args.qat_bits == 4
    assert args.qat_mode == "per_channel"
    assert args.seed == 42


def test_diagnostic_reports_a_missing_checkpoint(tmp_path, capsys):
    mod = _load_script("m9c_rd_diagnostic")
    manifest, _, calibration = _tiny_setup(tmp_path)

    exit_code = mod.main([
        "--manifest", str(manifest), "--calibration", str(calibration),
        "--checkpoint", str(tmp_path / "absent.pt"), "--device", "cpu",
    ])

    assert exit_code == 1
    assert "--checkpoint not found" in capsys.readouterr().err


def test_diagnostic_writes_a_report_and_leaves_model_weights_untouched(tmp_path):
    mod = _load_script("m9c_rd_diagnostic")
    manifest, checkpoint, calibration = _tiny_setup(tmp_path)
    output_dir = tmp_path / "diag"

    exit_code = mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--batch-size", "2",
        "--max-batches", "2", "--warm-fit-steps", "3",
        "--output-dir", str(output_dir), "--device", "cpu",
    ])

    assert exit_code == 0
    report = json.loads((output_dir / "dr_diagnostic.json").read_text())

    # The measurement this milestone's lambda grid is derived from.
    assert report["lambda_balance"]["from_fitted_rate"] > 0
    assert report["lambda_balance"]["recommended_basis"] == "from_fitted_rate"
    # The script's central promise: it trains nothing but the rate estimator.
    assert report["model_weights_unchanged"]["unchanged"] is True
    for split in ("train_split_at_init", "train_split_fitted", "val_split_fitted"):
        assert report[split]["all_finite"] is True
        assert report[split]["rate_non_negative"] is True


def test_diagnostic_sanity_checks_all_pass_on_a_tiny_model(tmp_path):
    """Phase 3's contract, on CPU where kernels are deterministic anyway."""
    mod = _load_script("m9c_rd_diagnostic")
    manifest, checkpoint, calibration = _tiny_setup(tmp_path)
    output_dir = tmp_path / "diag"

    mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--batch-size", "2",
        "--max-batches", "2", "--warm-fit-steps", "3",
        "--output-dir", str(output_dir), "--device", "cpu",
    ])
    checks = json.loads((output_dir / "dr_diagnostic.json").read_text())["sanity_checks"]

    assert checks["A_more_mass_less_rate"]["mode_cheaper_than_tail"] is True
    assert checks["A_more_mass_less_rate"]["sweep_is_non_decreasing"] is True
    assert checks["A2_cdf_form_matches_reference"]["agrees"] is True
    assert checks["B_finite_non_negative"]["non_negative"] is True
    assert checks["C_gradients_exist"]["all_nonzero"] is True
    # lambda=0 must be the distortion-only path exactly, not approximately.
    assert checks["D_lambda_zero_equals_distortion_only"]["gradients_identical"] is True
    assert checks["D_lambda_zero_equals_distortion_only"]["params_identical_after_step"] is True
    # lambda>0 must reach the encoder, and cannot reach the decoder.
    assert checks["E_lambda_positive_adds_rate_gradient"]["encoder_receives_rate_contribution"] is True
    assert checks["E_lambda_positive_adds_rate_gradient"]["decoder_unchanged_as_expected"] is True


def test_diagnostic_rejects_a_non_positive_batch_count(tmp_path):
    mod = _load_script("m9c_rd_diagnostic")
    manifest, checkpoint, calibration = _tiny_setup(tmp_path)

    with pytest.raises(SystemExit):
        mod.main([
            "--manifest", str(manifest), "--checkpoint", str(checkpoint),
            "--calibration", str(calibration), "--max-batches", "0", "--device", "cpu",
        ])


# --- m9c_lambda_pilot ------------------------------------------------------


def test_pilot_arg_parser_defaults_and_lambda_grid_shape():
    mod = _load_script("m9c_lambda_pilot")
    args = mod.build_arg_parser(load_default_config()).parse_args([])

    assert args.checkpoint == Path("outputs/qat_combined/checkpoints_qat_noise/best.pt")
    assert args.diagnostic == Path("outputs/m9c_lambda_pilot/dr_diagnostic.json")
    assert args.lambda_balance is None, "a real run must measure the balance, not assume it"
    assert len(mod.LAMBDA_MULTIPLIERS) == len(mod.REGIME_LABELS) == 5


def test_pilot_lambda_grid_is_log_spaced_around_the_balance_point():
    mod = _load_script("m9c_lambda_pilot")
    balance = 9.0757e-4
    grid = mod._lambda_grid(balance)

    assert [entry["lambda"] for entry in grid] == pytest.approx(
        [balance * m for m in mod.LAMBDA_MULTIPLIERS]
    )
    # Balance sits in the middle, one decade either side.
    assert grid[2]["lambda"] == pytest.approx(balance)
    assert grid[0]["lambda"] == pytest.approx(balance / 10)
    assert grid[-1]["lambda"] == pytest.approx(balance * 10)
    # Constant ratio between neighbours = genuinely logarithmic spacing.
    ratios = [b["lambda"] / a["lambda"] for a, b in zip(grid, grid[1:])]
    assert ratios == pytest.approx([ratios[0]] * len(ratios))


def test_pilot_requires_a_diagnostic_file(tmp_path, capsys):
    """Rule 4/5: lambda must come from measurement, so a missing diagnostic is
    a hard error rather than a silent fallback to some default."""
    mod = _load_script("m9c_lambda_pilot")
    manifest, checkpoint, calibration = _tiny_setup(tmp_path)

    exit_code = mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--diagnostic", str(tmp_path / "absent.json"),
        "--device", "cpu",
    ])

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "--diagnostic not found" in error
    assert "m9c_rd_diagnostic.py" in error


def test_pilot_reports_a_missing_start_checkpoint(tmp_path, capsys):
    mod = _load_script("m9c_lambda_pilot")
    manifest, _, calibration = _tiny_setup(tmp_path)

    exit_code = mod.main([
        "--manifest", str(manifest), "--calibration", str(calibration),
        "--checkpoint", str(tmp_path / "absent.pt"), "--lambda-balance", "1e-3",
        "--device", "cpu",
    ])

    assert exit_code == 1
    assert "--checkpoint not found" in capsys.readouterr().err


def test_pilot_rejects_a_non_positive_lambda_balance(tmp_path, capsys):
    mod = _load_script("m9c_lambda_pilot")
    manifest, checkpoint, calibration = _tiny_setup(tmp_path)

    exit_code = mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--lambda-balance", "0", "--device", "cpu",
    ])

    assert exit_code == 1
    assert "must be finite and > 0" in capsys.readouterr().err


def test_pilot_end_to_end_writes_all_summaries(tmp_path):
    mod = _load_script("m9c_lambda_pilot")
    manifest, checkpoint, calibration = _tiny_setup(tmp_path)
    output_dir = tmp_path / "pilot"

    exit_code = mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--lambda-balance", "1e-3",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--epochs", "1", "--max-batches", "1", "--batch-size", "2",
        "--output-dir", str(output_dir), "--device", "cpu", "--no-plot",
    ])

    assert exit_code == 0
    summary = json.loads((output_dir / "lambda_pilot_summary.json").read_text())
    assert summary["lambda_balance"] == 1e-3
    # Five lambda arms plus the control.
    assert len(summary["arms"]) == 6
    assert summary["arms"][0]["lambda"] == 0.0
    assert [arm["lambda"] for arm in summary["arms"][1:]] == pytest.approx(
        [1e-3 * m for m in mod.LAMBDA_MULTIPLIERS]
    )
    for arm in summary["arms"]:
        assert arm["all_finite"] is True
        assert arm["rate_estimator_state_restored"] is True, "each arm must be scored by its own density"
        assert Path(arm["checkpoint_dir"]).is_dir()

    assert (output_dir / "lambda_pilot_summary.csv").is_file()
    table = (output_dir / "lambda_pilot_table.md").read_text()
    assert table.startswith("| lambda |")
    assert table.count("\n") == 8  # header + separator + 6 arms


def test_pilot_control_arm_can_be_skipped(tmp_path):
    mod = _load_script("m9c_lambda_pilot")
    manifest, checkpoint, calibration = _tiny_setup(tmp_path)
    output_dir = tmp_path / "pilot_nc"

    exit_code = mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--lambda-balance", "1e-3",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--epochs", "1", "--max-batches", "1", "--batch-size", "2",
        "--output-dir", str(output_dir), "--device", "cpu", "--no-plot", "--skip-control",
    ])

    assert exit_code == 0
    summary = json.loads((output_dir / "lambda_pilot_summary.json").read_text())
    assert len(summary["arms"]) == 5
    assert all(arm["lambda"] > 0 for arm in summary["arms"])


def test_pilot_arms_do_not_share_a_checkpoint_directory(tmp_path):
    """Every arm must be independent - M9C forbids continuing one lambda from
    another, and colliding directories would silently do exactly that."""
    mod = _load_script("m9c_lambda_pilot")
    names = {mod._arm_name(value) for value in (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)}
    assert len(names) == 6
    assert mod._arm_name(0.0) == "control_lambda0"
    assert all("." not in name for name in names), "directory names must be filesystem-safe"


# --- m9c_failure_modes -----------------------------------------------------


def test_failure_modes_arg_parser_defaults():
    mod = _load_script("m9c_failure_modes")
    args = mod.build_arg_parser(load_default_config()).parse_args([])

    assert args.summary == Path("outputs/m9c_lambda_pilot/lambda_pilot_summary.json")
    assert args.calibration == Path("outputs/calibration/vimeo_epoch17_4bit.json")
    assert args.refit_steps == 600
    assert args.qat_bits == 4


def test_failure_modes_reports_a_missing_summary(tmp_path, capsys):
    mod = _load_script("m9c_failure_modes")
    manifest, _, calibration = _tiny_setup(tmp_path)

    exit_code = mod.main([
        "--manifest", str(manifest), "--calibration", str(calibration),
        "--summary", str(tmp_path / "absent.json"), "--device", "cpu",
    ])

    assert exit_code == 1
    assert "--summary not found" in capsys.readouterr().err


def test_symbol_entropy_of_a_constant_latent_is_zero():
    """A latent quantizing to a single symbol carries no information."""
    mod = _load_script("m9c_failure_modes")
    from nvc.compression.quantization import QuantizationParams

    params = QuantizationParams(
        scale=torch.ones(1, 2, 1, 1), zero_point=torch.zeros(1, 2, 1, 1),
        bits=4, mode="per_channel",
    )
    symbols = torch.full((1, 2, 4, 4), 3, dtype=torch.int32)

    assert mod._symbol_entropy_bpp(symbols, params, image_pixels=16) == pytest.approx(0.0)


def test_symbol_entropy_of_a_uniform_latent_matches_the_hand_computed_value():
    """Two channels, each uniform over 4 symbols = 2 bits per element.

    32 elements x 2 bits = 64 bits over 16 input pixels = 4.0 bpp.
    """
    mod = _load_script("m9c_failure_modes")
    from nvc.compression.quantization import QuantizationParams

    params = QuantizationParams(
        scale=torch.ones(1, 2, 1, 1), zero_point=torch.zeros(1, 2, 1, 1),
        bits=4, mode="per_channel",
    )
    pattern = torch.tensor([0, 1, 2, 3], dtype=torch.int32).repeat(4)
    symbols = pattern.reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)

    assert mod._symbol_entropy_bpp(symbols, params, image_pixels=16) == pytest.approx(4.0)


def test_failure_modes_end_to_end_on_a_tiny_pilot(tmp_path):
    """Runs the real pilot first, then analyses its own output - this is the
    schema handoff between the two scripts, so it is exercised end to end."""
    pilot = _load_script("m9c_lambda_pilot")
    analysis = _load_script("m9c_failure_modes")
    manifest, checkpoint, calibration = _tiny_setup(tmp_path)
    output_dir = tmp_path / "pilot"

    assert pilot.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--lambda-balance", "1e-3",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--epochs", "1", "--max-batches", "1", "--batch-size", "2",
        "--output-dir", str(output_dir), "--device", "cpu", "--no-plot",
    ]) == 0

    exit_code = analysis.main([
        "--manifest", str(manifest), "--calibration", str(calibration),
        "--summary", str(output_dir / "lambda_pilot_summary.json"),
        "--batch-size", "2", "--max-batches", "2", "--refit-steps", "5",
        "--output-dir", str(output_dir), "--device", "cpu",
    ])

    assert exit_code == 0
    report = json.loads((output_dir / "failure_mode_analysis.json").read_text())
    assert len(report["arms"]) == 6
    for arm in report["arms"]:
        assert arm["flags"]["non_finite"] is False
        assert arm["rate_matched_refit_bpp"] >= 0
        assert arm["symbol_entropy_bpp"] >= 0
        assert 0.0 <= arm["clipped_percent_on_frozen_grid"] <= 100.0
    assert "estimator_gaming" in report["verdict"]
    assert "frozen_grid_mismatched" in report["verdict"]


# --- m9c1_adaptation_check + the --rate-lr pass-through (Milestone 9C.1) ----


def test_pilot_passes_rate_lr_through_to_every_arm(tmp_path):
    """The sweep must train every arm at the rate LR it was asked for -
    otherwise M9C.1's fix silently would not reach the pilot."""
    mod = _load_script("m9c_lambda_pilot")
    manifest, checkpoint, calibration = _tiny_setup(tmp_path)
    output_dir = tmp_path / "pilot"

    assert mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--lambda-balance", "1e-3",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--epochs", "1", "--max-batches", "1", "--batch-size", "2",
        "--rate-lr", "0.05",
        "--output-dir", str(output_dir), "--device", "cpu", "--no-plot",
    ]) == 0

    summary = json.loads((output_dir / "lambda_pilot_summary.json").read_text())
    assert summary["rate_lr"] == pytest.approx(0.05)
    for arm in summary["arms"]:
        saved = torch.load(
            Path(arm["checkpoint_dir"]) / "latest.pt", map_location="cpu", weights_only=False,
        )
        groups = saved["optimizer_state_dict"]["param_groups"]
        assert len(groups) == 2
        assert groups[1]["lr"] == pytest.approx(0.05)
        assert saved["extra"]["rate_lr"] == pytest.approx(0.05)


def test_adaptation_check_arg_parser_defaults():
    mod = _load_script("m9c1_adaptation_check")
    args = mod.build_arg_parser(load_default_config()).parse_args([])

    assert args.checkpoint == Path("outputs/qat_combined/checkpoints_qat_noise/best.pt")
    assert args.output_dir == Path("outputs/m9c1_rate_lr_pilot")
    # Compares M9C's shared LR against M9C.1's separate one.
    assert args.rate_lrs == [1e-4, 1e-2]
    assert args.rate_lambda == pytest.approx(mod.M9C_LAMBDA_BALANCE)


def test_adaptation_check_reports_a_missing_checkpoint(tmp_path, capsys):
    mod = _load_script("m9c1_adaptation_check")
    manifest, _, calibration = _tiny_setup(tmp_path)

    exit_code = mod.main([
        "--manifest", str(manifest), "--calibration", str(calibration),
        "--checkpoint", str(tmp_path / "absent.pt"), "--device", "cpu",
    ])

    assert exit_code == 1
    assert "--checkpoint not found" in capsys.readouterr().err


def test_adaptation_check_rejects_a_non_positive_rate_lr(tmp_path):
    mod = _load_script("m9c1_adaptation_check")
    manifest, checkpoint, calibration = _tiny_setup(tmp_path)

    with pytest.raises(SystemExit):
        mod.main([
            "--manifest", str(manifest), "--checkpoint", str(checkpoint),
            "--calibration", str(calibration), "--rate-lrs", "0.01", "0",
            "--device", "cpu",
        ])


def test_adaptation_check_end_to_end_shows_the_lr_ratio(tmp_path):
    """Adam's first step is ~lr per parameter, so a 100x rate LR must produce
    ~100x more estimator movement. That ratio is the whole claim of M9C.1."""
    mod = _load_script("m9c1_adaptation_check")
    manifest, checkpoint, calibration = _tiny_setup(tmp_path)
    output_dir = tmp_path / "adapt"

    exit_code = mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration),
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--epochs", "1", "--max-batches", "2", "--batch-size", "2",
        "--refit-steps", "5", "--rate-lrs", "1e-4", "1e-2",
        "--output-dir", str(output_dir), "--device", "cpu",
    ])

    assert exit_code == 0
    report = json.loads((output_dir / "adaptation_check.json").read_text())
    assert len(report["arms"]) == 2
    assert report["estimator_initialization"]["scale_mean"] == pytest.approx(1.0)
    for arm in report["arms"]:
        assert arm["all_finite"] is True
        assert arm["rate_own_estimator_bpp"] >= 0
        assert arm["rate_matched_refit_bpp"] >= 0
    # 100x the LR, ~100x the movement.
    assert report["verdict"]["loc_movement_ratio"] == pytest.approx(100.0, rel=0.2)
    assert report["verdict"]["scale_movement_ratio"] == pytest.approx(100.0, rel=0.2)
    assert report["verdict"]["all_finite"] is True
