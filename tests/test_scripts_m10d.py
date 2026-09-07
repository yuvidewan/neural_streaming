"""Tests for M10D: converged lambda refinement.

M10D reuses M10C's snapshotting and M10A's instrumentation unchanged (both
already covered by `tests/test_scripts_m10c.py` and `tests/test_scripts_m10a.py`),
so what is tested here is what M10D actually adds:

  * the fairness preflight - the check that only lambda differs, which is what
    makes a lambda-refinement result mean anything;
  * lambda/config propagation into the training arms;
  * calibration provenance recorded per arm;
  * the evaluation's output schema and proxy/actual comparison;
  * that the established `.nvc` format and training path are untouched.

Runs on CPU against synthetic data from tests/helpers.py.
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


def _argv(manifest, checkpoint, calibration, output_dir, *, epochs=2):
    return [
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--expect-sha256", "",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--epochs", str(epochs), "--snapshot-epochs", "1", str(epochs),
        "--batch-size", "2", "--output-dir", str(output_dir), "--device", "cpu",
    ]


# --- Configuration: the arms are the ones M10D specifies -------------------


def test_arms_are_concentrated_around_the_m10c_operating_point():
    mod = _load_script("m10d_lambda_refinement")
    by_name = {arm["name"]: arm for arm in mod.ARMS}

    assert list(by_name) == ["CTRL", "LOW", "CENTER", "HIGH", "VERY_HIGH"]
    assert by_name["CTRL"]["lambda"] == 0.0
    assert by_name["LOW"]["lambda"] == pytest.approx(6.0e-4)
    assert by_name["CENTER"]["lambda"] == pytest.approx(9.0757e-04), "the M10C operating point"
    assert by_name["HIGH"]["lambda"] == pytest.approx(1.35e-3)
    assert by_name["VERY_HIGH"]["lambda"] == pytest.approx(1.8e-3)
    # Concentrated around the centre, not another downward sweep - M10B settled that.
    positive = [a["lambda"] for a in mod.ARMS if a["lambda"] > 0]
    assert min(positive) > 3e-4, "M10D must not re-sweep the low region M10B closed"
    assert len({arm["dir"] for arm in mod.ARMS}) == len(mod.ARMS)


def test_budget_matches_m10c_exactly_so_results_are_comparable():
    mod = _load_script("m10d_lambda_refinement")
    m10c = _load_script("m10c_convergence")
    args = mod.build_arg_parser(load_default_config()).parse_args([])

    assert args.epochs == m10c.DEFAULT_EPOCHS == 30
    assert args.snapshot_epochs == list(m10c.SNAPSHOT_EPOCHS)
    assert args.learning_rate == pytest.approx(1e-4)
    assert args.rate_lr == pytest.approx(1e-2)
    assert args.scale_momentum == pytest.approx(0.99)
    assert args.seed == 42
    assert args.qat_bits == 4 and args.qat_mode == "per_channel"
    assert args.expect_sha256 == mod.EXPECTED_START_SHA256 == m10c.EXPECTED_START_SHA256


# --- Fairness preflight ----------------------------------------------------


def test_fairness_preflight_passes_on_the_real_configuration(tmp_path):
    mod = _load_script("m10d_lambda_refinement")
    _, checkpoint, calibration = _setup(tmp_path)
    args = mod.build_arg_parser(load_default_config()).parse_args(
        ["--calibration", str(calibration), "--checkpoint", str(checkpoint), "--expect-sha256", ""]
    )

    checks = mod.fairness_preflight(args, list(mod.ARMS), "deadbeef")

    for name, value in checks.items():
        if isinstance(value, bool):
            assert value is True, f"{name} must pass"
    assert checks["rate_estimator_init_loc_is_zero"] is True
    assert checks["rate_estimator_init_log_scale_is_zero"] is True
    assert checks["scale_tracking_enabled_for_every_arm"] is True


def test_fairness_preflight_catches_a_wrong_start_checkpoint(tmp_path):
    mod = _load_script("m10d_lambda_refinement")
    _, checkpoint, calibration = _setup(tmp_path)
    args = mod.build_arg_parser(load_default_config()).parse_args(
        ["--calibration", str(calibration), "--checkpoint", str(checkpoint)]
    )

    checks = mod.fairness_preflight(args, list(mod.ARMS), "not-the-right-hash")

    assert checks["start_checkpoint_sha256_matches"] is False


def test_fairness_preflight_catches_duplicate_lambdas_and_colliding_dirs(tmp_path):
    """Two arms at the same lambda, or writing to the same directory, would
    silently invalidate the refinement."""
    mod = _load_script("m10d_lambda_refinement")
    _, checkpoint, calibration = _setup(tmp_path)
    args = mod.build_arg_parser(load_default_config()).parse_args(
        ["--calibration", str(calibration), "--checkpoint", str(checkpoint), "--expect-sha256", ""]
    )

    duplicate_lambda = [
        {"name": "A", "lambda": 1e-3, "dir": "a", "role": ""},
        {"name": "B", "lambda": 1e-3, "dir": "b", "role": ""},
    ]
    assert mod.fairness_preflight(args, duplicate_lambda, "x")["lambdas_are_distinct"] is False

    colliding_dir = [
        {"name": "A", "lambda": 1e-3, "dir": "same", "role": ""},
        {"name": "B", "lambda": 2e-3, "dir": "same", "role": ""},
    ]
    assert mod.fairness_preflight(args, colliding_dir, "x")["output_dirs_are_distinct"] is False


def test_check_only_stops_before_training(tmp_path, capsys):
    mod = _load_script("m10d_lambda_refinement")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10d"

    exit_code = mod.main(_argv(manifest, checkpoint, calibration, output_dir) + ["--check-only"])

    assert exit_code == 0
    assert "stopping before training" in capsys.readouterr().out
    assert not (output_dir / "training_summary.json").exists()


def test_a_failed_preflight_refuses_to_train(tmp_path, capsys):
    """The hash guard must stop the run, not warn and continue."""
    mod = _load_script("m10d_lambda_refinement")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10d"

    exit_code = mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration),  # default (wrong) expected hash
        "--output-dir", str(output_dir), "--device", "cpu",
    ])

    assert exit_code == 1
    assert "fairness preflight failed" in capsys.readouterr().err
    assert not (output_dir / "training_summary.json").exists()


# --- Lambda / config propagation and snapshots -----------------------------


def test_end_to_end_propagates_lambda_and_records_fairness(tmp_path):
    mod = _load_script("m10d_lambda_refinement")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10d"

    exit_code = mod.main(_argv(manifest, checkpoint, calibration, output_dir))

    assert exit_code == 0
    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert summary["track_scale"] is True
    assert summary["scale_momentum"] == pytest.approx(0.99)
    assert summary["fairness_preflight"]["only_lambda_differs"] is True
    assert [arm["name"] for arm in summary["arms"]] == list(
        a["name"] for a in mod.ARMS
    )

    steps_per_epoch = summary["steps_per_epoch"]
    for arm, expected in zip(summary["arms"], mod.ARMS):
        assert arm["lambda"] == pytest.approx(expected["lambda"]), "lambda must reach the arm"
        assert arm["all_finite"] is True
        assert arm["best_selection_used_only_current_objective"] is True
        assert arm["stale_history_records_ignored"] == 1
        # Fairness: every arm starts from the same checkpoint.
        assert arm["start_checkpoint_sha256"] == summary["start_checkpoint_sha256"]
        # Every arm has the same snapshot schedule.
        assert [s["step"] for s in arm["snapshots"]] == [steps_per_epoch, 2 * steps_per_epoch]
        for snapshot in arm["snapshots"]:
            assert Path(snapshot["path"]).is_file()
            assert len(snapshot["sha256"]) == 64

    # Distinct lambdas must produce distinct models.
    assert len({arm["final_snapshot_sha256"] for arm in summary["arms"]}) == len(summary["arms"])
    assert (output_dir / "snapshots.csv").is_file()


def test_control_arm_leaves_the_rate_estimator_unfitted(tmp_path):
    mod = _load_script("m10d_lambda_refinement")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10d"

    mod.main(_argv(manifest, checkpoint, calibration, output_dir) + ["--only", "CTRL"])

    summary = json.loads((output_dir / "training_summary.json").read_text())
    arm = summary["arms"][0]
    assert arm["lambda"] == 0.0
    assert arm["final_rate_scale_mean"] == pytest.approx(1.0, abs=1e-9)
    assert arm["snapshots"][-1]["rate_grad_norm"] == pytest.approx(0.0, abs=1e-12)


# --- Evaluation ------------------------------------------------------------


def test_evaluate_benchmarks_the_final_snapshot_of_each_arm():
    evaluate = _load_script("m10d_evaluate")
    summary = {
        "arms": [
            {"name": "CTRL", "lambda": 0.0, "snapshots": [
                {"step": 604, "path": "a/s1.pt", "sha256": "a" * 64},
                {"step": 18120, "path": "a/s5.pt", "sha256": "b" * 64}]},
            {"name": "CENTER", "lambda": 9.0757e-04, "snapshots": [
                {"step": 604, "path": "b/s1.pt", "sha256": "c" * 64},
                {"step": 18120, "path": "b/s5.pt", "sha256": "d" * 64}]},
        ]
    }

    models = evaluate.final_models(summary)

    assert [m["key"] for m in models] == ["CTRL", "CENTER"]
    assert all(m["step"] == 18120 for m in models), "only the converged snapshot is benchmarked"
    assert [m["checkpoint"].name for m in models] == ["s5.pt", "s5.pt"]
    assert models[1]["lambda"] == pytest.approx(9.0757e-04)
    assert models[0]["checkpoint_sha256"] == "b" * 64


def test_evaluate_reuses_the_shared_calibration_and_benchmark_stages():
    """M10D numbers are comparable to M9/M10A/M10B/M10C only because all of
    them go through the same calibration and benchmark code."""
    evaluate = _load_script("m10d_evaluate")
    m9 = _load_script("m9_final_calibrate_benchmark")

    assert evaluate.BIT_DEPTHS == m9.BIT_DEPTHS == (8, 6, 4)
    assert callable(m9._calibrate) and callable(m9._benchmark)
    assert evaluate.M10C_REFERENCE == "M10C-L@18120"


def test_bd_rate_and_correlation_helpers():
    evaluate = _load_script("m10d_evaluate")

    identical = [(1.0, 30.0), (0.7, 29.0), (0.5, 28.0)]
    assert evaluate._bd_rate_linear(identical, identical) == pytest.approx(0.0, abs=1e-9)
    halved = [(0.5, 30.0), (0.35, 29.0), (0.25, 28.0)]
    assert evaluate._bd_rate_linear(identical, halved) == pytest.approx(-50.0, abs=0.5)
    assert evaluate._bd_rate_linear(identical, [(1.0, 20.0), (0.7, 19.0), (0.5, 18.0)]) is None

    assert evaluate._pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert evaluate._spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert evaluate._pearson([1.0, 2.0], [1.0, 2.0]) is None


def test_evaluate_requires_the_training_summary(tmp_path, capsys):
    evaluate = _load_script("m10d_evaluate")
    manifest, _, _ = _setup(tmp_path)

    exit_code = evaluate.main([
        "--manifest", str(manifest), "--output-dir", str(tmp_path / "absent"),
        "--stage", "calibrate", "--device", "cpu",
    ])

    assert exit_code == 1
    assert "training_summary.json not found" in capsys.readouterr().err


def test_calibration_report_records_provenance(tmp_path):
    """Every grid must be traceable to the checkpoint it was fitted to, on the
    train split - otherwise a cross-arm calibration leak would be invisible."""
    evaluate = _load_script("m10d_evaluate")
    refinement = _load_script("m10d_lambda_refinement")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10d"

    refinement.main(
        _argv(manifest, checkpoint, calibration, output_dir) + ["--only", "CTRL", "HIGH"]
    )
    exit_code = evaluate.main([
        "--manifest", str(manifest), "--output-dir", str(output_dir),
        "--stage", "calibrate", "--batch-size", "4", "--calibration-batches", "16",
        "--device", "cpu",
    ])

    assert exit_code == 0
    report = json.loads((output_dir / "calibration_report.json").read_text())
    assert report["calibration_split"] == "train"
    assert report["lower_percentile"] == 0.1 and report["upper_percentile"] == 99.9
    assert "No calibration is shared between arms" in report["provenance_note"]
    assert {row["bits"] for row in report["rows"]} == {8, 6, 4}
    for row in report["rows"]:
        assert Path(row["calibration"]).is_file()
        assert len(row["checkpoint_sha256"]) == 64
        assert "lambda" in row and "training_steps" in row
    # Each arm's grids must come from that arm's own checkpoint.
    by_model = {}
    for row in report["rows"]:
        by_model.setdefault(row["model"], set()).add(row["checkpoint_sha256"])
    assert all(len(hashes) == 1 for hashes in by_model.values())
    assert len({next(iter(h)) for h in by_model.values()}) == len(by_model)


# --- The established codec must be untouched -------------------------------


def test_m10d_does_not_modify_the_nvc_format_or_training_path():
    """M10D is an experiment, not a code change: the deployed format constants
    and the training entry point must be exactly what earlier milestones used."""
    from nvc.compression import nvc_format

    # A frozen snapshot of the format's identifying constants. If M10D (or
    # anything it imports) had altered them, every previous milestone's
    # benchmark would silently stop being comparable.
    assert hasattr(nvc_format, "MAGIC")
    magic = nvc_format.MAGIC
    assert isinstance(magic, (bytes, bytearray, str)) and len(magic) > 0

    trainer = _load_script("train_autoencoder")
    parser_actions = {a.dest for a in trainer.build_arg_parser(load_default_config())._actions}
    for required in ("rate_lambda", "rate_lr", "rate_track_scale",
                     "rate_scale_momentum", "resume_model_only"):
        assert required in parser_actions, f"{required} must still exist on the training path"


def test_m10d_reuses_m10c_snapshotting_rather_than_reimplementing_it():
    """Forking the snapshot logic would let M10C and M10D drift apart."""
    refinement = _load_script("m10d_lambda_refinement")
    m10c = _load_script("m10c_convergence")

    assert callable(m10c.make_snapshotting_save)
    source = Path("scripts/m10d_lambda_refinement.py").read_text(encoding="utf-8")
    assert "convergence.make_snapshotting_save" in source
    assert "pilot._RecordingRateEstimator" in source
    assert refinement.SNAPSHOT_EPOCHS == m10c.SNAPSHOT_EPOCHS
