"""Tests for M10E: final converged lambda selection with two seeds per lambda.

M10E reuses M10C's snapshotting and M10A's instrumentation unchanged (covered
by `tests/test_scripts_m10c.py` / `test_scripts_m10a.py`), so what is tested
here is what M10E adds:

  * the (lambda x seed) grid, and that the two seeds of one lambda can never
    collide on disk;
  * the fail-closed fairness preflight, including the seed-specific conditions
    M10D's preflight could not check;
  * paired-per-seed BD-rate and the noise floor derived from the two controls;
  * the deliberate separation of "does the proxy order bitrate" from "does the
    proxy pick the best RD lambda";
  * that the deployed `.nvc` format and training path stay untouched.

Runs on CPU against synthetic data from tests/helpers.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

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


def _argv(manifest, checkpoint, calibration, output_dir, *, epochs=1):
    return [
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--expect-sha256", "",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--epochs", str(epochs), "--snapshot-epochs", str(epochs),
        "--batch-size", "2", "--output-dir", str(output_dir), "--device", "cpu",
    ]


# --- The (lambda x seed) grid ----------------------------------------------


def test_lambda_set_targets_the_region_below_m10ds_best():
    mod = _load_script("m10e_lambda_lock")
    by_name = {entry["name"]: entry for entry in mod.LAMBDAS}

    assert list(by_name) == ["CTRL", "LOWER", "MID_LOW", "CURRENT_BEST", "UPPER_REFERENCE"]
    assert by_name["CTRL"]["lambda"] == 0.0
    assert by_name["LOWER"]["lambda"] == pytest.approx(3.0e-4)
    assert by_name["MID_LOW"]["lambda"] == pytest.approx(4.5e-4)
    assert by_name["CURRENT_BEST"]["lambda"] == pytest.approx(6.0e-4), "M10D's winner"
    assert by_name["UPPER_REFERENCE"]["lambda"] == pytest.approx(7.5e-4)
    assert mod.SEEDS == (42, 43)


def test_grid_is_every_lambda_crossed_with_every_seed():
    mod = _load_script("m10e_lambda_lock")
    runs = mod.build_runs()

    assert len(runs) == 10, "5 lambdas x 2 seeds"
    assert len({r["dir"] for r in runs}) == 10, "no two runs may share a directory"
    for value in {r["lambda"] for r in runs}:
        assert {r["seed"] for r in runs if r["lambda"] == value} == {42, 43}
    # A directory name must identify BOTH the lambda and the seed - the two
    # seeds of one lambda must not be able to overwrite each other.
    for run in runs:
        assert f"seed{run['seed']}" in run["dir"]
        assert run["dir"].startswith("lambda_")


def test_budget_matches_m10c_and_m10d_exactly():
    mod = _load_script("m10e_lambda_lock")
    m10d = _load_script("m10d_lambda_refinement")
    args = mod.build_arg_parser(load_default_config()).parse_args([])

    assert args.epochs == m10d.DEFAULT_EPOCHS == 30
    assert args.snapshot_epochs == list(m10d.SNAPSHOT_EPOCHS)
    assert args.learning_rate == pytest.approx(1e-4)
    assert args.rate_lr == pytest.approx(1e-2)
    assert args.scale_momentum == pytest.approx(0.99)
    assert args.qat_bits == 4 and args.qat_mode == "per_channel"
    assert args.expect_sha256 == mod.EXPECTED_START_SHA256 == m10d.EXPECTED_START_SHA256


# --- Fairness preflight, fail-closed ---------------------------------------


def test_preflight_passes_on_the_real_grid(tmp_path):
    mod = _load_script("m10e_lambda_lock")
    _, checkpoint, calibration = _setup(tmp_path)
    args = mod.build_arg_parser(load_default_config()).parse_args(
        ["--calibration", str(calibration), "--checkpoint", str(checkpoint), "--expect-sha256", ""]
    )

    checks = mod.fairness_preflight(args, mod.build_runs(), "deadbeef", tmp_path / "fresh")

    for name, value in checks.items():
        if isinstance(value, bool):
            assert value is True, f"{name} must pass"
    assert checks["exactly_two_seeds"] is True
    assert checks["each_selected_lambda_has_every_seed"] is True
    assert checks["five_lambda_values_in_design"] is True
    assert checks["full_grid_selected"] is True
    assert checks["rate_estimator_init_identical_across_seeds"] is True


def test_preflight_catches_a_wrong_start_checkpoint(tmp_path):
    mod = _load_script("m10e_lambda_lock")
    _, checkpoint, calibration = _setup(tmp_path)
    args = mod.build_arg_parser(load_default_config()).parse_args(
        ["--calibration", str(calibration), "--checkpoint", str(checkpoint)]
    )

    checks = mod.fairness_preflight(args, mod.build_runs(), "wrong-hash", tmp_path / "fresh")

    assert checks["start_checkpoint_sha256_matches"] is False


def test_preflight_catches_a_missing_seed(tmp_path):
    """A lambda with only one seed would silently lose the replication this
    whole experiment is built on."""
    mod = _load_script("m10e_lambda_lock")
    _, checkpoint, calibration = _setup(tmp_path)
    args = mod.build_arg_parser(load_default_config()).parse_args(
        ["--calibration", str(calibration), "--checkpoint", str(checkpoint), "--expect-sha256", ""]
    )
    runs = [r for r in mod.build_runs() if not (r["lambda_name"] == "LOWER" and r["seed"] == 43)]

    checks = mod.fairness_preflight(args, runs, "x", tmp_path / "fresh")

    assert checks["each_selected_lambda_has_every_seed"] is False


def test_preflight_catches_a_preexisting_checkpoint_directory(tmp_path):
    """An existing latest.pt would let epoch numbering or best.pt tracking
    carry over into a run that is supposed to start clean."""
    mod = _load_script("m10e_lambda_lock")
    _, checkpoint, calibration = _setup(tmp_path)
    args = mod.build_arg_parser(load_default_config()).parse_args(
        ["--calibration", str(calibration), "--checkpoint", str(checkpoint), "--expect-sha256", ""]
    )
    runs = mod.build_runs()
    output_dir = tmp_path / "dirty"
    (output_dir / runs[0]["dir"]).mkdir(parents=True)
    (output_dir / runs[0]["dir"] / "latest.pt").write_bytes(b"stale")

    checks = mod.fairness_preflight(args, runs, "x", output_dir)

    assert checks["no_preexisting_run_checkpoints"] is False
    assert checks["preexisting_dirs"] == [runs[0]["dir"]]


def test_a_failed_preflight_refuses_to_train(tmp_path, capsys):
    mod = _load_script("m10e_lambda_lock")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10e"

    exit_code = mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration),  # default (wrong) expected hash
        "--output-dir", str(output_dir), "--device", "cpu",
    ])

    assert exit_code == 1
    assert "fairness preflight failed" in capsys.readouterr().err
    assert not (output_dir / "training_summary.json").exists()


def test_check_only_stops_before_training(tmp_path, capsys):
    mod = _load_script("m10e_lambda_lock")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10e"

    exit_code = mod.main(_argv(manifest, checkpoint, calibration, output_dir) + ["--check-only"])

    assert exit_code == 0
    assert "stopping before training" in capsys.readouterr().out
    assert not (output_dir / "training_summary.json").exists()


# --- Training propagates lambda AND seed -----------------------------------


def test_end_to_end_propagates_lambda_and_seed(tmp_path):
    mod = _load_script("m10e_lambda_lock")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10e"

    exit_code = mod.main(
        _argv(manifest, checkpoint, calibration, output_dir) + ["--only", "CTRL", "CURRENT_BEST"]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert summary["track_scale"] is True
    assert summary["seeds"] == [42, 43]
    assert summary["fairness_preflight"]["only_lambda_and_seed_differ"] is True
    assert len(summary["runs"]) == 4, "2 lambdas x 2 seeds"

    for run in summary["runs"]:
        assert run["all_finite"] is True
        assert run["best_selection_used_only_current_objective"] is True
        assert run["stale_history_records_ignored"] == 1
        assert run["start_checkpoint_sha256"] == summary["start_checkpoint_sha256"]
        assert Path(run["final_snapshot"]).is_file()
        assert len(run["final_snapshot_sha256"]) == 64
        assert run["seed"] in (42, 43)

    # Different seeds at the same lambda must produce different models -
    # otherwise the seed is not actually varying anything.
    best = [r for r in summary["runs"] if r["lambda_name"] == "CURRENT_BEST"]
    assert len({r["final_snapshot_sha256"] for r in best}) == 2
    assert {r["seed"] for r in best} == {42, 43}
    assert (output_dir / "snapshots.csv").is_file()


def test_control_runs_leave_the_rate_estimator_unfitted(tmp_path):
    mod = _load_script("m10e_lambda_lock")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10e"

    mod.main(_argv(manifest, checkpoint, calibration, output_dir) + ["--only", "CTRL"])

    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert len(summary["runs"]) == 2
    for run in summary["runs"]:
        assert run["lambda"] == 0.0
        assert run["final_rate_scale_mean"] == pytest.approx(1.0, abs=1e-9)
        assert run["snapshots"][-1]["rate_grad_norm"] == pytest.approx(0.0, abs=1e-12)


# --- Evaluation ------------------------------------------------------------


def test_evaluate_benchmarks_the_final_snapshot_of_every_run():
    evaluate = _load_script("m10e_evaluate")
    summary = {"runs": [
        {"name": "CTRL@s42", "lambda_name": "CTRL", "lambda": 0.0, "seed": 42,
         "snapshots": [{"step": 604, "path": "a/s1.pt", "sha256": "a" * 64},
                       {"step": 18120, "path": "a/s5.pt", "sha256": "b" * 64}]},
        {"name": "CTRL@s43", "lambda_name": "CTRL", "lambda": 0.0, "seed": 43,
         "snapshots": [{"step": 18120, "path": "b/s5.pt", "sha256": "c" * 64}]},
    ]}

    models = evaluate.final_models(summary)

    assert [m["key"] for m in models] == ["CTRL_s42", "CTRL_s43"]
    assert all(m["step"] == 18120 for m in models)
    assert models[0]["seed"] == 42 and models[1]["seed"] == 43
    assert len({m["key"] for m in models}) == 2


def test_evaluate_reuses_the_shared_calibration_and_benchmark_stages():
    """M10E numbers are comparable to M9-M10D only because every milestone
    goes through the same calibration and benchmark code."""
    evaluate = _load_script("m10e_evaluate")
    m9 = _load_script("m9_final_calibrate_benchmark")

    assert evaluate.BIT_DEPTHS == m9.BIT_DEPTHS == (8, 6, 4)
    assert callable(m9._calibrate) and callable(m9._benchmark)
    assert evaluate.NOISE_FACTOR == 2.0, "the stated above-noise convention"


def test_bd_rate_and_correlation_helpers():
    evaluate = _load_script("m10e_evaluate")

    identical = [(1.0, 30.0), (0.7, 29.0), (0.5, 28.0)]
    assert evaluate._bd_rate_linear(identical, identical) == pytest.approx(0.0, abs=1e-9)
    halved = [(0.5, 30.0), (0.35, 29.0), (0.25, 28.0)]
    assert evaluate._bd_rate_linear(identical, halved) == pytest.approx(-50.0, abs=0.5)
    assert evaluate._bd_rate_linear(identical, [(1.0, 20.0), (0.7, 19.0), (0.5, 18.0)]) is None

    assert evaluate._pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert evaluate._spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert evaluate._pearson([1.0, 2.0], [1.0, 2.0]) is None, "n<3 must refuse"


def test_evaluate_requires_the_training_summary(tmp_path, capsys):
    evaluate = _load_script("m10e_evaluate")
    manifest, _, _ = _setup(tmp_path)

    exit_code = evaluate.main([
        "--manifest", str(manifest), "--output-dir", str(tmp_path / "absent"),
        "--stage", "calibrate", "--device", "cpu",
    ])

    assert exit_code == 1
    assert "training_summary.json not found" in capsys.readouterr().err


def test_calibration_records_per_run_provenance(tmp_path):
    """Each grid must be traceable to its own checkpoint, lambda and seed, on
    the train split - a cross-run calibration leak would otherwise be invisible."""
    evaluate = _load_script("m10e_evaluate")
    lock = _load_script("m10e_lambda_lock")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10e"

    lock.main(_argv(manifest, checkpoint, calibration, output_dir) + ["--only", "CTRL"])
    exit_code = evaluate.main([
        "--manifest", str(manifest), "--output-dir", str(output_dir),
        "--stage", "calibrate", "--batch-size", "4", "--calibration-batches", "16",
        "--device", "cpu",
    ])

    assert exit_code == 0
    report = json.loads((output_dir / "calibration_report.json").read_text())
    assert report["calibration_split"] == "train"
    assert report["lower_percentile"] == 0.1 and report["upper_percentile"] == 99.9
    assert "no sharing between runs" in report["provenance_note"]
    for row in report["rows"]:
        assert Path(row["calibration"]).is_file()
        assert len(row["checkpoint_sha256"]) == 64
        assert row["seed"] in (42, 43)
        assert "lambda" in row and "training_steps" in row
    # The two seeds of one lambda must have been calibrated from DIFFERENT
    # checkpoints, not one shared grid.
    by_seed = {}
    for row in report["rows"]:
        by_seed.setdefault(row["seed"], set()).add(row["checkpoint_sha256"])
    assert len(by_seed) == 2
    assert len({next(iter(h)) for h in by_seed.values()}) == 2


# --- The established codec and training path must be untouched -------------


def test_m10e_does_not_modify_the_nvc_format_or_training_path():
    from nvc.compression import nvc_format

    assert hasattr(nvc_format, "MAGIC")
    assert isinstance(nvc_format.MAGIC, (bytes, bytearray, str)) and len(nvc_format.MAGIC) > 0

    trainer = _load_script("train_autoencoder")
    actions = {a.dest for a in trainer.build_arg_parser(load_default_config())._actions}
    for required in ("rate_lambda", "rate_lr", "rate_track_scale",
                     "rate_scale_momentum", "resume_model_only", "seed"):
        assert required in actions, f"{required} must still exist on the training path"


def test_m10e_reuses_m10c_snapshotting_and_m10a_instrumentation():
    """Forking either would let the milestones' training paths drift apart."""
    lock = _load_script("m10e_lambda_lock")
    m10c = _load_script("m10c_convergence")

    assert callable(m10c.make_snapshotting_save)
    source = Path("scripts/m10e_lambda_lock.py").read_text(encoding="utf-8")
    assert "convergence.make_snapshotting_save" in source
    assert "pilot._RecordingRateEstimator" in source
    assert lock.SNAPSHOT_EPOCHS == m10c.SNAPSHOT_EPOCHS
