"""Tests for M10F: closing the lower lambda boundary.

M10F reuses M10C's snapshotting, M10A's instrumentation and M10E's BD-rate /
correlation math unchanged (covered by their own test modules), so what is
tested here is what M10F adds:

  * the boundary-closing design - the exact lambda set, and the BRIDGE and
    UPPER_ANCHOR arms that make an inter-experiment comparison possible;
  * a fail-closed preflight extended with the conditions M10E's could not
    check: manifest identity, and that the evaluation procedure still matches
    M10E's (without which the bridge arm compares nothing);
  * a preflight failure now writing its result as an artifact instead of
    leaving no record - the one deliberate behavioural difference from M10E;
  * best-vs-final recorded per run at training time rather than reconstructed;
  * `classify_boundary`, which decides the shape of lambda -> BD-rate from the
    measured points and the measured noise floor rather than from whichever
    number is smallest;
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


# --- The boundary-closing design -------------------------------------------


def test_lambda_set_is_exactly_the_boundary_design():
    mod = _load_script("m10f_lambda_boundary")
    by_name = {entry["name"]: entry for entry in mod.LAMBDAS}

    assert list(by_name) == ["CTRL", "VERY_LOW", "LOW", "BRIDGE", "UPPER_ANCHOR"]
    assert by_name["CTRL"]["lambda"] == 0.0
    assert by_name["VERY_LOW"]["lambda"] == pytest.approx(1.0e-4)
    assert by_name["LOW"]["lambda"] == pytest.approx(2.0e-4)
    assert by_name["BRIDGE"]["lambda"] == pytest.approx(3.0e-4), "M10E's best, repeated"
    assert by_name["UPPER_ANCHOR"]["lambda"] == pytest.approx(4.5e-4), "M10E's candidate"
    assert mod.SEEDS == (42, 43)
    assert {e["lambda"] for e in mod.LAMBDAS} == set(mod.EXPECTED_LAMBDA_SET)


def test_bridge_and_anchor_reproduce_lambdas_m10e_actually_ran():
    """The bridge only works if these two lambdas exist in M10E as well - a
    typo in either would silently turn the comparison into nonsense."""
    mod = _load_script("m10f_lambda_boundary")
    m10e = _load_script("m10e_lambda_lock")

    m10e_lambdas = {entry["lambda"] for entry in m10e.LAMBDAS}
    by_name = {entry["name"]: entry["lambda"] for entry in mod.LAMBDAS}

    assert by_name["BRIDGE"] in m10e_lambdas
    assert by_name["UPPER_ANCHOR"] in m10e_lambdas


def test_grid_is_every_lambda_crossed_with_every_seed():
    mod = _load_script("m10f_lambda_boundary")
    runs = mod.build_runs()

    assert len(runs) == 10, "5 lambdas x 2 seeds"
    assert len({run["dir"] for run in runs}) == 10, "seeds must never collide on disk"
    for entry in mod.LAMBDAS:
        seeds = {r["seed"] for r in runs if r["lambda_name"] == entry["name"]}
        assert seeds == {42, 43}, f"{entry['name']} is missing a seed"
    for run in runs:
        assert str(run["seed"]) in run["dir"]


def test_budget_matches_the_earlier_full_budget_milestones_exactly():
    """A different budget would make M10F incomparable to M10D/M10E, and the
    bridge arm exists precisely to be compared."""
    mod = _load_script("m10f_lambda_boundary")
    m10c = _load_script("m10c_convergence")
    m10e = _load_script("m10e_lambda_lock")

    assert mod.DEFAULT_EPOCHS == m10e.DEFAULT_EPOCHS == 30
    assert mod.SNAPSHOT_EPOCHS == m10c.SNAPSHOT_EPOCHS == m10e.SNAPSHOT_EPOCHS
    assert mod.EXPECTED_START_SHA256 == m10e.EXPECTED_START_SHA256


# --- Fail-closed fairness preflight ----------------------------------------


def test_preflight_passes_on_the_real_grid(tmp_path):
    mod = _load_script("m10f_lambda_boundary")
    _, checkpoint, calibration = _setup(tmp_path)
    args = mod.build_arg_parser(load_default_config()).parse_args(
        ["--calibration", str(calibration), "--checkpoint", str(checkpoint), "--expect-sha256", ""]
    )

    checks = mod.fairness_preflight(args, mod.build_runs(), "deadbeef", tmp_path / "fresh")

    for name, value in checks.items():
        if isinstance(value, bool):
            assert value is True, f"{name} must pass"
    assert checks["lambda_set_is_exactly_the_m10f_design"] is True
    assert checks["design_run_count_is_ten"] is True
    assert checks["control_has_every_seed"] is True
    assert checks["rate_estimator_init_identical_across_seeds"] is True


def test_preflight_verifies_the_evaluation_procedure_still_matches_m10e(tmp_path):
    """The bridge arm compares M10F numbers against M10E numbers, so the two
    must be produced by the same calibration and benchmark settings."""
    mod = _load_script("m10f_lambda_boundary")
    _, checkpoint, calibration = _setup(tmp_path)
    args = mod.build_arg_parser(load_default_config()).parse_args(
        ["--calibration", str(calibration), "--checkpoint", str(checkpoint), "--expect-sha256", ""]
    )

    checks = mod.fairness_preflight(args, mod.build_runs(), "x", tmp_path / "fresh")

    assert checks["m10e_evaluator_readable"] is True
    assert checks["m10f_evaluator_readable"] is True
    assert checks["calibration_and_benchmark_procedure_matches_m10e"] is True
    assert checks["evaluation_constants"]["split"] == "test"
    assert checks["evaluation_constants"]["bit_depths"] == (8, 6, 4)


def test_preflight_records_the_manifest_identity(tmp_path):
    """'Identical dataset split' is only checkable if the manifest itself is
    pinned - the manifest IS the split."""
    mod = _load_script("m10f_lambda_boundary")
    manifest, checkpoint, calibration = _setup(tmp_path)
    args = mod.build_arg_parser(load_default_config()).parse_args(
        ["--calibration", str(calibration), "--checkpoint", str(checkpoint),
         "--manifest", str(manifest), "--expect-sha256", ""]
    )

    checks = mod.fairness_preflight(args, mod.build_runs(), "x", tmp_path / "fresh")

    assert checks["shared_manifest"] == str(manifest)
    assert len(checks["shared_manifest_sha256"]) == 64
    assert checks["identical_split_and_transforms_for_every_run"] is True


def test_preflight_catches_a_wrong_start_checkpoint(tmp_path):
    mod = _load_script("m10f_lambda_boundary")
    _, checkpoint, calibration = _setup(tmp_path)
    args = mod.build_arg_parser(load_default_config()).parse_args(
        ["--calibration", str(calibration), "--checkpoint", str(checkpoint)]
    )

    checks = mod.fairness_preflight(args, mod.build_runs(), "wrong-hash", tmp_path / "fresh")

    assert checks["start_checkpoint_sha256_matches"] is False


def test_preflight_catches_a_missing_seed(tmp_path):
    mod = _load_script("m10f_lambda_boundary")
    _, checkpoint, calibration = _setup(tmp_path)
    args = mod.build_arg_parser(load_default_config()).parse_args(
        ["--calibration", str(calibration), "--checkpoint", str(checkpoint), "--expect-sha256", ""]
    )
    runs = [r for r in mod.build_runs() if not (r["lambda_name"] == "BRIDGE" and r["seed"] == 43)]

    checks = mod.fairness_preflight(args, runs, "x", tmp_path / "fresh")

    assert checks["each_selected_lambda_has_every_seed"] is False


def test_preflight_catches_a_control_missing_a_seed(tmp_path):
    """BD-rate here is paired per seed, so a control with one seed would leave
    half the grid with nothing to be scored against."""
    mod = _load_script("m10f_lambda_boundary")
    _, checkpoint, calibration = _setup(tmp_path)
    args = mod.build_arg_parser(load_default_config()).parse_args(
        ["--calibration", str(calibration), "--checkpoint", str(checkpoint), "--expect-sha256", ""]
    )
    runs = [r for r in mod.build_runs() if not (r["lambda_name"] == "CTRL" and r["seed"] == 43)]

    checks = mod.fairness_preflight(args, runs, "x", tmp_path / "fresh")

    assert checks["control_has_every_seed"] is False


def test_preflight_catches_a_preexisting_checkpoint_directory(tmp_path):
    mod = _load_script("m10f_lambda_boundary")
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


def test_a_failed_preflight_refuses_to_train_but_records_why(tmp_path, capsys):
    """The one deliberate difference from M10E: a failed preflight is itself a
    result, so it is written out instead of leaving no trace."""
    mod = _load_script("m10f_lambda_boundary")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10f"

    exit_code = mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration),  # default (wrong) expected hash
        "--output-dir", str(output_dir), "--device", "cpu",
    ])

    assert exit_code == 1
    assert "fairness preflight failed" in capsys.readouterr().err
    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert summary["status"] == "PREFLIGHT FAILED - no training performed"
    assert "start_checkpoint_sha256_matches" in summary["failed_checks"]
    assert "runs" not in summary, "nothing may be reported as trained"


def test_check_only_stops_before_training(tmp_path, capsys):
    mod = _load_script("m10f_lambda_boundary")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10f"

    exit_code = mod.main(_argv(manifest, checkpoint, calibration, output_dir) + ["--check-only"])

    assert exit_code == 0
    assert "stopping before training" in capsys.readouterr().out
    assert not (output_dir / "training_summary.json").exists()


# --- Training propagates lambda AND seed -----------------------------------


def test_end_to_end_propagates_lambda_and_seed(tmp_path):
    mod = _load_script("m10f_lambda_boundary")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10f"

    exit_code = mod.main(
        _argv(manifest, checkpoint, calibration, output_dir) + ["--only", "CTRL", "BRIDGE"]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert summary["track_scale"] is True
    assert summary["seeds"] == [42, 43]
    assert summary["fairness_preflight"]["only_lambda_and_seed_differ"] is True
    assert len(summary["runs"]) == 4, "2 lambdas x 2 seeds"
    assert "FINAL 18,120-STEP SNAPSHOT" in summary["evaluation_convention"]

    for run in summary["runs"]:
        assert run["all_finite"] is True
        assert run["best_selection_used_only_current_objective"] is True
        assert run["start_checkpoint_sha256"] == summary["start_checkpoint_sha256"]
        assert Path(run["final_snapshot"]).is_file()
        assert len(run["final_snapshot_sha256"]) == 64
        assert run["seed"] in (42, 43)

    bridge = [r for r in summary["runs"] if r["lambda_name"] == "BRIDGE"]
    assert len({r["final_snapshot_sha256"] for r in bridge}) == 2
    assert {r["seed"] for r in bridge} == {42, 43}
    assert (output_dir / "snapshots.csv").is_file()


def test_best_vs_final_gap_is_recorded_for_every_run(tmp_path):
    """M10E's contaminated arm had to be found by hand afterwards; here the gap
    is a first-class field written at training time."""
    mod = _load_script("m10f_lambda_boundary")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10f"

    mod.main(_argv(manifest, checkpoint, calibration, output_dir, epochs=2)
             + ["--only", "CTRL", "--snapshot-epochs", "1", "2"])

    summary = json.loads((output_dir / "training_summary.json").read_text())
    for run in summary["runs"]:
        gap = run["best_vs_final"]
        assert gap["best_epoch"] <= gap["final_epoch"]
        assert gap["gap_absolute"] >= -1e-12, "the best objective cannot exceed the final one"
        assert gap["final_is_best"] == (gap["best_epoch"] == gap["final_epoch"])
        if gap["final_is_best"]:
            assert gap["gap_percent"] == pytest.approx(0.0, abs=1e-9)


def test_control_runs_leave_the_rate_estimator_unfitted(tmp_path):
    mod = _load_script("m10f_lambda_boundary")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10f"

    mod.main(_argv(manifest, checkpoint, calibration, output_dir) + ["--only", "CTRL"])

    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert len(summary["runs"]) == 2
    for run in summary["runs"]:
        assert run["lambda"] == 0.0
        assert run["final_rate_scale_mean"] == pytest.approx(1.0, abs=1e-9)
        assert run["snapshots"][-1]["rate_grad_norm"] == pytest.approx(0.0, abs=1e-12)


# --- The boundary verdict --------------------------------------------------


def _curve(*pairs):
    return [{"lambda_name": f"L{i}", "lambda": lam, "mean_bd_rate_psnr": bd}
            for i, (lam, bd) in enumerate(pairs)]


def test_boundary_case_a_when_the_smallest_lambda_is_still_the_best():
    mod = _load_script("m10f_evaluate")

    verdict = mod.classify_boundary(
        _curve((1.0e-4, -20.0), (2.0e-4, -18.0), (3.0e-4, -16.0), (4.5e-4, -13.0)), floor=1.5)

    assert verdict["case"] == "A_KEEPS_IMPROVING_AS_LAMBDA_DECREASES"
    assert verdict["bracketed"] is False
    assert verdict["best_lambda"] == pytest.approx(1.0e-4)


def test_boundary_interior_minimum_is_bracketed_and_separation_is_judged():
    mod = _load_script("m10f_evaluate")

    resolved = mod.classify_boundary(
        _curve((1.0e-4, -12.0), (2.0e-4, -18.0), (3.0e-4, -14.0), (4.5e-4, -11.0)), floor=1.5)
    assert resolved["case"] == "B_OR_C_INTERIOR_MINIMUM"
    assert resolved["bracketed"] is True
    assert resolved["separated_from_neighbours"] is True
    assert resolved["best_lambda"] == pytest.approx(2.0e-4)

    # Same shape, but the neighbours are inside the floor: bracketed, not resolved.
    unresolved = mod.classify_boundary(
        _curve((1.0e-4, -16.0), (2.0e-4, -17.0), (3.0e-4, -16.2), (4.5e-4, -11.0)), floor=1.5)
    assert unresolved["case"] == "B_OR_C_INTERIOR_MINIMUM"
    assert unresolved["separated_from_neighbours"] is False


def test_boundary_case_d_when_everything_is_inside_the_noise_floor():
    """Flatness must win over 'something is numerically smallest', or the
    analysis would report a minimum that the measurement cannot support."""
    mod = _load_script("m10f_evaluate")

    verdict = mod.classify_boundary(
        _curve((1.0e-4, -16.5), (2.0e-4, -16.9), (3.0e-4, -16.6), (4.5e-4, -16.4)), floor=1.5)

    assert verdict["case"] == "D_FLAT_WITHIN_NOISE"
    assert verdict["total_span_points"] < 1.5


def test_boundary_refuses_to_guess_without_enough_arms():
    mod = _load_script("m10f_evaluate")

    assert mod.classify_boundary(_curve((1.0e-4, -16.0)), floor=1.5)["case"] == "INDETERMINATE"


# --- Evaluation ------------------------------------------------------------


def test_evaluate_benchmarks_the_final_snapshot_not_best(tmp_path):
    """The primary convention is the final 18,120-step snapshot; silently
    switching to best.pt is exactly what the design forbids."""
    evaluate = _load_script("m10f_evaluate")
    summary = {"runs": [
        {"name": "BRIDGE@s42", "lambda_name": "BRIDGE", "lambda": 3.0e-4, "seed": 42,
         "snapshots": [
             {"path": "a/snapshot_step000604.pt", "step": 604, "sha256": "a" * 64},
             {"path": "a/snapshot_step018120.pt", "step": 18120, "sha256": "b" * 64},
         ]},
    ]}

    models = evaluate.final_models(summary)

    assert len(models) == 1
    assert models[0]["step"] == 18120
    assert models[0]["checkpoint"].name == "snapshot_step018120.pt"
    assert models[0]["key"] == "BRIDGE_s42"
    assert models[0]["checkpoint_sha256"] == "b" * 64


def test_evaluate_reuses_the_shared_calibration_benchmark_and_bd_rate_code():
    """M10F numbers are comparable to M9-M10E only because every milestone goes
    through the same calibration, benchmark and BD-rate code."""
    evaluate = _load_script("m10f_evaluate")
    m9 = _load_script("m9_final_calibrate_benchmark")
    m10e = _load_script("m10e_evaluate")

    assert evaluate.BIT_DEPTHS == m9.BIT_DEPTHS == m10e.BIT_DEPTHS == (8, 6, 4)
    assert evaluate.NOISE_FACTOR == m10e.NOISE_FACTOR == 2.0
    assert callable(m9._calibrate) and callable(m9._benchmark)

    source = Path("scripts/m10f_evaluate.py").read_text(encoding="utf-8")
    assert "m10e._bd_rate_linear" in source, "BD-rate math must be reused, not recopied"
    assert "m10e._spearman" in source and "m10e._pearson" in source


def test_evaluate_requires_the_training_summary(tmp_path, capsys):
    evaluate = _load_script("m10f_evaluate")
    manifest, _, _ = _setup(tmp_path)

    exit_code = evaluate.main([
        "--manifest", str(manifest), "--output-dir", str(tmp_path / "absent"),
        "--stage", "calibrate", "--device", "cpu",
    ])

    assert exit_code == 1
    assert "training_summary.json not found" in capsys.readouterr().err


def test_evaluate_refuses_a_preflight_failure_summary(tmp_path, capsys):
    """A summary written by a failed preflight has no runs; analysing it would
    otherwise raise instead of explaining."""
    evaluate = _load_script("m10f_evaluate")
    manifest, _, _ = _setup(tmp_path)
    output_dir = tmp_path / "m10f"
    output_dir.mkdir()
    (output_dir / "training_summary.json").write_text(
        json.dumps({"status": "PREFLIGHT FAILED - no training performed", "failed_checks": ["x"]}),
        encoding="utf-8")

    exit_code = evaluate.main([
        "--manifest", str(manifest), "--output-dir", str(output_dir),
        "--stage", "calibrate", "--device", "cpu",
    ])

    assert exit_code == 1
    assert "no completed runs" in capsys.readouterr().err


def test_calibration_records_per_run_provenance(tmp_path):
    evaluate = _load_script("m10f_evaluate")
    boundary = _load_script("m10f_lambda_boundary")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10f"

    boundary.main(_argv(manifest, checkpoint, calibration, output_dir) + ["--only", "CTRL"])
    exit_code = evaluate.main([
        "--manifest", str(manifest), "--output-dir", str(output_dir),
        "--stage", "calibrate", "--batch-size", "4", "--calibration-batches", "16",
        "--device", "cpu",
    ])

    assert exit_code == 0
    report = json.loads((output_dir / "calibration_report.json").read_text())
    assert report["calibration_split"] == "train"
    assert report["lower_percentile"] == 0.1 and report["upper_percentile"] == 99.9
    assert "nothing reused from M10D or M10E" in report["provenance_note"]
    for row in report["rows"]:
        assert Path(row["calibration"]).is_file()
        assert len(row["checkpoint_sha256"]) == 64
        assert row["seed"] in (42, 43)
        assert "lambda" in row and "training_steps" in row
    by_seed = {}
    for row in report["rows"]:
        by_seed.setdefault(row["seed"], set()).add(row["checkpoint_sha256"])
    assert len(by_seed) == 2
    assert len({next(iter(h)) for h in by_seed.values()}) == 2


# --- The established codec and training path must be untouched -------------


def test_m10f_does_not_modify_the_nvc_format_or_training_path():
    from nvc.compression import nvc_format

    assert hasattr(nvc_format, "MAGIC")
    assert isinstance(nvc_format.MAGIC, (bytes, bytearray, str)) and len(nvc_format.MAGIC) > 0

    trainer = _load_script("train_autoencoder")
    actions = {a.dest for a in trainer.build_arg_parser(load_default_config())._actions}
    for required in ("rate_lambda", "rate_lr", "rate_track_scale",
                     "rate_scale_momentum", "resume_model_only", "seed"):
        assert required in actions, f"{required} must still exist on the training path"


def test_m10f_reuses_m10c_snapshotting_and_m10a_instrumentation():
    """Forking either would let the milestones' training paths drift apart."""
    boundary = _load_script("m10f_lambda_boundary")
    m10c = _load_script("m10c_convergence")

    assert callable(m10c.make_snapshotting_save)
    source = Path("scripts/m10f_lambda_boundary.py").read_text(encoding="utf-8")
    assert "convergence.make_snapshotting_save" in source
    assert "pilot._RecordingRateEstimator" in source
    assert boundary.SNAPSHOT_EPOCHS == m10c.SNAPSHOT_EPOCHS
