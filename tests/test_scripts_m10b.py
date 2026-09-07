"""Tests for M10B: the parameterised low-lambda sweep and its evaluation.

M10B introduces exactly two new behaviours, and only these are tested here -
everything else (scale tracking, the recording estimator, best-checkpoint
selection, calibration, benchmarking) is unchanged code already covered by
`tests/test_scripts_m10a.py`, `tests/test_rate_estimator.py`,
`tests/test_m9_checkpoint_selection.py` and `tests/test_scripts_m9_final_eval.py`:

  1. `arms_for_lambdas()` - running the M10A harness over an arbitrary lambda
     sweep, WITHOUT changing what a default M10A invocation does;
  2. `m10b_evaluate` - joining three freshly benchmarked arms with the two
     retained from M10A, and the Pareto/BD-rate analysis over the union.

Everything runs on CPU against synthetic data from tests/helpers.py.
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


# --- 1. The parameterised sweep --------------------------------------------


def test_m10a_default_arms_are_unchanged_by_the_parameterisation():
    """The load-bearing backward-compatibility check: adding --lambdas must not
    alter what a plain M10A run does, or M10A's recorded results stop being
    reproducible from this harness."""
    mod = _load_script("m10a_pilot")
    args = mod.build_arg_parser(load_default_config()).parse_args([])

    assert args.lambdas is None, "M10A's own four arms must remain the default"
    assert args.no_control is False
    assert [arm["name"] for arm in mod.ARMS] == ["CTRL", "M10A-L", "M10A-M", "M10A-H"]
    assert [arm["lambda"] for arm in mod.ARMS] == pytest.approx(
        [0.0, 9.0757e-04, 2.8700e-03, 9.0757e-03]
    )


def test_arms_for_lambdas_builds_the_m10b_sweep():
    mod = _load_script("m10a_pilot")
    arms = mod.arms_for_lambdas([3e-4, 1e-4, 3e-5], prefix="M10B")

    assert [arm["name"] for arm in arms] == ["M10B-1", "M10B-2", "M10B-3"]
    assert [arm["lambda"] for arm in arms] == pytest.approx([3e-4, 1e-4, 3e-5])
    # Directory names carry the lambda, not just an index, so an arm's output
    # directory always identifies which lambda produced it.
    assert [arm["dir"] for arm in arms] == ["lambda_3e-04", "lambda_1e-04", "lambda_3e-05"]
    assert len({arm["dir"] for arm in arms}) == 3


def test_arms_for_lambdas_rejects_duplicates_that_would_collide_on_disk():
    mod = _load_script("m10a_pilot")

    with pytest.raises(ValueError, match="collide"):
        mod.arms_for_lambdas([1e-4, 1e-4], prefix="M10B")
    with pytest.raises(ValueError):
        mod.arms_for_lambdas([], prefix="M10B")


def test_cli_rejects_a_non_positive_lambda_in_the_sweep(tmp_path):
    mod = _load_script("m10a_pilot")
    manifest, checkpoint, calibration = _setup(tmp_path)

    with pytest.raises(SystemExit):
        mod.main([
            "--manifest", str(manifest), "--checkpoint", str(checkpoint),
            "--calibration", str(calibration), "--expect-sha256", "",
            "--lambdas", "1e-4", "0", "--device", "cpu",
        ])


def test_sweep_runs_the_control_unless_suppressed(tmp_path):
    """M10B reuses M10A's control rather than retraining it, so --no-control
    must actually omit it - and its absence must not disturb the rest."""
    mod = _load_script("m10a_pilot")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "sweep"

    exit_code = mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--expect-sha256", "",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--epochs", "1", "--max-batches", "2", "--batch-size", "2",
        "--lambdas", "3e-4", "1e-4", "--no-control",
        "--output-dir", str(output_dir), "--device", "cpu",
    ])

    assert exit_code == 0
    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert [arm["name"] for arm in summary["arms"]] == ["M10B-1", "M10B-2"]
    assert all(arm["lambda"] > 0 for arm in summary["arms"])
    for arm in summary["arms"]:
        assert arm["all_finite"] is True
        assert Path(arm["best_checkpoint"]).is_file()
        assert arm["best_selection_used_only_current_objective"] is True
        # Scale tracking stays on for every M10B arm.
        assert summary["track_scale"] is True
        steps = json.loads((Path(arm["checkpoint_dir"]) / "step_log.json").read_text())
        assert len(steps) == arm["steps"]
        # Validation batches must never enter the training step log.
        assert all("bin_width_after" in record for record in steps)


def test_sweep_arms_write_to_distinct_directories(tmp_path):
    mod = _load_script("m10a_pilot")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "sweep"

    mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--expect-sha256", "",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--epochs", "1", "--max-batches", "2", "--batch-size", "2",
        "--lambdas", "3e-4", "1e-4", "3e-5", "--no-control",
        "--output-dir", str(output_dir), "--device", "cpu",
    ])

    summary = json.loads((output_dir / "training_summary.json").read_text())
    directories = {arm["checkpoint_dir"] for arm in summary["arms"]}
    hashes = {arm["best_checkpoint_sha256"] for arm in summary["arms"]}
    assert len(directories) == 3, "no arm may overwrite another"
    assert len(hashes) == 3, "distinct lambdas must produce distinct models"


# --- 2. The joined evaluation ----------------------------------------------


def test_evaluate_only_calibrates_the_new_arms():
    """The retained control and M10A-L must NOT be recalibrated - re-running
    them would add nondeterminism between two copies of the same experiment."""
    evaluate = _load_script("m10b_evaluate")
    models = evaluate._model_set(Path("outputs/m10b_pilot"))

    assert [m["key"] for m in models] == ["M10B-1", "M10B-2", "M10B-3"]
    assert evaluate.RETAINED == {"CTRL": "M10A-CTRL", "M10A-L": "M10A-L"}
    assert evaluate.REPORT_ORDER == ("CTRL", "M10A-L", "M10B-1", "M10B-2", "M10B-3")
    for model in models:
        assert model["checkpoint"].parts[:2] == ("outputs", "m10b_pilot")
        assert model["checkpoint"].name == "best.pt"


def test_evaluate_reuses_the_shared_calibration_and_benchmark_stages():
    evaluate = _load_script("m10b_evaluate")
    m9 = _load_script("m9_final_calibrate_benchmark")

    assert evaluate.BIT_DEPTHS == m9.BIT_DEPTHS
    assert callable(m9._calibrate) and callable(m9._benchmark)


def test_bd_rate_is_piecewise_linear_and_undefined_without_overlap():
    """M10A settled on the conservative piecewise-linear methodology; a curve
    with no PSNR overlap must report undefined rather than a misleading zero."""
    evaluate = _load_script("m10b_evaluate")

    identical = [(1.0, 30.0), (0.7, 29.0), (0.5, 28.0)]
    assert evaluate._bd_rate_linear(identical, identical) == pytest.approx(0.0, abs=1e-9)

    # Same quality for uniformly half the bits = -50% BD-rate.
    cheaper = [(0.5, 30.0), (0.35, 29.0), (0.25, 28.0)]
    assert evaluate._bd_rate_linear(identical, cheaper) == pytest.approx(-50.0, abs=0.5)

    disjoint = [(1.0, 20.0), (0.7, 19.0), (0.5, 18.0)]
    assert evaluate._bd_rate_linear(identical, disjoint) is None


def test_correlation_helpers_refuse_too_few_points():
    evaluate = _load_script("m10b_evaluate")

    assert evaluate._pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert evaluate._spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert evaluate._pearson([1.0, 2.0], [1.0, 2.0]) is None
    assert evaluate._spearman([1.0, 2.0], [1.0, 2.0]) is None


def test_evaluate_reports_a_missing_manifest(tmp_path, capsys):
    evaluate = _load_script("m10b_evaluate")

    exit_code = evaluate.main([
        "--manifest", str(tmp_path / "absent.json"), "--stage", "calibrate", "--device", "cpu",
    ])

    assert exit_code == 1
    assert "--manifest not found" in capsys.readouterr().err


def test_evaluate_analysis_requires_the_m10a_results_it_joins_against(tmp_path, capsys):
    evaluate = _load_script("m10b_evaluate")
    manifest, _, _ = _setup(tmp_path)
    output_dir = tmp_path / "m10b"
    output_dir.mkdir()
    (output_dir / "benchmark_aggregate.json").write_text('{"rows": []}', encoding="utf-8")

    exit_code = evaluate.main([
        "--manifest", str(manifest), "--output-dir", str(output_dir),
        "--m10a-dir", str(tmp_path / "absent"), "--stage", "analyse", "--device", "cpu",
    ])

    assert exit_code == 1
    assert "M10A benchmark_aggregate.json not found" in capsys.readouterr().err
