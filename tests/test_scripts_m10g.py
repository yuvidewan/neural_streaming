"""Tests for M10G: the checkpoint evaluation convention and the temporal scripts.

Part A's whole value is that checkpoint selection is deterministic, declared in
advance, and STRUCTURALLY unable to see the test split. The first of those is
easy to get right by accident; the last is the one that actually protects the
benchmark, so it is tested as a hard failure rather than a convention.

Part B must be non-destructive: it re-reads historical M10E/M10F artifacts and
must never rewrite them.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from nvc.utils.config import load_default_config


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _history(values, *, rate_enabled=True):
    """A minimal validation history: (epoch, val_loss, val_psnr) triples."""
    return [{"epoch": epoch, "val_loss": loss, "val_psnr": psnr,
             "val_distortion": loss, "val_rate_bpp": 0.5, "rate_enabled": rate_enabled}
            for epoch, loss, psnr in values]


# --- Part A: the selection rule ---------------------------------------------


def test_selection_picks_the_best_validation_epoch_not_the_last():
    """The entire point: M10F deployed a degraded final epoch because the
    convention said 'final', not 'best'."""
    mod = _load_script("m10g_evaluation_convention")
    history = _history([(1, 5.0e-4, 30.0), (2, 4.0e-4, 31.0), (3, 4.6e-4, 30.2)])

    selection = mod.select_checkpoint(history)

    assert selection["selected_epoch"] == 2
    assert selection["selected_objective"] == pytest.approx(4.0e-4)
    assert selection["final_epoch"] == 3
    assert selection["final_objective"] == pytest.approx(4.6e-4)
    assert selection["final_is_best"] is False
    assert selection["best_vs_final_percent"] == pytest.approx(15.0)


def test_selection_is_deterministic_and_breaks_ties_by_earliest_epoch():
    mod = _load_script("m10g_evaluation_convention")
    history = _history([(1, 4.0e-4, 30.0), (2, 4.0e-4, 31.0), (3, 4.0e-4, 29.0)])

    first = mod.select_checkpoint(history)
    second = mod.select_checkpoint(list(reversed(history)))

    # Three epochs tie exactly; the earliest must win, and must keep winning
    # regardless of the order the records happen to arrive in.
    assert first["selected_epoch"] == 1
    assert second["selected_epoch"] == 1
    assert first["tie_break"] == "earliest epoch"


def test_final_checkpoint_is_retained_alongside_the_selected_one():
    """The final checkpoint is SECONDARY, not discarded - it is the convergence
    diagnostic that would have revealed M10F's problem sooner."""
    mod = _load_script("m10g_evaluation_convention")
    history = _history([(1, 5.0e-4, 30.0), (2, 4.0e-4, 31.0), (3, 4.6e-4, 30.2)])

    selection = mod.select_checkpoint(history)

    for key in ("final_epoch", "final_objective", "final_psnr_db",
                "best_vs_final_absolute", "best_vs_final_percent", "psnr_difference_db"):
        assert key in selection, f"{key} must be recorded for the secondary checkpoint"
    assert selection["epoch_difference"] == 1


def test_selection_records_its_own_metric_and_domain():
    mod = _load_script("m10g_evaluation_convention")

    selection = mod.select_checkpoint(_history([(1, 5.0e-4, 30.0), (2, 4.0e-4, 31.0)]))

    assert selection["selection_metric"] == "val_loss"
    assert selection["selection_domain"] == "validation"
    assert "best validation checkpoint" in selection["convention"]


# --- Part A: selection CANNOT reach the test set ----------------------------


@pytest.mark.parametrize("leaked_key", [
    "test_psnr", "test_bpp", "bd_rate_psnr", "aggregate_bpp",
    "deployed_bitrate", "benchmark_msssim",
])
def test_a_test_metric_in_the_history_is_a_hard_error(leaked_key):
    """Selection must not merely ignore a test metric - it must refuse. A
    filtered-away key would leave the caller believing selection was clean
    while the history file itself is wrong and keeps being reused."""
    mod = _load_script("m10g_evaluation_convention")
    history = _history([(1, 5.0e-4, 30.0), (2, 4.0e-4, 31.0)])
    history[1][leaked_key] = 0.42

    with pytest.raises(mod.TestMetricLeakError, match=leaked_key.split("_")[0]):
        mod.select_checkpoint(history)


def test_selecting_on_a_non_validation_objective_is_refused():
    mod = _load_script("m10g_evaluation_convention")
    history = _history([(1, 5.0e-4, 30.0), (2, 4.0e-4, 31.0)])
    for record in history:
        record["nvc_bpp"] = 1.0

    with pytest.raises(mod.TestMetricLeakError):
        mod.select_checkpoint(history, objective_key="nvc_bpp")


def test_only_validation_prefixed_objectives_are_accepted():
    mod = _load_script("m10g_evaluation_convention")
    history = _history([(1, 5.0e-4, 30.0), (2, 4.0e-4, 31.0)])

    assert mod.select_checkpoint(history, objective_key="val_loss")["selected_epoch"] == 2
    with pytest.raises(mod.TestMetricLeakError):
        mod.select_checkpoint(history, objective_key="loss")


def test_selection_signature_takes_no_test_data():
    """Structural guarantee, not a runtime one: there is no parameter through
    which a benchmark result could be supplied."""
    import inspect

    mod = _load_script("m10g_evaluation_convention")
    parameters = set(inspect.signature(mod.select_checkpoint).parameters)

    assert parameters == {"history", "objective_key", "objective_filter"}


# --- Part A: objective filtering and error handling -------------------------


def test_records_from_a_different_objective_are_ignored():
    """`--resume-model-only` carries stale pure-MSE epochs in from an earlier
    milestone; comparing against those would select an unrelated checkpoint.
    Same rule M9C.1 established."""
    mod = _load_script("m10g_evaluation_convention")
    history = (_history([(1, 1.0e-5, 40.0)], rate_enabled=False)
               + _history([(2, 5.0e-4, 30.0), (3, 4.0e-4, 31.0)]))

    selection = mod.select_checkpoint(history)

    assert selection["selected_epoch"] == 3
    assert selection["stale_records_ignored"] == 1
    assert selection["epochs_considered"] == 2


def test_an_empty_history_cannot_select_anything():
    mod = _load_script("m10g_evaluation_convention")

    with pytest.raises(mod.SelectionError):
        mod.select_checkpoint([])


def test_a_missing_objective_on_any_epoch_is_refused():
    mod = _load_script("m10g_evaluation_convention")
    history = _history([(1, 5.0e-4, 30.0), (2, 4.0e-4, 31.0)])
    del history[1]["val_loss"]

    with pytest.raises(mod.SelectionError, match="deterministic"):
        mod.select_checkpoint(history)


def test_a_missing_best_checkpoint_is_reported_not_invented(tmp_path):
    mod = _load_script("m10g_evaluation_convention")

    artifacts = mod.resolve_checkpoint_paths(tmp_path, {}, None)

    assert artifacts["primary_available"] is False
    assert artifacts["primary_checkpoint"] is None
    assert "not substituted" in artifacts["note"]


# --- Part B: non-destructive historical analysis ----------------------------


def test_part_b_reads_history_and_never_writes_into_the_experiment(tmp_path):
    mod = _load_script("m10g_evaluation_convention")
    run_dir = tmp_path / "lambda_3.0e-04_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "history.json").write_text(json.dumps(
        _history([(1, 5.0e-4, 30.0), (2, 4.0e-4, 31.0), (3, 4.6e-4, 30.2)])), encoding="utf-8")
    (run_dir / "best.pt").write_bytes(b"stub")
    (tmp_path / "training_summary.json").write_text(json.dumps({
        "runs": [{"name": "BRIDGE@s42", "lambda": 3.0e-4, "seed": 42,
                  "checkpoint_dir": str(run_dir), "snapshots": []}]
    }), encoding="utf-8")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    result = mod.analyse_experiment(tmp_path, "TEST")

    assert result["runs"][0]["selection"]["selected_epoch"] == 2
    assert result["runs"][0]["artifacts"]["primary_available"] is True
    after = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after, "Part B must not modify historical artifacts"


def test_part_b_skips_an_experiment_with_no_runs(tmp_path):
    mod = _load_script("m10g_evaluation_convention")
    (tmp_path / "training_summary.json").write_text(
        json.dumps({"status": "PREFLIGHT FAILED - no training performed"}), encoding="utf-8")

    assert mod.analyse_experiment(tmp_path, "TEST") is None


def test_part_b_reports_a_missing_history_rather_than_guessing(tmp_path):
    mod = _load_script("m10g_evaluation_convention")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "training_summary.json").write_text(json.dumps({
        "runs": [{"name": "X@s42", "lambda": 0.0, "seed": 42,
                  "checkpoint_dir": str(run_dir), "snapshots": []}]
    }), encoding="utf-8")

    result = mod.analyse_experiment(tmp_path, "TEST")

    assert "history.json not found" in result["runs"][0]["selection_error"]
    assert "selection" not in result["runs"][0]


# --- Script contract and repository safety ----------------------------------


@pytest.mark.parametrize("name", [
    "m10g_evaluation_convention", "m10g_temporal_baseline", "m10g_evaluate",
])
def test_every_m10g_script_follows_the_project_script_contract(name):
    mod = _load_script(name)

    assert callable(mod.build_arg_parser)
    assert callable(mod.main)
    parser = mod.build_arg_parser(load_default_config())
    assert parser.parse_args([]) is not None


def test_the_benchmark_control_arm_is_the_same_coder_at_gop_1():
    """Arm A and arm B must differ ONLY in whether P-frames exist - otherwise
    container overhead would be compared across two different formats."""
    evaluate = _load_script("m10g_evaluate")
    source = Path("scripts/m10g_evaluate.py").read_text(encoding="utf-8")

    assert '("A_intra_only", 1)' in source
    assert evaluate.FROZEN_LAMBDA == pytest.approx(3.0e-4)
    assert "best.pt" in str(evaluate.DEFAULT_CHECKPOINT), "must use the Part A convention"


def test_the_benchmark_counts_every_container_byte():
    """Reporting P-frame payload alone would hide the I-frames it depends on."""
    evaluate = _load_script("m10g_evaluate")
    source = Path("scripts/m10g_evaluate.py").read_text(encoding="utf-8")

    assert 'total_bytes * 8 / sequence.total_pixels' in source
    assert "container_overhead_bytes" in source


def test_drift_classification_distinguishes_accumulation_from_a_fixed_penalty():
    """These need opposite fixes, so the analysis must decide from the data
    rather than assume one. A shorter GOP helps accumulation and does nothing
    for a fixed per-P-frame penalty."""
    evaluate = _load_script("m10g_evaluate")

    def positions(values):
        return [{"gop_position": i, "frames": 5, "mean_psnr_db": v}
                for i, v in enumerate(values)]

    accumulating = evaluate.classify_drift(positions([30.0, 28.0, 26.0, 24.0, 22.0]))
    assert accumulating["accumulating"] is True
    assert "ACCUMULATION" in accumulating["shape"]

    fixed = evaluate.classify_drift(positions([30.2, 26.3, 26.2, 26.1, 26.6]))
    assert fixed["accumulating"] is False
    assert "FIXED" in fixed["shape"]
    assert fixed["i_to_first_p_drop_db"] == pytest.approx(-3.9, abs=0.05)

    assert evaluate.classify_drift(positions([30.0, 28.0])) is None


def test_m10g_does_not_modify_the_shipped_codec_or_training_path():
    from nvc.compression import nvc_format

    assert nvc_format.MAGIC == b"NVC1"
    assert nvc_format.STREAM_MAGIC == b"NVCS"

    trainer = _load_script("train_autoencoder")
    actions = {a.dest for a in trainer.build_arg_parser(load_default_config())._actions}
    for required in ("rate_lambda", "rate_lr", "rate_track_scale",
                     "rate_scale_momentum", "resume_model_only", "seed"):
        assert required in actions
