"""Tests for M10C: retained step snapshots and budget-paired evaluation.

M10C introduces two genuinely new pieces; everything else is unchanged code
already covered elsewhere (scale tracking, the recording estimator, checkpoint
selection, calibration, benchmarking, BD-rate methodology):

  1. `make_snapshotting_save` - retaining `latest.pt` at chosen epochs WITHOUT
     touching `train_autoencoder.py` or perturbing the optimization;
  2. `snapshot_models` - turning those snapshots into benchmark models keyed by
     (arm, step) so the RD comparison can be paired by training budget.

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


# --- 1. Snapshotting must be additive and non-invasive ---------------------


def test_snapshotting_save_delegates_and_copies_only_on_target_epochs(tmp_path):
    mod = _load_script("m10c_convergence")
    calls: list[tuple[Path, int]] = []

    def fake_save(path, **kwargs):
        Path(path).write_text(f"epoch{kwargs['epoch']}", encoding="utf-8")
        calls.append((Path(path), kwargs["epoch"]))

    recorded: list[dict] = []
    save = mod.make_snapshotting_save(
        fake_save, target_epochs={1, 3}, steps_per_epoch=100,
        first_epoch_holder={"value": None}, recorded=recorded,
    )

    # Global epochs 41..44 == own epochs 1..4 (the run resumes M8's epoch 40).
    for epoch in (41, 42, 43, 44):
        save(tmp_path / "latest.pt", epoch=epoch)

    assert len(calls) == 4, "the real save must run on every epoch"
    assert [r["own_epoch"] for r in recorded] == [1, 3]
    assert [r["step"] for r in recorded] == [100, 300]
    assert [r["global_epoch"] for r in recorded] == [41, 43]
    assert (tmp_path / "snapshot_step000100.pt").is_file()
    assert (tmp_path / "snapshot_step000300.pt").is_file()
    assert not (tmp_path / "snapshot_step000200.pt").exists()
    # The snapshot is a copy of what the real save wrote, not a re-serialisation.
    assert (tmp_path / "snapshot_step000300.pt").read_text() == "epoch43"


def test_snapshotting_save_ignores_best_pt(tmp_path):
    """best.pt is written on its own schedule; only the per-epoch latest.pt is
    a valid step marker."""
    mod = _load_script("m10c_convergence")
    recorded: list[dict] = []
    save = mod.make_snapshotting_save(
        lambda path, **kw: Path(path).write_text("x", encoding="utf-8"),
        target_epochs={1}, steps_per_epoch=100,
        first_epoch_holder={"value": None}, recorded=recorded,
    )

    save(tmp_path / "best.pt", epoch=41)

    assert recorded == []
    assert not list(tmp_path.glob("snapshot_*"))


def test_snapshotting_save_derives_the_offset_from_the_first_epoch_seen(tmp_path):
    """The run's own epoch 1 is whatever epoch it actually starts at - it must
    not be hardcoded to 41."""
    mod = _load_script("m10c_convergence")
    recorded: list[dict] = []
    save = mod.make_snapshotting_save(
        lambda path, **kw: Path(path).write_text("x", encoding="utf-8"),
        target_epochs={1, 2}, steps_per_epoch=50,
        first_epoch_holder={"value": None}, recorded=recorded,
    )

    for epoch in (7, 8, 9):
        save(tmp_path / "latest.pt", epoch=epoch)

    assert [r["own_epoch"] for r in recorded] == [1, 2]
    assert [r["global_epoch"] for r in recorded] == [7, 8]
    assert [r["step"] for r in recorded] == [50, 100]


# --- Configuration ---------------------------------------------------------


def test_default_budget_is_the_established_m9_final_configuration():
    mod = _load_script("m10c_convergence")
    args = mod.build_arg_parser(load_default_config()).parse_args([])

    assert args.epochs == 30, "the M9-final budget, not an invented schedule"
    assert args.learning_rate == pytest.approx(1e-4)
    assert args.rate_lr == pytest.approx(1e-2)
    assert args.scale_momentum == pytest.approx(0.99)
    assert args.seed == 42
    assert args.qat_bits == 4 and args.qat_mode == "per_channel"
    assert args.expect_sha256 == mod.EXPECTED_START_SHA256
    assert args.snapshot_epochs == [1, 3, 8, 17, 30]


def test_exactly_two_arms_at_the_bracketed_optimum():
    mod = _load_script("m10c_convergence")
    by_name = {arm["name"]: arm for arm in mod.ARMS}

    assert set(by_name) == {"CTRL", "M10C-L"}
    assert by_name["CTRL"]["lambda"] == 0.0
    assert by_name["M10C-L"]["lambda"] == pytest.approx(9.0757e-04)
    assert by_name["CTRL"]["dir"] != by_name["M10C-L"]["dir"]


def test_refuses_a_start_checkpoint_with_the_wrong_hash(tmp_path, capsys):
    mod = _load_script("m10c_convergence")
    manifest, checkpoint, calibration = _setup(tmp_path)

    exit_code = mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--device", "cpu",
    ])

    assert exit_code == 1
    assert "sha256 mismatch" in capsys.readouterr().err


def test_rejects_snapshot_epochs_outside_the_budget(tmp_path):
    mod = _load_script("m10c_convergence")
    manifest, checkpoint, calibration = _setup(tmp_path)

    with pytest.raises(SystemExit):
        mod.main([
            "--manifest", str(manifest), "--checkpoint", str(checkpoint),
            "--calibration", str(calibration), "--expect-sha256", "",
            "--epochs", "3", "--snapshot-epochs", "1", "9", "--device", "cpu",
        ])


def test_end_to_end_retains_snapshots_for_both_arms(tmp_path):
    mod = _load_script("m10c_convergence")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10c"

    exit_code = mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--expect-sha256", "",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--epochs", "2", "--snapshot-epochs", "1", "2", "--batch-size", "2",
        "--output-dir", str(output_dir), "--device", "cpu",
    ])

    assert exit_code == 0
    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert [arm["name"] for arm in summary["arms"]] == ["CTRL", "M10C-L"]
    assert summary["track_scale"] is True

    steps_per_epoch = summary["steps_per_epoch"]
    for arm in summary["arms"]:
        assert arm["all_finite"] is True
        assert arm["best_selection_used_only_current_objective"] is True
        assert arm["stale_history_records_ignored"] == 1
        assert arm["start_checkpoint_sha256"] == summary["start_checkpoint_sha256"]
        # Both arms must produce the SAME snapshot step counts - that is what
        # makes the comparison paired by budget.
        assert [s["step"] for s in arm["snapshots"]] == [steps_per_epoch, 2 * steps_per_epoch]
        for snapshot in arm["snapshots"]:
            assert Path(snapshot["path"]).is_file()
            assert len(snapshot["sha256"]) == 64
            assert "val_distortion" in snapshot and "val_psnr_db" in snapshot

    # The two arms must be paired step-for-step.
    ctrl, rate = summary["arms"]
    assert [s["step"] for s in ctrl["snapshots"]] == [s["step"] for s in rate["snapshots"]]


def test_control_arm_leaves_the_rate_estimator_unfitted(tmp_path):
    """lambda=0 gives exactly zero rate gradient, so its proxy R is not a
    meaningful quantity - pinned so the analysis never treats it as one."""
    mod = _load_script("m10c_convergence")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "m10c"

    mod.main([
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration), "--expect-sha256", "",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--epochs", "1", "--snapshot-epochs", "1", "--batch-size", "2",
        "--only", "CTRL", "--output-dir", str(output_dir), "--device", "cpu",
    ])

    summary = json.loads((output_dir / "training_summary.json").read_text())
    snapshot = summary["arms"][0]["snapshots"][0]
    assert snapshot["rate_scale_mean"] == pytest.approx(1.0, abs=1e-9)
    assert snapshot["rate_grad_norm"] == pytest.approx(0.0, abs=1e-12)


# --- 2. Budget-paired evaluation -------------------------------------------


def test_snapshot_models_are_keyed_by_arm_and_step():
    evaluate = _load_script("m10c_evaluate")
    summary = {
        "arms": [
            {"name": "CTRL", "lambda": 0.0,
             "snapshots": [{"step": 604, "path": "a/snapshot_step000604.pt"},
                           {"step": 1812, "path": "a/snapshot_step001812.pt"}]},
            {"name": "M10C-L", "lambda": 9.0757e-04,
             "snapshots": [{"step": 604, "path": "b/snapshot_step000604.pt"},
                           {"step": 1812, "path": "b/snapshot_step001812.pt"}]},
        ]
    }

    models = evaluate.snapshot_models(summary)

    assert [m["key"] for m in models] == [
        "CTRL@604", "CTRL@1812", "M10C-L@604", "M10C-L@1812",
    ]
    assert len({m["key"] for m in models}) == 4, "keys must be unique per (arm, step)"
    # Every step present for one arm must be present for the other, or the
    # comparison could not be paired.
    steps = {m["arm"]: sorted(x["step"] for x in models if x["arm"] == m["arm"]) for m in models}
    assert steps["CTRL"] == steps["M10C-L"]


def test_bd_rate_is_piecewise_linear_and_undefined_without_overlap():
    evaluate = _load_script("m10c_evaluate")

    identical = [(1.0, 30.0), (0.7, 29.0), (0.5, 28.0)]
    assert evaluate._bd_rate_linear(identical, identical) == pytest.approx(0.0, abs=1e-9)
    halved = [(0.5, 30.0), (0.35, 29.0), (0.25, 28.0)]
    assert evaluate._bd_rate_linear(identical, halved) == pytest.approx(-50.0, abs=0.5)
    assert evaluate._bd_rate_linear(identical, [(1.0, 20.0), (0.7, 19.0), (0.5, 18.0)]) is None


def test_evaluate_requires_the_training_summary(tmp_path, capsys):
    evaluate = _load_script("m10c_evaluate")
    manifest, _, _ = _setup(tmp_path)

    exit_code = evaluate.main([
        "--manifest", str(manifest), "--output-dir", str(tmp_path / "absent"),
        "--stage", "calibrate", "--device", "cpu",
    ])

    assert exit_code == 1
    assert "training_summary.json not found" in capsys.readouterr().err
