"""Tests for scripts/m9_final_train.py, the Milestone 9 final-training driver.

Per TESTING.md's "Added a new script?" rule: `build_arg_parser` defaults, a
missing-required-file error path, and a happy-path `main(argv)` run against
synthetic data checking the exit code and the files actually written.

Runs on CPU against tiny synthetic data. The real 30-epoch DAVIS runs this
script performs for the milestone are recorded in MILESTONE_9_PLAN.md, not
re-executed here.
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
    manifest = make_tiny_manifest(tmp_path, num_sequences=8, frames_per_sequence=4, width=32, height=32)
    checkpoint = make_tiny_checkpoint(
        tmp_path / "m8_qat.pt", epoch=40, model_kwargs={"base_channels": 32},
        history=[{"epoch": 40, "train_loss": 5e-4, "val_loss": 4.554493e-04, "val_psnr": 34.3}],
    )
    calibration = make_tiny_calibration(
        tmp_path / "calib.json", checkpoint_path=checkpoint, bits=4, mode="per_channel",
    )
    return manifest, checkpoint, calibration


def _argv(manifest: Path, checkpoint: Path, calibration: Path, output_dir: Path) -> list[str]:
    return [
        "--manifest", str(manifest), "--checkpoint", str(checkpoint),
        "--calibration", str(calibration),
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--epochs", "1", "--max-batches", "1", "--batch-size", "2",
        "--output-dir", str(output_dir), "--device", "cpu",
    ]


def test_arg_parser_defaults_match_the_approved_m9_configuration():
    mod = _load_script("m9_final_train")
    args = mod.build_arg_parser(load_default_config()).parse_args([])

    assert args.checkpoint == Path("outputs/qat_combined/checkpoints_qat_noise/best.pt")
    assert args.output_dir == Path("outputs/m9_final")
    assert args.epochs == 30
    assert args.max_batches is None, "final runs use the full train split"
    assert args.rate_lr == pytest.approx(1e-2)
    assert args.learning_rate == pytest.approx(1e-4), "model LR must stay at M8's"
    assert args.qat_bits == 4 and args.qat_mode == "per_channel"


def test_the_three_approved_lambdas_and_the_control_are_the_arms():
    mod = _load_script("m9_final_train")
    by_name = {arm["name"]: arm for arm in mod.ARMS}

    assert by_name["M9-L"]["lambda"] == pytest.approx(9.0757e-04)
    assert by_name["M9-M"]["lambda"] == pytest.approx(2.8700e-03)
    assert by_name["M9-H"]["lambda"] == pytest.approx(9.0757e-03)
    assert by_name["CTRL"]["lambda"] == 0.0
    # Directory names are the ones the M9 brief specifies.
    assert by_name["M9-L"]["dir"] == "lambda_9e-4"
    assert by_name["M9-M"]["dir"] == "lambda_2.87e-3"
    assert by_name["M9-H"]["dir"] == "lambda_9e-3"
    # Every arm writes somewhere distinct - no arm may overwrite another.
    assert len({arm["dir"] for arm in mod.ARMS}) == len(mod.ARMS)


def test_reports_a_missing_start_checkpoint(tmp_path, capsys):
    mod = _load_script("m9_final_train")
    manifest, _, calibration = _setup(tmp_path)

    exit_code = mod.main([
        "--manifest", str(manifest), "--calibration", str(calibration),
        "--checkpoint", str(tmp_path / "absent.pt"), "--device", "cpu",
    ])

    assert exit_code == 1
    assert "--checkpoint not found" in capsys.readouterr().err


def test_rejects_a_non_positive_rate_lr(tmp_path):
    mod = _load_script("m9_final_train")
    manifest, checkpoint, calibration = _setup(tmp_path)

    with pytest.raises(SystemExit):
        mod.main(_argv(manifest, checkpoint, calibration, tmp_path / "out") + ["--rate-lr=0"])


def test_only_selects_a_subset_of_arms(tmp_path):
    mod = _load_script("m9_final_train")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "out"

    assert mod.main(
        _argv(manifest, checkpoint, calibration, output_dir) + ["--only", "M9-L"]
    ) == 0
    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert [arm["name"] for arm in summary["arms"]] == ["M9-L"]


def test_unknown_only_name_is_rejected(tmp_path):
    mod = _load_script("m9_final_train")
    manifest, checkpoint, calibration = _setup(tmp_path)

    with pytest.raises(SystemExit):
        mod.main(_argv(manifest, checkpoint, calibration, tmp_path / "out") + ["--only", "M9-X"])


def test_end_to_end_writes_summaries_checkpoints_and_hashes(tmp_path):
    mod = _load_script("m9_final_train")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "out"

    exit_code = mod.main(_argv(manifest, checkpoint, calibration, output_dir))

    assert exit_code == 0
    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert len(summary["arms"]) == 4, "three lambdas plus the control"
    assert summary["model_learning_rate"] == pytest.approx(1e-4)
    assert summary["rate_lr"] == pytest.approx(1e-2)

    for arm in summary["arms"]:
        # Every arm must start from the same checkpoint, and say so verifiably.
        assert arm["start_checkpoint_sha256"] == summary["start_checkpoint_sha256"]
        assert len(arm["best_checkpoint_sha256"]) == 64
        assert arm["all_finite"] is True
        assert arm["rate_estimator_state_restored"] is True
        # best.pt must exist - its absence was the M9 pre-flight defect.
        assert Path(arm["best_checkpoint"]).is_file()
        assert arm["val_total_objective"] == pytest.approx(
            arm["val_distortion"] + arm["lambda"] * arm["val_rate_bpp_proxy"]
        )

    # Distinct arms must produce distinct checkpoints.
    hashes = {arm["best_checkpoint_sha256"] for arm in summary["arms"]}
    assert len(hashes) == len(summary["arms"])

    assert (output_dir / "training_summary.csv").is_file()


def test_the_start_checkpoint_is_never_modified(tmp_path):
    """M9 must not touch the M8 artifact it builds on."""
    mod = _load_script("m9_final_train")
    manifest, checkpoint, calibration = _setup(tmp_path)
    before = checkpoint.read_bytes()

    mod.main(_argv(manifest, checkpoint, calibration, tmp_path / "out") + ["--only", "M9-M"])

    assert checkpoint.read_bytes() == before


def test_control_can_be_skipped(tmp_path):
    mod = _load_script("m9_final_train")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "out"

    assert mod.main(
        _argv(manifest, checkpoint, calibration, output_dir) + ["--skip-control"]
    ) == 0
    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert all(arm["lambda"] > 0 for arm in summary["arms"])
    assert len(summary["arms"]) == 3


def test_summary_records_that_rate_is_a_proxy_not_bitrate(tmp_path):
    """The rate figures must never be presented as .nvc bitrate."""
    mod = _load_script("m9_final_train")
    manifest, checkpoint, calibration = _setup(tmp_path)
    output_dir = tmp_path / "out"

    mod.main(_argv(manifest, checkpoint, calibration, output_dir) + ["--only", "M9-H"])

    summary = json.loads((output_dir / "training_summary.json").read_text())
    assert "NOT .nvc bitrate" in summary["note"]
    assert "val_rate_bpp_proxy" in summary["arms"][0]
