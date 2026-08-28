"""Script-level tests: scripts/train_autoencoder.py, plot_training_history.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from helpers import TINY_MODEL_KWARGS, make_tiny_calibration, make_tiny_manifest  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- train_autoencoder.py ------------------------------------------------


def test_train_autoencoder_smoke_test_writes_checkpoints_and_history(tmp_path):
    manifest = make_tiny_manifest(tmp_path)
    mod = _load_script("train_autoencoder")
    checkpoint_dir = tmp_path / "checkpoints"

    exit_code = mod.main([
        "--manifest", str(manifest),
        "--epochs", "1", "--max-batches", "2",
        "--batch-size", "2",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--checkpoint-dir", str(checkpoint_dir),
        "--device", "cpu",
    ])

    assert exit_code == 0
    assert (checkpoint_dir / "latest.pt").is_file()
    assert (checkpoint_dir / "best.pt").is_file()
    history = json.loads((checkpoint_dir / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 1
    assert history[0]["epoch"] == 1
    assert history[0]["qat_enabled"] is False


def test_train_autoencoder_resume_continues_epoch_numbering(tmp_path):
    manifest = make_tiny_manifest(tmp_path)
    mod = _load_script("train_autoencoder")
    checkpoint_dir = tmp_path / "checkpoints"
    common_args = [
        "--manifest", str(manifest), "--max-batches", "2", "--batch-size", "2",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--checkpoint-dir", str(checkpoint_dir), "--device", "cpu",
    ]

    assert mod.main([*common_args, "--epochs", "1"]) == 0
    assert mod.main([*common_args, "--epochs", "1", "--resume", str(checkpoint_dir / "latest.pt")]) == 0

    history = json.loads((checkpoint_dir / "history.json").read_text(encoding="utf-8"))
    assert [record["epoch"] for record in history] == [1, 2]


def test_train_autoencoder_qat_enabled_records_qat_fields_in_history(tmp_path):
    manifest = make_tiny_manifest(tmp_path)
    mod = _load_script("train_autoencoder")
    checkpoint_dir = tmp_path / "checkpoints"
    calibration = make_tiny_calibration(
        tmp_path / "calib.json", checkpoint_path=tmp_path / "unused.pt", bits=4, mode="per_channel",
    )

    exit_code = mod.main([
        "--manifest", str(manifest),
        "--epochs", "1", "--max-batches", "2", "--batch-size", "2",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--checkpoint-dir", str(checkpoint_dir), "--device", "cpu",
        "--qat-enabled", "--qat-bits", "4", "--qat-mode", "per_channel",
        "--qat-calibration", str(calibration),
    ])

    assert exit_code == 0
    history = json.loads((checkpoint_dir / "history.json").read_text(encoding="utf-8"))
    assert history[0]["qat_enabled"] is True
    assert history[0]["qat_bits"] == 4
    assert history[0]["qat_mode"] == "per_channel"


def test_train_autoencoder_qat_enabled_without_calibration_errors(tmp_path):
    manifest = make_tiny_manifest(tmp_path)
    mod = _load_script("train_autoencoder")

    exit_code = mod.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1",
        "--qat-enabled", "--qat-calibration", "",  # blank -> falls through to None-check path
    ])
    assert exit_code != 0


def test_train_autoencoder_missing_manifest_fails_cleanly(tmp_path):
    mod = _load_script("train_autoencoder")
    exit_code = mod.main(["--manifest", str(tmp_path / "nope.json"), "--epochs", "1"])
    assert exit_code != 0


def test_train_autoencoder_rejects_zero_epochs():
    import pytest

    mod = _load_script("train_autoencoder")
    with pytest.raises(SystemExit):
        mod.main(["--manifest", "irrelevant.json", "--epochs", "0"])


# --- plot_training_history.py ---------------------------------------------


def test_plot_training_history_end_to_end(tmp_path):
    mod = _load_script("plot_training_history")
    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps([
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_psnr": 10.0},
        {"epoch": 2, "train_loss": 0.3, "val_loss": 0.25, "val_psnr": 13.0},
    ]), encoding="utf-8")
    output_path = tmp_path / "curves.png"

    exit_code = mod.main(["--history", str(history_path), "--output", str(output_path)])

    assert exit_code == 0
    assert output_path.is_file()


def test_plot_training_history_single_epoch_warns_but_succeeds(tmp_path):
    mod = _load_script("plot_training_history")
    history_path = tmp_path / "history.json"
    history_path.write_text(json.dumps([
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_psnr": 10.0},
    ]), encoding="utf-8")

    exit_code = mod.main(["--history", str(history_path), "--output", str(tmp_path / "curves.png")])
    assert exit_code == 0


def test_plot_training_history_missing_file_fails_cleanly(tmp_path):
    mod = _load_script("plot_training_history")
    exit_code = mod.main(["--history", str(tmp_path / "nope.json")])
    assert exit_code != 0


def test_plot_training_history_empty_history_fails_cleanly(tmp_path):
    mod = _load_script("plot_training_history")
    history_path = tmp_path / "history.json"
    history_path.write_text("[]", encoding="utf-8")
    exit_code = mod.main(["--history", str(history_path)])
    assert exit_code != 0
