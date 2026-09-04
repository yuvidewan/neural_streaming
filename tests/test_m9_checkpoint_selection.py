"""Regression tests for M9's best-checkpoint selection across objectives.

THE BUG
--------
`best_val_loss` was seeded by minimising `val_loss` over the whole resumed
history. But `val_loss` means different things in different runs:

    distortion-only run:  plain MSE
    rate-aware run:       D + lambda * R

Resuming the M8 QAT checkpoint into an M9 rate run therefore seeded the
best-so-far with M8's pure-MSE minimum (4.55e-04, measured on Vimeo), which no
`D + lambda*R` epoch can beat. `best.pt` was silently never written - every
M9C and M9C.1 pilot arm produced only `latest.pt`.

THE FIX
--------
Seed only from history records produced under the SAME objective as this run.
The comparison criterion itself is unchanged and was already correct:
`val_metrics["loss"]` for a rate run already is `D + lambda*R`.

Backward compatibility is the point of most of these tests: a distortion-only
run must behave exactly as it always has, including against pre-M8A history
records that have no `rate_enabled` key at all.
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

from nvc.models import BaselineAutoencoder  # noqa: E402
from nvc.training import save_checkpoint  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_with_history(path: Path, history: list[dict], *, epoch: int) -> Path:
    """A resumable checkpoint (base_channels=32, the CLI default) carrying a
    hand-built history, so each test controls exactly what the seeding sees."""
    model = BaselineAutoencoder(
        in_channels=3, latent_channels=TINY_MODEL_KWARGS["latent_channels"], base_channels=32,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    save_checkpoint(
        path, model=model, optimizer=optimizer, epoch=epoch,
        history=history, model_config=model.config_dict(),
    )
    return path


def _setup(tmp_path: Path):
    manifest = make_tiny_manifest(tmp_path, num_sequences=8, frames_per_sequence=4, width=32, height=32)
    calibration = make_tiny_calibration(
        tmp_path / "calib.json",
        checkpoint_path=make_tiny_checkpoint(tmp_path / "seed.pt"),
        bits=4, mode="per_channel",
    )
    return manifest, calibration


def _rate_argv(manifest: Path, calibration: Path, checkpoint_dir: Path, lambda_rate: str) -> list[str]:
    return [
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1", "--batch-size", "2",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--qat-enabled", "--qat-bits", "4", "--qat-mode", "per_channel",
        "--qat-calibration", str(calibration),
        "--rate-enabled", "--rate-lambda", lambda_rate, "--rate-lr", "1e-2",
        "--checkpoint-dir", str(checkpoint_dir), "--device", "cpu",
    ]


# --- The bug ---------------------------------------------------------------


def test_rate_run_resuming_an_m8_style_checkpoint_still_writes_best(tmp_path):
    """The exact failure: an M8-style pure-MSE history with a val_loss far
    below any achievable D+lambda*R must not suppress best.pt."""
    mod = _load_script("train_autoencoder")
    manifest, calibration = _setup(tmp_path)
    m8_history = [
        {"epoch": e, "train_loss": 5e-4, "val_loss": 4.554493e-04, "val_psnr": 34.3,
         "qat_enabled": True, "qat_bits": 4, "qat_mode": "per_channel"}
        for e in range(1, 41)
    ]
    checkpoint = _checkpoint_with_history(tmp_path / "m8.pt", m8_history, epoch=40)
    checkpoint_dir = tmp_path / "m9"

    exit_code = mod.main(
        _rate_argv(manifest, calibration, checkpoint_dir, "0.0009075672")
        + ["--resume", str(checkpoint), "--resume-model-only"]
    )

    assert exit_code == 0
    assert (checkpoint_dir / "best.pt").is_file(), "best.pt must be written for the M9 objective"
    saved = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
    assert saved["extra"]["rate_lambda"] == pytest.approx(0.0009075672)


def test_the_ignored_history_is_reported_not_silent(tmp_path, capsys):
    mod = _load_script("train_autoencoder")
    manifest, calibration = _setup(tmp_path)
    checkpoint = _checkpoint_with_history(
        tmp_path / "m8.pt",
        [{"epoch": 1, "train_loss": 5e-4, "val_loss": 4.5e-04, "val_psnr": 34.3}],
        epoch=1,
    )

    mod.main(
        _rate_argv(manifest, calibration, tmp_path / "m9", "0.001")
        + ["--resume", str(checkpoint), "--resume-model-only"]
    )

    assert "produced under a different objective" in capsys.readouterr().out


# --- The criterion is the rate-aware objective, not MSE --------------------


def test_best_is_selected_on_d_plus_lambda_r_for_a_rate_run(tmp_path, capsys):
    """The saved best epoch must be the one minimising the run's own reported
    val_loss (D + lambda*R), which is what history records."""
    mod = _load_script("train_autoencoder")
    manifest, calibration = _setup(tmp_path)
    checkpoint_dir = tmp_path / "m9"

    exit_code = mod.main(
        _rate_argv(manifest, calibration, checkpoint_dir, "0.001")[:2]
        + ["--epochs", "3"]
        + _rate_argv(manifest, calibration, checkpoint_dir, "0.001")[4:]
    )

    assert exit_code == 0
    history = json.loads((checkpoint_dir / "history.json").read_text())
    rate_records = [r for r in history if r.get("rate_enabled")]
    best_epoch = min(rate_records, key=lambda r: r["val_loss"])["epoch"]
    saved = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
    assert saved["epoch"] == best_epoch
    # And the reported objective is named correctly in the log.
    assert "New best validation D + lambda*R" in capsys.readouterr().out


def test_best_log_still_says_mse_for_a_distortion_only_run(tmp_path, capsys):
    mod = _load_script("train_autoencoder")
    manifest, _ = _setup(tmp_path)

    mod.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1", "--batch-size", "2",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--checkpoint-dir", str(tmp_path / "plain"), "--device", "cpu",
    ])

    assert "New best validation MSE" in capsys.readouterr().out


# --- Backward compatibility ------------------------------------------------


def test_distortion_only_run_still_honours_a_pre_m8a_history(tmp_path):
    """Records with NO rate_enabled key at all (pre-M8A checkpoints) must still
    count for a distortion-only run - this is the compatibility case the fix
    could most easily have broken."""
    mod = _load_script("train_autoencoder")
    manifest, _ = _setup(tmp_path)
    legacy_history = [{"epoch": 1, "train_loss": 1e-9, "val_loss": 1e-9, "val_psnr": 90.0}]
    checkpoint = _checkpoint_with_history(tmp_path / "legacy.pt", legacy_history, epoch=1)
    checkpoint_dir = tmp_path / "plain"

    exit_code = mod.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1", "--batch-size", "2",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--resume", str(checkpoint), "--resume-model-only",
        "--checkpoint-dir", str(checkpoint_dir), "--device", "cpu",
    ])

    assert exit_code == 0
    # An unbeatably-low historical val_loss must still suppress best.pt here,
    # exactly as it always did for same-objective runs.
    assert not (checkpoint_dir / "best.pt").exists()


def test_distortion_only_run_honours_an_m8_style_history(tmp_path):
    """Same, for records that DO carry rate_enabled: False."""
    mod = _load_script("train_autoencoder")
    manifest, _ = _setup(tmp_path)
    history = [{"epoch": 1, "train_loss": 1e-9, "val_loss": 1e-9, "val_psnr": 90.0,
                "qat_enabled": False, "rate_enabled": False, "rate_lambda": None}]
    checkpoint = _checkpoint_with_history(tmp_path / "m8.pt", history, epoch=1)
    checkpoint_dir = tmp_path / "plain"

    mod.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1", "--batch-size", "2",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--resume", str(checkpoint), "--resume-model-only",
        "--checkpoint-dir", str(checkpoint_dir), "--device", "cpu",
    ])

    assert not (checkpoint_dir / "best.pt").exists()


def test_a_rate_run_continues_its_own_history_at_the_same_lambda(tmp_path):
    """Resuming the SAME objective must keep the prior best, not restart - so a
    genuinely worse continuation does not overwrite a better checkpoint."""
    mod = _load_script("train_autoencoder")
    manifest, calibration = _setup(tmp_path)
    history = [{"epoch": 1, "train_loss": 1e-9, "val_loss": 1e-9, "val_psnr": 90.0,
                "qat_enabled": True, "rate_enabled": True, "rate_lambda": 0.001, "rate_lr": 1e-2}]
    checkpoint = _checkpoint_with_history(tmp_path / "m9prev.pt", history, epoch=1)
    checkpoint_dir = tmp_path / "cont"

    mod.main(
        _rate_argv(manifest, calibration, checkpoint_dir, "0.001")
        + ["--resume", str(checkpoint), "--resume-model-only"]
    )

    assert not (checkpoint_dir / "best.pt").exists(), "an unbeatable same-objective best must hold"


def test_a_different_lambda_is_a_different_objective(tmp_path):
    """Two rate runs at different lambdas are not comparable either, so a
    lambda change must reset best-checkpoint tracking."""
    mod = _load_script("train_autoencoder")
    manifest, calibration = _setup(tmp_path)
    history = [{"epoch": 1, "train_loss": 1e-9, "val_loss": 1e-9, "val_psnr": 90.0,
                "qat_enabled": True, "rate_enabled": True, "rate_lambda": 0.001, "rate_lr": 1e-2}]
    checkpoint = _checkpoint_with_history(tmp_path / "m9prev.pt", history, epoch=1)
    checkpoint_dir = tmp_path / "other"

    mod.main(
        _rate_argv(manifest, calibration, checkpoint_dir, "0.005")
        + ["--resume", str(checkpoint), "--resume-model-only"]
    )

    assert (checkpoint_dir / "best.pt").is_file()


def test_a_fresh_run_with_no_history_writes_best_on_the_first_epoch(tmp_path):
    mod = _load_script("train_autoencoder")
    manifest, calibration = _setup(tmp_path)
    checkpoint_dir = tmp_path / "fresh"

    mod.main(_rate_argv(manifest, calibration, checkpoint_dir, "0.001"))

    assert (checkpoint_dir / "best.pt").is_file()
