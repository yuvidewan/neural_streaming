"""Regression tests for the Milestone 9C resume bug.

THE BUG
--------
`--rate-enabled` puts the rate estimator's `loc`/`log_scale` into the
optimizer's parameter list alongside the model's own parameters. Every
pre-M9 checkpoint (M7, M8 QAT, M8 control) was written by a run whose
optimizer held ONLY the model's parameters. Resuming one of those into a
rate-enabled run therefore asked `optimizer.load_state_dict` to fit a
16-parameter saved state into an 18-parameter optimizer, which raises
`ValueError: loaded state dict contains a parameter group that doesn't match
the size of optimizer's group`. `scripts/train_autoencoder.py` caught only
`RuntimeError`, so this surfaced as an uncaught traceback.

That blocked M9C's Rule 6 exactly - "keep the M8 QAT checkpoint as the
starting model" was impossible through the CLI, for any lambda.

THE FIX
--------
`nvc.training.checkpoint.resume_model_only` (weights + epoch/history, no
optimizer state) and a `--resume-model-only` CLI flag, plus a `ValueError`
handler that names the real cause instead of surfacing PyTorch's bare
message. See MILESTONE_9_PLAN.md's M9C section.

Everything here runs on synthetic data via tests/helpers.py, per this
project's testing standard - no GPU, no DAVIS, no real checkpoint needed.
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
from nvc.training import (  # noqa: E402
    QuantizationNoise,
    RateEstimator,
    resume_model_only,
    resume_training_state,
)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli_checkpoint(path: Path, *, epoch: int = 1) -> Path:
    """A checkpoint the CLI can actually resume.

    `train_autoencoder.py` exposes `--latent-channels` but no
    `--base-channels`, so a resumable checkpoint has to be built at the CLI's
    own default `base_channels=32` - `TINY_MODEL_KWARGS`' 8 would fail
    `load_state_dict` on a shape mismatch before any of this file's actual
    subject matter was reached. Latent channels stay tiny, so the model is
    still small.
    """
    return make_tiny_checkpoint(path, epoch=epoch, model_kwargs={"base_channels": 32})


def _rate_enabled_setup(calibration: Path):
    """A model + optimizer configured exactly as train_autoencoder.py builds
    them when --qat-enabled and --rate-enabled are both passed."""
    noise = QuantizationNoise.from_calibration(calibration, bits=4, mode="per_channel")
    model = BaselineAutoencoder(quantization_noise=noise, **TINY_MODEL_KWARGS)
    estimator = RateEstimator(noise.scale, bits=noise.bits, mode=noise.mode)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(estimator.parameters()), lr=1e-4)
    return model, estimator, optimizer


# --- The bug itself --------------------------------------------------------


def test_resume_training_state_rejects_a_pre_rate_checkpoint(tmp_path):
    """The original failure, pinned so it cannot silently come back."""
    checkpoint = make_tiny_checkpoint(tmp_path / "pre_rate.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint)
    model, _, optimizer = _rate_enabled_setup(calibration)

    with pytest.raises(ValueError, match="parameter group"):
        resume_training_state(checkpoint, model=model, optimizer=optimizer, map_location="cpu")


def test_rate_enabled_optimizer_holds_two_more_params_than_a_pre_rate_one(tmp_path):
    """Documents *why* the above fails: the rate estimator's loc/log_scale."""
    checkpoint = make_tiny_checkpoint(tmp_path / "pre_rate.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint)
    model, estimator, optimizer = _rate_enabled_setup(calibration)

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_count = sum(len(group["params"]) for group in saved["optimizer_state_dict"]["param_groups"])
    live_count = sum(len(group["params"]) for group in optimizer.state_dict()["param_groups"])

    assert live_count == saved_count + 2
    assert [name for name, _ in estimator.named_parameters()] == ["loc", "log_scale"]


# --- The fix ---------------------------------------------------------------


def test_resume_model_only_restores_weights_from_a_pre_rate_checkpoint(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path / "pre_rate.pt", epoch=7, history=[{"epoch": 7}])
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint)
    model, _, _ = _rate_enabled_setup(calibration)

    next_epoch, history = resume_model_only(checkpoint, model=model, map_location="cpu")

    assert next_epoch == 8
    assert history == [{"epoch": 7}]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, saved[name]), f"{name} was not restored"


def test_resume_model_only_leaves_the_optimizer_state_empty(tmp_path):
    """The deliberate part: Adam moments accumulated under a pure-distortion
    objective are NOT carried into a D+lambda*R run, so every lambda arm of a
    sweep starts from an identical, empty optimizer state."""
    checkpoint = make_tiny_checkpoint(tmp_path / "pre_rate.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint)
    model, _, optimizer = _rate_enabled_setup(calibration)

    resume_model_only(checkpoint, model=model, map_location="cpu")

    assert optimizer.state_dict()["state"] == {}


def test_resume_model_only_matches_resume_training_state_on_weights(tmp_path):
    """The two paths must restore identical weights - they differ only in
    whether optimizer state comes along."""
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")

    with_optimizer = BaselineAutoencoder(**TINY_MODEL_KWARGS)
    resume_training_state(
        checkpoint, model=with_optimizer,
        optimizer=torch.optim.Adam(with_optimizer.parameters(), lr=1e-4), map_location="cpu",
    )
    weights_only = BaselineAutoencoder(**TINY_MODEL_KWARGS)
    resume_model_only(checkpoint, model=weights_only, map_location="cpu")

    for (name, a), (_, b) in zip(with_optimizer.state_dict().items(), weights_only.state_dict().items()):
        assert torch.equal(a, b), f"{name} differs between the two resume paths"


def test_resume_model_only_still_works_for_an_ordinary_non_rate_model(tmp_path):
    """The new function is not rate-specific - it must work for any model."""
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt", epoch=3)
    model = BaselineAutoencoder(**TINY_MODEL_KWARGS)

    next_epoch, _ = resume_model_only(checkpoint, model=model, map_location="cpu")

    assert next_epoch == 4


# --- CLI integration -------------------------------------------------------


def test_cli_resume_model_only_requires_resume(tmp_path, capsys):
    mod = _load_script("train_autoencoder")
    manifest = make_tiny_manifest(tmp_path, num_sequences=6, frames_per_sequence=4, width=32, height=32)

    with pytest.raises(SystemExit):
        mod.main(["--manifest", str(manifest), "--resume-model-only", "--device", "cpu"])
    assert "only makes sense together with --resume" in capsys.readouterr().err


def test_cli_reports_the_pre_rate_checkpoint_mismatch_actionably(tmp_path, capsys):
    """Without --resume-model-only the run must fail with an error that names
    the cause and the remedy - not an uncaught ValueError traceback."""
    mod = _load_script("train_autoencoder")
    manifest = make_tiny_manifest(tmp_path, num_sequences=6, frames_per_sequence=4, width=32, height=32)
    checkpoint = _cli_checkpoint(tmp_path / "pre_rate.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint)

    exit_code = mod.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--qat-enabled", "--qat-bits", "4", "--qat-mode", "per_channel",
        "--qat-calibration", str(calibration),
        "--rate-enabled", "--rate-lambda", "0.001",
        "--resume", str(checkpoint),
        "--checkpoint-dir", str(tmp_path / "run"), "--device", "cpu",
    ])

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "predates --rate-enabled" in stderr
    assert "--resume-model-only" in stderr


def test_cli_qat_plus_rate_resumes_a_pre_rate_checkpoint_with_the_flag(tmp_path):
    """The end-to-end path M9C's pilot actually uses: M8-style checkpoint in,
    QAT + rate training out."""
    mod = _load_script("train_autoencoder")
    manifest = make_tiny_manifest(tmp_path, num_sequences=6, frames_per_sequence=4, width=32, height=32)
    checkpoint = _cli_checkpoint(tmp_path / "pre_rate.pt", epoch=40)
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint)
    checkpoint_dir = tmp_path / "pilot"

    exit_code = mod.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--qat-enabled", "--qat-bits", "4", "--qat-mode", "per_channel",
        "--qat-calibration", str(calibration),
        "--rate-enabled", "--rate-lambda", "0.001",
        "--resume", str(checkpoint), "--resume-model-only",
        "--checkpoint-dir", str(checkpoint_dir), "--device", "cpu",
    ])

    assert exit_code == 0
    history = json.loads((checkpoint_dir / "history.json").read_text())
    assert history[-1]["epoch"] == 41, "epoch numbering must continue from the checkpoint"
    assert history[-1]["rate_enabled"] is True
    assert history[-1]["rate_lambda"] == 0.001
    assert history[-1]["train_rate_bpp"] is not None

    # The saved pilot checkpoint must still carry the rate estimator's state.
    saved = torch.load(checkpoint_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert "rate_estimator_state_dict" in saved["extra"]
    assert saved["extra"]["rate_lambda"] == 0.001


def test_cli_ordinary_resume_still_restores_optimizer_state(tmp_path):
    """The pre-existing --resume behavior must be unchanged when the new flag
    is absent and the optimizer's parameter list still matches."""
    mod = _load_script("train_autoencoder")
    manifest = make_tiny_manifest(tmp_path, num_sequences=6, frames_per_sequence=4, width=32, height=32)
    checkpoint = _cli_checkpoint(tmp_path / "ckpt.pt", epoch=5)
    checkpoint_dir = tmp_path / "resumed"

    exit_code = mod.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--resume", str(checkpoint),
        "--checkpoint-dir", str(checkpoint_dir), "--device", "cpu",
    ])

    assert exit_code == 0
    history = json.loads((checkpoint_dir / "history.json").read_text())
    assert history[-1]["epoch"] == 6
    assert history[-1]["rate_enabled"] is False
