"""Tests for Milestone 9C.1: a separate learning rate for the rate estimator.

WHY THIS EXISTS
----------------
M9C measured that `RateEstimator`'s `loc`/`log_scale` need to move by O(1)-O(5)
to fit the latent, but sharing the model's 1e-4 let them move by at most ~0.05
over a 500-step pilot. They stayed at initialization (`scale` 1.00 -> 1.02-1.05),
so `-log2 P(z)` stayed nearly proportional to `abs(z)` and the rate term acted
as an L1 penalty on latent magnitude rather than a fitted entropy model.

M9C.1 puts the estimator's parameters in their own optimizer parameter group at
`--rate-lr`, leaving the model's group at `--learning-rate`.

Everything here runs on CPU against synthetic data from tests/helpers.py, per
this project's testing standard.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from helpers import TINY_MODEL_KWARGS, make_tiny_calibration, make_tiny_checkpoint, make_tiny_manifest  # noqa: E402

from nvc.evaluation.basic_metrics import mse  # noqa: E402
from nvc.models import BaselineAutoencoder  # noqa: E402
from nvc.training import QuantizationNoise, RateEstimator, save_checkpoint  # noqa: E402
from nvc.utils.config import Config, load_default_config  # noqa: E402

MODEL_LR = 1e-4
RATE_LR = 1e-2


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cli_checkpoint(path: Path, *, epoch: int = 1) -> Path:
    """Built at the CLI's default base_channels=32 so --resume can load it."""
    return make_tiny_checkpoint(path, epoch=epoch, model_kwargs={"base_channels": 32})


def _setup(tmp_path: Path):
    manifest = make_tiny_manifest(tmp_path, num_sequences=8, frames_per_sequence=4, width=32, height=32)
    checkpoint = _cli_checkpoint(tmp_path / "m8_qat.pt", epoch=40)
    calibration = make_tiny_calibration(
        tmp_path / "calib.json", checkpoint_path=checkpoint, bits=4, mode="per_channel",
    )
    return manifest, checkpoint, calibration


def _base_argv(manifest: Path, calibration: Path, checkpoint_dir: Path) -> list[str]:
    return [
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1",
        "--batch-size", "2", "--learning-rate", str(MODEL_LR),
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--qat-enabled", "--qat-bits", "4", "--qat-mode", "per_channel",
        "--qat-calibration", str(calibration),
        "--rate-enabled", "--rate-lambda", "0.001",
        "--checkpoint-dir", str(checkpoint_dir), "--device", "cpu",
    ]


# --- 1-3. The parameter groups carry the right LRs and the right params -----


def test_config_exposes_a_rate_lr_default():
    assert Config().rate_lr == pytest.approx(1e-2)
    assert load_default_config().rate_lr > 0


def test_rate_estimator_parameters_are_exactly_loc_and_log_scale(tmp_path):
    """Guards the claim the parameter-group split depends on: nothing else of
    the estimator's may leak into the rate group (bin_width is a buffer)."""
    _, _, calibration = _setup(tmp_path)
    noise = QuantizationNoise.from_calibration(calibration, bits=4, mode="per_channel")
    estimator = RateEstimator(noise.scale, bits=noise.bits, mode=noise.mode)

    assert [name for name, _ in estimator.named_parameters()] == ["loc", "log_scale"]
    assert "bin_width" in dict(estimator.named_buffers())


def test_optimizer_has_two_groups_with_the_two_distinct_learning_rates(tmp_path, capsys):
    mod = _load_script("train_autoencoder")
    manifest, _, calibration = _setup(tmp_path)

    exit_code = mod.main(
        _base_argv(manifest, calibration, tmp_path / "run") + ["--rate-lr", str(RATE_LR)]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert f"model lr={MODEL_LR}" in out
    assert f"rate estimator (loc/log_scale) lr={RATE_LR}" in out

    saved = torch.load(tmp_path / "run" / "latest.pt", map_location="cpu", weights_only=False)
    groups = saved["optimizer_state_dict"]["param_groups"]
    assert len(groups) == 2, "model and rate parameters must be in separate groups"
    assert groups[0]["lr"] == pytest.approx(MODEL_LR), "the model's LR must be untouched"
    assert groups[1]["lr"] == pytest.approx(RATE_LR)
    # Exactly two parameters in the rate group: loc and log_scale.
    assert len(groups[1]["params"]) == 2
    assert len(groups[0]["params"]) == len(saved["model_state_dict"])


def test_a_non_rate_run_keeps_a_single_parameter_group(tmp_path):
    """Runs without --rate-enabled must be structurally unchanged from every
    checkpoint this script has ever written."""
    mod = _load_script("train_autoencoder")
    manifest, _, _ = _setup(tmp_path)

    exit_code = mod.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1",
        "--batch-size", "2", "--learning-rate", str(MODEL_LR),
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--checkpoint-dir", str(tmp_path / "plain"), "--device", "cpu",
    ])

    assert exit_code == 0
    saved = torch.load(tmp_path / "plain" / "latest.pt", map_location="cpu", weights_only=False)
    assert len(saved["optimizer_state_dict"]["param_groups"]) == 1
    assert saved["optimizer_state_dict"]["param_groups"][0]["lr"] == pytest.approx(MODEL_LR)


# --- 4. Invalid rate_lr is rejected, never silently replaced ----------------


# Passed as a single "--rate-lr=VALUE" token, not a two-token pair: argparse
# treats a bare leading "-" as the start of another flag and rejects negatives
# with "expected one argument" before any validation of ours can run. The "="
# form is how a caller actually supplies a negative number, and it is the form
# that reaches the check being tested here.
@pytest.mark.parametrize(
    "argument",
    ["--rate-lr=0", "--rate-lr=-1e-3", "--rate-lr=nan", "--rate-lr=inf", "--rate-lr=-inf"],
)
def test_cli_rejects_a_non_positive_or_non_finite_rate_lr(tmp_path, capsys, argument):
    mod = _load_script("train_autoencoder")
    manifest, _, calibration = _setup(tmp_path)

    with pytest.raises(SystemExit):
        mod.main(_base_argv(manifest, calibration, tmp_path / "run") + [argument])
    assert "--rate-lr must be finite and > 0" in capsys.readouterr().err


def test_rate_lr_is_never_silently_replaced_by_the_model_lr(tmp_path):
    """A rejected value must stop the run - not quietly fall back to 1e-4,
    which would silently reintroduce the M9C starvation this flag fixes."""
    mod = _load_script("train_autoencoder")
    manifest, _, calibration = _setup(tmp_path)

    with pytest.raises(SystemExit):
        mod.main(_base_argv(manifest, calibration, tmp_path / "run") + ["--rate-lr", "0"])
    assert not (tmp_path / "run" / "latest.pt").exists(), "no checkpoint may be written"


# --- 5. rate_lr is recorded and reproducible -------------------------------


def test_rate_lr_is_recorded_in_history_and_checkpoint_extra(tmp_path):
    mod = _load_script("train_autoencoder")
    manifest, _, calibration = _setup(tmp_path)
    checkpoint_dir = tmp_path / "run"

    mod.main(_base_argv(manifest, calibration, checkpoint_dir) + ["--rate-lr", str(RATE_LR)])

    history = json.loads((checkpoint_dir / "history.json").read_text())
    assert history[-1]["rate_lr"] == pytest.approx(RATE_LR)
    assert history[-1]["rate_lambda"] == pytest.approx(0.001)

    saved = torch.load(checkpoint_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert saved["extra"]["rate_lr"] == pytest.approx(RATE_LR)


def test_history_rate_lr_is_none_when_rate_training_is_off(tmp_path):
    mod = _load_script("train_autoencoder")
    manifest, _, _ = _setup(tmp_path)
    checkpoint_dir = tmp_path / "plain"

    mod.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1",
        "--batch-size", "2", "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--checkpoint-dir", str(checkpoint_dir), "--device", "cpu",
    ])

    history = json.loads((checkpoint_dir / "history.json").read_text())
    assert history[-1]["rate_lr"] is None
    assert history[-1]["rate_enabled"] is False


# --- 6-7. Checkpoint compatibility -----------------------------------------


def test_pre_m9_checkpoint_model_only_resume_still_works(tmp_path):
    """M7/M8 checkpoint -> rate-enabled M9C.1 run."""
    mod = _load_script("train_autoencoder")
    manifest, checkpoint, calibration = _setup(tmp_path)
    checkpoint_dir = tmp_path / "resumed"

    exit_code = mod.main(
        _base_argv(manifest, calibration, checkpoint_dir)
        + ["--rate-lr", str(RATE_LR), "--resume", str(checkpoint), "--resume-model-only"]
    )

    assert exit_code == 0
    history = json.loads((checkpoint_dir / "history.json").read_text())
    assert history[-1]["epoch"] == 41


def test_pre_m9_checkpoint_without_the_flag_reports_the_right_reason(tmp_path, capsys):
    mod = _load_script("train_autoencoder")
    manifest, checkpoint, calibration = _setup(tmp_path)

    exit_code = mod.main(
        _base_argv(manifest, calibration, tmp_path / "run")
        + ["--rate-lr", str(RATE_LR), "--resume", str(checkpoint)]
    )

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "predates --rate-enabled" in stderr
    assert "--resume-model-only" in stderr


def test_single_group_m9a_checkpoint_is_diagnosed_as_such(tmp_path, capsys):
    """An M9A/M9C checkpoint kept model+rate params in ONE group. Resuming it
    into M9C.1's two-group optimizer must say THAT, not 'predates rate'."""
    mod = _load_script("train_autoencoder")
    manifest, _, calibration = _setup(tmp_path)

    # Build a checkpoint with the old single-group layout, by hand.
    noise = QuantizationNoise.from_calibration(calibration, bits=4, mode="per_channel")
    model = BaselineAutoencoder(
        quantization_noise=noise, in_channels=3,
        latent_channels=TINY_MODEL_KWARGS["latent_channels"], base_channels=32,
    )
    estimator = RateEstimator(noise.scale, bits=noise.bits, mode=noise.mode)
    legacy_optimizer = torch.optim.Adam(
        list(model.parameters()) + list(estimator.parameters()), lr=MODEL_LR,
    )
    legacy = tmp_path / "m9a_style.pt"
    save_checkpoint(
        legacy, model=model, optimizer=legacy_optimizer, epoch=45, history=[],
        model_config=model.config_dict(),
        extra={"rate_estimator_state_dict": estimator.state_dict(), "rate_lambda": 0.001},
    )
    saved_groups = torch.load(legacy, map_location="cpu", weights_only=False)
    assert len(saved_groups["optimizer_state_dict"]["param_groups"]) == 1

    exit_code = mod.main(
        _base_argv(manifest, calibration, tmp_path / "run")
        + ["--rate-lr", str(RATE_LR), "--resume", str(legacy)]
    )

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "M9A/M9C" in stderr
    assert "--resume-model-only" in stderr


def test_m9c1_checkpoint_loads_through_the_standard_inference_path(tmp_path):
    """A two-group rate checkpoint must still rebuild via the ordinary
    inference loader, which knows nothing about optimizers or rate."""
    from nvc.training import load_model_from_checkpoint

    mod = _load_script("train_autoencoder")
    manifest, _, calibration = _setup(tmp_path)
    checkpoint_dir = tmp_path / "run"
    mod.main(_base_argv(manifest, calibration, checkpoint_dir) + ["--rate-lr", str(RATE_LR)])

    model, checkpoint = load_model_from_checkpoint(checkpoint_dir / "latest.pt", device="cpu")
    assert model.quantization_noise is None, "inference must not re-attach QAT noise"
    assert checkpoint["extra"]["rate_lr"] == pytest.approx(RATE_LR)
    with torch.no_grad():
        assert model(torch.rand(1, 3, 32, 32)).shape == (1, 3, 32, 32)


def test_m9c1_run_can_resume_its_own_checkpoint_with_optimizer_state(tmp_path):
    """Same-structure resume must still restore the optimizer - the two-group
    layout is self-consistent, so --resume-model-only is not needed here."""
    mod = _load_script("train_autoencoder")
    manifest, _, calibration = _setup(tmp_path)
    first = tmp_path / "first"
    mod.main(_base_argv(manifest, calibration, first) + ["--rate-lr", str(RATE_LR)])

    exit_code = mod.main(
        _base_argv(manifest, calibration, tmp_path / "second")
        + ["--rate-lr", str(RATE_LR), "--resume", str(first / "latest.pt")]
    )

    assert exit_code == 0
    history = json.loads((tmp_path / "second" / "history.json").read_text())
    assert history[-1]["epoch"] == 2


# --- 8. One step actually moves loc/log_scale materially more than before ---


def _one_step_movement(rate_lr: float, calibration: Path, seed: int = 0) -> float:
    """Max abs delta on the estimator's parameters after ONE optimizer step."""
    torch.manual_seed(seed)
    noise = QuantizationNoise.from_calibration(calibration, bits=4, mode="per_channel")
    model = BaselineAutoencoder(quantization_noise=noise, **TINY_MODEL_KWARGS)
    estimator = RateEstimator(noise.scale, bits=noise.bits, mode=noise.mode)
    optimizer = torch.optim.Adam([
        {"params": list(model.parameters()), "lr": MODEL_LR},
        {"params": list(estimator.parameters()), "lr": rate_lr},
    ])

    before = torch.cat([p.detach().flatten().clone() for p in estimator.parameters()])
    batch = torch.rand(2, 3, 32, 32)
    latent = noise.apply(model.encode(batch))
    loss = mse(model.decode(latent), batch) + 0.001 * estimator(latent, 32 * 32)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = torch.cat([p.detach().flatten() for p in estimator.parameters()])
    return (after - before).abs().max().item()


def test_one_step_moves_the_estimator_about_a_hundred_times_further(tmp_path):
    """The quantitative core of M9C.1. Adam's first step is ~lr per parameter,
    so a 100x LR must buy ~100x movement."""
    _, _, calibration = _setup(tmp_path)

    old_movement = _one_step_movement(MODEL_LR, calibration)
    new_movement = _one_step_movement(RATE_LR, calibration)

    assert old_movement == pytest.approx(MODEL_LR, rel=0.05), "M9C's starved step size"
    assert new_movement == pytest.approx(RATE_LR, rel=0.05)
    assert new_movement > 50 * old_movement


def test_model_parameters_move_at_the_model_lr_regardless_of_rate_lr(tmp_path):
    """The split must not leak: raising --rate-lr must not change how far the
    model's own parameters move."""
    _, _, calibration = _setup(tmp_path)

    def model_movement(rate_lr: float) -> float:
        torch.manual_seed(0)
        noise = QuantizationNoise.from_calibration(calibration, bits=4, mode="per_channel")
        model = BaselineAutoencoder(quantization_noise=noise, **TINY_MODEL_KWARGS)
        estimator = RateEstimator(noise.scale, bits=noise.bits, mode=noise.mode)
        optimizer = torch.optim.Adam([
            {"params": list(model.parameters()), "lr": MODEL_LR},
            {"params": list(estimator.parameters()), "lr": rate_lr},
        ])
        before = torch.cat([p.detach().flatten().clone() for p in model.parameters()])
        batch = torch.rand(2, 3, 32, 32)
        latent = noise.apply(model.encode(batch))
        loss = mse(model.decode(latent), batch) + 0.001 * estimator(latent, 32 * 32)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        after = torch.cat([p.detach().flatten() for p in model.parameters()])
        return (after - before).abs().max().item()

    assert model_movement(RATE_LR) == pytest.approx(model_movement(MODEL_LR), rel=1e-6)
    assert model_movement(RATE_LR) == pytest.approx(MODEL_LR, rel=0.05)


# --- 9. lambda=0 still contributes exactly nothing to model gradients -------


def test_lambda_zero_gives_the_model_exactly_zero_rate_gradient_under_a_high_rate_lr(tmp_path):
    """A higher rate LR must not create a rate contribution where lambda=0
    guarantees none - the estimator's LR scales its OWN step, not the model's."""
    _, _, calibration = _setup(tmp_path)
    noise = QuantizationNoise.from_calibration(calibration, bits=4, mode="per_channel")
    batch = torch.rand(2, 3, 32, 32)

    torch.manual_seed(0)
    model = BaselineAutoencoder(**TINY_MODEL_KWARGS)
    estimator = RateEstimator(noise.scale, bits=noise.bits, mode=noise.mode)
    latent = model.encode(batch)
    loss = mse(model.decode(latent), batch) + 0.0 * estimator(latent, 32 * 32)
    model.zero_grad()
    loss.backward()
    with_rate = {name: p.grad.detach().clone() for name, p in model.named_parameters()}

    torch.manual_seed(0)
    plain_model = BaselineAutoencoder(**TINY_MODEL_KWARGS)
    plain_loss = mse(plain_model(batch), batch)
    plain_model.zero_grad()
    plain_loss.backward()

    for name, parameter in plain_model.named_parameters():
        assert torch.equal(with_rate[name], parameter.grad), f"{name} differs at lambda=0"


def test_lambda_zero_leaves_the_estimator_unmoved_even_at_a_high_rate_lr(tmp_path):
    """Documents the M9C observation, now under M9C.1: `0.0 * rate` yields
    exactly zero gradient, so no LR can make the estimator learn at lambda=0."""
    _, _, calibration = _setup(tmp_path)

    assert _one_step_movement(RATE_LR, calibration) > 0  # sanity: lambda>0 does move it

    torch.manual_seed(0)
    noise = QuantizationNoise.from_calibration(calibration, bits=4, mode="per_channel")
    model = BaselineAutoencoder(quantization_noise=noise, **TINY_MODEL_KWARGS)
    estimator = RateEstimator(noise.scale, bits=noise.bits, mode=noise.mode)
    optimizer = torch.optim.Adam([
        {"params": list(model.parameters()), "lr": MODEL_LR},
        {"params": list(estimator.parameters()), "lr": RATE_LR},
    ])
    batch = torch.rand(2, 3, 32, 32)
    latent = noise.apply(model.encode(batch))
    loss = mse(model.decode(latent), batch) + 0.0 * estimator(latent, 32 * 32)
    optimizer.zero_grad()
    loss.backward()

    assert estimator.loc.grad.abs().max().item() == 0.0
    assert estimator.log_scale.grad.abs().max().item() == 0.0


# --- 10. No NaN/Inf --------------------------------------------------------


@pytest.mark.parametrize("rate_lr", [1e-3, 1e-2, 1e-1])
def test_no_nan_or_inf_after_several_steps_at_a_range_of_rate_lrs(tmp_path, rate_lr):
    _, _, calibration = _setup(tmp_path)
    torch.manual_seed(0)
    noise = QuantizationNoise.from_calibration(calibration, bits=4, mode="per_channel")
    model = BaselineAutoencoder(quantization_noise=noise, **TINY_MODEL_KWARGS)
    estimator = RateEstimator(noise.scale, bits=noise.bits, mode=noise.mode)
    optimizer = torch.optim.Adam([
        {"params": list(model.parameters()), "lr": MODEL_LR},
        {"params": list(estimator.parameters()), "lr": rate_lr},
    ])

    for _ in range(20):
        batch = torch.rand(2, 3, 32, 32)
        latent = noise.apply(model.encode(batch))
        rate = estimator(latent, 32 * 32)
        loss = mse(model.decode(latent), batch) + 0.01 * rate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert math.isfinite(loss.item())
        assert math.isfinite(rate.item())
        assert rate.item() >= 0

    for name, parameter in estimator.named_parameters():
        assert torch.isfinite(parameter).all(), f"{name} went non-finite"
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), f"{name} went non-finite"
