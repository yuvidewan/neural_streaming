"""Tests for Milestone 9A: nvc.training.rate_estimator.RateEstimator and
its trainer/checkpoint/CLI integration.

Organized the same way tests/test_quantization_aware_training.py is
(construction/validation, from_calibration, apply/forward behavior,
model+trainer integration, checkpoint compatibility, config/CLI) - this is
the sibling milestone's test file and deliberately mirrors its shape.

Nothing here needs a GPU or real Vimeo/DAVIS data - everything is built via
tests/helpers.py's real code paths (make_tiny_checkpoint, make_tiny_manifest)
or hand-built tiny tensors, per this project's testing standard.
"""

from __future__ import annotations

import copy
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
from nvc.training import QuantizationNoise, RateEstimator  # noqa: E402
from nvc.training.checkpoint import load_model_from_checkpoint, save_checkpoint  # noqa: E402
from nvc.training.trainer import (  # noqa: E402
    train_one_epoch,
    train_one_epoch_with_rate,
    validate_one_epoch_with_rate,
)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bin_width(num_channels: int = 4, value: float = 0.5) -> torch.Tensor:
    return torch.full((1, num_channels, 1, 1), value)


# --- Construction / validation ---------------------------------------------


def test_rate_estimator_rejects_non_positive_bin_width():
    with pytest.raises(ValueError, match="strictly positive"):
        RateEstimator(torch.zeros(1, 4, 1, 1), bits=8, mode="per_channel")


def test_rate_estimator_rejects_non_finite_bin_width():
    bad = _bin_width()
    bad[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        RateEstimator(bad, bits=8, mode="per_channel")


def test_rate_estimator_rejects_wrong_rank_bin_width():
    with pytest.raises(ValueError, match="shape"):
        RateEstimator(torch.ones(4), bits=8, mode="per_channel")


def test_rate_estimator_rejects_out_of_range_bits():
    with pytest.raises(ValueError, match="bits"):
        RateEstimator(_bin_width(), bits=17, mode="per_channel")
    with pytest.raises(ValueError, match="bits"):
        RateEstimator(_bin_width(), bits=0, mode="per_channel")


def test_rate_estimator_has_learnable_loc_and_log_scale():
    estimator = RateEstimator(_bin_width(), bits=8, mode="per_channel")
    names = {name for name, _ in estimator.named_parameters()}
    assert names == {"loc", "log_scale"}
    assert estimator.loc.requires_grad
    assert estimator.log_scale.requires_grad


def test_rate_estimator_is_not_in_baseline_autoencoder_state_dict():
    # Never attached to the model - inference must never need it.
    model = BaselineAutoencoder(**TINY_MODEL_KWARGS)
    assert not any("rate" in key.lower() for key in model.state_dict())


# --- from_calibration --------------------------------------------------


def test_from_calibration_loads_bin_width_bits_and_mode(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint, bits=4, mode="per_channel")
    estimator = RateEstimator.from_calibration(calibration)
    assert estimator.bits == 4
    assert estimator.mode == "per_channel"
    assert estimator.bin_width.shape == (1, TINY_MODEL_KWARGS["latent_channels"], 1, 1)


def test_from_calibration_rejects_bit_depth_mismatch(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint, bits=4)
    with pytest.raises(ValueError, match="bit"):
        RateEstimator.from_calibration(calibration, bits=8)


def test_from_calibration_rejects_mode_mismatch(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint, mode="per_channel")
    with pytest.raises(ValueError, match="mode"):
        RateEstimator.from_calibration(calibration, mode="global")


def test_from_calibration_rejects_a_non_train_split(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(
        tmp_path / "calib.json", checkpoint_path=checkpoint, calibration_split="test",
    )
    with pytest.raises(ValueError, match="train"):
        RateEstimator.from_calibration(calibration)


def test_from_calibration_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RateEstimator.from_calibration(tmp_path / "nope.json")


# --- Rate computation: finiteness, non-negativity, stability -----------


@pytest.mark.parametrize("z_value", [0.0, 1.0, -1.0, 5.0, -5.0])
def test_rate_is_finite_for_normal_latent_values(z_value):
    estimator = RateEstimator(_bin_width(), bits=8, mode="per_channel")
    z = torch.full((2, 4, 3, 3), z_value)
    bpp = estimator(z, image_pixels=64 * 64)
    assert torch.isfinite(bpp)
    assert bpp >= 0


@pytest.mark.parametrize("z_value", [1e3, -1e3, 1e6, -1e6, 1e12, -1e12])
def test_rate_is_finite_for_extreme_latent_values(z_value):
    estimator = RateEstimator(_bin_width(), bits=8, mode="per_channel")
    z = torch.full((1, 4, 2, 2), z_value)
    bpp = estimator(z, image_pixels=64 * 64)
    assert torch.isfinite(bpp)
    assert bpp >= 0


def test_rate_is_finite_for_very_small_bin_width():
    estimator = RateEstimator(_bin_width(value=1e-8), bits=8, mode="per_channel")
    z = torch.randn(2, 4, 4, 4)
    bpp = estimator(z, image_pixels=64 * 64)
    assert torch.isfinite(bpp)
    assert bpp >= 0


def test_rate_is_finite_for_large_bin_width():
    estimator = RateEstimator(_bin_width(value=1e8), bits=8, mode="per_channel")
    z = torch.randn(2, 4, 4, 4)
    bpp = estimator(z, image_pixels=64 * 64)
    assert torch.isfinite(bpp)
    assert bpp >= 0


@pytest.mark.parametrize("log_scale_value", [-30.0, 30.0])
def test_rate_is_finite_for_extreme_learned_scale(log_scale_value):
    # very small (-30 -> scale ~ 9e-14) and very large (30 -> scale ~ 1e13)
    # learned scale - both must stay finite through exp()/expm1().
    estimator = RateEstimator(_bin_width(), bits=8, mode="per_channel")
    with torch.no_grad():
        estimator.log_scale.fill_(log_scale_value)
    z = torch.randn(2, 4, 4, 4)
    bpp = estimator(z, image_pixels=64 * 64)
    assert torch.isfinite(bpp)
    assert bpp >= 0


def test_rate_is_always_non_negative_over_many_random_cases():
    torch.manual_seed(0)
    estimator = RateEstimator(_bin_width(num_channels=8, value=0.3), bits=8, mode="per_channel")
    for _ in range(20):
        z = torch.randn(3, 8, 4, 4) * torch.rand(1).item() * 100
        bpp = estimator(z, image_pixels=64 * 64)
        assert bpp >= 0
        assert torch.isfinite(bpp)


def test_rate_bpp_normalization_matches_hand_computed_value():
    # z exactly at loc=0, one channel, one pixel, bin width w -> the bin
    # probability is the exact Laplace(0, scale=1) mass in [-w/2, w/2]:
    # P = 1 - exp(-w/2) (by symmetry: 2 * 0.5 * (1 - exp(-(w/2)/1))).
    bin_width = 2.0
    estimator = RateEstimator(_bin_width(num_channels=1, value=bin_width), bits=8, mode="per_channel")
    z = torch.zeros(1, 1, 1, 1)  # single latent element, at loc=0, scale=exp(0)=1
    expected_prob = 1.0 - math.exp(-(bin_width / 2) / 1.0)
    expected_bits = -math.log2(expected_prob)
    image_pixels = 100  # one latent element -> expected_bits total, /100 pixels
    bpp = estimator(z, image_pixels=image_pixels)
    assert bpp.item() == pytest.approx(expected_bits / image_pixels, rel=1e-5)


# --- Gradients -----------------------------------------------------------


def test_loc_receives_gradient():
    estimator = RateEstimator(_bin_width(), bits=8, mode="per_channel")
    z = torch.randn(2, 4, 3, 3)
    estimator(z, image_pixels=256).backward()
    assert estimator.loc.grad is not None
    assert torch.isfinite(estimator.loc.grad).all()
    assert (estimator.loc.grad != 0).any()


def test_log_scale_receives_gradient():
    estimator = RateEstimator(_bin_width(), bits=8, mode="per_channel")
    z = torch.randn(2, 4, 3, 3)
    estimator(z, image_pixels=256).backward()
    assert estimator.log_scale.grad is not None
    assert torch.isfinite(estimator.log_scale.grad).all()
    assert (estimator.log_scale.grad != 0).any()


def test_latent_input_receives_gradient():
    estimator = RateEstimator(_bin_width(), bits=8, mode="per_channel")
    z = torch.randn(2, 4, 3, 3, requires_grad=True)
    estimator(z, image_pixels=256).backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert (z.grad != 0).any()


def test_changing_loc_changes_estimated_rate():
    estimator = RateEstimator(_bin_width(), bits=8, mode="per_channel")
    z = torch.randn(2, 4, 3, 3)
    with torch.no_grad():
        before = estimator(z, image_pixels=256).clone()
        estimator.loc.add_(5.0)
        after = estimator(z, image_pixels=256)
    assert not torch.equal(before, after)


def test_changing_log_scale_changes_estimated_rate():
    estimator = RateEstimator(_bin_width(), bits=8, mode="per_channel")
    z = torch.randn(2, 4, 3, 3)
    with torch.no_grad():
        before = estimator(z, image_pixels=256).clone()
        estimator.log_scale.add_(2.0)
        after = estimator(z, image_pixels=256)
    assert not torch.equal(before, after)


def test_changing_latent_values_changes_estimated_rate():
    estimator = RateEstimator(_bin_width(), bits=8, mode="per_channel")
    torch.manual_seed(1)
    z1 = torch.randn(2, 4, 3, 3)
    z2 = z1 + 3.0
    with torch.no_grad():
        assert not torch.equal(estimator(z1, image_pixels=256), estimator(z2, image_pixels=256))


# --- Rate gradient test (lambda > 0) contributes to model/latent -------


def test_rate_contributes_nonzero_gradient_to_latent_when_lambda_positive():
    estimator = RateEstimator(_bin_width(), bits=8, mode="per_channel")
    z_distortion_only = torch.randn(2, 4, 3, 3, requires_grad=True)
    z_with_rate = z_distortion_only.detach().clone().requires_grad_(True)

    # A trivial "distortion" that doesn't depend on z at all, so any
    # gradient on z_with_rate is attributable ONLY to the rate term.
    lambda_rate = 0.5
    rate = estimator(z_with_rate, image_pixels=256)
    (lambda_rate * rate).backward()

    assert z_with_rate.grad is not None
    assert torch.isfinite(z_with_rate.grad).all()
    assert (z_with_rate.grad != 0).any()


def test_rate_estimator_parameters_receive_gradient_when_lambda_positive():
    estimator = RateEstimator(_bin_width(), bits=8, mode="per_channel")
    z = torch.randn(2, 4, 3, 3)
    lambda_rate = 0.7
    (lambda_rate * estimator(z, image_pixels=256)).backward()
    assert (estimator.loc.grad != 0).any()
    assert (estimator.log_scale.grad != 0).any()


# --- Lambda handling -------------------------------------------------------


def test_check_lambda_rate_rejects_negative():
    from nvc.training.trainer import _check_lambda_rate
    with pytest.raises(ValueError, match="lambda_rate"):
        _check_lambda_rate(-0.1)


def test_check_lambda_rate_rejects_nan_and_inf():
    from nvc.training.trainer import _check_lambda_rate
    with pytest.raises(ValueError, match="lambda_rate"):
        _check_lambda_rate(float("nan"))
    with pytest.raises(ValueError, match="lambda_rate"):
        _check_lambda_rate(float("inf"))


def test_check_lambda_rate_accepts_zero():
    from nvc.training.trainer import _check_lambda_rate
    _check_lambda_rate(0.0)  # must not raise


def test_different_lambda_values_produce_different_total_loss(tmp_path):
    manifest = make_tiny_manifest(tmp_path, num_sequences=2, frames_per_sequence=4, width=32, height=32)
    from nvc.data.loaders import create_train_loader
    loader = create_train_loader(manifest, batch_size=2, seed=0)

    def _run(lambda_rate):
        torch.manual_seed(0)
        model = BaselineAutoencoder(**TINY_MODEL_KWARGS)
        estimator = RateEstimator(_bin_width(num_channels=TINY_MODEL_KWARGS["latent_channels"]), bits=8, mode="per_channel")
        optimizer = torch.optim.Adam(list(model.parameters()) + list(estimator.parameters()), lr=1e-4)
        return train_one_epoch_with_rate(
            model, loader, optimizer, torch.device("cpu"),
            rate_estimator=estimator, lambda_rate=lambda_rate, max_batches=1,
        )

    low = _run(0.0)
    high = _run(10.0)
    assert low["loss"] != high["loss"]
    # Distortion after ONE step differs too (different loss landscape ->
    # different gradient -> different parameter update), but both must stay finite.
    assert math.isfinite(low["loss"]) and math.isfinite(high["loss"])


# --- The critical lambda=0 equivalence test -------------------------------


def test_lambda_zero_matches_distortion_only_trainer_exactly(tmp_path):
    """A: the pre-existing, untouched train_one_epoch.
    B: train_one_epoch_with_rate at lambda_rate=0.0.
    Same initial model weights (deep copy), same optimizer settings, same
    batch, no QAT (no randomness involved at all) - so A and B's
    reconstruction, distortion, total loss, MODEL gradients, and the
    resulting parameters after one optimizer step must match exactly.
    """
    manifest = make_tiny_manifest(tmp_path, num_sequences=2, frames_per_sequence=4, width=32, height=32)
    from nvc.data.loaders import create_train_loader
    loader = create_train_loader(manifest, batch_size=2, seed=0)
    batch = next(iter(loader))

    torch.manual_seed(123)
    model_a = BaselineAutoencoder(**TINY_MODEL_KWARGS)
    model_b = copy.deepcopy(model_a)
    assert all(torch.equal(pa, pb) for pa, pb in zip(model_a.parameters(), model_b.parameters()))

    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)

    # A: plain forward, for a direct reconstruction/distortion comparison
    # independent of the trainer functions' own bookkeeping.
    model_a.train()
    reconstruction_a = model_a(batch)
    distortion_a = mse(reconstruction_a, batch)

    # B: the new path, still evaluating the rate estimator (numerical
    # stability requirement) but at lambda_rate=0.0.
    estimator = RateEstimator(_bin_width(num_channels=TINY_MODEL_KWARGS["latent_channels"]), bits=8, mode="per_channel")
    model_b.train()
    latent_b = model_b.encode(batch)
    assert model_b.quantization_noise is None  # no QAT in this test - deterministic
    rate_b = estimator(latent_b, image_pixels=batch.shape[-2] * batch.shape[-1])
    reconstruction_b = model_b.decode(latent_b)
    distortion_b = mse(reconstruction_b, batch)
    total_loss_b = distortion_b + 0.0 * rate_b

    assert torch.equal(reconstruction_a, reconstruction_b)
    assert torch.equal(distortion_a, distortion_b)
    assert total_loss_b.item() == distortion_b.item()  # lambda=0 -> total == distortion, exactly

    distortion_a.backward()
    total_loss_b.backward()
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa.grad, pb.grad)

    optimizer_a.step()
    optimizer_b.step()
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)

    # Now via the actual trainer functions end-to-end (not the manual
    # replication above), confirming the same equivalence at the
    # public-API level with a fresh pair of models.
    #
    # Two SEPARATE loader instances, not one reused across two `for batch
    # in loader` consumptions: create_train_loader's shuffling Generator
    # advances its internal state on every iteration, so the SAME loader
    # object gives a DIFFERENT order on its second pass (this is correct,
    # ordinary epoch-to-epoch behavior) - two loaders built from the same
    # seed each start fresh and give the same first-epoch order as each other.
    loader_c = create_train_loader(manifest, batch_size=2, seed=0)
    loader_d = create_train_loader(manifest, batch_size=2, seed=0)

    torch.manual_seed(456)
    model_c = BaselineAutoencoder(**TINY_MODEL_KWARGS)
    model_d = copy.deepcopy(model_c)
    optimizer_c = torch.optim.Adam(model_c.parameters(), lr=1e-3)
    estimator_d = RateEstimator(_bin_width(num_channels=TINY_MODEL_KWARGS["latent_channels"]), bits=8, mode="per_channel")
    optimizer_d = torch.optim.Adam(list(model_d.parameters()) + list(estimator_d.parameters()), lr=1e-3)

    metrics_c = train_one_epoch(model_c, loader_c, optimizer_c, torch.device("cpu"), max_batches=1)
    metrics_d = train_one_epoch_with_rate(
        model_d, loader_d, optimizer_d, torch.device("cpu"),
        rate_estimator=estimator_d, lambda_rate=0.0, max_batches=1,
    )
    assert metrics_c["loss"] == pytest.approx(metrics_d["loss"], abs=1e-7)
    assert metrics_c["loss"] == pytest.approx(metrics_d["distortion"], abs=1e-7)
    for pc, pd in zip(model_c.parameters(), model_d.parameters()):
        assert torch.allclose(pc, pd, atol=1e-6)


# --- QAT interaction -------------------------------------------------------


def test_rate_training_without_qat(tmp_path):
    manifest = make_tiny_manifest(tmp_path, num_sequences=2, frames_per_sequence=4, width=32, height=32)
    from nvc.data.loaders import create_train_loader
    loader = create_train_loader(manifest, batch_size=2, seed=0)

    model = BaselineAutoencoder(**TINY_MODEL_KWARGS)  # quantization_noise=None
    estimator = RateEstimator(_bin_width(num_channels=TINY_MODEL_KWARGS["latent_channels"]), bits=8, mode="per_channel")
    optimizer = torch.optim.Adam(list(model.parameters()) + list(estimator.parameters()), lr=1e-4)

    metrics = train_one_epoch_with_rate(
        model, loader, optimizer, torch.device("cpu"),
        rate_estimator=estimator, lambda_rate=0.1, max_batches=2,
    )
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["distortion"])
    assert math.isfinite(metrics["rate"])
    assert metrics["rate"] >= 0


def test_rate_training_with_qat_reuses_the_same_scale(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint, bits=4, mode="per_channel")
    quantization_noise = QuantizationNoise.from_calibration(calibration)

    # This is exactly what scripts/train_autoencoder.py does when both
    # --qat-enabled and --rate-enabled are passed: NOT a second,
    # independently loaded calibration - the literal same tensor.
    estimator = RateEstimator(quantization_noise.scale, bits=quantization_noise.bits, mode=quantization_noise.mode)

    assert torch.equal(estimator.bin_width, quantization_noise.scale)
    assert estimator.bits == quantization_noise.bits
    assert estimator.mode == quantization_noise.mode


def test_train_one_epoch_with_rate_works_with_qat_attached(tmp_path):
    manifest = make_tiny_manifest(tmp_path, num_sequences=2, frames_per_sequence=4, width=32, height=32)
    from nvc.data.loaders import create_train_loader
    loader = create_train_loader(manifest, batch_size=2, seed=0)

    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint, bits=4, mode="per_channel")
    quantization_noise = QuantizationNoise.from_calibration(calibration)

    model = BaselineAutoencoder(**TINY_MODEL_KWARGS, quantization_noise=quantization_noise)
    estimator = RateEstimator(quantization_noise.scale, bits=quantization_noise.bits, mode=quantization_noise.mode)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(estimator.parameters()), lr=1e-4)

    train_metrics = train_one_epoch_with_rate(
        model, loader, optimizer, torch.device("cpu"),
        rate_estimator=estimator, lambda_rate=0.05, max_batches=2,
    )
    val_metrics = validate_one_epoch_with_rate(
        model, loader, torch.device("cpu"), rate_estimator=estimator, lambda_rate=0.05, max_batches=2,
    )
    for metrics in (train_metrics, val_metrics):
        assert math.isfinite(metrics["loss"])
        assert math.isfinite(metrics["distortion"])
        assert math.isfinite(metrics["rate"])
    assert "psnr" in val_metrics


def test_existing_qat_behavior_remains_intact_when_rate_is_not_used(tmp_path):
    # train_one_epoch itself (QAT's own existing integration) must be
    # completely untouched by adding train_one_epoch_with_rate alongside it.
    manifest = make_tiny_manifest(tmp_path, num_sequences=6, frames_per_sequence=4, width=32, height=32)
    from nvc.data.loaders import create_train_loader, create_val_loader
    train_loader = create_train_loader(manifest, batch_size=2, seed=0)
    val_loader = create_val_loader(manifest, batch_size=2)

    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint, bits=4, mode="per_channel")
    quantization_noise = QuantizationNoise.from_calibration(calibration)
    model = BaselineAutoencoder(**TINY_MODEL_KWARGS, quantization_noise=quantization_noise)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    from nvc.training.trainer import validate_one_epoch
    train_metrics = train_one_epoch(model, train_loader, optimizer, torch.device("cpu"), max_batches=2)
    val_metrics = validate_one_epoch(model, val_loader, torch.device("cpu"), max_batches=2)
    assert set(train_metrics) == {"loss"}
    assert set(val_metrics) == {"loss", "psnr"}


# --- Checkpoint compatibility ----------------------------------------------


def test_save_checkpoint_without_extra_is_byte_identical_in_structure(tmp_path):
    model = BaselineAutoencoder(**TINY_MODEL_KWARGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, epoch=1, history=[], model_config=model.config_dict())
    raw = torch.load(path, weights_only=False)
    assert "extra" not in raw
    assert set(raw.keys()) == {"model_state_dict", "optimizer_state_dict", "epoch", "history", "model_config"}


def test_old_checkpoint_without_extra_loads_normally(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    model, loaded = load_model_from_checkpoint(checkpoint)
    assert isinstance(model, BaselineAutoencoder)
    assert "extra" not in loaded


def test_rate_trained_checkpoint_loads_through_standard_inference_path(tmp_path):
    model = BaselineAutoencoder(**TINY_MODEL_KWARGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    estimator = RateEstimator(_bin_width(num_channels=TINY_MODEL_KWARGS["latent_channels"]), bits=8, mode="per_channel")
    path = tmp_path / "rate_ckpt.pt"

    save_checkpoint(
        path, model=model, optimizer=optimizer, epoch=1, history=[], model_config=model.config_dict(),
        extra={"rate_estimator_state_dict": estimator.state_dict(), "rate_lambda": 0.01},
    )

    # Inference does not instantiate or require RateEstimator at all.
    loaded_model, checkpoint = load_model_from_checkpoint(path)
    assert isinstance(loaded_model, BaselineAutoencoder)
    assert "extra" in checkpoint
    assert "rate_estimator_state_dict" in checkpoint["extra"]
    # Model behaves as an ordinary checkpoint for inference.
    x = torch.rand(1, TINY_MODEL_KWARGS["in_channels"], 32, 32)
    with torch.no_grad():
        reconstruction = loaded_model(x)
    assert reconstruction.shape == x.shape


def test_rate_estimator_state_round_trips_from_checkpoint_extra(tmp_path):
    model = BaselineAutoencoder(**TINY_MODEL_KWARGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    estimator = RateEstimator(_bin_width(num_channels=TINY_MODEL_KWARGS["latent_channels"]), bits=8, mode="per_channel")
    with torch.no_grad():
        estimator.loc.fill_(3.5)
        estimator.log_scale.fill_(-1.25)
    path = tmp_path / "rate_ckpt.pt"
    save_checkpoint(
        path, model=model, optimizer=optimizer, epoch=1, history=[], model_config=model.config_dict(),
        extra={"rate_estimator_state_dict": estimator.state_dict(), "rate_lambda": 0.02},
    )

    _, checkpoint = load_model_from_checkpoint(path)
    restored = RateEstimator(_bin_width(num_channels=TINY_MODEL_KWARGS["latent_channels"]), bits=8, mode="per_channel")
    restored.load_state_dict(checkpoint["extra"]["rate_estimator_state_dict"])
    assert torch.equal(restored.loc, estimator.loc)
    assert torch.equal(restored.log_scale, estimator.log_scale)
    assert checkpoint["extra"]["rate_lambda"] == 0.02


@pytest.mark.parametrize("checkpoint_path", [
    "outputs/checkpoints/vimeo_epoch17_best.pt",
    "outputs/qat_combined/checkpoints_qat_control/best.pt",
    "outputs/qat_combined/checkpoints_qat_noise/best.pt",
])
def test_existing_m7_m8_checkpoints_remain_loadable(checkpoint_path):
    path = Path(checkpoint_path)
    if not path.is_file():
        pytest.skip(f"{path} not present on this machine")
    # Explicit CPU map_location: these real checkpoints were saved from a
    # CUDA machine (see MILESTONE_8_RESULTS.md) - map_location=None would
    # try to restore CUDA tensors and fail on this CPU-only dev machine,
    # unrelated to anything M9A changed.
    model, checkpoint = load_model_from_checkpoint(path, device=torch.device("cpu"))
    assert isinstance(model, BaselineAutoencoder)
    assert "extra" not in checkpoint  # M7/M8 checkpoints predate M9A, must be unaffected


# --- Config / CLI -----------------------------------------------------------


def test_config_rate_defaults_preserve_baseline_behavior():
    from nvc.utils.config import Config
    config = Config()
    assert config.rate_enabled is False
    assert config.rate_lambda == 0.0
    assert config.rate_calibration_path is None


def test_config_from_json_coerces_rate_calibration_path_to_a_path(tmp_path):
    from nvc.utils.config import Config
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"rate_enabled": True, "rate_calibration_path": "calib.json"}))
    config = Config.from_json(config_path)
    assert isinstance(config.rate_calibration_path, Path)


def test_config_from_json_accepts_a_null_rate_calibration_path(tmp_path):
    from nvc.utils.config import Config
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"rate_calibration_path": None}))
    config = Config.from_json(config_path)
    assert config.rate_calibration_path is None


def test_cli_requires_calibration_when_rate_enabled_without_qat(tmp_path, capsys):
    mod = _load_script("train_autoencoder")
    manifest = make_tiny_manifest(tmp_path, num_sequences=2, frames_per_sequence=4, width=32, height=32)
    with pytest.raises(SystemExit):
        mod.main([
            "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1",
            "--rate-enabled", "--checkpoint-dir", str(tmp_path / "ckpt"),
        ])
    assert "rate-calibration" in capsys.readouterr().err


def test_cli_rejects_rate_calibration_together_with_qat(tmp_path, capsys):
    mod = _load_script("train_autoencoder")
    manifest = make_tiny_manifest(tmp_path, num_sequences=2, frames_per_sequence=4, width=32, height=32)
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint, bits=4)
    with pytest.raises(SystemExit):
        mod.main([
            "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1",
            "--qat-enabled", "--qat-bits", "4", "--qat-calibration", str(calibration),
            "--rate-enabled", "--rate-calibration", str(calibration),
            "--checkpoint-dir", str(tmp_path / "ckpt"),
        ])
    assert "--rate-calibration" in capsys.readouterr().err


def test_cli_rejects_negative_lambda(tmp_path, capsys):
    mod = _load_script("train_autoencoder")
    manifest = make_tiny_manifest(tmp_path, num_sequences=2, frames_per_sequence=4, width=32, height=32)
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint, bits=4)
    with pytest.raises(SystemExit):
        mod.main([
            "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1",
            "--rate-enabled", "--rate-lambda", "-1.0", "--rate-calibration", str(calibration),
            "--checkpoint-dir", str(tmp_path / "ckpt"),
        ])
    assert "rate-lambda" in capsys.readouterr().err


def test_cli_reports_missing_rate_calibration_file(tmp_path, capsys):
    mod = _load_script("train_autoencoder")
    manifest = make_tiny_manifest(tmp_path, num_sequences=2, frames_per_sequence=4, width=32, height=32)
    exit_code = mod.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1",
        "--rate-enabled", "--rate-calibration", str(tmp_path / "nope.json"),
        "--checkpoint-dir", str(tmp_path / "ckpt"),
    ])
    assert exit_code != 0
    assert "not found" in capsys.readouterr().err


def test_cli_end_to_end_rate_only_smoke(tmp_path):
    mod = _load_script("train_autoencoder")
    manifest = make_tiny_manifest(tmp_path, num_sequences=6, frames_per_sequence=4, width=32, height=32)
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint, bits=4, mode="per_channel")
    checkpoint_dir = tmp_path / "rate_run"

    exit_code = mod.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--rate-enabled", "--rate-lambda", "0.01", "--rate-calibration", str(calibration),
        "--checkpoint-dir", str(checkpoint_dir), "--device", "cpu",
    ])
    assert exit_code == 0
    history = json.loads((checkpoint_dir / "history.json").read_text())
    assert history[-1]["rate_enabled"] is True
    assert history[-1]["rate_lambda"] == 0.01
    assert history[-1]["train_rate_bpp"] is not None

    # And it still loads through the ordinary inference path.
    model, checkpoint_doc = load_model_from_checkpoint(checkpoint_dir / "best.pt")
    assert isinstance(model, BaselineAutoencoder)
    assert "rate_estimator_state_dict" in checkpoint_doc["extra"]


def test_cli_end_to_end_qat_plus_rate_smoke(tmp_path):
    mod = _load_script("train_autoencoder")
    manifest = make_tiny_manifest(tmp_path, num_sequences=6, frames_per_sequence=4, width=32, height=32)
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint, bits=4, mode="per_channel")
    checkpoint_dir = tmp_path / "qat_rate_run"

    exit_code = mod.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "1",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--qat-enabled", "--qat-bits", "4", "--qat-mode", "per_channel", "--qat-calibration", str(calibration),
        "--rate-enabled", "--rate-lambda", "0.01",
        "--checkpoint-dir", str(checkpoint_dir), "--device", "cpu",
    ])
    assert exit_code == 0
    history = json.loads((checkpoint_dir / "history.json").read_text())
    assert history[-1]["qat_enabled"] is True
    assert history[-1]["rate_enabled"] is True
