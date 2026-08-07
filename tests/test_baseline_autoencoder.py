"""Tests for Milestone 4: BaselineAutoencoder (Encoder/Decoder), the
training engine (train/validate epoch), checkpointing + resume, and
reconstruction-metric integration.

Uses tiny synthetic tensors and a tiny manifest-backed dataset (via
helpers.make_tiny_manifest, same fixture as Milestone 3) so these tests
stay fast and never touch a real dataset or GPU.
"""

from __future__ import annotations

import torch

from nvc.evaluation.basic_metrics import mse, psnr
from nvc.data.loaders import create_train_loader, create_val_loader
from nvc.models import BaselineAutoencoder, Decoder, Encoder
from nvc.training import (
    load_checkpoint,
    resume_training_state,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)

from helpers import make_tiny_manifest

# Tiny architecture config shared by shape/gradient tests: 32x32 input (the
# smallest size that is both divisible by 16 and gives a non-degenerate
# 2x2 latent), few channels, so these tests run in milliseconds on CPU.
_TINY_KW = {"in_channels": 3, "latent_channels": 4, "base_channels": 8}


# --- Encoder / Decoder / BaselineAutoencoder shapes ---


def test_encoder_output_shape():
    encoder = Encoder(**_TINY_KW)
    x = torch.rand(2, 3, 32, 32)

    z = encoder(x)

    assert z.shape == (2, 4, 2, 2)


def test_decoder_output_shape():
    decoder = Decoder(out_channels=3, latent_channels=4, base_channels=8)
    z = torch.rand(2, 4, 2, 2)

    x = decoder(z)

    assert x.shape == (2, 3, 32, 32)


def test_autoencoder_forward_shape_matches_input():
    model = BaselineAutoencoder(**_TINY_KW)
    x = torch.rand(2, 3, 32, 32)

    reconstruction = model(x)

    assert reconstruction.shape == x.shape


def test_autoencoder_output_in_unit_range():
    model = BaselineAutoencoder(**_TINY_KW)
    x = torch.rand(2, 3, 32, 32)

    reconstruction = model(x)

    assert reconstruction.min() >= 0.0
    assert reconstruction.max() <= 1.0


def test_autoencoder_encode_decode_interface_matches_forward():
    model = BaselineAutoencoder(**_TINY_KW)
    x = torch.rand(2, 3, 32, 32)

    z = model.encode(x)
    reconstruction_via_decode = model.decode(z)
    reconstruction_via_forward = model(x)

    assert z.shape == (2, 4, 2, 2)
    assert torch.equal(reconstruction_via_decode, reconstruction_via_forward)


def test_encoder_rejects_dimensions_not_divisible_by_16():
    encoder = Encoder(**_TINY_KW)
    x = torch.rand(1, 3, 30, 30)

    try:
        encoder(x)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-divisible input dimensions")


def test_autoencoder_config_dict_roundtrip():
    model = BaselineAutoencoder(**_TINY_KW)
    rebuilt = BaselineAutoencoder(**model.config_dict())

    x = torch.rand(1, 3, 32, 32)
    assert rebuilt(x).shape == model(x).shape


# --- Backward pass / optimizer step ---


def test_backward_pass_populates_gradients():
    model = BaselineAutoencoder(**_TINY_KW)
    x = torch.rand(2, 3, 32, 32)

    loss = mse(model(x), x)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads, "model has no trainable parameters"
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads)


def test_optimizer_step_changes_parameters():
    model = BaselineAutoencoder(**_TINY_KW)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.rand(2, 3, 32, 32)
    before = [p.clone() for p in model.parameters()]

    optimizer.zero_grad()
    mse(model(x), x).backward()
    optimizer.step()

    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


# --- Training engine ---


def _tiny_loaders(tmp_path):
    manifest_path = make_tiny_manifest(tmp_path, width=32, height=32)
    train_loader = create_train_loader(manifest_path, batch_size=2, seed=1)
    val_loader = create_val_loader(manifest_path, batch_size=2)
    return train_loader, val_loader


def test_train_one_epoch_returns_loss_and_updates_parameters(tmp_path):
    train_loader, _ = _tiny_loaders(tmp_path)
    model = BaselineAutoencoder(**_TINY_KW)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = [p.clone() for p in model.parameters()]

    metrics = train_one_epoch(model, train_loader, optimizer, torch.device("cpu"))

    assert "loss" in metrics
    assert metrics["loss"] >= 0.0
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_train_one_epoch_respects_max_batches(tmp_path):
    train_loader, _ = _tiny_loaders(tmp_path)
    model = BaselineAutoencoder(**_TINY_KW)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    metrics = train_one_epoch(model, train_loader, optimizer, torch.device("cpu"), max_batches=1)

    assert "loss" in metrics


def test_validate_one_epoch_returns_loss_and_psnr_without_updating_parameters(tmp_path):
    _, val_loader = _tiny_loaders(tmp_path)
    model = BaselineAutoencoder(**_TINY_KW)
    before = [p.clone() for p in model.parameters()]

    metrics = validate_one_epoch(model, val_loader, torch.device("cpu"))

    assert set(metrics) == {"loss", "psnr"}
    after = list(model.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))


# --- Checkpointing ---


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    model = BaselineAutoencoder(**_TINY_KW)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_psnr": 10.0}]

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path, model=model, optimizer=optimizer, epoch=1,
        history=history, model_config=model.config_dict(),
    )
    checkpoint = load_checkpoint(checkpoint_path)

    assert checkpoint["epoch"] == 1
    assert checkpoint["history"] == history
    assert checkpoint["model_config"] == model.config_dict()
    assert set(checkpoint["model_state_dict"]) == set(model.state_dict())


def test_load_checkpoint_missing_file_raises(tmp_path):
    try:
        load_checkpoint(tmp_path / "does_not_exist.pt")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError for missing checkpoint")


def test_resume_training_state_restores_epoch_history_and_weights(tmp_path):
    model = BaselineAutoencoder(**_TINY_KW)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.rand(2, 3, 32, 32)
    optimizer.zero_grad()
    mse(model(x), x).backward()
    optimizer.step()  # move parameters away from init so the roundtrip is meaningful

    history = [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_psnr": 10.0}]
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path, model=model, optimizer=optimizer, epoch=1,
        history=history, model_config=model.config_dict(),
    )

    resumed_model = BaselineAutoencoder(**_TINY_KW)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-2)

    next_epoch, resumed_history = resume_training_state(
        checkpoint_path, model=resumed_model, optimizer=resumed_optimizer,
    )

    assert next_epoch == 2
    assert resumed_history == history
    for original, resumed in zip(model.parameters(), resumed_model.parameters()):
        assert torch.equal(original, resumed)


# --- Reconstruction metric integration ---


def test_reconstruction_metrics_on_model_output(tmp_path):
    train_loader, _ = _tiny_loaders(tmp_path)
    model = BaselineAutoencoder(**_TINY_KW)
    batch = next(iter(train_loader))

    with torch.no_grad():
        reconstruction = model(batch)

    batch_mse = mse(reconstruction, batch)
    batch_psnr = psnr(reconstruction, batch)

    assert 0.0 <= batch_mse.item() <= 1.0
    assert torch.isfinite(batch_psnr)
