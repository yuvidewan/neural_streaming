"""Tests for Milestone 8A: quantization-aware training via differentiable
uniform noise relaxation (`nvc.training.quantization_noise.QuantizationNoise`).

Distortion-only: nothing here trains against a rate/bitrate objective. The
central property under test throughout is that the mechanism is inert
unless BOTH explicitly attached to a model AND that model is in train()
mode - eval()/inference must never be able to observe it, by construction,
not by convention.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from nvc.compression.calibration import calibrate_quantization_params, save_calibration
from nvc.models import BaselineAutoencoder
from nvc.training import (
    QuantizationNoise,
    load_checkpoint,
    load_model_from_checkpoint,
    save_checkpoint,
)
from nvc.utils.config import Config

from helpers import make_tiny_manifest

_TINY_KW = {"in_channels": 3, "latent_channels": 4, "base_channels": 8}


def _write_calibration(
    tmp_path: Path,
    *,
    bits: int = 8,
    mode: str = "per_channel",
    split: str = "train",
    channels: int = 4,
    name: str = "calibration.json",
) -> tuple[Path, "object"]:
    """A minimal, real calibration file - same on-disk shape
    scripts/calibrate_quantizer.py writes, built without needing a full
    model/encode pass. entropy_model_data is a placeholder: QuantizationNoise
    never reads it, and load_calibration only checks the key is present.
    """
    torch.manual_seed(0)
    latents = torch.randn(6, channels, 4, 4)
    params = calibrate_quantization_params(latents, bits=bits, mode=mode)
    path = tmp_path / name
    save_calibration(
        path, params=params, entropy_model_data={},
        metadata={"calibration_split": split, "bits": bits, "mode": mode},
    )
    return path, params


# --- QuantizationNoise construction ---


def test_quantization_noise_rejects_non_positive_scale():
    with pytest.raises(ValueError):
        QuantizationNoise(torch.zeros(1, 1, 1, 1), bits=8, mode="global")


def test_quantization_noise_rejects_negative_scale():
    with pytest.raises(ValueError):
        QuantizationNoise(torch.full((1, 2, 1, 1), -0.1), bits=8, mode="per_channel")


def test_quantization_noise_rejects_wrong_rank():
    with pytest.raises(ValueError):
        QuantizationNoise(torch.ones(4), bits=8, mode="per_channel")


def test_quantization_noise_rejects_out_of_range_bits():
    with pytest.raises(ValueError):
        QuantizationNoise(torch.ones(1, 1, 1, 1), bits=0, mode="global")


# --- Loading from a calibration artifact (the scale source, req: "SCALE SOURCE") ---


def test_from_calibration_loads_scale_bits_and_mode(tmp_path):
    path, params = _write_calibration(tmp_path, bits=4, mode="per_channel", channels=4)

    noise = QuantizationNoise.from_calibration(path)

    assert noise.bits == 4
    assert noise.mode == "per_channel"
    assert torch.allclose(noise.scale, params.scale)


def test_from_calibration_rejects_bit_depth_mismatch(tmp_path):
    path, _ = _write_calibration(tmp_path, bits=8)

    with pytest.raises(ValueError, match="8-bit"):
        QuantizationNoise.from_calibration(path, bits=4)


def test_from_calibration_rejects_mode_mismatch(tmp_path):
    path, _ = _write_calibration(tmp_path, mode="global")

    with pytest.raises(ValueError, match="global"):
        QuantizationNoise.from_calibration(path, mode="per_channel")


def test_from_calibration_rejects_a_non_train_split():
    # This is the leakage guard: training noise must never be sourced from
    # DAVIS val/test statistics or any non-train calibration.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path, _ = _write_calibration(Path(tmp), split="test")
        with pytest.raises(ValueError, match="train"):
            QuantizationNoise.from_calibration(path)


def test_from_calibration_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        QuantizationNoise.from_calibration(tmp_path / "does_not_exist.json")


# --- Noise magnitude / broadcasting (reqs 4, 5) ---


def test_apply_noise_stays_within_half_the_quantization_step():
    scale = torch.full((1, 1, 1, 1), 0.1)
    noise = QuantizationNoise(scale, bits=8, mode="global")
    z = torch.zeros(4, 3, 8, 8)

    torch.manual_seed(0)
    z_tilde = noise.apply(z)

    deviation = (z_tilde - z).abs()
    assert deviation.max().item() <= 0.05 + 1e-6  # scale/2
    assert deviation.max().item() > 0.0  # a real perturbation, not a no-op


def test_apply_noise_broadcasts_per_channel_scale_independently():
    # Channel 0 gets a tiny step, channel 1 a huge one - if broadcasting
    # were wrong (e.g. global noise reused for every channel, or channels
    # swapped) one of these bounds would be violated.
    scale = torch.tensor([0.01, 2.0]).reshape(1, 2, 1, 1)
    noise = QuantizationNoise(scale, bits=8, mode="per_channel")
    z = torch.zeros(3, 2, 4, 4)

    torch.manual_seed(0)
    z_tilde = noise.apply(z)
    deviation = (z_tilde - z).abs()

    assert deviation[:, 0].max().item() <= 0.005 + 1e-6
    assert deviation[:, 1].max().item() <= 1.0 + 1e-6
    assert deviation[:, 1].mean().item() > deviation[:, 0].mean().item()


def test_apply_noise_moves_scale_to_the_input_device_and_dtype():
    # scale is not a registered buffer (see module docstring), so this
    # exercises the manual .to(z.device, z.dtype) path directly.
    scale = torch.full((1, 1, 1, 1), 0.2, dtype=torch.float64)
    noise = QuantizationNoise(scale, bits=8, mode="global")
    z = torch.zeros(2, 1, 4, 4, dtype=torch.float32)

    z_tilde = noise.apply(z)

    assert z_tilde.dtype == torch.float32


def test_apply_noise_does_not_mutate_the_input_tensor():
    scale = torch.full((1, 1, 1, 1), 0.2)
    noise = QuantizationNoise(scale, bits=8, mode="global")
    z = torch.randn(2, 3, 4, 4)
    z_copy = z.clone()

    torch.manual_seed(0)
    noise.apply(z)

    assert torch.equal(z, z_copy)


def test_apply_noise_rejects_non_4d_latent():
    noise = QuantizationNoise(torch.full((1, 1, 1, 1), 0.1), bits=8, mode="global")
    with pytest.raises(ValueError):
        noise.apply(torch.zeros(3, 8, 8))


# --- BaselineAutoencoder integration: training vs. evaluation behavior ---


def test_forward_is_unchanged_when_quantization_noise_is_disabled():
    # Default construction (quantization_noise=None): must be bit-identical
    # to the pre-Milestone-8A model - the existing float32 path is untouched.
    torch.manual_seed(0)
    model = BaselineAutoencoder(**_TINY_KW)
    model.eval()
    x = torch.rand(2, 3, 32, 32)

    with torch.no_grad():
        direct = model(x)
        staged = model.decode(model.encode(x))

    assert torch.equal(direct, staged)
    assert model.quantization_noise is None


def test_noise_is_injected_in_train_mode_when_enabled():
    scale = torch.full((1, 4, 1, 1), 0.3)
    noise = QuantizationNoise(scale, bits=4, mode="per_channel")
    model = BaselineAutoencoder(quantization_noise=noise, **_TINY_KW)
    model.train()
    x = torch.rand(2, 3, 32, 32)

    torch.manual_seed(10)
    out_a = model(x)
    torch.manual_seed(20)
    out_b = model(x)

    # Same weights, same input, different RNG draw for the noise -> the two
    # forwards must differ. (If noise were not being applied, both encode()
    # and decode() are deterministic and these would be equal.)
    assert not torch.equal(out_a, out_b)


def test_no_noise_is_injected_in_eval_mode_even_if_attached():
    scale = torch.full((1, 4, 1, 1), 0.3)
    noise = QuantizationNoise(scale, bits=4, mode="per_channel")
    model = BaselineAutoencoder(quantization_noise=noise, **_TINY_KW)
    model.eval()
    x = torch.rand(2, 3, 32, 32)

    with torch.no_grad():
        torch.manual_seed(10)
        out_a = model(x)
        torch.manual_seed(20)
        out_b = model(x)

    assert torch.equal(out_a, out_b)


def test_repeated_evaluation_forwards_remain_deterministic():
    torch.manual_seed(0)
    model = BaselineAutoencoder(**_TINY_KW)
    model.eval()
    x = torch.rand(2, 3, 32, 32)

    with torch.no_grad():
        first = model(x)
        second = model(x)

    assert torch.equal(first, second)


def test_seeded_training_noise_is_reproducible():
    scale = torch.full((1, 4, 1, 1), 0.3)
    noise = QuantizationNoise(scale, bits=4, mode="per_channel")
    model = BaselineAutoencoder(quantization_noise=noise, **_TINY_KW)
    model.train()
    x = torch.rand(2, 3, 32, 32)

    torch.manual_seed(42)
    out_a = model(x)
    torch.manual_seed(42)
    out_b = model(x)

    assert torch.equal(out_a, out_b)


def test_output_shape_is_unchanged_across_all_qat_configurations():
    x = torch.rand(2, 3, 32, 32)
    scale = torch.full((1, 4, 1, 1), 0.3)
    noise = QuantizationNoise(scale, bits=4, mode="per_channel")

    baseline = BaselineAutoencoder(**_TINY_KW)
    qat_train = BaselineAutoencoder(quantization_noise=noise, **_TINY_KW)
    qat_train.train()
    qat_eval = BaselineAutoencoder(quantization_noise=noise, **_TINY_KW)
    qat_eval.eval()

    for model in (baseline, qat_train, qat_eval):
        assert model(x).shape == x.shape


def test_model_call_api_is_unchanged():
    # The plain model(x) call - no new required argument, no new keyword -
    # must keep working exactly as every existing caller (train_one_epoch,
    # validate_one_epoch, reconstruct.py, benchmark scripts) expects.
    model = BaselineAutoencoder(**_TINY_KW)
    x = torch.rand(1, 3, 32, 32)

    reconstruction = model(x)

    assert reconstruction.shape == x.shape


def test_eval_mode_inference_does_not_modify_weights_even_with_qat_attached():
    scale = torch.full((1, 4, 1, 1), 0.3)
    noise = QuantizationNoise(scale, bits=4, mode="per_channel")
    model = BaselineAutoencoder(quantization_noise=noise, **_TINY_KW)
    model.eval()
    before = [p.clone() for p in model.parameters()]
    x = torch.rand(2, 3, 32, 32)

    with torch.no_grad():
        model(x)

    for original, current in zip(before, model.parameters()):
        assert torch.equal(original, current)


def test_quantization_noise_is_not_a_registered_submodule_or_buffer():
    # This is what keeps a QAT checkpoint loadable by the ordinary inference
    # path (see test_checkpoint_from_qat_model_loads_via_standard_inference_
    # path below): if QuantizationNoise were an nn.Module/Parameter/buffer,
    # its keys would show up in state_dict() and break strict loading on a
    # model built without it attached.
    scale = torch.full((1, 4, 1, 1), 0.3)
    noise = QuantizationNoise(scale, bits=4, mode="per_channel")
    model = BaselineAutoencoder(quantization_noise=noise, **_TINY_KW)

    assert not any("quantization_noise" in key for key in model.state_dict())
    assert "quantization_noise" not in dict(model.named_modules())
    assert "quantization_noise" not in model.config_dict()


# --- Checkpoint compatibility (reqs 10, 11) ---


def test_checkpoint_without_qat_round_trips_and_infers(tmp_path):
    model = BaselineAutoencoder(**_TINY_KW)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "baseline.pt"
    save_checkpoint(
        path, model=model, optimizer=optimizer, epoch=1, history=[],
        model_config=model.config_dict(),
    )

    loaded, checkpoint = load_model_from_checkpoint(path)
    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        out = loaded(x)

    assert out.shape == x.shape
    assert loaded.quantization_noise is None
    assert checkpoint["epoch"] == 1


def test_checkpoint_from_a_qat_trained_model_loads_via_the_standard_inference_path(tmp_path):
    # Simulates what a real QAT run produces: a checkpoint whose weights
    # were updated under noise injection, but whose saved state_dict/config
    # are otherwise ordinary - so the standard load_model_from_checkpoint
    # (used by reconstruct.py, benchmark_rd.py, calibrate_quantizer.py, ...)
    # must load it with zero special-casing, and get quantization_noise=None
    # (correct for inference - eval mode would ignore it anyway).
    scale = torch.full((1, 4, 1, 1), 0.3)
    noise = QuantizationNoise(scale, bits=4, mode="per_channel")
    model = BaselineAutoencoder(quantization_noise=noise, **_TINY_KW)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.rand(2, 3, 32, 32)
    optimizer.zero_grad()
    torch.nn.functional.mse_loss(model(x), x).backward()
    optimizer.step()

    path = tmp_path / "qat.pt"
    save_checkpoint(
        path, model=model, optimizer=optimizer, epoch=1, history=[],
        model_config=model.config_dict(),
    )

    loaded, _ = load_model_from_checkpoint(path)  # no quantization_noise re-supplied

    assert loaded.quantization_noise is None
    with torch.no_grad():
        out = loaded(torch.rand(1, 3, 32, 32))
    assert out.shape == (1, 3, 32, 32)


def test_load_checkpoint_missing_file_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")


# --- End-to-end smoke test through the real training loop ---


def test_train_one_epoch_and_validate_one_epoch_work_with_qat_attached(tmp_path):
    """SMOKE TEST - proves the mechanism plugs into the existing
    train_one_epoch/validate_one_epoch loop with zero changes to either,
    not a claim about training quality. See helpers.make_tiny_manifest.
    """
    from nvc.data.loaders import create_train_loader, create_val_loader
    from nvc.training import train_one_epoch, validate_one_epoch

    manifest_path = make_tiny_manifest(tmp_path, width=32, height=32)
    train_loader = create_train_loader(manifest_path, batch_size=2, seed=1)
    val_loader = create_val_loader(manifest_path, batch_size=2)

    scale = torch.full((1, 4, 1, 1), 0.3)
    noise = QuantizationNoise(scale, bits=4, mode="per_channel")
    model = BaselineAutoencoder(quantization_noise=noise, **_TINY_KW)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    before = [p.clone() for p in model.parameters()]

    train_metrics = train_one_epoch(model, train_loader, optimizer, torch.device("cpu"))
    val_metrics = validate_one_epoch(model, val_loader, torch.device("cpu"))

    assert "loss" in train_metrics and train_metrics["loss"] >= 0.0
    assert set(val_metrics) == {"loss", "psnr"}
    assert any(not torch.equal(b, a) for b, a in zip(before, model.parameters()))


# --- Configuration (Milestone 8A fields) ---


def test_config_qat_defaults_preserve_baseline_behavior():
    config = Config()

    assert config.qat_enabled is False
    assert config.qat_calibration_path is None
    assert config.qat_mode in ("global", "per_channel")


def test_config_from_json_accepts_a_null_qat_calibration_path(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"qat_calibration_path": None}), encoding="utf-8")

    config = Config.from_json(path)

    assert config.qat_calibration_path is None


def test_config_from_json_coerces_qat_calibration_path_to_a_path(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(
        json.dumps({"qat_enabled": True, "qat_calibration_path": "outputs/calibration/x.json"}),
        encoding="utf-8",
    )

    config = Config.from_json(path)

    assert config.qat_enabled is True
    assert isinstance(config.qat_calibration_path, Path)


# --- CLI validation (scripts/train_autoencoder.py) ---


def _load_train_autoencoder_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_autoencoder.py"
    spec = importlib.util.spec_from_file_location("train_autoencoder_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_requires_calibration_when_qat_enabled(tmp_path, capsys):
    module = _load_train_autoencoder_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        module.main(["--manifest", str(manifest_path), "--qat-enabled"])

    assert "--qat-calibration" in capsys.readouterr().err


def test_cli_reports_missing_calibration_file(tmp_path, capsys):
    module = _load_train_autoencoder_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    exit_code = module.main([
        "--manifest", str(manifest_path),
        "--qat-enabled", "--qat-calibration", str(tmp_path / "missing.json"),
    ])

    assert exit_code == 1
    assert "not found" in capsys.readouterr().err
