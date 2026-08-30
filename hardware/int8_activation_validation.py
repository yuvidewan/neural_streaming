"""Closes ARCHITECTURE.md open risk #1: is the CNN engine's INT8
weights/activations assumption actually viable, quality-wise?

Simulates the accelerator's proposed precision (per-output-channel INT8
weights, per-tensor INT8 activations, both fake-quantized in float so the
numerics - not the speed - match real INT8 hardware) on the REAL QAT
checkpoint, then runs the full real .nvc pipeline (existing calibration,
real entropy coding via encode_symbols/decode_symbols) over the REAL DAVIS
test split, and compares PSNR/MS-SSIM against the float32 numbers already
measured in Milestone 8 (outputs/benchmarks/m8_qat_close_out/qat/).

This is the same rigor M8 applied to QAT: measure actual reconstructed
image quality after the real codec path, not a proxy.

Usage (from the project root, with .venv activated):

    python hardware\\int8_activation_validation.py [--max-frames N]
"""

from __future__ import annotations

import argparse
import copy
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nvc.compression.calibration import load_calibration
from nvc.compression.codec import channel_table_index, latent_to_symbols, symbols_to_latent
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.compression.quantization import QuantizationParams
from nvc.compression.range_coder import decode_symbols, encode_symbols
from nvc.data.image_io import read_image_as_tensor
from nvc.data.loaders import create_train_loader
from nvc.evaluation.basic_metrics import psnr
from nvc.evaluation.perceptual_metrics import msssim
from nvc.evaluation.rd_benchmark import check_calibration_fit
from nvc.evaluation.sequences import discover_sequences
from nvc.training import load_model_from_checkpoint

CHECKPOINT = Path("outputs/qat_combined/checkpoints_qat_noise/best.pt")
CALIBRATION = Path("outputs/calibration/qat_combined_noise.json")  # 8-bit, from M8
MANIFEST = Path("data/processed/manifest.json")

# M8's own float32 measurement for this exact checkpoint+calibration, 8-bit,
# full 719-frame DAVIS test split - outputs/benchmarks/m8_qat_close_out/qat/results.json.
# The reference this script's INT8-simulated numbers are judged against.
M8_FLOAT32_QAT_8BIT_MEAN_PSNR = 29.748
M8_FLOAT32_QAT_8BIT_MEAN_MSSSIM = 0.9730

# Percentile calibration, same convention this project already uses for the
# latent (nvc.compression.calibration.DEFAULT_LOWER/UPPER_PERCENTILE).
ACT_LOWER_PERCENTILE = 0.1
ACT_UPPER_PERCENTILE = 99.9
ACT_CALIBRATION_BATCHES = 5
ACT_CALIBRATION_BATCH_SIZE = 8

_INT8_QMAX = 127  # symmetric signed INT8


def _conv_layers(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]


def _weight_out_channel_dim(module: nn.Module) -> int:
    # Conv2d.weight: [out_c, in_c, kh, kw]; ConvTranspose2d.weight: [in_c, out_c, kh, kw].
    return 0 if isinstance(module, nn.Conv2d) else 1


def _fake_quantize_weights_per_output_channel(module: nn.Module) -> None:
    """In-place, symmetric INT8 fake-quant of one conv layer's weight,
    per output channel - the standard, more-accurate-than-per-tensor
    convention (TFLite/TensorRT both default to this for weights)."""
    dim = _weight_out_channel_dim(module)
    weight = module.weight.data
    other_dims = tuple(d for d in range(weight.dim()) if d != dim)
    max_abs = weight.abs().amax(dim=other_dims, keepdim=True).clamp_min(1e-8)
    scale = max_abs / _INT8_QMAX
    quantized = torch.clamp(torch.round(weight / scale), -_INT8_QMAX, _INT8_QMAX) * scale
    module.weight.data.copy_(quantized)


def _percentile_scale(values: np.ndarray) -> float:
    lower = np.percentile(values, ACT_LOWER_PERCENTILE)
    upper = np.percentile(values, ACT_UPPER_PERCENTILE)
    max_abs = max(abs(lower), abs(upper), 1e-8)
    return max_abs / _INT8_QMAX


def _calibrate_activation_scales(
    model: nn.Module, calibration_params: QuantizationParams, device: torch.device,
) -> dict[int, float]:
    """One real forward pass per calibration frame - THROUGH THE REAL
    quantize/dequantize step for the decoder's input, so decoder layers see
    the same discretized latents they would at real inference time, not
    unquantized float latents. Encoder layers see raw calibration frames.
    """
    layers = _conv_layers(model)
    collected: dict[int, list[np.ndarray]] = {id(layer): [] for layer in layers}

    hooks = []
    for layer in layers:
        def _collect(module, inputs, _id=id(layer)):
            collected[_id].append(inputs[0].detach().cpu().numpy().reshape(-1))
        hooks.append(layer.register_forward_pre_hook(_collect))

    loader = create_train_loader(
        MANIFEST, batch_size=ACT_CALIBRATION_BATCH_SIZE, seed=42, crop_size=None,
    )
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= ACT_CALIBRATION_BATCHES:
                break
            batch = batch.to(device)
            latent = model.encode(batch)
            symbols = np.stack([
                latent_to_symbols(latent[i : i + 1], calibration_params)
                for i in range(latent.shape[0])
            ])
            dequantized = torch.cat([
                symbols_to_latent(symbols[i], latent.shape[1:], calibration_params)
                for i in range(latent.shape[0])
            ], dim=0).to(device)
            model.decode(dequantized)

    for hook in hooks:
        hook.remove()

    scales = {}
    for layer in layers:
        values = np.concatenate(collected[id(layer)])
        scales[id(layer)] = _percentile_scale(values)
    return scales


def _build_int8_simulated_model(
    float_model: nn.Module, activation_scales: dict[int, float],
) -> nn.Module:
    """Deep-copies the model (the real checkpoint's weights are never
    mutated) and applies fake-quant: weights once, in place; activations via
    a forward-pre-hook using the fixed calibrated scale per layer."""
    int8_model = copy.deepcopy(float_model)
    layers = _conv_layers(int8_model)
    float_layers = _conv_layers(float_model)

    for layer in layers:
        _fake_quantize_weights_per_output_channel(layer)

    for float_layer, layer in zip(float_layers, layers):
        scale = activation_scales[id(float_layer)]

        def _fake_quant_activation(module, inputs, _scale=scale):
            x = inputs[0]
            q = torch.clamp(torch.round(x / _scale), -_INT8_QMAX, _INT8_QMAX) * _scale
            return (q,)

        layer.register_forward_pre_hook(_fake_quant_activation)

    return int8_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Cap on DAVIS test frames evaluated (default: the full test split, matching M8).",
    )
    args = parser.parse_args(argv)

    device = torch.device("cpu")
    float_model, _ = load_model_from_checkpoint(CHECKPOINT, device=device)
    float_model.eval()

    calib_doc = load_calibration(CALIBRATION)
    params = QuantizationParams.from_dict(calib_doc["quantization"])
    entropy_model = EmpiricalEntropyModel.from_dict(calib_doc["entropy_model"])
    bits = calib_doc["quantization"]["bits"]

    print(f"Calibrating INT8 activation ranges ({ACT_CALIBRATION_BATCHES * ACT_CALIBRATION_BATCH_SIZE} "
          f"train frames, {ACT_LOWER_PERCENTILE}/{ACT_UPPER_PERCENTILE} percentile, per-layer)...")
    activation_scales = _calibrate_activation_scales(float_model, params, device)
    print(f"  {len(activation_scales)} conv layers calibrated "
          f"(scale range: {min(activation_scales.values()):.5f} - {max(activation_scales.values()):.5f})")

    int8_model = _build_int8_simulated_model(float_model, activation_scales)
    int8_model.eval()

    sequences = discover_sequences(MANIFEST, split="test")
    frame_paths = [p for seq in sequences for p in seq.frame_paths]
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]
    print(f"\nEvaluating {len(frame_paths)} real DAVIS test frames "
          f"({'full test split, matching M8' if args.max_frames is None else 'capped subset'})...")

    psnr_values: list[float] = []
    msssim_values: list[float] = []
    clip_percents: list[float] = []

    with torch.no_grad():
        for i, frame_path in enumerate(frame_paths):
            frame = read_image_as_tensor(frame_path).unsqueeze(0).to(device)
            latent = int8_model.encode(frame)

            symbols = latent_to_symbols(latent, params)
            channels, height, width = latent.shape[1], latent.shape[2], latent.shape[3]
            idx = channel_table_index(channels, height, width)
            payload = encode_symbols(symbols, entropy_model.cumulative, idx)
            decoded_symbols = decode_symbols(payload, len(symbols), entropy_model.cumulative, idx)
            assert np.array_equal(decoded_symbols, symbols), "entropy coding must still be lossless"

            dequantized = symbols_to_latent(decoded_symbols, (channels, height, width), params).to(device)
            reconstruction = int8_model.decode(dequantized)

            psnr_values.append(psnr(reconstruction, frame).item())
            try:
                msssim_values.append(msssim(reconstruction, frame).item())
            except Exception:
                pass  # frame too small for MS-SSIM's 5 scales - shouldn't happen at 256x256, guard anyway

            if i == 0:
                fit = check_calibration_fit(int8_model, sequences, params, device=device)
                clip_percents.append(fit["clipped_percent"])
                print(f"  Calibration fit check (INT8-simulated encoder): "
                      f"{fit['clipped_percent']:.4f}% clipped ({'OK' if fit['fits'] else 'MISMATCH'}, "
                      f"threshold {fit['threshold_percent']:.1f}%)")
                if not fit["fits"]:
                    print("  [WARNING] existing calibration does not fit the INT8-simulated encoder's "
                          "latents - results below are not methodologically valid. See stdout above.",
                          file=sys.stderr)

    mean_psnr = statistics.mean(psnr_values)
    mean_msssim = statistics.mean(msssim_values) if msssim_values else None

    print(f"\n=== INT8-simulated vs. float32 (M8, same checkpoint+calibration, {bits}-bit) ===")
    print(f"{'Metric':<12s} {'float32 (M8)':>14s} {'INT8-simulated':>16s} {'delta':>10s}")
    print(f"{'PSNR (dB)':<12s} {M8_FLOAT32_QAT_8BIT_MEAN_PSNR:>14.3f} {mean_psnr:>16.3f} "
          f"{mean_psnr - M8_FLOAT32_QAT_8BIT_MEAN_PSNR:>+10.3f}")
    if mean_msssim is not None:
        print(f"{'MS-SSIM':<12s} {M8_FLOAT32_QAT_8BIT_MEAN_MSSSIM:>14.4f} {mean_msssim:>16.4f} "
              f"{mean_msssim - M8_FLOAT32_QAT_8BIT_MEAN_MSSSIM:>+10.4f}")
    print(f"\nFrames evaluated: {len(psnr_values)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
