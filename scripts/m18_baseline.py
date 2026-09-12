"""M18 Phase 0/A - confirm the M13/M14/M15/M17 baseline hasn't drifted, then
audit the deployed INTRA vs RESIDUAL quantizer precisely, normalized (never
comparing raw scale alone - the two latents have very different dynamic
ranges by construction: intra represents whole-frame content, residual
represents a motion-compensated difference that is mostly near zero).

Both quantizers are `nvc.compression.quantization.UniformQuantizer`-style
affine grids (see that module's own docstring for the exact formula); the
only structural difference between the deployed intra and residual
QuantizationParams objects is WHICH calibration tensor produced their
scale/zero_point (`calibrate_grids`'s intra_stack vs residual_stack) - same
code path, same `calibrate_quantization_params` call, same percentile
convention (0.1/99.9), same per-channel mode. Both entropy models
(`EmpiricalEntropyModel`) use the identical alphabet size `2**bits` (a
structural invariant of that class, not a per-milestone choice) - so any
effective-precision gap comes entirely from the CALIBRATED SCALE, not from
a different alphabet or a different quantization formula.

Run:
  ./.venv/Scripts/python.exe scripts/m18_baseline.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.quantization import UniformQuantizer, count_clipped, quantization_error
from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m18_intra_quantizer_audit")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M13_DAVIS_JSON = Path("outputs/m13_recalibration/m13_davis_benchmark.json")
RATE_POINTS = (5, 4, 3)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snr_db(std: float, rmse: float) -> float:
    if rmse <= 0:
        return float("inf")
    return 20.0 * math.log10(std / rmse)


def _audit_quantizer(name: str, calibration_stack: torch.Tensor, eval_stack: torch.Tensor,
                     *, bits: int) -> dict[str, Any]:
    """TRAIN-fit params (from `calibration_stack`), measured on BOTH the
    TRAIN set itself and a held-out `eval_stack` (VAL) - generalization gap
    matters as much as the fit itself."""
    params = calibrate_quantization_params(calibration_stack, bits=bits, mode="per_channel")
    quantizer = UniformQuantizer(bits, mode="per_channel")

    per_channel_std = calibration_stack.permute(1, 0, 2, 3).reshape(
        calibration_stack.shape[1], -1).std(dim=1)
    step = params.scale.flatten()  # one quantization level's width, per channel

    results = {}
    for split_name, stack in (("train_fit", calibration_stack), ("held_out_eval", eval_stack)):
        dequantized, _ = quantizer.quantize_dequantize(stack, params)
        error = quantization_error(stack, dequantized)
        clipped = count_clipped(stack, params)
        per_channel_mse = ((dequantized - stack) ** 2).mean(dim=(0, 2, 3))
        per_channel_rmse = per_channel_mse.sqrt()
        results[split_name] = {
            "latent_mse": error["latent_mse"], "latent_mae": error["latent_mae"],
            "latent_max_abs_error": error["latent_max_abs_error"],
            "clipped_percent": clipped["clipped_percent"],
            "mean_step": float(step.mean()),
            "mean_error_over_step_ratio": float((per_channel_rmse / step).mean()),
            "mean_snr_db": float(np.mean([
                _snr_db(float(per_channel_std[c]), float(per_channel_rmse[c]))
                for c in range(len(step))])),
        }
    return {
        "name": name, "bits": bits,
        "num_channels": calibration_stack.shape[1],
        "global_min": float(calibration_stack.min()), "global_max": float(calibration_stack.max()),
        "mean_per_channel_std": float(per_channel_std.mean()),
        "mean_step": float(step.mean()),
        "mean_step_over_std_ratio": float((step / per_channel_std).mean()),
        **results,
    }


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M18 Phase 0/A: baseline confirmation + intra-vs-residual quantizer audit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--m13-davis-json", type=Path, default=DEFAULT_M13_DAVIS_JSON)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--val-frames-per-sequence", type=int, default=40)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    mc = _load_script("m10h_motion_compensation")
    ev = _load_script("m10l_evaluate")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    train_sequences = discover_sequences(args.manifest, split="train")
    val_sequences = discover_sequences(args.manifest, split="val",
                                       max_frames_per_sequence=args.val_frames_per_sequence)
    val_a, val_b = val_sequences[0::2], val_sequences[1::2]

    print("=" * 112, flush=True)
    print("M18 PHASE 0/A - BASELINE + INTRA vs RESIDUAL QUANTIZER AUDIT")
    print("=" * 112)

    report: dict[str, Any] = {
        "phase": "M18 Phase 0/A", "checkpoint": str(args.checkpoint),
        "formula_note": (
            "Both intra and residual are UniformQuantizer per-channel affine grids: "
            "scale=(x_max-x_min)/(2**bits-1), zero_point=-round(x_min/scale), "
            "q=clamp(round(x/scale)+zero_point,0,2**bits-1). Percentile source: "
            "calibrate_quantization_params, (0.1, 99.9) percentile, per_channel mode, "
            "identical code path for both - the ONLY difference is which calibration "
            "tensor (calibrate_grids's intra_stack vs residual_stack) produced scale/"
            "zero_point. Both entropy models use alphabet size 2**bits identically - "
            "EmpiricalEntropyModel enforces frequencies.shape[1] == 2**bits structurally, "
            "so there is no alphabet/range asymmetry between intra and residual tables."),
        "rate_points": [],
    }

    for bits in args.rate_points:
        with torch.no_grad():
            intra_latents: list[torch.Tensor] = []
            seen = 0
            for sequence in train_sequences:
                frames = sequence.load_frames()
                for index in range(frames.shape[0]):
                    if seen >= args.calibration_frames:
                        break
                    intra_latents.append(model.encode(frames[index:index + 1].to(device)).cpu())
                    seen += 1
                if seen >= args.calibration_frames:
                    break
            intra_stack = torch.cat(intra_latents, dim=0)

        calibration = mc.calibrate_grids(
            model, train_sequences, bits=bits, mode="per_channel", gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, reference_mode="mc",
            max_frames=args.calibration_frames)
        residual_params = calibration["residual_params"]
        # Reconstruct the residual calibration tensor via the SAME formula
        # calibrate_grids used internally (its own residual_stack is not
        # returned - only the fitted params are) - re-derive it identically
        # rather than approximate it, since Phase A needs the raw tensor.
        residual_stack = _redo_residual_stack(
            mc, model, train_sequences, bits=bits, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            max_frames=args.calibration_frames, device=device,
            intra_params=calibration["intra_params"],
            intra_entropy_model=calibration["intra_entropy_model"])

        with torch.no_grad():
            val_intra_parts = []
            for sequence in val_a:
                frames = sequence.load_frames()
                for index in range(frames.shape[0]):
                    val_intra_parts.append(model.encode(frames[index:index + 1].to(device)).cpu())
            val_intra = torch.cat(val_intra_parts, dim=0)
        val_residual = _redo_residual_stack(
            mc, model, val_a, bits=bits, gop_size=args.gop, block_size=args.block_size,
            search_range=args.search_range, max_frames=10 ** 9, device=device,
            intra_params=calibration["intra_params"],
            intra_entropy_model=calibration["intra_entropy_model"])

        # Self-consistency check: refitting from the re-derived residual_stack
        # must reproduce calibrate_grids's own residual_params EXACTLY - if
        # _redo_residual_stack had any bug, this catches it immediately
        # rather than silently reporting wrong Phase A numbers.
        refit = calibrate_quantization_params(residual_stack, bits=bits, mode="per_channel")
        assert torch.equal(refit.scale, residual_params.scale), (
            f"{bits}-bit: re-derived residual_stack does not reproduce calibrate_grids's "
            f"own residual_params - _redo_residual_stack has a bug")
        assert torch.equal(refit.zero_point, residual_params.zero_point)

        intra_audit = _audit_quantizer("intra", intra_stack, val_intra, bits=bits)
        residual_audit = _audit_quantizer("residual", residual_stack, val_residual, bits=bits)

        print(f"\n  ---- {bits}-bit ----")
        for a in (intra_audit, residual_audit):
            h = a["held_out_eval"]
            print(f"    {a['name']:<9} mean_step={a['mean_step']:.5f}  "
                 f"step/std={a['mean_step_over_std_ratio']:.4f}  "
                 f"VAL: MSE={h['latent_mse']:.6f} err/step={h['mean_error_over_step_ratio']:.4f} "
                 f"SNR={h['mean_snr_db']:.2f}dB clipped={h['clipped_percent']:.3f}%")
        step_ratio = residual_audit["mean_step"] / intra_audit["mean_step"]
        print(f"    residual step is {step_ratio:.4f}x intra step (raw, NOT normalized - "
             f"see step/std ratio above for the fair comparison)")

        report["rate_points"].append({"bits": bits, "intra": intra_audit, "residual": residual_audit,
                                      "raw_step_ratio_residual_over_intra": step_ratio})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "m18_baseline.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


def _redo_residual_stack(mc, model, sequences, *, bits, gop_size, block_size, search_range,
                         max_frames, device, intra_params, intra_entropy_model) -> torch.Tensor:
    """Reproduces EXACTLY calibrate_grids's own residual-collection loop
    (same true-latent reference advance, same GOP typing), since that
    function only returns the FITTED params, not the raw calibration
    tensor Phase A needs to audit directly."""
    from nvc.compression.codec import decode_payload_to_latent, encode_latent_to_payload
    residuals: list[torch.Tensor] = []
    seen = 0
    with torch.no_grad(), mc.deterministic_kernels():
        for sequence in sequences:
            frames = sequence.load_frames()
            types = mc.gop_frame_types(frames.shape[0], gop_size)
            previous_reconstruction = None
            for index in range(frames.shape[0]):
                if seen >= max_frames:
                    break
                frame = frames[index:index + 1].to(device)
                latent = model.encode(frame)
                if types[index] == mc.FRAME_TYPE_I:
                    payload, _ = encode_latent_to_payload(
                        latent, params=intra_params, entropy_model=intra_entropy_model)
                    decoded, _ = decode_payload_to_latent(
                        payload, entropy_model=intra_entropy_model, params=intra_params,
                        shape=tuple(latent.shape[1:]))
                    previous_reconstruction = model.decode(decoded.to(device))
                elif previous_reconstruction is not None:
                    motion = mc.estimate_block_motion(
                        previous_reconstruction, frame, block_size=block_size,
                        search_range=search_range)
                    warped = mc.warp_blocks(previous_reconstruction, motion, block_size=block_size)
                    residuals.append((latent - model.encode(warped)).cpu())
                    previous_reconstruction = model.decode(latent)
                seen += 1
            if seen >= max_frames:
                break
    return torch.cat(residuals, dim=0)


if __name__ == "__main__":
    sys.exit(main())
