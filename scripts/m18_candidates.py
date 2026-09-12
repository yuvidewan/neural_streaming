"""M18 Phase B - candidate intra quantizer variants, diagnosed OFFLINE
(latent-space MSE/SNR only - cheap, no G16/entropy-coder pass) before any
candidate is pushed through the expensive real-pipeline bridge
(m18_reference_bridge.py, Phases C/D).

Every candidate is TRAIN-only fitted (VAL is evaluation-only) and produces
a `QuantizationParams` of the EXACT SAME shape/semantics the deployed one
already has (`bits_per_channel=None`, `companding_gamma=None`, per-channel
affine scale/zero_point) - so every candidate here is realizable without
any `.nvct` format change, by construction, not by later inspection.
Two mechanisms explicitly considered and rejected from this candidate set,
with the reasoning kept here rather than silently omitted:

  - per-channel bit reallocation (`bits_per_channel`, already implemented
    in nvc.compression.calibration.allocate_bits_per_channel): the
    deployed `EmpiricalEntropyModel`/`.nvct` v2 entropy-identity scheme
    assumes ONE alphabet size (`2**bits`) shared by every channel's table
    - a per-channel-varying alphabet is a real format change, forbidden by
      this milestone's freezes. Excluded from Phase C onward; its
      THEORETICAL latent-MSE benefit is still measured below as a
      diagnostic upper bound, clearly labeled as unrealizable here.
  - companding (`companding_gamma`): the quantization.py docstring notes
    this needs no NEW `.nvc` field, but VERIFYING it is actually plumbed
    through `.nvct` v2's existing (de)serialization path is itself
    nontrivial and outside this audit's tight scope - not attempted.

Run:
  ./.venv/Scripts/python.exe scripts/m18_candidates.py
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

from nvc.compression.calibration import allocate_bits_per_channel, calibrate_quantization_params
from nvc.compression.quantization import UniformQuantizer, count_clipped, quantization_error
from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m18_intra_quantizer_audit")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
RATE_POINTS = (5, 4, 3)

# (label, percentile_pair, use_broad_train_coverage)
CANDIDATES = [
    ("A_current_deployed", (0.1, 99.9), False),
    ("BROAD_same_percentile", (0.1, 99.9), True),
    ("TIGHT_1_99", (1.0, 99.0), True),
    ("TIGHTER_2_98", (2.0, 98.0), True),
    ("LOOSE_0.01_99.99", (0.01, 99.99), True),
]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snr_db(std: float, rmse: float) -> float:
    return 20.0 * math.log10(std / rmse) if rmse > 0 else float("inf")


def _evaluate_candidate(params, eval_stack: torch.Tensor, *, bits: int) -> dict[str, Any]:
    quantizer = UniformQuantizer(bits, mode="per_channel")
    dequantized, _ = quantizer.quantize_dequantize(eval_stack, params)
    error = quantization_error(eval_stack, dequantized)
    clipped = count_clipped(eval_stack, params)
    per_channel_std = eval_stack.permute(1, 0, 2, 3).reshape(eval_stack.shape[1], -1).std(dim=1)
    per_channel_rmse = ((dequantized - eval_stack) ** 2).mean(dim=(0, 2, 3)).sqrt()
    return {
        "latent_mse": error["latent_mse"], "latent_mae": error["latent_mae"],
        "clipped_percent": clipped["clipped_percent"],
        "mean_snr_db": float(np.mean([_snr_db(float(per_channel_std[c]), float(per_channel_rmse[c]))
                                      for c in range(eval_stack.shape[1])])),
    }


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M18 Phase B: offline (latent-space) candidate intra quantizer sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--broad-frames-per-sequence", type=int, default=8)
    parser.add_argument("--val-frames-per-sequence", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    if not args.manifest.is_file() or not args.checkpoint.is_file():
        print("[ERROR] --manifest/--checkpoint not found", file=sys.stderr)
        return 1

    m15cal = _load_script("m15_calibration_policy")
    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    train_full = discover_sequences(args.manifest, split="train")
    val_sequences = discover_sequences(args.manifest, split="val",
                                       max_frames_per_sequence=args.val_frames_per_sequence)
    val_a = val_sequences[0::2]

    def _collect_intra(sequences, max_frames):
        parts, seen = [], 0
        with torch.no_grad():
            for sequence in sequences:
                frames = sequence.load_frames()
                for index in range(frames.shape[0]):
                    if seen >= max_frames:
                        break
                    parts.append(model.encode(frames[index:index + 1].to(device)).cpu())
                    seen += 1
                if seen >= max_frames:
                    break
        return torch.cat(parts, dim=0)

    narrow_train_intra = _collect_intra(train_full, args.calibration_frames)
    broad_train_sequences = m15cal.build_policy("C_broad_576", train_full, seed=args.seed,
                                                frames_per_sequence=args.broad_frames_per_sequence)
    broad_train_intra = _collect_intra(broad_train_sequences, 10 ** 9)
    val_intra = _collect_intra(val_a, 10 ** 9)

    print("=" * 118, flush=True)
    print("M18 PHASE B - CANDIDATE INTRA QUANTIZER SWEEP (offline, latent-space)")
    print("=" * 118)
    print(f"  narrow TRAIN: {narrow_train_intra.shape[0]} frames   "
         f"broad TRAIN: {broad_train_intra.shape[0]} frames   VAL-A: {val_intra.shape[0]} frames")

    report: dict[str, Any] = {"phase": "M18 Phase B candidate sweep", "rate_points": []}
    for bits in args.rate_points:
        print(f"\n  ---- {bits}-bit ----")
        rate_point: dict[str, Any] = {"bits": bits, "candidates": {}}
        for label, (lower, upper), broad in CANDIDATES:
            fit_stack = broad_train_intra if broad else narrow_train_intra
            params = calibrate_quantization_params(
                fit_stack, bits=bits, mode="per_channel",
                lower_percentile=lower, upper_percentile=upper)
            metrics = _evaluate_candidate(params, val_intra, bits=bits)
            print(f"    {label:<24} VAL MSE={metrics['latent_mse']:.6f}  "
                 f"SNR={metrics['mean_snr_db']:.2f}dB  clipped={metrics['clipped_percent']:.3f}%")
            rate_point["candidates"][label] = {
                "percentiles": [lower, upper], "broad_train": broad,
                "scale": params.scale.flatten().tolist(),
                "zero_point": params.zero_point.flatten().tolist(), **metrics,
            }

        # Diagnostic-only, explicitly unrealizable within .nvct v2 (see module
        # docstring): per-channel bit reallocation's THEORETICAL MSE bound.
        bits_per_channel = allocate_bits_per_channel(broad_train_intra, average_bits=bits)
        oracle_params = calibrate_quantization_params(
            broad_train_intra, bits=bits, mode="per_channel", bits_per_channel=bits_per_channel)
        oracle_metrics = _evaluate_candidate(oracle_params, val_intra, bits=bits)
        print(f"    [UNREALIZABLE - format change needed] per_channel_bits  "
             f"VAL MSE={oracle_metrics['latent_mse']:.6f}  SNR={oracle_metrics['mean_snr_db']:.2f}dB "
             f"(theoretical bound only, not carried into Phase C)")
        rate_point["unrealizable_per_channel_bits_diagnostic"] = {
            "bits_per_channel": bits_per_channel, **oracle_metrics}

        baseline_mse = rate_point["candidates"]["A_current_deployed"]["latent_mse"]
        best_label = min(rate_point["candidates"], key=lambda k: rate_point["candidates"][k]["latent_mse"])
        best_mse = rate_point["candidates"][best_label]["latent_mse"]
        rate_point["best_candidate"] = best_label
        rate_point["best_candidate_mse_improvement_percent"] = (
            (baseline_mse - best_mse) / baseline_mse * 100)
        print(f"    best candidate: {best_label} "
             f"({rate_point['best_candidate_mse_improvement_percent']:+.3f}% latent MSE vs deployed)")
        report["rate_points"].append(rate_point)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "m18_candidates.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
