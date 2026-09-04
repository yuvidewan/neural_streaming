"""Milestone 9C.1, phase 2: does a separate rate LR actually let the estimator adapt?

THE QUESTION
-------------
M9C found the rate estimator effectively frozen: sharing the model's 1e-4, its
`loc`/`log_scale` moved by at most ~0.05 over 500 Adam steps and ended at
`scale` 1.016-1.050 (initialization is exactly 1.0). With `loc~0, scale~1` the
Laplace code length is very nearly proportional to `abs(z)`, so the rate term
degenerated into an L1 penalty on latent magnitude rather than a fitted
entropy model.

This script runs the SAME short pilot twice - once at the old rate LR, once at
the new one - changing nothing else, and reports how far the estimator's
parameters actually travelled in each. It is the controlled A/B the lambda
re-sweep is gated on: if the estimator still cannot move, re-running five
lambdas would only reproduce M9C.

WHAT "ADAPTED" MEANS HERE
--------------------------
Deliberately NOT "reached scale 6.4". The diagnostic's warm fit reached about
that, but only as evidence that substantial movement is POSSIBLE at a
sufficient LR - it is not a target, and the joint objective's optimum need not
coincide with a rate-only fit's. The evidence looked for instead is:

  * loc/log_scale move by O(1), not O(0.01)
  * the estimator's own R converges toward the independently refitted R
    (the gap between them IS the estimator's misfit)
  * nothing diverges

NOT VALIDATED HERE: any correspondence to real `.nvc` payload bitrate. Every
rate figure is the training-time Laplace proxy or an independent refit of the
same family - see MILESTONE_9_PLAN.md.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\m9c1_adaptation_check.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

from nvc.data.loaders import create_val_loader
from nvc.evaluation.basic_metrics import mse, psnr
from nvc.training import QuantizationNoise, RateEstimator, load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_CHECKPOINT = Path("outputs/qat_combined/checkpoints_qat_noise/best.pt")
DEFAULT_CALIBRATION = Path("outputs/calibration/vimeo_epoch17_4bit.json")
DEFAULT_OUTPUT_DIR = Path("outputs/m9c1_rate_lr_pilot")

# The M9C balance point, from that milestone's own measured mean(D)/mean(R).
# Used here only to put both arms under identical, realistic rate pressure -
# the lambda grid itself is re-derived later, in the full sweep.
M9C_LAMBDA_BALANCE = 9.075672e-04


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Milestone 9C.1 phase 2: run the same short pilot at the old and new rate "
            "learning rates and compare how far the rate estimator's parameters move."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--qat-bits", type=int, default=4)
    parser.add_argument("--qat-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--rate-lambda", type=float, default=M9C_LAMBDA_BALANCE)
    parser.add_argument(
        "--rate-lrs", type=float, nargs="+", default=[1e-4, 1e-2],
        help="Rate learning rates to compare. The first is M9C's (shared with the model).",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--latent-channels", type=int, default=defaults.latent_channels)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--refit-steps", type=int, default=600)
    parser.add_argument("--refit-lr", type=float, default=1e-2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _resolve_device(name: str) -> torch.device:
    return get_device() if name == "auto" else torch.device(name)


def _load_script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _estimator_summary(estimator: RateEstimator) -> dict[str, float]:
    loc = estimator.loc.detach().flatten()
    scale = torch.exp(estimator.log_scale.detach()).flatten()
    return {
        "loc_mean": loc.mean().item(), "loc_std": loc.std().item(),
        "loc_min": loc.min().item(), "loc_max": loc.max().item(),
        "loc_abs_max": loc.abs().max().item(),
        "scale_mean": scale.mean().item(), "scale_std": scale.std().item(),
        "scale_min": scale.min().item(), "scale_max": scale.max().item(),
    }


def _refit_rate(
    latents: list[torch.Tensor], estimator_template: RateEstimator, *,
    image_pixels: int, steps: int, lr: float, seed: int, device: torch.device,
) -> float:
    """Independent matched-capacity refit, identical protocol for both arms."""
    seed_everything(seed)
    fresh = RateEstimator(
        estimator_template.bin_width, bits=estimator_template.bits, mode=estimator_template.mode,
    ).to(device)
    optimizer = torch.optim.Adam(fresh.parameters(), lr=lr)
    step = 0
    while step < steps:
        for latent in latents:
            if step >= steps:
                break
            optimizer.zero_grad()
            rate = fresh(latent, image_pixels)
            rate.backward()
            optimizer.step()
            step += 1
    with torch.no_grad():
        return sum(fresh(latent, image_pixels).item() for latent in latents) / len(latents)


def _evaluate(
    checkpoint_path: Path, *, manifest: Path, noise: QuantizationNoise,
    batch_size: int, max_batches: int, num_workers: int, crop_size: int | None,
    refit_steps: int, refit_lr: float, seed: int, device: torch.device,
) -> dict[str, Any]:
    """Val D / PSNR / own-estimator R / matched-refit R for one finished arm."""
    seed_everything(seed)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device, eval_mode=True)
    model.quantization_noise = noise

    estimator = RateEstimator(noise.scale, bits=noise.bits, mode=noise.mode).to(device)
    extra = checkpoint.get("extra") or {}
    if "rate_estimator_state_dict" in extra:
        estimator.load_state_dict(extra["rate_estimator_state_dict"])

    loader = create_val_loader(
        manifest, batch_size=batch_size, num_workers=num_workers, crop_size=crop_size,
    )
    latents, distortions, rates, psnrs, latent_values = [], [], [], [], []
    image_pixels = None
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            batch = batch.to(device)
            image_pixels = batch.shape[-2] * batch.shape[-1]
            latent = model.encode(batch)
            latents.append(latent)
            latent_values.append(latent.flatten().cpu())
            reconstruction = model.decode(latent)
            distortions.append(mse(reconstruction, batch).item())
            rates.append(estimator(latent, image_pixels).item())
            psnrs.append(psnr(reconstruction, batch).item())

    own_rate = sum(rates) / len(rates)
    refit = _refit_rate(
        latents, estimator, image_pixels=image_pixels,
        steps=refit_steps, lr=refit_lr, seed=seed, device=device,
    )
    all_values = torch.cat(latent_values)
    return {
        "val_distortion": sum(distortions) / len(distortions),
        "val_psnr_db": sum(psnrs) / len(psnrs),
        "rate_own_estimator_bpp": own_rate,
        "rate_matched_refit_bpp": refit,
        # The estimator's misfit: how much worse its own density is than an
        # equally-sized one fitted properly to the same latent. This going to
        # ~0 is what "the estimator adapted" actually means.
        "own_minus_refit_bpp": own_rate - refit,
        "latent_std": all_values.std().item(),
        "latent_abs_mean": all_values.abs().mean().item(),
        "latent_abs_max": all_values.abs().max().item(),
        "estimator": _estimator_summary(estimator),
        "all_finite": bool(
            math.isfinite(own_rate) and math.isfinite(refit)
            and torch.isfinite(all_values).all()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    for label, path in (
        ("--manifest", args.manifest),
        ("--checkpoint", args.checkpoint),
        ("--calibration", args.calibration),
    ):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1
    if any(not math.isfinite(value) or value <= 0 for value in args.rate_lrs):
        parser.error("every --rate-lrs value must be finite and > 0")

    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        noise = QuantizationNoise.from_calibration(
            args.calibration, bits=args.qat_bits, mode=args.qat_mode,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    # Initialization is the common starting point both arms are measured from.
    initial = _estimator_summary(RateEstimator(noise.scale, bits=noise.bits, mode=noise.mode))

    print("=" * 74)
    print("M9C.1 phase 2: rate-estimator adaptation check")
    print("=" * 74)
    print(f"lambda {args.rate_lambda:.6e} (M9C balance) | model lr {args.learning_rate}")
    print(f"rate LRs compared: {args.rate_lrs}")
    print(f"{args.epochs} epochs x {args.max_batches} batches = "
          f"{args.epochs * args.max_batches} steps per arm, seed {args.seed}, device {device}")
    print(f"estimator init: loc 0.0, scale {initial['scale_mean']:.4f}")
    print("=" * 74)

    trainer = _load_script("train_autoencoder")
    results = []
    for rate_lr in args.rate_lrs:
        arm_dir = args.output_dir / f"adaptation_ratelr_{rate_lr:.0e}".replace(".", "p")
        print(f"\n--- rate_lr = {rate_lr:g} -> {arm_dir}")
        exit_code = trainer.main([
            "--manifest", str(args.manifest),
            "--epochs", str(args.epochs), "--max-batches", str(args.max_batches),
            "--batch-size", str(args.batch_size), "--learning-rate", repr(args.learning_rate),
            "--latent-channels", str(args.latent_channels), "--seed", str(args.seed),
            "--device", str(device),
            "--resume", str(args.checkpoint), "--resume-model-only",
            "--qat-enabled", "--qat-bits", str(args.qat_bits), "--qat-mode", args.qat_mode,
            "--qat-calibration", str(args.calibration),
            "--rate-enabled", "--rate-lambda", repr(args.rate_lambda),
            "--rate-lr", repr(rate_lr),
            "--checkpoint-dir", str(arm_dir),
        ])
        if exit_code != 0:
            print(f"[ERROR] arm rate_lr={rate_lr} failed ({exit_code})", file=sys.stderr)
            return 1

        measured = _evaluate(
            arm_dir / "latest.pt", manifest=args.manifest, noise=noise,
            batch_size=args.batch_size, max_batches=args.max_batches,
            num_workers=defaults.num_workers, crop_size=defaults.random_crop_size,
            refit_steps=args.refit_steps, refit_lr=args.refit_lr,
            seed=args.seed, device=device,
        )
        estimator = measured["estimator"]
        measured.update({
            "rate_lr": rate_lr,
            "checkpoint_dir": str(arm_dir),
            # Movement from the shared initialization (loc=0, scale=1).
            "loc_movement_abs_max": estimator["loc_abs_max"],
            "scale_movement_max": max(
                abs(estimator["scale_max"] - 1.0), abs(estimator["scale_min"] - 1.0)
            ),
        })
        results.append(measured)
        print(f"      loc: mean {estimator['loc_mean']:+.4f} std {estimator['loc_std']:.4f} "
              f"(|max| {estimator['loc_abs_max']:.4f})")
        print(f"      scale: mean {estimator['scale_mean']:.4f} "
              f"[{estimator['scale_min']:.4f}, {estimator['scale_max']:.4f}]")
        print(f"      D {measured['val_distortion']:.6e}  PSNR {measured['val_psnr_db']:.2f} dB  "
              f"R_own {measured['rate_own_estimator_bpp']:.4f}  "
              f"R_refit {measured['rate_matched_refit_bpp']:.4f}  "
              f"misfit {measured['own_minus_refit_bpp']:+.4f}")

    verdict: dict[str, Any] = {"adapted": None}
    if len(results) >= 2:
        old, new = results[0], results[-1]
        verdict = {
            "old_rate_lr": old["rate_lr"],
            "new_rate_lr": new["rate_lr"],
            "loc_movement_ratio": (
                new["loc_movement_abs_max"] / old["loc_movement_abs_max"]
                if old["loc_movement_abs_max"] > 0 else None
            ),
            "scale_movement_ratio": (
                new["scale_movement_max"] / old["scale_movement_max"]
                if old["scale_movement_max"] > 0 else None
            ),
            "old_misfit_bpp": old["own_minus_refit_bpp"],
            "new_misfit_bpp": new["own_minus_refit_bpp"],
            "misfit_reduced": new["own_minus_refit_bpp"] < old["own_minus_refit_bpp"],
            # O(1) movement is the threshold M9C's finding was stated against
            # (it measured O(0.01)); not a target value for scale itself.
            "estimator_moved_order_one": (
                new["loc_movement_abs_max"] > 0.5 or new["scale_movement_max"] > 0.5
            ),
            "all_finite": all(row["all_finite"] for row in results),
        }
        verdict["adapted"] = bool(
            verdict["estimator_moved_order_one"]
            and verdict["misfit_reduced"]
            and verdict["all_finite"]
        )

    print("\n" + "=" * 74)
    for key, value in verdict.items():
        print(f"  {key}: {value}")
    print("=" * 74)
    print("Rate figures are the training-time proxy / an independent refit of the same")
    print("family - NOT measured .nvc payload bitrate.")

    output_path = args.output_dir / "adaptation_check.json"
    output_path.write_text(
        json.dumps(
            {
                "phase": "M9C.1 phase 2 (rate-estimator adaptation check)",
                "rate_lambda": args.rate_lambda,
                "model_learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "max_batches": args.max_batches,
                "steps_per_arm": args.epochs * args.max_batches,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "refit_steps": args.refit_steps,
                "refit_lr": args.refit_lr,
                "estimator_initialization": initial,
                "arms": results,
                "verdict": verdict,
                "note": (
                    "rate_own_estimator_bpp and rate_matched_refit_bpp are training-time "
                    "Laplace estimates, not measured .nvc bitrate."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWritten to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
