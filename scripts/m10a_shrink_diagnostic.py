"""M10A Phase 3: does scale tracking remove the latent-shrinkage exploit?

THE EXPLOIT, RESTATED
----------------------
MILESTONE_9_PLAN.md Section 9F.5 measured that the deployed codec recalibrates
its quantization grid to each model's own latent range, so shrinking the latent
uniformly costs almost nothing in real `.nvc` bits - the grid shrinks with it.
The original `RateEstimator` scored against a bin width FROZEN at construction,
so that same shrinkage read to it as a large rate reduction. Gradient descent
duly found it: M9-H shrank its latent 2.5x versus M9-L for a 3.7x lower proxy R
and +0.7% actual BPP.

WHAT THIS SCRIPT MEASURES
--------------------------
On a real checkpoint's real latent, take z, 0.75z, 0.50z, 0.25z and ask how
many proxy bits each configuration "saves" for shrinkage that the real codec
would be nearly indifferent to. Two configurations:

    frozen    bin width fixed at the calibration value (M9 behaviour)
    tracked   bin width allowed to follow the latent, as it would mid-training

The tracked estimator is given the same EMA adaptation opportunity a training
run would give it (`update_bin_width` on the shrunk latent), because that is
the mechanism under test - not a hypothetical instantaneous recalibration.

For reference the script also reports what the DEPLOYED calibrator would do
with each shrunk latent, computed with the project's own
`calibrate_quantization_params`. That is the ground truth the proxy is supposed
to imitate: its bin width scales with the latent, so its bits-per-symbol
picture is unchanged by pure shrinkage.

This is a DIAGNOSTIC. It does not alter production behaviour, and its numbers
are proxy bits, never `.nvc` bitrate.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\m10a_shrink_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from nvc.compression.calibration import calibrate_quantization_params
from nvc.data.loaders import create_val_loader
from nvc.training import QuantizationNoise, RateEstimator, load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_CHECKPOINT = Path("outputs/qat_combined/checkpoints_qat_noise/best.pt")
DEFAULT_CALIBRATION = Path("outputs/calibration/vimeo_epoch17_4bit.json")
DEFAULT_OUTPUT_DIR = Path("outputs/m10a_pilot")
SHRINK_FACTORS = (1.0, 0.75, 0.50, 0.25)


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "M10A Phase 3: measure the proxy-rate reward for pure latent shrinkage, "
            "with a frozen versus a scale-tracked bin width."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--qat-bits", type=int, default=4)
    parser.add_argument("--qat-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument(
        "--adapt-steps", type=int, default=200,
        help="EMA steps the tracked estimator gets to follow each shrunk latent, "
             "standing in for the adaptation a training run would give it.",
    )
    parser.add_argument("--scale-momentum", type=float, default=0.99)
    parser.add_argument(
        "--density-fit-steps", type=int, default=600,
        help="Steps for configuration C, where loc/log_scale are allowed to follow the "
             "latent as they do in real training under --rate-lr.",
    )
    parser.add_argument("--density-fit-lr", type=float, default=1e-2,
                        help="Matches the project's --rate-lr default.")
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _resolve_device(name: str) -> torch.device:
    return get_device() if name == "auto" else torch.device(name)


def _mean_rate(estimator: RateEstimator, latents: list[torch.Tensor], image_pixels: int) -> float:
    with torch.no_grad():
        return sum(estimator(z, image_pixels).item() for z in latents) / len(latents)


def _fit_density(
    estimator: RateEstimator, latents: list[torch.Tensor], image_pixels: int,
    *, steps: int, lr: float,
) -> None:
    """Let the estimator's own loc/log_scale adapt to this latent, in place.

    Configurations A and B below hold the density frozen at its initialization
    (loc=0, scale=1). Real training does not: `--rate-lr` gives loc/log_scale
    their own optimizer group precisely so they follow the latent (M9C.1
    measured the fitted scale moving 1.018 -> 0.253 across lambdas). Since
    Laplace is a location-scale family, a bin width AND a density that both
    follow the latent make the bin probability - and hence the rate - exactly
    invariant to uniform scaling. Configuration C measures how close gradient
    descent actually gets to that, which is the question that matters.
    """
    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)
    for step in range(steps):
        optimizer.zero_grad()
        estimator(latents[step % len(latents)], image_pixels).backward()
        optimizer.step()


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint),
                        ("--calibration", args.calibration)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    device = _resolve_device(args.device)
    seed_everything(args.seed)

    try:
        noise = QuantizationNoise.from_calibration(
            args.calibration, bits=args.qat_bits, mode=args.qat_mode,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    model, _ = load_model_from_checkpoint(args.checkpoint, device=device, eval_mode=True)
    loader = create_val_loader(
        args.manifest, batch_size=args.batch_size,
        num_workers=defaults.num_workers, crop_size=defaults.random_crop_size,
    )

    latents: list[torch.Tensor] = []
    image_pixels = None
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= args.max_batches:
                break
            batch = batch.to(device)
            image_pixels = batch.shape[-2] * batch.shape[-1]
            latents.append(model.encode(batch))

    print("=" * 84)
    print("M10A Phase 3 - proxy-rate reward for PURE latent shrinkage")
    print("=" * 84)
    print(f"checkpoint: {args.checkpoint}")
    print(f"latent batches: {len(latents)}  |  calibration bin width source: {args.calibration}")
    print(f"tracked estimator gets {args.adapt_steps} EMA steps @ momentum {args.scale_momentum}")
    print("Proxy bits - NOT .nvc bitrate.")
    print("=" * 84)
    print(f"{'factor':>7} {'A frozen':>9} {'reward':>8} | {'B tracked':>10} {'reward':>8} | "
          f"{'C trk+fit':>10} {'reward':>8} | {'trk bw':>8} {'depl bw':>8}")

    rows: list[dict[str, Any]] = []
    baseline_frozen = baseline_tracked = baseline_adapted = None

    for factor in SHRINK_FACTORS:
        shrunk = [z * factor for z in latents]

        # --- A. frozen bin width (the M9 behaviour) ----------------------
        seed_everything(args.seed)
        frozen = RateEstimator(
            noise.scale, bits=noise.bits, mode=noise.mode, track_scale=False,
        ).to(device)
        frozen_rate = _mean_rate(frozen, shrunk, image_pixels)

        # --- B. scale-tracked -------------------------------------------
        # Same construction, then given the adaptation opportunity training
        # would give it. Momentum 0.99 has a ~100-step time constant, so a
        # few hundred steps is what "has converged" means here.
        seed_everything(args.seed)
        tracked = RateEstimator(
            noise.scale, bits=noise.bits, mode=noise.mode,
            track_scale=True, scale_momentum=args.scale_momentum,
        ).to(device)
        for step in range(args.adapt_steps):
            tracked.update_bin_width(shrunk[step % len(shrunk)])
        tracked_rate = _mean_rate(tracked, shrunk, image_pixels)

        # --- Reference: what the DEPLOYED calibrator would derive --------
        deployed = calibrate_quantization_params(
            torch.cat([z.detach().cpu() for z in shrunk]), bits=args.qat_bits, mode=args.qat_mode,
        )

        # --- C. scale-tracked AND density-adapted (what training reaches) --
        seed_everything(args.seed)
        adapted = RateEstimator(
            noise.scale, bits=noise.bits, mode=noise.mode,
            track_scale=True, scale_momentum=args.scale_momentum,
        ).to(device)
        for step in range(args.adapt_steps):
            adapted.update_bin_width(shrunk[step % len(shrunk)])
        _fit_density(adapted, shrunk, image_pixels,
                     steps=args.density_fit_steps, lr=args.density_fit_lr)
        adapted_rate = _mean_rate(adapted, shrunk, image_pixels)

        if factor == 1.0:
            baseline_frozen, baseline_tracked = frozen_rate, tracked_rate
            baseline_adapted = adapted_rate

        row = {
            "factor": factor,
            "frozen_rate_bpp": frozen_rate,
            "frozen_reward_bpp": baseline_frozen - frozen_rate,
            "tracked_rate_bpp": tracked_rate,
            "tracked_reward_bpp": baseline_tracked - tracked_rate,
            "tracked_and_fitted_rate_bpp": adapted_rate,
            "tracked_and_fitted_reward_bpp": baseline_adapted - adapted_rate,
            "tracked_bin_width_mean": tracked.bin_width.mean().item(),
            "deployed_bin_width_mean": deployed.scale.mean().item(),
            "latent_abs_mean": torch.cat([z.flatten() for z in shrunk]).abs().mean().item(),
        }
        rows.append(row)
        print(f"{factor:>7.2f} {frozen_rate:>9.4f} {row['frozen_reward_bpp']:>+8.4f} | "
              f"{tracked_rate:>10.4f} {row['tracked_reward_bpp']:>+8.4f} | "
              f"{adapted_rate:>10.4f} {row['tracked_and_fitted_reward_bpp']:>+8.4f} | "
              f"{row['tracked_bin_width_mean']:>8.4f} {row['deployed_bin_width_mean']:>8.4f}")

    frozen_total = rows[-1]["frozen_reward_bpp"]
    tracked_total = rows[-1]["tracked_reward_bpp"]
    reduction = (
        (frozen_total - tracked_total) / frozen_total * 100 if abs(frozen_total) > 1e-12 else None
    )

    adapted_total = rows[-1]["tracked_and_fitted_reward_bpp"]
    adapted_reduction = (
        (frozen_total - adapted_total) / frozen_total * 100 if abs(frozen_total) > 1e-12 else None
    )
    print("=" * 84)
    print(f"Reward for shrinking to 0.25z:")
    print(f"  A frozen bin width            {frozen_total:+.4f} bpp")
    print(f"  B tracked bin width only      {tracked_total:+.4f} bpp"
          + (f"   ({reduction:.1f}% less than A)" if reduction is not None else ""))
    print(f"  C tracked + density adapted   {adapted_total:+.4f} bpp"
          + (f"   ({adapted_reduction:.1f}% less than A)" if adapted_reduction is not None else ""))
    print()
    print("C is the configuration real training reaches: --rate-lr lets loc/log_scale")
    print("follow the latent alongside the bin width. Laplace being a location-scale")
    print("family, both together make the rate exactly scale-invariant in the limit.")
    print("Deployed bin width scales with the latent (see last column) - that is the")
    print("behaviour the tracked proxy is imitating and the frozen one was not.")
    print("=" * 84)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "shrink_diagnostic.json"
    output.write_text(
        json.dumps(
            {
                "phase": "M10A Phase 3 (scale-shrink diagnostic)",
                "checkpoint": str(args.checkpoint),
                "calibration": str(args.calibration),
                "adapt_steps": args.adapt_steps,
                "scale_momentum": args.scale_momentum,
                "batches": len(latents),
                "seed": args.seed,
                "note": (
                    "All rates are the differentiable Laplace training proxy in bits per "
                    "input pixel. None of these figures is a measured .nvc bitrate."
                ),
                "rows": rows,
                "frozen_reward_at_quarter_bpp": frozen_total,
                "tracked_reward_at_quarter_bpp": tracked_total,
                "tracked_and_fitted_reward_at_quarter_bpp": adapted_total,
                "exploit_reward_reduction_percent_tracked_only": reduction,
                "exploit_reward_reduction_percent_tracked_and_fitted": adapted_reduction,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Written to: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
