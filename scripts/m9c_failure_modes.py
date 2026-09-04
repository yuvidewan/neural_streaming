"""Milestone 9C, phase 7: failure-mode analysis of the lambda pilot arms.

THE QUESTION THIS EXISTS TO ANSWER
-----------------------------------
The pilot's headline R comes from each arm's OWN trained rate estimator. That
number can fall for two completely different reasons:

    (1) the encoder genuinely produced a cheaper latent          - what we want
    (2) the Laplace density simply got better at describing an   - "estimator
        essentially unchanged latent                               gaming"

Both look identical in the training log. Separating them needs a measure of
the latent that does not depend on how well that arm's own estimator was
trained, so this script computes two:

  R_refit   Fit a FRESH rate estimator (same init, same steps, same LR) to
            each arm's final latent. Every arm is then scored by a density of
            identical capacity fitted with identical effort, so differences
            come from the latent alone.

  H_symbol  Empirical discrete entropy of the latent after quantization on
            the EXISTING frozen 4-bit calibration grid, in bits per input
            pixel. Independent of the Laplace family entirely - it is a
            property of the symbol histogram, not of any fitted model.

`H_symbol` reads the existing calibration and uses `UniformQuantizer.quantize`
read-only. It does NOT recalibrate anything (M9C Rule 8), does NOT touch
`EmpiricalEntropyModel`, the range coder or the `.nvc` format (Rule 2), and is
NOT a claim about actual `.nvc` payload size (Rule 9) - a real payload also
carries the coder's own overhead and its static per-channel table's mismatch,
neither of which a plug-in entropy figure includes. It is a directional
diagnostic, and the order-of-magnitude sanity check on the Laplace proxy.

ALSO CHECKED
-------------
Rate collapse / explosion (implausible R, pathological fitted scales),
non-finite values anywhere, decoder-only compensation (D moved but the latent
did not), and lambda insensitivity (all arms landing on the same point).

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\m9c_failure_modes.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

from nvc.compression.calibration import load_calibration
from nvc.compression.quantization import QuantizationParams, UniformQuantizer, count_clipped
from nvc.data.loaders import create_val_loader
from nvc.training import QuantizationNoise, RateEstimator, load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m9c_lambda_pilot")
DEFAULT_CALIBRATION = Path("outputs/calibration/vimeo_epoch17_4bit.json")

# A fitted Laplace scale this far outside the latent's own spread means the
# density has stopped describing the data - the signature of rate collapse
# (a scale driven to nothing) or of a degenerate flat fit.
_MIN_PLAUSIBLE_SCALE = 1e-3
_MAX_PLAUSIBLE_SCALE = 1e4
# Below this, an estimated bits-per-pixel is not a plausible compression
# operating point for this architecture - it means the density collapsed
# rather than the latent becoming cheap.
_IMPLAUSIBLY_LOW_BPP = 1e-3


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Milestone 9C phase 7: check the lambda pilot arms for rate collapse, "
            "rate explosion, estimator gaming, decoder-only compensation and "
            "lambda insensitivity."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument(
        "--summary", type=Path, default=DEFAULT_OUTPUT_DIR / "lambda_pilot_summary.json",
        help="Output of scripts/m9c_lambda_pilot.py, naming the arms to analyse.",
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--qat-bits", type=int, default=4)
    parser.add_argument("--qat-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--max-batches", type=int, default=40)
    parser.add_argument(
        "--refit-steps", type=int, default=600,
        help="Steps for the matched fresh-estimator refit. Identical for every arm.",
    )
    parser.add_argument("--refit-lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _resolve_device(name: str) -> torch.device:
    return get_device() if name == "auto" else torch.device(name)


def _symbol_entropy_bpp(
    symbols: torch.Tensor, params: QuantizationParams, image_pixels: int
) -> float:
    """Per-channel empirical entropy of quantized symbols, in bits per pixel.

    Normalized exactly as `RateEstimator.forward` normalizes: total bits over
    the whole latent divided by the INPUT image's height*width, so the two
    numbers are directly comparable. A plug-in (maximum-likelihood) entropy
    estimate - it is a lower bound on what any real coder using this symbol
    alphabet would spend, not a prediction of the coded size.
    """
    batch, channels, height, width = symbols.shape
    flat = symbols.reshape(batch, channels, -1).to(torch.int64)
    total_bits = 0.0
    for channel in range(channels):
        values = flat[:, channel, :].reshape(-1)
        counts = torch.bincount(values - int(params.q_min))
        probabilities = counts[counts > 0].to(torch.float64) / values.numel()
        entropy = float(-(probabilities * torch.log2(probabilities)).sum())
        total_bits += entropy * values.numel()
    return total_bits / (batch * image_pixels)


def _refit_estimator(
    latents: list[torch.Tensor],
    bin_width: torch.Tensor,
    *,
    bits: int,
    mode: str,
    image_pixels: int,
    steps: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[RateEstimator, float]:
    """Fit a fresh estimator to fixed latents. Identical protocol per arm."""
    seed_everything(seed)
    estimator = RateEstimator(bin_width, bits=bits, mode=mode).to(device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)

    step = 0
    while step < steps:
        for latent in latents:
            if step >= steps:
                break
            optimizer.zero_grad()
            rate = estimator(latent, image_pixels)
            rate.backward()
            optimizer.step()
            step += 1

    with torch.no_grad():
        final = sum(estimator(latent, image_pixels).item() for latent in latents) / len(latents)
    return estimator, final


def _analyse_arm(
    arm: dict[str, Any],
    *,
    manifest: Path,
    quantization_noise: QuantizationNoise,
    quantization_params: QuantizationParams,
    quantizer: UniformQuantizer,
    batch_size: int,
    max_batches: int,
    num_workers: int,
    crop_size: int | None,
    refit_steps: int,
    refit_lr: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(seed)
    checkpoint_path = Path(arm["checkpoint_dir"]) / "latest.pt"
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device, eval_mode=True)
    model.quantization_noise = quantization_noise

    loader = create_val_loader(
        manifest, batch_size=batch_size, num_workers=num_workers, crop_size=crop_size,
    )

    # Collect this arm's latents once, in eval mode (no QAT noise) - the
    # latent the deployed encoder would actually emit.
    latents: list[torch.Tensor] = []
    image_pixels = None
    latent_values: list[torch.Tensor] = []
    symbol_entropies: list[float] = []
    device_params = quantization_params.to(device)
    clipped_total = 0.0
    clipped_values = 0.0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            batch = batch.to(device)
            image_pixels = batch.shape[-2] * batch.shape[-1]
            latent = model.encode(batch)
            latents.append(latent)
            latent_values.append(latent.flatten().cpu())
            # quantize() returns (symbols, params_used); the params are the
            # frozen ones passed in, so only the symbols are of interest here.
            symbols, _ = quantizer.quantize(latent, device_params)
            symbol_entropies.append(
                _symbol_entropy_bpp(symbols.cpu(), quantization_params, image_pixels)
            )
            # How much of the latent the FROZEN grid cannot represent. This is
            # the confounder for symbol_entropy_bpp above: a latent that
            # saturates the grid piles onto the two extreme symbols, which
            # LOWERS its measured entropy while destroying information. Read
            # the two columns together, never the entropy alone.
            clipping = count_clipped(latent, device_params)
            clipped_total += clipping["clipped_total"]
            clipped_values += clipping["total_values"]

    all_values = torch.cat(latent_values)
    _, refit_rate = _refit_estimator(
        latents, quantization_noise.scale, bits=quantization_noise.bits,
        mode=quantization_noise.mode, image_pixels=image_pixels,
        steps=refit_steps, lr=refit_lr, seed=seed, device=device,
    )

    own_scale = arm["rate_estimator_scale"]
    return {
        "lambda": arm["lambda"],
        "regime": arm["regime"],
        "val_distortion": arm["val_distortion"],
        "val_psnr_db": arm["val_psnr_db"],
        "rate_own_estimator_bpp": arm["val_rate_bpp"],
        "rate_matched_refit_bpp": refit_rate,
        "symbol_entropy_bpp": sum(symbol_entropies) / len(symbol_entropies),
        "clipped_percent_on_frozen_grid": 100.0 * clipped_total / clipped_values,
        "latent_stats": {
            "mean": all_values.mean().item(),
            "std": all_values.std().item(),
            "abs_mean": all_values.abs().mean().item(),
            "min": all_values.min().item(),
            "max": all_values.max().item(),
        },
        "own_estimator_scale": own_scale,
        "checkpoint_epoch": checkpoint["epoch"],
        "flags": {
            "non_finite": not (
                math.isfinite(refit_rate)
                and math.isfinite(arm["val_rate_bpp"])
                and math.isfinite(arm["val_distortion"])
                and bool(torch.isfinite(all_values).all())
            ),
            "rate_collapse": (
                refit_rate < _IMPLAUSIBLY_LOW_BPP
                or own_scale["min"] < _MIN_PLAUSIBLE_SCALE
            ),
            "rate_explosion": (
                own_scale["max"] > _MAX_PLAUSIBLE_SCALE
                or arm["val_rate_bpp"] > 64.0
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    for label, path in (
        ("--manifest", args.manifest),
        ("--summary", args.summary),
        ("--calibration", args.calibration),
    ):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    device = _resolve_device(args.device)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    arms = summary["arms"]

    try:
        quantization_noise = QuantizationNoise.from_calibration(
            args.calibration, bits=args.qat_bits, mode=args.qat_mode,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    quantization_params = QuantizationParams.from_dict(
        load_calibration(args.calibration)["quantization"]
    )
    quantizer = UniformQuantizer(bits=quantization_params.bits, mode=quantization_params.mode)

    print("=" * 70)
    print("M9C phase 7: failure-mode analysis")
    print("=" * 70)
    print(f"arms: {len(arms)}   matched refit: {args.refit_steps} steps @ lr {args.refit_lr}")
    print(f"symbol entropy on the EXISTING {quantization_params.bits}-bit grid "
          f"(read-only; nothing recalibrated)")
    print("=" * 70)

    results = []
    for arm in arms:
        analysis = _analyse_arm(
            arm, manifest=args.manifest, quantization_noise=quantization_noise,
            quantization_params=quantization_params, quantizer=quantizer,
            batch_size=args.batch_size, max_batches=args.max_batches,
            num_workers=defaults.num_workers, crop_size=defaults.random_crop_size,
            refit_steps=args.refit_steps, refit_lr=args.refit_lr,
            seed=args.seed, device=device,
        )
        results.append(analysis)
        print(f"  lambda={analysis['lambda']:.4e} ({analysis['regime']:<26}) "
              f"D={analysis['val_distortion']:.4e}  R_own={analysis['rate_own_estimator_bpp']:.4f}  "
              f"R_refit={analysis['rate_matched_refit_bpp']:.4f}  "
              f"H_symbol={analysis['symbol_entropy_bpp']:.4f}  "
              f"clip={analysis['clipped_percent_on_frozen_grid']:5.2f}%  "
              f"latent|mean|={analysis['latent_stats']['abs_mean']:.4f}")

    positive = [r for r in results if r["lambda"] > 0]
    control = next((r for r in results if r["lambda"] == 0), None)

    # Lambda insensitivity: if the strongest and weakest rate-aware arms land
    # on effectively the same latent, lambda is not doing anything.
    refit_spread = (
        max(r["rate_matched_refit_bpp"] for r in positive)
        - min(r["rate_matched_refit_bpp"] for r in positive)
    ) if positive else 0.0
    entropy_spread = (
        max(r["symbol_entropy_bpp"] for r in positive)
        - min(r["symbol_entropy_bpp"] for r in positive)
    ) if positive else 0.0

    # Estimator gaming: the own-estimator rate moved but neither the matched
    # refit nor the symbol entropy did. Compared against the control's latent,
    # which had no rate pressure at all.
    gaming = False
    decoder_only = False
    if control is not None and positive:
        strongest = max(positive, key=lambda r: r["lambda"])
        own_delta = control["rate_own_estimator_bpp"] - strongest["rate_own_estimator_bpp"]
        refit_delta = control["rate_matched_refit_bpp"] - strongest["rate_matched_refit_bpp"]
        entropy_delta = control["symbol_entropy_bpp"] - strongest["symbol_entropy_bpp"]
        # Gated on the matched refit ONLY, never on symbol entropy: the refit
        # is the one latent-side measure that is not confounded by how much of
        # the latent the frozen grid clips (see above). "The estimator's own
        # number moved a lot while the latent's matched-capacity cost did not"
        # is exactly estimator gaming; entropy_delta is reported alongside for
        # context but must not gate the verdict.
        gaming = own_delta > 0.1 and refit_delta < 0.01
        decoder_only = refit_delta < 0.01
    else:
        own_delta = refit_delta = entropy_delta = None

    # The project's existing calibration-fit guard (see rd_benchmark's
    # check_calibration_fit / MILESTONE_8_RESULTS.md) treats >2% clipping as a
    # mismatched grid. Reported per arm because it is the confounder that makes
    # symbol_entropy_bpp non-comparable ACROSS arms when it varies this widely.
    worst_clip = max(r["clipped_percent_on_frozen_grid"] for r in results)
    verdict = {
        "max_clipped_percent_on_frozen_grid": worst_clip,
        "frozen_grid_mismatched": worst_clip > 2.0,
        "symbol_entropy_comparable_across_arms": (
            max(r["clipped_percent_on_frozen_grid"] for r in results)
            - min(r["clipped_percent_on_frozen_grid"] for r in results)
        ) < 2.0,
        "non_finite_any_arm": any(r["flags"]["non_finite"] for r in results),
        "rate_collapse_any_arm": any(r["flags"]["rate_collapse"] for r in results),
        "rate_explosion_any_arm": any(r["flags"]["rate_explosion"] for r in results),
        "matched_refit_spread_bpp": refit_spread,
        "symbol_entropy_spread_bpp": entropy_spread,
        "lambda_insensitivity": refit_spread < 0.01 and entropy_spread < 0.01,
        "strongest_vs_control_own_rate_delta_bpp": own_delta,
        "strongest_vs_control_refit_delta_bpp": refit_delta,
        "strongest_vs_control_symbol_entropy_delta_bpp": entropy_delta,
        "estimator_gaming": gaming,
        "decoder_only_compensation": decoder_only,
    }

    print("=" * 70)
    for key, value in verdict.items():
        print(f"  {key}: {value}")
    print("=" * 70)
    print("H_symbol is a plug-in entropy on the frozen grid - NOT a measured "
          ".nvc payload bitrate, and nothing was recalibrated (Rules 2, 8, 9).")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "failure_mode_analysis.json"
    output_path.write_text(
        json.dumps(
            {
                "phase": "M9C phase 7 (failure modes)",
                "refit_steps": args.refit_steps,
                "refit_lr": args.refit_lr,
                "calibration": str(args.calibration),
                "max_batches": args.max_batches,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "note": (
                    "rate_matched_refit_bpp and symbol_entropy_bpp are latent-side "
                    "diagnostics, not measured .nvc bitrate. No recalibration was "
                    "performed and the deployed codec path was not modified or run."
                ),
                "arms": results,
                "verdict": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nAnalysis written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
