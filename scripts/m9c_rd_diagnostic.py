"""Milestone 9C, phases 2-3: measure the D/R scale and sanity-check the rate proxy.

WHAT THIS IS FOR
-----------------
M9C must choose `lambda` in `L = D + lambda * R` from THIS implementation's
own measured magnitudes, not from a conventional value quoted in another
paper (whose D might be MSE on 0-255 pixels, or 1-MS-SSIM, and whose R might
be bits-per-latent-element rather than bits-per-input-pixel). A lambda is
only meaningful relative to the units of the two terms it balances, so this
script measures both terms on real data first and reports the ratio.

It changes NO model weights. The autoencoder is loaded frozen from the M8
QAT checkpoint and its parameters are asserted bit-identical at the end. The
only thing this script trains is the rate estimator's own 128 scalars
(`loc`/`log_scale`), and only in the "warm fit" stage below, which exists
because of this measurement subtlety:

    `RateEstimator` initializes to loc=0, log_scale=0 (scale=1) - a prior
    that has not yet seen the latent at all. The rate it reports at that
    moment is dominated by the prior's misfit, not by the latent's actual
    entropy, and it drops sharply within the first optimizer steps of any
    real run. Choosing lambda against that inflated initial R would pick a
    lambda too small by whatever factor the misfit inflates it.

So this script reports R twice - at initialization, and after fitting the
estimator alone against the frozen latent - and derives the balancing lambda
from the FITTED value, which is the R that actually persists through
training. Both numbers are written to the JSON so the choice is auditable.

WHAT "D" AND "R" MEAN HERE, EXACTLY
------------------------------------
Measured in the same regime the training loss sees (`model.train()`, QAT
noise applied to the latent before both the rate estimator and the decoder),
mirroring `trainer.train_one_epoch_with_rate` step for step - not eval mode,
which would silently drop the noise and measure a different D and a
different R than the objective is actually built from. Eval-mode figures are
also recorded, separately labelled, for reference.

    D = mse(decode(z_noised), x)        [0,1] pixel range, per-batch mean
    R = rate_estimator(z_noised, H*W)   estimated bits per INPUT pixel

NOT VALIDATED HERE (deliberately): that R corresponds to any real `.nvc`
payload size. R is a training-time proxy from a learned Laplace density; the
deployed coder uses `EmpiricalEntropyModel` and is untouched. Comparing the
two is a later phase's job - see MILESTONE_9_PLAN.md.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\m9c_rd_diagnostic.py
    python scripts\\m9c_rd_diagnostic.py --max-batches 10 --warm-fit-steps 50
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from nvc.data.loaders import create_train_loader, create_val_loader
from nvc.data.validation import DatasetValidationError
from nvc.evaluation.basic_metrics import mse, psnr
from nvc.training import (
    QuantizationNoise,
    RateEstimator,
    load_model_from_checkpoint,
)
from nvc.training.rate_estimator import _half_cdf_offset
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

# Defaults point at the M8 QAT artifacts this milestone is required to build
# on (MILESTONE_8_RESULTS.md): the QAT-noise checkpoint, and the same 4-bit
# per-channel train-split calibration that run's noise scale came from.
DEFAULT_CHECKPOINT = Path("outputs/qat_combined/checkpoints_qat_noise/best.pt")
DEFAULT_CALIBRATION = Path("outputs/calibration/vimeo_epoch17_4bit.json")
DEFAULT_OUTPUT_DIR = Path("outputs/m9c_lambda_pilot")


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Milestone 9C phases 2-3: measure the distortion/rate scale of the "
            "M8 QAT model on real data and sanity-check the rate estimator. "
            "Changes no model weights."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json",
        help="Dataset manifest produced by scripts/prepare_dataset.py (the DAVIS manifest).",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Starting model. M9C requires the M8 QAT checkpoint; weights are never modified.",
    )
    parser.add_argument(
        "--calibration", type=Path, default=DEFAULT_CALIBRATION,
        help="Train-split calibration supplying BOTH the QAT noise scale and the rate "
             "estimator's bin width - one scale for the whole run, as in train_autoencoder.py.",
    )
    parser.add_argument("--qat-bits", type=int, default=4, help="Must match --calibration's own bit depth.")
    parser.add_argument(
        "--qat-mode", choices=["global", "per_channel"], default="per_channel",
        help="Must match --calibration's own mode.",
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--max-batches", type=int, default=40,
        help="Number of real batches to measure D and R over.",
    )
    parser.add_argument(
        "--warm-fit-steps", type=int, default=200,
        help="Optimizer steps spent fitting ONLY the rate estimator's loc/log_scale "
             "against the frozen latent, before re-measuring R. See module docstring.",
    )
    parser.add_argument(
        "--warm-fit-lr", type=float, default=1e-2,
        help="Learning rate for the warm fit above. Higher than the model's own LR on "
             "purpose - 128 scalars fitting a fixed target, not a network being trained.",
    )
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _resolve_device(name: str) -> torch.device:
    return get_device() if name == "auto" else torch.device(name)


def _summarize(values: list[float]) -> dict[str, float]:
    """mean/median/std/min/max for one measured series."""
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def _measure(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    rate_estimator: RateEstimator,
    max_batches: int,
    train_mode: bool,
    seed: int,
) -> dict[str, Any]:
    """One measurement pass. No optimizer, no weight updates.

    `train_mode=True` reproduces `train_one_epoch_with_rate`'s regime exactly
    (QAT noise applied before both the rate estimator and the decoder);
    `train_mode=False` is the eval-mode reference with no noise at all. The
    RNG is re-seeded per pass so two passes over the same loader see the same
    QAT noise draws and are therefore directly comparable.
    """
    seed_everything(seed)
    model.train(train_mode)

    distortions: list[float] = []
    rates: list[float] = []
    psnrs: list[float] = []
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            image_pixels = batch.shape[-2] * batch.shape[-1]

            latent = model.encode(batch)
            if model.training and model.quantization_noise is not None:
                latent = model.quantization_noise.apply(latent)
            rate = rate_estimator(latent, image_pixels)
            reconstruction = model.decode(latent)

            distortions.append(mse(reconstruction, batch).item())
            rates.append(rate.item())
            psnrs.append(psnr(reconstruction, batch).item())

            num_batches += 1
            if num_batches >= max_batches:
                break

    if num_batches == 0:
        raise ValueError("_measure: the data loader produced no batches")

    # Per-batch D/R ratios, then averaged - deliberately not mean(D)/mean(R),
    # which is a different statistic. Both are reported; the per-batch spread
    # is what says whether a single global lambda is even a sensible choice.
    ratios = [d / r for d, r in zip(distortions, rates)]
    return {
        "mode": "train (QAT noise on)" if train_mode else "eval (no noise)",
        "distortion": _summarize(distortions),
        "rate_bpp": _summarize(rates),
        "psnr_db": _summarize(psnrs),
        "d_over_r_per_batch": _summarize(ratios),
        "mean_d_over_mean_r": statistics.fmean(distortions) / statistics.fmean(rates),
        "mean_r_over_mean_d": statistics.fmean(rates) / statistics.fmean(distortions),
        "all_finite": all(
            torch.isfinite(torch.tensor(series)).all().item()
            for series in (distortions, rates, psnrs)
        ),
        "rate_non_negative": min(rates) >= 0.0,
    }


def _warm_fit_rate_estimator(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    rate_estimator: RateEstimator,
    steps: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    """Fit ONLY loc/log_scale against the frozen latent, minimizing R alone.

    The model is in train mode (so the latent carries QAT noise, matching the
    real regime) but under `torch.no_grad()` for the encode step: the latent
    is a fixed target here, and no gradient may reach the autoencoder. Model
    parameters are never handed to this optimizer at all - only the rate
    estimator's own parameters are.
    """
    seed_everything(seed)
    model.train()
    optimizer = torch.optim.Adam(rate_estimator.parameters(), lr=lr)

    history: list[float] = []
    step = 0
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            batch = batch.to(device)
            image_pixels = batch.shape[-2] * batch.shape[-1]

            with torch.no_grad():
                latent = model.encode(batch)
                if model.quantization_noise is not None:
                    latent = model.quantization_noise.apply(latent)

            optimizer.zero_grad()
            rate = rate_estimator(latent, image_pixels)
            rate.backward()
            optimizer.step()

            history.append(rate.item())
            step += 1

    return {
        "steps": step,
        "learning_rate": lr,
        "first_rate_bpp": history[0] if history else None,
        "last_rate_bpp": history[-1] if history else None,
        "rate_trajectory_every_10": history[::10],
    }


def _estimator_stats(rate_estimator: RateEstimator) -> dict[str, Any]:
    loc = rate_estimator.loc.detach().flatten()
    scale = torch.exp(rate_estimator.log_scale.detach()).flatten()
    bin_width = rate_estimator.bin_width.detach().flatten()
    return {
        "loc": {"mean": loc.mean().item(), "std": loc.std().item(),
                "min": loc.min().item(), "max": loc.max().item()},
        "scale": {"mean": scale.mean().item(), "std": scale.std().item(),
                  "min": scale.min().item(), "max": scale.max().item()},
        "bin_width": {"mean": bin_width.mean().item(),
                      "min": bin_width.min().item(), "max": bin_width.max().item()},
        "all_finite": bool(torch.isfinite(loc).all() and torch.isfinite(scale).all()),
    }


@contextmanager
def _deterministic_kernels():
    """Force bit-reproducible conv kernels for the duration of the block.

    Checks D and E below compare two forward/backward passes and assert the
    difference is EXACTLY zero. cuDNN's default autotuned convolution
    backward is non-deterministic, so without this those comparisons bottom
    out at a ~1e-9 floor of pure kernel noise and can no longer distinguish
    "the rate term contributed nothing" from "the rate term contributed a
    little" - which is the entire question being asked. Uses the same knobs
    `nvc.utils.seed.seed_everything(deterministic=True)` sets, restored on
    exit so the surrounding measurement passes keep their normal speed.
    """
    previous_deterministic = torch.backends.cudnn.deterministic
    previous_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = previous_deterministic
        torch.backends.cudnn.benchmark = previous_benchmark


def _sanity_checks(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    rate_estimator: RateEstimator,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    """Phase 3: empirical verification of the rate estimator's contract.

    Every check runs against the REAL M8 QAT model and REAL DAVIS batches -
    tests/test_rate_estimator.py already covers the same properties on
    synthetic data, so the value added here is confirming they hold at
    production scale and with the real latent distribution.
    """
    results: dict[str, Any] = {}
    batch = next(iter(loader)).to(device)
    image_pixels = batch.shape[-2] * batch.shape[-1]

    # --- A. More probability mass => less rate -------------------------
    # A latent sitting exactly on the density's mode must cost fewer bits
    # than one far out in its tail. Also checked as a monotone sweep, since
    # a single pair could pass by coincidence.
    with torch.no_grad():
        loc = rate_estimator.loc
        scale = torch.exp(rate_estimator.log_scale)
        at_mode = rate_estimator.rate_bits(loc.expand(1, -1, 4, 4).contiguous()).mean().item()
        in_tail = rate_estimator.rate_bits(
            (loc + 20.0 * scale).expand(1, -1, 4, 4).contiguous()
        ).mean().item()
        sweep = [
            rate_estimator.rate_bits(
                (loc + offset * scale).expand(1, -1, 4, 4).contiguous()
            ).mean().item()
            for offset in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
        ]
    results["A_more_mass_less_rate"] = {
        "rate_bits_at_mode": at_mode,
        "rate_bits_in_tail": in_tail,
        "mode_cheaper_than_tail": at_mode < in_tail,
        "monotone_sweep_bits": sweep,
        "sweep_is_non_decreasing": all(a <= b + 1e-6 for a, b in zip(sweep, sweep[1:])),
    }

    # --- A2. The stabilized CDF form matches a naive float64 reference ---
    # Guards the algebraic cancellation in _half_cdf_offset: if that rewrite
    # were wrong, every number in this milestone would be wrong. Compared
    # against torch.distributions.Laplace's own CDF at float64.
    with torch.no_grad():
        latent = model.encode(batch)
        z64 = latent.double()
        loc64, scale64 = rate_estimator.loc.double(), torch.exp(rate_estimator.log_scale.double())
        half64 = rate_estimator.bin_width.double() * 0.5
        reference_dist = torch.distributions.Laplace(loc64, scale64)
        reference_p = reference_dist.cdf(z64 + half64) - reference_dist.cdf(z64 - half64)
        stabilized_p = (
            _half_cdf_offset(z64 + half64 - loc64, scale64)
            - _half_cdf_offset(z64 - half64 - loc64, scale64)
        )
        max_abs_error = (reference_p - stabilized_p).abs().max().item()
    results["A2_cdf_form_matches_reference"] = {
        "max_abs_probability_error_vs_torch_laplace": max_abs_error,
        "agrees": max_abs_error < 1e-9,
    }

    # --- B. Finite and non-negative -------------------------------------
    with torch.no_grad():
        bits = rate_estimator.rate_bits(latent)
    results["B_finite_non_negative"] = {
        "all_finite": bool(torch.isfinite(bits).all()),
        "min_bits_per_element": bits.min().item(),
        "max_bits_per_element": bits.max().item(),
        "non_negative": bool((bits >= 0).all()),
    }

    # --- C. Gradients reach the latent, loc and log_scale ---------------
    model.train()
    latent = model.encode(batch)
    latent.retain_grad()
    rate = rate_estimator(latent, image_pixels)
    rate.backward()
    results["C_gradients_exist"] = {
        "latent_grad_norm": latent.grad.norm().item(),
        "loc_grad_norm": rate_estimator.loc.grad.norm().item(),
        "log_scale_grad_norm": rate_estimator.log_scale.grad.norm().item(),
        "all_nonzero": all(
            t.grad is not None and t.grad.norm().item() > 0
            for t in (latent, rate_estimator.loc, rate_estimator.log_scale)
        ),
    }

    # --- D. lambda = 0 is exactly the distortion-only path --------------
    # Not merely "the scalar loss equals D": the model's gradients and its
    # parameters after one real optimizer step must match the reconstruction-
    # only path bit for bit. Both paths get identical starting weights
    # (deepcopy) and identical QAT noise draws (re-seeded before each).
    def _one_step(lambda_rate: float | None) -> tuple[torch.nn.Module, dict[str, torch.Tensor], float]:
        """lambda_rate=None runs the plain distortion-only path instead."""
        step_model = copy.deepcopy(model)
        step_model.train()
        step_estimator = copy.deepcopy(rate_estimator)
        params = list(step_model.parameters())
        if lambda_rate is not None:
            params += list(step_estimator.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        seed_everything(seed)
        optimizer.zero_grad()
        if lambda_rate is None:
            reconstruction = step_model(batch)
            loss = mse(reconstruction, batch)
        else:
            z = step_model.encode(batch)
            if step_model.quantization_noise is not None:
                z = step_model.quantization_noise.apply(z)
            r = step_estimator(z, image_pixels)
            reconstruction = step_model.decode(z)
            loss = mse(reconstruction, batch) + lambda_rate * r
        loss.backward()
        gradients = {
            name: p.grad.detach().clone()
            for name, p in step_model.named_parameters() if p.grad is not None
        }
        optimizer.step()
        return step_model, gradients, loss.item()

    with _deterministic_kernels():
        plain_model, plain_grads, plain_loss = _one_step(None)
        zero_model, zero_grads, zero_loss = _one_step(0.0)
        positive_lambda = 0.01
        _, positive_grads, positive_loss = _one_step(positive_lambda)

    max_grad_diff = max(
        (plain_grads[name] - zero_grads[name]).abs().max().item() for name in plain_grads
    )
    max_param_diff = max(
        (a - b).abs().max().item()
        for a, b in zip(plain_model.parameters(), zero_model.parameters())
    )
    results["D_lambda_zero_equals_distortion_only"] = {
        "distortion_only_loss": plain_loss,
        "lambda_zero_loss": zero_loss,
        "loss_abs_diff": abs(plain_loss - zero_loss),
        "max_abs_gradient_diff_over_all_model_params": max_grad_diff,
        "max_abs_param_diff_after_one_optimizer_step": max_param_diff,
        "num_params_compared": len(plain_grads),
        "gradients_identical": max_grad_diff == 0.0,
        "params_identical_after_step": max_param_diff == 0.0,
    }

    # --- E. lambda > 0 genuinely changes the encoder's gradient ---------
    # The rate term must reach the ENCODER (not just the rate estimator's own
    # parameters), otherwise the objective could never trade rate for
    # distortion in the latent at all.
    encoder_grad_delta = max(
        (positive_grads[name] - zero_grads[name]).abs().max().item()
        for name in positive_grads if name.startswith("encoder.")
    )
    decoder_grad_delta = max(
        (positive_grads[name] - zero_grads[name]).abs().max().item()
        for name in positive_grads if name.startswith("decoder.")
    )
    results["E_lambda_positive_adds_rate_gradient"] = {
        "lambda": positive_lambda,
        "loss": positive_loss,
        "max_encoder_gradient_change_vs_lambda_zero": encoder_grad_delta,
        "max_decoder_gradient_change_vs_lambda_zero": decoder_grad_delta,
        "encoder_receives_rate_contribution": encoder_grad_delta > 0.0,
        # The decoder sits AFTER the latent the rate is measured on, so the
        # rate term contributes no gradient path to it - expected to be 0.
        "decoder_unchanged_as_expected": decoder_grad_delta == 0.0,
    }
    return results


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
    if args.max_batches < 1:
        parser.error("--max-batches must be >= 1")
    if args.warm_fit_steps < 0:
        parser.error("--warm-fit-steps must be >= 0")

    seed_everything(args.seed)
    device = _resolve_device(args.device)

    try:
        quantization_noise = QuantizationNoise.from_calibration(
            args.calibration, bits=args.qat_bits, mode=args.qat_mode,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device=device, eval_mode=True)
    model.quantization_noise = quantization_noise
    # Snapshot to prove at the end that this script trained nothing.
    weights_before = {name: p.detach().clone() for name, p in model.named_parameters()}

    # Same construction train_autoencoder.py uses when QAT and rate are both
    # on: the rate estimator reuses the QAT noise's literal scale tensor, so
    # there is exactly one quantization scale in play.
    rate_estimator = RateEstimator(
        quantization_noise.scale, bits=quantization_noise.bits, mode=quantization_noise.mode,
    ).to(device)

    def fresh_train_loader():
        return create_train_loader(
            args.manifest, batch_size=args.batch_size, num_workers=defaults.num_workers,
            seed=args.seed, crop_size=defaults.random_crop_size,
        )

    try:
        train_loader = fresh_train_loader()
        val_loader = create_val_loader(
            args.manifest, batch_size=args.batch_size, num_workers=defaults.num_workers,
            crop_size=defaults.random_crop_size,
        )
    except DatasetValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("M9C phases 2-3: D/R scale measurement + rate-estimator sanity checks")
    print("=" * 70)
    print(f"checkpoint:   {args.checkpoint} (epoch {checkpoint['epoch']})")
    print(f"calibration:  {args.calibration} ({quantization_noise.bits}-bit / {quantization_noise.mode})")
    print(f"manifest:     {args.manifest}")
    print(f"device:       {device}   batch_size: {args.batch_size}   seed: {args.seed}")
    print(f"batches:      {args.max_batches}   warm-fit steps: {args.warm_fit_steps}")
    print("Model weights are FROZEN - only the rate estimator's loc/log_scale move.")
    print("=" * 70)

    report: dict[str, Any] = {
        "phase": "M9C phases 2-3 (D/R scale + sanity checks)",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "model_config": checkpoint["model_config"],
        "calibration": str(args.calibration),
        "qat_bits": quantization_noise.bits,
        "qat_mode": quantization_noise.mode,
        "manifest": str(args.manifest),
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "seed": args.seed,
        "device": str(device),
    }

    # --- Phase 2a: D and R at rate-estimator initialization -------------
    print("\n[1/5] Measuring D and R with the rate estimator at initialization...")
    report["train_split_at_init"] = _measure(
        model, fresh_train_loader(), device, rate_estimator=rate_estimator,
        max_batches=args.max_batches, train_mode=True, seed=args.seed,
    )
    report["rate_estimator_at_init"] = _estimator_stats(rate_estimator)
    initial = report["train_split_at_init"]
    print(f"      D mean {initial['distortion']['mean']:.6e}   "
          f"R mean {initial['rate_bpp']['mean']:.4f} bpp   "
          f"PSNR {initial['psnr_db']['mean']:.2f} dB")

    # --- Phase 2b: warm-fit the estimator, then re-measure --------------
    print(f"\n[2/5] Warm-fitting the rate estimator alone ({args.warm_fit_steps} steps, model frozen)...")
    report["warm_fit"] = _warm_fit_rate_estimator(
        model, fresh_train_loader(), device, rate_estimator=rate_estimator,
        steps=args.warm_fit_steps, lr=args.warm_fit_lr, seed=args.seed,
    )
    print(f"      R: {report['warm_fit']['first_rate_bpp']:.4f} -> "
          f"{report['warm_fit']['last_rate_bpp']:.4f} bpp")

    print("\n[3/5] Re-measuring D and R with the fitted rate estimator...")
    report["train_split_fitted"] = _measure(
        model, fresh_train_loader(), device, rate_estimator=rate_estimator,
        max_batches=args.max_batches, train_mode=True, seed=args.seed,
    )
    report["val_split_fitted"] = _measure(
        model, val_loader, device, rate_estimator=rate_estimator,
        max_batches=args.max_batches, train_mode=False, seed=args.seed,
    )
    report["rate_estimator_fitted"] = _estimator_stats(rate_estimator)
    fitted = report["train_split_fitted"]
    print(f"      D mean {fitted['distortion']['mean']:.6e}   "
          f"R mean {fitted['rate_bpp']['mean']:.4f} bpp   "
          f"PSNR {fitted['psnr_db']['mean']:.2f} dB")

    # --- Phase 4 input: the balancing lambda ----------------------------
    # lambda_balance is the value at which lambda*R equals D, i.e. the point
    # where the two loss terms contribute equally. Derived from the FITTED R
    # (see module docstring), with the init-R figure kept alongside it so the
    # difference between the two is visible rather than buried.
    balance_fitted = fitted["mean_d_over_mean_r"]
    balance_init = initial["mean_d_over_mean_r"]
    report["lambda_balance"] = {
        "definition": "lambda at which lambda*mean(R) == mean(D), i.e. mean(D)/mean(R)",
        "from_fitted_rate": balance_fitted,
        "from_initial_rate": balance_init,
        "recommended_basis": "from_fitted_rate",
        "why": (
            "The initialization-time R reflects an unfitted Laplace prior (loc=0, "
            "scale=1) and collapses within the first optimizer steps of any real "
            "run; the fitted R is the magnitude that actually persists through "
            "training, so it is the one a lambda must balance against."
        ),
    }
    print(f"\n      lambda_balance (fitted R):  {balance_fitted:.6e}")
    print(f"      lambda_balance (init R):    {balance_init:.6e}")

    # --- Phase 3: sanity checks -----------------------------------------
    print("\n[4/5] Running rate-estimator sanity checks on the real model...")
    report["sanity_checks"] = _sanity_checks(
        model, fresh_train_loader(), device, rate_estimator=rate_estimator,
        learning_rate=defaults.learning_rate, seed=args.seed,
    )
    for name, block in report["sanity_checks"].items():
        verdict = all(v for v in block.values() if isinstance(v, bool))
        print(f"      {'PASS' if verdict else 'FAIL'}  {name}")

    # --- Prove no model weight moved ------------------------------------
    print("\n[5/5] Verifying the autoencoder's weights are unchanged...")
    max_weight_drift = max(
        (weights_before[name] - p.detach()).abs().max().item()
        for name, p in model.named_parameters()
    )
    report["model_weights_unchanged"] = {
        "max_abs_drift": max_weight_drift,
        "unchanged": max_weight_drift == 0.0,
    }
    print(f"      max |drift| = {max_weight_drift} "
          f"({'unchanged' if max_weight_drift == 0.0 else 'CHANGED - BUG'})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "dr_diagnostic.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDiagnostic written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
