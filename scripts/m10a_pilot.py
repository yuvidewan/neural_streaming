"""M10A Phase 4: four 500-step arms with the scale-tracked rate proxy.

WHAT THIS RUNS
---------------
Four independent 500-step fine-tunes from the same M8-QAT checkpoint, differing
only in lambda, all with `--rate-track-scale` enabled:

    CTRL      lambda = 0            separates DAVIS fine-tuning from the rate term
    M10A-L    lambda = 9.0757e-04
    M10A-M    lambda = 2.8700e-03
    M10A-H    lambda = 9.0757e-03

Same lambdas as M9's final runs, deliberately: the whole question is whether
the SAME rate pressure now behaves differently, so changing the lambdas would
confound the comparison. This is a 500-step diagnostic pilot, not a final
training run.

HOW TRAINING IS RUN
--------------------
Through `scripts/train_autoencoder.py`'s own `main()`, unmodified - so the
production path is what is under test, including the two-parameter-group
optimizer (M9C.1), objective-aware best-checkpoint selection (M9F.1), and
`resume_model_only`.

Per-step instrumentation comes from `_RecordingRateEstimator`, a subclass that
overrides `forward`/`update_bin_width` to append to a log and then defers to
`super()`. It changes no numerics - the assertions in
`tests/test_scripts_m10a.py` pin that - and is injected into the training
script's module namespace only for the duration of an arm. Observation only.

RATE IS NOT BITRATE
--------------------
Every `R` here is the differentiable Laplace proxy in bits per input pixel.
Actual `.nvc` bitrate comes only from `scripts/m10a_evaluate.py`, after fresh
per-model calibration.

A NOTE ON COMPARING R ACROSS ARMS
----------------------------------
With `track_scale=True` each arm's bin width follows its OWN latent, so each
arm's R is measured against its own grid. That is deliberate - it mirrors the
deployed codec, which calibrates per model - but it means R is "bits under this
model's own grid", not bits on a shared ruler. The comparison that matters is
therefore R's ORDERING against actual BPP's ordering, which is exactly what
Phase 7 tests.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\m10a_pilot.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
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
EXPECTED_START_SHA256 = "90d51157356953db85d01441508526e02cf00645992c7815df755913aadaa776"
DEFAULT_CALIBRATION = Path("outputs/calibration/vimeo_epoch17_4bit.json")
DEFAULT_OUTPUT_DIR = Path("outputs/m10a_pilot")

ARMS: tuple[dict[str, Any], ...] = (
    {"name": "CTRL", "lambda": 0.0, "dir": "control_lambda0", "role": "control (distortion-only)"},
    {"name": "M10A-L", "lambda": 9.0757e-04, "dir": "lambda_9e-4", "role": "quality-preserving"},
    {"name": "M10A-M", "lambda": 2.8700e-03, "dir": "lambda_2.87e-3", "role": "balanced"},
    {"name": "M10A-H", "lambda": 9.0757e-03, "dir": "lambda_9e-3", "role": "aggressive"},
)

# Filled by _RecordingRateEstimator during an arm, drained after it.
_STEP_LOG: list[dict[str, float]] = []


class _RecordingRateEstimator(RateEstimator):
    """RateEstimator that logs per-step statistics. Numerics unchanged.

    Both overrides call `super()` for all real work; they only read tensors
    that already exist. `update_bin_width` is called by the trainer AFTER
    `optimizer.step()`, so the gradients still on `loc`/`log_scale` at that
    moment are the ones the step just consumed - which is why the rate
    estimator's gradient norm can be captured here without touching the
    training loop.
    """

    def forward(self, z: torch.Tensor, image_pixels: int) -> torch.Tensor:
        rate = super().forward(z, image_pixels)
        with torch.no_grad():
            flat = z.detach().flatten()
            # Staged, not appended: `validate_one_epoch_with_rate` also calls
            # forward(), and its batches are NOT training steps. Only
            # update_bin_width - which the trainer calls exclusively on the
            # training path - commits a record, so validation passes overwrite
            # this slot harmlessly and never enter the log.
            self._pending = {
                "rate_bpp_proxy": rate.item(),
                "latent_mean": flat.mean().item(),
                "latent_std": flat.std().item(),
                "latent_abs_mean": flat.abs().mean().item(),
                "latent_range": (flat.max() - flat.min()).item(),
                "bin_width_before": self.bin_width.mean().item(),
            }
        return rate

    @torch.no_grad()
    def update_bin_width(self, z: torch.Tensor) -> None:
        super().update_bin_width(z)
        pending = getattr(self, "_pending", None)
        if pending is None:
            return
        record = dict(pending)
        record["step"] = len(_STEP_LOG) + 1
        record["bin_width_after"] = self.bin_width.mean().item()
        record["loc_mean"] = self.loc.mean().item()
        record["loc_std"] = self.loc.std().item()
        record["scale_mean"] = torch.exp(self.log_scale).mean().item()
        record["scale_std"] = torch.exp(self.log_scale).std().item()
        record["rate_grad_norm"] = float(
            sum(
                p.grad.pow(2).sum().item()
                for p in (self.loc, self.log_scale) if p.grad is not None
            ) ** 0.5
        )
        _STEP_LOG.append(record)
        self._pending = None


def arms_for_lambdas(values: list[float], *, prefix: str) -> list[dict[str, Any]]:
    """Build an arm list for an arbitrary lambda sweep (Milestone 10B).

    M10A's own `ARMS` are left exactly as they were - passing no `--lambdas`
    reproduces that run unchanged. M10B needs the same harness at three lower
    lambdas, and forking the file would mean two copies of the training,
    instrumentation and best-checkpoint logic drifting apart, so the sweep is
    parameterised instead.

    Directory names are derived from the lambda itself (`lambda_3e-04`) rather
    than an index, so an arm's output directory always says which lambda
    produced it even if the sweep is later reordered or extended.
    """
    if not values:
        raise ValueError("at least one lambda is required")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate lambdas would collide on disk: {values}")
    return [
        {
            "name": f"{prefix}-{index}",
            "lambda": value,
            "dir": f"lambda_{value:.0e}".replace("+", ""),
            "role": f"lambda {value:.4e}",
        }
        for index, value in enumerate(values, start=1)
    ]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10A Phase 4: four 500-step arms with the scale-tracked rate proxy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--expect-sha256", default=EXPECTED_START_SHA256,
        help="Refuse to run unless the start checkpoint hashes to this. Pass '' to skip.",
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--qat-bits", type=int, default=4)
    parser.add_argument("--qat-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-batches", type=int, default=100, help="5 x 100 = 500 steps per arm.")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--rate-lr", type=float, default=defaults.rate_lr)
    parser.add_argument("--scale-momentum", type=float, default=defaults.rate_scale_momentum)
    parser.add_argument("--latent-channels", type=int, default=defaults.latent_channels)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only", nargs="+", default=None, metavar="NAME")
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=None, metavar="LAMBDA",
        help="Run this lambda sweep instead of M10A's own four arms. Omitted (the "
             "default) reproduces M10A exactly. Used by M10B to sweep below M10A-L.",
    )
    parser.add_argument(
        "--arm-prefix", default="M10B",
        help="Name prefix for --lambdas arms (M10B-1, M10B-2, ...). Ignored without --lambdas.",
    )
    parser.add_argument(
        "--no-control", action="store_true",
        help="Omit the lambda=0 control. Valid only when the control already exists from "
             "a previous run in the comparison set - M10B reuses M10A's.",
    )
    return parser


def _resolve_device(name: str) -> torch.device:
    return get_device() if name == "auto" else torch.device(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": values.mean().item(), "std": values.std().item(),
        "min": values.min().item(), "max": values.max().item(),
        "abs_mean": values.abs().mean().item(), "abs_max": values.abs().max().item(),
        "range": (values.max() - values.min()).item(),
    }


def _evaluate(
    checkpoint_path: Path, *, manifest: Path, noise: QuantizationNoise, lambda_rate: float,
    batch_size: int, num_workers: int, crop_size: int | None, device: torch.device, seed: int,
    scale_momentum: float,
) -> dict[str, Any]:
    """Post-hoc validation metrics, latent/estimator statistics and gradient norms."""
    seed_everything(seed)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device, eval_mode=True)
    model.quantization_noise = noise

    extra = checkpoint.get("extra") or {}
    estimator = RateEstimator(
        noise.scale, bits=noise.bits, mode=noise.mode,
        track_scale=bool(extra.get("rate_track_scale", False)), scale_momentum=scale_momentum,
    ).to(device)
    restored = "rate_estimator_state_dict" in extra
    if restored:
        # Carries the arm's own trained loc/log_scale AND its tracked bin_width
        # (a registered buffer), so the arm is scored on the grid it learned.
        estimator.load_state_dict(extra["rate_estimator_state_dict"])

    loader = create_val_loader(
        manifest, batch_size=batch_size, num_workers=num_workers, crop_size=crop_size,
    )
    distortions, rates, psnrs, latent_values = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            latent = model.encode(batch)
            reconstruction = model.decode(latent)
            distortions.append(mse(reconstruction, batch).item())
            rates.append(estimator(latent, batch.shape[-2] * batch.shape[-1]).item())
            psnrs.append(psnr(reconstruction, batch).item())
            latent_values.append(latent.flatten().cpu())

    # Gradient norms on one fixed batch under this arm's own lambda.
    model.train()
    batch = next(iter(loader)).to(device)
    latent = noise.apply(model.encode(batch))
    rate = estimator(latent, batch.shape[-2] * batch.shape[-1])
    loss = mse(model.decode(latent), batch) + lambda_rate * rate
    model.zero_grad()
    estimator.zero_grad()
    loss.backward()

    def grad_norm(parameters) -> float:
        return float(sum(p.grad.pow(2).sum().item() for p in parameters if p.grad is not None) ** 0.5)

    val_d = sum(distortions) / len(distortions)
    val_r = sum(rates) / len(rates)
    return {
        "val_distortion": val_d,
        "val_rate_bpp_proxy": val_r,
        "val_total_objective": val_d + lambda_rate * val_r,
        "val_psnr_db": sum(psnrs) / len(psnrs),
        "val_batches": len(distortions),
        "rate_estimator_state_restored": restored,
        "final_bin_width_mean": estimator.bin_width.mean().item(),
        "final_bin_width_min": estimator.bin_width.min().item(),
        "final_bin_width_max": estimator.bin_width.max().item(),
        "rate_estimator_loc": _stats(estimator.loc.detach().flatten()),
        "rate_estimator_scale": _stats(torch.exp(estimator.log_scale.detach()).flatten()),
        "latent": _stats(torch.cat(latent_values)),
        "model_grad_norm": grad_norm(model.parameters()),
        "rate_grad_norm": grad_norm(estimator.parameters()),
        "all_finite": bool(
            math.isfinite(val_d) and math.isfinite(val_r)
            and torch.isfinite(torch.cat(latent_values)).all()
            and torch.isfinite(estimator.loc).all() and torch.isfinite(estimator.log_scale).all()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint),
                        ("--calibration", args.calibration)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    start_hash = _sha256(args.checkpoint)
    if args.expect_sha256 and start_hash != args.expect_sha256:
        print(f"[ERROR] start checkpoint sha256 mismatch.\n  expected {args.expect_sha256}\n"
              f"  actual   {start_hash}", file=sys.stderr)
        return 1

    if args.lambdas is not None:
        if any(value <= 0 or not math.isfinite(value) for value in args.lambdas):
            parser.error("every --lambdas value must be finite and > 0")
        arms = arms_for_lambdas(args.lambdas, prefix=args.arm_prefix)
        if not args.no_control:
            arms = [dict(ARMS[0])] + arms
    else:
        arms = list(ARMS)
        if args.no_control:
            arms = [arm for arm in arms if arm["lambda"] != 0.0]
    if args.only is not None:
        wanted = {name.upper() for name in args.only}
        arms = [arm for arm in arms if arm["name"].upper() in wanted]
        if not arms:
            parser.error(f"--only matched no arms; choose from {[a['name'] for a in arms]}")

    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        noise = QuantizationNoise.from_calibration(
            args.calibration, bits=args.qat_bits, mode=args.qat_mode,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print("=" * 80)
    print("M10A PHASE 4 - scale-tracked rate proxy, 500-step controlled pilot")
    print("=" * 80)
    print(f"start checkpoint: {args.checkpoint}")
    print(f"  sha256 verified: {start_hash}")
    print(f"budget: {args.epochs} x {args.max_batches} = {args.epochs * args.max_batches} steps/arm")
    print(f"model lr {args.learning_rate} | rate lr {args.rate_lr} | "
          f"track_scale=True momentum {args.scale_momentum}")
    print(f"seed {args.seed} | batch {args.batch_size} | QAT {args.qat_bits}-bit/{args.qat_mode} | {device}")
    print("R is the training proxy, NOT .nvc bitrate.")
    print("=" * 80)

    trainer_script = _load_script("train_autoencoder")
    original_estimator_class = trainer_script.RateEstimator
    rows: list[dict[str, Any]] = []

    for index, arm in enumerate(arms, start=1):
        lambda_rate = arm["lambda"]
        arm_dir = args.output_dir / arm["dir"]
        print(f"\n[{index}/{len(arms)}] {arm['name']}  lambda={lambda_rate:.4e}  -> {arm_dir}")

        _STEP_LOG.clear()
        # Observation-only substitution, reverted in the finally below.
        trainer_script.RateEstimator = _RecordingRateEstimator
        try:
            started = time.time()
            exit_code = trainer_script.main([
                "--manifest", str(args.manifest),
                "--epochs", str(args.epochs), "--max-batches", str(args.max_batches),
                "--batch-size", str(args.batch_size),
                "--learning-rate", repr(args.learning_rate),
                "--latent-channels", str(args.latent_channels),
                "--seed", str(args.seed), "--device", str(device),
                "--resume", str(args.checkpoint), "--resume-model-only",
                "--qat-enabled", "--qat-bits", str(args.qat_bits), "--qat-mode", args.qat_mode,
                "--qat-calibration", str(args.calibration),
                "--rate-enabled", "--rate-lambda", repr(lambda_rate),
                "--rate-lr", repr(args.rate_lr),
                "--rate-track-scale", "--rate-scale-momentum", repr(args.scale_momentum),
                "--checkpoint-dir", str(arm_dir),
            ])
            elapsed = time.time() - started
        finally:
            trainer_script.RateEstimator = original_estimator_class

        if exit_code != 0:
            print(f"[ERROR] {arm['name']} failed with exit code {exit_code}", file=sys.stderr)
            return 1

        steps = [dict(record) for record in _STEP_LOG]
        (arm_dir / "step_log.json").write_text(json.dumps(steps, indent=2), encoding="utf-8")

        history = json.loads((arm_dir / "history.json").read_text(encoding="utf-8"))
        own = [record for record in history if record.get("rate_enabled")]
        if not own:
            print(f"[ERROR] {arm['name']} recorded no rate-enabled history", file=sys.stderr)
            return 1

        best_path = arm_dir / "best.pt"
        if not best_path.is_file():
            print(f"[ERROR] {arm['name']} never wrote best.pt", file=sys.stderr)
            return 1

        best_record = min(own, key=lambda record: record["val_loss"])
        # The M9F.1 guarantee: selection must consider only this run's own
        # objective, never the 40 resumed pure-MSE records from M8.
        stale = [record for record in history if not record.get("rate_enabled")]
        selection_ok = best_record["val_loss"] == min(record["val_loss"] for record in own)

        measured = _evaluate(
            best_path, manifest=args.manifest, noise=noise, lambda_rate=lambda_rate,
            batch_size=args.batch_size, num_workers=defaults.num_workers,
            crop_size=defaults.random_crop_size, device=device, seed=args.seed,
            scale_momentum=args.scale_momentum,
        )

        first_step, last_step = steps[0], steps[-1]
        row = {
            "name": arm["name"], "role": arm["role"], "lambda": lambda_rate,
            "checkpoint_dir": str(arm_dir), "best_checkpoint": str(best_path),
            "best_checkpoint_sha256": _sha256(best_path),
            "start_checkpoint_sha256": start_hash,
            "model_learning_rate": args.learning_rate, "rate_lr": args.rate_lr,
            "scale_momentum": args.scale_momentum, "seed": args.seed,
            "steps": len(steps), "epochs_run": len(own), "best_epoch": best_record["epoch"],
            "best_selection_value": best_record["val_loss"],
            "best_selection_used_only_current_objective": selection_ok,
            "stale_history_records_ignored": len(stale),
            "train_distortion_first_epoch": own[0]["train_distortion"],
            "train_distortion_last_epoch": own[-1]["train_distortion"],
            "train_rate_proxy_first_epoch": own[0]["train_rate_bpp"],
            "train_rate_proxy_last_epoch": own[-1]["train_rate_bpp"],
            "train_total_first_epoch": own[0]["train_loss"],
            "train_total_last_epoch": own[-1]["train_loss"],
            # The M10A question in two numbers: how far the latent moved, and
            # whether the bin width followed it.
            "step1_latent_abs_mean": first_step["latent_abs_mean"],
            "final_latent_abs_mean": last_step["latent_abs_mean"],
            "step1_latent_range": first_step["latent_range"],
            "final_latent_range": last_step["latent_range"],
            "step1_bin_width": first_step["bin_width_before"],
            "final_bin_width": last_step["bin_width_after"],
            "step1_rate_proxy": first_step["rate_bpp_proxy"],
            "final_rate_proxy": last_step["rate_bpp_proxy"],
            "elapsed_seconds": elapsed,
            **measured,
        }
        rows.append(row)

        shrink = (
            row["step1_latent_abs_mean"] / row["final_latent_abs_mean"]
            if row["final_latent_abs_mean"] > 0 else float("nan")
        )
        bin_shrink = (
            row["step1_bin_width"] / row["final_bin_width"] if row["final_bin_width"] > 0 else float("nan")
        )
        print(f"  best epoch {row['best_epoch']} ({elapsed / 60:.1f} min), "
              f"selection ok: {selection_ok}, stale records ignored: {len(stale)}")
        print(f"  val D {measured['val_distortion']:.6e}  R_proxy {measured['val_rate_bpp_proxy']:.4f}  "
              f"PSNR {measured['val_psnr_db']:.2f} dB")
        print(f"  latent abs-mean {row['step1_latent_abs_mean']:.4f} -> "
              f"{row['final_latent_abs_mean']:.4f} ({shrink:.2f}x)   "
              f"bin width {row['step1_bin_width']:.4f} -> {row['final_bin_width']:.4f} ({bin_shrink:.2f}x)")
        if not measured["all_finite"]:
            print(f"[ERROR] {arm['name']} produced non-finite values", file=sys.stderr)
            return 1

    summary = {
        "phase": "M10A Phase 4 (scale-tracked 500-step pilot)",
        "start_checkpoint": str(args.checkpoint),
        "start_checkpoint_sha256": start_hash,
        "calibration_for_qat_and_initial_bin_width": str(args.calibration),
        "qat_bits": args.qat_bits, "qat_mode": args.qat_mode,
        "epochs": args.epochs, "max_batches": args.max_batches,
        "steps_per_arm": args.epochs * args.max_batches,
        "batch_size": args.batch_size,
        "model_learning_rate": args.learning_rate, "rate_lr": args.rate_lr,
        "track_scale": True, "scale_momentum": args.scale_momentum,
        "seed": args.seed, "device": str(device),
        "note": (
            "val_rate_bpp_proxy is the differentiable Laplace training proxy in bits per "
            "input pixel, measured against each arm's OWN tracked bin width. It is NOT "
            ".nvc bitrate and no bitrate claim follows from it."
        ),
        "arms": rows,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "training_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [key for key in rows[0] if not isinstance(rows[0][key], dict)]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\n" + "=" * 80)
    print(f"{'arm':<8} {'lambda':>11} {'val D':>11} {'R*':>8} {'PSNR':>7} "
          f"{'latent|.|':>10} {'bin width':>10}")
    for row in rows:
        print(f"{row['name']:<8} {row['lambda']:>11.4e} {row['val_distortion']:>11.4e} "
              f"{row['val_rate_bpp_proxy']:>8.4f} {row['val_psnr_db']:>7.2f} "
              f"{row['final_latent_abs_mean']:>10.4f} {row['final_bin_width']:>10.4f}")
    print("=" * 80)
    print("* proxy, measured against each arm's own tracked grid - NOT .nvc bitrate.")
    print(f"\nSummary: {args.output_dir / 'training_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
