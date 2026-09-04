"""Milestone 9 final training: three rate-aware models, plus a lambda=0 control.

WHAT THIS RUNS
---------------
Four independent fine-tuning runs, each starting from the SAME M8 QAT
checkpoint with `--resume-model-only` (weights restored, optimizer fresh), and
differing only in lambda:

    M9-L   lambda = 9.0757e-04    quality-preserving
    M9-M   lambda = 2.8700e-03    balanced
    M9-H   lambda = 9.0757e-03    aggressive
    CTRL   lambda = 0            control - see below

None is a continuation of another.

WHY A LAMBDA=0 CONTROL IS INCLUDED
-----------------------------------
The scientific question is whether *rate-aware training* improves the deployed
RD tradeoff over M8 QAT. But these runs also fine-tune on DAVIS, while the M8
QAT checkpoint was trained on Vimeo. Any M8 -> M9 difference therefore has two
possible causes, and comparing only against M8 cannot separate them:

    (a) the rate term did something          <- the milestone's actual question
    (b) 30 more epochs on DAVIS did something <- a confound

The control receives byte-identical treatment except that lambda is 0, so it
measures (b) alone. Every rate-aware claim in the final report is made against
the control as well as against M8. It is a measurement instrument, not a
fourth deliverable model, and it is not part of the lambda sweep.

BUDGET
-------
30 full epochs over the DAVIS train split (604 batches/epoch = 18,120 optimizer
steps per model), identical for every arm. That is 36x the M9C.1 pilot's 500
steps and comparable in epoch count to M8's 40. No arm is stopped early, and no
arm is shortened because it looks worse - all four get the same opportunity.

RATE IS NOT BITRATE
--------------------
`R` recorded here is the differentiable Laplace training proxy in bits per
input pixel. It is NOT `.nvc` bitrate, and nothing in this script's output
supports a bitrate claim. Actual bitrate comes only from fresh calibration plus
real encoding, in the benchmark that follows this.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\m9_final_train.py
    python scripts\\m9_final_train.py --epochs 5      # a shorter rehearsal
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
DEFAULT_CALIBRATION = Path("outputs/calibration/vimeo_epoch17_4bit.json")
DEFAULT_OUTPUT_DIR = Path("outputs/m9_final")

# Selected in M9C.1 section 9C1.8 from the measured D/R balance point
# (9.0757e-04) and the observed rate-reduction-per-dB efficiency curve.
# Directory names are the ones the M9 final-training brief specifies.
ARMS: tuple[dict[str, Any], ...] = (
    {"name": "M9-L", "lambda": 9.0757e-04, "dir": "lambda_9e-4", "role": "quality-preserving"},
    {"name": "M9-M", "lambda": 2.8700e-03, "dir": "lambda_2.87e-3", "role": "balanced"},
    {"name": "M9-H", "lambda": 9.0757e-03, "dir": "lambda_9e-3", "role": "aggressive"},
    {"name": "CTRL", "lambda": 0.0, "dir": "control_lambda0", "role": "control (distortion-only)"},
)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Milestone 9 final training: three rate-aware models (plus a lambda=0 "
            "control), each independently fine-tuned from the M8 QAT checkpoint."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Start model for EVERY arm. Never modified.",
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--qat-bits", type=int, default=4)
    parser.add_argument("--qat-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--epochs", type=int, default=30, help="Full epochs per arm.")
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Batches per epoch. None (the default) means the full train split.",
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--rate-lr", type=float, default=defaults.rate_lr)
    parser.add_argument("--latent-channels", type=int, default=defaults.latent_channels)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-control", action="store_true",
        help="Omit the lambda=0 control. Not recommended - see the module docstring.",
    )
    parser.add_argument(
        "--only", nargs="+", default=None, metavar="NAME",
        help="Run only these arms by name (M9-L, M9-M, M9-H, CTRL).",
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


def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": values.mean().item(), "std": values.std().item(),
        "min": values.min().item(), "max": values.max().item(),
        "abs_mean": values.abs().mean().item(), "abs_max": values.abs().max().item(),
    }


def _evaluate(
    checkpoint_path: Path, *, manifest: Path, noise: QuantizationNoise, lambda_rate: float,
    batch_size: int, num_workers: int, crop_size: int | None, device: torch.device, seed: int,
) -> dict[str, Any]:
    """Validation D / R / total / PSNR plus latent and estimator statistics.

    Eval mode (no QAT noise) - the regime the deployed encoder runs in, and the
    same convention `validate_one_epoch_with_rate` uses.
    """
    seed_everything(seed)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device, eval_mode=True)
    model.quantization_noise = noise

    estimator = RateEstimator(noise.scale, bits=noise.bits, mode=noise.mode).to(device)
    extra = checkpoint.get("extra") or {}
    restored = "rate_estimator_state_dict" in extra
    if restored:
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

    val_d = sum(distortions) / len(distortions)
    val_r = sum(rates) / len(rates)
    loc = estimator.loc.detach().flatten()
    scale = torch.exp(estimator.log_scale.detach()).flatten()
    return {
        "val_distortion": val_d,
        "val_rate_bpp_proxy": val_r,
        "val_total_objective": val_d + lambda_rate * val_r,
        "val_psnr_db": sum(psnrs) / len(psnrs),
        "val_batches": len(distortions),
        "rate_estimator_state_restored": restored,
        "rate_estimator_loc": _tensor_stats(loc),
        "rate_estimator_scale": _tensor_stats(scale),
        "latent": _tensor_stats(torch.cat(latent_values)),
        "all_finite": bool(
            math.isfinite(val_d) and math.isfinite(val_r)
            and torch.isfinite(loc).all() and torch.isfinite(scale).all()
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
    if not math.isfinite(args.rate_lr) or args.rate_lr <= 0:
        parser.error("--rate-lr must be finite and > 0")

    arms = [arm for arm in ARMS if not (args.skip_control and arm["lambda"] == 0.0)]
    if args.only is not None:
        wanted = {name.upper() for name in args.only}
        arms = [arm for arm in arms if arm["name"].upper() in wanted]
        if not arms:
            parser.error(f"--only matched no arms; choose from {[a['name'] for a in ARMS]}")

    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        noise = QuantizationNoise.from_calibration(
            args.calibration, bits=args.qat_bits, mode=args.qat_mode,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    start_hash = _sha256(args.checkpoint)
    print("=" * 78)
    print("M9 FINAL TRAINING")
    print("=" * 78)
    print(f"start checkpoint: {args.checkpoint}")
    print(f"  sha256: {start_hash}")
    print(f"budget: {args.epochs} epochs"
          + (f" x {args.max_batches} batches" if args.max_batches else " (full train split)")
          + " - identical for every arm")
    print(f"model lr {args.learning_rate} | rate lr {args.rate_lr} | batch {args.batch_size} | "
          f"seed {args.seed} | QAT {args.qat_bits}-bit/{args.qat_mode} | {device}")
    for arm in arms:
        print(f"  {arm['name']:<5} lambda={arm['lambda']:.4e}  ({arm['role']})")
    print("R below is the TRAINING PROXY, not .nvc bitrate.")
    print("=" * 78)

    trainer = _load_script("train_autoencoder")
    rows: list[dict[str, Any]] = []

    for index, arm in enumerate(arms, start=1):
        lambda_rate = arm["lambda"]
        arm_dir = args.output_dir / arm["dir"]
        print(f"\n{'=' * 78}\n[{index}/{len(arms)}] {arm['name']}  lambda={lambda_rate:.4e}  -> {arm_dir}\n{'=' * 78}")

        argv_arm = [
            "--manifest", str(args.manifest),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", repr(args.learning_rate),
            "--latent-channels", str(args.latent_channels),
            "--seed", str(args.seed),
            "--device", str(device),
            "--resume", str(args.checkpoint), "--resume-model-only",
            "--qat-enabled", "--qat-bits", str(args.qat_bits), "--qat-mode", args.qat_mode,
            "--qat-calibration", str(args.calibration),
            "--rate-enabled", "--rate-lambda", repr(lambda_rate),
            "--rate-lr", repr(args.rate_lr),
            "--checkpoint-dir", str(arm_dir),
        ]
        if args.max_batches is not None:
            argv_arm += ["--max-batches", str(args.max_batches)]

        started = time.time()
        exit_code = trainer.main(argv_arm)
        elapsed = time.time() - started
        if exit_code != 0:
            print(f"[ERROR] {arm['name']} failed with exit code {exit_code}", file=sys.stderr)
            return 1

        history = json.loads((arm_dir / "history.json").read_text(encoding="utf-8"))
        own = [record for record in history if record.get("rate_enabled")]
        if not own:
            print(f"[ERROR] {arm['name']} wrote no rate-enabled history records", file=sys.stderr)
            return 1

        best_path = arm_dir / "best.pt"
        if not best_path.is_file():
            # This is the M9 pre-flight defect's signature. It must not recur
            # silently: without best.pt there is no principled model to deploy.
            print(
                f"[ERROR] {arm['name']} never wrote best.pt - best-checkpoint selection "
                "did not fire. See tests/test_m9_checkpoint_selection.py.",
                file=sys.stderr,
            )
            return 1

        best_record = min(own, key=lambda record: record["val_loss"])
        measured = _evaluate(
            best_path, manifest=args.manifest, noise=noise, lambda_rate=lambda_rate,
            batch_size=args.batch_size, num_workers=defaults.num_workers,
            crop_size=defaults.random_crop_size, device=device, seed=args.seed,
        )
        first = own[0]

        row = {
            "name": arm["name"],
            "role": arm["role"],
            "lambda": lambda_rate,
            "checkpoint_dir": str(arm_dir),
            "best_checkpoint": str(best_path),
            "best_checkpoint_sha256": _sha256(best_path),
            "latest_checkpoint_sha256": _sha256(arm_dir / "latest.pt"),
            "start_checkpoint": str(args.checkpoint),
            "start_checkpoint_sha256": start_hash,
            "model_learning_rate": args.learning_rate,
            "rate_lr": args.rate_lr,
            "seed": args.seed,
            "epochs_run": len(own),
            "best_epoch": best_record["epoch"],
            # Training-side, from the run's own history.
            "train_distortion_first": first["train_distortion"],
            "train_distortion_best": best_record["train_distortion"],
            "train_rate_bpp_proxy_first": first["train_rate_bpp"],
            "train_rate_bpp_proxy_best": best_record["train_rate_bpp"],
            "train_total_first": first["train_loss"],
            "train_total_best": best_record["train_loss"],
            "val_total_at_best_epoch": best_record["val_loss"],
            "elapsed_seconds": elapsed,
            **measured,
        }
        rows.append(row)

        print(f"\n  best epoch {row['best_epoch']} (of {row['epochs_run']}), {elapsed / 60:.1f} min")
        print(f"  val D {measured['val_distortion']:.6e}  R_proxy {measured['val_rate_bpp_proxy']:.4f}  "
              f"total {measured['val_total_objective']:.6e}  PSNR {measured['val_psnr_db']:.2f} dB")
        print(f"  estimator scale mean {measured['rate_estimator_scale']['mean']:.4f}  "
              f"latent abs-mean {measured['latent']['abs_mean']:.4f}")
        if not measured["all_finite"]:
            print(f"[ERROR] {arm['name']} produced non-finite values", file=sys.stderr)
            return 1

    summary = {
        "phase": "M9 final training",
        "start_checkpoint": str(args.checkpoint),
        "start_checkpoint_sha256": start_hash,
        "calibration_for_qat_and_rate_bin_width": str(args.calibration),
        "qat_bits": args.qat_bits,
        "qat_mode": args.qat_mode,
        "epochs": args.epochs,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "model_learning_rate": args.learning_rate,
        "rate_lr": args.rate_lr,
        "seed": args.seed,
        "device": str(device),
        "note": (
            "val_rate_bpp_proxy is the differentiable Laplace training proxy in bits per "
            "input pixel. It is NOT .nvc bitrate; no bitrate claim follows from it. "
            "Actual bitrate requires fresh calibration and real encoding."
        ),
        "arms": rows,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    with (args.output_dir / "training_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[key for key in rows[0] if not isinstance(rows[0][key], dict)],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if not isinstance(v, dict)})

    print("\n" + "=" * 78)
    print(f"{'model':<6} {'lambda':>11} {'best ep':>8} {'val D':>11} {'val R*':>8} "
          f"{'val total':>11} {'PSNR':>7}")
    for row in rows:
        print(f"{row['name']:<6} {row['lambda']:>11.4e} {row['best_epoch']:>8} "
              f"{row['val_distortion']:>11.4e} {row['val_rate_bpp_proxy']:>8.4f} "
              f"{row['val_total_objective']:>11.4e} {row['val_psnr_db']:>7.2f}")
    print("=" * 78)
    print("* R is the training proxy, NOT .nvc bitrate.")
    print(f"\nSummary: {args.output_dir / 'training_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
