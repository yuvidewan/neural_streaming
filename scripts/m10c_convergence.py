"""M10C: does M10A-L's rate-aware RD advantage survive to convergence?

THE QUESTION
-------------
M10A measured -6.59% BD-rate for lambda=9.0757e-04 versus a lambda=0 control,
and M10B established that the optimum is bracketed there (every lower lambda
was worse). But every one of those numbers came from a 500-step pilot, and M9
Section 9F showed pilot-scale RD orderings can reverse by convergence. So the
-6.59% is not yet a durable result, and this script is what decides that.

Two arms only - lambda=0 and lambda=9.0757e-04 - trained to the project's
established full-training budget, with checkpoints retained along the way so
the advantage can be tracked as a function of training budget rather than read
off a single endpoint.

THE BUDGET, AND WHY THESE CHECKPOINTS
--------------------------------------
The full-training configuration is the one that produced the M9 final runs: 30
epochs over the full DAVIS train split. At 604 batches/epoch that is **18,120
optimizer steps**, which is short of the 20,000 the brief asks for as a final
milestone - so the last retained checkpoint is the run's own final one at
18,120 rather than an invented longer schedule.

Checkpoints are written per epoch by `train_autoencoder.py`, so the requested
step milestones are served by the nearest epoch boundary rather than by
changing the optimization to stop mid-epoch:

    requested   500    2,000   5,000   10,000   20,000
    epoch         1        3       8       17       30
    actual step 604    1,812   4,832   10,268   18,120

HOW SNAPSHOTS ARE TAKEN WITHOUT TOUCHING THE TRAINER
-----------------------------------------------------
`train_autoencoder.py` is left byte-identical. This driver wraps the
`save_checkpoint` symbol in that script's module namespace so that, after the
real save runs, `latest.pt` is additionally copied to `snapshot_step<N>.pt` on
the target epochs. That is pure extra I/O after the fact - it reads no state,
changes no numerics, and cannot affect the optimization. Same precedent as
M10A's `_RecordingRateEstimator`, and pinned by tests.

FAIRNESS
---------
Both arms get identical budgets, identical data exposure, identical seeds and
identical evaluation. Every comparison in the analysis is paired by training
budget - a snapshot is only ever compared against the other arm's snapshot at
the same step count.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\m10c_convergence.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

from nvc.training import QuantizationNoise
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device

DEFAULT_CHECKPOINT = Path("outputs/qat_combined/checkpoints_qat_noise/best.pt")
EXPECTED_START_SHA256 = "90d51157356953db85d01441508526e02cf00645992c7815df755913aadaa776"
DEFAULT_CALIBRATION = Path("outputs/calibration/vimeo_epoch17_4bit.json")
DEFAULT_OUTPUT_DIR = Path("outputs/m10c_convergence")

# The M9-final budget, unchanged.
DEFAULT_EPOCHS = 30
# Epoch boundaries nearest the brief's requested step milestones. See the
# module docstring for the mapping; epoch 30 stands in for 20,000 because the
# established budget stops at 18,120.
SNAPSHOT_EPOCHS: tuple[int, ...] = (1, 3, 8, 17, 30)

ARMS: tuple[dict[str, Any], ...] = (
    {"name": "CTRL", "lambda": 0.0, "dir": "control_lambda0",
     "role": "control (distortion-only)"},
    {"name": "M10C-L", "lambda": 9.0757e-04, "dir": "lambda_9e-4",
     "role": "rate-aware at the M10A/M10B optimum"},
)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "M10C: train lambda=0 and lambda=9.0757e-04 to the established full budget, "
            "retaining checkpoints so the RD advantage can be tracked against training budget."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expect-sha256", default=EXPECTED_START_SHA256)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--qat-bits", type=int, default=4)
    parser.add_argument("--qat-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--snapshot-epochs", type=int, nargs="+", default=list(SNAPSHOT_EPOCHS),
        help="Own-run epochs (1-based) whose latest.pt is retained as a step snapshot.",
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--rate-lr", type=float, default=defaults.rate_lr)
    parser.add_argument("--scale-momentum", type=float, default=defaults.rate_scale_momentum)
    parser.add_argument("--latent-channels", type=int, default=defaults.latent_channels)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only", nargs="+", default=None, metavar="NAME")
    return parser


def _resolve_device(name: str) -> torch.device:
    return get_device() if name == "auto" else torch.device(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_snapshotting_save(original_save, *, target_epochs: set[int], steps_per_epoch: int,
                           first_epoch_holder: dict[str, int], recorded: list[dict[str, Any]]):
    """Wrap `save_checkpoint` so target epochs also leave a retained snapshot.

    Called only for `latest.pt` (written every epoch); `best.pt` is left alone.
    The real save happens first and its result is untouched - this only copies
    the file afterwards, so it cannot influence training in any way.

    Epoch numbers arriving here are GLOBAL (the run resumes M8's epoch 40, so
    they start at 41). The first one seen establishes the offset, which is why
    the caller passes a mutable holder rather than assuming 41.
    """

    def save(path, **kwargs):
        original_save(path, **kwargs)
        epoch = kwargs.get("epoch")
        path = Path(path)
        if path.name != "latest.pt" or epoch is None:
            return
        if first_epoch_holder.get("value") is None:
            first_epoch_holder["value"] = epoch
        own_epoch = epoch - first_epoch_holder["value"] + 1
        if own_epoch in target_epochs:
            step = own_epoch * steps_per_epoch
            destination = path.with_name(f"snapshot_step{step:06d}.pt")
            shutil.copy2(path, destination)
            recorded.append({
                "own_epoch": own_epoch, "global_epoch": epoch,
                "step": step, "path": str(destination),
            })

    return save


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

    bad = [e for e in args.snapshot_epochs if not 1 <= e <= args.epochs]
    if bad:
        parser.error(f"--snapshot-epochs outside 1..{args.epochs}: {bad}")

    arms = list(ARMS)
    if args.only is not None:
        wanted = {name.upper() for name in args.only}
        arms = [arm for arm in arms if arm["name"].upper() in wanted]
        if not arms:
            parser.error(f"--only matched no arms; choose from {[a['name'] for a in ARMS]}")

    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        QuantizationNoise.from_calibration(args.calibration, bits=args.qat_bits, mode=args.qat_mode)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    from nvc.data.loaders import create_train_loader
    steps_per_epoch = len(create_train_loader(
        args.manifest, batch_size=args.batch_size, num_workers=defaults.num_workers,
        seed=args.seed, crop_size=defaults.random_crop_size,
    ))

    print("=" * 84)
    print("M10C - TRAINING-BUDGET / CONVERGENCE VALIDATION")
    print("=" * 84)
    print(f"start checkpoint: {args.checkpoint}")
    print(f"  sha256 verified: {start_hash}")
    print(f"budget: {args.epochs} epochs x {steps_per_epoch} batches = "
          f"{args.epochs * steps_per_epoch} steps per arm (the M9-final configuration)")
    print(f"snapshots at own-epochs {args.snapshot_epochs} = steps "
          f"{[e * steps_per_epoch for e in args.snapshot_epochs]}")
    print(f"model lr {args.learning_rate} | rate lr {args.rate_lr} | "
          f"track_scale=True momentum {args.scale_momentum}")
    print(f"seed {args.seed} | batch {args.batch_size} | QAT {args.qat_bits}-bit/{args.qat_mode} | {device}")
    for arm in arms:
        print(f"  {arm['name']:<7} lambda={arm['lambda']:.4e}  ({arm['role']})")
    print("Both arms receive an IDENTICAL budget; every comparison is paired by step count.")
    print("=" * 84)

    pilot = _load_script("m10a_pilot")
    trainer_script = _load_script("train_autoencoder")
    original_estimator = trainer_script.RateEstimator
    original_save = trainer_script.save_checkpoint

    rows: list[dict[str, Any]] = []
    for index, arm in enumerate(arms, start=1):
        lambda_rate = arm["lambda"]
        arm_dir = args.output_dir / arm["dir"]
        print(f"\n{'=' * 84}\n[{index}/{len(arms)}] {arm['name']} lambda={lambda_rate:.4e} -> {arm_dir}\n{'=' * 84}")

        pilot._STEP_LOG.clear()
        snapshots: list[dict[str, Any]] = []
        first_epoch: dict[str, int] = {"value": None}
        trainer_script.RateEstimator = pilot._RecordingRateEstimator
        trainer_script.save_checkpoint = make_snapshotting_save(
            original_save, target_epochs=set(args.snapshot_epochs),
            steps_per_epoch=steps_per_epoch, first_epoch_holder=first_epoch, recorded=snapshots,
        )
        try:
            started = time.time()
            exit_code = trainer_script.main([
                "--manifest", str(args.manifest),
                "--epochs", str(args.epochs),
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
            trainer_script.RateEstimator = original_estimator
            trainer_script.save_checkpoint = original_save

        if exit_code != 0:
            print(f"[ERROR] {arm['name']} failed with exit code {exit_code}", file=sys.stderr)
            return 1

        steps = [dict(record) for record in pilot._STEP_LOG]
        (arm_dir / "step_log.json").write_text(json.dumps(steps, indent=2), encoding="utf-8")

        history = json.loads((arm_dir / "history.json").read_text(encoding="utf-8"))
        own = [record for record in history if record.get("rate_enabled")]
        stale = [record for record in history if not record.get("rate_enabled")]
        best_path = arm_dir / "best.pt"
        if not best_path.is_file():
            print(f"[ERROR] {arm['name']} never wrote best.pt", file=sys.stderr)
            return 1
        best_record = min(own, key=lambda record: record["val_loss"])

        for snapshot in snapshots:
            snapshot["sha256"] = _sha256(Path(snapshot["path"]))
            record = own[snapshot["own_epoch"] - 1]
            snapshot.update({
                "val_distortion": record["val_distortion"],
                "val_rate_bpp_proxy": record["val_rate_bpp"],
                "val_psnr_db": record["val_psnr"],
                "val_total_objective": record["val_loss"],
                "train_distortion": record["train_distortion"],
                "train_rate_bpp_proxy": record["train_rate_bpp"],
            })
            # Per-step instrumentation at the snapshot's own step index.
            if snapshot["step"] <= len(steps):
                step_record = steps[snapshot["step"] - 1]
                snapshot.update({
                    "latent_abs_mean": step_record["latent_abs_mean"],
                    "latent_std": step_record["latent_std"],
                    "latent_range": step_record["latent_range"],
                    "bin_width": step_record["bin_width_after"],
                    "rate_loc_std": step_record["loc_std"],
                    "rate_scale_mean": step_record["scale_mean"],
                    "rate_grad_norm": step_record["rate_grad_norm"],
                })

        row = {
            "name": arm["name"], "role": arm["role"], "lambda": lambda_rate,
            "checkpoint_dir": str(arm_dir),
            "start_checkpoint_sha256": start_hash,
            "best_checkpoint": str(best_path), "best_checkpoint_sha256": _sha256(best_path),
            "best_epoch": best_record["epoch"], "best_selection_value": best_record["val_loss"],
            "best_selection_used_only_current_objective": (
                best_record["val_loss"] == min(r["val_loss"] for r in own)
            ),
            "stale_history_records_ignored": len(stale),
            "epochs_run": len(own), "steps": len(steps), "steps_per_epoch": steps_per_epoch,
            "model_learning_rate": args.learning_rate, "rate_lr": args.rate_lr,
            "scale_momentum": args.scale_momentum, "seed": args.seed,
            "elapsed_seconds": elapsed,
            "snapshots": snapshots,
            "final_val_distortion": own[-1]["val_distortion"],
            "final_val_rate_bpp_proxy": own[-1]["val_rate_bpp"],
            "final_val_psnr_db": own[-1]["val_psnr"],
            "all_finite": all(
                math.isfinite(record["val_distortion"]) and math.isfinite(record["val_rate_bpp"])
                for record in own
            ),
        }
        rows.append(row)
        print(f"\n  {len(own)} epochs / {len(steps)} steps in {elapsed / 60:.1f} min, "
              f"{len(snapshots)} snapshots, best epoch {best_record['epoch']}")
        print(f"  final val D {row['final_val_distortion']:.6e}  "
              f"R_proxy {row['final_val_rate_bpp_proxy']:.4f}  PSNR {row['final_val_psnr_db']:.2f} dB")
        if not row["all_finite"]:
            print(f"[ERROR] {arm['name']} produced non-finite values", file=sys.stderr)
            return 1

    summary = {
        "phase": "M10C (training-budget / convergence validation)",
        "start_checkpoint": str(args.checkpoint), "start_checkpoint_sha256": start_hash,
        "calibration_for_qat_and_initial_bin_width": str(args.calibration),
        "qat_bits": args.qat_bits, "qat_mode": args.qat_mode,
        "epochs": args.epochs, "steps_per_epoch": steps_per_epoch,
        "total_steps": args.epochs * steps_per_epoch,
        "snapshot_epochs": list(args.snapshot_epochs),
        "snapshot_steps": [e * steps_per_epoch for e in args.snapshot_epochs],
        "batch_size": args.batch_size,
        "model_learning_rate": args.learning_rate, "rate_lr": args.rate_lr,
        "track_scale": True, "scale_momentum": args.scale_momentum,
        "seed": args.seed, "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "note": (
            "val_rate_bpp_proxy is the differentiable Laplace training proxy against each "
            "arm's own tracked bin width. It is NOT .nvc bitrate."
        ),
        "arms": rows,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "snapshots.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["arm", "lambda", "step", "own_epoch", "global_epoch", "val_distortion",
                        "val_psnr_db", "val_rate_bpp_proxy", "val_total_objective",
                        "latent_abs_mean", "latent_range", "bin_width", "rate_scale_mean",
                        "rate_grad_norm", "sha256", "path"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            for snapshot in row["snapshots"]:
                writer.writerow({"arm": row["name"], "lambda": row["lambda"], **snapshot})

    print("\n" + "=" * 84)
    print(f"{'arm':<8} {'step':>7} {'val D':>11} {'PSNR':>7} {'R*':>8} {'latent|.|':>10} {'bin w':>8}")
    for row in rows:
        for snapshot in row["snapshots"]:
            print(f"{row['name']:<8} {snapshot['step']:>7} {snapshot['val_distortion']:>11.4e} "
                  f"{snapshot['val_psnr_db']:>7.2f} {snapshot['val_rate_bpp_proxy']:>8.4f} "
                  f"{snapshot.get('latent_abs_mean', float('nan')):>10.4f} "
                  f"{snapshot.get('bin_width', float('nan')):>8.4f}")
    print("=" * 84)
    print("* proxy, NOT .nvc bitrate.")
    print(f"\nSummary: {args.output_dir / 'training_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
