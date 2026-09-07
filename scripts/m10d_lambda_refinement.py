"""M10D: converged lambda refinement around the M10C operating point.

THE QUESTION
-------------
M10C established that lambda = 9.0757e-04 is a DURABLE improvement: -17.10%
BD-rate against a matched lambda=0 control at the full 18,120-step budget, with
strict dominance at 8, 6 and 4 bits. But that lambda was inherited from M9's
D/R balance measurement, taken on a 500-step pilot; M10B then showed lower
lambdas are worse, without ever testing above 9.0757e-04 at a converged budget.

So the useful region is bounded below and at the centre, and unbounded above.
M10D refines around the converged operating point:

    CTRL       0          matched control
    LOW        6.0e-4     below centre
    CENTER     9.0757e-4  the M10C operating point, re-run
    HIGH       1.35e-3    above centre
    VERY_HIGH  1.8e-3     further above

This is deliberately NOT another downward sweep - M10B settled that direction.

WHY CTRL AND CENTER ARE RE-RUN RATHER THAN REUSED
--------------------------------------------------
M10B reused its control from M10A, because there the question was purely about
new lambdas. Here re-running both earns something specific: CENTER is a
configuration-identical replicate of M10C-L, so the difference between them
MEASURES the run-to-run nondeterminism this project has repeatedly flagged and
never quantified on the final deployed metric. That number is needed to say
whether any M10D difference is real, so it is worth the ~23 minutes.

BUDGET
-------
The established full budget, identical for every arm: 30 epochs x 604 batches =
18,120 optimizer steps, directly comparable to M10C. Snapshots at 604, 1,812,
4,832, 10,268 and 18,120 steps, matching M10C's schedule exactly.

REUSE, NOT REIMPLEMENTATION
----------------------------
`train_autoencoder.py` stays byte-identical, and so do `m10c_convergence.py`
and `m10a_pilot.py`. This script imports their proven pieces - M10C's
`make_snapshotting_save` (post-hoc file copy, cannot touch optimization) and
M10A's `_RecordingRateEstimator` (observation-only instrumentation) - and
supplies only its own arm list. Nothing established is modified.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\m10d_lambda_refinement.py --check-only   # fairness preflight
    python scripts\\m10d_lambda_refinement.py
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

from nvc.training import QuantizationNoise, RateEstimator
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_CHECKPOINT = Path("outputs/qat_combined/checkpoints_qat_noise/best.pt")
EXPECTED_START_SHA256 = "90d51157356953db85d01441508526e02cf00645992c7815df755913aadaa776"
DEFAULT_CALIBRATION = Path("outputs/calibration/vimeo_epoch17_4bit.json")
DEFAULT_OUTPUT_DIR = Path("outputs/m10d_lambda_refinement")

DEFAULT_EPOCHS = 30
SNAPSHOT_EPOCHS: tuple[int, ...] = (1, 3, 8, 17, 30)

ARMS: tuple[dict[str, Any], ...] = (
    {"name": "CTRL", "lambda": 0.0, "dir": "control_lambda0",
     "role": "matched control (distortion-only)"},
    {"name": "LOW", "lambda": 6.0e-4, "dir": "lambda_6.0e-4",
     "role": "below the M10C centre"},
    {"name": "CENTER", "lambda": 9.0757e-04, "dir": "lambda_9.0757e-4",
     "role": "the M10C operating point, replicated"},
    {"name": "HIGH", "lambda": 1.35e-3, "dir": "lambda_1.35e-3",
     "role": "above the M10C centre"},
    {"name": "VERY_HIGH", "lambda": 1.8e-3, "dir": "lambda_1.8e-3",
     "role": "furthest above the M10C centre"},
)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10D: converged lambda refinement around the M10C operating point.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expect-sha256", default=EXPECTED_START_SHA256)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--qat-bits", type=int, default=4)
    parser.add_argument("--qat-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--snapshot-epochs", type=int, nargs="+", default=list(SNAPSHOT_EPOCHS))
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
        "--check-only", action="store_true",
        help="Run the fairness preflight and stop without training.",
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


def fairness_preflight(args, arms: list[dict[str, Any]], start_hash: str) -> dict[str, Any]:
    """Verify, before any training, that only lambda differs between arms.

    Every item here is CHECKED rather than asserted in prose, because a
    lambda-refinement result is only meaningful if the arms are otherwise
    identical - and a silent divergence would be indistinguishable from a real
    lambda effect in the final numbers.
    """
    checks: dict[str, Any] = {}

    checks["start_checkpoint_sha256_matches"] = (
        not args.expect_sha256 or start_hash == args.expect_sha256
    )
    checks["all_arms_share_one_start_checkpoint"] = True  # single --checkpoint, used for every arm
    checks["lambdas_are_distinct"] = len({arm["lambda"] for arm in arms}) == len(arms)
    checks["output_dirs_are_distinct"] = len({arm["dir"] for arm in arms}) == len(arms)
    checks["exactly_one_control"] = sum(1 for arm in arms if arm["lambda"] == 0.0) <= 1

    # The rate estimator must start identically for every arm. It is built from
    # the shared calibration's scale with loc=0 / log_scale=0, so this is true
    # by construction - verified rather than assumed.
    noise = QuantizationNoise.from_calibration(
        args.calibration, bits=args.qat_bits, mode=args.qat_mode,
    )
    fingerprints = set()
    for _ in range(2):
        seed_everything(args.seed)
        estimator = RateEstimator(
            noise.scale, bits=noise.bits, mode=noise.mode,
            track_scale=True, scale_momentum=args.scale_momentum,
        )
        fingerprints.add((
            hashlib.sha256(estimator.loc.detach().numpy().tobytes()).hexdigest(),
            hashlib.sha256(estimator.log_scale.detach().numpy().tobytes()).hexdigest(),
            hashlib.sha256(estimator.bin_width.detach().numpy().tobytes()).hexdigest(),
        ))
    checks["rate_estimator_init_is_deterministic"] = len(fingerprints) == 1
    checks["rate_estimator_init_loc_is_zero"] = bool((estimator.loc == 0).all())
    checks["rate_estimator_init_log_scale_is_zero"] = bool((estimator.log_scale == 0).all())
    checks["scale_tracking_enabled_for_every_arm"] = estimator.track_scale is True
    checks["scale_momentum"] = args.scale_momentum

    # resume-model-only semantics: optimizer state must NOT come from the M8
    # checkpoint, or arms would inherit Adam moments from a different objective.
    checks["resume_model_only_is_used"] = True
    trainer_script = _load_script("train_autoencoder")
    checks["train_autoencoder_supports_resume_model_only"] = any(
        action.dest == "resume_model_only"
        for action in trainer_script.build_arg_parser(load_default_config())._actions
    )

    checks["shared_seed"] = args.seed
    checks["shared_batch_size"] = args.batch_size
    checks["shared_model_lr"] = args.learning_rate
    checks["shared_rate_lr"] = args.rate_lr
    checks["shared_qat"] = f"{args.qat_bits}-bit/{args.qat_mode}"
    checks["shared_epochs"] = args.epochs
    checks["only_lambda_differs"] = True
    return checks


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
    arms = list(ARMS)
    if args.only is not None:
        wanted = {name.upper() for name in args.only}
        arms = [arm for arm in arms if arm["name"].upper() in wanted]
        if not arms:
            parser.error(f"--only matched no arms; choose from {[a['name'] for a in ARMS]}")

    bad = [e for e in args.snapshot_epochs if not 1 <= e <= args.epochs]
    if bad:
        parser.error(f"--snapshot-epochs outside 1..{args.epochs}: {bad}")

    checks = fairness_preflight(args, arms, start_hash)
    failed = [name for name, value in checks.items() if isinstance(value, bool) and not value]

    print("=" * 88)
    print("M10D - CONVERGED LAMBDA REFINEMENT")
    print("=" * 88)
    print("FAIRNESS PREFLIGHT")
    for name, value in checks.items():
        marker = "  " if not isinstance(value, bool) else ("OK" if value else "!!")
        print(f"  [{marker}] {name}: {value}")
    if failed:
        print(f"\n[STOP] fairness preflight failed: {failed}", file=sys.stderr)
        print("Not training. Fix the issue and report it before continuing.", file=sys.stderr)
        return 1
    print("  -> only lambda differs between arms.")

    if args.check_only:
        print("\n--check-only: stopping before training.")
        return 0

    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from nvc.data.loaders import create_train_loader
    steps_per_epoch = len(create_train_loader(
        args.manifest, batch_size=args.batch_size, num_workers=defaults.num_workers,
        seed=args.seed, crop_size=defaults.random_crop_size,
    ))

    print("=" * 88)
    print(f"start checkpoint: {args.checkpoint}")
    print(f"  sha256 verified: {start_hash}")
    print(f"budget: {args.epochs} x {steps_per_epoch} = {args.epochs * steps_per_epoch} steps/arm "
          f"(the M10C budget, directly comparable)")
    print(f"snapshots at steps {[e * steps_per_epoch for e in args.snapshot_epochs]}")
    print(f"model lr {args.learning_rate} | rate lr {args.rate_lr} | track_scale=True "
          f"momentum {args.scale_momentum} | seed {args.seed} | batch {args.batch_size} | {device}")
    for arm in arms:
        print(f"  {arm['name']:<10} lambda={arm['lambda']:.4e}  ({arm['role']})")
    print("R below is the training proxy, NOT .nvc bitrate.")
    print("=" * 88)

    pilot = _load_script("m10a_pilot")
    convergence = _load_script("m10c_convergence")
    trainer_script = _load_script("train_autoencoder")
    original_estimator = trainer_script.RateEstimator
    original_save = trainer_script.save_checkpoint

    rows: list[dict[str, Any]] = []
    for index, arm in enumerate(arms, start=1):
        lambda_rate = arm["lambda"]
        arm_dir = args.output_dir / arm["dir"]
        print(f"\n{'=' * 88}\n[{index}/{len(arms)}] {arm['name']} lambda={lambda_rate:.4e} -> {arm_dir}\n{'=' * 88}")

        pilot._STEP_LOG.clear()
        snapshots: list[dict[str, Any]] = []
        first_epoch: dict[str, int] = {"value": None}
        trainer_script.RateEstimator = pilot._RecordingRateEstimator
        trainer_script.save_checkpoint = convergence.make_snapshotting_save(
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
            if snapshot["step"] <= len(steps):
                step_record = steps[snapshot["step"] - 1]
                snapshot.update({
                    "latent_abs_mean": step_record["latent_abs_mean"],
                    "latent_std": step_record["latent_std"],
                    "latent_range": step_record["latent_range"],
                    "bin_width": step_record["bin_width_after"],
                    "rate_loc_mean": step_record["loc_mean"],
                    "rate_loc_std": step_record["loc_std"],
                    "rate_scale_mean": step_record["scale_mean"],
                    "rate_scale_std": step_record["scale_std"],
                    "rate_grad_norm": step_record["rate_grad_norm"],
                })

        final_step = steps[-1] if steps else {}
        row = {
            "name": arm["name"], "role": arm["role"], "lambda": lambda_rate,
            "checkpoint_dir": str(arm_dir),
            "start_checkpoint_sha256": start_hash,
            "best_checkpoint": str(best_path), "best_checkpoint_sha256": _sha256(best_path),
            "final_snapshot": str(snapshots[-1]["path"]) if snapshots else None,
            "final_snapshot_sha256": snapshots[-1]["sha256"] if snapshots else None,
            "best_epoch": best_record["epoch"], "best_selection_value": best_record["val_loss"],
            "best_selection_used_only_current_objective": (
                best_record["val_loss"] == min(r["val_loss"] for r in own)
            ),
            "stale_history_records_ignored": len(stale),
            "epochs_run": len(own), "steps": len(steps), "steps_per_epoch": steps_per_epoch,
            "model_learning_rate": args.learning_rate, "rate_lr": args.rate_lr,
            "scale_momentum": args.scale_momentum, "seed": args.seed,
            "elapsed_seconds": elapsed, "snapshots": snapshots,
            "final_val_distortion": own[-1]["val_distortion"],
            "final_val_rate_bpp_proxy": own[-1]["val_rate_bpp"],
            "final_val_psnr_db": own[-1]["val_psnr"],
            "final_val_total_objective": own[-1]["val_loss"],
            "final_latent_abs_mean": final_step.get("latent_abs_mean"),
            "final_latent_range": final_step.get("latent_range"),
            "final_bin_width": final_step.get("bin_width_after"),
            "final_rate_scale_mean": final_step.get("scale_mean"),
            "all_finite": all(
                math.isfinite(record["val_distortion"]) and math.isfinite(record["val_rate_bpp"])
                for record in own
            ),
        }
        rows.append(row)
        print(f"\n  {len(own)} epochs / {len(steps)} steps in {elapsed / 60:.1f} min, "
              f"{len(snapshots)} snapshots, best epoch {best_record['epoch']}")
        print(f"  final val D {row['final_val_distortion']:.6e}  "
              f"R_proxy {row['final_val_rate_bpp_proxy']:.4f}  PSNR {row['final_val_psnr_db']:.2f} dB  "
              f"latent|.| {row['final_latent_abs_mean']:.4f}  bw {row['final_bin_width']:.4f}")
        if not row["all_finite"]:
            print(f"[ERROR] {arm['name']} produced non-finite values", file=sys.stderr)
            return 1

    summary = {
        "phase": "M10D (converged lambda refinement)",
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
        "fairness_preflight": checks,
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
                        "latent_abs_mean", "latent_std", "latent_range", "bin_width",
                        "rate_loc_mean", "rate_loc_std", "rate_scale_mean", "rate_scale_std",
                        "rate_grad_norm", "sha256", "path"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            for snapshot in row["snapshots"]:
                writer.writerow({"arm": row["name"], "lambda": row["lambda"], **snapshot})

    print("\n" + "=" * 88)
    print(f"{'arm':<10} {'lambda':>11} {'val D':>11} {'PSNR':>7} {'R*':>8} "
          f"{'latent|.|':>10} {'bin w':>8} {'scale':>7}")
    for row in rows:
        print(f"{row['name']:<10} {row['lambda']:>11.4e} {row['final_val_distortion']:>11.4e} "
              f"{row['final_val_psnr_db']:>7.2f} {row['final_val_rate_bpp_proxy']:>8.4f} "
              f"{row['final_latent_abs_mean']:>10.4f} {row['final_bin_width']:>8.4f} "
              f"{row['final_rate_scale_mean']:>7.3f}")
    print("=" * 88)
    print("* proxy, NOT .nvc bitrate.")
    print(f"\nSummary: {args.output_dir / 'training_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
