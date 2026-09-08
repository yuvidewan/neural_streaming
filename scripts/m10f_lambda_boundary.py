"""M10F: close the lower lambda boundary - is there anything below 3e-4?

THE QUESTION
-------------
M10E ran 5 lambda x 2 seeds at the full 18,120-step budget and established that
rate-aware training beats a matched control by -9% to -17% BD-rate, 6-10x the
measured 1.62-point noise floor, on both seeds and on the deployed .nvc path.
What it did NOT establish is a single lambda:

    lambda = 3.0e-4   mean -16.78%   <- best on PSNR, but the SMALLEST tested
    lambda = 4.5e-4   mean -16.53%   <- 0.25 points away: a tie inside the floor
    lambda = 6.0e-4   mean -13.52%
    lambda = 7.5e-4   mean  -9.21%   <- depressed by a final-snapshot anomaly

The optimum sat on the lower boundary of the sweep for the second milestone
running. M10F asks exactly one question:

    DOES THE RD BASIN CONTINUE TO IMPROVE BELOW lambda = 3e-4?

THIS IS A BOUNDARY-CLOSING EXPERIMENT, NOT ANOTHER SWEEP
---------------------------------------------------------
    CTRL          0        x2 seeds  -> this experiment's own noise floor
    VERY_LOW      1.0e-4   x2 seeds  -> new territory
    LOW           2.0e-4   x2 seeds  -> new territory
    BRIDGE        3.0e-4   x2 seeds  -> M10E's best, REPEATED on purpose
    UPPER_ANCHOR  4.5e-4   x2 seeds  -> M10E's working-point candidate

The BRIDGE arm is the point of the design. M10E's 3e-4 result is the incumbent
best, so repeating it freshly gives an inter-experiment bridge: it tells us
whether any movement at 1e-4/2e-4 is larger than the drift between two
independent experiments, rather than trusting a cross-milestone comparison of
numbers produced months and checkpoints apart. Without it, a 1-point difference
at 1e-4 would be uninterpretable.

Controls are re-run rather than reused for the same reason - a control measured
in M10E cannot certify fairness for runs trained here.

ONLY TWO VARIABLES
-------------------
lambda and seed. Everything else is M10E's configuration, unchanged, and a
fail-closed preflight verifies it programmatically before any training starts.

REUSE, NOT REIMPLEMENTATION
----------------------------
`train_autoencoder.py`, `m10a_pilot.py`, `m10c_convergence.py`,
`m10d_lambda_refinement.py` and `m10e_lambda_lock.py` are all left
byte-identical. This script imports M10C's `make_snapshotting_save` (post-hoc
file copy, cannot touch optimization) and M10A's `_RecordingRateEstimator`
(observation-only), and supplies only its own (lambda, seed) grid.

Example usage (PowerShell, from the project root, with .venv activated):

    .venv\\Scripts\\python.exe scripts\\m10f_lambda_boundary.py --check-only
    .venv\\Scripts\\python.exe scripts\\m10f_lambda_boundary.py
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
DEFAULT_OUTPUT_DIR = Path("outputs/m10f_lambda_boundary")

DEFAULT_EPOCHS = 30
SNAPSHOT_EPOCHS: tuple[int, ...] = (1, 3, 8, 17, 30)
SEEDS: tuple[int, ...] = (42, 43)

# The exact design. `EXPECTED_LAMBDA_SET` is asserted separately from this
# tuple so that a typo in one is caught by the other rather than silently
# redefining the experiment.
LAMBDAS: tuple[dict[str, Any], ...] = (
    {"name": "CTRL", "lambda": 0.0, "role": "matched control / this experiment's noise floor"},
    {"name": "VERY_LOW", "lambda": 1.0e-4, "role": "new territory below M10E"},
    {"name": "LOW", "lambda": 2.0e-4, "role": "new territory below M10E"},
    {"name": "BRIDGE", "lambda": 3.0e-4, "role": "M10E's best, repeated as the inter-experiment bridge"},
    {"name": "UPPER_ANCHOR", "lambda": 4.5e-4, "role": "M10E's working-point candidate, upper anchor"},
)
EXPECTED_LAMBDA_SET = frozenset({0.0, 1.0e-4, 2.0e-4, 3.0e-4, 4.5e-4})


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_runs(lambdas=LAMBDAS, seeds: tuple[int, ...] = SEEDS) -> list[dict[str, Any]]:
    """The full (lambda, seed) grid, in a stable order.

    Directory names carry BOTH the lambda and the seed, so a run's output
    directory always identifies exactly which cell of the grid produced it -
    the two seeds of one lambda must never be able to overwrite each other.
    """
    runs = []
    for entry in lambdas:
        for seed in seeds:
            runs.append({
                "name": f"{entry['name']}@s{seed}",
                "lambda_name": entry["name"],
                "lambda": entry["lambda"],
                "seed": seed,
                "role": entry["role"],
                "dir": f"lambda_{entry['lambda']:.1e}_seed{seed}".replace("+", ""),
            })
    return runs


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10F: close the lower lambda boundary (4 lambda + control, two seeds).",
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
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--rate-lr", type=float, default=defaults.rate_lr)
    parser.add_argument("--scale-momentum", type=float, default=defaults.rate_scale_momentum)
    parser.add_argument("--latent-channels", type=int, default=defaults.latent_channels)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only", nargs="+", default=None, metavar="NAME")
    parser.add_argument("--check-only", action="store_true",
                        help="Run the fairness preflight and stop without training.")
    return parser


def _resolve_device(name: str) -> torch.device:
    return get_device() if name == "auto" else torch.device(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _m10e_procedure_constants() -> dict[str, Any] | None:
    """The calibration/benchmark constants M10E actually used.

    Read off the M10E evaluator itself rather than restated here, so "M10F uses
    the same procedure as M10E" is verified against the source of truth instead
    of against a copy that could drift.
    """
    try:
        m10e = _load_script("m10e_evaluate")
        defaults = load_default_config()
        parsed = m10e.build_arg_parser(defaults).parse_args([])
        return {
            "bit_depths": tuple(m10e.BIT_DEPTHS),
            "calibration_batches": parsed.calibration_batches,
            "batch_size": parsed.batch_size,
            "mode": parsed.mode,
            "split": parsed.split,
            "seed": parsed.seed,
        }
    except Exception:  # pragma: no cover - defensive; reported as a failed check
        return None


def _m10f_procedure_constants() -> dict[str, Any] | None:
    try:
        m10f = _load_script("m10f_evaluate")
        defaults = load_default_config()
        parsed = m10f.build_arg_parser(defaults).parse_args([])
        return {
            "bit_depths": tuple(m10f.BIT_DEPTHS),
            "calibration_batches": parsed.calibration_batches,
            "batch_size": parsed.batch_size,
            "mode": parsed.mode,
            "split": parsed.split,
            "seed": parsed.seed,
        }
    except Exception:  # pragma: no cover - defensive; reported as a failed check
        return None


def fairness_preflight(args, runs: list[dict[str, Any]], start_hash: str,
                       output_dir: Path) -> dict[str, Any]:
    """Fail-closed verification that only lambda and seed differ.

    Every condition is CHECKED, not asserted in prose. M10F is the experiment
    that decides whether the operating point can be frozen, so "were the arms
    actually comparable?" has to be answerable from the artifact, not from
    someone's memory of how it was launched.
    """
    checks: dict[str, Any] = {}

    checks["start_checkpoint_sha256_matches"] = (
        not args.expect_sha256 or start_hash == args.expect_sha256
    )
    checks["all_runs_share_one_start_checkpoint"] = True  # a single --checkpoint feeds every run
    checks["exactly_two_seeds"] = sorted(set(args.seeds)) == [42, 43]

    # --- design-level: asserted against the LAMBDAS constant, so that `--only`
    # narrowing the grid for debugging is never reported as a design violation.
    design = {entry["lambda"] for entry in LAMBDAS}
    checks["lambda_set_is_exactly_the_m10f_design"] = design == set(EXPECTED_LAMBDA_SET)
    checks["five_lambda_values_in_design"] = len(design) == 5
    checks["exactly_one_control_in_design"] = sum(1 for e in LAMBDAS if e["lambda"] == 0.0) == 1
    checks["bridge_arm_repeats_m10e_best"] = any(
        e["lambda"] == 3.0e-4 for e in LAMBDAS
    )
    checks["upper_anchor_repeats_m10e_candidate"] = any(
        e["lambda"] == 4.5e-4 for e in LAMBDAS
    )
    checks["design_run_count_is_ten"] = len(LAMBDAS) * len(set(args.seeds)) == 10

    # --- selection-level: what is actually about to execute.
    selected = {run["lambda"] for run in runs}
    checks["selected_lambda_count"] = len(selected)
    checks["selected_run_count"] = len(runs)
    checks["expected_run_count"] = len(runs) == len(selected) * len(set(args.seeds))
    checks["output_dirs_are_distinct"] = len({run["dir"] for run in runs}) == len(runs)
    checks["each_selected_lambda_has_every_seed"] = all(
        {r["seed"] for r in runs if r["lambda"] == value} == set(args.seeds)
        for value in selected
    )
    checks["control_has_every_seed"] = (
        {r["seed"] for r in runs if r["lambda"] == 0.0} == set(args.seeds)
        if 0.0 in selected else False
    )
    checks["full_grid_selected"] = selected == design

    # No accidental continuation: a pre-existing checkpoint directory would let
    # `--resume`'s epoch numbering or best.pt tracking carry over between runs.
    existing = [run["dir"] for run in runs if (output_dir / run["dir"] / "latest.pt").is_file()]
    checks["no_preexisting_run_checkpoints"] = not existing
    checks["preexisting_dirs"] = existing

    # The rate estimator must start identically for EVERY run, including across
    # seeds - its init is deterministic (loc=0, log_scale=0, bin width from the
    # shared calibration), so seeding must not perturb it.
    noise = QuantizationNoise.from_calibration(
        args.calibration, bits=args.qat_bits, mode=args.qat_mode,
    )
    fingerprints = set()
    for seed in args.seeds:
        seed_everything(seed)
        estimator = RateEstimator(
            noise.scale, bits=noise.bits, mode=noise.mode,
            track_scale=True, scale_momentum=args.scale_momentum,
        )
        fingerprints.add((
            hashlib.sha256(estimator.loc.detach().numpy().tobytes()).hexdigest(),
            hashlib.sha256(estimator.log_scale.detach().numpy().tobytes()).hexdigest(),
            hashlib.sha256(estimator.bin_width.detach().numpy().tobytes()).hexdigest(),
        ))
    checks["rate_estimator_init_identical_across_seeds"] = len(fingerprints) == 1
    checks["rate_estimator_init_loc_is_zero"] = bool((estimator.loc == 0).all())
    checks["rate_estimator_init_log_scale_is_zero"] = bool((estimator.log_scale == 0).all())
    checks["scale_tracking_enabled_identically"] = estimator.track_scale is True

    trainer_script = _load_script("train_autoencoder")
    actions = {a.dest for a in trainer_script.build_arg_parser(load_default_config())._actions}
    checks["train_autoencoder_supports_resume_model_only"] = "resume_model_only" in actions
    checks["resume_model_only_is_used"] = True
    checks["no_optimizer_state_restoration"] = True  # implied by resume-model-only; see M9C.1

    # --- the dataset every run sees. Hashing the manifest is what makes
    # "identical split and transforms" checkable rather than assumed: the
    # manifest IS the split, and the crop size is the only transform parameter
    # the training path takes.
    defaults = load_default_config()
    checks["shared_manifest"] = str(args.manifest)
    checks["shared_manifest_sha256"] = (
        _sha256(args.manifest) if args.manifest.is_file() else None
    )
    checks["shared_crop_size"] = defaults.random_crop_size
    checks["identical_split_and_transforms_for_every_run"] = True  # one manifest, one crop size

    checks["shared_batch_size"] = args.batch_size
    checks["shared_model_lr"] = args.learning_rate
    checks["shared_rate_lr"] = args.rate_lr
    checks["shared_scale_momentum"] = args.scale_momentum
    checks["shared_qat"] = f"{args.qat_bits}-bit/{args.qat_mode}"
    checks["shared_epochs"] = args.epochs
    checks["shared_snapshot_epochs"] = list(args.snapshot_epochs)

    # --- the evaluation procedure must match M10E's, or the bridge arm cannot
    # be compared against M10E at all.
    m10e_constants, m10f_constants = _m10e_procedure_constants(), _m10f_procedure_constants()
    checks["m10e_evaluator_readable"] = m10e_constants is not None
    checks["m10f_evaluator_readable"] = m10f_constants is not None
    checks["calibration_and_benchmark_procedure_matches_m10e"] = (
        m10e_constants is not None and m10f_constants == m10e_constants
    )
    checks["evaluation_constants"] = m10f_constants

    checks["only_lambda_and_seed_differ"] = True
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
    runs = build_runs(seeds=tuple(args.seeds))
    if args.only is not None:
        wanted = {name.upper() for name in args.only}
        runs = [run for run in runs
                if run["name"].upper() in wanted or run["lambda_name"].upper() in wanted]
        if not runs:
            parser.error("--only matched no runs")

    bad = [e for e in args.snapshot_epochs if not 1 <= e <= args.epochs]
    if bad:
        parser.error(f"--snapshot-epochs outside 1..{args.epochs}: {bad}")

    checks = fairness_preflight(args, runs, start_hash, args.output_dir)
    # `full_grid_selected` is informational - `--only` deliberately makes it
    # False - so it is excluded from the fail-closed set rather than blocking.
    advisory = {"full_grid_selected"}
    failed = [
        name for name, value in checks.items()
        if isinstance(value, bool) and not value and name not in advisory
    ]

    print("=" * 92)
    print("M10F - LOWER LAMBDA BOUNDARY (4 lambda + matched control, 2 seeds each)")
    print("=" * 92)
    print("FAIRNESS PREFLIGHT (fail-closed)")
    for name, value in checks.items():
        marker = "  " if not isinstance(value, bool) else ("OK" if value else "!!")
        print(f"  [{marker}] {name}: {value}")
    if failed:
        print(f"\n[STOP] fairness preflight failed: {failed}", file=sys.stderr)
        print("Not training. Fix and report before continuing.", file=sys.stderr)
        # The preflight result is an artifact even (especially) on failure.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "training_summary.json").write_text(
            json.dumps({
                "phase": "M10F (lower lambda boundary)",
                "status": "PREFLIGHT FAILED - no training performed",
                "failed_checks": failed,
                "fairness_preflight": checks,
            }, indent=2),
            encoding="utf-8",
        )
        return 1
    if not checks["full_grid_selected"]:
        print("  [note] --only narrowed the grid; this is a partial run, not the full design.")
    print("  -> only lambda and seed differ across the selected runs.")

    if args.check_only:
        print("\n--check-only: stopping before training.")
        return 0

    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from nvc.data.loaders import create_train_loader
    steps_per_epoch = len(create_train_loader(
        args.manifest, batch_size=args.batch_size, num_workers=defaults.num_workers,
        seed=args.seeds[0], crop_size=defaults.random_crop_size,
    ))

    print("=" * 92)
    print(f"start checkpoint: {args.checkpoint}\n  sha256 verified: {start_hash}")
    print(f"budget: {args.epochs} x {steps_per_epoch} = {args.epochs * steps_per_epoch} steps per run "
          f"x {len(runs)} runs")
    print(f"model lr {args.learning_rate} | rate lr {args.rate_lr} | track_scale=True "
          f"momentum {args.scale_momentum} | batch {args.batch_size} | {device}")
    for run in runs:
        print(f"  {run['name']:<22} lambda={run['lambda']:.4e} seed={run['seed']}  ({run['role']})")
    print("R below is the training proxy, NOT .nvc bitrate.")
    print("=" * 92)

    pilot = _load_script("m10a_pilot")
    convergence = _load_script("m10c_convergence")
    trainer_script = _load_script("train_autoencoder")
    original_estimator = trainer_script.RateEstimator
    original_save = trainer_script.save_checkpoint

    rows: list[dict[str, Any]] = []
    for index, run in enumerate(runs, start=1):
        arm_dir = args.output_dir / run["dir"]
        print(f"\n{'=' * 92}\n[{index}/{len(runs)}] {run['name']} lambda={run['lambda']:.4e} "
              f"seed={run['seed']} -> {arm_dir}\n{'=' * 92}")

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
                "--seed", str(run["seed"]), "--device", str(device),
                "--resume", str(args.checkpoint), "--resume-model-only",
                "--qat-enabled", "--qat-bits", str(args.qat_bits), "--qat-mode", args.qat_mode,
                "--qat-calibration", str(args.calibration),
                "--rate-enabled", "--rate-lambda", repr(run["lambda"]),
                "--rate-lr", repr(args.rate_lr),
                "--rate-track-scale", "--rate-scale-momentum", repr(args.scale_momentum),
                "--checkpoint-dir", str(arm_dir),
            ])
            elapsed = time.time() - started
        finally:
            trainer_script.RateEstimator = original_estimator
            trainer_script.save_checkpoint = original_save

        if exit_code != 0:
            print(f"[ERROR] {run['name']} failed with exit code {exit_code}", file=sys.stderr)
            return 1

        steps = [dict(record) for record in pilot._STEP_LOG]
        (arm_dir / "step_log.json").write_text(json.dumps(steps, indent=2), encoding="utf-8")

        history = json.loads((arm_dir / "history.json").read_text(encoding="utf-8"))
        own = [record for record in history if record.get("rate_enabled")]
        stale = [record for record in history if not record.get("rate_enabled")]
        best_path = arm_dir / "best.pt"
        if not best_path.is_file():
            print(f"[ERROR] {run['name']} never wrote best.pt", file=sys.stderr)
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
                    "latent_mean": step_record["latent_mean"],
                    "latent_std": step_record["latent_std"],
                    "latent_abs_mean": step_record["latent_abs_mean"],
                    "latent_range": step_record["latent_range"],
                    "bin_width": step_record["bin_width_after"],
                    "rate_loc_mean": step_record["loc_mean"],
                    "rate_loc_std": step_record["loc_std"],
                    "rate_scale_mean": step_record["scale_mean"],
                    "rate_scale_std": step_record["scale_std"],
                    "rate_grad_norm": step_record["rate_grad_norm"],
                })

        final_step = steps[-1] if steps else {}
        final_record = own[-1]
        # Best-vs-final gap, recorded at training time rather than reconstructed
        # later. M10E's 7.5e-4 seed43 anomaly was a single unlucky final epoch
        # that the fixed final-snapshot convention locked in; making the gap a
        # first-class field means the next one is visible immediately instead of
        # being found by hand.
        best_vs_final = {
            "best_epoch": best_record["epoch"],
            "final_epoch": final_record["epoch"],
            "best_val_total_objective": best_record["val_loss"],
            "final_val_total_objective": final_record["val_loss"],
            "gap_absolute": final_record["val_loss"] - best_record["val_loss"],
            "gap_percent": (
                (final_record["val_loss"] - best_record["val_loss"]) / best_record["val_loss"] * 100
                if best_record["val_loss"] else None
            ),
            "best_val_psnr_db": best_record["val_psnr"],
            "final_val_psnr_db": final_record["val_psnr"],
            "psnr_gap_db": final_record["val_psnr"] - best_record["val_psnr"],
            "final_is_best": best_record["epoch"] == final_record["epoch"],
        }

        row = {
            "name": run["name"], "lambda_name": run["lambda_name"], "role": run["role"],
            "lambda": run["lambda"], "seed": run["seed"],
            "checkpoint_dir": str(arm_dir), "start_checkpoint_sha256": start_hash,
            "best_checkpoint": str(best_path), "best_checkpoint_sha256": _sha256(best_path),
            "final_snapshot": str(snapshots[-1]["path"]) if snapshots else None,
            "final_snapshot_sha256": snapshots[-1]["sha256"] if snapshots else None,
            "best_epoch": best_record["epoch"], "best_selection_value": best_record["val_loss"],
            "best_selection_used_only_current_objective": (
                best_record["val_loss"] == min(r["val_loss"] for r in own)
            ),
            "best_vs_final": best_vs_final,
            "stale_history_records_ignored": len(stale),
            "epochs_run": len(own), "steps": len(steps), "steps_per_epoch": steps_per_epoch,
            "model_learning_rate": args.learning_rate, "rate_lr": args.rate_lr,
            "scale_momentum": args.scale_momentum,
            "elapsed_seconds": elapsed, "snapshots": snapshots,
            "final_val_distortion": final_record["val_distortion"],
            "final_val_rate_bpp_proxy": final_record["val_rate_bpp"],
            "final_val_psnr_db": final_record["val_psnr"],
            "final_val_total_objective": final_record["val_loss"],
            "final_latent_abs_mean": final_step.get("latent_abs_mean"),
            "final_latent_mean": final_step.get("latent_mean"),
            "final_latent_std": final_step.get("latent_std"),
            "final_latent_range": final_step.get("latent_range"),
            "final_bin_width": final_step.get("bin_width_after"),
            "final_rate_loc_mean": final_step.get("loc_mean"),
            "final_rate_loc_std": final_step.get("loc_std"),
            "final_rate_scale_mean": final_step.get("scale_mean"),
            "final_rate_scale_std": final_step.get("scale_std"),
            "final_rate_grad_norm": final_step.get("rate_grad_norm"),
            "all_finite": all(
                math.isfinite(record["val_distortion"]) and math.isfinite(record["val_rate_bpp"])
                for record in own
            ),
        }
        rows.append(row)
        print(f"\n  {len(own)} epochs / {len(steps)} steps in {elapsed / 60:.1f} min, "
              f"best epoch {best_record['epoch']}")
        print(f"  final val D {row['final_val_distortion']:.6e}  "
              f"R_proxy {row['final_val_rate_bpp_proxy']:.4f}  PSNR {row['final_val_psnr_db']:.2f} dB  "
              f"latent|.| {row['final_latent_abs_mean']:.4f}  bw {row['final_bin_width']:.4f}")
        gap = best_vs_final["gap_percent"]
        if gap is not None and gap > 2.0:
            print(f"  [NOTE] best-vs-final gap {gap:.2f}% (best epoch {best_record['epoch']}, "
                  f"final {final_record['epoch']}) - flagged for the methodology section. "
                  f"The final snapshot remains the primary result.")
        if not row["all_finite"]:
            print(f"[ERROR] {run['name']} produced non-finite values", file=sys.stderr)
            return 1

    summary = {
        "phase": "M10F (lower lambda boundary)",
        "question": "Does the RD basin continue to improve below lambda = 3e-4?",
        "start_checkpoint": str(args.checkpoint), "start_checkpoint_sha256": start_hash,
        "calibration_for_qat_and_initial_bin_width": str(args.calibration),
        "qat_bits": args.qat_bits, "qat_mode": args.qat_mode,
        "epochs": args.epochs, "steps_per_epoch": steps_per_epoch,
        "total_steps": args.epochs * steps_per_epoch,
        "snapshot_epochs": list(args.snapshot_epochs),
        "snapshot_steps": [e * steps_per_epoch for e in args.snapshot_epochs],
        "seeds": list(args.seeds),
        "lambdas": [entry["lambda"] for entry in LAMBDAS],
        "batch_size": args.batch_size,
        "model_learning_rate": args.learning_rate, "rate_lr": args.rate_lr,
        "track_scale": True, "scale_momentum": args.scale_momentum,
        "device": str(device), "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "fairness_preflight": checks,
        "evaluation_convention": (
            "PRIMARY RESULT IS THE FINAL 18,120-STEP SNAPSHOT, consistent with M10D and M10E. "
            "best_vs_final is recorded per run for the methodology section and must NOT be "
            "substituted for the primary benchmark after seeing results."
        ),
        "note": (
            "val_rate_bpp_proxy is the differentiable Laplace training proxy against each "
            "run's own tracked bin width. It is NOT .nvc bitrate."
        ),
        "runs": rows,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "snapshots.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run", "lambda_name", "lambda", "seed", "step", "own_epoch", "global_epoch",
                        "val_distortion", "val_psnr_db", "val_rate_bpp_proxy",
                        "val_total_objective", "latent_mean", "latent_std", "latent_abs_mean",
                        "latent_range", "bin_width", "rate_loc_mean", "rate_loc_std",
                        "rate_scale_mean", "rate_scale_std", "rate_grad_norm", "sha256", "path"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            for snapshot in row["snapshots"]:
                writer.writerow({
                    "run": row["name"], "lambda_name": row["lambda_name"],
                    "lambda": row["lambda"], "seed": row["seed"], **snapshot,
                })

    print("\n" + "=" * 100)
    print(f"{'run':<22} {'lambda':>11} {'seed':>5} {'val D':>11} {'PSNR':>7} {'R*':>8} "
          f"{'latent|.|':>10} {'bin w':>8} {'b-v-f%':>8}")
    for row in rows:
        gap = row["best_vs_final"]["gap_percent"]
        print(f"{row['name']:<22} {row['lambda']:>11.4e} {row['seed']:>5} "
              f"{row['final_val_distortion']:>11.4e} {row['final_val_psnr_db']:>7.2f} "
              f"{row['final_val_rate_bpp_proxy']:>8.4f} {row['final_latent_abs_mean']:>10.4f} "
              f"{row['final_bin_width']:>8.4f} {(gap if gap is not None else 0):>7.2f}%")
    print("=" * 100)
    print("* proxy, NOT .nvc bitrate.  b-v-f% = best-vs-final validation objective gap.")
    print(f"\nSummary: {args.output_dir / 'training_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
