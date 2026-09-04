"""Milestone 9C, phases 4-6: short lambda pilot runs and their comparison.

WHAT THIS IS AND IS NOT
------------------------
This is a SHORT, deliberately non-convergent sweep whose only question is
"does lambda move D and R in the directions the objective says it should?".
It is NOT the final M9 training run, it does not recalibrate anything, and
it never touches the `.nvc` codec path. The rate figures here are the
training-time Laplace proxy (`nvc.training.rate_estimator`), NOT measured
`.nvc` payload bitrate - see MILESTONE_9_PLAN.md.

WHERE THE LAMBDA VALUES COME FROM
----------------------------------
Read out of `dr_diagnostic.json` (written by scripts/m9c_rd_diagnostic.py),
never typed in by hand: that file measures this model's actual mean D and
mean R on real data, and `lambda_balance = mean(D)/mean(R)` is the value at
which the two loss terms contribute equally. The sweep is a half-decade log
grid spanning one decade either side of it:

    lambda_balance * [1/10, 1/sqrt(10), 1, sqrt(10), 10]

which maps onto the five regimes the milestone asks for, because at
`lambda = k * lambda_balance` the rate term is `k/(1+k)` of the total loss:
9% (quality-oriented), 24% (mildly rate-aware), 50% (balanced), 76%
(strongly rate-aware), 91% (rate-oriented).

THE CONTROL ARM
----------------
A `lambda = 0` arm runs alongside them. Without it, any change in D across
the sweep is unattributable: a few hundred more optimizer steps move D on
their own, whatever lambda is. The control measures that drift so the
lambda arms can be read against it rather than against the starting
checkpoint. It is a control, not a sixth operating point.

EVERY ARM IS INDEPENDENT
-------------------------
Each lambda gets the same M8 QAT checkpoint, the same seed, batch size,
learning rate, QAT bit depth and calibration, and a fresh (empty) optimizer
and rate estimator. No arm continues from another. `train_autoencoder.main`
re-seeds and rebuilds the model, loaders, optimizer and rate estimator on
every invocation, so the arms do not interact.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\m9c_rd_diagnostic.py           # writes dr_diagnostic.json
    python scripts\\m9c_lambda_pilot.py            # reads it, runs the sweep
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
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
DEFAULT_OUTPUT_DIR = Path("outputs/m9c_lambda_pilot")

# Half-decade log grid, one decade either side of the measured balance point.
# See the module docstring for why these five and what each represents.
LAMBDA_MULTIPLIERS: tuple[float, ...] = (0.1, 1 / math.sqrt(10), 1.0, math.sqrt(10), 10.0)
REGIME_LABELS: tuple[str, ...] = (
    "quality-oriented", "mildly rate-aware", "balanced",
    "strongly rate-aware", "rate-oriented",
)


def _load_script(name: str):
    """Import a sibling script by path, the same way the test-suite does."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Milestone 9C phases 4-6: run short D+lambda*R pilot trainings across a "
            "lambda grid derived from the measured D/R scale, and summarize the trend."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Starting model for EVERY arm (M9C Rule 6: the M8 QAT checkpoint).",
    )
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--qat-bits", type=int, default=4)
    parser.add_argument("--qat-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument(
        "--diagnostic", type=Path, default=DEFAULT_OUTPUT_DIR / "dr_diagnostic.json",
        help="Output of scripts/m9c_rd_diagnostic.py, supplying the measured lambda balance.",
    )
    parser.add_argument(
        "--lambda-balance", type=float, default=None,
        help="Override the balance point instead of reading --diagnostic. For tests only - "
             "a real run must derive it from measurement (M9C Rule 4/5).",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Short on purpose - not a convergence run.")
    parser.add_argument("--max-batches", type=int, default=100, help="Batches per epoch, train and val.")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument(
        "--rate-lr", type=float, default=defaults.rate_lr,
        help="Milestone 9C.1: learning rate for the rate estimator's own parameter "
             "group, passed straight through to train_autoencoder.py. The model keeps "
             "--learning-rate.",
    )
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--latent-channels", type=int, default=defaults.latent_channels)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-control", action="store_true",
        help="Omit the lambda=0 control arm. Not recommended - see the module docstring.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip writing the D-vs-R plot.")
    return parser


def _resolve_device(name: str) -> torch.device:
    return get_device() if name == "auto" else torch.device(name)


def _lambda_grid(balance: float) -> list[dict[str, Any]]:
    return [
        {"lambda": balance * multiplier, "multiplier": multiplier, "regime": label}
        for multiplier, label in zip(LAMBDA_MULTIPLIERS, REGIME_LABELS)
    ]


def _arm_name(lambda_rate: float) -> str:
    """Filesystem-safe, sortable directory name for one arm."""
    return "control_lambda0" if lambda_rate == 0.0 else f"lambda_{lambda_rate:.6e}".replace(".", "p")


def _measure_arm(
    checkpoint_path: Path,
    *,
    manifest: Path,
    calibration: Path,
    qat_bits: int,
    qat_mode: str,
    lambda_rate: float,
    batch_size: int,
    max_batches: int,
    num_workers: int,
    crop_size: int | None,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Post-hoc validation of one finished arm, plus gradient diagnostics.

    Evaluates in eval mode (no QAT noise), which is the regime the deployed
    model actually runs in - the training loop's own val numbers use the same
    convention. The rate estimator is rebuilt from the arm's checkpoint
    `extra` so each arm is scored by the density IT learned, not a shared one.
    """
    seed_everything(seed)
    noise = QuantizationNoise.from_calibration(calibration, bits=qat_bits, mode=qat_mode)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device, eval_mode=True)
    model.quantization_noise = noise

    estimator = RateEstimator(noise.scale, bits=noise.bits, mode=noise.mode).to(device)
    extra = checkpoint.get("extra")
    rate_state_restored = False
    if extra is not None and "rate_estimator_state_dict" in extra:
        estimator.load_state_dict(extra["rate_estimator_state_dict"])
        rate_state_restored = True

    loader = create_val_loader(
        manifest, batch_size=batch_size, num_workers=num_workers, crop_size=crop_size,
    )

    distortions, rates, psnrs = [], [], []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            batch = batch.to(device)
            latent = model.encode(batch)
            reconstruction = model.decode(latent)
            distortions.append(mse(reconstruction, batch).item())
            rates.append(estimator(latent, batch.shape[-2] * batch.shape[-1]).item())
            psnrs.append(psnr(reconstruction, batch).item())

    val_d = sum(distortions) / len(distortions)
    val_r = sum(rates) / len(rates)
    val_psnr = sum(psnrs) / len(psnrs)

    # Gradient norms on one fixed batch, under this arm's own lambda - cheap,
    # and the direct way to see how much of the encoder's update is rate-driven.
    model.train()
    batch = next(iter(loader)).to(device)
    latent = model.encode(batch)
    latent = noise.apply(latent)
    rate = estimator(latent, batch.shape[-2] * batch.shape[-1])
    loss = mse(model.decode(latent), batch) + lambda_rate * rate
    model.zero_grad()
    estimator.zero_grad()
    loss.backward()

    def _grad_norm(prefix: str) -> float:
        total = sum(
            p.grad.pow(2).sum().item()
            for name, p in model.named_parameters()
            if p.grad is not None and name.startswith(prefix)
        )
        return math.sqrt(total)

    loc = estimator.loc.detach().flatten()
    scale = torch.exp(estimator.log_scale.detach()).flatten()
    return {
        "val_distortion": val_d,
        "val_rate_bpp": val_r,
        "val_total_loss": val_d + lambda_rate * val_r,
        "val_psnr_db": val_psnr,
        "val_batches": len(distortions),
        "rate_estimator_state_restored": rate_state_restored,
        "encoder_grad_norm": _grad_norm("encoder."),
        "decoder_grad_norm": _grad_norm("decoder."),
        "rate_loc_grad_norm": estimator.loc.grad.norm().item(),
        "rate_log_scale_grad_norm": estimator.log_scale.grad.norm().item(),
        "rate_estimator_loc": {
            "mean": loc.mean().item(), "std": loc.std().item(),
            "min": loc.min().item(), "max": loc.max().item(),
        },
        "rate_estimator_scale": {
            "mean": scale.mean().item(), "std": scale.std().item(),
            "min": scale.min().item(), "max": scale.max().item(),
        },
        "all_finite": bool(
            math.isfinite(val_d) and math.isfinite(val_r)
            and torch.isfinite(loc).all() and torch.isfinite(scale).all()
        ),
    }


def _write_plot(rows: list[dict[str, Any]], path: Path) -> bool:
    """D-vs-R scatter across the sweep. Returns False if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    lambdas = [r["lambda"] for r in rows]
    val_d = [r["val_distortion"] for r in rows]
    val_r = [r["val_rate_bpp"] for r in rows]

    axes[0].plot(val_r, val_d, "o-", color="#1f77b4")
    for row in rows:
        axes[0].annotate(
            "control" if row["lambda"] == 0 else f"{row['lambda']:.2e}",
            (row["val_rate_bpp"], row["val_distortion"]),
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )
    axes[0].set_xlabel("estimated rate R (bpp, training proxy - NOT .nvc bitrate)")
    axes[0].set_ylabel("distortion D (MSE)")
    axes[0].set_title("M9C pilot: D vs R across the lambda sweep")
    axes[0].grid(alpha=0.3)

    positive = [r for r in rows if r["lambda"] > 0]
    axes[1].semilogx([r["lambda"] for r in positive], [r["val_rate_bpp"] for r in positive],
                     "o-", color="#d62728", label="val R (bpp)")
    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("estimated rate R (bpp)", color="#d62728")
    axes[1].grid(alpha=0.3)
    twin = axes[1].twinx()
    twin.semilogx([r["lambda"] for r in positive], [r["val_distortion"] for r in positive],
                  "s--", color="#1f77b4", label="val D (MSE)")
    twin.set_ylabel("distortion D (MSE)", color="#1f77b4")
    axes[1].set_title("Rate and distortion vs lambda")

    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return True


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

    if args.lambda_balance is not None:
        balance = args.lambda_balance
        balance_source = "--lambda-balance override"
    else:
        if not args.diagnostic.is_file():
            print(
                f"[ERROR] --diagnostic not found: {args.diagnostic}. Run "
                "scripts/m9c_rd_diagnostic.py first - the lambda grid must come from "
                "measured D/R statistics, not a hardcoded guess.",
                file=sys.stderr,
            )
            return 1
        diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
        balance = diagnostic["lambda_balance"]["from_fitted_rate"]
        balance_source = str(args.diagnostic)
    if not math.isfinite(balance) or balance <= 0:
        print(f"[ERROR] lambda balance must be finite and > 0, got {balance}", file=sys.stderr)
        return 1

    device = _resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arms = _lambda_grid(balance)
    if not args.skip_control:
        arms.insert(0, {"lambda": 0.0, "multiplier": 0.0, "regime": "control (distortion-only)"})

    print("=" * 70)
    print("M9C phases 4-6: lambda pilot sweep")
    print("=" * 70)
    print(f"lambda balance:  {balance:.6e}   (from {balance_source})")
    print(f"start checkpoint: {args.checkpoint}  (identical for every arm)")
    print(f"schedule:        {args.epochs} epochs x {args.max_batches} batches = "
          f"{args.epochs * args.max_batches} optimizer steps per arm")
    print(f"seed {args.seed} | batch {args.batch_size} | model lr {args.learning_rate} | "
          f"rate lr {args.rate_lr} | QAT {args.qat_bits}-bit/{args.qat_mode} | device {device}")
    print("SHORT pilot - not the final M9 training run.")
    print("=" * 70)
    for arm in arms:
        print(f"  lambda={arm['lambda']:.6e}  ({arm['regime']})")
    print("=" * 70)

    trainer_script = _load_script("train_autoencoder")
    rows: list[dict[str, Any]] = []

    for index, arm in enumerate(arms, start=1):
        lambda_rate = arm["lambda"]
        arm_dir = args.output_dir / _arm_name(lambda_rate)
        print(f"\n[{index}/{len(arms)}] lambda={lambda_rate:.6e} ({arm['regime']}) -> {arm_dir}")

        argv_arm = [
            "--manifest", str(args.manifest),
            "--epochs", str(args.epochs),
            "--max-batches", str(args.max_batches),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--latent-channels", str(args.latent_channels),
            "--seed", str(args.seed),
            "--device", str(device),
            "--resume", str(args.checkpoint),
            "--resume-model-only",
            "--qat-enabled", "--qat-bits", str(args.qat_bits), "--qat-mode", args.qat_mode,
            "--qat-calibration", str(args.calibration),
            "--rate-enabled", "--rate-lambda", repr(lambda_rate),
            "--rate-lr", repr(args.rate_lr),
            "--checkpoint-dir", str(arm_dir),
        ]
        exit_code = trainer_script.main(argv_arm)
        if exit_code != 0:
            print(f"[ERROR] arm lambda={lambda_rate} failed with exit code {exit_code}", file=sys.stderr)
            return 1

        history = json.loads((arm_dir / "history.json").read_text(encoding="utf-8"))
        # The arm's own epochs only - `history` carries the starting
        # checkpoint's prior epochs too, since --resume restores them.
        arm_history = history[-args.epochs:]
        first, last = arm_history[0], arm_history[-1]

        measured = _measure_arm(
            arm_dir / "latest.pt",
            manifest=args.manifest, calibration=args.calibration,
            qat_bits=args.qat_bits, qat_mode=args.qat_mode, lambda_rate=lambda_rate,
            batch_size=args.batch_size, max_batches=args.max_batches,
            num_workers=defaults.num_workers, crop_size=defaults.random_crop_size,
            device=device, seed=args.seed,
        )

        row = {
            "lambda": lambda_rate,
            "multiplier_of_balance": arm["multiplier"],
            "regime": arm["regime"],
            "checkpoint_dir": str(arm_dir),
            "initial_train_distortion": first["train_distortion"],
            "final_train_distortion": last["train_distortion"],
            "initial_train_rate_bpp": first["train_rate_bpp"],
            "final_train_rate_bpp": last["train_rate_bpp"],
            "initial_train_total_loss": first["train_loss"],
            "final_train_total_loss": last["train_loss"],
            "epoch_val_distortion": last["val_distortion"],
            "epoch_val_rate_bpp": last["val_rate_bpp"],
            "epoch_val_psnr_db": last["val_psnr"],
            **measured,
            "epochs_per_arm": args.epochs,
            "steps_per_arm": args.epochs * args.max_batches,
        }
        rows.append(row)
        print(f"      train D {first['train_distortion']:.6e} -> {last['train_distortion']:.6e} | "
              f"R {first['train_rate_bpp']:.4f} -> {last['train_rate_bpp']:.4f} bpp")
        print(f"      val   D {measured['val_distortion']:.6e}  R {measured['val_rate_bpp']:.4f} bpp  "
              f"PSNR {measured['val_psnr_db']:.2f} dB")

    summary = {
        "phase": "M9C phases 4-6 (lambda pilot)",
        "lambda_balance": balance,
        "lambda_balance_source": balance_source,
        "lambda_multipliers": list(LAMBDA_MULTIPLIERS),
        "start_checkpoint": str(args.checkpoint),
        "calibration": str(args.calibration),
        "qat_bits": args.qat_bits,
        "qat_mode": args.qat_mode,
        "epochs": args.epochs,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "rate_lr": args.rate_lr,
        "seed": args.seed,
        "device": str(device),
        "note": (
            "R is the training-time Laplace proxy in bits per input pixel, NOT a "
            "measured .nvc payload bitrate. No recalibration or codec benchmarking "
            "was performed in M9C."
        ),
        "arms": rows,
    }
    summary_path = args.output_dir / "lambda_pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = args.output_dir / "lambda_pilot_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    table_lines = [
        "| lambda | x balance | regime | init D | final D | init R | final R | init total | final total | val D | val R | val PSNR |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['lambda']:.4e} | {row['multiplier_of_balance']:.3g} | {row['regime']} | "
            f"{row['initial_train_distortion']:.4e} | {row['final_train_distortion']:.4e} | "
            f"{row['initial_train_rate_bpp']:.4f} | {row['final_train_rate_bpp']:.4f} | "
            f"{row['initial_train_total_loss']:.4e} | {row['final_train_total_loss']:.4e} | "
            f"{row['val_distortion']:.4e} | {row['val_rate_bpp']:.4f} | {row['val_psnr_db']:.2f} |"
        )
    table = "\n".join(table_lines)
    (args.output_dir / "lambda_pilot_table.md").write_text(table + "\n", encoding="utf-8")

    plotted = False
    if not args.no_plot:
        plotted = _write_plot(rows, args.output_dir / "lambda_pilot_rd.png")

    print("\n" + "=" * 70)
    print(table)
    print("=" * 70)
    print(f"summary: {summary_path}")
    print(f"csv:     {csv_path}")
    if plotted:
        print(f"plot:    {args.output_dir / 'lambda_pilot_rd.png'}")
    print("\nR is the training-time proxy, NOT measured .nvc bitrate (M9C Rule 9).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
