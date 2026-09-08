"""M10G Part A/B: a deterministic, pre-declared checkpoint evaluation convention.

WHY THIS EXISTS
----------------
M10E and M10F both evaluated the FINAL training snapshot, and both were bitten
by it. In M10F two of ten runs ended on a degraded epoch, and one of those was a
CONTROL: `CTRL@s42` finished 3.04% worse than its own best epoch, which inflated
that experiment's measured noise floor from ~1.6 to 4.86 BD-rate points and
lifted every seed-42 candidate scored against it. The boundary question had to
be answered by a control-independent analysis instead.

The failure mode is not noise in the model - it is the *selection rule*. A single
unlucky final epoch is a coin flip that the convention promotes to the deployed
result.

THE CONVENTION (declared here, before any future experiment runs)
------------------------------------------------------------------
    PRIMARY   : the best VALIDATION checkpoint under the experiment's own
                declared training objective.
    SECONDARY : the final training checkpoint, retained for convergence
                diagnostics - not deleted, not deployed.

Three properties make it safe:

1. **Deterministic.** Minimum objective value wins; exact ties go to the
   EARLIEST epoch. Two runs over the same history always select the same
   checkpoint, on any machine.

2. **Declared before training.** The objective key is an input to the run, not
   something chosen after looking at results. `--objective-key` exists so an
   experiment records what it selected on.

3. **Structurally unable to see the test set.** `select_checkpoint` takes only
   validation history. Every record is screened against a forbidden-key list, so
   a test metric that leaks into a history file raises `TestMetricLeakError`
   rather than quietly influencing the choice. Selecting on deployed `.nvc`
   BD-rate, test PSNR or test BPP would be selection on the evaluation set - the
   exact thing the whole benchmark is supposed to measure honestly.

For rate-aware training the objective is `val_loss` = D + lambda*R, the quantity
actually optimised - NOT proxy bitrate alone, which would prefer a model that
throws away quality to save rate.

WHAT PART B DOES
-----------------
A NON-DESTRUCTIVE re-reading of the M10F (and, as supporting evidence, M10E)
artifacts. It does not retrain, does not re-benchmark, and does not overwrite a
single historical result. It answers one question: how often does the final
snapshot differ from the best one, and by how much? Historical M10A-M10F results
remain valid under the final-snapshot convention they declared.

Example usage (PowerShell, from the project root, with .venv activated):

    .venv\\Scripts\\python.exe scripts\\m10g_evaluation_convention.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from nvc.utils.config import load_default_config

DEFAULT_OUTPUT_DIR = Path("outputs/m10g_temporal_baseline")
M10F_DIR = Path("outputs/m10f_lambda_boundary")
M10E_DIR = Path("outputs/m10e_lambda_lock")

# The declared default objective. For rate-aware training this is D + lambda*R,
# the quantity the optimiser actually minimises.
DEFAULT_OBJECTIVE_KEY = "val_loss"

# A checkpoint may never be selected using anything measured on the test split.
# Substrings, matched case-insensitively against every history key - a history
# record carrying any of these is a leak, not a selection input.
FORBIDDEN_KEY_SUBSTRINGS: tuple[str, ...] = (
    "test_", "bd_rate", "bdrate", "aggregate_bpp", "deployed", "benchmark",
    "nvc_bpp", "msssim_test", "test-",
)

# Only these prefixes may be used as a selection objective.
ALLOWED_OBJECTIVE_PREFIXES: tuple[str, ...] = ("val_", "validation_")


class TestMetricLeakError(RuntimeError):
    """Raised when checkpoint selection is handed a test-set metric.

    Deliberately an exception rather than a filter: silently ignoring the key
    would leave the caller believing selection was clean when the history file
    itself is wrong, and that file would go on being reused.
    """


class SelectionError(RuntimeError):
    """Raised when a history cannot produce a deterministic selection."""


def assert_validation_only(record: dict[str, Any]) -> None:
    """Screen one history record for test-set contamination."""
    for key in record:
        lowered = str(key).lower()
        for forbidden in FORBIDDEN_KEY_SUBSTRINGS:
            if forbidden in lowered:
                raise TestMetricLeakError(
                    f"History record contains '{key}', which looks like a test-set "
                    f"metric (matched '{forbidden}'). Checkpoint selection must use "
                    f"validation metrics only - selecting on the test split would "
                    f"invalidate every number the benchmark then reports."
                )


def select_checkpoint(
    history: Iterable[dict[str, Any]],
    *,
    objective_key: str = DEFAULT_OBJECTIVE_KEY,
    objective_filter: str | None = "rate_enabled",
) -> dict[str, Any]:
    """The convention, as executable code.

    Returns a selection record describing BOTH the primary (best-validation)
    and secondary (final) checkpoints, plus the gap between them.

    `objective_filter`, when given, is a boolean history key used to drop
    records produced under a DIFFERENT objective - the stale pure-MSE epochs
    that `--resume-model-only` carries in from a previous milestone. Comparing
    a rate-aware objective against those would select an unrelated checkpoint;
    this is the same rule M9C.1's `_same_objective` fix established.
    """
    records = list(history)
    if not records:
        raise SelectionError("Cannot select a checkpoint from an empty history.")

    if not any(objective_key.lower().startswith(p) for p in ALLOWED_OBJECTIVE_PREFIXES):
        raise TestMetricLeakError(
            f"Objective key '{objective_key}' is not a validation metric (expected one "
            f"starting with {ALLOWED_OBJECTIVE_PREFIXES}). Checkpoint selection may not "
            f"use test-set quantities."
        )

    for record in records:
        assert_validation_only(record)

    if objective_filter is not None and any(objective_filter in r for r in records):
        own = [r for r in records if r.get(objective_filter)]
        stale = len(records) - len(own)
    else:
        own, stale = records, 0
    if not own:
        raise SelectionError(
            f"No history record matches the current objective "
            f"(filter '{objective_filter}'); nothing can be selected."
        )

    missing = [r for r in own if objective_key not in r]
    if missing:
        raise SelectionError(
            f"{len(missing)} history record(s) have no '{objective_key}'; the objective "
            f"must be present on every epoch for selection to be deterministic."
        )

    # Deterministic: minimum objective, EARLIEST epoch breaks an exact tie.
    best = min(own, key=lambda r: (r[objective_key], r.get("epoch", 0)))
    final = own[-1]

    best_value, final_value = best[objective_key], final[objective_key]
    return {
        "convention": "M10G: best validation checkpoint (primary), final checkpoint (secondary)",
        "selection_metric": objective_key,
        "selection_domain": "validation",
        "tie_break": "earliest epoch",
        "objective_filter": objective_filter,
        "stale_records_ignored": stale,
        "epochs_considered": len(own),
        "selected_epoch": best.get("epoch"),
        "selected_objective": best_value,
        "selected_psnr_db": best.get("val_psnr"),
        "final_epoch": final.get("epoch"),
        "final_objective": final_value,
        "final_psnr_db": final.get("val_psnr"),
        "epoch_difference": (final.get("epoch") or 0) - (best.get("epoch") or 0),
        "best_vs_final_absolute": final_value - best_value,
        "best_vs_final_percent": (
            (final_value - best_value) / best_value * 100 if best_value else None
        ),
        "psnr_difference_db": (
            final.get("val_psnr") - best.get("val_psnr")
            if final.get("val_psnr") is not None and best.get("val_psnr") is not None else None
        ),
        "final_is_best": best.get("epoch") == final.get("epoch"),
    }


def resolve_checkpoint_paths(run_dir: Path, selection: dict[str, Any],
                             snapshots: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Locate the artifacts the selection refers to, and say plainly when one
    is missing rather than inventing a substitute."""
    best_path = run_dir / "best.pt"
    final_path = None
    if snapshots:
        candidate = Path(snapshots[-1]["path"])
        final_path = candidate if candidate.is_file() else None
    if final_path is None:
        candidate = run_dir / "latest.pt"
        final_path = candidate if candidate.is_file() else None

    return {
        "run_dir": str(run_dir),
        "primary_checkpoint": str(best_path) if best_path.is_file() else None,
        "primary_available": best_path.is_file(),
        "secondary_checkpoint": str(final_path) if final_path else None,
        "secondary_available": final_path is not None,
        "note": (
            None if best_path.is_file()
            else "best.pt is NOT present for this run; the new convention could not be "
                 "applied to it retrospectively. Reported as unavailable, not substituted."
        ),
    }


def analyse_experiment(directory: Path, label: str) -> dict[str, Any] | None:
    """Part B for one historical experiment - read only, never write into it."""
    summary_path = directory / "training_summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = summary.get("runs") or summary.get("arms") or []
    if not runs:
        return None

    rows = []
    for run in runs:
        run_dir = Path(run["checkpoint_dir"])
        history_path = run_dir / "history.json"
        row: dict[str, Any] = {
            "experiment": label, "run": run.get("name"),
            "lambda": run.get("lambda"), "seed": run.get("seed"),
        }
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            try:
                row["selection"] = select_checkpoint(history)
            except (SelectionError, TestMetricLeakError) as error:
                row["selection_error"] = str(error)
        else:
            row["selection_error"] = f"history.json not found at {history_path}"
        row["artifacts"] = resolve_checkpoint_paths(run_dir, row.get("selection", {}),
                                                    run.get("snapshots"))
        rows.append(row)
    return {"experiment": label, "directory": str(directory), "runs": rows}


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10G Part A/B: declare the checkpoint convention and apply it "
                    "retrospectively, non-destructively, to M10E/M10F.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--m10f-dir", type=Path, default=M10F_DIR)
    parser.add_argument("--m10e-dir", type=Path, default=M10E_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--objective-key", default=DEFAULT_OBJECTIVE_KEY)
    parser.add_argument("--flag-percent", type=float, default=2.0,
                        help="Best-vs-final gap above which a run is flagged as degraded.")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    print("=" * 100)
    print("M10G PART A - CHECKPOINT EVALUATION CONVENTION (declared for all FUTURE experiments)")
    print("=" * 100)
    print(f"  PRIMARY   : best validation checkpoint by '{args.objective_key}'")
    print("  SECONDARY : final training checkpoint (kept for convergence diagnostics)")
    print("  tie-break : earliest epoch")
    print("  domain    : validation only - selection cannot access test metrics")
    print(f"  forbidden : any history key containing {FORBIDDEN_KEY_SUBSTRINGS}")
    print("  rationale : the final snapshot is a coin flip; M10F promoted a degraded")
    print("              CONTROL epoch to the deployed result and inflated that")
    print("              experiment's noise floor ~3x.")
    print()
    print("  Historical M10A-M10F results are NOT restated under this convention.")
    print("  They remain valid under the final-snapshot convention they declared.")

    experiments = []
    for directory, label in ((args.m10f_dir, "M10F"), (args.m10e_dir, "M10E")):
        result = analyse_experiment(directory, label)
        if result is None:
            print(f"\n[skip] {label}: no training_summary.json with runs at {directory}")
            continue
        experiments.append(result)

    if not experiments:
        print("[ERROR] no historical experiment artifacts found to analyse", file=sys.stderr)
        return 1

    print()
    print("=" * 100)
    print("M10G PART B - WHAT THE NEW CONVENTION WOULD HAVE SELECTED (non-destructive)")
    print("=" * 100)
    print(f"{'exp':<6} {'run':<20} {'best ep':>8} {'final ep':>9} {'d ep':>5} "
          f"{'best obj':>12} {'final obj':>12} {'gap %':>7} {'dPSNR':>8} {'best.pt':>8}  flag")

    totals = {"runs": 0, "degraded": 0, "missing_best": 0, "errors": 0}
    for experiment in experiments:
        for row in experiment["runs"]:
            totals["runs"] += 1
            if "selection_error" in row:
                totals["errors"] += 1
                print(f"{experiment['experiment']:<6} {str(row['run']):<20} "
                      f"  [ERROR] {row['selection_error']}")
                continue
            s = row["selection"]
            available = row["artifacts"]["primary_available"]
            if not available:
                totals["missing_best"] += 1
            gap = s["best_vs_final_percent"] or 0.0
            degraded = gap > args.flag_percent
            if degraded:
                totals["degraded"] += 1
            print(f"{experiment['experiment']:<6} {str(row['run']):<20} "
                  f"{s['selected_epoch']:>8} {s['final_epoch']:>9} {s['epoch_difference']:>5} "
                  f"{s['selected_objective']:>12.6e} {s['final_objective']:>12.6e} "
                  f"{gap:>6.2f}% {(s['psnr_difference_db'] or 0):>+8.3f} "
                  f"{'yes' if available else 'NO':>8}  {'<-- DEGRADED' if degraded else ''}")

    print()
    print(f"  runs analysed              : {totals['runs']}")
    print(f"  degraded final snapshots   : {totals['degraded']} "
          f"({totals['degraded'] / totals['runs'] * 100:.0f}% of runs, "
          f"gap > {args.flag_percent}%)")
    print(f"  best.pt unavailable        : {totals['missing_best']}")
    print(f"  selection errors           : {totals['errors']}")
    print()
    print("  The new convention is implementable on every run where best.pt exists:")
    print("  it needs only validation history, which every run already records.")
    print("  Historical benchmark results are untouched - nothing above was re-run.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "M10G Part A/B (evaluation convention)",
        "convention": {
            "primary": "best validation checkpoint under the declared training objective",
            "secondary": "final training checkpoint, retained for convergence diagnostics",
            "selection_metric_default": args.objective_key,
            "selection_domain": "validation",
            "tie_break": "earliest epoch",
            "declared_before_training": True,
            "forbidden_key_substrings": list(FORBIDDEN_KEY_SUBSTRINGS),
            "allowed_objective_prefixes": list(ALLOWED_OBJECTIVE_PREFIXES),
            "rationale": (
                "M10E and M10F both evaluated the final snapshot; M10F promoted a degraded "
                "CONTROL epoch (3.04% worse than its own best) to the deployed result, "
                "inflating that experiment's measured noise floor from ~1.6 to 4.86 BD-rate "
                "points. The failure is in the selection rule, not the model."
            ),
            "historical_note": (
                "M10A-M10F are NOT restated under this convention. They remain valid under "
                "the final-snapshot convention they declared before running."
            ),
        },
        "part_b_totals": totals,
        "part_b_flag_percent": args.flag_percent,
        "experiments": experiments,
    }
    path = args.output_dir / "evaluation_convention.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
