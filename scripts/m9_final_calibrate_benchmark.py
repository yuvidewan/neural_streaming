"""Milestone 9 final evaluation: fresh calibration, then real `.nvc` benchmarking.

THE POINT OF THIS SCRIPT
-------------------------
Everything M9 has measured so far is a training-time proxy. This is where the
milestone's actual question gets answered against the real codec:

    does rate-aware training improve the DEPLOYED rate-distortion tradeoff?

A lower proxy R is not sufficient evidence, and this script is deliberately
built so it cannot be mistaken for such evidence: the calibration step and the
benchmark step are separate, and the benchmark reports measured payload bytes
from real `.nvc` encode/decode, not any Laplace estimate.

FRESH CALIBRATION, PER MODEL
-----------------------------
Each model gets its OWN calibration at each bit depth, computed from the TRAIN
split only. Nothing is reused - not M7's, not M8's, not the M9C diagnostic
grid, and not another lambda's. This matters more than usual here: M9C measured
that the M8 QAT model clips 48.9% against the M7-derived grid because its
latent drifted far outside it, and the rate-aware models shrink their latents
by very different amounts (M9C.1 saw latent abs-mean spanning 7.8 to 1.8 across
lambdas). A shared grid would systematically favour whichever model happened to
match it.

Methodology is the established one, unchanged from M7/M8
(`scripts/calibrate_quantizer.py` defaults): per-channel percentile,
0.1/99.9 bounds, 50 batches x batch size frames, train split, seed 42.

THE BASELINES ARE RECALIBRATED TOO
-----------------------------------
M8's own calibration files (`qat_combined_noise*.json`) are not on disk, and in
any case a comparison is only clean if every model goes through the identical
procedure in the same run. So M7 and M8-QAT are recalibrated here with the same
script and the same parameters, rather than citing published numbers measured
under a different invocation. M7's and M8's own historical calibration files
and results are left untouched; these are written to the M9 output directory.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\m9_final_calibrate_benchmark.py                  # both stages
    python scripts\\m9_final_calibrate_benchmark.py --stage calibrate
    python scripts\\m9_final_calibrate_benchmark.py --stage benchmark
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from nvc.compression.calibration import load_calibration
from nvc.utils.config import load_default_config

DEFAULT_OUTPUT_DIR = Path("outputs/m9_final")
# The project's existing calibration-fit guard (see rd_benchmark's
# check_calibration_fit and MILESTONE_8_RESULTS.md section 4).
CLIP_GUARD_PERCENT = 2.0
BIT_DEPTHS = (8, 6, 4)
# 8-bit lives in the base filename; other depths take a _<n>bit suffix. This is
# the convention benchmark_rd.py's own _calibration_path_for_bits expects.
PRIMARY_BITS = 8


def _model_set(output_dir: Path) -> list[dict[str, Any]]:
    """Every model in the final comparison, in report order."""
    return [
        {"key": "M7", "label": "M7 baseline",
         "checkpoint": Path("outputs/checkpoints/vimeo_epoch17_best.pt")},
        {"key": "M8-QAT", "label": "M8 QAT (immediate baseline)",
         "checkpoint": Path("outputs/qat_combined/checkpoints_qat_noise/best.pt")},
        {"key": "M9-CTRL", "label": "M9 control (lambda=0)",
         "checkpoint": output_dir / "control_lambda0" / "best.pt"},
        {"key": "M9-L", "label": "M9-L (lambda 9.0757e-04)",
         "checkpoint": output_dir / "lambda_9e-4" / "best.pt"},
        {"key": "M9-M", "label": "M9-M (lambda 2.8700e-03)",
         "checkpoint": output_dir / "lambda_2.87e-3" / "best.pt"},
        {"key": "M9-H", "label": "M9-H (lambda 9.0757e-03)",
         "checkpoint": output_dir / "lambda_9e-3" / "best.pt"},
    ]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Milestone 9 final evaluation: fresh per-model calibration at 8/6/4-bit, "
            "then real .nvc encode/decode benchmarking on the DAVIS test split."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stage", choices=["calibrate", "benchmark", "both"], default="both",
        help="Run only one stage. Benchmarking requires calibration to have run first.",
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--calibration-batches", type=int, default=50,
        help="Calibration batches (the established M7/M8 value; 50 x batch size frames).",
    )
    parser.add_argument("--mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--only", nargs="+", default=None, metavar="KEY",
        help="Restrict to these model keys (M7, M8-QAT, M9-CTRL, M9-L, M9-M, M9-H).",
    )
    parser.add_argument(
        "--allow-clipping", action="store_true",
        help="Continue even if a calibration exceeds the project's clipping guard. "
             "Off by default: pathological clipping is a STOP condition for M9.",
    )
    return parser


def _calibration_path(calibration_dir: Path, key: str, bits: int) -> Path:
    stem = key.replace("-", "_").lower()
    if bits == PRIMARY_BITS:
        return calibration_dir / f"{stem}.json"
    return calibration_dir / f"{stem}_{bits}bit.json"


def _calibrate(args, models: list[dict[str, Any]], calibration_dir: Path) -> dict[str, Any]:
    calibrate = _load_script("calibrate_quantizer")
    calibration_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    print("=" * 78)
    print("STAGE 1 - FRESH CALIBRATION (train split only, per model, per bit depth)")
    print(f"method: per-channel percentile 0.1/99.9, {args.calibration_batches} batches "
          f"x {args.batch_size} frames, seed {args.seed}")
    print("=" * 78)

    for model in models:
        if not model["checkpoint"].is_file():
            print(f"[ERROR] checkpoint missing for {model['key']}: {model['checkpoint']}",
                  file=sys.stderr)
            return {"ok": False, "rows": rows, "error": f"missing checkpoint for {model['key']}"}

        for bits in BIT_DEPTHS:
            output = _calibration_path(calibration_dir, model["key"], bits)
            exit_code = calibrate.main([
                "--checkpoint", str(model["checkpoint"]),
                "--manifest", str(args.manifest),
                "--bits", str(bits), "--mode", args.mode,
                "--batch-size", str(args.batch_size),
                "--max-batches", str(args.calibration_batches),
                "--seed", str(args.seed), "--device", args.device,
                "--output", str(output),
                "--metrics-dir", str(calibration_dir / "metrics" / model["key"]),
                "--visualizations-dir", str(calibration_dir / "visualizations" / model["key"]),
            ])
            if exit_code != 0:
                failures.append(f"{model['key']} @ {bits}-bit (exit {exit_code})")
                continue

            document = load_calibration(output)
            clipping = document["calibration_metadata"]["clipping"]
            clipped = float(clipping["clipped_percent"])
            healthy = clipped <= CLIP_GUARD_PERCENT
            if not healthy:
                failures.append(f"{model['key']} @ {bits}-bit clips {clipped:.3f}%")
            rows.append({
                "model": model["key"],
                "label": model["label"],
                "bits": bits,
                "calibration": str(output),
                "clipped_percent": clipped,
                "clipped_low": clipping["clipped_low"],
                "clipped_high": clipping["clipped_high"],
                "guard_percent": CLIP_GUARD_PERCENT,
                "within_guard": healthy,
                "checkpoint": str(model["checkpoint"]),
            })
            flag = "OK " if healthy else "OVER"
            print(f"  [{flag}] {model['key']:<8} {bits}-bit  clipping {clipped:6.3f}%  -> {output.name}")

    return {"ok": not failures, "rows": rows, "failures": failures}


def _benchmark(args, models: list[dict[str, Any]], calibration_dir: Path,
               benchmark_dir: Path) -> dict[str, Any]:
    benchmark = _load_script("benchmark_rd")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    print("=" * 78)
    print(f"STAGE 2 - REAL .nvc ENCODE/DECODE on the DAVIS {args.split} split")
    print("Measured payload bytes - NOT the training proxy.")
    print("=" * 78)

    for model in models:
        base = _calibration_path(calibration_dir, model["key"], PRIMARY_BITS)
        if not base.is_file():
            print(f"[ERROR] no calibration for {model['key']} - run --stage calibrate first",
                  file=sys.stderr)
            return {"ok": False, "rows": rows, "error": f"missing calibration for {model['key']}"}

        run_name = f"m9_final_{model['key'].replace('-', '_').lower()}"
        argv = [
            "--dataset", "davis", "--manifest", str(args.manifest), "--split", args.split,
            "--codecs", "nvc",
            "--checkpoint", str(model["checkpoint"]), "--calibration", str(base),
            "--nvc-bits", *[str(b) for b in BIT_DEPTHS],
            "--output-dir", str(benchmark_dir), "--run-name", run_name,
            "--device", args.device, "--seed", str(args.seed),
        ]
        if args.max_sequences is not None:
            argv += ["--max-sequences", str(args.max_sequences)]
        if args.max_frames_per_sequence is not None:
            argv += ["--max-frames-per-sequence", str(args.max_frames_per_sequence)]

        print(f"\n--- {model['key']}: {model['checkpoint']}")
        exit_code = benchmark.main(argv)
        if exit_code != 0:
            print(f"[ERROR] benchmark failed for {model['key']} (exit {exit_code})", file=sys.stderr)
            return {"ok": False, "rows": rows, "error": f"benchmark failed for {model['key']}"}

        results_path = benchmark_dir / run_name / "results.json"
        if not results_path.is_file():
            print(f"[ERROR] no results.json for {model['key']}", file=sys.stderr)
            return {"ok": False, "rows": rows, "error": f"no results for {model['key']}"}

        payload = json.loads(results_path.read_text(encoding="utf-8"))
        for entry in payload["aggregate"]:
            row = {"model": model["key"], "label": model["label"], **entry}
            # bytes/frame is not in the aggregate record but is asked for in the
            # M9 report, and is exactly derivable from what is.
            if entry.get("total_frames"):
                row["bytes_per_frame"] = entry["total_bytes"] / entry["total_frames"]
            rows.append(row)

    return {"ok": True, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        print(f"[ERROR] --manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    models = _model_set(args.output_dir)
    if args.only is not None:
        wanted = {key.upper() for key in args.only}
        models = [model for model in models if model["key"].upper() in wanted]
        if not models:
            parser.error("--only matched no models")

    calibration_dir = args.output_dir / "calibration"
    benchmark_dir = args.output_dir / "benchmarks"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration_result: dict[str, Any] = {}
    if args.stage in ("calibrate", "both"):
        calibration_result = _calibrate(args, models, calibration_dir)
        (args.output_dir / "calibration_report.json").write_text(
            json.dumps(
                {
                    "method": "per_channel_percentile",
                    "lower_percentile": 0.1, "upper_percentile": 99.9,
                    "calibration_batches": args.calibration_batches,
                    "batch_size": args.batch_size,
                    "calibration_split": "train",
                    "seed": args.seed,
                    "clip_guard_percent": CLIP_GUARD_PERCENT,
                    "rows": calibration_result["rows"],
                    "failures": calibration_result.get("failures", []),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if not calibration_result["ok"]:
            print("\n[STOP] calibration did not pass the clipping guard / failed:",
                  file=sys.stderr)
            for failure in calibration_result.get("failures", []):
                print(f"  - {failure}", file=sys.stderr)
            if not args.allow_clipping:
                print("Refusing to benchmark on a calibration that fails the project's own "
                      "fit guard. Re-run with --allow-clipping only if this is understood.",
                      file=sys.stderr)
                return 1

    if args.stage in ("benchmark", "both"):
        benchmark_result = _benchmark(args, models, calibration_dir, benchmark_dir)
        if not benchmark_result["ok"]:
            return 1
        rows = benchmark_result["rows"]
        (args.output_dir / "benchmark_aggregate.json").write_text(
            json.dumps(
                {
                    "split": args.split,
                    "bit_depths": list(BIT_DEPTHS),
                    "note": (
                        "bpp here is MEASURED .nvc payload, not the training-time Laplace "
                        "proxy. The two are different quantities and must not be compared "
                        "as if they were the same."
                    ),
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if rows:
            fieldnames: list[str] = []
            for row in rows:
                for key in row:
                    if key not in fieldnames and not isinstance(row[key], (dict, list)):
                        fieldnames.append(key)
            with (args.output_dir / "benchmark_aggregate.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        print(f"\nAggregate: {args.output_dir / 'benchmark_aggregate.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
