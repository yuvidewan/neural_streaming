"""M10F evaluation: fresh calibration, deployed benchmark, and the lower-boundary decision.

Turns the 10-run (lambda x seed) boundary grid into an answer to one question -
does the RD basin keep improving below lambda = 3e-4? - using the deployed
`.nvc` codec as the authority throughout.

WHAT THIS ADDS OVER M10E
-------------------------
1. **A bridge to M10E.** The 3e-4 and 4.5e-4 arms were retrained here on
   purpose. Comparing them against M10E's own 3e-4 / 4.5e-4 results measures
   inter-experiment drift directly, so a movement at 1e-4 / 2e-4 can be judged
   against how much two independent experiments disagree about the SAME lambda
   rather than against an assumption that they would agree exactly.

2. **Best-vs-final as a first-class analysis.** M10E's 7.5e-4 seed43 arm was
   contaminated by one unlucky final epoch under the fixed final-snapshot
   convention. Here every run's gap is reported up front. The convention does
   not change - the final 18,120-step snapshot remains the primary result - but
   a systematic problem becomes visible instead of being discovered by hand.

3. **An explicit boundary verdict.** The lambda -> BD-rate shape is classified
   into one of five cases (keeps improving / minimum near 2e-4 / minimum near
   3e-4 / flat within noise / unstable at low lambda) rather than left to
   narrative.

REUSE, NOT REIMPLEMENTATION
----------------------------
The BD-rate integration, correlation math and bit-depth parsing are imported
from `m10e_evaluate` rather than copied, so the two milestones cannot drift
apart in how they compute the numbers being compared. Calibration and
benchmarking come from `m9_final_calibrate_benchmark`, exactly as M10E did.

FOUR QUESTIONS ABOUT THE PROXY, KEPT SEPARATE
----------------------------------------------
    1. does lower lambda produce lower proxy R?
    2. does proxy R order actual .nvc bitrate?
    3. does the lambda minimising proxy R minimise PSNR BD-rate?
    4. does it minimise MS-SSIM BD-rate?

M10D and M10E both found 2 can hold while 3 fails. Conflating them is how a
rate proxy gets mistaken for an RD criterion.

Example usage (PowerShell, from the project root, with .venv activated):

    .venv\\Scripts\\python.exe scripts\\m10f_evaluate.py --stage all
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from nvc.utils.config import load_default_config

DEFAULT_OUTPUT_DIR = Path("outputs/m10f_lambda_boundary")
M10E_DIR = Path("outputs/m10e_lambda_lock")
BIT_DEPTHS = (8, 6, 4)
# A candidate is "clearly above noise" only if its improvement exceeds the
# measured control-to-control spread by this factor. 2x is a deliberate,
# stated convention - not a significance test, which n=2 cannot support.
NOISE_FACTOR = 2.0
# A best-vs-final validation gap above this is flagged as a methodology caveat
# (M10E's contaminated arm was 4.08%; every clean run there was under 1.3%).
BEST_VS_FINAL_FLAG_PERCENT = 2.0


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def final_models(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """One benchmark model per run: its final 18,120-step snapshot.

    Same convention as M10D and M10E. Deliberately NOT best.pt - switching
    after seeing results would be exactly the post-hoc change the design
    forbids.
    """
    models = []
    for run in summary["runs"]:
        final = run["snapshots"][-1]
        models.append({
            "key": run["name"].replace("@", "_"),
            "label": f"{run['lambda_name']} lambda={run['lambda']:.4e} seed={run['seed']}",
            "checkpoint": Path(final["path"]),
            "lambda": run["lambda"], "lambda_name": run["lambda_name"],
            "seed": run["seed"], "step": final["step"],
            "checkpoint_sha256": final["sha256"],
        })
    return models


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10F: calibrate, benchmark and close the lower lambda boundary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--m10e-dir", type=Path, default=M10E_DIR)
    parser.add_argument("--stage", choices=["calibrate", "benchmark", "analyse", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--calibration-batches", type=int, default=50)
    parser.add_argument("--mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow-clipping", action="store_true")
    return parser


def classify_boundary(curve_rows: list[dict[str, Any]], floor: float) -> dict[str, Any]:
    """Which of the five shapes does lambda -> BD-rate actually have?

    Decided from the measured points and the measured noise floor, not from
    whichever number happens to be smallest. `curve_rows` must be the
    rate-aware arms only, ascending in lambda.
    """
    points = [(r["lambda"], r["mean_bd_rate_psnr"]) for r in curve_rows
              if r["mean_bd_rate_psnr"] is not None]
    if len(points) < 2:
        return {"case": "INDETERMINATE", "reason": "not enough measured arms"}

    points.sort(key=lambda p: p[0])
    values = [v for _, v in points]
    best_index = min(range(len(values)), key=lambda i: values[i])
    best_lambda, best_value = points[best_index]
    span = max(values) - min(values)

    # Flat first: if nothing separates the arms, no minimum can be claimed.
    if span <= floor:
        case, reason = "D_FLAT_WITHIN_NOISE", (
            f"total spread {span:.2f} pts <= noise floor {floor:.2f} pts")
    elif best_index == 0:
        case, reason = "A_KEEPS_IMPROVING_AS_LAMBDA_DECREASES", (
            f"best arm is the smallest lambda tested ({best_lambda:.1e}); "
            "the basin is still open below")
    elif best_index == len(points) - 1:
        case, reason = "E_BEST_AT_UPPER_EDGE", (
            f"best arm is the largest lambda tested ({best_lambda:.1e}); "
            "the sweep does not bracket the optimum from above")
    else:
        # Interior minimum: bracketed on both sides. Whether it is a RESOLVED
        # minimum depends on the neighbours being separated from it by noise.
        left, right = values[best_index - 1], values[best_index + 1]
        margin = min(left - best_value, right - best_value)
        resolved = margin > floor
        case = "B_OR_C_INTERIOR_MINIMUM"
        reason = (
            f"minimum at lambda={best_lambda:.1e}, bracketed on both sides; "
            f"nearest neighbour is {margin:.2f} pts worse "
            f"({'above' if resolved else 'within'} the {floor:.2f}-pt floor)"
        )
        return {
            "case": case, "reason": reason, "best_lambda": best_lambda,
            "best_bd_rate": best_value, "neighbour_margin_points": margin,
            "bracketed": True, "separated_from_neighbours": resolved,
            "total_span_points": span, "noise_floor_points": floor,
        }

    return {
        "case": case, "reason": reason, "best_lambda": best_lambda,
        "best_bd_rate": best_value, "bracketed": False,
        "total_span_points": span, "noise_floor_points": floor,
    }


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        print(f"[ERROR] --manifest not found: {args.manifest}", file=sys.stderr)
        return 1
    summary_path = args.output_dir / "training_summary.json"
    if not summary_path.is_file():
        print(f"[ERROR] training_summary.json not found: {summary_path}. Run "
              "scripts/m10f_lambda_boundary.py first.", file=sys.stderr)
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "runs" not in summary:
        print(f"[ERROR] {summary_path} has no completed runs "
              f"(status: {summary.get('status', 'unknown')}).", file=sys.stderr)
        return 1

    m9 = _load_script("m9_final_calibrate_benchmark")
    m10e = _load_script("m10e_evaluate")  # BD-rate / correlation math, reused not copied
    models = final_models(summary)
    calibration_dir = args.output_dir / "calibration"
    benchmark_dir = args.output_dir / "benchmarks"

    if args.stage in ("calibrate", "all"):
        result = m9._calibrate(args, models, calibration_dir)
        for row in result["rows"]:
            model = next((m for m in models if m["key"] == row["model"]), None)
            if model is not None:
                row.update({
                    "checkpoint_sha256": model["checkpoint_sha256"],
                    "lambda": model["lambda"], "lambda_name": model["lambda_name"],
                    "seed": model["seed"], "training_steps": model["step"],
                })
        (args.output_dir / "calibration_report.json").write_text(
            json.dumps({
                "method": "per_channel_percentile", "lower_percentile": 0.1,
                "upper_percentile": 99.9, "calibration_batches": args.calibration_batches,
                "calibration_frames": args.calibration_batches * args.batch_size,
                "batch_size": args.batch_size, "calibration_split": "train",
                "calibration_seed": args.seed, "clip_guard_percent": m9.CLIP_GUARD_PERCENT,
                "provenance_note": (
                    "Every grid is fitted to its OWN checkpoint on the TRAIN split only - no "
                    "validation or test frames, no sharing between runs, and nothing reused "
                    "from M10D or M10E."
                ),
                "rows": result["rows"], "failures": result.get("failures", []),
            }, indent=2),
            encoding="utf-8",
        )
        if not result["ok"] and not args.allow_clipping:
            print("\n[STOP] calibration failed the clipping guard:", file=sys.stderr)
            for failure in result.get("failures", []):
                print(f"  - {failure}", file=sys.stderr)
            return 1

    if args.stage in ("benchmark", "all"):
        result = m9._benchmark(args, models, calibration_dir, benchmark_dir)
        if not result["ok"]:
            return 1
        (args.output_dir / "benchmark_aggregate.json").write_text(
            json.dumps({
                "split": args.split, "bit_depths": list(BIT_DEPTHS),
                "note": "aggregate_bpp is MEASURED .nvc payload, not the training proxy.",
                "rows": result["rows"],
            }, indent=2),
            encoding="utf-8",
        )

    if args.stage not in ("analyse", "all"):
        return 0

    aggregate_path = args.output_dir / "benchmark_aggregate.json"
    if not aggregate_path.is_file():
        print(f"[ERROR] benchmark_aggregate.json not found: {aggregate_path}", file=sys.stderr)
        return 1

    measurements: dict[tuple[str, int], dict[str, Any]] = {}
    for row in json.loads(aggregate_path.read_text(encoding="utf-8"))["rows"]:
        bits = m10e._bits_of(row)
        if bits is not None:
            measurements[(row["model"], bits)] = row

    runs = {run["name"].replace("@", "_"): run for run in summary["runs"]}
    seeds = summary["seeds"]
    lambda_names, seen = [], set()
    for run in summary["runs"]:
        if run["lambda_name"] not in seen:
            seen.add(run["lambda_name"])
            lambda_names.append(run["lambda_name"])

    def key(lambda_name: str, seed: int) -> str:
        return f"{lambda_name}_s{seed}"

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    def curve(model_key: str, metric: str = "mean_psnr"):
        return [(measurements[(model_key, b)]["aggregate_bpp"],
                 measurements[(model_key, b)][metric])
                for b in BIT_DEPTHS if (model_key, b) in measurements]

    emit("=" * 116)
    emit("M10F - DEPLOYED .nvc BENCHMARK (18,120 steps, DAVIS test, 719 frames, 10 runs)")
    emit("=" * 116)
    emit(f"{'lambda name':<14} {'lambda':>10} {'seed':>5} {'bits':>5} {'BPP':>9} "
         f"{'PSNR dB':>9} {'MS-SSIM':>9} {'bytes/frm':>10} {'ratio':>7}")
    for name in lambda_names:
        for seed in seeds:
            model_key = key(name, seed)
            for bits in BIT_DEPTHS:
                row = measurements.get((model_key, bits))
                if row is None:
                    continue
                emit(f"{name:<14} {runs[model_key]['lambda']:>10.4e} {seed:>5} {bits:>5} "
                     f"{row['aggregate_bpp']:>9.4f} {row['mean_psnr']:>9.3f} "
                     f"{row['mean_msssim']:>9.4f} {row.get('bytes_per_frame', 0):>10.1f} "
                     f"{row.get('compression_ratio', 0):>7.2f}")

    # --- paired BD-rate, per seed -------------------------------------------
    emit()
    emit("=" * 116)
    emit("PAIRED BD-RATE - each run vs the CONTROL OF ITS OWN SEED")
    emit("=" * 116)
    emit(f"{'lambda name':<14} {'lambda':>10} " +
         " ".join(f"{'s' + str(s) + ' PSNR':>12}" for s in seeds) +
         f" {'mean':>9} {'std':>7} {'spread':>7} | " +
         " ".join(f"{'s' + str(s) + ' MSSSIM':>13}" for s in seeds) + f" {'mean':>9} {'spread':>7}")
    bd_rows = []
    for name in lambda_names:
        if name == "CTRL":
            continue
        psnr_values, msssim_values = [], []
        for seed in seeds:
            control, candidate = key("CTRL", seed), key(name, seed)
            if (control, 8) not in measurements or (candidate, 8) not in measurements:
                continue
            psnr_values.append(m10e._bd_rate_linear(curve(control), curve(candidate)))
            msssim_values.append(m10e._bd_rate_linear(
                curve(control, "mean_msssim"), curve(candidate, "mean_msssim")))
        if not psnr_values or any(v is None for v in psnr_values):
            continue
        clean_ms = [v for v in msssim_values if v is not None]
        entry = {
            "lambda_name": name, "lambda": runs[key(name, seeds[0])]["lambda"],
            "bd_rate_psnr_per_seed": dict(zip(map(str, seeds), psnr_values)),
            "bd_rate_psnr_mean": statistics.fmean(psnr_values),
            "bd_rate_psnr_std": statistics.stdev(psnr_values) if len(psnr_values) > 1 else 0.0,
            "bd_rate_psnr_spread": max(psnr_values) - min(psnr_values),
            "bd_rate_msssim_per_seed": dict(zip(map(str, seeds), msssim_values)),
            "bd_rate_msssim_mean": statistics.fmean(clean_ms) if clean_ms else None,
            "bd_rate_msssim_spread": (max(clean_ms) - min(clean_ms)) if len(clean_ms) > 1 else 0.0,
        }
        bd_rows.append(entry)
        emit(f"{name:<14} {entry['lambda']:>10.4e} " +
             " ".join(f"{v:>+11.2f}%" for v in psnr_values) +
             f" {entry['bd_rate_psnr_mean']:>+8.2f}% {entry['bd_rate_psnr_std']:>6.2f} "
             f"{entry['bd_rate_psnr_spread']:>6.2f} | " +
             " ".join(f"{(v if v is not None else float('nan')):>+12.2f}%" for v in msssim_values) +
             f" {(entry['bd_rate_msssim_mean'] or float('nan')):>+8.2f}% "
             f"{entry['bd_rate_msssim_spread']:>6.2f}")
    emit()
    emit("  n=2 seeds per lambda. 'std' over two points is a spread, not a confidence")
    emit("  interval - no significance is claimed from it.")

    # --- deltas vs matched control ------------------------------------------
    emit()
    emit("=" * 116)
    emit("DELTAS vs MATCHED CONTROL (same seed, seed-averaged)")
    emit("=" * 116)
    emit(f"{'lambda name':<14} {'bits':>5} {'dBPP%':>9} {'dPSNR dB':>10} {'dMS-SSIM':>10}")
    delta_rows = []
    for name in lambda_names:
        if name == "CTRL":
            continue
        for bits in BIT_DEPTHS:
            pairs = [(measurements[(key("CTRL", s), bits)], measurements[(key(name, s), bits)])
                     for s in seeds
                     if (key("CTRL", s), bits) in measurements and (key(name, s), bits) in measurements]
            if not pairs:
                continue
            entry = {
                "lambda_name": name, "bits": bits,
                "bpp_change_percent": statistics.fmean(
                    (c["aggregate_bpp"] - b["aggregate_bpp"]) / b["aggregate_bpp"] * 100
                    for b, c in pairs),
                "psnr_delta_db": statistics.fmean(c["mean_psnr"] - b["mean_psnr"] for b, c in pairs),
                "msssim_delta": statistics.fmean(c["mean_msssim"] - b["mean_msssim"] for b, c in pairs),
            }
            delta_rows.append(entry)
            emit(f"{name:<14} {bits:>5} {entry['bpp_change_percent']:>+8.2f}% "
                 f"{entry['psnr_delta_db']:>+10.3f} {entry['msssim_delta']:>+10.4f}")

    # --- noise floor from this experiment's own controls ---------------------
    emit()
    emit("=" * 116)
    emit("M10F NOISE FLOOR - CTRL seed 42 vs CTRL seed 43 (identical but for the seed)")
    emit("=" * 116)
    noise: dict[str, Any] = {}
    floor = None
    if all((key("CTRL", s), 8) in measurements for s in seeds):
        emit(f"{'bits':>5} " + " ".join(f"{'s' + str(s) + ' BPP':>11}" for s in seeds) +
             f" {'dBPP%':>8} " + " ".join(f"{'s' + str(s) + ' PSNR':>12}" for s in seeds) +
             f" {'dPSNR':>8} {'dMS-SSIM':>10}")
        for bits in BIT_DEPTHS:
            a, b = (measurements[(key("CTRL", s), bits)] for s in seeds)
            noise[f"{bits}bit"] = {
                "bpp": [a["aggregate_bpp"], b["aggregate_bpp"]],
                "bpp_change_percent": (b["aggregate_bpp"] - a["aggregate_bpp"])
                / a["aggregate_bpp"] * 100,
                "psnr": [a["mean_psnr"], b["mean_psnr"]],
                "psnr_delta_db": b["mean_psnr"] - a["mean_psnr"],
                "msssim": [a["mean_msssim"], b["mean_msssim"]],
                "msssim_delta": b["mean_msssim"] - a["mean_msssim"],
            }
            e = noise[f"{bits}bit"]
            emit(f"{bits:>5} {a['aggregate_bpp']:>11.4f} {b['aggregate_bpp']:>11.4f} "
                 f"{e['bpp_change_percent']:>+7.2f}% {a['mean_psnr']:>12.3f} "
                 f"{b['mean_psnr']:>12.3f} {e['psnr_delta_db']:>+8.3f} "
                 f"{e['msssim_delta']:>+10.4f}")
        control_bd = m10e._bd_rate_linear(curve(key("CTRL", seeds[0])), curve(key("CTRL", seeds[1])))
        control_bd_ms = m10e._bd_rate_linear(
            curve(key("CTRL", seeds[0]), "mean_msssim"), curve(key("CTRL", seeds[1]), "mean_msssim"))
        noise["control_to_control_bd_rate_psnr"] = control_bd
        noise["control_to_control_bd_rate_msssim"] = control_bd_ms
        noise["floor_magnitude"] = abs(control_bd) if control_bd is not None else None
        floor = abs(control_bd) if control_bd is not None else None
        emit()
        emit(f"  CTRL-vs-CTRL PSNR BD-rate    = {control_bd:+.2f}%  ->  noise floor "
             f"|{abs(control_bd):.2f}| percentage points")
        if control_bd_ms is not None:
            emit(f"  CTRL-vs-CTRL MS-SSIM BD-rate = {control_bd_ms:+.2f}%")
        emit("  This experiment's OWN floor is the primary estimate. M10D (1.41) and")
        emit("  M10E (1.62) are historical references only.")

        emit()
        emit(f"{'lambda name':<14} {'mean BD':>9} {'seed spread':>12} {'vs floor':>10}   classification")
        for entry in bd_rows:
            magnitude = abs(entry["bd_rate_psnr_mean"])
            ratio = magnitude / floor if floor else float("inf")
            classification = (
                "clearly above noise" if magnitude > NOISE_FACTOR * floor else
                "comparable to noise" if magnitude > floor else "below noise"
            )
            entry["vs_noise_floor_ratio"] = ratio
            entry["noise_classification"] = classification
            emit(f"{entry['lambda_name']:<14} {entry['bd_rate_psnr_mean']:>+8.2f}% "
                 f"{entry['bd_rate_psnr_spread']:>11.2f} {ratio:>9.1f}x   {classification}")
        emit()
        emit(f"  Convention: 'clearly above noise' means |mean BD-rate| > {NOISE_FACTOR}x the")
        emit("  measured control-to-control floor. A stated threshold, not a significance")
        emit("  test - n=2 cannot support one.")
        emit()
        emit("  NOTE: this classifies each lambda against the CONTROL. It does NOT say")
        emit("  whether two lambdas differ from each other - see the pairwise table below.")

    # --- lambda -> RD curve ---------------------------------------------------
    emit()
    emit("=" * 116)
    emit("CONVERGED lambda -> RD CURVE (seed-averaged)")
    emit("=" * 116)
    emit(f"{'lambda name':<14} {'lambda':>10} {'proxy R':>9} {'BPP 8':>9} {'BPP 6':>9} "
         f"{'BPP 4':>9} {'BD PSNR':>9} {'BD MSSSIM':>10} {'BD std':>7} {'latent|.|':>10}")
    curve_rows = []
    for name in lambda_names:
        present = [key(name, s) for s in seeds if (key(name, s), 8) in measurements]
        if not present:
            continue
        bd = next((e for e in bd_rows if e["lambda_name"] == name), None)
        entry = {
            "lambda_name": name, "lambda": runs[present[0]]["lambda"],
            "mean_proxy_R": statistics.fmean(runs[k]["final_val_rate_bpp_proxy"] for k in present),
            **{f"mean_bpp_{b}bit": statistics.fmean(
                measurements[(k, b)]["aggregate_bpp"] for k in present) for b in BIT_DEPTHS},
            **{f"mean_psnr_{b}bit": statistics.fmean(
                measurements[(k, b)]["mean_psnr"] for k in present) for b in BIT_DEPTHS},
            **{f"mean_msssim_{b}bit": statistics.fmean(
                measurements[(k, b)]["mean_msssim"] for k in present) for b in BIT_DEPTHS},
            "mean_bd_rate_psnr": bd["bd_rate_psnr_mean"] if bd else None,
            "mean_bd_rate_msssim": bd["bd_rate_msssim_mean"] if bd else None,
            "bd_rate_psnr_std": bd["bd_rate_psnr_std"] if bd else None,
            "mean_latent_abs_mean": statistics.fmean(runs[k]["final_latent_abs_mean"] for k in present),
            "mean_latent_range": statistics.fmean(runs[k]["final_latent_range"] for k in present),
            "mean_bin_width": statistics.fmean(runs[k]["final_bin_width"] for k in present),
            "mean_rate_scale": statistics.fmean(runs[k]["final_rate_scale_mean"] for k in present),
        }
        curve_rows.append(entry)
        emit(f"{name:<14} {entry['lambda']:>10.4e} {entry['mean_proxy_R']:>9.4f} "
             f"{entry['mean_bpp_8bit']:>9.4f} {entry['mean_bpp_6bit']:>9.4f} "
             f"{entry['mean_bpp_4bit']:>9.4f} "
             f"{(entry['mean_bd_rate_psnr'] if entry['mean_bd_rate_psnr'] is not None else 0):>+8.2f}% "
             f"{(entry['mean_bd_rate_msssim'] if entry['mean_bd_rate_msssim'] is not None else 0):>+9.2f}% "
             f"{(entry['bd_rate_psnr_std'] if entry['bd_rate_psnr_std'] is not None else 0):>6.2f} "
             f"{entry['mean_latent_abs_mean']:>10.4f}")

    # --- pairwise lambda separation ------------------------------------------
    rate_aware_curve = [r for r in curve_rows if r["lambda_name"] != "CTRL"]
    pairwise = []
    if floor:
        emit()
        emit("=" * 116)
        emit("PAIRWISE lambda SEPARATION - can these two lambdas be told apart?")
        emit("=" * 116)
        import itertools
        for x, y in itertools.combinations(rate_aware_curve, 2):
            if x["mean_bd_rate_psnr"] is None or y["mean_bd_rate_psnr"] is None:
                continue
            gap = abs(x["mean_bd_rate_psnr"] - y["mean_bd_rate_psnr"])
            verdict = ("DISTINGUISHABLE" if gap > NOISE_FACTOR * floor
                       else "marginal" if gap > floor else "TIED within noise")
            pairwise.append({
                "a": x["lambda_name"], "b": y["lambda_name"],
                "gap_points": gap, "ratio_to_floor": gap / floor, "verdict": verdict,
            })
            emit(f"  {x['lambda_name']:<14} vs {y['lambda_name']:<14} gap={gap:5.2f} pts "
                 f"({gap / floor:4.1f}x floor)  -> {verdict}")

    # --- the boundary question ------------------------------------------------
    boundary = {}
    if floor:
        boundary = classify_boundary(rate_aware_curve, floor)
        emit()
        emit("=" * 116)
        emit("BOUNDARY VERDICT - does the basin keep improving below 3e-4?")
        emit("=" * 116)
        emit(f"  case  : {boundary['case']}")
        emit(f"  reason: {boundary['reason']}")

    # --- best vs final --------------------------------------------------------
    emit()
    emit("=" * 116)
    emit("BEST-vs-FINAL CHECKPOINT GAP (methodology - the FINAL snapshot remains primary)")
    emit("=" * 116)
    emit(f"{'run':<18} {'best ep':>8} {'final ep':>9} {'best obj':>12} {'final obj':>12} "
         f"{'gap %':>8} {'dPSNR':>8}  flag")
    best_final_rows = []
    for run in summary["runs"]:
        bvf = run.get("best_vs_final")
        if not bvf:
            continue
        flagged = bvf["gap_percent"] is not None and bvf["gap_percent"] > BEST_VS_FINAL_FLAG_PERCENT
        best_final_rows.append({"run": run["name"], "lambda": run["lambda"], "seed": run["seed"],
                                "flagged": flagged, **bvf})
        emit(f"{run['name']:<18} {bvf['best_epoch']:>8} {bvf['final_epoch']:>9} "
             f"{bvf['best_val_total_objective']:>12.6e} {bvf['final_val_total_objective']:>12.6e} "
             f"{(bvf['gap_percent'] or 0):>7.2f}% {bvf['psnr_gap_db']:>+8.3f}  "
             f"{'<-- FLAGGED' if flagged else ''}")
    flagged_runs = [r["run"] for r in best_final_rows if r["flagged"]]
    emit()
    if flagged_runs:
        emit(f"  {len(flagged_runs)} run(s) exceed the {BEST_VS_FINAL_FLAG_PERCENT}% gap threshold: "
             f"{', '.join(flagged_runs)}")
        emit("  These are reported as a methodology caveat. The predefined final-snapshot")
        emit("  result is RETAINED - it is not replaced by best.pt after the fact.")
    else:
        emit(f"  No run exceeds the {BEST_VS_FINAL_FLAG_PERCENT}% gap threshold; the final-snapshot")
        emit("  convention is not distorting any arm in this experiment.")

    # --- proxy vs actual: four separate questions -----------------------------
    emit()
    emit("=" * 116)
    emit("PROXY vs ACTUAL - four questions, kept apart")
    emit("=" * 116)
    rated = [k for name in lambda_names if name != "CTRL"
             for k in (key(name, s) for s in seeds) if (k, 8) in measurements]
    proxy_values = [runs[k]["final_val_rate_bpp_proxy"] for k in rated]

    lambda_vs_proxy = sorted(rate_aware_curve, key=lambda r: r["lambda"])
    proxy_monotonic = all(
        lambda_vs_proxy[i]["mean_proxy_R"] >= lambda_vs_proxy[i + 1]["mean_proxy_R"]
        for i in range(len(lambda_vs_proxy) - 1)
    )
    emit("1. Does LOWER lambda produce LOWER proxy R?")
    emit(f"   lambda ascending -> proxy R: " +
         ", ".join(f"{r['lambda']:.1e}:{r['mean_proxy_R']:.4f}" for r in lambda_vs_proxy))
    emit(f"   proxy R decreases as lambda increases: {proxy_monotonic}")
    emit(f"   -> lower lambda gives HIGHER proxy R (weaker rate pressure): "
         f"{'as expected' if proxy_monotonic else 'NOT monotonic - investigate'}")

    emit()
    emit("2. Does proxy R order actual .nvc bitrate?")
    per_depth = {}
    for bits in BIT_DEPTHS:
        actual = [measurements[(k, bits)]["aggregate_bpp"] for k in rated]
        proxy_rank = sorted(rated, key=lambda k: runs[k]["final_val_rate_bpp_proxy"])
        actual_rank = sorted(rated, key=lambda k: measurements[(k, bits)]["aggregate_bpp"])
        # Cross-lambda ordering, which is the question that actually matters:
        # within-lambda seed pairs are tied to within noise and their order is
        # not meaningful.
        by_lambda = sorted(rate_aware_curve, key=lambda r: r["mean_proxy_R"])
        cross_monotonic = all(
            by_lambda[i][f"mean_bpp_{bits}bit"] <= by_lambda[i + 1][f"mean_bpp_{bits}bit"]
            for i in range(len(by_lambda) - 1)
        )
        per_depth[f"{bits}bit"] = {
            "rank_agreement_per_run": proxy_rank == actual_rank,
            "cross_lambda_monotonic": cross_monotonic,
            "spearman": m10e._spearman(proxy_values, actual),
            "pearson": m10e._pearson(proxy_values, actual),
        }
        e = per_depth[f"{bits}bit"]
        emit(f"   {bits}-bit: cross-lambda monotonic={e['cross_lambda_monotonic']}  "
             f"per-run agreement={e['rank_agreement_per_run']}  "
             f"spearman={(e['spearman'] if e['spearman'] is not None else float('nan')):+.3f}  "
             f"pearson={(e['pearson'] if e['pearson'] is not None else float('nan')):+.3f}")

    best_psnr = min((r for r in bd_rows), key=lambda r: r["bd_rate_psnr_mean"], default=None)
    best_ms = min((r for r in bd_rows if r["bd_rate_msssim_mean"] is not None),
                  key=lambda r: r["bd_rate_msssim_mean"], default=None)
    lowest_proxy = min(rate_aware_curve, key=lambda r: r["mean_proxy_R"], default=None)
    emit()
    emit("3. Does the lambda minimising proxy R minimise PSNR BD-rate?")
    emit(f"   lowest proxy R      : {lowest_proxy['lambda_name']} (lambda {lowest_proxy['lambda']:.4e})")
    emit(f"   best PSNR BD-rate   : {best_psnr['lambda_name']} (lambda {best_psnr['lambda']:.4e}, "
         f"{best_psnr['bd_rate_psnr_mean']:+.2f}%)")
    emit(f"   same lambda? {lowest_proxy['lambda_name'] == best_psnr['lambda_name']}")
    emit()
    emit("4. Does it minimise MS-SSIM BD-rate?")
    if best_ms is not None:
        emit(f"   best MS-SSIM BD-rate: {best_ms['lambda_name']} (lambda {best_ms['lambda']:.4e}, "
             f"{best_ms['bd_rate_msssim_mean']:+.2f}%)")
        emit(f"   same lambda? {lowest_proxy['lambda_name'] == best_ms['lambda_name']}")
    emit()
    emit("   The proxy estimates RATE. Questions 3 and 4 are about the rate/QUALITY")
    emit("   tradeoff, which the proxy does not represent - it must not pick lambda alone.")

    # --- M10E bridge ----------------------------------------------------------
    emit()
    emit("=" * 116)
    emit("M10E -> M10F BRIDGE - the repeated 3e-4 and 4.5e-4 arms")
    emit("=" * 116)
    bridge: dict[str, Any] = {}
    m10e_aggregate = args.m10e_dir / "benchmark_aggregate.json"
    m10e_analysis = args.m10e_dir / "rd_analysis.json"
    if m10e_aggregate.is_file():
        m10e_measure: dict[tuple[str, int], dict[str, Any]] = {}
        for row in json.loads(m10e_aggregate.read_text(encoding="utf-8"))["rows"]:
            bits = m10e._bits_of(row)
            if bits is not None:
                m10e_measure[(row["model"], bits)] = row
        # M10E's names for the same lambdas: LOWER = 3e-4, MID_LOW = 4.5e-4.
        pairs = [("BRIDGE", "LOWER", 3.0e-4), ("UPPER_ANCHOR", "MID_LOW", 4.5e-4)]
        emit(f"{'lambda':>10} {'seed':>5} {'bits':>5} {'M10E BPP':>10} {'M10F BPP':>10} "
             f"{'dBPP%':>8} {'M10E PSNR':>10} {'M10F PSNR':>10} {'dPSNR':>8} {'dMS-SSIM':>10}")
        for m10f_name, m10e_name, lam in pairs:
            for seed in seeds:
                fkey, ekey = key(m10f_name, seed), f"{m10e_name}_s{seed}"
                for bits in BIT_DEPTHS:
                    if (fkey, bits) not in measurements or (ekey, bits) not in m10e_measure:
                        continue
                    f_row, e_row = measurements[(fkey, bits)], m10e_measure[(ekey, bits)]
                    record = {
                        "lambda": lam, "seed": seed, "bits": bits,
                        "m10e_bpp": e_row["aggregate_bpp"], "m10f_bpp": f_row["aggregate_bpp"],
                        "bpp_change_percent": (f_row["aggregate_bpp"] - e_row["aggregate_bpp"])
                        / e_row["aggregate_bpp"] * 100,
                        "m10e_psnr": e_row["mean_psnr"], "m10f_psnr": f_row["mean_psnr"],
                        "psnr_delta_db": f_row["mean_psnr"] - e_row["mean_psnr"],
                        "msssim_delta": f_row["mean_msssim"] - e_row["mean_msssim"],
                    }
                    bridge.setdefault("per_measurement", []).append(record)
                    emit(f"{lam:>10.1e} {seed:>5} {bits:>5} {e_row['aggregate_bpp']:>10.4f} "
                         f"{f_row['aggregate_bpp']:>10.4f} {record['bpp_change_percent']:>+7.2f}% "
                         f"{e_row['mean_psnr']:>10.3f} {f_row['mean_psnr']:>10.3f} "
                         f"{record['psnr_delta_db']:>+8.3f} {record['msssim_delta']:>+10.4f}")
        if m10e_analysis.is_file():
            m10e_bd = {r["lambda_name"]: r for r in
                       json.loads(m10e_analysis.read_text(encoding="utf-8"))["bd_rate_paired"]}
            emit()
            emit(f"{'lambda':>10} {'M10E BD':>10} {'M10F BD':>10} {'difference':>12}")
            for m10f_name, m10e_name, lam in pairs:
                f_entry = next((e for e in bd_rows if e["lambda_name"] == m10f_name), None)
                e_entry = m10e_bd.get(m10e_name)
                if f_entry is None or e_entry is None:
                    continue
                diff = f_entry["bd_rate_psnr_mean"] - e_entry["bd_rate_psnr_mean"]
                bridge.setdefault("bd_rate", []).append({
                    "lambda": lam, "m10e_bd_rate": e_entry["bd_rate_psnr_mean"],
                    "m10f_bd_rate": f_entry["bd_rate_psnr_mean"], "difference_points": diff,
                })
                emit(f"{lam:>10.1e} {e_entry['bd_rate_psnr_mean']:>+9.2f}% "
                     f"{f_entry['bd_rate_psnr_mean']:>+9.2f}% {diff:>+11.2f} pts")
            if floor:
                emit()
                emit(f"  Inter-experiment drift should be read against the {floor:.2f}-point")
                emit("  noise floor. Differences inside it are not evidence of anything.")
    else:
        emit("  [skip] M10E benchmark_aggregate.json not found; bridge analysis unavailable.")

    # --- Pareto ----------------------------------------------------------------
    emit()
    emit("=" * 116)
    emit("PARETO FRONTIER (seed-averaged points)")
    emit("=" * 116)
    pareto = {}
    for bits in BIT_DEPTHS:
        points = [(r["lambda_name"], r[f"mean_bpp_{bits}bit"], r[f"mean_psnr_{bits}bit"])
                  for r in curve_rows]
        nondominated = [
            name for name, bpp, psnr in points
            if not any(o_bpp <= bpp and o_psnr >= psnr and (o_bpp, o_psnr) != (bpp, psnr)
                       for _, o_bpp, o_psnr in points)
        ]
        pareto[f"{bits}bit"] = nondominated
        emit(f"  {bits}-bit non-dominated: {', '.join(sorted(nondominated))}")
    emit()
    emit("  A lambda sweep necessarily spreads points along a curve, so dominance")
    emit("  counting is weak evidence here - BD-rate is the discriminating metric.")

    analysis = {
        "phase": "M10F (lower lambda boundary)",
        "question": "Does the RD basin continue to improve below lambda = 3e-4?",
        "split": args.split, "seeds": seeds,
        "noise_factor_convention": NOISE_FACTOR,
        "best_vs_final_flag_percent": BEST_VS_FINAL_FLAG_PERCENT,
        "evaluation_convention": summary.get("evaluation_convention"),
        "note": (
            "aggregate_bpp is MEASURED deployed .nvc payload. The differentiable proxy is "
            "never substituted for it."
        ),
        "measurements": [
            {"model": model_key, "bits": bits, "lambda": runs[model_key]["lambda"],
             "lambda_name": runs[model_key]["lambda_name"], "seed": runs[model_key]["seed"],
             **{k: row[k] for k in ("aggregate_bpp", "mean_psnr", "mean_msssim",
                                    "compression_ratio", "total_bytes", "total_frames")
                if k in row}}
            for (model_key, bits), row in sorted(measurements.items())
        ],
        "bd_rate_paired": bd_rows,
        "deltas_vs_control": delta_rows,
        "noise_floor": noise,
        "pairwise_lambda_separation": pairwise,
        "boundary_verdict": boundary,
        "lambda_rd_curve": curve_rows,
        "best_vs_final": best_final_rows,
        "proxy_vs_actual": {
            "Q1_lower_lambda_gives_lower_proxy": {
                "proxy_decreases_with_lambda": proxy_monotonic,
                "per_lambda": [{"lambda": r["lambda"], "mean_proxy_R": r["mean_proxy_R"]}
                               for r in lambda_vs_proxy],
            },
            "Q2_proxy_orders_bitrate": per_depth,
            "Q3_proxy_minimiser_minimises_psnr_bd": {
                "lowest_proxy_lambda": lowest_proxy["lambda_name"] if lowest_proxy else None,
                "best_psnr_bd_lambda": best_psnr["lambda_name"] if best_psnr else None,
                "same": bool(lowest_proxy and best_psnr
                             and lowest_proxy["lambda_name"] == best_psnr["lambda_name"]),
            },
            "Q4_proxy_minimiser_minimises_msssim_bd": {
                "best_msssim_bd_lambda": best_ms["lambda_name"] if best_ms else None,
                "same": bool(lowest_proxy and best_ms
                             and lowest_proxy["lambda_name"] == best_ms["lambda_name"]),
            },
        },
        "m10e_bridge": bridge,
        "pareto_frontier": pareto,
    }
    (args.output_dir / "rd_analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8")
    (args.output_dir / "rd_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nAnalysis: {args.output_dir / 'rd_analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
