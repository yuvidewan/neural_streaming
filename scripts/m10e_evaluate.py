"""M10E evaluation: fresh calibration, deployed benchmark, and the final lambda decision.

Turns the 10-run (lambda x seed) grid into a defensible operating-point choice,
using the deployed `.nvc` codec as the authority throughout.

TWO THINGS THIS DOES THAT EARLIER MILESTONES COULD NOT
-------------------------------------------------------
1. **Paired BD-rate.** Every rate-aware run is scored against the CONTROL OF
   ITS OWN SEED, so seed-driven differences in the control cancel instead of
   leaking into the lambda comparison. Mean and spread across the two seeds are
   both reported; the spread is never hidden.

2. **A noise floor measured inside this experiment.** CTRL seed 42 vs CTRL
   seed 43 are configuration-identical apart from the seed, so their BD-rate
   difference IS the run-to-run noise. Every candidate's improvement is then
   classified against that floor rather than an arbitrary p-value.

TWO QUESTIONS ABOUT THE PROXY, KEPT SEPARATE
---------------------------------------------
    A. does proxy R order actual .nvc bitrate correctly?
    B. does the lambda that minimises proxy R also minimise BD-rate?

These are different, and conflating them is how a rate proxy gets mistaken for
an RD criterion. M10D already showed A can hold while B fails - the proxy
ranked the highest lambda cheapest, and that lambda had the worst BD-rate.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\m10e_evaluate.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from nvc.utils.config import load_default_config

DEFAULT_OUTPUT_DIR = Path("outputs/m10e_lambda_lock")
M10D_DIR = Path("outputs/m10d_lambda_refinement")
BIT_DEPTHS = (8, 6, 4)
# A candidate is "clearly above noise" only if its improvement exceeds the
# measured control-to-control spread by this factor. 2x is a deliberate,
# stated convention - not a significance test, which n=2 cannot support.
NOISE_FACTOR = 2.0


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def final_models(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """One benchmark model per run: its final 18,120-step snapshot."""
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
        description="M10E: calibrate, benchmark and select the final lambda.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--m10d-dir", type=Path, default=M10D_DIR)
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


def _bits_of(row: dict[str, Any]) -> int | None:
    configuration = str(row.get("codec_configuration", ""))
    for bits in BIT_DEPTHS:
        if f"{bits}bit" in configuration:
            return bits
    return None


def _bd_rate_linear(base, test):
    """Piecewise-linear BD-rate - the conservative methodology used since M9."""
    import numpy as np

    base_psnr = np.array([p[1] for p in base]); base_rate = np.log10([p[0] for p in base])
    test_psnr = np.array([p[1] for p in test]); test_rate = np.log10([p[0] for p in test])
    order = np.argsort(base_psnr); base_psnr, base_rate = base_psnr[order], base_rate[order]
    order = np.argsort(test_psnr); test_psnr, test_rate = test_psnr[order], test_rate[order]
    low = max(base_psnr.min(), test_psnr.min()); high = min(base_psnr.max(), test_psnr.max())
    if high - low <= 1e-9:
        return None
    grid = np.linspace(low, high, 2000)
    return float((10 ** float(np.mean(
        np.interp(grid, test_psnr, test_rate) - np.interp(grid, base_psnr, base_rate)
    )) - 1) * 100)


def _pearson(a, b):
    n = len(a)
    if n < 3:
        return None
    mean_a, mean_b = sum(a) / n, sum(b) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    den = (sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)) ** 0.5
    return num / den if den > 0 else None


def _spearman(a, b):
    n = len(a)
    if n < 3:
        return None

    def ranks(values):
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    return _pearson(ranks(a), ranks(b))


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
              "scripts/m10e_lambda_lock.py first.", file=sys.stderr)
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    m9 = _load_script("m9_final_calibrate_benchmark")
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
                    "validation or test frames, no sharing between runs, nothing reused from "
                    "another milestone."
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

    if args.stage in ("analyse", "all"):
        aggregate_path = args.output_dir / "benchmark_aggregate.json"
        if not aggregate_path.is_file():
            print(f"[ERROR] benchmark_aggregate.json not found: {aggregate_path}", file=sys.stderr)
            return 1

        measurements: dict[tuple[str, int], dict[str, Any]] = {}
        for row in json.loads(aggregate_path.read_text(encoding="utf-8"))["rows"]:
            bits = _bits_of(row)
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

        emit("=" * 110)
        emit("M10E - DEPLOYED .nvc BENCHMARK (18,120 steps, DAVIS test, 719 frames, 10 runs)")
        emit("=" * 110)
        emit(f"{'lambda name':<16} {'lambda':>10} {'seed':>5} {'bits':>5} {'BPP':>9} "
             f"{'PSNR dB':>9} {'MS-SSIM':>9} {'bytes/frm':>10} {'ratio':>7}")
        for name in lambda_names:
            for seed in seeds:
                model_key = key(name, seed)
                for bits in BIT_DEPTHS:
                    row = measurements.get((model_key, bits))
                    if row is None:
                        continue
                    run = runs[model_key]
                    emit(f"{name:<16} {run['lambda']:>10.4e} {seed:>5} {bits:>5} "
                         f"{row['aggregate_bpp']:>9.4f} {row['mean_psnr']:>9.3f} "
                         f"{row['mean_msssim']:>9.4f} {row.get('bytes_per_frame', 0):>10.1f} "
                         f"{row.get('compression_ratio', 0):>7.2f}")

        # --- paired BD-rate, per seed ---------------------------------------
        emit()
        emit("=" * 110)
        emit("PAIRED BD-RATE - each run vs the CONTROL OF ITS OWN SEED")
        emit("=" * 110)
        emit(f"{'lambda name':<16} {'lambda':>10} " +
             " ".join(f"{'s' + str(s) + ' PSNR':>12}" for s in seeds) +
             f" {'mean':>9} {'std':>8} {'spread':>8} | " +
             " ".join(f"{'s' + str(s) + ' MSSSIM':>13}" for s in seeds) + f" {'mean':>9}")
        bd_rows = []
        for name in lambda_names:
            if name == "CTRL":
                continue
            psnr_values, msssim_values = [], []
            for seed in seeds:
                control, candidate = key("CTRL", seed), key(name, seed)
                if (control, 8) not in measurements or (candidate, 8) not in measurements:
                    continue
                psnr_values.append(_bd_rate_linear(curve(control), curve(candidate)))
                msssim_values.append(_bd_rate_linear(
                    curve(control, "mean_msssim"), curve(candidate, "mean_msssim")))
            if not psnr_values or any(v is None for v in psnr_values):
                continue
            entry = {
                "lambda_name": name, "lambda": runs[key(name, seeds[0])]["lambda"],
                "bd_rate_psnr_per_seed": dict(zip(map(str, seeds), psnr_values)),
                "bd_rate_psnr_mean": statistics.fmean(psnr_values),
                "bd_rate_psnr_std": statistics.stdev(psnr_values) if len(psnr_values) > 1 else 0.0,
                "bd_rate_psnr_spread": max(psnr_values) - min(psnr_values),
                "bd_rate_msssim_per_seed": dict(zip(map(str, seeds), msssim_values)),
                "bd_rate_msssim_mean": (
                    statistics.fmean([v for v in msssim_values if v is not None])
                    if any(v is not None for v in msssim_values) else None
                ),
            }
            bd_rows.append(entry)
            emit(f"{name:<16} {entry['lambda']:>10.4e} " +
                 " ".join(f"{v:>+11.2f}%" for v in psnr_values) +
                 f" {entry['bd_rate_psnr_mean']:>+8.2f}% {entry['bd_rate_psnr_std']:>7.2f} "
                 f"{entry['bd_rate_psnr_spread']:>7.2f} | " +
                 " ".join(f"{(v if v is not None else float('nan')):>+12.2f}%" for v in msssim_values) +
                 f" {(entry['bd_rate_msssim_mean'] or float('nan')):>+8.2f}%")
        emit()
        emit(f"  n=2 seeds per lambda. 'std' over two points is a spread, not a confidence")
        emit("  interval - no significance is claimed from it.")

        # --- noise floor from the two controls -------------------------------
        emit()
        emit("=" * 110)
        emit("EMPIRICAL NOISE FLOOR - CTRL seed 42 vs CTRL seed 43 (identical but for the seed)")
        emit("=" * 110)
        noise: dict[str, Any] = {}
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
                    "msssim_delta": b["mean_msssim"] - a["mean_msssim"],
                }
                entry = noise[f"{bits}bit"]
                emit(f"{bits:>5} {a['aggregate_bpp']:>11.4f} {b['aggregate_bpp']:>11.4f} "
                     f"{entry['bpp_change_percent']:>+7.2f}% {a['mean_psnr']:>12.3f} "
                     f"{b['mean_psnr']:>12.3f} {entry['psnr_delta_db']:>+8.3f} "
                     f"{entry['msssim_delta']:>+10.4f}")
            control_bd = _bd_rate_linear(curve(key("CTRL", seeds[0])), curve(key("CTRL", seeds[1])))
            noise["control_to_control_bd_rate"] = control_bd
            noise["floor_magnitude"] = abs(control_bd) if control_bd is not None else None
            emit()
            emit(f"  CTRL-vs-CTRL BD-rate = {control_bd:+.2f}%  ->  noise floor "
                 f"|{abs(control_bd):.2f}| percentage points")
            emit(f"  (M10D's independent estimate was 1.41 points.)")

            floor = abs(control_bd)
            emit()
            emit(f"{'lambda name':<16} {'mean BD':>9} {'seed spread':>12} {'vs floor':>10}   classification")
            for entry in bd_rows:
                magnitude = abs(entry["bd_rate_psnr_mean"])
                ratio = magnitude / floor if floor > 0 else float("inf")
                classification = (
                    "clearly above noise" if magnitude > NOISE_FACTOR * floor else
                    "comparable to noise" if magnitude > floor else "below noise"
                )
                entry["vs_noise_floor_ratio"] = ratio
                entry["noise_classification"] = classification
                emit(f"{entry['lambda_name']:<16} {entry['bd_rate_psnr_mean']:>+8.2f}% "
                     f"{entry['bd_rate_psnr_spread']:>11.2f} {ratio:>9.1f}x   {classification}")
            emit()
            emit(f"  Convention: 'clearly above noise' means |mean BD-rate| > {NOISE_FACTOR}x the")
            emit("  measured control-to-control floor. This is a stated threshold, not a")
            emit("  significance test - n=2 cannot support one.")

        # --- lambda -> RD curve ----------------------------------------------
        emit()
        emit("=" * 110)
        emit("CONVERGED lambda -> RD CURVE (seed-averaged)")
        emit("=" * 110)
        emit(f"{'lambda name':<16} {'lambda':>10} {'proxy R':>9} {'BPP 8':>9} {'BPP 6':>9} "
             f"{'BPP 4':>9} {'BD PSNR':>9} {'BD MSSSIM':>10} {'BD std':>8}")
        curve_rows = []
        for name in lambda_names:
            present = [key(name, s) for s in seeds if (key(name, s), 8) in measurements]
            if not present:
                continue
            bd = next((e for e in bd_rows if e["lambda_name"] == name), None)
            entry = {
                "lambda_name": name, "lambda": runs[present[0]]["lambda"],
                "mean_proxy_R": statistics.fmean(
                    runs[k]["final_val_rate_bpp_proxy"] for k in present),
                **{f"mean_bpp_{b}bit": statistics.fmean(
                    measurements[(k, b)]["aggregate_bpp"] for k in present) for b in BIT_DEPTHS},
                **{f"mean_psnr_{b}bit": statistics.fmean(
                    measurements[(k, b)]["mean_psnr"] for k in present) for b in BIT_DEPTHS},
                "mean_bd_rate_psnr": bd["bd_rate_psnr_mean"] if bd else None,
                "mean_bd_rate_msssim": bd["bd_rate_msssim_mean"] if bd else None,
                "bd_rate_psnr_std": bd["bd_rate_psnr_std"] if bd else None,
                "mean_latent_abs_mean": statistics.fmean(
                    runs[k]["final_latent_abs_mean"] for k in present),
                "mean_bin_width": statistics.fmean(runs[k]["final_bin_width"] for k in present),
            }
            curve_rows.append(entry)
            emit(f"{name:<16} {entry['lambda']:>10.4e} {entry['mean_proxy_R']:>9.4f} "
                 f"{entry['mean_bpp_8bit']:>9.4f} {entry['mean_bpp_6bit']:>9.4f} "
                 f"{entry['mean_bpp_4bit']:>9.4f} "
                 f"{(entry['mean_bd_rate_psnr'] if entry['mean_bd_rate_psnr'] is not None else 0):>+8.2f}% "
                 f"{(entry['mean_bd_rate_msssim'] if entry['mean_bd_rate_msssim'] is not None else 0):>+9.2f}% "
                 f"{(entry['bd_rate_psnr_std'] if entry['bd_rate_psnr_std'] is not None else 0):>7.2f}")

        # --- proxy vs actual: the two separate questions ----------------------
        emit()
        emit("=" * 110)
        emit("PROXY vs ACTUAL - two questions, kept apart")
        emit("=" * 110)
        rated = [k for name in lambda_names if name != "CTRL"
                 for k in (key(name, s) for s in seeds) if (k, 8) in measurements]
        proxy_values = [runs[k]["final_val_rate_bpp_proxy"] for k in rated]
        emit("A. Does proxy R order actual .nvc bitrate?")
        per_depth = {}
        for bits in BIT_DEPTHS:
            actual = [measurements[(k, bits)]["aggregate_bpp"] for k in rated]
            proxy_rank = sorted(rated, key=lambda k: runs[k]["final_val_rate_bpp_proxy"])
            actual_rank = sorted(rated, key=lambda k: measurements[(k, bits)]["aggregate_bpp"])
            per_depth[f"{bits}bit"] = {
                "rank_agreement": proxy_rank == actual_rank,
                "spearman": _spearman(proxy_values, actual),
                "pearson": _pearson(proxy_values, actual),
            }
            block = per_depth[f"{bits}bit"]
            emit(f"   {bits}-bit: agreement={block['rank_agreement']}  "
                 f"spearman={block['spearman']:+.3f}  pearson={block['pearson']:+.3f}")
        emit()
        emit("B. Does the lambda minimising proxy R also minimise BD-rate?")
        by_proxy = min(curve_rows, key=lambda e: e["mean_proxy_R"])
        rate_aware = [e for e in curve_rows if e["mean_bd_rate_psnr"] is not None]
        by_bd = min(rate_aware, key=lambda e: e["mean_bd_rate_psnr"]) if rate_aware else None
        agree = by_bd is not None and by_proxy["lambda_name"] == by_bd["lambda_name"]
        emit(f"   lowest mean proxy R : {by_proxy['lambda_name']} (lambda {by_proxy['lambda']:.4e})")
        if by_bd is not None:
            emit(f"   best mean BD-rate   : {by_bd['lambda_name']} (lambda {by_bd['lambda']:.4e}, "
                 f"{by_bd['mean_bd_rate_psnr']:+.2f}%)")
        emit(f"   same lambda? {agree}")
        emit("   -> the proxy estimates RATE. It does not by itself identify the best RD")
        emit("      operating point, and must not be used alone to choose lambda.")

        # --- M10D cross-check -------------------------------------------------
        cross_check: dict[str, Any] = {}
        m10d_aggregate = args.m10d_dir / "benchmark_aggregate.json"
        if m10d_aggregate.is_file() and (key("CURRENT_BEST", 42), 8) in measurements:
            m10d = {}
            for row in json.loads(m10d_aggregate.read_text(encoding="utf-8"))["rows"]:
                bits = _bits_of(row)
                if bits is not None and row["model"] == "LOW":  # M10D's lambda=6.0e-4, seed 42
                    m10d[bits] = row
            if m10d:
                emit()
                emit("=" * 110)
                emit("M10D CROSS-CHECK - lambda=6.0e-4 seed 42, M10D vs M10E (reproducibility)")
                emit("=" * 110)
                emit(f"{'bits':>5} {'M10D BPP':>10} {'M10E BPP':>10} {'dBPP%':>8} "
                     f"{'M10D PSNR':>11} {'M10E PSNR':>11} {'dPSNR':>8}")
                for bits in BIT_DEPTHS:
                    a = m10d[bits]
                    b = measurements[(key("CURRENT_BEST", 42), bits)]
                    entry = {
                        "m10d_bpp": a["aggregate_bpp"], "m10e_bpp": b["aggregate_bpp"],
                        "bpp_change_percent": (b["aggregate_bpp"] - a["aggregate_bpp"])
                        / a["aggregate_bpp"] * 100,
                        "m10d_psnr": a["mean_psnr"], "m10e_psnr": b["mean_psnr"],
                        "psnr_delta_db": b["mean_psnr"] - a["mean_psnr"],
                    }
                    cross_check[f"{bits}bit"] = entry
                    emit(f"{bits:>5} {entry['m10d_bpp']:>10.4f} {entry['m10e_bpp']:>10.4f} "
                         f"{entry['bpp_change_percent']:>+7.2f}% {entry['m10d_psnr']:>11.3f} "
                         f"{entry['m10e_psnr']:>11.3f} {entry['psnr_delta_db']:>+8.3f}")

        # --- Pareto ------------------------------------------------------------
        emit()
        emit("=" * 110)
        emit("PARETO FRONTIER (seed-averaged points)")
        emit("=" * 110)
        pareto: dict[str, list[str]] = {}
        for bits in BIT_DEPTHS:
            points = {e["lambda_name"]: (e[f"mean_bpp_{bits}bit"], e[f"mean_psnr_{bits}bit"])
                      for e in curve_rows}
            frontier = [
                name for name, (bpp, quality) in points.items()
                if not any(
                    other != name and other_bpp <= bpp and other_quality >= quality
                    and (other_bpp < bpp or other_quality > quality)
                    for other, (other_bpp, other_quality) in points.items()
                )
            ]
            pareto[f"{bits}bit"] = sorted(frontier)
            emit(f"  {bits}-bit non-dominated: {', '.join(sorted(frontier))}")

        report = {
            "phase": "M10E evaluation (final lambda selection)",
            "split": args.split, "seeds": seeds,
            "noise_factor_convention": NOISE_FACTOR,
            "note": (
                "aggregate_bpp is MEASURED .nvc payload; final_val_rate_bpp_proxy is the "
                "training-time Laplace estimate. BD-rate is paired per seed against that "
                "seed's own control. n=2 - spreads are reported, significance is not claimed."
            ),
            "measurements": [
                {"model": k[0], "bits": k[1],
                 "lambda": runs[k[0]]["lambda"], "seed": runs[k[0]]["seed"],
                 **{kk: vv for kk, vv in measurements[k].items()
                    if not isinstance(vv, (dict, list))}}
                for k in sorted(measurements, key=lambda k: (k[0], -k[1]))
            ],
            "bd_rate_paired": bd_rows,
            "noise_floor": noise,
            "lambda_rd_curve": curve_rows,
            "proxy_vs_actual": {
                "A_orders_bitrate": per_depth,
                "B_minimiser_agrees": {
                    "lowest_proxy_lambda": by_proxy["lambda_name"],
                    "best_bd_rate_lambda": by_bd["lambda_name"] if by_bd else None,
                    "same": agree,
                },
            },
            "m10d_cross_check": cross_check,
            "pareto_frontier": pareto,
        }
        (args.output_dir / "rd_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (args.output_dir / "rd_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        with (args.output_dir / "rd_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["model", "lambda", "seed", "bits", "aggregate_bpp", "mean_psnr",
                            "mean_msssim", "bytes_per_frame", "compression_ratio", "total_bytes",
                            "total_frames", "encode_seconds_per_frame", "decode_seconds_per_frame"],
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in report["measurements"]:
                writer.writerow(row)
        emit()
        emit(f"Analysis: {args.output_dir / 'rd_analysis.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
