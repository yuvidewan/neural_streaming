"""M22 Phases 5/14/17 + the gate decision - a pure analysis pass over the JSON
the earlier phases wrote. Runs no model and reads no data, so the verdict is
reproducible from the recorded artifacts alone.

Produces:
  * the VAL-B rate gate, with the pre-declared distortion guard applied;
  * the STALE-vs-REFIT split, which prices the coupling the residual quantizer
    has with the entropy stack;
  * the GOP-position analysis (position 1 against positions 2-9);
  * the candidate-selection rule, applied mechanically;
  * the classification, graded on held-out DAVIS TEST once Phase 19 has run.

Run:
  ./.venv/Scripts/python.exe scripts/m22_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

WEAK_BELOW_PERCENT = 0.5
MEANINGFUL_ABOVE_PERCENT = 1.0
INDISTINGUISHABLE_PERCENT = 0.05
MAX_PSNR_REGRESSION_DB = 0.10
MAX_MSSSIM_REGRESSION = 0.0010


def _bd_rate(base_curve, test_curve):
    """The project's settled piecewise-linear BD-rate, reused rather than
    reimplemented (`m10b_evaluate._bd_rate_linear`, the conservative methodology
    M10A chose over a polynomial fit).

    M22 needs this because changing a quantizer grid moves rate AND distortion
    together: a coarser grid buys bytes by coding at a lower quality point, which
    raw byte deltas at a fixed nominal bit depth cannot distinguish from a real
    representation improvement. Negative means the candidate needs FEWER bits for
    the same quality; positive means more.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m10b_evaluate", Path(__file__).parent / "m10b_evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._bd_rate_linear(base_curve, test_curve)


def rate_distortion_curves(sweep: dict[str, Any]) -> dict[str, list[tuple[float, float, float]]]:
    """(bpp, PSNR, MS-SSIM) per variant/arm across every rate point measured, so
    a candidate that wins bytes while losing quality can be priced honestly
    instead of being read as a free win."""
    curves: dict[str, list[tuple[float, float, float]]] = {}
    for point in sweep["rate_points"]:
        for row in point["candidates"]:
            aggregate = row["aggregate"]
            curves.setdefault(row["label"], []).append(
                (aggregate["stream_bpp"], aggregate["mean_psnr_db"], aggregate["mean_msssim"]))
    return {label: sorted(points) for label, points in curves.items()}


def bd_rate_table(sweep: dict[str, Any], *, control: str = "deployed/stale"
                  ) -> list[dict[str, Any]]:
    """BD-rate of every arm against the deployed control, where enough rate
    points overlap. Returns `None` for a curve with too few points or no PSNR
    overlap - undefined, which is different from zero."""
    curves = rate_distortion_curves(sweep)
    base = curves.get(control)
    if base is None or len(base) < 2:
        return []
    rows = []
    for label, points in curves.items():
        if label == control or len(points) < 2:
            rows.append({"label": label, "points": len(points),
                         "bd_rate_psnr": None, "bd_rate_msssim": None})
            continue
        rows.append({
            "label": label, "points": len(points),
            "bd_rate_psnr": _bd_rate([(p[0], p[1]) for p in base],
                                     [(p[0], p[1]) for p in points]),
            "bd_rate_msssim": _bd_rate([(p[0], p[2]) for p in base],
                                       [(p[0], p[2]) for p in points]),
        })
    return rows


def verdict(gain: float) -> str:
    return ("meaningful" if gain >= MEANINGFUL_ABOVE_PERCENT else
            "marginal" if gain >= WEAK_BELOW_PERCENT else "weak")


def distortion_regression(delta_psnr_db: float, delta_msssim: float) -> str | None:
    reasons = []
    if delta_psnr_db < -MAX_PSNR_REGRESSION_DB:
        reasons.append(f"PSNR {delta_psnr_db:+.4f} dB")
    if delta_msssim < -MAX_MSSSIM_REGRESSION:
        reasons.append(f"MS-SSIM {delta_msssim:+.6f}")
    return "; ".join(reasons) if reasons else None


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Phase 17's pre-declared rule, applied mechanically:

      1. actual coded TOTAL-STREAM bytes on VAL-B;
      2. subject to decoder compatibility AND the declared distortion guard;
      3. then residual-byte improvement;
      4. then simplicity/runtime.

    Never by PSNR, MS-SSIM, proxy rate, training loss or oracle-gap closure
    alone - quality enters only as a disqualifier, never as a tiebreaker.
    """
    eligible = [r for r in rows
                if r.get("decoder_compatible")
                and not r.get("is_control")
                and distortion_regression(r.get("delta_psnr_db", 0.0),
                                          r.get("delta_msssim", 0.0)) is None]
    if not eligible:
        return None
    best = max(r["total_stream_gain_percent"] for r in eligible)
    tied = [r for r in eligible
            if best - r["total_stream_gain_percent"] <= INDISTINGUISHABLE_PERCENT]
    return max(tied, key=lambda r: (r.get("residual_gain_percent", 0.0),
                                    -r.get("seconds", 0.0)))


def gop_split(candidate_gop: dict[str, Any], baseline_gop: dict[str, Any]) -> dict[str, Any]:
    """Position 1 against positions 2-9 - the split M16-M21 all found anomalous,
    reported so a global aggregate cannot hide a boundary-specific effect."""
    def _bucket(source, positions):
        out = {"residual_bytes": 0, "motion_bytes": 0, "frames": 0}
        for key, values in source.items():
            if int(key) in positions:
                for field in out:
                    out[field] += values[field]
        return out

    result = {}
    for name, positions in (("boundary", {1}), ("ordinary", set(range(2, 10)))):
        base = _bucket(baseline_gop, positions)
        cand = _bucket(candidate_gop, positions)
        result[name] = {
            "frames": base["frames"],
            "baseline_residual_bytes": base["residual_bytes"],
            "candidate_residual_bytes": cand["residual_bytes"],
            "delta_residual_bytes": cand["residual_bytes"] - base["residual_bytes"],
            "residual_gain_percent": ((base["residual_bytes"] - cand["residual_bytes"])
                                      / base["residual_bytes"] * 100
                                      if base["residual_bytes"] else 0.0),
        }
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M22 Phases 5/14/17 and the gate decision, from recorded JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/m22_residual_freeze_lift"))
    parser.add_argument("--sweep-name", type=str, default="m22_sweep.json")
    parser.add_argument("--repro", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    baseline = json.loads((args.output_dir / "m22_baseline.json").read_text(encoding="utf-8"))
    sweep = json.loads((args.output_dir / args.sweep_name).read_text(encoding="utf-8"))
    diagnostics_path = args.output_dir / "m22_diagnostics.json"
    diagnostics = (json.loads(diagnostics_path.read_text(encoding="utf-8"))
                   if diagnostics_path.is_file() else None)

    print("=" * 142)
    print("M22 PHASES 5/14/17 - VAL-B GATE, GOP SPLIT, SELECTION, CLASSIFICATION")
    print("=" * 142)

    report: dict[str, Any] = {
        "phase": "M22 Phases 5/14/17",
        "baseline_status": baseline.get("status"),
        "distortion_guard": {"max_psnr_regression_db": MAX_PSNR_REGRESSION_DB,
                             "max_msssim_regression": MAX_MSSSIM_REGRESSION},
        "rate_points": [],
    }

    if diagnostics:
        print("\n  PHASE 2 MECHANISM SUMMARY (why the symbols are unstable)")
        print(f"    {'bits':>5} {'step/std':>9} {'ref shift RMS':>14} {'1-step flips':>13} "
              f"{'their excess share':>19} {'>=2-step flips':>15}")
        for point in diagnostics["rate_points"]:
            stats, sensitivity = point["train_grid_statistics"], point["val_b_sensitivity"]
            displacement = sensitivity["symbol_displacement"]
            one_step = displacement["fraction_of_positions"][1]
            one_share = displacement["share_of_excess_bits"][1]
            multi = sum(displacement["fraction_of_positions"][2:])
            print(f"    {point['bits']:>5} {stats['step_over_std_mean']:>9.4f} "
                  f"{sensitivity['rms_reference_shift_steps']:>14.4f} "
                  f"{one_step * 100:>12.2f}% {one_share * 100:>18.2f}% {multi * 100:>14.2f}%")
        report["phase2_summary"] = [
            {"bits": p["bits"],
             "step_over_std": p["train_grid_statistics"]["step_over_std_mean"],
             "rms_reference_shift_steps": p["val_b_sensitivity"]["rms_reference_shift_steps"],
             "one_step_flip_fraction": p["val_b_sensitivity"][
                 "symbol_displacement"]["fraction_of_positions"][1],
             "one_step_excess_share": p["val_b_sensitivity"][
                 "symbol_displacement"]["share_of_excess_bits"][1]}
            for p in diagnostics["rate_points"]]

    for point in sweep["rate_points"]:
        bits = point["bits"]
        rows = point["candidates"]
        control = next(r for r in rows if r["is_control"])
        base = control["aggregate"]

        print(f"\n  ================ {bits}-bit  (deployed residual="
              f"{point['deployed_identity']}) ================")
        print(f"    control: total={base['total_container_bytes']:,}  "
              f"residual={base['total_residual_bytes']:,}  "
              f"motion={base['total_motion_bytes']:,}  BPP={base['stream_bpp']:.6f}  "
              f"PSNR={base['mean_psnr_db']:.4f}  MS-SSIM={base['mean_msssim']:.6f}")
        print(f"\n    {'variant/arm':>24} {'step':>8} {'clip%':>7} {'Hsym':>6} {'d total':>10} "
              f"{'stream %':>9} {'resid %':>9} {'dPSNR':>8} {'dMS-SSIM':>10} {'dec':>4} "
              f"{'verdict':>9}")
        for row in rows:
            if row["is_control"]:
                continue
            stats = row["train_grid_statistics"]
            guard = distortion_regression(row.get("delta_psnr_db", 0.0),
                                          row.get("delta_msssim", 0.0))
            print(f"    {row['label']:>24} {stats['step_mean']:>8.4f} "
                  f"{stats['clipping_fraction'] * 100:>7.3f} "
                  f"{stats['symbol_entropy_bits']:>6.3f} "
                  f"{row['delta_total_bytes']:>+10,} "
                  f"{row['total_stream_gain_percent']:>+9.4f} "
                  f"{row['residual_gain_percent']:>+9.4f} "
                  f"{row['delta_psnr_db']:>+8.4f} {row['delta_msssim']:>+10.6f} "
                  f"{str(row['decoder_compatible'])[0]:>4} {row['verdict']:>9}"
                  + ("  [DISTORTION]" if guard else ""))

        stale = [r for r in rows if r["arm"] == "stale" and not r["is_control"]]
        refit = [r for r in rows if r["arm"] == "refit" and not r["is_control"]]
        if stale and refit:
            by_variant = {r["variant"]: r for r in refit}
            paired = [(r, by_variant[r["variant"]]) for r in stale if r["variant"] in by_variant]
            if paired:
                coupling = sum(b["total_stream_gain_percent"] - a["total_stream_gain_percent"]
                               for a, b in paired) / len(paired)
                print(f"\n    COUPLING PRICE: refitting the downstream stack is worth "
                      f"{coupling:+.4f}% of the total stream on average over "
                      f"{len(paired)} variants - what the calibration-signature guard protects")
                report.setdefault("coupling_price_percent", {})[str(bits)] = coupling

        selected = select_candidate(rows)
        entry: dict[str, Any] = {
            "bits": bits, "deployed_identity": point["deployed_identity"],
            "control": base,
            "candidates": [{k: v for k, v in row.items()
                            if k not in ("per_sequence", "gop_position", "aggregate")}
                           for row in rows],
            "all_decoder_compatible": all(r["decoder_compatible"] for r in rows),
        }
        if selected:
            entry["selected"] = {
                "variant": selected["variant"], "arm": selected["arm"],
                "label": selected["label"],
                "total_stream_gain_percent": selected["total_stream_gain_percent"],
                "delta_total_bytes": selected["delta_total_bytes"],
                "residual_gain_percent": selected["residual_gain_percent"],
                "delta_psnr_db": selected["delta_psnr_db"],
                "delta_msssim": selected["delta_msssim"],
                "verdict": verdict(selected["total_stream_gain_percent"]),
                "gop": gop_split(selected["gop_position"], control["gop_position"])
                if "gop_position" in selected and "gop_position" in control else None,
            }
            gop = gop_split(
                next(r for r in rows if r["label"] == selected["label"])["gop_position"],
                control["gop_position"])
            entry["selected"]["gop"] = gop
            print(f"\n    BEST BY THE DECLARED RULE: {selected['label']}  "
                  f"{selected['delta_total_bytes']:+,} bytes "
                  f"({selected['total_stream_gain_percent']:+.4f}% total stream, "
                  f"'{verdict(selected['total_stream_gain_percent'])}')")
            print(f"    Phase 14 GOP  {'group':>10} {'frames':>7} {'d residual':>12} {'resid %':>9}")
            for name, values in gop.items():
                print(f"                  {name:>10} {values['frames']:>7} "
                      f"{values['delta_residual_bytes']:>+12,} "
                      f"{values['residual_gain_percent']:>+9.4f}")
        else:
            print("\n    BEST BY THE DECLARED RULE: none "
                  "(no candidate is both decoder-compatible and within the distortion guard)")
        report["rate_points"].append(entry)
        print(flush=True)

    # Phase 12, the honest framing: a residual-grid change moves rate AND
    # distortion, so a byte win at a fixed nominal depth may be nothing more than
    # a lower quality point. BD-rate collapses the two into one number wherever
    # enough rate points were measured for a curve to exist.
    # the control label differs by run: the screen's control is the deployed grid
    # with the DEPLOYED stack, while a refit-only run's control is the deployed
    # grid with a REFITTED stack. Pick whichever this file actually contains.
    labels = {row["label"] for point in sweep["rate_points"] for row in point["candidates"]}
    control_label = "deployed/stale" if "deployed/stale" in labels else "deployed/refit"
    bd_rows = bd_rate_table(sweep, control=control_label)
    if any(row["bd_rate_psnr"] is not None for row in bd_rows):
        print("\n  RATE/DISTORTION (BD-rate vs the deployed control; negative = fewer bits "
              "at equal quality)")
        print(f"    {'variant/arm':>24} {'points':>7} {'BD-rate PSNR':>14} "
              f"{'BD-rate MS-SSIM':>16}")
        for row in sorted(bd_rows, key=lambda r: (r["bd_rate_psnr"] is None,
                                                  r["bd_rate_psnr"] or 0.0)):
            psnr = ("n/a" if row["bd_rate_psnr"] is None
                    else f"{row['bd_rate_psnr']:+.4f}%")
            msssim = ("n/a" if row["bd_rate_msssim"] is None
                      else f"{row['bd_rate_msssim']:+.4f}%")
            print(f"    {row['label']:>24} {row['points']:>7} {psnr:>14} {msssim:>16}")
        print("    (a candidate that only slid along the existing R/D curve reads ~0% here, "
              "however many bytes it saved)", flush=True)
    report["bd_rate_val_b"] = bd_rows

    selections = [e["selected"] for e in report["rate_points"] if "selected" in e]
    best_gain = max((s["total_stream_gain_percent"] for s in selections), default=0.0)
    all_compatible = all(e["all_decoder_compatible"] for e in report["rate_points"])
    report["best_val_b_total_stream_gain_percent"] = best_gain
    report["all_decoder_compatible"] = all_compatible
    report["coded_validation_gated"] = best_gain >= WEAK_BELOW_PERCENT
    report["full_davis_gated"] = best_gain >= WEAK_BELOW_PERCENT

    davis_path = args.output_dir / "m22_davis.json"
    davis_gain = None
    if davis_path.is_file():
        davis = json.loads(davis_path.read_text(encoding="utf-8"))
        davis_gain = max(p["delta"]["total_stream_gain_percent"] for p in davis["rate_points"])
        report["davis"] = {
            "best_total_stream_gain_percent": davis_gain,
            "per_rate_point": {str(p["bits"]): p["delta"]["total_stream_gain_percent"]
                               for p in davis["rate_points"]},
            "bd_rate": davis.get("bd_rate"),
            "generalization_gap_percent": best_gain - davis_gain,
        }
    graded_on = "davis_test" if davis_gain is not None else "val_b"
    graded_gain = davis_gain if davis_gain is not None else best_gain
    report["graded_on"] = graded_on
    report["graded_gain_percent"] = graded_gain

    if not all_compatible:
        classification = "D - DECODER/PROVENANCE FAILURE"
    elif graded_gain >= MEANINGFUL_ABOVE_PERCENT:
        classification = "A - MEANINGFUL RESIDUAL-REPRESENTATION SUCCESS"
    elif graded_gain >= WEAK_BELOW_PERCENT:
        classification = "B - MARGINAL RESIDUAL-RATE IMPROVEMENT"
    else:
        classification = "C - NO CODED-RATE IMPROVEMENT"
    report["classification"] = classification

    if args.repro and args.repro.is_file():
        other = json.loads(args.repro.read_text(encoding="utf-8"))
        other_by_bits = {p["bits"]: p for p in other["rate_points"]}
        fields = ("total_container_bytes", "total_residual_bytes", "total_motion_bytes",
                  "total_i_frame_residual_bytes", "total_p_frame_residual_bytes", "stream_bpp",
                  "mean_psnr_db", "mean_msssim", "p_frame_ideal_bits")
        checks = []
        for point in sweep["rate_points"]:
            if point["bits"] not in other_by_bits:
                continue
            mine = {r["label"]: r["aggregate"] for r in point["candidates"]}
            theirs = {r["label"]: r["aggregate"]
                      for r in other_by_bits[point["bits"]]["candidates"]}
            shared = sorted(set(mine) & set(theirs))
            checks.append({"bits": point["bits"], "labels": len(shared),
                           "identical": all(mine[name][field] == theirs[name][field]
                                            for name in shared for field in fields)})
        report["reproducibility"] = {"compared": str(args.repro), "checks": checks,
                                     "all_identical": all(c["identical"] for c in checks)}

    print("=" * 142)
    print("M22 SUMMARY")
    print("=" * 142)
    print(f"  DECODER COMPATIBLE        : {all_compatible}")
    print(f"  BEST VAL-B GAIN           : {best_gain:+.4f}%  ({verdict(best_gain)})")
    print(f"  CODED VALIDATION GATED    : {report['coded_validation_gated']} "
          f"(needs >= {WEAK_BELOW_PERCENT}%)")
    print(f"  FULL DAVIS GATED          : {report['full_davis_gated']}")
    if davis_gain is not None:
        entry = report["davis"]
        print(f"  DAVIS TEST (held out)     : {davis_gain:+.4f}%  ({verdict(davis_gain)})   "
              + "  ".join(f"{b}-bit={g:+.4f}%" for b, g in
                          sorted(entry["per_rate_point"].items(), reverse=True)))
        print(f"  GENERALIZATION GAP        : {entry['generalization_gap_percent']:+.4f} points")
    print(f"\n  GRADED ON: {graded_on}  ({graded_gain:+.4f}%)")
    print(f"  CLASSIFICATION: {classification}")

    path = args.output_dir / "m22_analysis.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
