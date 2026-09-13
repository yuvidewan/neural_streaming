"""M21 Phases 6-7 + the gate decision - a pure analysis pass over the JSON the
earlier phases wrote. Runs no model and reads no data, so the verdict is
reproducible from the recorded artifacts alone.

Produces:
  * the mechanism decomposition (Phase 6): how each candidate moved reference
    pixel quality, reference latent quality, motion, the residual symbols, the
    G16 ideal bits, and the actual coded bytes - kept separate on purpose,
    because M18 established they do not move together;
  * the GOP-position split (Phase 6): position 1 against positions 2-9;
  * the candidate-selection rule (Phase 7), applied mechanically;
  * the total-stream gate (Phase 5) against the project's frozen thresholds.

Run:
  ./.venv/Scripts/python.exe scripts/m21_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

WEAK_BELOW_PERCENT = 0.5
MEANINGFUL_ABOVE_PERCENT = 1.0

# Two candidates whose total-stream effects differ by less than this are treated
# as operationally indistinguishable, and the Phase 7 tie-break applies.
INDISTINGUISHABLE_PERCENT = 0.05
# A candidate that buys bytes while losing more quality than this is flagged as a
# rate/distortion move rather than a compression win.
PSNR_LOSS_FLAG_DB = 0.05
MSSSIM_LOSS_FLAG = 0.0005


def verdict(gain: float) -> str:
    return ("meaningful" if gain >= MEANINGFUL_ABOVE_PERCENT else
            "marginal" if gain >= WEAK_BELOW_PERCENT else "weak")


def rate_distortion_flag(row: dict[str, Any]) -> str | None:
    """A byte saving paid for with quality is not a compression win. Returns a
    human-readable flag, or None when the candidate is rate-only."""
    psnr = row.get("delta_psnr_db", 0.0)
    quality = row.get("delta_msssim", 0.0)
    if row.get("total_stream_gain_percent", 0.0) <= 0:
        return None
    if psnr <= -PSNR_LOSS_FLAG_DB or quality <= -MSSSIM_LOSS_FLAG:
        return (f"rate/distortion move: {psnr:+.4f} dB PSNR, {quality:+.6f} MS-SSIM - the byte "
                f"saving is partly bought with quality, not a pure rate win")
    return None


def select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Phase 7's pre-declared rule, applied mechanically:

      1. highest actual coded-byte improvement on VAL-B (total stream);
      2. subject to decoder compatibility;
      3. if operationally indistinguishable, prefer zero side information (every
         pre-registered candidate qualifies), then lower compute, then the
         simpler implementation.

    Reconstruction quality is deliberately NOT a selection input - it is a
    reported constraint (see `rate_distortion_flag`), never a tiebreaker, so a
    candidate cannot be chosen for looking better.
    """
    eligible = [r for r in rows if r.get("decoder_compatible")
                and r.get("candidate") != "identity"]
    if not eligible:
        return None
    best = max(r["total_stream_gain_percent"] for r in eligible)
    tied = [r for r in eligible
            if best - r["total_stream_gain_percent"] <= INDISTINGUISHABLE_PERCENT]
    return min(tied, key=lambda r: (r.get("seconds", 0.0), r.get("candidate", "")))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M21 Phases 6/7 and the gate decision, from recorded JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/m21_reference_refinement"))
    parser.add_argument("--repro", type=Path, default=None,
                        help="Optional second, independent-process sweep JSON to compare.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    baseline = json.loads((args.output_dir / "m21_baseline.json").read_text(encoding="utf-8"))
    sweep = json.loads((args.output_dir / "m21_sweep.json").read_text(encoding="utf-8"))
    oracle_path = args.output_dir / "m21_oracle.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8")) if oracle_path.is_file() else None

    print("=" * 138)
    print("M21 PHASES 6/7 - MECHANISM DECOMPOSITION, GOP POSITION, SELECTION, GATE")
    print("=" * 138)

    report: dict[str, Any] = {
        "phase": "M21 Phases 6/7",
        "baseline_status": baseline.get("status"),
        "screen_bits": sweep["declared_screen_bits"],
        "confirm_bits": sweep["declared_confirm_bits"],
        "stage2_admitted": sweep.get("stage2_admitted"),
        "rate_points": [],
    }

    oracle_by_bits = {p["bits"]: p for p in oracle["rate_points"]} if oracle else {}

    for point in sweep["rate_points"]:
        bits = point["bits"]
        rows = point["candidates"]
        by_name = {r["candidate"]: r for r in rows}
        base = by_name["identity"]["aggregate"]

        print(f"\n  ================ {bits}-bit  ({point['stage']} stage, "
              f"residual={point['residual_identity']}) ================")
        print(f"    identity control: total={base['total_container_bytes']:,}  "
              f"residual={base['total_residual_bytes']:,}  "
              f"motion={base['total_motion_bytes']:,}  "
              f"intra={base['total_i_frame_residual_bytes']:,}  "
              f"BPP={base['stream_bpp']:.6f}  PSNR={base['mean_psnr_db']:.4f}  "
              f"MS-SSIM={base['mean_msssim']:.6f}")

        if bits in oracle_by_bits:
            entry = oracle_by_bits[bits]
            print(f"    M17 oracle bound (open loop): residual "
                  f"{entry['deployed_coded_residual_bytes']:,} -> "
                  f"{entry['oracle_coded_residual_bytes']:,} "
                  f"({entry['oracle_gap_percent']:+.3f}% of the residual channel)")

        print(f"\n    {'candidate':>22} {'d total':>10} {'stream %':>9} {'d resid':>9} "
              f"{'d motion':>9} {'ideal %':>9} {'dPSNR':>8} {'dMS-SSIM':>10} {'dBPP':>10} "
              f"{'dec':>5} {'verdict':>9}")
        for row in rows:
            if row["candidate"] == "identity":
                continue
            print(f"    {row['candidate']:>22} {row['delta_total_bytes']:>+10,} "
                  f"{row['total_stream_gain_percent']:>+9.4f} "
                  f"{row['delta_residual_bytes']:>+9,} {row['delta_motion_bytes']:>+9,} "
                  f"{row['ideal_bits_gain_percent']:>+9.4f} {row['delta_psnr_db']:>+8.4f} "
                  f"{row['delta_msssim']:>+10.6f} {row['delta_bpp']:>+10.6f} "
                  f"{str(row['decoder_compatible'])[0]:>5} {row['verdict']:>9}")

        selected = select_candidate(rows)
        entry: dict[str, Any] = {
            "bits": bits, "stage": point["stage"],
            "residual_identity": point["residual_identity"],
            "motion_identity": point["motion_identity"],
            "identity_control": base,
            "candidates": [{k: v for k, v in row.items()
                            if k not in ("per_sequence", "gop_position", "aggregate")}
                           for row in rows],
            "all_decoder_compatible": all(r["decoder_compatible"] for r in rows),
        }
        if selected:
            flag = rate_distortion_flag(selected)
            entry["selected"] = {
                "candidate": selected["candidate"],
                "total_stream_gain_percent": selected["total_stream_gain_percent"],
                "delta_total_bytes": selected["delta_total_bytes"],
                "verdict": verdict(selected["total_stream_gain_percent"]),
                "rate_distortion_flag": flag,
                "gop_comparison": selected.get("gop_comparison"),
            }
            print(f"\n    BEST BY THE DECLARED RULE: {selected['candidate']}  "
                  f"{selected['delta_total_bytes']:+,} bytes  "
                  f"({selected['total_stream_gain_percent']:+.4f}% total stream, "
                  f"'{verdict(selected['total_stream_gain_percent'])}')")
            if flag:
                print(f"    FLAG: {flag}")
            gop = selected.get("gop_comparison") or {}
            if gop:
                print(f"    Phase 6 GOP split  {'group':>10} {'frames':>7} "
                      f"{'d residual':>12} {'resid %':>9} {'d motion':>10}")
                for name, values in gop.items():
                    print(f"                       {name:>10} {values['frames']:>7} "
                          f"{values['delta_residual_bytes']:>+12,} "
                          f"{values['residual_gain_percent']:>+9.4f} "
                          f"{values['delta_motion_bytes']:>+10,}")
        report["rate_points"].append(entry)
        print(flush=True)

    # --- Phase 5 gate + classification -----------------------------------------
    selections = [e["selected"] for e in report["rate_points"] if "selected" in e]
    best_gain = max((s["total_stream_gain_percent"] for s in selections), default=0.0)
    all_compatible = all(e["all_decoder_compatible"] for e in report["rate_points"])
    report["best_total_stream_gain_percent"] = best_gain
    report["all_decoder_compatible"] = all_compatible
    report["coded_validation_gated"] = best_gain >= WEAK_BELOW_PERCENT
    report["full_davis_gated"] = best_gain >= WEAK_BELOW_PERCENT

    # If Phase 9 has run, the HELD-OUT DAVIS TEST result is what the
    # classification is based on - VAL-B selected the candidate, so grading on
    # VAL-B would grade on the selection set. VAL-B's own number is kept and
    # reported either way, and the gap between them is itself a result.
    davis_path = args.output_dir / "m21_davis.json"
    davis_gain = None
    if davis_path.is_file():
        davis = json.loads(davis_path.read_text(encoding="utf-8"))
        davis_gain = max(p["delta"]["total_stream_gain_percent"] for p in davis["rate_points"])
        report["davis"] = {
            "best_total_stream_gain_percent": davis_gain,
            "per_rate_point": {str(p["bits"]): p["delta"]["total_stream_gain_percent"]
                               for p in davis["rate_points"]},
            "bd_rate": davis["bd_rate"],
            "identity_matches_m14_recorded": davis["baseline_matches_m14"],
            "generalization_gap_percent": best_gain - davis_gain,
        }
    graded_on = "davis_test" if davis_gain is not None else "val_b"
    graded_gain = davis_gain if davis_gain is not None else best_gain
    report["graded_on"] = graded_on
    report["graded_gain_percent"] = graded_gain

    if not all_compatible:
        classification = "D - DECODER-INCOMPATIBLE"
    elif graded_gain >= MEANINGFUL_ABOVE_PERCENT:
        classification = "A - MEANINGFUL CAUSAL REFERENCE-REFINEMENT SUCCESS"
    elif graded_gain >= WEAK_BELOW_PERCENT:
        classification = "B - MARGINAL RATE IMPROVEMENT"
    else:
        classification = "C - NO CODED-RATE IMPROVEMENT"
    report["classification"] = classification

    if args.repro and args.repro.is_file():
        other = json.loads(args.repro.read_text(encoding="utf-8"))
        other_by_bits = {p["bits"]: p for p in other["rate_points"]}
        checks = []
        fields = ("total_container_bytes", "total_residual_bytes", "total_motion_bytes",
                  "total_i_frame_residual_bytes", "total_p_frame_residual_bytes",
                  "stream_bpp", "mean_psnr_db", "mean_msssim", "p_frame_ideal_bits")
        for point in sweep["rate_points"]:
            if point["bits"] not in other_by_bits:
                continue
            mine = {r["candidate"]: r["aggregate"] for r in point["candidates"]}
            theirs = {r["candidate"]: r["aggregate"]
                      for r in other_by_bits[point["bits"]]["candidates"]}
            shared = sorted(set(mine) & set(theirs))
            checks.append({
                "bits": point["bits"], "candidates": len(shared),
                "identical": all(mine[name][field] == theirs[name][field]
                                 for name in shared for field in fields)})
        report["reproducibility"] = {"compared": str(args.repro), "checks": checks,
                                     "all_identical": all(c["identical"] for c in checks)}

    print("=" * 138)
    print("M21 SUMMARY")
    print("=" * 138)
    header = (f"  {'candidate':>22} | {'5-bit stream %':>15} {'4-bit stream %':>15} "
              f"{'3-bit stream %':>15} | {'best verdict':>13}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    by_bits = {e["bits"]: {c["candidate"]: c for c in e["candidates"]}
               for e in report["rate_points"]}
    screen = by_bits.get(sweep["declared_screen_bits"], {})
    for name in screen:
        if name == "identity":
            continue
        cells, gains = [], []
        for bits in (5, 4, 3):
            row = by_bits.get(bits, {}).get(name)
            if row is None:
                cells.append("       --      ")
            else:
                cells.append(f"{row['total_stream_gain_percent']:>+15.4f}")
                gains.append(row["total_stream_gain_percent"])
        print(f"  {name:>22} | {cells[0]} {cells[1]} {cells[2]} | "
              f"{verdict(max(gains)) if gains else 'n/a':>13}")
    print("  " + "-" * (len(header) - 2))
    print(f"  {'identity (deployed)':>22} | {0.0:>+15.4f} {0.0:>+15.4f} {0.0:>+15.4f} | "
          f"{'control':>13}")

    print(f"\n  DECODER COMPATIBLE     : {all_compatible}")
    print(f"  BEST VAL-B GAIN        : {best_gain:+.4f}%  ({verdict(best_gain)})")
    print(f"  CODED VALIDATION GATED : {report['coded_validation_gated']} "
          f"(needs >= {WEAK_BELOW_PERCENT}%)")
    print(f"  FULL DAVIS GATED       : {report['full_davis_gated']}")
    if davis_gain is not None:
        entry = report["davis"]
        print(f"  DAVIS TEST (held out)  : {davis_gain:+.4f}%  ({verdict(davis_gain)})   "
              f"per rate point " + "  ".join(
                  f"{b}-bit={g:+.4f}%" for b, g in sorted(
                      entry["per_rate_point"].items(), reverse=True)))
        print(f"  GENERALIZATION GAP     : {entry['generalization_gap_percent']:+.4f} points "
              f"(VAL-B minus TEST)")
        print(f"  BD-rate applied at every rate point: PSNR "
              f"{entry['bd_rate']['psnr']:+.3f}%  MS-SSIM {entry['bd_rate']['msssim']:+.3f}%")
    print(f"\n  GRADED ON: {graded_on}  ({graded_gain:+.4f}%)")
    print(f"  CLASSIFICATION: {classification}")

    path = args.output_dir / "m21_analysis.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
