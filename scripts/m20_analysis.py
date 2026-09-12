"""M20 Phases F/G/H + the decision - a pure analysis pass over the JSON the
earlier phases wrote. Runs no model and touches no data, so the verdict is
reproducible from the recorded artifacts alone.

Produces:
  * Phase F  the rate-response curve (margin vs churn vs bytes) and its shape
             classification per bit depth - monotone-improving, finite optimum,
             flat, immediately harmful, or bit-depth dependent;
  * Phase G  the GOP-position split for the best candidate, if any;
  * Phase H  the total-stream gate against the project's frozen thresholds;
  * the opportunity accounting: M19's routing-only share, the maximum M20 could
    possibly recover, what it actually recovered, and the fraction of the
    opportunity that represents.

Run:
  ./.venv/Scripts/python.exe scripts/m20_analysis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

WEAK_BELOW_PERCENT = 0.5
MEANINGFUL_ABOVE_PERCENT = 1.0


def _verdict(gain: float) -> str:
    return ("meaningful" if gain >= MEANINGFUL_ABOVE_PERCENT else
            "marginal" if gain >= WEAK_BELOW_PERCENT else "weak")


def classify_response(margins: list[float], gains: list[float], *,
                      flat_below: float = 0.01, negligible: float = 0.01) -> str:
    """Shape of the rate response, without assuming monotonicity.

    `gains` are total-stream gain percentages (positive = fewer bytes), ordered
    by increasing margin, with margin = 0 first and gain 0.

    `negligible` guards against dressing up a +0.0004% blip at the smallest
    margin - four bytes out of half a megabyte - as "a finite optimum". A point
    only counts as a gain if it clears 0.01% of the total stream, which is still
    fifty times below the project's "weak" line, so the guard cannot mask
    anything the gate would have cared about.
    """
    if all(abs(g) < flat_below for g in gains[1:]):
        return "flat (no margin moves the rate measurably)"
    best_index = max(range(len(gains)), key=lambda i: gains[i])
    if gains[best_index] < negligible:
        return ("immediately harmful (no margin produces a gain above 0.01% of the "
                "total stream; the best non-zero point is "
                f"{gains[best_index]:+.4f}%)")
    if best_index == len(gains) - 1:
        return "monotonic improvement up to the largest margin tested"
    if best_index == 0:
        return "immediately harmful (margin = 0 is the optimum)"
    return f"optimum at a finite margin ({margins[best_index]})"


def counterproductive(row: dict[str, Any], zero: dict[str, Any]) -> bool:
    """Phase E's explicit failure mode: routing churn falls while coded bytes
    rise. This is the case the milestone says must be called counterproductive
    rather than reported as a churn win."""
    return (row["routing_churn_vs_oracle"] < zero["routing_churn_vs_oracle"]
            and row["real_residual_bytes"] > zero["real_residual_bytes"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M20 Phases F/G/H and the decision, from recorded JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m20_codebook_hysteresis"))
    parser.add_argument("--repro", type=Path, default=None,
                        help="Optional second, independent-process sweep JSON to compare.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    baseline = json.loads((args.output_dir / "m20_baseline.json").read_text(encoding="utf-8"))
    sweep = json.loads((args.output_dir / "m20_sweep.json").read_text(encoding="utf-8"))
    trace = json.loads((args.output_dir / "m20_assignment_trace.json").read_text(encoding="utf-8"))
    share = {int(k): v for k, v in sweep["p_residual_share_of_total_stream"].items()}
    baseline_by_bits = {rp["bits"]: rp for rp in baseline["rate_points"]}

    print("=" * 132)
    print("M20 PHASES F/G/H - RATE RESPONSE, GOP POSITION, TOTAL-STREAM GATE")
    print("=" * 132)

    report: dict[str, Any] = {"phase": "M20 Phases F/G/H", "rate_points": [],
                              "decoder_compatible": trace["all_decoder_compatible"]}
    best_overall = None

    for point in sweep["rate_points"]:
        bits = point["bits"]
        rows = point["configurations"]
        by_key = {(r["state"], r["margin"]): r for r in rows}
        zero = by_key[("baseline", 0.0)]
        base_routing = baseline_by_bits[bits]["routing"]
        routing_only_bits = base_routing["routing_only_excess_bits"]
        ceiling_bytes = routing_only_bits / 8.0
        ceiling_residual_percent = ceiling_bytes / zero["real_residual_bytes"] * 100
        ceiling_stream_percent = ceiling_residual_percent * share[bits]

        print(f"\n  ================ {bits}-bit "
              f"({point['p_frames']} VAL-B P-frames, residual={point['residual_identity']}) "
              "================")
        print(f"    M19 routing-only opportunity : {base_routing['routing_only_share_of_excess_bits'] * 100:.2f}% "
              f"of excess bits = {routing_only_bits:,.0f} bits = {ceiling_bytes:,.0f} bytes")
        print(f"    M20 maximum possible target  : {ceiling_residual_percent:+.4f}% residual "
              f"= {ceiling_stream_percent:+.4f}% total stream "
              f"(and even that needs the ORACLE's assignment, which no decoder has)")

        curves = {}
        for state in sweep["declared_states"]:
            margins = [0.0] + [m for m in sweep["declared_margins"] if m > 0]
            gains, deltas, churn, held, worse = [], [], [], [], []
            for margin in margins:
                row = zero if margin == 0.0 else by_key[(state, margin)]
                gains.append(row["total_stream_gain_percent"])
                deltas.append(row["delta_residual_bytes_vs_margin0"])
                churn.append(row["routing_churn_vs_oracle"])
                held.append(row["held_fraction"])
                worse.append(row["held_made_code_length_worse"] / max(row["held_positions"], 1))
            counter = [False] + [counterproductive(by_key[(state, m)], zero)
                                 for m in margins[1:]]
            curves[state] = {"margins": margins, "total_stream_gain_percent": gains,
                             "delta_residual_bytes": deltas, "routing_churn": churn,
                             "held_fraction": held, "held_worse_fraction": worse,
                             "counterproductive": counter,
                             "response_shape": classify_response(margins, gains)}
            print(f"\n    [{state}]  {curves[state]['response_shape']}")
            print(f"      {'margin':>8} {'d bytes':>10} {'residual %':>11} {'stream %':>10} "
                  f"{'churn':>8} {'churn red.':>11} {'held':>8} {'held worse':>11} "
                  f"{'routing-only':>13}")
            for i, margin in enumerate(margins):
                row = zero if margin == 0.0 else by_key[(state, margin)]
                cell = row["decomposition_2x2"]["symbol_changed=False_assignment_changed=True"]
                base_cell = zero["decomposition_2x2"][
                    "symbol_changed=False_assignment_changed=True"]
                routing_only_change = ((cell["positions"] - base_cell["positions"])
                                       / base_cell["positions"] * 100
                                       if base_cell["positions"] else 0.0)
                print(f"      {margin:>8.3f} {deltas[i]:>+10,} "
                      f"{row['residual_gain_percent']:>+11.4f} {gains[i]:>+10.4f} "
                      f"{churn[i] * 100:>7.2f}% "
                      f"{(zero['routing_churn_vs_oracle'] - churn[i]) * 100:>+10.2f}% "
                      f"{held[i] * 100:>7.2f}% {worse[i] * 100:>10.2f}% "
                      f"{routing_only_change:>+12.2f}%"
                      + ("  <- churn DOWN, bits UP: counterproductive" if counter[i] else ""))

        # Phase E items 4 and 5: hysteresis must not change WHICH symbols are
        # coded, only which table codes them - so the symbol-change rate against
        # the oracle (and hence every G16 context plane, which is a function of
        # the symbols alone) must be bit-identical across every configuration.
        symbol_rates = {(r["state"], r["margin"]): r["symbol_change_rate_vs_oracle"]
                        for r in rows}
        positions = {(r["state"], r["margin"]): r["positions"] for r in rows}
        invariants = {
            "symbol_change_rate_identical_across_configurations":
                len(set(symbol_rates.values())) == 1,
            "symbol_change_rate": zero["symbol_change_rate_vs_oracle"],
            "positions_identical_across_configurations": len(set(positions.values())) == 1,
            "distinct_symbol_change_rates": sorted(set(symbol_rates.values())),
        }
        print(f"\n    Phase E invariants: symbol-change rate identical across all "
              f"{len(rows)} configurations = "
              f"{invariants['symbol_change_rate_identical_across_configurations']} "
              f"({zero['symbol_change_rate_vs_oracle'] * 100:.4f}%)  -> residual symbols and "
              f"therefore G16 context are untouched by the rule")

        candidates = [r for r in rows if r["is_hysteresis_candidate"] and r["margin"] > 0]
        best = max(candidates, key=lambda r: r["total_stream_gain_percent"])
        diagnostics = {r["state"]: r for r in rows if not r["is_hysteresis_candidate"]}

        print(f"\n    BEST CANDIDATE: {best['state']} @ margin {best['margin']}  "
              f"-> {best['delta_residual_bytes_vs_margin0']:+,} bytes "
              f"({best['total_stream_gain_percent']:+.4f}% total stream, "
              f"verdict '{_verdict(best['total_stream_gain_percent'])}')")

        # Phase G - GOP position, for the best candidate whether or not it wins.
        gop = {}
        for name in ("boundary", "ordinary"):
            zero_bucket = zero["gop_position"][name]
            best_bucket = best["gop_position"][name]
            gop[name] = {
                "frames": zero_bucket["frames"],
                "baseline_bytes": zero_bucket["real_bytes"],
                "candidate_bytes": best_bucket["real_bytes"],
                "delta_bytes": best_bucket["real_bytes"] - zero_bucket["real_bytes"],
                "residual_gain_percent": (-(best_bucket["real_bytes"] - zero_bucket["real_bytes"])
                                          / zero_bucket["real_bytes"] * 100
                                          if zero_bucket["real_bytes"] else 0.0),
                "baseline_routing_only_positions": zero_bucket["routing_only"],
                "candidate_routing_only_positions": best_bucket["routing_only"],
                "routing_only_reduction_percent": (
                    (zero_bucket["routing_only"] - best_bucket["routing_only"])
                    / zero_bucket["routing_only"] * 100 if zero_bucket["routing_only"] else 0.0),
                "held_fraction": (best_bucket["held"] / best_bucket["positions"]
                                  if best_bucket["positions"] else 0.0),
            }
        print(f"    Phase G  {'group':>10} {'frames':>7} {'bytes d':>10} {'resid %':>9} "
              f"{'routing-only d':>15} {'held':>8}")
        for name, values in gop.items():
            print(f"             {name:>10} {values['frames']:>7} {values['delta_bytes']:>+10,} "
                  f"{values['residual_gain_percent']:>+9.4f} "
                  f"{values['routing_only_reduction_percent']:>+14.2f}% "
                  f"{values['held_fraction'] * 100:>7.2f}%")

        print("\n    Reference points (NOT M20 candidates):")
        for name, row in diagnostics.items():
            label = ("decoder-available" if row["is_decoder_compatible"]
                     else "NON-CAUSAL upper bound")
            print(f"      {name:>14} [{label:>21}]  "
                  f"{row['delta_residual_bytes_vs_margin0']:>+10,} bytes  "
                  f"{row['residual_gain_percent']:>+9.4f}% residual  "
                  f"{row['total_stream_gain_percent']:>+9.4f}% total stream")

        recovered = (best["total_stream_gain_percent"] / ceiling_stream_percent * 100
                     if ceiling_stream_percent else 0.0)
        entry = {
            "bits": bits, "p_frames": point["p_frames"],
            "residual_identity": point["residual_identity"],
            "baseline_residual_bytes": zero["real_residual_bytes"],
            "m19_routing_only_share": base_routing["routing_only_share_of_excess_bits"],
            "m19_routing_only_excess_bits": routing_only_bits,
            "m20_ceiling_bytes": ceiling_bytes,
            "m20_ceiling_residual_percent": ceiling_residual_percent,
            "m20_ceiling_total_stream_percent": ceiling_stream_percent,
            "curves": curves,
            "phase_e_invariants": invariants,
            "any_counterproductive_configuration": any(
                any(curve["counterproductive"]) for curve in curves.values()),
            "counterproductive_configurations": [
                {"state": state, "margin": curve["margins"][i]}
                for state, curve in curves.items()
                for i, flag in enumerate(curve["counterproductive"]) if flag],
            "best_candidate": {
                "state": best["state"], "margin": best["margin"],
                "delta_residual_bytes": best["delta_residual_bytes_vs_margin0"],
                "residual_gain_percent": best["residual_gain_percent"],
                "total_stream_gain_percent": best["total_stream_gain_percent"],
                "verdict": _verdict(best["total_stream_gain_percent"]),
                "churn_reduction_absolute": best["churn_reduction_absolute"],
                "held_fraction": best["held_fraction"],
                "held_worse_fraction": (best["held_made_code_length_worse"]
                                        / max(best["held_positions"], 1)),
                "fraction_of_opportunity_recovered_percent": recovered,
            },
            "gop_position": gop,
            "reference_points": {
                name: {"delta_residual_bytes": row["delta_residual_bytes_vs_margin0"],
                       "residual_gain_percent": row["residual_gain_percent"],
                       "total_stream_gain_percent": row["total_stream_gain_percent"],
                       "decoder_compatible": row["is_decoder_compatible"]}
                for name, row in diagnostics.items()},
            "roundtrip_all_exact": point["roundtrip_all_exact"],
            "roundtrip_checked": point["roundtrip_checked"],
            "margin0_identical_to_deployed": point["margin0_identical_to_deployed"],
        }
        report["rate_points"].append(entry)
        if best_overall is None or best["total_stream_gain_percent"] > best_overall[1]:
            best_overall = (bits, best["total_stream_gain_percent"])

    # --- Phase H + decision ----------------------------------------------------
    gains = [e["best_candidate"]["total_stream_gain_percent"] for e in report["rate_points"]]
    best_gain = max(gains)
    if not trace["all_decoder_compatible"]:
        classification = "D - HYSTERESIS IS NOT DECODER-COMPATIBLE"
    elif best_gain >= MEANINGFUL_ABOVE_PERCENT:
        classification = "A - HYSTERESIS PRODUCES A MEANINGFUL TOTAL-STREAM GAIN"
    elif best_gain >= WEAK_BELOW_PERCENT:
        classification = "B - HYSTERESIS PRODUCES A REAL BUT MARGINAL GAIN"
    else:
        classification = "C - HYSTERESIS DOES NOT IMPROVE CODED RATE"
    report["best_total_stream_gain_percent"] = best_gain
    report["classification"] = classification
    report["coded_validation_gated"] = best_gain >= WEAK_BELOW_PERCENT
    report["full_davis_gated"] = best_gain >= WEAK_BELOW_PERCENT

    if args.repro and args.repro.is_file():
        other = json.loads(args.repro.read_text(encoding="utf-8"))
        checks = []
        other_by_bits = {p["bits"]: p for p in other["rate_points"]}
        for point in sweep["rate_points"]:
            if point["bits"] not in other_by_bits:
                continue
            mine = {(r["state"], r["margin"]): r for r in point["configurations"]}
            theirs = {(r["state"], r["margin"]): r
                      for r in other_by_bits[point["bits"]]["configurations"]}
            fields = ("real_residual_bytes", "oracle_residual_bytes", "real_ideal_bits",
                      "held_positions", "held_made_code_length_worse", "routing_churn_vs_oracle",
                      "mean_code_length_bits", "total_excess_bits_vs_oracle",
                      "delta_residual_bytes_vs_margin0")
            identical = all(mine[key][field] == theirs[key][field]
                            for key in mine for field in fields)
            checks.append({"bits": point["bits"], "configurations": len(mine),
                           "identical": identical})
        report["reproducibility"] = {"compared": str(args.repro), "checks": checks,
                                     "all_identical": all(c["identical"] for c in checks)}

    print("\n" + "=" * 132)
    print("M20 SUMMARY")
    print("=" * 132)
    header = (f"  {'margin':>8} {'state':>14} | {'5-bit d bytes':>14} {'4-bit d bytes':>14} "
              f"{'3-bit d bytes':>14} | {'churn reduction':>16} | {'total-stream':>13} | verdict")
    print(header)
    print("  " + "-" * (len(header) - 2))
    by_bits = {e["bits"]: e for e in report["rate_points"]}
    for state in sweep["declared_states"]:
        for margin in sweep["declared_margins"]:
            if margin == 0.0:
                continue
            cells, churns, streams = [], [], []
            for bits in (5, 4, 3):
                curve = by_bits[bits]["curves"][state]
                index = curve["margins"].index(margin)
                cells.append(curve["delta_residual_bytes"][index])
                churns.append(curve["routing_churn"][0] - curve["routing_churn"][index])
                streams.append(curve["total_stream_gain_percent"][index])
            mean_stream = sum(streams) / len(streams)
            print(f"  {margin:>8.3f} {state:>14} | {cells[0]:>+14,} {cells[1]:>+14,} "
                  f"{cells[2]:>+14,} | {sum(churns) / 3 * 100:>+15.2f}% | "
                  f"{mean_stream:>+12.4f}% | {_verdict(max(streams))}")
    print("  " + "-" * (len(header) - 2))
    print(f"  {'margin 0':>8} {'(deployed)':>14} | {0:>+14,} {0:>+14,} {0:>+14,} | "
          f"{0.0:>+15.2f}% | {0.0:>+12.4f}% | identity control")

    print("\n  OPPORTUNITY ACCOUNTING")
    print(f"    {'bits':>5} {'M19 routing-only':>18} {'M20 max target':>16} {'M20 measured':>14} "
          f"{'fraction recovered':>20}")
    for entry in report["rate_points"]:
        best = entry["best_candidate"]
        print(f"    {entry['bits']:>5} "
              f"{entry['m19_routing_only_share'] * 100:>17.2f}% "
              f"{entry['m20_ceiling_total_stream_percent']:>+15.4f}% "
              f"{best['total_stream_gain_percent']:>+13.4f}% "
              f"{best['fraction_of_opportunity_recovered_percent']:>19.2f}%")

    print(f"\n  DECODER COMPATIBLE     : {trace['all_decoder_compatible']}")
    print(f"  BEST TOTAL-STREAM GAIN : {best_gain:+.4f}% (at {best_overall[0]}-bit)")
    print(f"  CODED VALIDATION GATED : {report['coded_validation_gated']} "
          f"(needs >= {WEAK_BELOW_PERCENT}%)")
    print(f"  FULL DAVIS GATED       : {report['full_davis_gated']}")
    print(f"  WORTH KEEPING          : {'yes' if report['coded_validation_gated'] else 'no'}")
    print(f"\n  CLASSIFICATION: {classification}")

    path = args.output_dir / "m20_analysis.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
