"""docs/PLAN_audit2.md: collect every check's JSON into results.json, applying the rules fixed in
the plan (and its two C2 amendments). Run from the repo root after all audit runs:
    .venv/Scripts/python outputs/p12_audit2/report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import benchmark_parity as bp  # noqa: E402

O = Path("outputs/p12_audit2")
C = bp.PRIMARY_CONVENTION


def load(name):
    p = O / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def curve(report, key="psnr"):
    return sorted((p["bpp"], p["quality"][C][key]) for p in report["points"].values())


def main() -> int:
    r = {}
    enc, dec, cpu = load("a_encode.json"), load("a_decode_cuda.json"), load("a_decode_cpu.json")
    if enc and dec:
        r["A1_fresh_process_decode"] = {"streams": len(dec), "identical": sum(v["same"] for v in dec.values()),
                                        "ok": all(v["same"] for v in dec.values()) and len(dec) == len(enc)}
        share = {k: e["bytes"]["unused_by_wide_i_frames"] / e["bytes"]["file"] for k, e in enc.items()}
        r["A3_byte_accounting"] = {
            "adds_up": all(e["bytes"]["adds_up"] for e in enc.values()),
            "records_match_payloads": all(e["bytes"]["records_match_payloads"] for e in enc.values()),
            "unused_quant_block_bytes_per_stream": next(iter(enc.values()))["bytes"]["unused_by_wide_i_frames"],
            "unused_share_max": round(max(share.values()), 5),
            "unused_share_of_smallest_stream": min(enc.items(), key=lambda kv: kv[1]["bytes"]["file"])[0]}
    wrong = load("a_wrong.json")
    if wrong:
        cases = {k: v for k, v in wrong.items() if isinstance(v, dict)}
        control = cases.pop("right coders (control)")
        r["A2_wrong_models"] = {"control_decodes": control.get("raised") is None,
                                "all_wrong_cases_raise": all(v.get("raised") for v in cases.values()),
                                "cases": cases}
    a4 = load("a4_range_coder.json")
    if a4:
        r["A4_range_coder"] = {"ok": a4["all_ok"], "random_1000": a4["random_1000"]}
    dmg = load("a_damage.json")
    if dmg:
        outcomes = {}
        for v in dmg.values():
            key = v["outcome"] if v["outcome"] != "returned frames" else (
                "returned frames, identical" if v.get("identical_to_original") else "returned frames, different")
            outcomes[key] = outcomes.get(key, 0) + 1
        r["A5_damaged_streams"] = {"outcomes": outcomes,
                                   "no_hang_or_crash": not any(k.startswith(("hang", "process died")) for k in outcomes)}
    if cpu:
        r["A6_cpu_decode"] = {"streams": len(cpu), "identical": sum(v["same"] for v in cpu.values())}
    b1 = load("b1_lossless.json")
    if b1 and "summary" in b1:
        r["B1_lossless_floor"] = {"summary": b1["summary"], "material": b1["material"],
                                  "conversion": b1["conversion"]}
    hl = load("d1_b2_headlines.json")
    if hl:
        r["D1_headlines"] = {"checked": len(hl["headlines"]), "mismatches": hl["d1_mismatches"]}
        r["B2_bd_method"] = {"findings": [h for h in hl["headlines"] if h["b2_finding"]]}
    b3 = load("b3_anchor_settings.json")
    if b3:
        structure = {}
        for name, s in b3["structure"].items():
            if s["lowdelay"]:
                ok = s["i_positions"] == list(range(0, s["frames"], 30)) and s["b_frames"] == 0
            else:
                ok = None   # encoder default GOP and B-frames, by design
            structure[name] = {"i_positions": s["i_positions"], "b_frames": s["b_frames"], "as_designed": ok}
        r["B3_anchor_settings"] = {"structure": structure, "fps": b3["fps"]}
    b4 = load("b4_same_frames.json")
    if b4:
        r["B4_same_frames"] = {k: v for k, v in b4.items() if k != "scoring"}
    c1 = load("c1_results.json")
    if c1:
        r["C1_vimeo_overlap"] = {k: c1[k] for k in ("threshold", "test_frames", "vimeo_frames",
                                                     "candidates", "candidates_by_video")}
        r["C1_vimeo_overlap"]["nearest_distance_by_video"] = {k: v["distance"] for k, v in c1["nearest"].items()}
    c3 = load("c3_splits.json")
    if c3:
        r["C3_splits"] = {"disjoint": c3["disjoint"], "foreign_in_folders": {k: v["foreign"] for k, v in c3["folders"].items()},
                          "external_names_in_manifests": c3["external_names_in_manifests"],
                          "davis_near_duplicate_pairs": c3["davis_near_duplicates"]["pairs"]}
    old, wide = load("c2_valb_old.json"), load("c2_valb_wide.json")
    if old and wide:
        bd = bp.bd_rate(curve(old), curve(wide))
        r["C2_repick_i_frames"] = {"bd_wide_vs_old_valb": round(bd, 2), "valid": old["valid"] and wide["valid"],
                                   "verdict": "wide confirmed" if bd <= 0.5 else "OLD WINS (material)"}
    gops = {g: load(f"c2_valb_gop{g}.json") for g in (10, 20, 30)}
    if all(gops.values()):
        bds = {g: round(bp.bd_rate(curve(gops[30]), curve(gops[g])), 2) for g in (10, 20)}
        best = min(bds, key=bds.get)
        r["C2_repick_gop"] = {"bd_vs_gop30_valb": bds, "valid": all(g["valid"] for g in gops.values()),
                              "verdict": "GOP 30 confirmed" if bds[best] >= -0.5 else f"GOP {best} WINS (material)"}
    d3 = load("d3_commands.json")
    if d3:
        r["D3_commands"] = d3
    (O / "results.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
