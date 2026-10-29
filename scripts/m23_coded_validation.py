"""M23 Phase B2 step 1 - does Phase B1's ideal-bits gain realise in REAL CODED
BYTES, and is the result decodable?

Phase B1 measured +1.3999 / +1.7691 / +1.7889% of ideal bits at 5/4/3-bit
against the full deployed stack, with the transmitted map charged. Ideal bits are
not bytes, and a context the decoder cannot rebuild is not a codec. This script
settles both with the real range coder.

WHAT MAKES THIS A DECODABILITY PROOF, NOT JUST A BYTE COUNT
-----------------------------------------------------------
The candidate arm codes the scale map first, then **decodes it back from its own
payload bytes**, and the residual's table indices are derived from the DECODED
map - never from the encoder's copy. So the only thing standing between this
measurement and a working codec is container plumbing, which is deliberately out
of scope here (see below). If the decode path needed anything the encoder has and
the decoder does not, this would fail rather than quietly report a gain.

Both arms round-trip: every frame's symbols are decoded back from the actual
payload and compared to what went in. A byte count from a stream that does not
decode is worthless, so exactness is checked rather than assumed.

THE ARMS
--------
  baseline    the full deployed stack - M11-G16's K=512 prototypes with M13's
              recalibrated frequencies - coded for real. 512 tables, selected by
              prototype index k.
  candidate   the same, conditioned additionally on the transmitted scale map:
              512 x 8 = 4096 joint tables, selected by (k, ctx). Plus the map's
              own payload, coded with its own table and counted in full.
  control     the candidate with the map permuted across the whole split. Kept
              because it costs almost nothing and it is the arm that proves the
              bytes are bought by content and not by table count.

Hierarchical smoothing, strengths, bin edges and the TRAIN/VAL-A/VAL-B split are
all exactly Phase B1's, so a difference between the two reports is realisation
and nothing else.

WHY NO CONTAINER CHANGE YET
---------------------------
`.nvct` v2's per-frame record is a fixed `uint8 frame_type, uint32 motion_length,
uint32 residual_length`, so carrying a third payload needs a format version. That
is a real change to a frozen item and it should be spent only on a gain that is
known to realise - which is what this script measures first. Step 2 is the v3
record, with v2 reading preserved and tested.

No `src/nvc/` module is modified: the range coder already accepts an arbitrary
per-symbol `table_index`, which is what makes 4096 joint tables codeable without
touching it.

TEST is never opened. Run:
  ./.venv/Scripts/python.exe scripts/m23_coded_validation.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.compression.range_coder import decode_symbols, encode_symbols
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

ARMS = ("baseline", "candidate", "control")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percent(baseline: float, value: float) -> float:
    return 100.0 * (baseline - value) / baseline


# --------------------------------------------------------------------------------------------
# Float distributions -> coder tables
# --------------------------------------------------------------------------------------------


def smoothed_child_probabilities(child_counts: np.ndarray, parent_probabilities: np.ndarray,
                                 *, card: int, strength: float) -> np.ndarray:
    """The same P(s | k, ctx) `m11_causal_context.child_bits` scores, materialised.

    `child_bits` computes the smoothed probability per SYMBOL to avoid building an
    [N, alphabet] array. The coder needs the table itself, so it is built here
    from the identical expression - counts plus `strength` times the parent,
    normalised by the row total plus `strength` - and nothing else, so the bytes
    below are the bytes of the distribution Phase B1 measured.
    """
    groups = child_counts.shape[0] // card
    parent_rows = np.repeat(parent_probabilities, card, axis=0)
    numerator = child_counts + strength * parent_rows
    denominator = child_counts.sum(axis=1, keepdims=True) + strength
    probabilities = numerator / denominator
    if probabilities.shape[0] != groups * card:
        raise ValueError("child table shape does not match groups x card")
    return probabilities


def coder_tables(m10k, probabilities: np.ndarray, *, bits: int) -> EmpiricalEntropyModel:
    """Integer coder tables from float probabilities, by M10K's own deterministic
    conversion - the one every learned table in this codec already goes through,
    so encoder and decoder agree on any machine."""
    frequencies = m10k.probabilities_to_frequencies(probabilities)
    return EmpiricalEntropyModel(frequencies, bits=bits)


def map_coder_table(train_binned: np.ndarray, *, bins: int) -> EmpiricalEntropyModel:
    """One table for the scale map, fitted on TRAIN only."""
    bit_width = int(np.log2(bins))
    return EmpiricalEntropyModel.from_symbols(
        train_binned.reshape(train_binned.shape[0], 1, -1), bits=bit_width, num_tables=1)


# --------------------------------------------------------------------------------------------
# Coding one frame, each arm
# --------------------------------------------------------------------------------------------


def code_baseline_frame(symbols: np.ndarray, k: np.ndarray,
                        table: EmpiricalEntropyModel) -> dict[str, Any]:
    """The deployed stack: table per prototype, nothing transmitted."""
    payload = encode_symbols(symbols, table.cumulative, k)
    decoded = decode_symbols(payload, symbols.size, table.cumulative, k)
    return {"residual_bytes": len(payload), "map_bytes": 0,
            "total_bytes": len(payload), "exact": bool(np.array_equal(decoded, symbols))}


def code_candidate_frame(symbols: np.ndarray, k: np.ndarray, binned_map: np.ndarray, *,
                         child_table: EmpiricalEntropyModel,
                         map_table: EmpiricalEntropyModel, bins: int, channels: int,
                         height: int, width: int, broadcast) -> dict[str, Any]:
    """Transmit the map, DECODE it back, then code the residual against the
    decoded map.

    The order is the point. The map's payload is produced, decoded, and only the
    decoded values are allowed to influence the residual's table indices, so this
    reproduces what a decoder can actually do rather than what an encoder happens
    to know.
    """
    flat_map = binned_map.reshape(-1)
    map_index = np.zeros(flat_map.size, dtype=np.int64)
    map_payload = encode_symbols(flat_map, map_table.cumulative, map_index)
    decoded_map = decode_symbols(map_payload, flat_map.size, map_table.cumulative, map_index)
    map_exact = bool(np.array_equal(decoded_map, flat_map))

    # Everything below uses `decoded_map`, never `binned_map`.
    context = broadcast(decoded_map.reshape(1, *binned_map.shape), channels=channels,
                        height=height, width=width).reshape(-1)
    joint = k * bins + context
    payload = encode_symbols(symbols, child_table.cumulative, joint)
    decoded = decode_symbols(payload, symbols.size, child_table.cumulative, joint)
    return {"residual_bytes": len(payload), "map_bytes": len(map_payload),
            "total_bytes": len(payload) + len(map_payload),
            "exact": bool(np.array_equal(decoded, symbols)) and map_exact,
            "map_exact": map_exact}


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    m23 = _load_script("m23_hyperprior")
    pb = _load_script("m23_phaseb_gate")
    parser = argparse.ArgumentParser(
        description="M23 Phase B2 step 1: does B1's gain realise in real coded bytes?",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=m23.DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=m23.DEFAULT_M11_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m23_hyperprior"))
    parser.add_argument("--output-name", type=str, default="m23_coded_validation.json")
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(m23.RATE_POINTS))
    parser.add_argument("--context-bins", type=int, default=pb.DEFAULT_CONTEXT_BINS)
    parser.add_argument("--context-stride", type=int, default=pb.DEFAULT_CONTEXT_STRIDE)
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    m23 = _load_script("m23_hyperprior")
    pb = _load_script("m23_phaseb_gate")
    gate_a = _load_script("m23_offline_gate")
    md = _load_script("m11_data")
    cx = _load_script("m11_causal_context")
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    m12 = _load_script("m12_spatial_offline_gate")
    m10k = _load_script("m10k_learned_entropy")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / args.output_name
    configuration = {
        "checkpoint": str(args.checkpoint), "m11_dir": str(args.m11_dir),
        "rate_points": list(args.rate_points), "arms": list(args.arms),
        "context_bins": args.context_bins, "context_stride": args.context_stride,
        "calibration_frames": args.calibration_frames, "seed": args.seed,
    }
    report: dict[str, Any] = {
        "phase": "M23 Phase B2 step 1 - coded validation of the transmitted scale map",
        "question": ("B1 measured +1.3999/+1.7691/+1.7889% of IDEAL BITS against the full "
                     "deployed stack. Does it realise in real coded bytes, and is the "
                     "context rebuildable from the bitstream alone?"),
        "decodability": ("the candidate codes the map, DECODES it back from its own payload, "
                         "and derives the residual's table indices from the decoded values "
                         "only - never from the encoder's copy"),
        "out_of_scope": (".nvct v2's per-frame record is a fixed (frame_type, motion_length, "
                         "residual_length), so carrying a third payload needs a format "
                         "version. That is step 2, and it should only be spent on a gain "
                         "known to realise."),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "pre_registered": {
            "arms": list(ARMS),
            "baseline": "M11-G16 K=512 + M13 recalibrated frequencies, coded for real",
            "primary_metric": "real coded bytes on VAL-B, map payload included",
            "weak_below_percent": m23.WEAK_BELOW_PERCENT,
            "meaningful_above_percent": m23.MEANINGFUL_ABOVE_PERCENT,
            "net_of_control_rule": "gain - max(control, 0) (M11's rule)",
            "round_trip_required": True,
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": str(device), "platform": platform.platform()},
        "rate_points": [],
    }
    if report_path.is_file() and args.resume:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("configuration") == configuration:
            report["rate_points"] = previous.get("rate_points", [])
            print(f"[resume] {len(report['rate_points'])} rate points already measured",
                  flush=True)

    def persist() -> None:
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    done = {entry["bits"] for entry in report["rate_points"]}
    b1_path = args.output_dir / "m23_phaseb_gate.json"
    b1 = (json.loads(b1_path.read_text(encoding="utf-8")) if b1_path.is_file() else None)

    from nvc.training.checkpoint import load_model_from_checkpoint

    print("=" * 118)
    print("M23 PHASE B2 STEP 1 - CODED VALIDATION (VAL-B, real range coder; TEST never "
          "opened)")
    print(f"  arms: {args.arms}   map: {args.context_bins} bins at stride "
          f"{args.context_stride}, transmitted and DECODED before the residual")
    print("=" * 118, flush=True)

    for bits in args.rate_points:
        if bits in done:
            print(f"  [resume] {bits}-bit already measured", flush=True)
            continue
        alphabet = 2 ** bits
        print(f"\n  ---- {bits}-bit ----", flush=True)
        base_model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
        base_model.eval()
        data = md.load_or_collect(
            base_model, checkpoint=args.checkpoint, manifest=args.manifest, bits=bits,
            device=device, calibration_frames=args.calibration_frames, cache_dir=cache_dir,
            log=lambda m: print("    " + m, flush=True))
        train_symbols = data["train_symbols"].astype(np.int64)
        val_symbols = data["val_symbols"].astype(np.int64)
        select_mask, report_mask = md.split_validation(data)
        channels, height, width = train_symbols.shape[1:]
        residual_params = data["calibration"]["residual_params"]
        zero = torch.from_numpy(cx.zero_symbols(residual_params, channels))

        model, _ = ma.load_model(args.m11_dir / f"m11_G16_entropy_{bits}bit.pt", device=device)
        codebook = ml.SharedCodebook.from_dict(json.loads(
            (args.m11_dir / f"m11_G16_codebook_{bits}bit_K512.json").read_text(
                encoding="utf-8")))
        train_k = m12.m11_g16_prototype_indices(model, codebook, train_symbols,
                                                data["train_references"], zero, device=device)
        val_k = m12.m11_g16_prototype_indices(model, codebook, val_symbols,
                                              data["val_references"], zero, device=device)

        flat = lambda a: a.reshape(-1)
        t_sym, t_k = flat(train_symbols), flat(train_k)
        a_sym, a_k = flat(val_symbols[select_mask]), flat(val_k[select_mask])

        # ---- the deployed parent (M13's mechanism), exactly as B1 built it ----------
        groups = codebook.size
        q = codebook.probabilities
        parent_counts = cx.fit_parent(t_sym, t_k, groups, alphabet)
        strength_parent, _ = cx.select_strength(lambda s: cx.parent_bits(
            cx.smoothed_parent(parent_counts, q, s), a_sym, a_k))
        parent = cx.smoothed_parent(parent_counts, q, strength_parent)
        parent_table = coder_tables(m10k, parent, bits=bits)

        # ---- the transmitted map, exactly as B1 built it -----------------------------
        train_residual = m23.dequantized_residual(
            train_symbols, residual_params, shape=(channels, height, width))
        val_residual = m23.dequantized_residual(
            val_symbols, residual_params, shape=(channels, height, width))
        train_coarse = pb.coarsen(pb.scale_map_signal(train_residual),
                                  stride=args.context_stride)
        val_coarse = pb.coarsen(pb.scale_map_signal(val_residual), stride=args.context_stride)
        edges = pb.train_bin_edges(train_coarse, bins=args.context_bins)
        train_binned = pb.to_bins(train_coarse, edges)
        val_binned = pb.to_bins(val_coarse, edges)
        map_table = map_coder_table(train_binned, bins=args.context_bins)

        card = args.context_bins
        t_ctx = flat(pb.broadcast_to_symbol_positions(train_binned, channels=channels,
                                                      height=height, width=width))
        v_ctx_frames = pb.broadcast_to_symbol_positions(val_binned, channels=channels,
                                                        height=height, width=width)
        child_counts = cx.fit_child(t_sym, t_k, t_ctx, card, groups, alphabet)
        a_ctx = flat(v_ctx_frames[select_mask])
        side_a = pb.context_map_bits(train_binned, val_binned[select_mask],
                                    bins=card) / a_sym.size
        strength_child, _ = cx.select_strength(lambda s: cx.child_bits(
            child_counts, parent, a_sym, a_k, a_ctx, card, s) / a_sym.size + side_a)
        child = smoothed_child_probabilities(child_counts, parent, card=card,
                                             strength=strength_child)
        child_table = coder_tables(m10k, child, bits=bits)
        print(f"    tables: parent {parent_table.frequencies.shape} (strength "
              f"{strength_parent:g})   child {child_table.frequencies.shape} (strength "
              f"{strength_child:g})   map {map_table.frequencies.shape}", flush=True)

        # the control's map, permuted across the WHOLE report split
        rng = np.random.default_rng(args.seed)
        permuted_binned = val_binned.copy()
        report_indices = np.flatnonzero(report_mask)
        order = m23.permuted_hyper_signal(report_indices.size, seed=args.seed)
        permuted_binned[report_indices] = val_binned[report_indices[order]]

        # ---- code every VAL-B frame, every arm ---------------------------------------
        totals = {arm: {"residual_bytes": 0, "map_bytes": 0, "total_bytes": 0,
                        "exact": True, "frames": 0} for arm in args.arms}
        started = time.perf_counter()
        for index in report_indices:
            symbols = val_symbols[index].reshape(-1)
            k = val_k[index].reshape(-1)
            if "baseline" in args.arms:
                row = code_baseline_frame(symbols, k, parent_table)
                for key in ("residual_bytes", "map_bytes", "total_bytes"):
                    totals["baseline"][key] += row[key]
                totals["baseline"]["exact"] &= row["exact"]
                totals["baseline"]["frames"] += 1
            for arm, source in (("candidate", val_binned), ("control", permuted_binned)):
                if arm not in args.arms:
                    continue
                row = code_candidate_frame(
                    symbols, k, source[index], child_table=child_table,
                    map_table=map_table, bins=card, channels=channels, height=height,
                    width=width, broadcast=pb.broadcast_to_symbol_positions)
                for key in ("residual_bytes", "map_bytes", "total_bytes"):
                    totals[arm][key] += row[key]
                totals[arm]["exact"] &= row["exact"]
                totals[arm]["frames"] += 1
        elapsed = time.perf_counter() - started

        symbols_scored = int(val_symbols[report_mask].size)
        for arm, row in totals.items():
            row["bits_per_symbol"] = row["total_bytes"] * 8.0 / symbols_scored
            row["map_share_of_total"] = (row["map_bytes"] / row["total_bytes"]
                                         if row["total_bytes"] else 0.0)

        base_bytes = totals.get("baseline", {}).get("total_bytes")
        entry: dict[str, Any] = {
            "bits": bits, "alphabet": alphabet, "groups": groups, "card": card,
            "val_b_p_frames": int(report_mask.sum()), "symbols_scored": symbols_scored,
            "strength_parent": strength_parent, "strength_child": strength_child,
            "arms": totals, "coding_seconds": elapsed,
        }
        if base_bytes:
            for arm in ("candidate", "control"):
                if arm in totals:
                    totals[arm]["gain_percent_vs_baseline"] = _percent(
                        base_bytes, totals[arm]["total_bytes"])
            if "candidate" in totals and "control" in totals:
                entry["net_gain_percent"] = gate_a.net_of_control(
                    totals["candidate"]["gain_percent_vs_baseline"],
                    totals["control"]["gain_percent_vs_baseline"])
                entry["control_interpretation"] = gate_a.control_interpretation(
                    totals["control"]["gain_percent_vs_baseline"])
            elif "candidate" in totals:
                entry["net_gain_percent"] = totals["candidate"]["gain_percent_vs_baseline"]
            entry["verdict"] = m23.verdict(entry["net_gain_percent"])

            # realisation against B1's ideal-bits prediction for the same arm
            if b1 is not None:
                b1_entry = next((e for e in b1["rate_points"] if e["bits"] == bits), None)
                if b1_entry is not None:
                    row = next((r for r in b1_entry["candidates"]
                                if r["arm"] == "scale_map"), None)
                    if row is not None:
                        predicted = row["net_gain_percent"]
                        entry["b1_ideal_bits_gain_percent"] = predicted
                        entry["realisation_ratio"] = (entry["net_gain_percent"] / predicted
                                                      if predicted else None)

        report["rate_points"].append(entry)
        persist()

        for arm in args.arms:
            row = totals[arm]
            gain = row.get("gain_percent_vs_baseline")
            print(f"    {arm:<10} {row['total_bytes']:>9,} B  "
                  f"(residual {row['residual_bytes']:>9,} + map {row['map_bytes']:>7,})  "
                  f"{row['bits_per_symbol']:.5f} b/sym  "
                  f"map {100 * row['map_share_of_total']:.2f}%  "
                  f"exact={row['exact']}"
                  + (f"  gain {gain:+.4f}%" if gain is not None else ""), flush=True)
        if "net_gain_percent" in entry:
            print(f"    NET {entry['net_gain_percent']:+.4f}%  {entry['verdict']}"
                  + (f"   B1 predicted {entry['b1_ideal_bits_gain_percent']:+.4f}% "
                     f"-> realisation {100 * entry['realisation_ratio']:.1f}%"
                     if "realisation_ratio" in entry else ""), flush=True)

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["all_round_trips_exact"] = all(
        row["exact"] for entry in report["rate_points"] for row in entry["arms"].values())
    persist()

    print("\n" + "=" * 118)
    print("PHASE B2 STEP 1 - REAL CODED BYTES ON VAL-B")
    print("=" * 118)
    print(f"{'bits':>5} {'baseline B':>12} {'candidate B':>12} {'map B':>9} {'gain':>10} "
          f"{'control':>10} {'NET':>10} {'B1 ideal':>10} {'realised':>9} {'exact':>6}")
    for entry in sorted(report["rate_points"], key=lambda e: -e["bits"]):
        arms = entry["arms"]
        if "baseline" not in arms or "candidate" not in arms:
            continue
        control = arms.get("control", {}).get("gain_percent_vs_baseline")
        print(f"{entry['bits']:>5} {arms['baseline']['total_bytes']:>12,} "
              f"{arms['candidate']['total_bytes']:>12,} "
              f"{arms['candidate']['map_bytes']:>9,} "
              f"{arms['candidate']['gain_percent_vs_baseline']:>+9.4f}% "
              + (f"{control:>+9.4f}% " if control is not None else " " * 11)
              + f"{entry.get('net_gain_percent', float('nan')):>+9.4f}% "
              f"{entry.get('b1_ideal_bits_gain_percent', float('nan')):>+9.4f}% "
              f"{100 * entry.get('realisation_ratio', float('nan')):>8.1f}% "
              f"{str(all(r['exact'] for r in arms.values())):>6}")
    print(f"\nall round trips exact: {report['all_round_trips_exact']}")
    print(f"Report: {report_path}")
    return 0 if report["all_round_trips_exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
