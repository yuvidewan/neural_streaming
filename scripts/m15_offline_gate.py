"""M15 Phase C/D/E - the central experiment. For each candidate calibration
POLICY (see m15_calibration_policy.py) and each rate point, fit intra and
motion entropy tables under the FROZEN quantizer/motion estimator (the
quantizer itself always comes from Policy A / the deployed
max_frames=400 recipe - only the ENTROPY TABLE varies by policy, exactly
M13/M14's own frozen-quantizer/fixed-symbol invariant), then measure
held-out bits/symbol on the SAME VAL-A/VAL-B split M13/M14 used.

Two comparisons are reported for every policy, because they answer
different questions:
  vs_policy_a   - the ROOT-CAUSE question: does this policy beat the
                  CURRENT (plain sequential, same total budget where
                  applicable) policy - measured through the identical
                  collector so only the sequence selection differs.
  vs_deployed_c - the INCREMENTAL question (M15's Phase G): does this
                  policy beat what M14 actually shipped? For motion,
                  policy C (8 frames/sequence, 576 total) IS M14's deployed
                  recipe exactly, so this column is the "is there more
                  room beyond current production" answer. For intra, M14
                  shipped no recalibration at all - policy A's own table
                  IS what's deployed - so vs_policy_a and vs_deployed_c
                  are identical for intra by construction.

Phase D (coverage statistics) and Phase E (per-VAL-B-sequence breakdown)
are computed in the same pass, since they reuse the same fitted tables.

Run:
  ./.venv/Scripts/python.exe scripts/m15_offline_gate.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m15_calibration_policy")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
RATE_POINTS = (5, 4, 3)

WEAK_BELOW_PERCENT = 0.5
MEANINGFUL_ABOVE_PERCENT = 1.0


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percent(baseline: float, value: float) -> float:
    return (baseline - value) / baseline * 100.0


def _verdict(gain: float) -> str:
    return ("meaningful" if gain >= MEANINGFUL_ABOVE_PERCENT else
           "marginal" if gain >= WEAK_BELOW_PERCENT else "weak")


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M15 Phase C/D/E: controlled offline comparison of calibration policies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--val-frames-per-sequence", type=int, default=40)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    mc = _load_script("m10h_motion_compensation")
    m14 = _load_script("m14_recalibration")
    m15cal = _load_script("m15_calibration_policy")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    train_sequences = discover_sequences(args.manifest, split="train")
    val_sequences = discover_sequences(args.manifest, split="val",
                                       max_frames_per_sequence=args.val_frames_per_sequence)
    val_a = val_sequences[0::2]
    val_b = val_sequences[1::2]

    print("=" * 122, flush=True)
    print("M15 PHASE C/D/E - CONTROLLED OFFLINE CALIBRATION-POLICY COMPARISON")
    print("=" * 122)
    print(f"  TRAIN: {len(train_sequences)} sequences, {sum(s.frame_count for s in train_sequences)} frames")
    print(f"  VAL-A: {[s.sequence_id for s in val_a]}")
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "M15 Phase C/D/E offline gate", "policies": {},
        "weak_below_percent": WEAK_BELOW_PERCENT, "meaningful_above_percent": MEANINGFUL_ABOVE_PERCENT,
        "val_a_sequence_ids": [s.sequence_id for s in val_a],
        "val_b_sequence_ids": [s.sequence_id for s in val_b],
        "rate_points": [],
    }

    # --- policy allocation + Phase D coverage statistics (bit-depth independent) -----------
    policy_sequences = {name: m15cal.build_policy(name, train_sequences, seed=args.seed)
                        for name in m15cal.POLICY_NAMES}
    for name, seqs in policy_sequences.items():
        coverage = m15cal.coverage_statistics(seqs, train_sequences)
        report["policies"][name] = {"coverage": coverage}
        print(f"\n  policy {name}: {coverage['sequences_represented']}/{coverage['total_train_sequences']} "
             f"sequences ({coverage['percent_sequences_covered']:.1f}%), "
             f"{coverage['total_frames_selected']} frames "
             f"(min {coverage['min_frames_per_represented_sequence']}, "
             f"max {coverage['max_frames_per_represented_sequence']}, "
             f"mean {coverage['mean_frames_per_represented_sequence']:.2f}, "
             f"stdev {coverage['stdev_frames_per_represented_sequence']:.2f})", flush=True)

    # --- motion: bit-depth independent (see module docstring / Phase A q7) - fit ONCE ------
    print("\n  fitting motion tables (bit-depth independent, computed once per policy)...", flush=True)
    motion_bits = mc.motion_alphabet_bits(args.search_range)
    policy_motion = {}
    for name, seqs in policy_sequences.items():
        started = time.perf_counter()
        symbols = m14.collect_motion_symbols(
            mc, model, seqs, block_size=args.block_size, search_range=args.search_range,
            gop_size=args.gop, max_frames=10 ** 9, reference_mode="mc", device=device)
        policy_motion[name] = m14.fit_empirical(symbols, bits=motion_bits, num_tables=2)
        report["policies"][name]["motion_p_frames"] = int(symbols.shape[0])
        report["policies"][name]["motion_identity"] = policy_motion[name].model_id().hex()
        print(f"    {name}: {symbols.shape[0]} P-frames, identity "
             f"{policy_motion[name].model_id().hex()}  ({time.perf_counter() - started:.1f}s)", flush=True)

    val_a_motion = m14.collect_motion_symbols(
        mc, model, val_a, block_size=args.block_size, search_range=args.search_range,
        gop_size=args.gop, max_frames=10 ** 9, reference_mode="mc", device=device)
    val_b_motion = m14.collect_motion_symbols(
        mc, model, val_b, block_size=args.block_size, search_range=args.search_range,
        gop_size=args.gop, max_frames=10 ** 9, reference_mode="mc", device=device)
    a_mtable = m14.motion_table_index(val_a_motion.shape[2], val_a_motion.shape[0])
    b_mtable = m14.motion_table_index(val_b_motion.shape[2], val_b_motion.shape[0])

    deployed_motion_name = "C_broad_576"  # M14's actual shipped recipe: 8 frames/sequence
    root_cause_motion_name = "A_sequential_400"  # the plain-sequential comparator

    motion_report: dict[str, Any] = {}
    for name in m15cal.POLICY_NAMES:
        h_a = m14.held_out_bits_per_symbol(policy_motion[name], val_a_motion, a_mtable)
        h_b = m14.held_out_bits_per_symbol(policy_motion[name], val_b_motion, b_mtable)
        motion_report[name] = {"val_a_h_bits_per_symbol": h_a, "val_b_h_bits_per_symbol": h_b}
    for name in m15cal.POLICY_NAMES:
        row = motion_report[name]
        row["gain_vs_policy_a_percent"] = _percent(
            motion_report[root_cause_motion_name]["val_b_h_bits_per_symbol"], row["val_b_h_bits_per_symbol"])
        row["gain_vs_deployed_c_percent"] = _percent(
            motion_report[deployed_motion_name]["val_b_h_bits_per_symbol"], row["val_b_h_bits_per_symbol"])
        row["verdict_vs_policy_a"] = _verdict(row["gain_vs_policy_a_percent"])
        row["verdict_vs_deployed_c"] = _verdict(row["gain_vs_deployed_c_percent"])
        print(f"    motion {name:<24} VAL-B H={row['val_b_h_bits_per_symbol']:.5f}  "
             f"vs A {row['gain_vs_policy_a_percent']:+.4f}% ({row['verdict_vs_policy_a']})  "
             f"vs deployed-C {row['gain_vs_deployed_c_percent']:+.4f}% "
             f"({row['verdict_vs_deployed_c']})", flush=True)
    report["motion"] = motion_report

    # --- Phase E: per-VAL-B-sequence motion breakdown ---------------------------------------
    print("\n  per-VAL-B-sequence motion breakdown...", flush=True)
    per_sequence_motion = []
    for sequence in val_b:
        symbols = m14.collect_motion_symbols(
            mc, model, [sequence], block_size=args.block_size, search_range=args.search_range,
            gop_size=args.gop, max_frames=10 ** 9, reference_mode="mc", device=device)
        table = m14.motion_table_index(symbols.shape[2], symbols.shape[0])
        row = {"sequence_id": sequence.sequence_id, "p_frames": int(symbols.shape[0])}
        for name in m15cal.POLICY_NAMES:
            row[f"h_{name}"] = m14.held_out_bits_per_symbol(policy_motion[name], symbols, table)
        row["gain_deployed_c_vs_a_percent"] = _percent(row[f"h_{root_cause_motion_name}"],
                                                        row[f"h_{deployed_motion_name}"])
        per_sequence_motion.append(row)
        print(f"    {sequence.sequence_id:<18} " +
             "  ".join(f"{name.split('_')[0]}={row[f'h_{name}']:.4f}" for name in m15cal.POLICY_NAMES),
             flush=True)
    report["per_sequence_motion_val_b"] = per_sequence_motion

    # --- per-bit-depth: intra (bit-depth dependent quantizer) -------------------------------
    for bits in args.rate_points:
        print(f"\n  ---- {bits}-bit (intra) ----", flush=True)
        calibration = mc.calibrate_grids(
            model, train_sequences, bits=bits, mode="per_channel", gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, reference_mode="mc",
            max_frames=args.calibration_frames)
        intra_params = calibration["intra_params"]
        deployed_current_intra = calibration["intra_entropy_model"]

        val_a_intra = m14.collect_intra_symbols(model, val_a, intra_params=intra_params,
                                                max_frames=10 ** 9, device=device)
        val_b_intra = m14.collect_intra_symbols(model, val_b, intra_params=intra_params,
                                                max_frames=10 ** 9, device=device)
        channels, plane = val_a_intra.shape[1], val_a_intra.shape[2]
        a_table = m14.intra_table_index(channels, plane, val_a_intra.shape[0])
        b_table = m14.intra_table_index(channels, plane, val_b_intra.shape[0])

        policy_intra = {}
        for name, seqs in policy_sequences.items():
            symbols = m14.collect_intra_symbols(model, seqs, intra_params=intra_params,
                                                max_frames=10 ** 9, device=device)
            policy_intra[name] = m14.fit_empirical(symbols, bits=bits, num_tables=channels)

        policy_a_matches_calibrate_grids = (
            policy_intra["A_sequential_400"].model_id() == deployed_current_intra.model_id())
        print(f"    Policy-A-via-collector matches calibrate_grids's own intra table: "
             f"{policy_a_matches_calibrate_grids}", flush=True)

        rate_point: dict[str, Any] = {
            "bits": bits, "deployed_current_intra_identity": deployed_current_intra.model_id().hex(),
            "policy_a_matches_calibrate_grids_intra": policy_a_matches_calibrate_grids,
            "intra": {},
        }
        for name in m15cal.POLICY_NAMES:
            h_a = m14.held_out_bits_per_symbol(policy_intra[name], val_a_intra, a_table)
            h_b = m14.held_out_bits_per_symbol(policy_intra[name], val_b_intra, b_table)
            rate_point["intra"][name] = {"val_a_h_bits_per_symbol": h_a, "val_b_h_bits_per_symbol": h_b,
                                         "identity": policy_intra[name].model_id().hex()}
        for name in m15cal.POLICY_NAMES:
            row = rate_point["intra"][name]
            # Intra was never recalibrated in production - policy A IS what's deployed -
            # so "vs deployed" and "vs policy A" are the same comparison for intra.
            row["gain_vs_policy_a_percent"] = _percent(
                rate_point["intra"]["A_sequential_400"]["val_b_h_bits_per_symbol"],
                row["val_b_h_bits_per_symbol"])
            row["verdict_vs_policy_a"] = _verdict(row["gain_vs_policy_a_percent"])
            print(f"    intra  {name:<24} VAL-B H={row['val_b_h_bits_per_symbol']:.5f}  "
                 f"vs A {row['gain_vs_policy_a_percent']:+.4f}% ({row['verdict_vs_policy_a']})", flush=True)
        report["rate_points"].append(rate_point)

    print()
    print("=" * 122)
    print("SUMMARY")
    print("=" * 122)
    for name in m15cal.POLICY_NAMES:
        if name == root_cause_motion_name:
            continue
        print(f"  motion {name:<24} vs A {report['motion'][name]['gain_vs_policy_a_percent']:+.4f}% "
             f"({report['motion'][name]['verdict_vs_policy_a']})  "
             f"vs deployed-C {report['motion'][name]['gain_vs_deployed_c_percent']:+.4f}% "
             f"({report['motion'][name]['verdict_vs_deployed_c']})")
    for rp in report["rate_points"]:
        for name in m15cal.POLICY_NAMES:
            if name == "A_sequential_400":
                continue
            row = rp["intra"][name]
            print(f"  {rp['bits']}-bit intra {name:<24} vs A {row['gain_vs_policy_a_percent']:+.4f}% "
                 f"({row['verdict_vs_policy_a']})")

    path = args.output_dir / "m15_offline_gate.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
