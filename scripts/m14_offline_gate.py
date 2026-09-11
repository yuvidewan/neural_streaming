"""M14 Phase B - offline calibration-gap measurement for the intra and motion
entropy tables (the two LIVE, not-yet-recalibrated candidates Phase A's
audit found - see outputs/m14_entropy_audit/m14_entropy_audit.json).

For each candidate: the SYMBOL STREAM IS NEVER REGENERATED WITH A MODIFIED
CODEC (per the brief). Both the CURRENT deployed table and the CANDIDATE
recalibrated table are compared on the SAME held-out VAL-B symbols, computed
once, under the FROZEN quantizer/motion estimator. Only the frequency table
differs between H_current and H_empirical - exactly M13's discipline,
generalized to two new tables.

Split discipline (identical rule M11/M12/M13 used - sequence-disjoint, by
INDEX PARITY over the SAME 9 DAVIS validation sequences):
  TRAIN   fit the empirical candidate table
  VAL-A   even-indexed val sequences - consistency check only (no
          hyperparameter to select here: Laplace has no tunable strength,
          and Phase B says keep the same smoothing convention unless an
          ablation proves it unstable - none was found, see the report)
  VAL-B   odd-indexed val sequences - the headline number
  TEST    never read by this script

BROAD vs SEQUENTIAL TRAIN COVERAGE - an audit finding, not an assumption
-------------------------------------------------------------------------
TRAIN has 72 sequences totaling 4,826 frames. `calibrate_grids` (the
DEPLOYED calibration recipe) applies its `max_frames` budget SEQUENTIALLY -
"walk sequences in manifest order, stop once N total frames are seen" - so
at the deployed `max_frames=400` it only ever sees the first ~6 of 72 TRAIN
sequences (~8% of TRAIN's sequences and frame volume), never the other 66.
An early smoke test at this script's own `--train-frames` cap applied the
SAME sequential strategy and found deeply NEGATIVE "gains" (a smaller,
similarly-narrow resample can only look worse than the larger deployed
sample it's compared against - not evidence of anything). This script
therefore samples TRAIN with `max_frames_per_sequence` - a fixed budget
FROM EVERY sequence - so the comparison is "broad coverage vs narrow
coverage" at comparable total volume, not "more of the same narrow slice."
Whether broad coverage actually helps is what Phase B measures, not
assumes; see the report for the answer.

Run:
  ./.venv/Scripts/python.exe scripts/m14_offline_gate.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m14_entropy_audit")
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


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M14 Phase B: offline gap measurement for intra/motion entropy tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--train-frames-per-sequence", type=int, default=8,
                        help="frames sampled from EVERY TRAIN sequence (72 sequences), not a "
                             "global sequential cap - see module docstring")
    parser.add_argument("--val-frames-per-sequence", type=int, default=40)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def _verdict(gain: float) -> str:
    return ("meaningful" if gain >= MEANINGFUL_ABOVE_PERCENT else
           "marginal" if gain >= WEAK_BELOW_PERCENT else "weak")


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    mc = _load_script("m10h_motion_compensation")
    m14 = _load_script("m14_recalibration")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    # TWO separate TRAIN sequence lists, deliberately: `calibration_sequences`
    # (full, untruncated) feeds `calibrate_grids` so its OWN internal
    # sequential max_frames cap behaves EXACTLY as the deployed recipe does
    # (seeing only the first few of 72 sequences) - this is the CURRENT
    # baseline. `broad_train_sequences` (per-sequence-capped) is the NEW
    # candidate's broad-coverage sample. Passing the truncated list to
    # calibrate_grids by mistake would make both collections draw from the
    # same narrow slice, silently erasing the comparison - see the module
    # docstring's "BROAD vs SEQUENTIAL" note for why this distinction matters.
    calibration_sequences = discover_sequences(args.manifest, split="train")
    broad_train_sequences = discover_sequences(
        args.manifest, split="train", max_frames_per_sequence=args.train_frames_per_sequence)
    val_sequences = discover_sequences(args.manifest, split="val",
                                       max_frames_per_sequence=args.val_frames_per_sequence)
    val_a = val_sequences[0::2]
    val_b = val_sequences[1::2]
    train_total_frames = sum(s.frame_count for s in broad_train_sequences)

    print("=" * 116, flush=True)
    print("M14 PHASE B - OFFLINE CALIBRATION-GAP MEASUREMENT (intra, motion)")
    print("=" * 116)
    print(f"  TRAIN: {len(broad_train_sequences)} sequences (ALL of TRAIN), "
         f"{args.train_frames_per_sequence} frames/sequence = {train_total_frames} total frames "
         f"(deployed calibration saw only ~{args.calibration_frames} frames from the first few "
         f"sequences in manifest order)")
    print(f"  VAL-A: {[s.sequence_id for s in val_a]}")
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}")
    print(f"  TEST : never read by this script", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "M14 Phase B offline gate",
        "weak_below_percent": WEAK_BELOW_PERCENT, "meaningful_above_percent": MEANINGFUL_ABOVE_PERCENT,
        "val_a_sequence_ids": [s.sequence_id for s in val_a],
        "val_b_sequence_ids": [s.sequence_id for s in val_b],
        "train_frames_per_sequence": args.train_frames_per_sequence,
        "train_total_frames": train_total_frames, "calibration_frames": args.calibration_frames,
        "rate_points": [],
    }

    for bits in args.rate_points:
        print(f"\n  ---- {bits}-bit ----", flush=True)
        calibration = mc.calibrate_grids(
            model, calibration_sequences, bits=bits, mode="per_channel", gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, reference_mode="mc",
            max_frames=args.calibration_frames)
        current_intra = calibration["intra_entropy_model"]
        current_motion = calibration["motion_entropy_model"]
        intra_params = calibration["intra_params"]

        rate_point: dict[str, Any] = {"bits": bits, "candidates": {}}

        # --- intra --------------------------------------------------------------------
        started = time.perf_counter()
        train_intra = m14.collect_intra_symbols(model, broad_train_sequences, intra_params=intra_params,
                                                max_frames=10 ** 9, device=device)
        val_a_intra = m14.collect_intra_symbols(model, val_a, intra_params=intra_params,
                                                max_frames=10 ** 9, device=device)
        val_b_intra = m14.collect_intra_symbols(model, val_b, intra_params=intra_params,
                                                max_frames=10 ** 9, device=device)
        channels, plane = train_intra.shape[1], train_intra.shape[2]
        new_intra = m14.fit_empirical(train_intra, bits=bits, num_tables=channels)

        a_table = m14.intra_table_index(channels, plane, val_a_intra.shape[0])
        b_table = m14.intra_table_index(channels, plane, val_b_intra.shape[0])
        h_old_a = m14.held_out_bits_per_symbol(current_intra, val_a_intra, a_table)
        h_new_a = m14.held_out_bits_per_symbol(new_intra, val_a_intra, a_table)
        h_old_b = m14.held_out_bits_per_symbol(current_intra, val_b_intra, b_table)
        h_new_b = m14.held_out_bits_per_symbol(new_intra, val_b_intra, b_table)
        gain_a, gain_b = _percent(h_old_a, h_new_a), _percent(h_old_b, h_new_b)
        elapsed = time.perf_counter() - started
        print(f"    intra   TRAIN {train_intra.shape[0]} frames | VAL-A gain {gain_a:+.4f}% | "
             f"VAL-B H_old {h_old_b:.5f} H_new {h_new_b:.5f} gain {gain_b:+.4f}% "
             f"-> {_verdict(gain_b)}  ({elapsed:.1f}s)", flush=True)
        rate_point["candidates"]["intra"] = {
            "train_frames": int(train_intra.shape[0]), "val_a_frames": int(val_a_intra.shape[0]),
            "val_b_frames": int(val_b_intra.shape[0]), "val_a_gain_percent": gain_a,
            "val_b_h_old_bits_per_symbol": h_old_b, "val_b_h_new_bits_per_symbol": h_new_b,
            "val_b_gain_percent": gain_b, "verdict": _verdict(gain_b),
            "current_identity": current_intra.model_id().hex(),
            "new_identity": new_intra.model_id().hex(),
        }

        # --- motion ---------------------------------------------------------------------
        started = time.perf_counter()
        train_motion = m14.collect_motion_symbols(
            mc, model, broad_train_sequences, block_size=args.block_size,
            search_range=args.search_range, gop_size=args.gop, max_frames=10 ** 9,
            reference_mode="mc", device=device)
        val_a_motion = m14.collect_motion_symbols(
            mc, model, val_a, block_size=args.block_size, search_range=args.search_range,
            gop_size=args.gop, max_frames=10 ** 9, reference_mode="mc", device=device)
        val_b_motion = m14.collect_motion_symbols(
            mc, model, val_b, block_size=args.block_size, search_range=args.search_range,
            gop_size=args.gop, max_frames=10 ** 9, reference_mode="mc", device=device)
        motion_bits = mc.motion_alphabet_bits(args.search_range)
        blocks = train_motion.shape[2]
        new_motion = m14.fit_empirical(train_motion, bits=motion_bits, num_tables=2)

        a_mtable = m14.motion_table_index(val_a_motion.shape[2], val_a_motion.shape[0])
        b_mtable = m14.motion_table_index(val_b_motion.shape[2], val_b_motion.shape[0])
        m_old_a = m14.held_out_bits_per_symbol(current_motion, val_a_motion, a_mtable)
        m_new_a = m14.held_out_bits_per_symbol(new_motion, val_a_motion, a_mtable)
        m_old_b = m14.held_out_bits_per_symbol(current_motion, val_b_motion, b_mtable)
        m_new_b = m14.held_out_bits_per_symbol(new_motion, val_b_motion, b_mtable)
        mgain_a, mgain_b = _percent(m_old_a, m_new_a), _percent(m_old_b, m_new_b)
        elapsed = time.perf_counter() - started
        print(f"    motion  TRAIN {train_motion.shape[0]} P-frames | VAL-A gain {mgain_a:+.4f}% | "
             f"VAL-B H_old {m_old_b:.5f} H_new {m_new_b:.5f} gain {mgain_b:+.4f}% "
             f"-> {_verdict(mgain_b)}  ({elapsed:.1f}s)", flush=True)
        rate_point["candidates"]["motion"] = {
            "train_p_frames": int(train_motion.shape[0]), "val_a_p_frames": int(val_a_motion.shape[0]),
            "val_b_p_frames": int(val_b_motion.shape[0]), "val_a_gain_percent": mgain_a,
            "val_b_h_old_bits_per_symbol": m_old_b, "val_b_h_new_bits_per_symbol": m_new_b,
            "val_b_gain_percent": mgain_b, "verdict": _verdict(mgain_b),
            "current_identity": current_motion.model_id().hex(),
            "new_identity": new_motion.model_id().hex(),
        }

        report["rate_points"].append(rate_point)

    print()
    print("=" * 116)
    print("SUMMARY")
    print("=" * 116)
    for rp in report["rate_points"]:
        for name, candidate in rp["candidates"].items():
            print(f"  {rp['bits']}-bit {name:<8} VAL-B gain {candidate['val_b_gain_percent']:+.4f}% "
                 f"-> {candidate['verdict']}")

    path = args.output_dir / "m14_offline_gate.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
