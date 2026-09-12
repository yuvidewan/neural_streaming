"""M15 Phase A - formalize exactly what the DEPLOYED calibration policy
(`m10h_motion_compensation.calibrate_grids`) does, before any experimental
policy is run against it. Every claim below is either read directly from
that function's source or measured from the real TRAIN/VAL manifest -
nothing here is asserted from memory of M14's report.

Run:
  ./.venv/Scripts/python.exe scripts/m15_policy_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config

DEFAULT_OUTPUT_DIR = Path("outputs/m15_calibration_policy")


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M15 Phase A: formalize the current (deployed) calibration policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calibration-frames", type=int, default=400)
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    if not args.manifest.is_file():
        print(f"[ERROR] --manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    train_sequences = discover_sequences(args.manifest, split="train")
    val_sequences = discover_sequences(args.manifest, split="val")

    train_counts = [s.frame_count for s in train_sequences]
    total_train_frames = sum(train_counts)

    # Reproduce calibrate_grids's own sequential walk (its EXACT algorithm,
    # not a paraphrase) to report which/how-many TRAIN sequences the
    # deployed max_frames=400 recipe actually reaches.
    remaining = args.calibration_frames
    sequential_touch: list[dict[str, Any]] = []
    for sequence in train_sequences:
        if remaining <= 0:
            break
        take = min(remaining, sequence.frame_count)
        sequential_touch.append({"sequence_id": sequence.sequence_id,
                                 "frames_taken": take, "frames_available": sequence.frame_count})
        remaining -= take

    answers = {
        "q1_sequence_ordering": (
            "The manifest's own (already-sorted, alphabetical) order - "
            "discover_sequences() never reorders or samples; confirmed here: "
            f"first 5 TRAIN sequences = {[s.sequence_id for s in train_sequences[:5]]}"),
        "q2_frame_consumption": (
            "calibrate_grids walks `sequences` in the given order and, within each "
            "sequence, `sequence.load_frames()` then `for index in range(frames.shape[0])` "
            "front-to-back; a single GLOBAL counter `seen` is incremented per frame and "
            "checked (`if seen >= max_frames: break`) in BOTH the inner (per-sequence) and "
            "outer (per-TRAIN-sequence) loop, so consumption stops mid-sequence the instant "
            "the budget is reached - never fills the current sequence first."),
        "q3_max_frames_meaning": (
            "A single GLOBAL scalar cap on TOTAL frames consumed, counted across the ENTIRE "
            "walk over `sequences` - not a per-sequence cap. It is applied identically, but "
            "independently (the counter restarts at 0), to TWO SEPARATE passes over the same "
            "`sequences` argument: (1) the intra-only pass (fits intra_params/intra_entropy_model), "
            "(2) the residual+motion pass (fits residual_params/residual_entropy_model and "
            "motion_entropy_model). Both passes therefore see the SAME substantive prefix "
            "of TRAIN (the same sequences, same frame range within each), just walked twice."),
        "q4_same_frames_for_every_table": (
            "Yes for WHICH FRAMES are in scope (both passes walk the identical `sequences` "
            "list under the identical `max_frames`), but NOT for which of those frames "
            "actually contribute a SAMPLE to a given table - see q5."),
        "q5_which_tables_from_which_frames": (
            "intra_entropy_model: every one of the max_frames consumed frames (I- and P-typed "
            "alike) contributes one intra-latent sample. residual_entropy_model and "
            "motion_entropy_model: only frames typed FRAME_TYPE_P by gop_frame_types() "
            "(gop_size-1 of every gop_size frames) contribute a sample; the I-frame at each "
            "GOP boundary only advances `previous_reconstruction`, contributing to neither. "
            "Measured at max_frames=400, gop_size=10 (from the current recorded audit): "
            "intra_frames=400, residual_frames=motion_frames=358 (~89.5% of 400, matching "
            "the 9/10 P-frame fraction minus boundary effects)."),
        "q6_residual_codebook_calibration_path": (
            "Completely different, and NOT calibrate_grids at all. The M11-G16 residual "
            "codebook's symbol-to-prototype ASSIGNMENT and its ORIGINAL frequencies come from "
            "m11_train.fit_model_codebook over the m11_data caching pipeline's own TRAIN "
            "sample; M13's RECALIBRATED frequencies (frozen, deployed, unaffected by this "
            "milestone) come from m13_recalibration.fit_recalibrated_frequencies over the "
            "SAME m11_data TRAIN/VAL-A split - never from calibrate_grids's sequential walk. "
            "This is WHY M13's recalibration did not inherit the coverage problem M14 found: "
            "it was never on this code path to begin with."),
        "q7_bit_depth_dependence": (
            "intra_params/intra_entropy_model and residual_params/residual_entropy_model: "
            "YES, fully bit-depth-dependent (quantization bin count scales with `bits`; "
            "calibrate_grids is called once per bit depth and produces a distinct table "
            "each time). motion_entropy_model: the ALPHABET is bit-depth-INDEPENDENT "
            "(motion_bits = motion_alphabet_bits(search_range), a function of search_range "
            "only). The FITTED FREQUENCIES are ALMOST bit-depth-independent but not exactly: "
            "calibrate_grids's own I-frame branch reconstructs the GOP-boundary reference "
            "through the ACTUAL bit-depth-dependent intra quantizer/entropy model "
            "(encode_latent_to_payload/decode_payload_to_latent under that call's intra_params), "
            "so the one P-frame immediately after each I-frame sees a reference whose quality "
            "depends on `bits` - confirmed empirically: the deployed motion_entropy_model_id "
            "differs across 5/4/3-bit in outputs/m14_entropy_audit/m14_entropy_audit.json "
            "despite the alphabet being fixed. M14's OWN collect_motion_symbols (used by every "
            "M15 policy below) instead advances even the I-frame reference via the TRUE, "
            "unquantized latent (`model.decode(latent)`, no intra_params/intra_entropy_model "
            "involved at all) - a simplification that makes ITS motion tables exactly "
            "bit-depth-independent (confirmed: M14's deployed recalibrated motion identity is "
            "IDENTICAL across 5/4/3-bit). Both are pre-existing, documented approximations "
            "relative to the coder's real closed loop (see calibrate_grids's own "
            "'residuals... very slightly optimistic' comment); M15 changes neither - every "
            "M15 policy reuses collect_motion_symbols/collect_intra_symbols UNMODIFIED, so this "
            "asymmetry applies identically across all four candidate policies and is not a "
            "confound between them."),
        "q8_determinism": (
            "Deterministic. discover_sequences() reads a fixed on-disk manifest in fixed "
            "order; the residual/motion pass runs inside deterministic_kernels(); no "
            "randomness anywhere in the walk. Cross-process reproducibility for this exact "
            "function was established and tested in M11 (test_m11_reproducibility.py) and is "
            "re-confirmed fresh for the current checkpoint in M15 Phase 0 "
            "(m15_phase0_baseline.json)."),
        "q9_sequential_or_balanced": (
            "SEQUENTIAL, not sequence-balanced. At the deployed max_frames=400, the walk "
            "reaches only the first few TRAIN sequences in manifest (alphabetical) order "
            "before the budget is exhausted - see 'sequential_walk_at_400' below for exactly "
            "which ones and how many frames each contributes."),
        "q10_existing_test_assumptions": (
            "No existing test asserts a specific TRAIN ORDERING or sequence-coverage "
            "property of calibrate_grids's calibration - tests instead pin its OUTPUT "
            "invariants (determinism, frequency-table validity, coder byte-exactness) using "
            "small synthetic sequences where sequential vs balanced coverage cannot differ. "
            "M14's own tests (test_m14_recalibration.py) assert TRAIN-only access and "
            "fixed-quantizer/fixed-motion-estimator invariants for the collector functions "
            "M15 reuses unmodified, and pin the recorded Phase B verdict "
            "(intra=weak, motion=meaningful) against calibrate_grids's SEQUENTIAL baseline - "
            "none of that pins sequential ordering as a requirement, so a broader policy is "
            "free to change coverage without breaking any existing assertion."),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "M15 Phase A - formalize the current calibration policy",
        "manifest": str(args.manifest),
        "train_sequence_count": len(train_sequences),
        "train_total_frames": total_train_frames,
        "train_min_frames_per_sequence": min(train_counts),
        "train_max_frames_per_sequence": max(train_counts),
        "train_mean_frames_per_sequence": total_train_frames / len(train_sequences),
        "val_sequence_ids": [s.sequence_id for s in val_sequences],
        "val_a_sequence_ids": [s.sequence_id for s in val_sequences[0::2]],
        "val_b_sequence_ids": [s.sequence_id for s in val_sequences[1::2]],
        "deployed_calibration_frames": args.calibration_frames,
        "sequential_walk_at_400": sequential_touch,
        "sequential_walk_sequences_touched": len(sequential_touch),
        "sequential_walk_percent_of_train_sequences": (
            len(sequential_touch) / len(train_sequences) * 100.0),
        "answers": answers,
    }

    print("=" * 110)
    print("M15 PHASE A - CURRENT CALIBRATION POLICY, FORMALIZED")
    print("=" * 110)
    print(f"  TRAIN: {len(train_sequences)} sequences, {total_train_frames} frames "
         f"(min {min(train_counts)}, max {max(train_counts)}, "
         f"mean {total_train_frames / len(train_sequences):.2f} per sequence)")
    print(f"  At max_frames={args.calibration_frames}, the deployed sequential walk touches "
         f"{len(sequential_touch)}/{len(train_sequences)} TRAIN sequences "
         f"({len(sequential_touch) / len(train_sequences) * 100:.1f}%):")
    for row in sequential_touch:
        print(f"    {row['sequence_id']:<20} {row['frames_taken']}/{row['frames_available']} frames")
    print(f"  VAL-A: {report['val_a_sequence_ids']}")
    print(f"  VAL-B: {report['val_b_sequence_ids']}")

    path = args.output_dir / "m15_baseline_policy.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
