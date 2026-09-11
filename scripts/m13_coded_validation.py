"""M13 Phase D - actual arithmetic-coder validation (mandatory: Phase C
passed at every rate point).

Runs the REAL closed loop (`m13_closed_loop.py`: motion estimation, warping,
`.nvct` v2 streams) over a small set of DAVIS VALIDATION sequences (not
TEST - reserved for Phase E's headline number) with two arms coding the
SAME residual symbols:

  m11_op      the deployed M11-G16 codebook, unchanged.
  m13_recal   the SAME model and SAME assignment, TRAIN-recalibrated
              coding frequencies (scripts/m13_recalibration.py).

For each of 5/4/3 bits, verifies residual symbols, motion bytes and
reconstruction are IDENTICAL between arms (structurally guaranteed by the
closed loop computing one set of true symbols and letting each arm just
re-code them - a failure here means a real bug, not an expected effect of
recalibration), then measures:

  ideal bits (both arms) vs ACTUAL coded bytes (both arms)
  coder overhead = (actual_bytes*8 - ideal_bits) / ideal_bits
  realized_gain  = actual_bit_reduction / ideal_bit_reduction

The milestone brief is explicit: do not assume ideal entropy gain survives
arithmetic coding - this is where that assumption is actually checked.

Run:
  ./.venv/Scripts/python.exe scripts/m13_coded_validation.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m13_recalibration")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M11_DIR = Path("outputs/m11_autoregressive_entropy")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
RATE_POINTS = (5, 4, 3)
M11_G16_GROUP_SIZE = 16  # frozen: M13's whole premise is "the deployed M11-G16 model"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M13 Phase D: real arithmetic-coded validation on a small VAL set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=DEFAULT_M11_DIR)
    parser.add_argument("--m10k-dir", type=Path, default=DEFAULT_M10K_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--val-sequences", type=int, default=3)
    parser.add_argument("--max-frames-per-sequence", type=int, default=30)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=None)
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
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    md = _load_script("m11_data")
    cx = _load_script("m11_causal_context")
    m13 = _load_script("m13_recalibration")
    cl = _load_script("m13_closed_loop")
    ev = _load_script("m10l_evaluate")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    print("=" * 120, flush=True)
    print("M13 PHASE D - ACTUAL ARITHMETIC-CODED VALIDATION (small VAL set, mandatory per Phase C)")
    print("=" * 120)

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    val_sequences = discover_sequences(args.manifest, split="val",
                                       max_sequences=args.val_sequences,
                                       max_frames_per_sequence=args.max_frames_per_sequence)
    print(f"  validation sequences: {[s.sequence_id for s in val_sequences]} "
         f"({sum(s.frame_count for s in val_sequences)} frames total)", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = args.output_dir / "validation_streams"
    stream_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"phase": "M13 Phase D coded validation",
                              "val_sequence_ids": [s.sequence_id for s in val_sequences],
                              "rate_points": []}

    for bits in args.rate_points:
        print(f"\n  ---- {bits}-bit ----", flush=True)
        data = md.load_or_collect(
            model, checkpoint=args.checkpoint, manifest=args.manifest, bits=bits, device=device,
            cache_dir=cache_dir, log=lambda m: print(m, flush=True))
        calibration = mc.calibrate_grids(
            model, discover_sequences(args.manifest, split="train"), bits=bits,
            mode="per_channel", gop_size=args.gop, block_size=args.block_size,
            search_range=args.search_range, reference_mode="mc", max_frames=400)
        signature = ev.calibration_signature(calibration, bits=bits, calibration_frames=400,
                                             quant_mode="per_channel")
        cached_signature = ev.calibration_signature(data["calibration"], bits=bits,
                                                     calibration_frames=400, quant_mode="per_channel")
        if signature != cached_signature:
            print(f"[ERROR] fresh calibration {signature} != cached {cached_signature} - "
                 f"determinism regressed. STOPPING.", file=sys.stderr)
            return 1

        train_symbols = data["train_symbols"].astype(np.int64)
        val_symbols = data["val_symbols"].astype(np.int64)
        select_mask, _ = md.split_validation(data)
        channels = train_symbols.shape[1]
        zero = torch.from_numpy(cx.zero_symbols(calibration["residual_params"], channels))

        m11k, m11k_checkpoint = _load_script("m10k_learned_entropy").load_entropy_model(
            args.m10k_dir / f"learned_entropy_{bits}bit.pt", device=device)
        m10k_identity = ev._m10k_model_identity(m11k_checkpoint["model_state_dict"])

        model11, checkpoint11 = ma.load_model(args.m11_dir / f"m11_G16_entropy_{bits}bit.pt",
                                              device=device)
        ev_m11 = _load_script("m11_evaluate")
        try:
            ev_m11.check_provenance(checkpoint11, signature=signature, bits=bits,
                                    group_size=M11_G16_GROUP_SIZE,
                                    context_definition_id=ma.context_definition_id(
                                        M11_G16_GROUP_SIZE), m10k_identity=m10k_identity)
        except ev_m11.ProvenanceError as error:
            print(f"[ERROR] M11-G16 checkpoint provenance mismatch: {error}. STOPPING.",
                 file=sys.stderr)
            return 1
        assign_codebook = ml.SharedCodebook.from_dict(json.loads(
            (args.m11_dir / f"m11_G16_codebook_{bits}bit_K512.json").read_text(encoding="utf-8")))

        train_k = m13.m11_g16_prototype_indices(model11, assign_codebook, train_symbols,
                                                data["train_references"], zero, device=device)
        val_k = m13.m11_g16_prototype_indices(model11, assign_codebook, val_symbols,
                                              data["val_references"], zero, device=device)
        flat = lambda a: a.reshape(-1)
        frequencies, strength, val_a_bits, val_a_baseline = m13.fit_recalibrated_frequencies(
            assign_codebook, flat(train_symbols), flat(train_k),
            flat(val_symbols[select_mask]), flat(val_k[select_mask]), alphabet=2 ** bits)
        coding_codebook = m13.build_recalibrated_codebook(
            assign_codebook, frequencies, provenance={"bits": bits, "strength": strength})
        print(f"    recalibrated with strength {strength:g} (VAL-A: baseline "
             f"{val_a_baseline:.5f} -> recalibrated {val_a_bits:.5f} bits/symbol)", flush=True)

        arms = cl.build_arms(ma, m13, model11, assign_codebook, coding_codebook, zero,
                             m10k_identity=m10k_identity, calibration_signature=signature,
                             bits=bits)
        assert arms["m11_op"]["identity"] != arms["m13_recal"]["identity"], \
            "recalibration must produce a distinct residual_entropy_model_id"

        run = cl.run_sequences(
            mc, ma, m13, model, val_sequences, arms, stream_dir,
            intra_params=calibration["intra_params"],
            intra_entropy_model=calibration["intra_entropy_model"],
            residual_params=calibration["residual_params"],
            motion_entropy_model=calibration["motion_entropy_model"], bits=bits,
            gop_size=args.gop, block_size=args.block_size, search_range=args.search_range)

        invariants = run["invariants"]
        print(f"    invariants - symbols {invariants['symbols']} | reconstruction "
             f"{invariants['reconstruction']} | motion {invariants['motion']} | "
             f"PSNR/MS-SSIM {invariants['metrics']}", flush=True)
        if not all(invariants.values()):
            print("[ERROR] m11_op and m13_recal diverged beyond entropy coding. STOPPING.",
                 file=sys.stderr)
            return 1

        old, new = run["results"]["m11_op"], run["results"]["m13_recal"]
        ideal_reduction = old["p_frame_ideal_bits"] - new["p_frame_ideal_bits"]
        actual_reduction = (old["total_p_frame_residual_bytes"]
                            - new["total_p_frame_residual_bytes"]) * 8
        realized_gain = (actual_reduction / ideal_reduction * 100.0
                         if abs(ideal_reduction) > 1e-9 else float("nan"))
        old_overhead = (old["total_p_frame_residual_bytes"] * 8 - old["p_frame_ideal_bits"]) \
            / old["p_frame_ideal_bits"] * 100
        new_overhead = (new["total_p_frame_residual_bytes"] * 8 - new["p_frame_ideal_bits"]) \
            / new["p_frame_ideal_bits"] * 100
        ideal_gain_pct = ideal_reduction / old["p_frame_ideal_bits"] * 100
        byte_gain_pct = (actual_reduction / 8) / old["total_p_frame_residual_bytes"] * 100

        print(f"    ideal bits: old {old['p_frame_ideal_bits']:,.0f}  "
             f"new {new['p_frame_ideal_bits']:,.0f}  ideal gain {ideal_gain_pct:+.3f}%",
             flush=True)
        print(f"    actual P-residual bytes: old {old['total_p_frame_residual_bytes']:,}  "
             f"new {new['total_p_frame_residual_bytes']:,}  byte gain {byte_gain_pct:+.3f}%",
             flush=True)
        print(f"    coder overhead: old {old_overhead:+.4f}%  new {new_overhead:+.4f}%",
             flush=True)
        print(f"    REALIZED GAIN (actual / ideal): {realized_gain:.2f}%", flush=True)
        print(f"    BPP: old {old['stream_bpp']:.5f}  new {new['stream_bpp']:.5f}   "
             f"PSNR old {old['mean_psnr_db']:.4f} new {new['mean_psnr_db']:.4f}   "
             f"MS-SSIM old {old['mean_msssim']:.6f} new {new['mean_msssim']:.6f}", flush=True)

        report["rate_points"].append({
            "bits": bits, "smoothing_strength": strength,
            "invariants": invariants,
            "old": old, "new": new, "decode_seconds": run["decode_times"],
            "ideal_gain_percent": ideal_gain_pct, "byte_gain_percent": byte_gain_pct,
            "old_coder_overhead_percent": old_overhead, "new_coder_overhead_percent": new_overhead,
            "realized_gain_percent": realized_gain,
            "old_identity": arms["m11_op"]["identity"].hex(),
            "new_identity": arms["m13_recal"]["identity"].hex(),
            "per_sequence": run["per_sequence"],
        })

    all_invariants_ok = all(rp["invariants"] and all(rp["invariants"].values())
                            for rp in report["rate_points"])
    all_gains_positive = all(rp["byte_gain_percent"] > 0 for rp in report["rate_points"])
    realized_ok = all(rp["realized_gain_percent"] == rp["realized_gain_percent"]  # not NaN
                      and rp["realized_gain_percent"] > 80.0 for rp in report["rate_points"])
    report["phase_d_passed"] = bool(all_invariants_ok and all_gains_positive)
    report["realized_gain_high"] = bool(realized_ok)
    print()
    print("=" * 120)
    print(f"PHASE D: invariants hold everywhere = {all_invariants_ok}, "
         f"actual byte gain positive everywhere = {all_gains_positive}, "
         f"realized gain > 80% everywhere = {realized_ok}")
    print("=" * 120)

    path = args.output_dir / "m13_coded_validation.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
