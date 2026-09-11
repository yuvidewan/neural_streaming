"""M14 Phase D - actual arithmetic-coded validation for whichever candidates
Phase B's offline gate found credible (>= 0.5%, evaluated per-candidate).

Every combination requested is run against the SAME frozen M13 residual arm
(model11 + assign_codebook + M13's recalibrated coding_codebook - re-derived
here exactly as scripts/m13_coded_validation.py did, proven deterministic in
M13's own cross-process check) on a small held-out VAL sequence sample, so
"only the intra/motion table changed" is true by construction: the closed
loop (scripts/m13_closed_loop.py + scripts/m14_closed_loop.py, both reused
unmodified) computes residual symbols and motion vectors ONCE per frame and
every combination just re-codes them under a different (intra_entropy_model,
motion_entropy_model) pair.

Combinations are named by which tables are NEW (recalibrated) vs CURRENT
(deployed): "baseline" (both current - the M13 configuration, unchanged),
"intra" (intra new, motion current), "motion" (intra current, motion new),
"both" (both new) - covering the interaction-effects requirement (A alone,
B alone, A+B) without assuming gains add.

Run:
  ./.venv/Scripts/python.exe scripts/m14_coded_validation.py --combinations intra motion both
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

DEFAULT_OUTPUT_DIR = Path("outputs/m14_entropy_audit")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M11_DIR = Path("outputs/m11_autoregressive_entropy")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
RATE_POINTS = (5, 4, 3)
M11_G16_GROUP_SIZE = 16
COMBINATIONS = ("baseline", "intra", "motion", "both")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M14 Phase D: real arithmetic-coded validation for intra/motion candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=DEFAULT_M11_DIR)
    parser.add_argument("--m10k-dir", type=Path, default=DEFAULT_M10K_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--combinations", nargs="+", default=["intra", "motion", "both"],
                        choices=list(COMBINATIONS))
    parser.add_argument("--val-sequences", type=int, default=3)
    parser.add_argument("--max-frames-per-sequence", type=int, default=30)
    parser.add_argument("--train-frames-per-sequence", type=int, default=8)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--calibration-frames", type=int, default=400)
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
    mk = _load_script("m10k_learned_entropy")
    m13 = _load_script("m13_recalibration")
    m14 = _load_script("m14_recalibration")
    cl14 = _load_script("m14_closed_loop")
    ev = _load_script("m10l_evaluate")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    print("=" * 122, flush=True)
    print("M14 PHASE D - ACTUAL ARITHMETIC-CODED VALIDATION (intra/motion candidates)")
    print("=" * 122)
    print(f"  combinations: {args.combinations}")

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    val_sequences = discover_sequences(args.manifest, split="val",
                                       max_sequences=args.val_sequences,
                                       max_frames_per_sequence=args.max_frames_per_sequence)
    calibration_sequences = discover_sequences(args.manifest, split="train")
    broad_train_sequences = discover_sequences(
        args.manifest, split="train", max_frames_per_sequence=args.train_frames_per_sequence)
    print(f"  validation sequences: {[s.sequence_id for s in val_sequences]}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = args.output_dir / "coded_validation_streams"
    stream_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"phase": "M14 Phase D coded validation",
                              "combinations_requested": args.combinations,
                              "val_sequence_ids": [s.sequence_id for s in val_sequences],
                              "rate_points": []}

    for bits in args.rate_points:
        print(f"\n  ---- {bits}-bit ----", flush=True)
        calibration = mc.calibrate_grids(
            model, calibration_sequences, bits=bits, mode="per_channel", gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, reference_mode="mc",
            max_frames=args.calibration_frames)
        signature = ev.calibration_signature(calibration, bits=bits,
                                             calibration_frames=args.calibration_frames,
                                             quant_mode="per_channel")
        current_intra, current_motion = calibration["intra_entropy_model"], calibration["motion_entropy_model"]
        intra_params, residual_params = calibration["intra_params"], calibration["residual_params"]

        # --- M13's frozen residual arm (re-derived, deterministic - see M13's own
        # cross-process check) -----------------------------------------------------------
        data = md.load_or_collect(model, checkpoint=args.checkpoint, manifest=args.manifest,
                                  bits=bits, device=device, cache_dir=cache_dir,
                                  log=lambda m: print(m, flush=True))
        train_symbols = data["train_symbols"].astype(np.int64)
        val_symbols = data["val_symbols"].astype(np.int64)
        select_mask, _ = md.split_validation(data)
        channels = train_symbols.shape[1]
        zero = torch.from_numpy(cx.zero_symbols(calibration["residual_params"], channels))
        m10k, m10k_checkpoint = mk.load_entropy_model(
            args.m10k_dir / f"learned_entropy_{bits}bit.pt", device=device)
        m10k_identity = ev._m10k_model_identity(m10k_checkpoint["model_state_dict"])
        model11, checkpoint11 = ma.load_model(args.m11_dir / f"m11_G16_entropy_{bits}bit.pt",
                                              device=device)
        ev_m11 = _load_script("m11_evaluate")
        ev_m11.check_provenance(checkpoint11, signature=signature, bits=bits,
                                group_size=M11_G16_GROUP_SIZE,
                                context_definition_id=ma.context_definition_id(M11_G16_GROUP_SIZE),
                                m10k_identity=m10k_identity)
        assign_codebook = ml.SharedCodebook.from_dict(json.loads(
            (args.m11_dir / f"m11_G16_codebook_{bits}bit_K512.json").read_text(encoding="utf-8")))
        train_k = m13.m11_g16_prototype_indices(model11, assign_codebook, train_symbols,
                                                data["train_references"], zero, device=device)
        val_k = m13.m11_g16_prototype_indices(model11, assign_codebook, val_symbols,
                                              data["val_references"], zero, device=device)
        flat = lambda a: a.reshape(-1)
        frequencies, strength, _, _ = m13.fit_recalibrated_frequencies(
            assign_codebook, flat(train_symbols), flat(train_k),
            flat(val_symbols[select_mask]), flat(val_k[select_mask]), alphabet=2 ** bits)
        coding_codebook = m13.build_recalibrated_codebook(
            assign_codebook, frequencies, provenance={"bits": bits, "strength": strength})
        m13_identity = ma.model_identity(model11, m10k_identity=m10k_identity,
                                         calibration_signature=signature, bits=bits,
                                         codebook=coding_codebook)
        residual_arm = {"m13_recal": {"model": model11, "zero": zero,
                                      "assign_codebook": assign_codebook,
                                      "coding_codebook": coding_codebook,
                                      "identity": m13_identity}}

        # --- new intra/motion tables, fit exactly as Phase B did --------------------------
        train_intra = m14.collect_intra_symbols(model, broad_train_sequences,
                                                intra_params=intra_params, max_frames=10 ** 9,
                                                device=device)
        new_intra = m14.fit_empirical(train_intra, bits=bits, num_tables=train_intra.shape[1])
        train_motion = m14.collect_motion_symbols(
            mc, model, broad_train_sequences, block_size=args.block_size,
            search_range=args.search_range, gop_size=args.gop, max_frames=10 ** 9,
            reference_mode="mc", device=device)
        motion_bits = mc.motion_alphabet_bits(args.search_range)
        new_motion = m14.fit_empirical(train_motion, bits=motion_bits, num_tables=2)

        tables_by_combination = {
            "baseline": (current_intra, current_motion),
            "intra": (new_intra, current_motion),
            "motion": (current_intra, new_motion),
            "both": (new_intra, new_motion),
        }

        rate_point: dict[str, Any] = {"bits": bits, "combinations": {}}
        baseline_result = None
        for combo in ["baseline"] + [c for c in args.combinations if c != "baseline"]:
            intra_model, motion_model = tables_by_combination[combo]
            run = cl14.run_sequences_for_arms(
                mc, ma, m13, model, val_sequences, residual_arm, stream_dir,
                intra_params=intra_params, intra_entropy_model=intra_model,
                residual_params=residual_params, motion_entropy_model=motion_model,
                bits=bits, gop_size=args.gop, block_size=args.block_size,
                search_range=args.search_range)
            invariants = run["invariants"]
            stats = run["results"]["m13_recal"]
            print(f"    [{combo:<8}] invariants symbols={invariants['symbols']} "
                 f"reconstruction={invariants['reconstruction']} motion={invariants['motion']} "
                 f"metrics={invariants['metrics']}  total_bytes={stats['total_container_bytes']:,}  "
                 f"BPP={stats['stream_bpp']:.5f}  PSNR={stats['mean_psnr_db']:.4f}  "
                 f"MS-SSIM={stats['mean_msssim']:.6f}", flush=True)
            if not all(invariants.values()):
                print(f"[ERROR] combination '{combo}' violated an invariant. STOPPING.",
                     file=sys.stderr)
                return 1
            if combo == "baseline":
                baseline_result = stats
            else:
                byte_delta = stats["total_container_bytes"] - baseline_result["total_container_bytes"]
                byte_gain = -byte_delta / baseline_result["total_container_bytes"] * 100
                print(f"               vs baseline: {byte_delta:+,} bytes "
                     f"({byte_gain:+.4f}% total container gain)", flush=True)
            rate_point["combinations"][combo] = {
                "invariants": invariants, "stats": stats,
                "byte_gain_vs_baseline_percent": (
                    None if combo == "baseline" else
                    -(stats["total_container_bytes"] - baseline_result["total_container_bytes"])
                    / baseline_result["total_container_bytes"] * 100),
            }
        report["rate_points"].append(rate_point)

    print()
    print("=" * 122)
    print("SUMMARY - total container byte gain vs baseline")
    print("=" * 122)
    for rp in report["rate_points"]:
        for combo, entry in rp["combinations"].items():
            if combo == "baseline":
                continue
            print(f"  {rp['bits']}-bit {combo:<8} {entry['byte_gain_vs_baseline_percent']:+.4f}%")

    path = args.output_dir / "m14_coded_validation.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
