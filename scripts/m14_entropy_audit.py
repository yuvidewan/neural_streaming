"""M14 Phase A - complete inventory of every static entropy probability/
frequency table the DEPLOYED (M13) codec actually uses.

Purely diagnostic: loads the real M13 calibration and models, INSPECTS them,
and writes what it finds. No behavior changes, no symbols regenerated, no
table refit.

WHAT ".nvct v2" ACTUALLY HOLDS (audited directly from
`m10h_motion_compensation.TemporalStreamHeader`'s docstring/layout, not
assumed): exactly THREE 8-byte entropy-model identity fields -
`intra_entropy_model_id`, `residual_entropy_model_id`, `motion_entropy_model_id`
- at fixed offsets 32/40/48 in the 56-byte fixed header. There is no fourth
slot, so any recalibrated table this milestone deploys must reuse one of
these three, never add a new one (a new slot would be a `.nvct v3` change,
forbidden).

Run:
  ./.venv/Scripts/python.exe scripts/m14_entropy_audit.py
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
RATE_POINTS = (5, 4, 3)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M14 Phase A: audit every static entropy table the deployed codec uses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=DEFAULT_M11_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--calibration-frames", type=int, default=400)
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
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    ev = _load_script("m10l_evaluate")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    train_sequences = discover_sequences(args.manifest, split="train")

    print("=" * 110, flush=True)
    print("M14 PHASE A - COMPLETE ENTROPY-TABLE INVENTORY (deployed M13 codec)")
    print("=" * 110)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "M14 Phase A entropy inventory", "checkpoint": str(args.checkpoint),
        "nvct_v2_entropy_identity_slots": ["intra_entropy_model_id", "residual_entropy_model_id",
                                           "motion_entropy_model_id"],
        "nvct_v2_slot_count": 3,
        "rate_points": [],
    }

    for bits in args.rate_points:
        print(f"\n  ---- {bits}-bit ----", flush=True)
        calibration = mc.calibrate_grids(
            model, train_sequences, bits=bits, mode="per_channel", gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, reference_mode="mc",
            max_frames=args.calibration_frames)
        signature = ev.calibration_signature(calibration, bits=bits,
                                             calibration_frames=args.calibration_frames,
                                             quant_mode="per_channel")

        intra_model = calibration["intra_entropy_model"]
        motion_model = calibration["motion_entropy_model"]
        dead_residual_model = calibration["residual_entropy_model"]
        provenance = calibration["provenance"]

        m11_16, checkpoint11 = ma.load_model(args.m11_dir / f"m11_G16_entropy_{bits}bit.pt",
                                             device=device)
        m13_codebook_path = args.m11_dir / f"m11_G16_codebook_{bits}bit_K512.json"
        m13_codebook = ml.SharedCodebook.from_dict(
            json.loads(m13_codebook_path.read_text(encoding="utf-8")))

        tables = [
            {
                "name": "intra_entropy_model",
                "identity_field": "intra_entropy_model_id",
                "identity_hex": intra_model.model_id().hex(),
                "live_in_m13_deployment": True,
                "used_for": "I-frame latent symbols (every GOP's first frame)",
                "alphabet_size": intra_model.num_symbols,
                "num_tables": intra_model.num_tables,
                "table_grouping": "one table per latent channel (channel-major, matching "
                                 "latent_to_symbols' coding order)",
                "symbol_source": "quantized latents of EVERY frame scanned during calibration "
                                 "(I- and P-typed alike - the intra grid/table must cover the "
                                 "full latent distribution, not just I-frame-typed positions)",
                "fitting_data": f"first {provenance['intra_frames']} frames reached while "
                               f"scanning TRAIN sequences in manifest order, capped at "
                               f"--calibration-frames={args.calibration_frames}",
                "fits_actual_deployed_symbols": True,
                "model_predicted": False,
                "smoothing_method": "EmpiricalEntropyModel: Laplace (+1) on raw counts",
                "normalization": "floor(p*65536), MIN_FREQUENCY=1 floor, deterministic "
                                "largest-first residual redistribution",
                "provenance_identity_fn": "EmpiricalEntropyModel.model_id() - hashes bits + "
                                         "frequencies",
                "coder_usage": "encode_latent_to_payload/decode_payload_to_latent -> "
                              "encode_symbols/decode_symbols, one table per channel, "
                              "STATELESS (not through ResumableDecoder today)",
                "test_could_influence": False,
                "recalibratable_without_changing_symbols": True,
                "recalibration_note": "quantization params (intra_params) are FROZEN and "
                                      "produce the symbols; only the FREQUENCY table can change "
                                      "without altering a single symbol",
            },
            {
                "name": "motion_entropy_model",
                "identity_field": "motion_entropy_model_id",
                "identity_hex": motion_model.model_id().hex(),
                "live_in_m13_deployment": True,
                "used_for": "P-frame block motion vectors (dy, dx)",
                "alphabet_size": motion_model.num_symbols,
                "num_tables": motion_model.num_tables,
                "table_grouping": "table 0 = every dy symbol, table 1 = every dx symbol "
                                  "(motion_table_index) - ONE shared identity for both",
                "symbol_source": "estimate_block_motion's real block-matched (dy, dx) vectors "
                                 "on frames scanned during calibration, shifted by "
                                 "+search_range (motion_to_symbols) - fully deterministic given "
                                 "the frozen motion estimator, no entropy model involved in "
                                 "SYMBOL generation at all",
                "fitting_data": f"{provenance['motion_frames']} P-frames reached while "
                               f"scanning TRAIN sequences in manifest order, capped at "
                               f"--calibration-frames={args.calibration_frames} TOTAL "
                               f"(I+P) frames scanned",
                "fits_actual_deployed_symbols": True,
                "model_predicted": False,
                "smoothing_method": "EmpiricalEntropyModel: Laplace (+1) on raw counts",
                "normalization": "floor(p*65536), MIN_FREQUENCY=1 floor, deterministic "
                                "largest-first residual redistribution",
                "provenance_identity_fn": "EmpiricalEntropyModel.model_id() - hashes bits + "
                                         "frequencies",
                "coder_usage": "encode_motion_payload/decode_motion_payload -> "
                              "encode_symbols/decode_symbols, 2 tables, ONE-SHOT "
                              "(not grouped, not through ResumableDecoder)",
                "test_could_influence": False,
                "recalibratable_without_changing_symbols": True,
                "recalibration_note": "motion vectors depend only on the FROZEN motion "
                                      "estimator/block_size/search_range/GOP, never on any "
                                      "entropy table - recalibrating changes zero motion "
                                      "symbols by construction",
            },
            {
                "name": "residual_entropy_model (calibrate_grids, M10H-style per-channel)",
                "identity_field": None,
                "identity_hex": dead_residual_model.model_id().hex(),
                "live_in_m13_deployment": False,
                "used_for": "NOTHING in the M13-deployed closed loop - superseded by "
                           "M11-G16 + M13's recalibrated 512-entry codebook for every "
                           "P-frame residual. `calibrate_grids` still computes and returns "
                           "it (a shared utility also used by earlier M10G/M10H-era "
                           "scripts), but scripts/m13_closed_loop.py never reads "
                           "calibration['residual_entropy_model'] - only "
                           "calibration['residual_params'] (the QUANTIZATION grid, which "
                           "IS live) is used from this part of the calibration dict.",
                "alphabet_size": dead_residual_model.num_symbols,
                "num_tables": dead_residual_model.num_tables,
                "table_grouping": "one table per latent channel",
                "symbol_source": "N/A - not used for deployed coding",
                "fitting_data": "N/A - not used for deployed coding",
                "fits_actual_deployed_symbols": None,
                "model_predicted": False,
                "smoothing_method": "N/A - not used for deployed coding",
                "normalization": "N/A - not used for deployed coding",
                "provenance_identity_fn": "N/A - not used for deployed coding",
                "coder_usage": "NONE (dead weight in the M13 deployment path)",
                "test_could_influence": False,
                "recalibratable_without_changing_symbols": False,
                "recalibration_note": "NOT A CANDIDATE: recalibrating a table nothing reads "
                                      "cannot produce any deployed byte change. Documented "
                                      "here because Phase A asked to inspect every static "
                                      "table calibrate_grids builds, not just the live ones.",
            },
            {
                "name": "m11_g16_residual_codebook (M13's ALREADY-recalibrated table)",
                "identity_field": "residual_entropy_model_id",
                # Filled in below from M13's own recorded benchmark provenance - avoids
                # re-deriving the exact M10K checkpoint hash bytes here just to reproduce
                # a value M13 already computed and persisted.
                "identity_hex": None,
                "live_in_m13_deployment": True,
                "used_for": "P-frame residual latent symbols (the M13 baseline this milestone "
                           "freezes)",
                "alphabet_size": m13_codebook.alphabet,
                "num_tables": m13_codebook.size,
                "table_grouping": "512 shared prototypes, assigned per-position via "
                                 "SharedCodebook.assign_tensor (M11-G16's own channel-"
                                 "autoregressive prediction)",
                "symbol_source": "TRAIN residual symbols actually assigned to each prototype "
                                 "by the DEPLOYED M11-G16 model (M13's recalibration target)",
                "fitting_data": "536 TRAIN P-frames (m11_data cache) - ALREADY recalibrated "
                               "by M13; FROZEN as this milestone's baseline, not re-touched",
                "fits_actual_deployed_symbols": True,
                "model_predicted": False,
                "smoothing_method": "M11 causal-context hierarchical smoothing toward the "
                                    "pre-recalibration prior, strength chosen on VAL-A "
                                    "(scripts/m13_recalibration.py, M13, frozen)",
                "normalization": "probabilities_to_frequencies (same function family as "
                                "the other tables' Laplace path, different smoothing input)",
                "provenance_identity_fn": "m11_ar_entropy.model_identity(model, "
                                         "codebook=coding_codebook) - hashes model weights + "
                                         "codebook frequencies together",
                "coder_usage": "encode_frame_recalibrated/decode_frame_recalibrated -> "
                              "ResumableDecoder (M12), group_size=16, 4 sequential steps",
                "test_could_influence": False,
                "recalibratable_without_changing_symbols": True,
                "recalibration_note": "FROZEN by M14's own rules - 'M13 recalibrated "
                                      "M11-G16 configuration as the temporal baseline'. Not "
                                      "a candidate; listed for completeness of the inventory.",
            },
        ]
        # The placeholder identity_hex above needs the real m11_op identity - pull it
        # straight from M13's own recorded benchmark provenance instead of re-deriving it
        # (avoids needing the exact M10K checkpoint hash bytes here).
        m13_benchmark = json.loads(
            Path("outputs/m13_recalibration/m13_davis_benchmark.json").read_text(encoding="utf-8")
        ) if Path("outputs/m13_recalibration/m13_davis_benchmark.json").is_file() else None
        if m13_benchmark is not None:
            tables[3]["identity_hex"] = m13_benchmark["provenance"][f"{bits}bit"]["new_identity"]

        for table in tables:
            print(f"    {table['name']:<55} live={table['live_in_m13_deployment']!s:<5} "
                 f"tables={table['num_tables']:>4} alphabet={table['alphabet_size']:>4} "
                 f"model_predicted={table['model_predicted']}", flush=True)

        report["rate_points"].append({
            "bits": bits, "calibration_signature": signature,
            "calibration_provenance": provenance, "tables": tables,
        })

    candidates = ["intra_entropy_model", "motion_entropy_model"]
    report["live_candidates_not_yet_recalibrated"] = candidates
    report["excluded_not_live"] = ["residual_entropy_model (calibrate_grids, M10H-style)"]
    report["frozen_already_recalibrated"] = ["m11_g16_residual_codebook (M13)"]
    print(f"\n  LIVE, NOT-YET-RECALIBRATED candidates: {candidates}")
    print(f"  excluded (not live in deployment): "
         f"{report['excluded_not_live']}")
    print(f"  frozen (already recalibrated by M13): {report['frozen_already_recalibrated']}")

    path = args.output_dir / "m14_entropy_audit.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
