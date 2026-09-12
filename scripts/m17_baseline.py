"""M17 Phase 0/A - confirm the M13/M14/M15 baseline this milestone probes has
not drifted, and document exactly how the deployed residual path is
assembled, before spending any compute on the real-vs-oracle diagnostic.

Run:
  ./.venv/Scripts/python.exe scripts/m17_baseline.py
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

DEFAULT_OUTPUT_DIR = Path("outputs/m17_residual_oracle_audit")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M11_DIR = Path("outputs/m11_autoregressive_entropy")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
DEFAULT_M13_DAVIS_JSON = Path("outputs/m13_recalibration/m13_davis_benchmark.json")
RATE_POINTS = (5, 4, 3)
M11_G16_GROUP_SIZE = 16


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M17 Phase 0/A: confirm baseline, document the deployed residual path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=DEFAULT_M11_DIR)
    parser.add_argument("--m10k-dir", type=Path, default=DEFAULT_M10K_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--m13-davis-json", type=Path, default=DEFAULT_M13_DAVIS_JSON)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint),
                        ("--m13-davis-json", args.m13_davis_json)):
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
    ev = _load_script("m10l_evaluate")
    ev_m11 = _load_script("m11_evaluate")
    gate = _load_script("m12_spatial_offline_gate")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    train_sequences = discover_sequences(args.manifest, split="train")

    recorded_davis = json.loads(args.m13_davis_json.read_text(encoding="utf-8"))
    recorded_provenance = recorded_davis.get("provenance", {})

    report: dict[str, Any] = {
        "phase": "M17 Phase 0/A - baseline confirmation + residual-path trace",
        "checkpoint": str(args.checkpoint), "rate_points": [],
        "all_identities_match_recorded": True,
        "pipeline_trace": {
            "step_1_real_residual": "delta = latent - reference_latent, where reference_latent "
                "= model.encode(warp_blocks(previous_reconstruction, motion, block_size)) - "
                "see m13_closed_loop.encode_multi lines ~140-143",
            "step_2_g16_causal_context": "model11.planes(target_symbols, zero) builds [B,C,3,H,W] "
                "causal planes (group_starts-based: only channel-groups BEFORE the current one "
                "are visible, using the TRUE target symbols since encode-time knows them; zero "
                "-padded for the first group) - m11_ar_entropy.context_planes, group_size=16",
            "step_3_g16_prediction": "model11.log_probabilities(reference_latent, planes) -> "
                "per-position predicted distribution over the 2**bits alphabet, conditioned on "
                "BOTH reference_latent and the causal context planes",
            "step_4_codebook_assignment": "assign_codebook.assign_tensor(rows) - `rows` = the "
                "G16 model's predicted probabilities (step 3), reshaped C-major "
                "(m11_ar_entropy._rows); argmin L1 or log2-cost distance to the DEPLOYED "
                "codebook's 512 prototypes, lowest-index tie-break. This is the ORIGINAL, "
                "unmodified assign_codebook - NEVER the recalibrated one (M13's own invariant, "
                "reused unmodified)",
            "step_5_m13_recalibrated_frequencies": "coding_codebook.cumulative[table_index] - a "
                "SEPARATE SharedCodebook object holding TRAIN-fit, VAL-A-smoothed frequencies "
                "(m13_recalibration.fit_recalibrated_frequencies), used ONLY for the coder's "
                "cumulative frequency table, never for assignment",
            "step_6_arithmetic_coding": "nvc.compression.range_coder.encode_symbols(flat_symbols, "
                "coding_codebook.cumulative, table_index) -> payload bytes, via "
                "m13_recalibration.encode_frame_recalibrated (called here UNMODIFIED, never "
                "reimplemented or approximated - M17's central methodological commitment)",
        },
    }

    for bits in args.rate_points:
        calibration = mc.calibrate_grids(
            model, train_sequences, bits=bits, mode="per_channel", gop_size=10,
            block_size=16, search_range=16, reference_mode="mc",
            max_frames=args.calibration_frames)
        signature = ev.calibration_signature(calibration, bits=bits,
                                             calibration_frames=args.calibration_frames,
                                             quant_mode="per_channel")
        data = md.load_or_collect(model, checkpoint=args.checkpoint, manifest=args.manifest,
                                  bits=bits, device=device, cache_dir=cache_dir,
                                  log=lambda m: None)
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
        ev_m11.check_provenance(checkpoint11, signature=signature, bits=bits,
                                group_size=M11_G16_GROUP_SIZE,
                                context_definition_id=ma.context_definition_id(M11_G16_GROUP_SIZE),
                                m10k_identity=m10k_identity)
        assign_codebook = ml.SharedCodebook.from_dict(json.loads(
            (args.m11_dir / f"m11_G16_codebook_{bits}bit_K512.json").read_text(encoding="utf-8")))
        train_k = gate.m11_g16_prototype_indices(model11, assign_codebook, train_symbols,
                                                 data["train_references"], zero, device=device)
        val_k = gate.m11_g16_prototype_indices(model11, assign_codebook, val_symbols,
                                               data["val_references"], zero, device=device)
        flat = lambda a: a.reshape(-1)
        frequencies, strength, _, _ = m13.fit_recalibrated_frequencies(
            assign_codebook, flat(train_symbols), flat(train_k),
            flat(val_symbols[select_mask]), flat(val_k[select_mask]), alphabet=2 ** bits)
        coding_codebook = m13.build_recalibrated_codebook(
            assign_codebook, frequencies, provenance={"bits": bits, "strength": strength})
        fresh_identity = ma.model_identity(model11, m10k_identity=m10k_identity,
                                           calibration_signature=signature, bits=bits,
                                           codebook=coding_codebook)
        recorded_identity = recorded_provenance.get(f"{bits}bit", {}).get("new_identity")
        match = fresh_identity.hex() == recorded_identity
        report["all_identities_match_recorded"] &= match
        print(f"  {bits}-bit: fresh residual-arm identity {fresh_identity.hex()}  "
             f"recorded {recorded_identity}  match={match}", flush=True)
        report["rate_points"].append({
            "bits": bits, "codebook_size": assign_codebook.size,
            "codebook_alphabet": assign_codebook.alphabet, "recalibration_strength": strength,
            "fresh_residual_identity": fresh_identity.hex(), "recorded_residual_identity": recorded_identity,
            "match": match,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "m17_baseline.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 110)
    status = "CONFIRMED - deterministic, reproduces the recorded M13/M14/M15 deployed arm exactly" \
        if report["all_identities_match_recorded"] else "MISMATCH - STOP before proceeding"
    print(f"BASELINE STATUS: {status}")
    print("=" * 110)
    print(f"\nReport: {path}")
    return 0 if report["all_identities_match_recorded"] else 1


if __name__ == "__main__":
    sys.exit(main())
