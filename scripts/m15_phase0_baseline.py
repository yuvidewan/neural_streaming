"""M15 Phase 0 - confirm the M13/M14 baseline this milestone builds on has
not silently drifted before spending any compute on new experiments.

Re-derives (never trusts blindly): the M14-recorded calibration signatures
and intra/motion entropy identities per bit depth (via a fresh
`calibrate_grids` call - Policy A, the deployed recipe, made explicit), and
confirms they match `outputs/m14_entropy_audit/m14_entropy_audit.json`
byte-for-byte. Also re-loads M14's own recorded full-DAVIS headline numbers
(never re-runs that benchmark - Phase 0 confirms the EXISTING record is
still consistent with a fresh calibration, it does not repeat compute
M14 already paid for).

Run:
  ./.venv/Scripts/python.exe scripts/m15_phase0_baseline.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m15_calibration_policy")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_AUDIT_JSON = Path("outputs/m14_entropy_audit/m14_entropy_audit.json")
DEFAULT_DAVIS_JSON = Path("outputs/m14_entropy_audit/m14_davis_benchmark.json")
RATE_POINTS = (5, 4, 3)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M15 Phase 0: confirm the M13/M14 baseline before new experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--davis-json", type=Path, default=DEFAULT_DAVIS_JSON)
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
    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint),
                        ("--audit-json", args.audit_json), ("--davis-json", args.davis_json)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    mc = _load_script("m10h_motion_compensation")
    ev = _load_script("m10l_evaluate")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    recorded_audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    recorded_davis = json.loads(args.davis_json.read_text(encoding="utf-8"))
    recorded_by_bits = {rp["bits"]: rp for rp in recorded_audit["rate_points"]}

    calibration_sequences = discover_sequences(args.manifest, split="train")
    train_total = sum(s.frame_count for s in calibration_sequences)

    print("=" * 110, flush=True)
    print("M15 PHASE 0 - BASELINE CONFIRMATION (re-derive M14's identities, don't trust the record)")
    print("=" * 110)
    print(f"  TRAIN: {len(calibration_sequences)} sequences, {train_total} frames total")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "M15 Phase 0 baseline confirmation",
        "checkpoint": str(args.checkpoint), "manifest": str(args.manifest),
        "train_sequences": len(calibration_sequences), "train_total_frames": train_total,
        "rate_points": [], "all_identities_match_recorded": True,
        "recorded_full_davis_bd_rate": recorded_davis.get("bd_rate"),
        "recorded_full_davis_motion_byte_gain": recorded_davis.get("motion_byte_gain"),
        "recorded_full_davis_invariants": recorded_davis.get("invariants"),
    }

    for bits in args.rate_points:
        calibration = mc.calibrate_grids(
            model, calibration_sequences, bits=bits, mode="per_channel", gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, reference_mode="mc",
            max_frames=args.calibration_frames)
        signature = ev.calibration_signature(calibration, bits=bits,
                                             calibration_frames=args.calibration_frames,
                                             quant_mode="per_channel")
        fresh = {
            "calibration_signature": signature,
            "intra_entropy_model_id": calibration["intra_entropy_model"].model_id().hex(),
            "motion_entropy_model_id": calibration["motion_entropy_model"].model_id().hex(),
            "residual_entropy_model_id": calibration["residual_entropy_model"].model_id().hex(),
        }
        recorded = recorded_by_bits[bits]
        recorded_flat = {
            "calibration_signature": recorded["calibration_signature"],
            "intra_entropy_model_id": recorded["calibration_provenance"]["intra_entropy_model_id"],
            "motion_entropy_model_id": recorded["calibration_provenance"]["motion_entropy_model_id"],
            "residual_entropy_model_id": recorded["calibration_provenance"]["residual_entropy_model_id"],
        }
        matches = {key: fresh[key] == recorded_flat[key] for key in fresh}
        all_match = all(matches.values())
        report["all_identities_match_recorded"] &= all_match
        print(f"  {bits}-bit: fresh vs recorded identities match = {all_match}  {matches}", flush=True)
        report["rate_points"].append({"bits": bits, "fresh": fresh, "recorded": recorded_flat,
                                      "matches": matches, "all_match": all_match})

    print()
    print("=" * 110)
    status = "CONFIRMED - deterministic, matches the recorded M14 baseline exactly" \
        if report["all_identities_match_recorded"] else \
        "MISMATCH - the recorded baseline has drifted, STOP before proceeding"
    print(f"BASELINE STATUS: {status}")
    print("=" * 110)

    path = args.output_dir / "m15_phase0_baseline.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0 if report["all_identities_match_recorded"] else 1


if __name__ == "__main__":
    sys.exit(main())
