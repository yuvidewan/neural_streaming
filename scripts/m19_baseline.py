"""M19 Phase 0 - confirm the M17/M18 baseline hasn't drifted, and reproduce
M17's real-vs-oracle residual pipeline EXACTLY (reusing
`m17_residual_diagnostic.diagnose_residual_oracle` unmodified - not a
re-derivation that could silently diverge) on the same VAL-B sample.

Run:
  ./.venv/Scripts/python.exe scripts/m19_baseline.py
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

DEFAULT_OUTPUT_DIR = Path("outputs/m19_reference_error_audit")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M11_DIR = Path("outputs/m11_autoregressive_entropy")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
DEFAULT_M17_JSON = Path("outputs/m17_residual_oracle_audit/m17_residual_diagnostic.json")
RATE_POINTS = (5, 4, 3)
M11_G16_GROUP_SIZE = 16


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M19 Phase 0: confirm baseline, reproduce M17 exactly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=DEFAULT_M11_DIR)
    parser.add_argument("--m10k-dir", type=Path, default=DEFAULT_M10K_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--m17-json", type=Path, default=DEFAULT_M17_JSON)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--val-sequences", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint),
                        ("--m17-json", args.m17_json)):
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
    gate_script = _load_script("m12_spatial_offline_gate")
    m17 = _load_script("m17_residual_diagnostic")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    val_sequences = discover_sequences(args.manifest, split="val")
    val_b = val_sequences[1::2][:args.val_sequences]
    train_full = discover_sequences(args.manifest, split="train")

    recorded = json.loads(args.m17_json.read_text(encoding="utf-8"))
    recorded_by_bits = {rp["bits"]: rp for rp in recorded["rate_points"]}

    print("=" * 112)
    print("M19 PHASE 0 - BASELINE CONFIRMATION (exact M17 reproduction)")
    print("=" * 112)
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"phase": "M19 Phase 0", "rate_points": [], "all_match": True}

    for bits in args.rate_points:
        calibration = mc.calibrate_grids(
            model, train_full, bits=bits, mode="per_channel", gop_size=10,
            block_size=16, search_range=16, reference_mode="mc",
            max_frames=args.calibration_frames)
        signature = ev.calibration_signature(calibration, bits=bits,
                                             calibration_frames=args.calibration_frames,
                                             quant_mode="per_channel")
        intra_params, intra_entropy_model = calibration["intra_params"], calibration["intra_entropy_model"]
        residual_params = calibration["residual_params"]

        data = md.load_or_collect(model, checkpoint=args.checkpoint, manifest=args.manifest,
                                  bits=bits, device=device, cache_dir=cache_dir, log=lambda m: None)
        train_symbols = data["train_symbols"].astype(np.int64)
        val_symbols = data["val_symbols"].astype(np.int64)
        select_mask, _ = md.split_validation(data)
        channels = train_symbols.shape[1]
        zero = torch.from_numpy(cx.zero_symbols(residual_params, channels))
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
        train_k = gate_script.m11_g16_prototype_indices(model11, assign_codebook, train_symbols,
                                                        data["train_references"], zero, device=device)
        val_k = gate_script.m11_g16_prototype_indices(model11, assign_codebook, val_symbols,
                                                      data["val_references"], zero, device=device)
        flat = lambda a: a.reshape(-1)
        frequencies, strength, _, _ = m13.fit_recalibrated_frequencies(
            assign_codebook, flat(train_symbols), flat(train_k),
            flat(val_symbols[select_mask]), flat(val_k[select_mask]), alphabet=2 ** bits)
        coding_codebook = m13.build_recalibrated_codebook(
            assign_codebook, frequencies, provenance={"bits": bits, "strength": strength})
        residual_identity = ma.model_identity(model11, m10k_identity=m10k_identity,
                                              calibration_signature=signature, bits=bits,
                                              codebook=coding_codebook)

        rows = []
        for sequence in val_b:
            rows.extend(m17.diagnose_residual_oracle(
                mc, ma, m13, mk, model, model11, assign_codebook, coding_codebook, sequence,
                bits=bits, intra_params=intra_params, intra_entropy_model=intra_entropy_model,
                residual_params=residual_params, zero=zero, gop_size=10, block_size=16,
                search_range=16, device=device))

        total_a = sum(r["A_real_bytes"] for r in rows)
        total_c = sum(r["C_full_oracle_bytes"] for r in rows)
        gain = (total_a - total_c) / total_a * 100
        recorded_gain = recorded_by_bits[bits]["channel_gain_percent"]["A_vs_C_full_oracle"]
        recorded_bytes = recorded_by_bits[bits]["total_val_b_bytes"]
        match = (abs(gain - recorded_gain) < 1e-6 and total_a == recorded_bytes["A_real"]
                and total_c == recorded_bytes["C_full_oracle"])
        report["all_match"] &= match
        print(f"  {bits}-bit: identity={residual_identity.hex()}  real={total_a:,} oracle={total_c:,} "
             f"gain={gain:+.4f}%  (recorded: real={recorded_bytes['A_real']:,} "
             f"oracle={recorded_bytes['C_full_oracle']:,} gain={recorded_gain:+.4f}%)  match={match}",
             flush=True)
        report["rate_points"].append({
            "bits": bits, "residual_identity": residual_identity.hex(),
            "fresh_real_bytes": total_a, "fresh_oracle_bytes": total_c, "fresh_gain_percent": gain,
            "recorded_real_bytes": recorded_bytes["A_real"],
            "recorded_oracle_bytes": recorded_bytes["C_full_oracle"],
            "recorded_gain_percent": recorded_gain, "match": match,
        })

    status = "CONFIRMED - byte-exact reproduction of M17" if report["all_match"] else "MISMATCH - STOP"
    print(f"\nBASELINE STATUS: {status}")
    path = args.output_dir / "m19_baseline.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Report: {path}")
    return 0 if report["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
