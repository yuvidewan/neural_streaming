"""M22 Phase 15 - quantifying the codebook incompatibility, without repairing it.

Section 15 of the milestone is explicit: if a new residual grid makes the
deployed K=512 assignment prototypes and M13 coding frequencies invalid, the
incompatibility must be RECORDED and QUANTIFIED rather than silently
recalibrated, and whether recalibration is a separately required phase must be
decided on that evidence.

The sweep's STALE arm already prices the incompatibility in actual container
bytes. This script decomposes that price into its two causes, over the same
held-out VAL-B P-frames:

  assignment drift   how often the deployed K=512 codebook routes a position to
                     a different prototype than the refitted one does, once the
                     grid has moved the symbol distribution underneath it;
  table mismatch     the cross-entropy penalty of coding the NEW grid's symbols
                     with the DEPLOYED M13 frequencies instead of refitted ones.

Neither number changes any artifact. Nothing is recalibrated here.

This script deliberately lives outside `m22_sweep.py` so that running it cannot
alter that file's hash, which `m22_residual.code_identity()` records inside every
refit checkpoint.

Run:
  ./.venv/Scripts/python.exe scripts/m22_codebook.py --variant broad_p001 --bits 3
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


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def locked_candidate(report: dict[str, Any], bits: int, analysis) -> dict[str, Any] | None:
    """The row Phase 17's rule selects, so Phase 15 describes the candidate that
    was actually locked rather than one chosen here."""
    for point in report["rate_points"]:
        if point["bits"] == bits:
            refit_rows = [r for r in point["candidates"] if r["arm"] == "refit"]
            return analysis.select_candidate(refit_rows)
    return None


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M22 Phase 15: quantify (do not repair) the codebook incompatibility.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt"))
    parser.add_argument("--m11-dir", type=Path, default=Path("outputs/m11_autoregressive_entropy"))
    parser.add_argument("--m10k-dir", type=Path, default=Path("outputs/m10k_learned_entropy"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/m22_residual_freeze_lift"))
    parser.add_argument("--sweep-name", type=str, default="m22_sweep.json")
    parser.add_argument("--output-name", type=str, default="m22_codebook.json")
    parser.add_argument("--variant", type=str, default=None,
                        help="default: whichever candidate Phase 17's rule locks")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--val-sequences", type=int, default=4)
    parser.add_argument("--val-frames-per-sequence", type=int, default=None)
    parser.add_argument("--train-frames-per-sequence", type=int, default=8)
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

    mc = _load_script("m10h_motion_compensation")
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    cx = _load_script("m11_causal_context")
    m13 = _load_script("m13_recalibration")
    m21 = _load_script("m21_refinement")
    m22 = _load_script("m22_residual")
    sweep = _load_script("m22_sweep")
    mech = _load_script("m22_mechanism")
    analysis = _load_script("m22_analysis")

    report_path = args.output_dir / args.sweep_name
    if not report_path.is_file():
        print(f"[ERROR] no sweep report at {report_path}; run m22_sweep.py first", file=sys.stderr)
        return 1
    sweep_report = json.loads(report_path.read_text(encoding="utf-8"))

    variant = args.variant
    if variant is None:
        row = locked_candidate(sweep_report, args.bits, analysis)
        if row is None:
            print("[ERROR] no eligible candidate in the sweep report; pass --variant",
                  file=sys.stderr)
            return 1
        variant = row["variant"]
        print(f"  locked candidate at {args.bits}-bit (Phase 17 rule): {variant}")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or _load_script("m11_data").DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    val_b = m21.val_b_sequences(args.manifest, count=args.val_sequences,
                                max_frames=args.val_frames_per_sequence)
    deployed_rig = m21.prepare_rate_point(
        model, bits=args.bits, manifest=args.manifest, checkpoint=args.checkpoint,
        m11_dir=args.m11_dir, m10k_dir=args.m10k_dir, device=device, cache_dir=cache_dir,
        train_full=discover_sequences(args.manifest, split="train"),
        motion_train=m21.broad_train_sequences(
            args.manifest, frames_per_sequence=args.train_frames_per_sequence),
        calibration_frames=args.calibration_frames, gop_size=args.gop,
        block_size=args.block_size, search_range=args.search_range,
        motion_cache=args.output_dir / "m22_deployed_motion_table.json")

    ckpt = args.output_dir / "checkpoints" / f"m22_{variant}_{args.bits}bit.pt"
    if not ckpt.is_file():
        print(f"[ERROR] no refit checkpoint at {ckpt}", file=sys.stderr)
        return 1
    loaded = mech.load_refit_checkpoint(ckpt, device=device, m22=m22, ma=ma, ml=ml, cx=cx)
    residual_params = loaded["residual_params"]

    intra = {"intra_params": deployed_rig["intra_params"],
             "intra_entropy_model": deployed_rig["intra_entropy_model"]}
    refit_spec = {**loaded["spec"], **intra}
    deployed_spec = {**deployed_rig["spec"], **intra}

    print("=" * 110)
    print(f"M22 PHASE 15 - CODEBOOK INCOMPATIBILITY UNDER THE {variant!r} GRID "
          f"({args.bits}-bit, VAL-B)")
    print("=" * 110)
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}")
    print(f"  deployed grid  {m22.grid_signature(deployed_rig['residual_params'])}   "
          f"candidate grid {m22.grid_signature(residual_params)}")
    print("  NOTHING IS RECALIBRATED HERE - this measures the mismatch only.", flush=True)

    started = time.perf_counter()
    result = sweep.codebook_incompatibility(
        mc, m13, m21, model, refit_spec, residual_params, val_b, bits=args.bits,
        gop_size=args.gop, block_size=args.block_size, search_range=args.search_range,
        device=device, deployed_spec=deployed_spec)
    result.update({"variant": variant, "bits": args.bits,
                   "seconds": time.perf_counter() - started,
                   "deployed_grid_signature": m22.grid_signature(
                       deployed_rig["residual_params"]),
                   "candidate_grid_signature": m22.grid_signature(residual_params),
                   "val_b_sequence_ids": [s.sequence_id for s in val_b],
                   "checkpoint_sha256": loaded["record"]["sha256"]})

    print(f"\n  positions measured            {result['positions']:,}")
    print(f"  K=512 assignment drift        {result['assignment_drift'] * 100:.4f}% of positions "
          "routed to a different prototype")
    print(f"  refitted tables               {result['refit_bits_per_symbol']:.6f} bits/symbol")
    print(f"  deployed tables (same symbols){result['deployed_bits_per_symbol']:>10.6f} "
          "bits/symbol")
    print(f"  deployed-table penalty        {result['deployed_table_penalty_percent']:+.4f}%")
    print(f"  symbol entropy under the new grid {result['symbol_entropy_bits']:.4f} bits")
    print(f"  ({result['seconds']:.1f}s)")

    path = args.output_dir / args.output_name
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
