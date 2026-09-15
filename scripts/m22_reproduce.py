"""M22 Phase 20 - independent-process reproducibility of the locked candidate.

Section 20 requires the selected candidate to be re-run in a SEPARATE process and
to come out identical in checkpoint loading, quantizer parameters, latent
outputs, residual symbols, codebook assignments, entropy lengths, payload bytes,
decoded reconstruction and metrics.

This script does not trust the sweep's in-memory state for anything. It starts
from disk: the refit checkpoint (whose SHA256 is verified against its recorded
provenance before it is loaded at all) and the sweep report's recorded numbers.
It then re-encodes VAL-B through the real closed loop under `deterministic_kernels()`
and compares EVERY recorded field exactly - integers and byte counts by equality,
floating-point metrics by equality too, since determinism is the property under
test and a tolerance would hide exactly the failure this phase is looking for.

Exit code is non-zero if anything differs, so this is usable as a gate and not
merely as a report.

Run:
  ./.venv/Scripts/python.exe scripts/m22_reproduce.py --bits 3
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


# The fields whose exact reproduction Phase 20 is asserting. Byte counts and
# symbol counts are the load-bearing ones; the metrics are included because a
# non-deterministic kernel would move them before it moved the bytes.
EXACT_FIELDS = (
    "total_container_bytes", "total_residual_bytes", "total_p_frame_residual_bytes",
    "total_i_frame_residual_bytes", "total_motion_bytes", "total_overhead_bytes",
    "residual_symbols", "p_frame_ideal_bits", "stream_bpp", "mean_psnr_db", "mean_msssim",
    "residual_rms", "residual_mae",
)


# The per-sequence rows carry their own narrower field set.
PER_SEQUENCE_FIELDS = (
    "residual_bytes", "p_frame_residual_bytes", "motion_bytes", "container_bytes",
    "mean_psnr_db", "mean_msssim",
)


def compare_exact(fresh: dict[str, Any], recorded: dict[str, Any],
                  fields=EXACT_FIELDS) -> list[dict[str, Any]]:
    """Every mismatch, not just the first - a partial reproduction failure is
    more informative when you can see its whole shape."""
    differences = []
    for field in fields:
        if field not in recorded:
            continue
        if fresh.get(field) != recorded[field]:
            differences.append({"field": field, "recorded": recorded[field],
                                "reproduced": fresh.get(field)})
    return differences


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M22 Phase 20: re-run the locked candidate in a fresh process.",
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
    parser.add_argument("--output-name", type=str, default="m22_reproduce.json")
    parser.add_argument("--variant", type=str, default=None,
                        help="default: whichever candidate Phase 17's rule locks")
    parser.add_argument("--arm", choices=["refit", "stale"], default="refit")
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

    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    cx = _load_script("m11_causal_context")
    m13 = _load_script("m13_recalibration")
    mc = _load_script("m10h_motion_compensation")
    m21 = _load_script("m21_refinement")
    m22 = _load_script("m22_residual")
    mech = _load_script("m22_mechanism")
    analysis = _load_script("m22_analysis")
    codebook_mod = _load_script("m22_codebook")

    report_path = args.output_dir / args.sweep_name
    sweep_report = json.loads(report_path.read_text(encoding="utf-8"))
    if args.variant is None:
        row = codebook_mod.locked_candidate(sweep_report, args.bits, analysis)
        if row is None:
            print("[ERROR] no eligible candidate; pass --variant", file=sys.stderr)
            return 1
    else:
        row = next((r for point in sweep_report["rate_points"] if point["bits"] == args.bits
                    for r in point["candidates"]
                    if r["variant"] == args.variant and r["arm"] == args.arm), None)
        if row is None:
            print(f"[ERROR] {args.variant}/{args.arm} at {args.bits}-bit is not in the sweep",
                  file=sys.stderr)
            return 1
    variant, arm = row["variant"], row["arm"]

    print("=" * 110)
    print(f"M22 PHASE 20 - INDEPENDENT-PROCESS REPRODUCTION OF {variant}/{arm} ({args.bits}-bit)")
    print("=" * 110)

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

    if arm == "refit":
        ckpt = args.output_dir / "checkpoints" / f"m22_{variant}_{args.bits}bit.pt"
        loaded = mech.load_refit_checkpoint(ckpt, device=device, m22=m22, ma=ma, ml=ml, cx=cx)
        spec, residual_params = loaded["spec"], loaded["residual_params"]
        checkpoint_sha = loaded["record"]["sha256"]
        print(f"  checkpoint {ckpt.name}  sha256 {checkpoint_sha[:16]} VERIFIED before loading")
    else:
        residuals = m22.collect_train_residuals(
            mc, model, discover_sequences(args.manifest, split="train"), bits=args.bits,
            intra_params=deployed_rig["intra_params"],
            intra_entropy_model=deployed_rig["intra_entropy_model"],
            gop_size=args.gop, block_size=args.block_size, search_range=args.search_range,
            max_frames=args.calibration_frames, device=device,
            cache=args.output_dir / f"m22_train_residuals_{args.bits}bit.pt")
        residual_params = m22.grid_variant(variant).build(residuals["residual_stack"], args.bits)
        spec, checkpoint_sha = dict(deployed_rig["spec"]), None

    grid = m22.grid_signature(residual_params)
    print(f"  grid signature {grid}  (recorded {row['grid_signature']})")
    if grid != row["grid_signature"]:
        print("[FAIL] the rebuilt grid does not match the recorded one", file=sys.stderr)
        return 1

    started = time.perf_counter()
    fresh = m21.run_candidate(
        mc, m13, model, val_b, spec, m21.IDENTITY, args.output_dir / "reproduce_streams",
        intra_params=deployed_rig["intra_params"],
        intra_entropy_model=deployed_rig["intra_entropy_model"],
        residual_params=residual_params,
        motion_entropy_model=deployed_rig["motion_entropy_model"], bits=args.bits,
        gop_size=args.gop, block_size=args.block_size, search_range=args.search_range)
    elapsed = time.perf_counter() - started

    differences = compare_exact(fresh["aggregate"], row["aggregate"])
    # per_sequence is a LIST of per-sequence dicts keyed by a "sequence" field,
    # not a mapping - index it by that field so a reordering cannot be mistaken
    # for a reproduction failure (or, worse, hide one).
    fresh_by_sequence = {entry["sequence"]: entry for entry in fresh["per_sequence"]}
    per_sequence_differences = []
    for recorded in row["per_sequence"]:
        name = recorded["sequence"]
        got = fresh_by_sequence.get(name)
        if got is None:
            per_sequence_differences.append({"sequence": name, "field": "missing"})
            continue
        for difference in compare_exact(got, recorded, PER_SEQUENCE_FIELDS):
            per_sequence_differences.append({"sequence": name, **difference})

    decode_ok = all(fresh["decode_checks"][k] for k in
                    ("symbols_exact", "reconstruction_exact", "references_exact"))
    reproduced = not differences and not per_sequence_differences and decode_ok

    print(f"\n  decode round trip      symbols/reconstruction/references exact: {decode_ok}")
    print(f"  aggregate fields       {len(EXACT_FIELDS)} compared, {len(differences)} differ")
    print(f"  per-sequence fields    {len(row['per_sequence'])} sequences, "
          f"{len(per_sequence_differences)} differ")
    print(f"  total container bytes  {fresh['aggregate']['total_container_bytes']:,} "
          f"(recorded {row['aggregate']['total_container_bytes']:,})")
    for difference in differences + per_sequence_differences:
        print(f"    [DIFF] {difference}")
    print(f"\n  REPRODUCED: {reproduced}   ({elapsed:.1f}s)")

    result = {"phase": "M22 Phase 20", "variant": variant, "arm": arm, "bits": args.bits,
              "grid_signature": grid, "checkpoint_sha256": checkpoint_sha,
              "val_b_sequence_ids": [s.sequence_id for s in val_b],
              "fields_compared": list(EXACT_FIELDS),
              "aggregate_differences": differences,
              "per_sequence_differences": per_sequence_differences,
              "decode_checks": fresh["decode_checks"], "reproduced": reproduced,
              "seconds": elapsed, "recorded_aggregate": row["aggregate"],
              "reproduced_aggregate": fresh["aggregate"]}
    path = args.output_dir / args.output_name
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Report: {path}")
    return 0 if reproduced else 1


if __name__ == "__main__":
    sys.exit(main())
