"""M21 Phases 3-7 - the closed-loop VAL-B rate gate.

Every pre-registered candidate is run through the REAL deployed closed loop:
`.nvct` v2 container, real motion coding, real M11-G16 + K=512 + M13 residual
coding, real arithmetic coder - then decoded back from the container and checked
for exact symbol/reconstruction/reference agreement. Actual coded bytes are the
primary metric; PSNR and MS-SSIM are recorded alongside them because refining
the reference moves the codec along a rate/distortion curve, so a byte saving
bought by a quality loss must be visible rather than hidden.

Two stages, both declared in `m21_refinement.py` before any result was read
(see SCREEN_BITS / STAGE2_ADMISSION_PERCENT there): screen every candidate at
one bit depth, then confirm the non-harmful ones at the other two. The
admission threshold is a fixed number, not "keep the best", so nothing is
selected on the held-out set.

DAVIS TEST is never opened by this script.

Run:
  ./.venv/Scripts/python.exe scripts/m21_sweep.py
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


def compare(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Deltas against the identity control. Positive percentages mean FEWER
    bytes (an improvement); PSNR/MS-SSIM deltas keep their natural sign."""
    def _gain(field):
        base = baseline[field]
        return (base - candidate[field]) / base * 100 if base else 0.0

    return {
        "delta_total_bytes": candidate["total_container_bytes"]
        - baseline["total_container_bytes"],
        "delta_residual_bytes": candidate["total_residual_bytes"]
        - baseline["total_residual_bytes"],
        "delta_p_residual_bytes": candidate["total_p_frame_residual_bytes"]
        - baseline["total_p_frame_residual_bytes"],
        "delta_motion_bytes": candidate["total_motion_bytes"] - baseline["total_motion_bytes"],
        "delta_i_residual_bytes": candidate["total_i_frame_residual_bytes"]
        - baseline["total_i_frame_residual_bytes"],
        "total_stream_gain_percent": _gain("total_container_bytes"),
        "residual_gain_percent": _gain("total_residual_bytes"),
        "p_residual_gain_percent": _gain("total_p_frame_residual_bytes"),
        "motion_gain_percent": _gain("total_motion_bytes"),
        "ideal_bits_gain_percent": _gain("p_frame_ideal_bits"),
        "delta_bpp": candidate["stream_bpp"] - baseline["stream_bpp"],
        "delta_psnr_db": candidate["mean_psnr_db"] - baseline["mean_psnr_db"],
        "delta_msssim": candidate["mean_msssim"] - baseline["mean_msssim"],
    }


def gop_comparison(candidate_gop: dict[str, Any], baseline_gop: dict[str, Any]) -> dict[str, Any]:
    """GOP position 1 (the frame after an I-frame, where M16/M17/M19 all found a
    spike) split from positions 2-9."""
    def _bucket(source, positions):
        out = {"residual_bytes": 0, "motion_bytes": 0, "frames": 0}
        for key, values in source.items():
            if int(key) in positions:
                for field in out:
                    out[field] += values[field]
        return out

    result = {}
    for name, positions in (("boundary", {1}), ("ordinary", set(range(2, 10)))):
        base = _bucket(baseline_gop, positions)
        cand = _bucket(candidate_gop, positions)
        result[name] = {
            "frames": base["frames"],
            "baseline_residual_bytes": base["residual_bytes"],
            "candidate_residual_bytes": cand["residual_bytes"],
            "delta_residual_bytes": cand["residual_bytes"] - base["residual_bytes"],
            "residual_gain_percent": ((base["residual_bytes"] - cand["residual_bytes"])
                                      / base["residual_bytes"] * 100
                                      if base["residual_bytes"] else 0.0),
            "baseline_motion_bytes": base["motion_bytes"],
            "candidate_motion_bytes": cand["motion_bytes"],
            "delta_motion_bytes": cand["motion_bytes"] - base["motion_bytes"],
        }
    return result


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M21 Phases 3-7: the closed-loop VAL-B rate gate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt"))
    parser.add_argument("--m11-dir", type=Path, default=Path("outputs/m11_autoregressive_entropy"))
    parser.add_argument("--m10k-dir", type=Path, default=Path("outputs/m10k_learned_entropy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m21_reference_refinement"))
    parser.add_argument("--output-name", type=str, default="m21_sweep.json")
    parser.add_argument("--stage", choices=["screen", "confirm", "both"], default="both")
    parser.add_argument("--candidates", type=str, nargs="*", default=None,
                        help="Restrict to these pre-registered names (default: all).")
    parser.add_argument("--rate-points", type=int, nargs="+", default=None,
                        help="Override the declared screen/confirm bit depths.")
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
    parser.add_argument("--keep-streams", action="store_true",
                        help="Keep the per-candidate .nvct files (default: delete after check).")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    mc = _load_script("m10h_motion_compensation")
    m13 = _load_script("m13_recalibration")
    md = _load_script("m11_data")
    m21 = _load_script("m21_refinement")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    val_b = m21.val_b_sequences(args.manifest, count=args.val_sequences,
                                max_frames=args.val_frames_per_sequence)
    train_full = discover_sequences(args.manifest, split="train")
    motion_train = m21.broad_train_sequences(
        args.manifest, frames_per_sequence=args.train_frames_per_sequence)
    all_candidates = [c for c in m21.CANDIDATES
                      if args.candidates is None or c.name in args.candidates]
    if args.candidates:
        missing = set(args.candidates) - {c.name for c in m21.CANDIDATES}
        if missing:
            print(f"[ERROR] not pre-registered: {sorted(missing)}", file=sys.stderr)
            return 1

    print("=" * 138)
    print("M21 PHASES 3-7 - CLOSED-LOOP VAL-B RATE GATE (held out; DAVIS TEST never opened)")
    print("=" * 138)
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}")
    print(f"  pre-registered candidates: {len(all_candidates)}")
    print(f"  declared protocol: screen at {m21.SCREEN_BITS}-bit, confirm "
          f"{m21.CONFIRM_BITS} for candidates with total-stream effect >= "
          f"{m21.STAGE2_ADMISSION_PERCENT}%", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = args.output_dir / "sweep_streams"
    report: dict[str, Any] = {
        "phase": "M21 Phases 3-7",
        "val_b_sequence_ids": [s.sequence_id for s in val_b],
        "declared_screen_bits": m21.SCREEN_BITS,
        "declared_confirm_bits": list(m21.CONFIRM_BITS),
        "declared_stage2_admission_percent": m21.STAGE2_ADMISSION_PERCENT,
        "candidate_definitions": [c.to_dict() for c in all_candidates],
        "rate_points": [],
    }

    if args.rate_points:
        schedule = [(bits, all_candidates) for bits in args.rate_points]
        admitted_names = None
    else:
        schedule = [(m21.SCREEN_BITS, all_candidates)]
        admitted_names = "pending"

    index = 0
    while index < len(schedule):
        bits, candidates = schedule[index]
        index += 1
        started = time.perf_counter()
        rig = m21.prepare_rate_point(
            model, bits=bits, manifest=args.manifest, checkpoint=args.checkpoint,
            m11_dir=args.m11_dir, m10k_dir=args.m10k_dir, device=device, cache_dir=cache_dir,
            train_full=train_full, motion_train=motion_train,
            calibration_frames=args.calibration_frames, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            motion_cache=args.output_dir / "m21_deployed_motion_table.json")

        runs: dict[str, Any] = {}
        for refinement in candidates:
            run_started = time.perf_counter()
            runs[refinement.name] = m21.run_candidate(
                mc, m13, model, val_b, rig["spec"], refinement, stream_dir,
                intra_params=rig["intra_params"],
                intra_entropy_model=rig["intra_entropy_model"],
                residual_params=rig["residual_params"],
                motion_entropy_model=rig["motion_entropy_model"], bits=bits,
                gop_size=args.gop, block_size=args.block_size, search_range=args.search_range)
            runs[refinement.name]["seconds"] = time.perf_counter() - run_started
            if not args.keep_streams:
                for path in stream_dir.glob(f"{refinement.name}_{bits}bit_*.nvct"):
                    path.unlink()

        baseline = runs["identity"]["aggregate"] if "identity" in runs else None
        rows = []
        for refinement in candidates:
            run = runs[refinement.name]
            summary = run["aggregate"]
            delta = compare(summary, baseline) if baseline else {}
            rows.append({
                "candidate": refinement.name, "family": refinement.family,
                "domain": refinement.domain, "definition": refinement.definition,
                "aggregate": summary, "per_sequence": run["per_sequence"],
                "gop_position": run["gop_position"],
                "gop_comparison": (gop_comparison(run["gop_position"],
                                                  runs["identity"]["gop_position"])
                                   if baseline else {}),
                "decode_checks": run["decode_checks"],
                "decoder_compatible": all(
                    run["decode_checks"][k] for k in
                    ("symbols_exact", "reconstruction_exact", "references_exact")),
                "seconds": run["seconds"], **delta,
                "verdict": m21.verdict(delta.get("total_stream_gain_percent", 0.0)),
            })

        print(f"\n  ================ {bits}-bit  "
              f"({time.perf_counter() - started:.1f}s, residual={rig['residual_identity']}, "
              f"motion={rig['motion_identity']}) ================")
        print(f"    {'candidate':>22} {'total bytes':>12} {'d total':>9} {'stream %':>9} "
              f"{'d resid':>9} {'d motion':>9} {'dPSNR':>8} {'dMS-SSIM':>10} {'dec ok':>7} "
              f"{'verdict':>10}")
        for row in rows:
            summary = row["aggregate"]
            print(f"    {row['candidate']:>22} {summary['total_container_bytes']:>12,} "
                  f"{row.get('delta_total_bytes', 0):>+9,} "
                  f"{row.get('total_stream_gain_percent', 0.0):>+9.4f} "
                  f"{row.get('delta_residual_bytes', 0):>+9,} "
                  f"{row.get('delta_motion_bytes', 0):>+9,} "
                  f"{row.get('delta_psnr_db', 0.0):>+8.4f} "
                  f"{row.get('delta_msssim', 0.0):>+10.6f} "
                  f"{str(row['decoder_compatible']):>7} {row['verdict']:>10}")
        print(flush=True)

        report["rate_points"].append({
            "bits": bits, "stage": "screen" if bits == m21.SCREEN_BITS else "confirm",
            "residual_identity": rig["residual_identity"],
            "intra_identity": rig["intra_identity"],
            "motion_identity": rig["motion_identity"],
            "assign_codebook_id": rig["assign_codebook_id"],
            "coding_codebook_id": rig["coding_codebook_id"],
            "candidates": rows,
        })

        if admitted_names == "pending" and args.stage in ("both", "confirm"):
            candidates_only = [row for row in rows if row["candidate"] != "identity"]
            passed = [row["candidate"] for row in candidates_only
                      if row["total_stream_gain_percent"] >= m21.STAGE2_ADMISSION_PERCENT]
            ranked = sorted(candidates_only,
                            key=lambda r: -r["total_stream_gain_percent"])
            floor = [r["candidate"] for r in ranked[:m21.STAGE2_MINIMUM_CANDIDATES]]
            carried = [name for name in floor if name not in passed]
            admitted = ["identity"] + passed + carried
            admitted_names = admitted
            report["stage2_admitted"] = admitted
            report["stage2_passed_threshold"] = passed
            report["stage2_carried_by_floor"] = carried
            report["stage2_rule"] = (
                f"identity control plus every candidate with a Stage-1 total-stream effect >= "
                f"{m21.STAGE2_ADMISSION_PERCENT}%; if fewer than "
                f"{m21.STAGE2_MINIMUM_CANDIDATES} qualify, the best-ranked are carried forward "
                f"as a labelled robustness check, never as a selection")
            print(f"  STAGE 2 ADMISSION ({len(admitted)} of {len(rows)}): "
                  f"threshold-passed={passed}  carried-by-floor={carried}", flush=True)
            admitted_set = [c for c in all_candidates if c.name in admitted]
            for confirm_bits in m21.CONFIRM_BITS:
                schedule.append((confirm_bits, admitted_set))

    path = args.output_dir / args.output_name
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
