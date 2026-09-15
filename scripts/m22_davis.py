"""M22 Phase 19 - the full 719-frame DAVIS TEST run for the locked candidate.

This is the ONLY script in M22 that opens the TEST split, and it refuses to run
unless the candidate was already locked by the declared Phase 17 rule and cleared
the 0.5% VAL-B gate. The lock is read from `m22_analysis.json` rather than passed
in, so TEST cannot be opened for a candidate chosen after the fact.

Two arms, both through the real deployed closed loop and both decoded back from
their containers:

  baseline   the frozen production codec - deployed grid, deployed M10K/G16/
             K=512/M13. Cross-checked byte-for-byte against the DAVIS totals
             M14 recorded and M21 reproduced; a mismatch here invalidates
             everything else, so it is reported, not assumed.
  candidate  the locked grid with its refitted stack, loaded from the
             SHA256-verified checkpoint written during Stage 2.

Reports per-sequence and per-GOP-position results, BD-rate at both metrics, and
runtime, against the frozen baseline - not against another candidate.

Run:
  ./.venv/Scripts/python.exe scripts/m22_davis.py --candidate symmetric_p01
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

# What M14 recorded and M21 reproduced for the frozen production codec on the
# full DAVIS TEST split. The baseline arm must land on these exactly.
RECORDED_DAVIS = {
    5: {"total_container_bytes": 4_292_887, "total_p_frame_residual_bytes": 3_527_769,
        "total_motion_bytes": 164_010, "total_i_frame_residual_bytes": 584_917},
    4: {"total_container_bytes": 3_009_891, "total_p_frame_residual_bytes": 2_398_204,
        "total_motion_bytes": 166_842, "total_i_frame_residual_bytes": 428_654},
    3: {"total_container_bytes": 1_900_744, "total_p_frame_residual_bytes": 1_439_926,
        "total_motion_bytes": 171_612, "total_i_frame_residual_bytes": 273_015},
}


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M22 Phase 19: full DAVIS TEST for the locked candidate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt"))
    parser.add_argument("--m11-dir", type=Path, default=Path("outputs/m11_autoregressive_entropy"))
    parser.add_argument("--m10k-dir", type=Path, default=Path("outputs/m10k_learned_entropy"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/m22_residual_freeze_lift"))
    parser.add_argument("--analysis-name", type=str, default="m22_analysis.json")
    parser.add_argument("--output-name", type=str, default="m22_davis.json")
    parser.add_argument("--rate-points", type=int, nargs="+", default=[5, 4, 3])
    parser.add_argument("--calibration-frames", type=int, default=400)
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
    md = _load_script("m11_data")
    m21 = _load_script("m21_refinement")
    m22 = _load_script("m22_residual")
    mech = _load_script("m22_mechanism")
    sweep = _load_script("m22_sweep")
    analysis_mod = _load_script("m22_analysis")

    # --- the lock, verified before TEST is touched -----------------------------------------
    analysis_path = args.output_dir / args.analysis_name
    if not analysis_path.is_file():
        print(f"[ERROR] {analysis_path} not found - the candidate must be locked by the "
              "declared Phase 17 rule BEFORE DAVIS TEST is opened.", file=sys.stderr)
        return 1
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    locked = sorted((entry["selected"] for entry in analysis["rate_points"]
                     if "selected" in entry),
                    key=lambda s: -s["total_stream_gain_percent"])
    if not locked:
        print("[ERROR] no candidate was selected; refusing to open TEST.", file=sys.stderr)
        return 1
    if locked[0]["variant"] != args.candidate:
        print(f"[ERROR] --candidate {args.candidate!r} is not the locked candidate "
              f"({locked[0]['variant']!r}); refusing to open TEST.", file=sys.stderr)
        return 1
    if not analysis.get("full_davis_gated"):
        print("[ERROR] the VAL-B gate did not clear 0.5%; refusing to open TEST.",
              file=sys.stderr)
        return 1

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    # The ONLY TEST access in M22, after the lock.
    davis = discover_sequences(args.manifest, split="test")
    train_full = discover_sequences(args.manifest, split="train")
    motion_train = m21.broad_train_sequences(
        args.manifest, frames_per_sequence=args.train_frames_per_sequence)

    print("=" * 134)
    print(f"M22 PHASE 19 - FULL DAVIS TEST, locked candidate {args.candidate!r}")
    print("=" * 134)
    print(f"  sequences: {len(davis)}  frames: {sum(s.frame_count for s in davis)}")
    print(f"  locked by the declared rule at {locked[0]['total_stream_gain_percent']:+.4f}% "
          "total stream on VAL-B", flush=True)

    stream_dir = args.output_dir / "davis_streams"
    report: dict[str, Any] = {
        "phase": "M22 Phase 19", "candidate": args.candidate,
        "sequences": len(davis), "frames": sum(s.frame_count for s in davis),
        "locked_val_b_gain_percent": locked[0]["total_stream_gain_percent"],
        "baseline_matches_recorded": True, "rate_points": [],
    }
    curves_psnr: dict[str, list[tuple[float, float]]] = {"baseline": [], "candidate": []}
    curves_msssim: dict[str, list[tuple[float, float]]] = {"baseline": [], "candidate": []}
    report_path = args.output_dir / args.output_name

    def persist() -> None:
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    for bits in args.rate_points:
        rig = m21.prepare_rate_point(
            model, bits=bits, manifest=args.manifest, checkpoint=args.checkpoint,
            m11_dir=args.m11_dir, m10k_dir=args.m10k_dir, device=device, cache_dir=cache_dir,
            train_full=train_full, motion_train=motion_train,
            calibration_frames=args.calibration_frames, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            motion_cache=args.output_dir / "m22_deployed_motion_table.json")

        ckpt = args.output_dir / "checkpoints" / f"m22_{args.candidate}_{bits}bit.pt"
        loaded = mech.load_refit_checkpoint(ckpt, device=device, m22=m22, ma=ma, ml=ml, cx=cx)
        print(f"\n  ---- {bits}-bit ----  checkpoint {ckpt.name} "
              f"sha256 {loaded['record']['sha256'][:16]} VERIFIED", flush=True)

        arms = {
            "baseline": (rig["spec"], rig["residual_params"]),
            "candidate": (loaded["spec"], loaded["residual_params"]),
        }
        runs = {}
        for name, (spec, residual_params) in arms.items():
            started = time.perf_counter()
            runs[name] = m21.run_candidate(
                mc, m13, model, davis, spec, m21.IDENTITY, stream_dir,
                intra_params=rig["intra_params"],
                intra_entropy_model=rig["intra_entropy_model"],
                residual_params=residual_params,
                motion_entropy_model=rig["motion_entropy_model"], bits=bits,
                gop_size=args.gop, block_size=args.block_size, search_range=args.search_range)
            runs[name]["seconds"] = time.perf_counter() - started
            for stream_path in stream_dir.glob(f"identity_{bits}bit_*.nvct"):
                stream_path.unlink()
            aggregate = runs[name]["aggregate"]
            print(f"    [{name}] {aggregate['total_container_bytes']:,} bytes  "
                  f"BPP {aggregate['stream_bpp']:.6f}  PSNR {aggregate['mean_psnr_db']:.4f}  "
                  f"MS-SSIM {aggregate['mean_msssim']:.6f}  "
                  f"({runs[name]['seconds']:.1f}s)", flush=True)

        base, cand = runs["baseline"]["aggregate"], runs["candidate"]["aggregate"]
        expected = RECORDED_DAVIS.get(bits, {})
        mismatches = {field: (base[field], value) for field, value in expected.items()
                      if base[field] != value}
        report["baseline_matches_recorded"] &= not mismatches
        if mismatches:
            print(f"    [WARN] the baseline arm does NOT reproduce the recorded DAVIS totals: "
                  f"{mismatches}", file=sys.stderr, flush=True)

        delta = sweep.compare(cand, base)
        guard = m22.distortion_regression(delta["delta_psnr_db"], delta["delta_msssim"])
        gop = analysis_mod.gop_split(runs["candidate"]["gop_position"],
                                     runs["baseline"]["gop_position"])
        curves_psnr["baseline"].append((base["stream_bpp"], base["mean_psnr_db"]))
        curves_psnr["candidate"].append((cand["stream_bpp"], cand["mean_psnr_db"]))
        curves_msssim["baseline"].append((base["stream_bpp"], base["mean_msssim"]))
        curves_msssim["candidate"].append((cand["stream_bpp"], cand["mean_msssim"]))

        print(f"    delta {delta['delta_total_bytes']:+,} bytes  "
              f"{delta['total_stream_gain_percent']:+.4f}% total stream  "
              f"residual {delta['p_residual_gain_percent']:+.4f}%  "
              f"dPSNR {delta['delta_psnr_db']:+.4f}  dMS-SSIM {delta['delta_msssim']:+.6f}  "
              f"guard={'FAIL' if guard else 'PASS'}  "
              f"verdict={m22.verdict(delta['total_stream_gain_percent'])}")
        print(f"    GOP  boundary {gop['boundary']['residual_gain_percent']:+.4f}%  "
              f"ordinary {gop['ordinary']['residual_gain_percent']:+.4f}%", flush=True)

        report["rate_points"].append({
            "bits": bits, "baseline": base, "candidate": cand, "delta": delta,
            "distortion_regression": guard,
            "verdict": m22.verdict(delta["total_stream_gain_percent"]),
            "gop_split": gop,
            "per_sequence": {"baseline": runs["baseline"]["per_sequence"],
                             "candidate": runs["candidate"]["per_sequence"]},
            "decode_checks": {"baseline": runs["baseline"]["decode_checks"],
                              "candidate": runs["candidate"]["decode_checks"]},
            "seconds": {"baseline": runs["baseline"]["seconds"],
                        "candidate": runs["candidate"]["seconds"]},
            "checkpoint_sha256": loaded["record"]["sha256"],
            "recorded_baseline": expected, "baseline_mismatches": mismatches,
        })
        persist()

    if len(report["rate_points"]) >= 2:
        report["bd_rate"] = {
            "psnr": analysis_mod._bd_rate(curves_psnr["baseline"], curves_psnr["candidate"]),
            "msssim": analysis_mod._bd_rate(curves_msssim["baseline"],
                                            curves_msssim["candidate"]),
            "points": len(report["rate_points"]),
        }
        print(f"\n  BD-rate vs the frozen production baseline: "
              f"PSNR {report['bd_rate']['psnr']:+.4f}%  "
              f"MS-SSIM {report['bd_rate']['msssim']:+.4f}%")

    best = max(p["delta"]["total_stream_gain_percent"] for p in report["rate_points"])
    report["best_total_stream_gain_percent"] = best
    report["generalization_gap_percent"] = report["locked_val_b_gain_percent"] - best
    print(f"\n  BEST DAVIS TEST GAIN: {best:+.4f}%  "
          f"(VAL-B said {report['locked_val_b_gain_percent']:+.4f}%, "
          f"gap {report['generalization_gap_percent']:+.4f} points)")
    print(f"  baseline reproduces the recorded DAVIS totals: "
          f"{report['baseline_matches_recorded']}")
    persist()
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
