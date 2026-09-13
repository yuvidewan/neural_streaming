"""M21 Phase 9 - full DAVIS TEST benchmark for the locked candidate.

Runs ONLY after `m21_analysis.py` has locked a candidate by the pre-declared
Phase 7 rule and the VAL-B gate has cleared 0.5% (Phase 8). DAVIS TEST is opened
here and nowhere else in M21: no calibration, no fitting, no threshold and no
candidate selection has ever touched it.

Both arms - the deployed identity control and the locked candidate - are run at
all three rate points through the same real closed loop, so:

  * the identity arm can be checked against M14's own recorded DAVIS totals
    (`m14_davis_benchmark.json`'s `motion@Nbit` arms), which is the direct
    comparison against the frozen M13/M14/M20 production baseline the milestone
    asks for;
  * a BD-rate can be computed over three real rate points, which is the only
    honest way to price a candidate that helps at one bit depth and hurts at the
    others.

Run:
  ./.venv/Scripts/python.exe scripts/m21_davis.py --candidate px_median3_a50
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

DEFAULT_M14_DAVIS = Path("outputs/m14_entropy_audit/m14_davis_benchmark.json")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M21 Phase 9: full DAVIS TEST for the locked candidate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate", type=str, required=True,
                        help="The candidate locked by m21_analysis.py's declared rule.")
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt"))
    parser.add_argument("--m11-dir", type=Path, default=Path("outputs/m11_autoregressive_entropy"))
    parser.add_argument("--m10k-dir", type=Path, default=Path("outputs/m10k_learned_entropy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m21_reference_refinement"))
    parser.add_argument("--m14-davis", type=Path, default=DEFAULT_M14_DAVIS)
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
    m13 = _load_script("m13_recalibration")
    md = _load_script("m11_data")
    m21 = _load_script("m21_refinement")
    m10e = _load_script("m10e_evaluate")

    analysis_path = args.output_dir / "m21_analysis.json"
    if not analysis_path.is_file():
        print("[ERROR] m21_analysis.json not found - the candidate must be locked by the "
              "declared Phase 7 rule BEFORE DAVIS TEST is opened.", file=sys.stderr)
        return 1
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    locked = sorted(
        (entry["selected"] for entry in analysis["rate_points"] if "selected" in entry),
        key=lambda s: -s["total_stream_gain_percent"])
    if not locked or locked[0]["candidate"] != args.candidate:
        print(f"[ERROR] --candidate {args.candidate!r} is not the locked candidate "
              f"({locked[0]['candidate'] if locked else 'none'}); refusing to open TEST.",
              file=sys.stderr)
        return 1
    if not analysis["full_davis_gated"]:
        print("[ERROR] the VAL-B gate did not clear 0.5%; refusing to open TEST.",
              file=sys.stderr)
        return 1

    refinement = m21.candidate(args.candidate)
    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    # The ONLY TEST access in M21, and it happens after the candidate is locked.
    davis = discover_sequences(args.manifest, split="test")
    train_full = discover_sequences(args.manifest, split="train")
    motion_train = m21.broad_train_sequences(
        args.manifest, frames_per_sequence=args.train_frames_per_sequence)
    recorded = json.loads(args.m14_davis.read_text(encoding="utf-8"))["arms"]

    print("=" * 134)
    print(f"M21 PHASE 9 - FULL DAVIS TEST, locked candidate '{args.candidate}'")
    print("=" * 134)
    print(f"  definition: {refinement.definition}")
    print(f"  sequences: {len(davis)}  frames: {sum(s.frame_count for s in davis)}")
    print(f"  locked by the declared rule at {locked[0]['total_stream_gain_percent']:+.4f}% "
          f"total stream on VAL-B", flush=True)

    stream_dir = args.output_dir / "davis_streams"
    report: dict[str, Any] = {
        "phase": "M21 Phase 9", "candidate": args.candidate,
        "definition": refinement.definition,
        "sequences": len(davis), "frames": sum(s.frame_count for s in davis),
        "rate_points": [], "baseline_matches_m14": True,
    }

    curves: dict[str, list[tuple[float, float]]] = {"identity": [], args.candidate: []}
    curves_msssim: dict[str, list[tuple[float, float]]] = {"identity": [], args.candidate: []}

    for bits in args.rate_points:
        rig = m21.prepare_rate_point(
            model, bits=bits, manifest=args.manifest, checkpoint=args.checkpoint,
            m11_dir=args.m11_dir, m10k_dir=args.m10k_dir, device=device, cache_dir=cache_dir,
            train_full=train_full, motion_train=motion_train,
            calibration_frames=args.calibration_frames, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            motion_cache=args.output_dir / "m21_deployed_motion_table.json")

        runs = {}
        for arm in (m21.IDENTITY, refinement):
            started = time.perf_counter()
            runs[arm.name] = m21.run_candidate(
                mc, m13, model, davis, rig["spec"], arm, stream_dir,
                intra_params=rig["intra_params"],
                intra_entropy_model=rig["intra_entropy_model"],
                residual_params=rig["residual_params"],
                motion_entropy_model=rig["motion_entropy_model"], bits=bits,
                gop_size=args.gop, block_size=args.block_size, search_range=args.search_range)
            runs[arm.name]["seconds"] = time.perf_counter() - started
            for path in stream_dir.glob(f"{arm.name}_{bits}bit_*.nvct"):
                path.unlink()

        base = runs["identity"]["aggregate"]
        cand = runs[args.candidate]["aggregate"]
        sweep = _load_script("m21_sweep")
        delta = sweep.compare(cand, base)
        gop = sweep.gop_comparison(runs[args.candidate]["gop_position"],
                                   runs["identity"]["gop_position"])

        expected = recorded.get(f"motion@{bits}bit", {})
        matches = all(base[field] == expected.get(field) for field in (
            "total_motion_bytes", "total_i_frame_residual_bytes",
            "total_p_frame_residual_bytes", "total_container_bytes")) if expected else None
        report["baseline_matches_m14"] &= bool(matches)

        for name, summary in (("identity", base), (args.candidate, cand)):
            curves[name].append((summary["stream_bpp"], summary["mean_psnr_db"]))
            curves_msssim[name].append((summary["stream_bpp"], summary["mean_msssim"]))

        print(f"\n  ---- {bits}-bit ----  residual={rig['residual_identity']}  "
              f"motion={rig['motion_identity']}")
        print(f"    identity  total={base['total_container_bytes']:,}  "
              f"P-resid={base['total_p_frame_residual_bytes']:,}  "
              f"I-resid={base['total_i_frame_residual_bytes']:,}  "
              f"motion={base['total_motion_bytes']:,}  BPP={base['stream_bpp']:.6f}  "
              f"PSNR={base['mean_psnr_db']:.4f}  MS-SSIM={base['mean_msssim']:.6f}")
        print(f"              matches M14's recorded motion@{bits}bit arm: {matches}")
        print(f"    {args.candidate:<9} total={cand['total_container_bytes']:,}  "
              f"P-resid={cand['total_p_frame_residual_bytes']:,}  "
              f"I-resid={cand['total_i_frame_residual_bytes']:,}  "
              f"motion={cand['total_motion_bytes']:,}  BPP={cand['stream_bpp']:.6f}  "
              f"PSNR={cand['mean_psnr_db']:.4f}  MS-SSIM={cand['mean_msssim']:.6f}")
        print(f"    DELTA     total={delta['delta_total_bytes']:+,} "
              f"({delta['total_stream_gain_percent']:+.4f}%)  "
              f"residual={delta['delta_residual_bytes']:+,}  "
              f"motion={delta['delta_motion_bytes']:+,}  "
              f"PSNR={delta['delta_psnr_db']:+.4f} dB  "
              f"MS-SSIM={delta['delta_msssim']:+.6f}  verdict="
              f"{m21.verdict(delta['total_stream_gain_percent'])}")
        print(f"    GOP       boundary {gop['boundary']['delta_residual_bytes']:+,} bytes "
              f"({gop['boundary']['residual_gain_percent']:+.4f}%)   ordinary "
              f"{gop['ordinary']['delta_residual_bytes']:+,} bytes "
              f"({gop['ordinary']['residual_gain_percent']:+.4f}%)")
        print(f"    runtime   identity {runs['identity']['seconds']:.1f}s   "
              f"{args.candidate} {runs[args.candidate]['seconds']:.1f}s   "
              f"(+{(runs[args.candidate]['seconds'] / runs['identity']['seconds'] - 1) * 100:.1f}%)")
        print(f"    decode    {runs[args.candidate]['decode_checks']}", flush=True)

        report["rate_points"].append({
            "bits": bits, "residual_identity": rig["residual_identity"],
            "motion_identity": rig["motion_identity"],
            "identity": base, "candidate": cand, "delta": delta,
            "gop_comparison": gop,
            "verdict": m21.verdict(delta["total_stream_gain_percent"]),
            "identity_matches_m14_recorded": matches,
            "m14_recorded": {k: expected.get(k) for k in (
                "total_motion_bytes", "total_i_frame_residual_bytes",
                "total_p_frame_residual_bytes", "total_container_bytes")},
            "per_sequence": {"identity": runs["identity"]["per_sequence"],
                             args.candidate: runs[args.candidate]["per_sequence"]},
            "decode_checks": runs[args.candidate]["decode_checks"],
            "seconds": {"identity": runs["identity"]["seconds"],
                        args.candidate: runs[args.candidate]["seconds"]},
        })

    bd_psnr = m10e._bd_rate_linear(curves["identity"], curves[args.candidate])
    bd_msssim = m10e._bd_rate_linear(curves_msssim["identity"], curves_msssim[args.candidate])
    report["bd_rate"] = {"psnr": bd_psnr, "msssim": bd_msssim,
                         "note": "computed with the candidate applied at ALL THREE rate points - "
                                 "the honest price of deploying it everywhere, which is not the "
                                 "same as its effect at the single rate point where it helps"}
    print(f"\n  BD-rate (candidate applied at every rate point): PSNR "
          f"{bd_psnr:+.3f}%  MS-SSIM {bd_msssim:+.3f}%  (negative = better)")

    path = args.output_dir / "m21_davis.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
