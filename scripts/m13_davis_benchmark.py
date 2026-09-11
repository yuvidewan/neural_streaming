"""M13 Phase E - full DAVIS TEST benchmark, M11-G16 (deployed) vs M13
(TRAIN-recalibrated codebook frequencies). ONLY run after Phase D's coded
validation confirms a real, positive, near-fully-realized byte gain (per the
milestone brief: "ONLY run this if the actual coded validation confirms a
real gain").

Same 719-frame DAVIS TEST set, same sequence staging, same M11-G16 model,
same motion compensation, same quantization, same GOP as M11's own
benchmark (scripts/m11_evaluate.py) - the only thing that can differ between
the two arms is which frequency table the arithmetic coder reads for
residual symbols, via `m13_closed_loop.py` (see its module docstring for
why this needs its own closed loop rather than reusing `m11_evaluate.py`'s
directly).

Run:
  ./.venv/Scripts/python.exe scripts/m13_davis_benchmark.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m13_recalibration")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M11_DIR = Path("outputs/m11_autoregressive_entropy")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
RATE_POINTS = (5, 4, 3)
M11_G16_GROUP_SIZE = 16  # frozen: M13's whole premise is "the deployed M11-G16 model"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M13 Phase E: full DAVIS TEST benchmark, M11-G16 vs M13 recalibrated.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=DEFAULT_M11_DIR)
    parser.add_argument("--m10k-dir", type=Path, default=DEFAULT_M10K_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
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
    cl = _load_script("m13_closed_loop")
    ev = _load_script("m10l_evaluate")
    m10e = _load_script("m10e_evaluate")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    test_sequences = discover_sequences(args.manifest, split="test",
                                        max_sequences=args.max_sequences,
                                        max_frames_per_sequence=args.max_frames_per_sequence)

    print("=" * 130, flush=True)
    print("M13 PHASE E - M11-G16 (deployed) vs M13 (TRAIN-recalibrated codebook), DAVIS TEST")
    print("=" * 130)
    print(f"  frozen model : {args.checkpoint}")
    print(f"  sequences    : {len(test_sequences)}  frames: "
         f"{sum(s.frame_count for s in test_sequences)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = args.output_dir / "davis_streams"
    stream_dir.mkdir(parents=True, exist_ok=True)

    results: dict[tuple[str, int], dict[str, Any]] = {}
    decode_times: dict[tuple[str, int], dict[str, float]] = {}
    provenance: dict[str, Any] = {}
    all_per_sequence: list[dict[str, Any]] = []

    for bits in args.rate_points:
        print(f"\n  preparing {bits}-bit ...", flush=True)
        data = md.load_or_collect(
            model, checkpoint=args.checkpoint, manifest=args.manifest, bits=bits, device=device,
            calibration_frames=args.calibration_frames, cache_dir=cache_dir,
            log=lambda m: print(m, flush=True))
        calibration = mc.calibrate_grids(
            model, discover_sequences(args.manifest, split="train"), bits=bits,
            mode="per_channel", gop_size=args.gop, block_size=args.block_size,
            search_range=args.search_range, reference_mode="mc",
            max_frames=args.calibration_frames)
        signature = ev.calibration_signature(calibration, bits=bits,
                                             calibration_frames=args.calibration_frames,
                                             quant_mode="per_channel")
        cached_signature = ev.calibration_signature(
            data["calibration"], bits=bits, calibration_frames=args.calibration_frames,
            quant_mode="per_channel")
        if signature != cached_signature:
            print(f"[ERROR] fresh calibration {signature} != cached {cached_signature} - "
                 f"determinism regressed. STOPPING.", file=sys.stderr)
            return 1

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
        try:
            ev_m11.check_provenance(checkpoint11, signature=signature, bits=bits,
                                    group_size=M11_G16_GROUP_SIZE,
                                    context_definition_id=ma.context_definition_id(
                                        M11_G16_GROUP_SIZE), m10k_identity=m10k_identity)
        except ev_m11.ProvenanceError as error:
            print(f"[ERROR] M11-G16 checkpoint provenance mismatch: {error}. STOPPING.",
                 file=sys.stderr)
            return 1
        assign_codebook = ml.SharedCodebook.from_dict(json.loads(
            (args.m11_dir / f"m11_G16_codebook_{bits}bit_K512.json").read_text(encoding="utf-8")))

        train_k = m13.m11_g16_prototype_indices(model11, assign_codebook, train_symbols,
                                                data["train_references"], zero, device=device)
        val_k = m13.m11_g16_prototype_indices(model11, assign_codebook, val_symbols,
                                              data["val_references"], zero, device=device)
        flat = lambda a: a.reshape(-1)
        frequencies, strength, val_a_bits, val_a_baseline = m13.fit_recalibrated_frequencies(
            assign_codebook, flat(train_symbols), flat(train_k),
            flat(val_symbols[select_mask]), flat(val_k[select_mask]), alphabet=2 ** bits)
        coding_codebook = m13.build_recalibrated_codebook(
            assign_codebook, frequencies, provenance={"bits": bits, "strength": strength})
        print(f"    recalibration strength {strength:g} (VAL-A {val_a_baseline:.5f} -> "
             f"{val_a_bits:.5f} bits/symbol)", flush=True)

        arms = cl.build_arms(ma, m13, model11, assign_codebook, coding_codebook, zero,
                             m10k_identity=m10k_identity, calibration_signature=signature,
                             bits=bits)
        provenance[f"{bits}bit"] = {
            "old_identity": arms["m11_op"]["identity"].hex(),
            "new_identity": arms["m13_recal"]["identity"].hex(),
            "calibration_signature": signature, "m10k_identity": m10k_identity.hex(),
            "old_codebook_id": assign_codebook.codebook_id(
                model_identity=m10k_identity, calibration_signature=signature).hex(),
            "new_codebook_id": coding_codebook.codebook_id(
                model_identity=m10k_identity, calibration_signature=signature).hex(),
            "smoothing_strength": strength,
            "m11_g16_checkpoint_selected_epoch": checkpoint11.get("selected_epoch"),
        }

        run = cl.run_sequences(
            mc, ma, m13, model, test_sequences, arms, stream_dir,
            intra_params=calibration["intra_params"],
            intra_entropy_model=calibration["intra_entropy_model"],
            residual_params=calibration["residual_params"],
            motion_entropy_model=calibration["motion_entropy_model"], bits=bits,
            gop_size=args.gop, block_size=args.block_size, search_range=args.search_range)

        invariants = run["invariants"]
        print(f"    invariants - symbols {invariants['symbols']} | reconstruction "
             f"{invariants['reconstruction']} | motion {invariants['motion']} | "
             f"PSNR/MS-SSIM {invariants['metrics']}", flush=True)
        if not all(invariants.values()):
            print("[ERROR] the arms diverged; results are not interpretable. STOPPING.",
                 file=sys.stderr)
            return 1

        for arm in cl.ARMS:
            results[(arm, bits)] = run["results"][arm]
            decode_times[(arm, bits)] = run["decode_times"][arm]
        for row in run["per_sequence"]:
            row["m13_vs_m11_op_percent"] = (
                (row["m13_recal_residual_bytes"] - row["m11_op_residual_bytes"])
                / row["m11_op_residual_bytes"] * 100)
        all_per_sequence.extend(run["per_sequence"])

    # --- report --------------------------------------------------------------------------
    print()
    print("=" * 130)
    print("FULL BYTE ACCOUNTING")
    print("=" * 130)
    print(f"{'arm':<12} {'bits':>5} {'I bytes':>11} {'P resid':>11} {'motion':>9} "
         f"{'overhead':>9} {'TOTAL':>11} {'BPP':>8} {'PSNR':>8} {'MS-SSIM':>8} {'vs old':>8}")
    for bits in args.rate_points:
        old = results[("m11_op", bits)]
        for arm in cl.ARMS:
            a = results[(arm, bits)]
            vs_old = "" if arm == "m11_op" else \
                f"{(a['total_residual_bytes'] - old['total_residual_bytes']) / old['total_residual_bytes'] * 100:+7.3f}%"
            print(f"{arm:<12} {bits:>5} {a['total_i_frame_residual_bytes']:>11,} "
                 f"{a['total_p_frame_residual_bytes']:>11,} {a['total_motion_bytes']:>9,} "
                 f"{a['total_container_overhead_bytes']:>9,} {a['total_container_bytes']:>11,} "
                 f"{a['stream_bpp']:>8.5f} {a['mean_psnr_db']:>8.4f} {a['mean_msssim']:>8.6f} "
                 f"{vs_old:>8}")

    print()
    print("=" * 130)
    print("IDEAL BITS vs EMITTED BYTES, CODER OVERHEAD, REALIZED GAIN (P-frame residuals)")
    print("=" * 130)
    theory_rows = []
    for bits in args.rate_points:
        old, new = results[("m11_op", bits)], results[("m13_recal", bits)]
        ideal_reduction = old["p_frame_ideal_bits"] - new["p_frame_ideal_bits"]
        actual_reduction = (old["total_p_frame_residual_bytes"]
                            - new["total_p_frame_residual_bytes"]) * 8
        ideal_gain = ideal_reduction / old["p_frame_ideal_bits"] * 100
        byte_gain = (actual_reduction / 8) / old["total_p_frame_residual_bytes"] * 100
        realized = (actual_reduction / ideal_reduction * 100
                   if abs(ideal_reduction) > 1e-9 else float("nan"))
        old_overhead = (old["total_p_frame_residual_bytes"] * 8 - old["p_frame_ideal_bits"]) \
            / old["p_frame_ideal_bits"] * 100
        new_overhead = (new["total_p_frame_residual_bytes"] * 8 - new["p_frame_ideal_bits"]) \
            / new["p_frame_ideal_bits"] * 100
        row = {"bits": bits, "old_ideal_bits": old["p_frame_ideal_bits"],
              "new_ideal_bits": new["p_frame_ideal_bits"], "ideal_gain_percent": ideal_gain,
              "byte_gain_percent": byte_gain, "old_coder_overhead_percent": old_overhead,
              "new_coder_overhead_percent": new_overhead, "realized_gain_percent": realized}
        theory_rows.append(row)
        print(f"{bits:>5}  ideal {old['p_frame_ideal_bits']:>14,.0f} -> "
             f"{new['p_frame_ideal_bits']:>14,.0f}  ({ideal_gain:+.3f}%)   "
             f"bytes {old['total_p_frame_residual_bytes']:>11,} -> "
             f"{new['total_p_frame_residual_bytes']:>11,}  ({byte_gain:+.3f}%)   "
             f"overhead {old_overhead:+.4f}%->{new_overhead:+.4f}%   realized {realized:.2f}%")

    print()
    print("=" * 130)
    print("BD-RATE (piecewise linear, no extrapolation)")
    print("=" * 130)

    def curve(arm, metric="mean_psnr_db"):
        return [(results[(arm, b)]["stream_bpp"], results[(arm, b)][metric])
               for b in args.rate_points]

    psnr_bd = m10e._bd_rate_linear(curve("m11_op"), curve("m13_recal"))
    ms_bd = m10e._bd_rate_linear(curve("m11_op", "mean_msssim"), curve("m13_recal", "mean_msssim"))
    print(f"  m13_recal vs m11_op   PSNR "
         f"{(f'{psnr_bd:+.3f}%' if psnr_bd is not None else 'n/a'):>9}   MS-SSIM "
         f"{(f'{ms_bd:+.3f}%' if ms_bd is not None else 'n/a'):>9}")

    print()
    print("=" * 130)
    print("LATENCY per P-frame - encode and decode, split by stage (ms)")
    print("=" * 130)
    latency_rows = []
    for bits in args.rate_points:
        for arm in cl.ARMS:
            a = results[(arm, bits)]
            frames_p = max(a["p_frames"], 1)
            encode = {k: v / frames_p * 1000 for k, v in a["encode_seconds"].items()}
            decode = {k: v / frames_p * 1000 for k, v in decode_times[(arm, bits)].items()}
            row = {"bits": bits, "arm": arm, "encode_ms": encode, "decode_ms": decode,
                  "encode_total_ms": sum(encode.values()), "decode_total_ms": sum(decode.values())}
            latency_rows.append(row)
            print(f"{bits:>5} {arm:<12} encode {row['encode_total_ms']:>7.3f}  "
                 + " ".join(f"{k} {v:.3f}" for k, v in sorted(encode.items()))
                 + f"   | decode {row['decode_total_ms']:>7.3f}  "
                 + " ".join(f"{k} {v:.3f}" for k, v in sorted(decode.items())))

    reference_bits = args.rate_points[1] if len(args.rate_points) > 1 else args.rate_points[0]
    print()
    print("=" * 130)
    print(f"PER-SEQUENCE residual bytes at {reference_bits}-bit")
    print("=" * 130)
    for record in [r for r in all_per_sequence if r["bits"] == reference_bits]:
        print(f"{record['sequence']:<17} m11_op {record['m11_op_residual_bytes']:>9,}  "
             f"m13_recal {record['m13_recal_residual_bytes']:>9,} "
             f"({record['m13_vs_m11_op_percent']:+6.3f}%)")

    closes = all(results[k]["byte_accounting_closes"] for k in results)
    print(f"\n  byte accounting closes everywhere: {closes}")
    report = {
        "phase": "M13 Phase E DAVIS benchmark", "checkpoint": str(args.checkpoint),
        "rate_points_bits": list(args.rate_points),
        "arms": {f"{k[0]}@{k[1]}bit": v for k, v in results.items()},
        "per_sequence": all_per_sequence, "ideal_vs_deployed": theory_rows,
        "bd_rate": {"m13_recal_vs_m11_op_psnr": psnr_bd, "m13_recal_vs_m11_op_msssim": ms_bd},
        "latency": latency_rows, "provenance": provenance,
        "invariants": {"symbols_identical": True, "reconstruction_identical": True,
                      "motion_identical": True, "metrics_identical": True,
                      "byte_accounting_closes": closes},
    }
    path = args.output_dir / "m13_davis_benchmark.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
