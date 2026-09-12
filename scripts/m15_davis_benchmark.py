"""M15 Phase H/I - full 719-frame DAVIS TEST benchmark, current (M14
deployed) vs whichever calibration-policy candidate Phase F/G's coded
validation confirmed a real gain for. Only run after Phase F/G, not before
(this milestone's own "only promote a policy if offline AND coded
validation show a meaningful improvement" rule).

Same sequence staging, same M13 temporal model/recalibrated residual
codebook, same motion compensation/quantization/GOP as M13/M14's own
benchmarks - the ONLY thing that can differ between combos is which
frequency table the arithmetic coder reads for intra and/or motion
symbols. Combo spec is identical to m15_coded_validation.py's
(LABEL:INTRA_POLICY:MOTION_POLICY, with "current"/"broad_motion"
shortcuts) so the exact same candidate can be pointed at TEST once it has
earned it.

Run:
  ./.venv/Scripts/python.exe scripts/m15_davis_benchmark.py --combos current broad_motion
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

DEFAULT_OUTPUT_DIR = Path("outputs/m15_calibration_policy")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M11_DIR = Path("outputs/m11_autoregressive_entropy")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
RATE_POINTS = (5, 4, 3)
M11_G16_GROUP_SIZE = 16
SHORTCUTS = {
    "current": ("A_sequential_400", "C_broad_576"),
    "broad_motion": ("A_sequential_400", "B_uniform_400"),
}


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_combo(spec: str) -> tuple[str, str, str]:
    if spec in SHORTCUTS:
        intra_policy, motion_policy = SHORTCUTS[spec]
        return spec, intra_policy, motion_policy
    label, intra_policy, motion_policy = spec.split(":")
    return label, intra_policy, motion_policy


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M15 Phase H/I: full DAVIS TEST, current vs a calibration-policy candidate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=DEFAULT_M11_DIR)
    parser.add_argument("--m10k-dir", type=Path, default=DEFAULT_M10K_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--combos", nargs="+", default=["current", "broad_motion"])
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

    combos = [_parse_combo(spec) for spec in args.combos]

    mc = _load_script("m10h_motion_compensation")
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    md = _load_script("m11_data")
    cx = _load_script("m11_causal_context")
    mk = _load_script("m10k_learned_entropy")
    m13 = _load_script("m13_recalibration")
    m14 = _load_script("m14_recalibration")
    cl14 = _load_script("m14_closed_loop")
    ev = _load_script("m10l_evaluate")
    ev_m11 = _load_script("m11_evaluate")
    m10e = _load_script("m10e_evaluate")
    m15cal = _load_script("m15_calibration_policy")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    test_sequences = discover_sequences(args.manifest, split="test",
                                        max_sequences=args.max_sequences,
                                        max_frames_per_sequence=args.max_frames_per_sequence)
    train_sequences = discover_sequences(args.manifest, split="train")

    print("=" * 130, flush=True)
    print("M15 PHASE H/I - CALIBRATION-POLICY CANDIDATES, FULL DAVIS TEST")
    print("=" * 130)
    print(f"  sequences: {len(test_sequences)}  frames: "
         f"{sum(s.frame_count for s in test_sequences)}  combos: {[c[0] for c in combos]}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = args.output_dir / "davis_streams"
    stream_dir.mkdir(parents=True, exist_ok=True)

    # Motion tables are bit-depth independent (Phase A q7) - fit once per policy actually used.
    motion_policy_names = sorted({m for _, _, m in combos})
    motion_bits = mc.motion_alphabet_bits(args.search_range)
    policy_motion_cache = {}
    for name in motion_policy_names:
        seqs = m15cal.build_policy(name, train_sequences, seed=args.seed)
        symbols = m14.collect_motion_symbols(
            mc, model, seqs, block_size=args.block_size, search_range=args.search_range,
            gop_size=args.gop, max_frames=10 ** 9, reference_mode="mc", device=device)
        policy_motion_cache[name] = m14.fit_empirical(symbols, bits=motion_bits, num_tables=2)
        print(f"  motion policy {name}: {symbols.shape[0]} TRAIN P-frames, identity "
             f"{policy_motion_cache[name].model_id().hex()}", flush=True)

    results: dict[tuple[str, int], dict[str, Any]] = {}
    decode_times: dict[tuple[str, int], dict[str, float]] = {}
    provenance: dict[str, Any] = {}
    all_per_sequence: list[dict[str, Any]] = []
    combo_labels = [c[0] for c in combos]

    for bits in args.rate_points:
        print(f"\n  preparing {bits}-bit ...", flush=True)
        calibration = mc.calibrate_grids(
            model, train_sequences, bits=bits, mode="per_channel", gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, reference_mode="mc",
            max_frames=args.calibration_frames)
        signature = ev.calibration_signature(calibration, bits=bits,
                                             calibration_frames=args.calibration_frames,
                                             quant_mode="per_channel")

        data = md.load_or_collect(model, checkpoint=args.checkpoint, manifest=args.manifest,
                                  bits=bits, device=device, calibration_frames=args.calibration_frames,
                                  cache_dir=cache_dir, log=lambda m: print(m, flush=True))
        cached_signature = ev.calibration_signature(data["calibration"], bits=bits,
                                                     calibration_frames=args.calibration_frames,
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
        ev_m11.check_provenance(checkpoint11, signature=signature, bits=bits,
                                group_size=M11_G16_GROUP_SIZE,
                                context_definition_id=ma.context_definition_id(M11_G16_GROUP_SIZE),
                                m10k_identity=m10k_identity)
        assign_codebook = ml.SharedCodebook.from_dict(json.loads(
            (args.m11_dir / f"m11_G16_codebook_{bits}bit_K512.json").read_text(encoding="utf-8")))
        train_k = m13.m11_g16_prototype_indices(model11, assign_codebook, train_symbols,
                                                data["train_references"], zero, device=device)
        val_k = m13.m11_g16_prototype_indices(model11, assign_codebook, val_symbols,
                                              data["val_references"], zero, device=device)
        flat = lambda a: a.reshape(-1)
        frequencies, strength, _, _ = m13.fit_recalibrated_frequencies(
            assign_codebook, flat(train_symbols), flat(train_k),
            flat(val_symbols[select_mask]), flat(val_k[select_mask]), alphabet=2 ** bits)
        coding_codebook = m13.build_recalibrated_codebook(
            assign_codebook, frequencies, provenance={"bits": bits, "strength": strength})
        m13_identity = ma.model_identity(model11, m10k_identity=m10k_identity,
                                         calibration_signature=signature, bits=bits,
                                         codebook=coding_codebook)
        residual_arm = {"m13_recal": {"model": model11, "zero": zero,
                                      "assign_codebook": assign_codebook,
                                      "coding_codebook": coding_codebook,
                                      "identity": m13_identity}}

        intra_policy_names = sorted({i for _, i, _ in combos})
        policy_intra: dict[str, Any] = {}
        for name in intra_policy_names:
            seqs = m15cal.build_policy(name, train_sequences, seed=args.seed)
            symbols = m14.collect_intra_symbols(model, seqs, intra_params=calibration["intra_params"],
                                                max_frames=10 ** 9, device=device)
            policy_intra[name] = m14.fit_empirical(symbols, bits=bits, num_tables=symbols.shape[1])

        provenance[f"{bits}bit"] = {
            "residual_identity": m13_identity.hex(), "calibration_signature": signature,
            "m10k_identity": m10k_identity.hex(),
            "intra_identities": {n: t.model_id().hex() for n, t in policy_intra.items()},
        }

        for label, intra_policy, motion_policy in combos:
            run = cl14.run_sequences_for_arms(
                mc, ma, m13, model, test_sequences, residual_arm, stream_dir,
                intra_params=calibration["intra_params"],
                intra_entropy_model=policy_intra[intra_policy],
                residual_params=calibration["residual_params"],
                motion_entropy_model=policy_motion_cache[motion_policy], bits=bits, gop_size=args.gop,
                block_size=args.block_size, search_range=args.search_range)
            invariants = run["invariants"]
            print(f"    [{label}] intra={intra_policy} motion={motion_policy}  "
                 f"invariants - symbols {invariants['symbols']} | reconstruction "
                 f"{invariants['reconstruction']} | motion {invariants['motion']} | "
                 f"PSNR/MS-SSIM {invariants['metrics']}", flush=True)
            if not all(invariants.values()):
                print(f"[ERROR] combo '{label}' diverged; results are not interpretable. "
                     f"STOPPING.", file=sys.stderr)
                return 1
            results[(label, bits)] = run["results"]["m13_recal"]
            decode_times[(label, bits)] = run["decode_times"]["m13_recal"]
            for row in run["per_sequence"]:
                row["combo"] = label
                all_per_sequence.append(row)

    # --- report --------------------------------------------------------------------------
    print()
    print("=" * 130)
    print("FULL BYTE ACCOUNTING")
    print("=" * 130)
    print(f"{'combo':<14} {'bits':>5} {'I bytes':>11} {'P resid':>11} {'motion':>9} "
         f"{'overhead':>9} {'TOTAL':>11} {'BPP':>8} {'PSNR':>8} {'MS-SSIM':>8} {'vs current':>12}")
    for bits in args.rate_points:
        base = results[("current", bits)]
        for combo in combo_labels:
            a = results[(combo, bits)]
            vs_base = "" if combo == "current" else \
                f"{(a['total_container_bytes'] - base['total_container_bytes']) / base['total_container_bytes'] * 100:+11.4f}%"
            print(f"{combo:<14} {bits:>5} {a['total_i_frame_residual_bytes']:>11,} "
                 f"{a['total_p_frame_residual_bytes']:>11,} {a['total_motion_bytes']:>9,} "
                 f"{a['total_container_overhead_bytes']:>9,} {a['total_container_bytes']:>11,} "
                 f"{a['stream_bpp']:>8.5f} {a['mean_psnr_db']:>8.4f} {a['mean_msssim']:>8.6f} "
                 f"{vs_base:>12}")

    print()
    print("=" * 130)
    print("BD-RATE vs current (piecewise linear, no extrapolation)")
    print("=" * 130)

    def curve(combo, metric="mean_psnr_db"):
        return [(results[(combo, b)]["stream_bpp"], results[(combo, b)][metric])
               for b in args.rate_points]

    bd_rates = {}
    for combo in combo_labels:
        if combo == "current":
            continue
        psnr_bd = m10e._bd_rate_linear(curve("current"), curve(combo))
        ms_bd = m10e._bd_rate_linear(curve("current", "mean_msssim"), curve(combo, "mean_msssim"))
        bd_rates[combo] = {"vs_current_psnr": psnr_bd, "vs_current_msssim": ms_bd}
        print(f"  {combo} vs current   PSNR "
             f"{(f'{psnr_bd:+.3f}%' if psnr_bd is not None else 'n/a'):>9}   MS-SSIM "
             f"{(f'{ms_bd:+.3f}%' if ms_bd is not None else 'n/a'):>9}")

    print()
    print("=" * 130)
    print("LATENCY per P-frame - encode and decode, split by stage (ms)")
    print("=" * 130)
    latency_rows = []
    for bits in args.rate_points:
        for combo in combo_labels:
            a = results[(combo, bits)]
            frames_p = max(a["p_frames"], 1)
            encode = {k: v / frames_p * 1000 for k, v in a["encode_seconds"].items()}
            decode = {k: v / frames_p * 1000 for k, v in decode_times[(combo, bits)].items()}
            row = {"bits": bits, "combo": combo, "encode_ms": encode, "decode_ms": decode,
                  "encode_total_ms": sum(encode.values()), "decode_total_ms": sum(decode.values())}
            latency_rows.append(row)
            print(f"{bits:>5} {combo:<14} encode {row['encode_total_ms']:>7.3f}  | decode "
                 f"{row['decode_total_ms']:>7.3f}", flush=True)

    closes = all(results[k]["byte_accounting_closes"] for k in results)
    print(f"\n  byte accounting closes everywhere: {closes}")
    report = {
        "phase": "M15 Phase H/I DAVIS benchmark", "checkpoint": str(args.checkpoint),
        "rate_points_bits": list(args.rate_points), "combos": combo_labels,
        "arms": {f"{k[0]}@{k[1]}bit": v for k, v in results.items()},
        "per_sequence": all_per_sequence,
        "bd_rate": bd_rates, "latency": latency_rows, "provenance": provenance,
        "invariants": {"symbols_identical": True, "reconstruction_identical": True,
                      "motion_identical": True, "metrics_identical": True,
                      "byte_accounting_closes": closes},
    }
    path = args.output_dir / "m15_davis_benchmark.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
