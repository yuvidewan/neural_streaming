"""M18 Phase C/D/H - does a candidate intra quantizer (Phase B) actually
reduce REAL, deployed P-frame residual bytes, once fed through the exact
same real closed loop M13/M14/M15/M16/M17 all used? Only BASE (the deployed
intra quantizer+entropy model) and P-frame coding (motion, residual
quantizer, G16, codebook, M13 frequencies) ever change across a run - the
candidate substitutes ONLY the I-frame quantizer+entropy model, nothing else.

Both `intra_entropy_model`s (base and candidate) are freshly TRAIN-fit under
their OWN quantizer's symbols - a candidate is judged on what its OWN
properly-fit I-frame table would cost, never on a stale table mismatched to
a different quantizer (that would unfairly penalize or flatter it).

Reuses `m13_recalibration.encode_frame_recalibrated` UNMODIFIED for P-frame
residual coding (exactly M17's own discipline) - only I-frame coding
(`encode_latent_to_payload`/`decode_payload_to_latent`) varies by candidate.

Run:
  ./.venv/Scripts/python.exe scripts/m18_reference_bridge.py --candidate BROAD_same_percentile --rate-points 5 4
  ./.venv/Scripts/python.exe scripts/m18_reference_bridge.py --candidate TIGHT_1_99 --rate-points 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.codec import (
    decode_payload_to_latent, encode_latent_to_payload, latent_to_symbols, symbols_to_latent,
)
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m18_intra_quantizer_audit")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M11_DIR = Path("outputs/m11_autoregressive_entropy")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
M11_G16_GROUP_SIZE = 16

# M17's own recorded real-to-oracle P-residual channel gain, VAL-B, per bit
# depth (outputs/m17_residual_oracle_audit/m17_residual_diagnostic.json) -
# the denominator for Phase D's "fraction of oracle gap recovered".
M17_REAL_TO_ORACLE_GAIN_PERCENT = {5: 2.4718831448875886, 4: 8.46616703581226, 3: 19.61324608483599}

CANDIDATE_PERCENTILES = {
    "BROAD_same_percentile": (0.1, 99.9, True),
    "TIGHT_1_99": (1.0, 99.0, True),
    "TIGHTER_2_98": (2.0, 98.0, True),
    "LOOSE_0.01_99.99": (0.01, 99.99, True),
}


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def run_closed_loop_with_intra_variant(mc, m13, model, model11, assign_codebook, coding_codebook,
                                       sequence, *, bits, intra_params, intra_entropy_model,
                                       residual_params, zero, gop_size, block_size, search_range,
                                       device) -> list[dict[str, Any]]:
    """The real closed loop (matches m13_closed_loop.encode_multi exactly),
    with the SUPPLIED intra_params/intra_entropy_model - lets a candidate
    intra quantizer stand in for I-frame coding while every other stage
    stays the frozen, deployed one. Returns one row per frame (I and P)."""
    frames = sequence.load_frames()
    frame_count = frames.shape[0]
    types = mc.gop_frame_types(frame_count, gop_size)
    previous = None
    latent_shape: tuple[int, ...] | None = None
    rows: list[dict[str, Any]] = []

    with mc.deterministic_kernels():
        for index in range(frame_count):
            frame = frames[index:index + 1].to(device)
            latent = model.encode(frame)
            latent_shape = tuple(latent.shape[1:])
            if types[index] == mc.FRAME_TYPE_I:
                payload, ideal = encode_latent_to_payload(latent, params=intra_params,
                                                          entropy_model=intra_entropy_model)
                decoded, _ = decode_payload_to_latent(
                    payload, entropy_model=intra_entropy_model, params=intra_params,
                    shape=latent_shape)
                previous = model.decode(decoded.to(device))
                rows.append({"index": index, "frame_type": "I", "i_bytes": len(payload),
                            "i_ideal_bits": ideal,
                            "reconstruction_mse": float(((previous - frame) ** 2).mean())})
                continue

            is_boundary = (index - 1) % gop_size == 0
            gop_position = index % gop_size
            motion = mc.estimate_block_motion(previous, frame, block_size=block_size,
                                              search_range=search_range)
            warped = mc.warp_blocks(previous, motion, block_size=block_size)
            reference_latent = model.encode(warped)
            delta = latent - reference_latent
            symbols = latent_to_symbols(delta, residual_params).reshape(latent_shape)
            payload, ideal = m13.encode_frame_recalibrated(
                model11, assign_codebook, coding_codebook, reference_latent, symbols, zero, bits=bits)

            reconstructed_latent = reference_latent + symbols_to_latent(
                symbols.reshape(-1), latent_shape, residual_params).to(device)
            reconstruction = model.decode(reconstructed_latent)
            rows.append({"index": index, "frame_type": "P", "is_boundary": is_boundary,
                        "gop_position": gop_position, "p_bytes": len(payload), "p_ideal_bits": ideal,
                        "residual_abs_mean": float(delta.abs().mean()),
                        "sad_reference": float((frame - warped).abs().mean()),
                        "reconstruction_mse": float(((reconstruction - frame) ** 2).mean())})
            previous = reconstruction

    return rows


def _fit_intra_entropy_model(model, sequences, *, params, bits, max_frames, device
                             ) -> EmpiricalEntropyModel:
    """TRAIN-only, matches calibrate_grids's own intra fitting convention
    exactly (EmpiricalEntropyModel.from_symbols over every consumed frame's
    quantized latent, under the SUPPLIED - candidate or baseline - params)."""
    symbols, seen = [], 0
    with torch.no_grad():
        for sequence in sequences:
            frames = sequence.load_frames()
            for index in range(frames.shape[0]):
                if seen >= max_frames:
                    break
                latent = model.encode(frames[index:index + 1].to(device))
                symbols.append(latent_to_symbols(latent, params).reshape(latent.shape[1], -1))
                seen += 1
            if seen >= max_frames:
                break
    stack = np.stack(symbols)
    return EmpiricalEntropyModel.from_symbols(stack, bits=bits, num_tables=stack.shape[1])


def _summarize_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    p_rows = [r for r in rows if r["frame_type"] == "P"]
    i_rows = [r for r in rows if r["frame_type"] == "I"]
    boundary = [r for r in p_rows if r["is_boundary"]]
    ordinary = [r for r in p_rows if not r["is_boundary"]]
    return {
        "i_frames": len(i_rows), "total_i_bytes": sum(r["i_bytes"] for r in i_rows),
        "mean_i_reconstruction_mse": statistics.fmean(r["reconstruction_mse"] for r in i_rows),
        "p_frames": len(p_rows), "total_p_bytes": sum(r["p_bytes"] for r in p_rows),
        "boundary_p_frames": len(boundary), "boundary_p_bytes": sum(r["p_bytes"] for r in boundary),
        "ordinary_p_frames": len(ordinary), "ordinary_p_bytes": sum(r["p_bytes"] for r in ordinary),
        "mean_p_reconstruction_mse": statistics.fmean(r["reconstruction_mse"] for r in p_rows),
    }


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M18 Phase C/D/H: does a candidate intra quantizer reduce real deployed bytes?",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=DEFAULT_M11_DIR)
    parser.add_argument("--m10k-dir", type=Path, default=DEFAULT_M10K_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate", required=True, choices=list(CANDIDATE_PERCENTILES))
    parser.add_argument("--rate-points", type=int, nargs="+", required=True)
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--broad-frames-per-sequence", type=int, default=8)
    parser.add_argument("--val-sequences", type=int, default=4)
    parser.add_argument("--val-frames-per-sequence", type=int, default=None)
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
    ev = _load_script("m10l_evaluate")
    ev_m11 = _load_script("m11_evaluate")
    gate_script = _load_script("m12_spatial_offline_gate")
    m15cal = _load_script("m15_calibration_policy")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    val_sequences = discover_sequences(args.manifest, split="val",
                                       max_frames_per_sequence=args.val_frames_per_sequence)
    val_b = val_sequences[1::2][:args.val_sequences]
    train_full = discover_sequences(args.manifest, split="train")
    lower, upper, use_broad = CANDIDATE_PERCENTILES[args.candidate]
    candidate_train_sequences = (
        m15cal.build_policy("C_broad_576", train_full, seed=args.seed,
                            frames_per_sequence=args.broad_frames_per_sequence)
        if use_broad else train_full)

    print("=" * 122, flush=True)
    print(f"M18 PHASE C/D/H - REFERENCE BRIDGE: candidate '{args.candidate}' "
         f"(percentiles {lower}/{upper}, broad_train={use_broad})")
    print("=" * 122)
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"phase": "M18 Phase C/D/H reference bridge",
                              "candidate": args.candidate, "percentiles": [lower, upper],
                              "broad_train": use_broad, "rate_points": []}

    for bits in args.rate_points:
        print(f"\n  ---- {bits}-bit ----", flush=True)
        calibration = mc.calibrate_grids(
            model, train_full, bits=bits, mode="per_channel", gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, reference_mode="mc",
            max_frames=args.calibration_frames)
        base_intra_params = calibration["intra_params"]
        base_intra_entropy_model = calibration["intra_entropy_model"]
        residual_params = calibration["residual_params"]

        candidate_intra_params = calibrate_quantization_params(
            torch.cat([model.encode(f[i:i + 1].to(device)).cpu()
                      for s in candidate_train_sequences for f in [s.load_frames()]
                      for i in range(f.shape[0])], dim=0).to(device),
            bits=bits, mode="per_channel", lower_percentile=lower, upper_percentile=upper)
        started = time.perf_counter()
        candidate_intra_entropy_model = _fit_intra_entropy_model(
            model, candidate_train_sequences, params=candidate_intra_params, bits=bits,
            max_frames=10 ** 9, device=device)
        print(f"    candidate intra entropy model fit ({time.perf_counter() - started:.1f}s), "
             f"identity {candidate_intra_entropy_model.model_id().hex()} vs base "
             f"{base_intra_entropy_model.model_id().hex()}", flush=True)

        # --- re-derive the FROZEN, DEPLOYED M13 residual arm, unchanged ---------------------
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
        signature = ev.calibration_signature(calibration, bits=bits,
                                             calibration_frames=args.calibration_frames,
                                             quant_mode="per_channel")
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

        # --- BASE vs CANDIDATE, identical VAL-B sequences, only intra varies ----------------
        base_rows, candidate_rows = [], []
        for sequence in val_b:
            base_rows.extend(run_closed_loop_with_intra_variant(
                mc, m13, model, model11, assign_codebook, coding_codebook, sequence, bits=bits,
                intra_params=base_intra_params, intra_entropy_model=base_intra_entropy_model,
                residual_params=residual_params, zero=zero, gop_size=args.gop,
                block_size=args.block_size, search_range=args.search_range, device=device))
            candidate_rows.extend(run_closed_loop_with_intra_variant(
                mc, m13, model, model11, assign_codebook, coding_codebook, sequence, bits=bits,
                intra_params=candidate_intra_params, intra_entropy_model=candidate_intra_entropy_model,
                residual_params=residual_params, zero=zero, gop_size=args.gop,
                block_size=args.block_size, search_range=args.search_range, device=device))

        base_summary = _summarize_run(base_rows)
        candidate_summary = _summarize_run(candidate_rows)

        delta_i_bytes = candidate_summary["total_i_bytes"] - base_summary["total_i_bytes"]
        delta_p_bytes = candidate_summary["total_p_bytes"] - base_summary["total_p_bytes"]
        delta_boundary_p = candidate_summary["boundary_p_bytes"] - base_summary["boundary_p_bytes"]
        delta_ordinary_p = candidate_summary["ordinary_p_bytes"] - base_summary["ordinary_p_bytes"]
        net_delta_bytes = delta_i_bytes + delta_p_bytes
        base_total = base_summary["total_i_bytes"] + base_summary["total_p_bytes"]
        net_gain_percent = -net_delta_bytes / base_total * 100

        p_channel_gain_percent = -delta_p_bytes / base_summary["total_p_bytes"] * 100
        fraction_oracle_gap_recovered = (
            p_channel_gain_percent / M17_REAL_TO_ORACLE_GAIN_PERCENT[bits] * 100)

        print(f"    I-bytes:  base={base_summary['total_i_bytes']:,}  "
             f"candidate={candidate_summary['total_i_bytes']:,}  Delta={delta_i_bytes:+,}")
        print(f"    P-bytes:  base={base_summary['total_p_bytes']:,}  "
             f"candidate={candidate_summary['total_p_bytes']:,}  Delta={delta_p_bytes:+,}  "
             f"(boundary Delta={delta_boundary_p:+,}, ordinary Delta={delta_ordinary_p:+,})")
        print(f"    P-channel gain: {p_channel_gain_percent:+.4f}%   "
             f"fraction of M17 oracle gap recovered: {fraction_oracle_gap_recovered:+.4f}%")
        print(f"    NET total-stream effect: {net_delta_bytes:+,} bytes "
             f"({net_gain_percent:+.4f}% of base total) "
             f"I-frame reconstruction MSE base={base_summary['mean_i_reconstruction_mse']:.6f} "
             f"candidate={candidate_summary['mean_i_reconstruction_mse']:.6f}", flush=True)

        report["rate_points"].append({
            "bits": bits, "base_intra_identity": base_intra_entropy_model.model_id().hex(),
            "candidate_intra_identity": candidate_intra_entropy_model.model_id().hex(),
            "base_summary": base_summary, "candidate_summary": candidate_summary,
            "delta_i_bytes": delta_i_bytes, "delta_p_bytes": delta_p_bytes,
            "delta_boundary_p_bytes": delta_boundary_p, "delta_ordinary_p_bytes": delta_ordinary_p,
            "p_channel_gain_percent": p_channel_gain_percent,
            "m17_real_to_oracle_gain_percent": M17_REAL_TO_ORACLE_GAIN_PERCENT[bits],
            "fraction_oracle_gap_recovered_percent": fraction_oracle_gap_recovered,
            "net_delta_bytes": net_delta_bytes, "net_total_stream_gain_percent": net_gain_percent,
        })

    path = args.output_dir / f"m18_reference_bridge_{args.candidate}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
