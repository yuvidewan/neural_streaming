"""M17 Phase B/C/D/E - real vs. oracle residual coding through the ACTUAL
deployed M11-G16 + M13 pipeline (`m13_recalibration.encode_frame_recalibrated`,
unmodified, called directly - never reimplemented or approximated).

Three variants per P-frame, matching Phase C's minimum requirement exactly:

  A_real                    real reference (`model.decode(reconstructed_latent)`,
                             the actual deployed chain) + real motion vectors.
                             This IS what the deployed coder produces.
  B_oracle_ref_real_motion  the reference is replaced by the oracle (true-latent
                             round trip of the raw previous frame, M16's own
                             idealization) but the MOTION VECTORS are still A's
                             real ones - isolates "does reference PIXEL quality
                             alone change residual/context/assignment" from any
                             motion-vector effect.
  C_full_oracle             oracle reference AND oracle-reference motion vectors
                             (re-estimated against the oracle reference) - the
                             full, unattainable upper bound. Per the milestone's
                             own instruction, this is a separate upper-bound
                             experiment since it changes motion vectors too.

For EVERY variant, `reference` and `symbols` are fed through the SAME, real,
unmodified `model11`/`assign_codebook`/`coding_codebook` via
`m13.encode_frame_recalibrated` - so `A_real`'s byte count for a P-frame is
byte-for-byte what the deployed coder actually produces for that frame (not
an approximation of it), and B/C's byte counts are what the SAME real
pipeline would produce if fed a differently-referenced residual - never a
simplified proxy model.

The REAL chain (A) is the only one that ever advances `real_previous` -
B/C are read-only diagnostic side channels, exactly as M16 established.

Run:
  ./.venv/Scripts/python.exe scripts/m17_residual_diagnostic.py
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

from nvc.compression.codec import (
    decode_payload_to_latent, encode_latent_to_payload, latent_to_symbols, symbols_to_latent,
)
from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m17_residual_oracle_audit")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M11_DIR = Path("outputs/m11_autoregressive_entropy")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
RATE_POINTS = (5, 4, 3)
M11_G16_GROUP_SIZE = 16
VARIANTS = ("A_real", "B_oracle_ref_real_motion", "C_full_oracle")

WEAK_BELOW_PERCENT = 0.5
MEANINGFUL_ABOVE_PERCENT = 1.0


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verdict(gain: float) -> str:
    return ("meaningful" if gain >= MEANINGFUL_ABOVE_PERCENT else
           "marginal" if gain >= WEAK_BELOW_PERCENT else "weak")


@torch.no_grad()
def diagnose_residual_oracle(mc, ma, m13, mk, model, model11, assign_codebook, coding_codebook,
                             sequence, *, bits, intra_params, intra_entropy_model,
                             residual_params, zero, gop_size, block_size, search_range,
                             device, keep_payloads_and_references: bool = False) -> list[dict[str, Any]]:
    """Runs the REAL closed loop (identical to encode_multi) frame by frame;
    at every P-frame, additionally runs B/C as read-only diagnostic side
    channels through the SAME real m13.encode_frame_recalibrated. Returns
    one row per P-frame."""
    frames = sequence.load_frames()
    frame_count = frames.shape[0]
    types = mc.gop_frame_types(frame_count, gop_size)

    real_previous = None
    oracle_previous_raw = None
    latent_shape: tuple[int, ...] | None = None
    rows_out: list[dict[str, Any]] = []

    with mc.deterministic_kernels():
        for index in range(frame_count):
            frame = frames[index:index + 1].to(device)
            latent = model.encode(frame)
            latent_shape = tuple(latent.shape[1:])
            if types[index] == mc.FRAME_TYPE_I:
                payload, _ = encode_latent_to_payload(latent, params=intra_params,
                                                      entropy_model=intra_entropy_model)
                decoded, _ = decode_payload_to_latent(
                    payload, entropy_model=intra_entropy_model, params=intra_params,
                    shape=latent_shape)
                real_previous = model.decode(decoded.to(device))
                oracle_previous_raw = frame
                continue

            is_boundary = (index - 1) % gop_size == 0
            gop_position = index % gop_size

            oracle_previous = model.decode(model.encode(oracle_previous_raw))
            mv_real = mc.estimate_block_motion(real_previous, frame, block_size=block_size,
                                               search_range=search_range)
            mv_oracle = mc.estimate_block_motion(oracle_previous, frame, block_size=block_size,
                                                 search_range=search_range)

            warped_A = mc.warp_blocks(real_previous, mv_real, block_size=block_size)
            warped_B = mc.warp_blocks(oracle_previous, mv_real, block_size=block_size)
            warped_C = mc.warp_blocks(oracle_previous, mv_oracle, block_size=block_size)

            reference_A = model.encode(warped_A)
            reference_B = model.encode(warped_B)
            reference_C = model.encode(warped_C)

            symbols_A = latent_to_symbols(latent - reference_A, residual_params).reshape(latent_shape)
            symbols_B = latent_to_symbols(latent - reference_B, residual_params).reshape(latent_shape)
            symbols_C = latent_to_symbols(latent - reference_C, residual_params).reshape(latent_shape)

            row: dict[str, Any] = {"sequence_id": sequence.sequence_id, "index": index,
                                   "is_boundary": is_boundary, "gop_position": gop_position}
            variant_data = {"A_real": (reference_A, symbols_A), "B_oracle_ref_real_motion": (reference_B, symbols_B),
                           "C_full_oracle": (reference_C, symbols_C)}
            for name, (reference, symbols) in variant_data.items():
                payload, ideal = m13.encode_frame_recalibrated(
                    model11, assign_codebook, coding_codebook, reference, symbols, zero, bits=bits)
                target = torch.from_numpy(np.asarray(symbols, dtype=np.int64))[None].to(device)
                model_rows = ma._rows(model11.log_probabilities(reference, model11.planes(target, zero.to(device))))
                table_index = assign_codebook.assign_tensor(model_rows)
                row[f"{name}_bytes"] = len(payload)
                row[f"{name}_ideal_bits"] = ideal
                row[f"{name}_table_index"] = table_index
                row[f"{name}_symbols"] = np.asarray(symbols, dtype=np.int64).reshape(-1)
                row[f"{name}_residual_abs_mean"] = float((latent - reference).abs().mean())
                row[f"{name}_residual_variance"] = float((latent - reference).var())
                if keep_payloads_and_references:
                    row[f"{name}_payload"] = payload
                    row[f"{name}_reference"] = reference.detach().cpu()

            row["fraction_symbols_changed_A_vs_C"] = float(
                np.mean(row["A_real_symbols"] != row["C_full_oracle_symbols"]))
            row["fraction_assignments_changed_A_vs_B"] = float(
                np.mean(row["A_real_table_index"] != row["B_oracle_ref_real_motion_table_index"]))
            row["fraction_assignments_changed_B_vs_C"] = float(
                np.mean(row["B_oracle_ref_real_motion_table_index"] != row["C_full_oracle_table_index"]))
            row["fraction_assignments_changed_A_vs_C"] = float(
                np.mean(row["A_real_table_index"] != row["C_full_oracle_table_index"]))
            rows_out.append(row)

            # Advance the REAL chain EXACTLY as encode_multi does - B/C above
            # never feed back into what actually advances the reference.
            reconstructed_latent = reference_A + symbols_to_latent(
                symbols_A.reshape(-1), latent_shape, residual_params).to(device)
            real_previous = model.decode(reconstructed_latent)
            oracle_previous_raw = frame

    return rows_out


def _histogram_divergence(a: np.ndarray, b: np.ndarray, alphabet: int) -> float:
    """Symmetric KL (Jensen-Shannon-lite, not normalized) between two
    symbol histograms over the same alphabet - a secondary diagnostic."""
    ca = np.bincount(a, minlength=alphabet).astype(np.float64) + 1.0
    cb = np.bincount(b, minlength=alphabet).astype(np.float64) + 1.0
    pa, pb = ca / ca.sum(), cb / cb.sum()
    return float(np.sum(pa * np.log2(pa / pb)) + np.sum(pb * np.log2(pb / pa))) / 2.0


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _mean(subset, field):
        values = [r[field] for r in subset]
        return statistics.fmean(values) if values else None

    groups = {"boundary": [r for r in rows if r["is_boundary"]],
             "ordinary": [r for r in rows if not r["is_boundary"]], "all": rows}
    out = {}
    for name, subset in groups.items():
        if not subset:
            out[name] = {"count": 0}
            continue
        out[name] = {
            "count": len(subset),
            "mean_bytes_A_real": _mean(subset, "A_real_bytes"),
            "mean_bytes_B_oracle_ref_real_motion": _mean(subset, "B_oracle_ref_real_motion_bytes"),
            "mean_bytes_C_full_oracle": _mean(subset, "C_full_oracle_bytes"),
            "mean_ideal_bits_A_real": _mean(subset, "A_real_ideal_bits"),
            "mean_ideal_bits_C_full_oracle": _mean(subset, "C_full_oracle_ideal_bits"),
            "mean_fraction_symbols_changed_A_vs_C": _mean(subset, "fraction_symbols_changed_A_vs_C"),
            "mean_fraction_assignments_changed_A_vs_B": _mean(subset, "fraction_assignments_changed_A_vs_B"),
            "mean_fraction_assignments_changed_B_vs_C": _mean(subset, "fraction_assignments_changed_B_vs_C"),
            "mean_fraction_assignments_changed_A_vs_C": _mean(subset, "fraction_assignments_changed_A_vs_C"),
            "mean_residual_abs_mean_A_real": _mean(subset, "A_real_residual_abs_mean"),
            "mean_residual_abs_mean_C_full_oracle": _mean(subset, "C_full_oracle_residual_abs_mean"),
        }
    by_position: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_position.setdefault(r["gop_position"], []).append(r)
    out["by_gop_position"] = {
        str(pos): {
            "count": len(subset),
            "mean_bytes_A_real": _mean(subset, "A_real_bytes"),
            "mean_bytes_C_full_oracle": _mean(subset, "C_full_oracle_bytes"),
            "bytes_gap_percent": (
                (_mean(subset, "A_real_bytes") - _mean(subset, "C_full_oracle_bytes"))
                / _mean(subset, "A_real_bytes") * 100 if _mean(subset, "A_real_bytes") else None),
        }
        for pos, subset in sorted(by_position.items())
    }
    return out


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M17 Phase B/C/D/E: real-vs-oracle residual diagnostic through the deployed M11-G16+M13 pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=DEFAULT_M11_DIR)
    parser.add_argument("--m10k-dir", type=Path, default=DEFAULT_M10K_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--calibration-frames", type=int, default=400)
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

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    # VAL-B ONLY per Phase B's own instruction ("For VAL-B only initially").
    # SAME 9 val sequences / index-parity split M11-M16 all used.
    val_sequences = discover_sequences(args.manifest, split="val",
                                       max_frames_per_sequence=args.val_frames_per_sequence)
    val_b = val_sequences[1::2][:args.val_sequences]
    train_sequences_full = discover_sequences(args.manifest, split="train")

    print("=" * 122, flush=True)
    print("M17 PHASE B/C/D/E - RESIDUAL REAL vs ORACLE, DEPLOYED M11-G16 + M13 PIPELINE")
    print("=" * 122)
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"phase": "M17 Phase B/C/D/E residual diagnostic",
                              "val_b_sequence_ids": [s.sequence_id for s in val_b], "rate_points": []}

    for bits in args.rate_points:
        print(f"\n  ---- {bits}-bit ----", flush=True)
        calibration = mc.calibrate_grids(
            model, train_sequences_full, bits=bits, mode="per_channel", gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, reference_mode="mc",
            max_frames=args.calibration_frames)
        signature = ev.calibration_signature(calibration, bits=bits,
                                             calibration_frames=args.calibration_frames,
                                             quant_mode="per_channel")
        intra_params = calibration["intra_params"]
        intra_entropy_model = calibration["intra_entropy_model"]
        residual_params = calibration["residual_params"]

        # --- re-derive the FROZEN, DEPLOYED M13 residual arm - identical to
        # m13/m14/m15's own coded-validation scripts, never re-fit differently ---
        started = time.perf_counter()
        data = md.load_or_collect(model, checkpoint=args.checkpoint, manifest=args.manifest,
                                  bits=bits, device=device, cache_dir=cache_dir,
                                  log=lambda m: print(m, flush=True))
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
        gate = _load_script("m12_spatial_offline_gate")
        train_k = gate.m11_g16_prototype_indices(model11, assign_codebook, train_symbols,
                                                 data["train_references"], zero, device=device)
        val_k = gate.m11_g16_prototype_indices(model11, assign_codebook, val_symbols,
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
        print(f"    deployed residual arm re-derived: identity {m13_identity.hex()} "
             f"({time.perf_counter() - started:.1f}s)", flush=True)

        # --- the core diagnostic, VAL-B only ------------------------------------------------
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        for sequence in val_b:
            rows.extend(diagnose_residual_oracle(
                mc, ma, m13, mk, model, model11, assign_codebook, coding_codebook, sequence,
                bits=bits, intra_params=intra_params, intra_entropy_model=intra_entropy_model,
                residual_params=residual_params, zero=zero, gop_size=args.gop,
                block_size=args.block_size, search_range=args.search_range, device=device))
        print(f"    diagnosed {len(rows)} VAL-B P-frames ({time.perf_counter() - started:.1f}s)",
             flush=True)

        summary = _summarize(rows)
        total_a_bytes = sum(r["A_real_bytes"] for r in rows)
        total_c_bytes = sum(r["C_full_oracle_bytes"] for r in rows)
        total_b_bytes = sum(r["B_oracle_ref_real_motion_bytes"] for r in rows)
        gain_a_vs_c = (total_a_bytes - total_c_bytes) / total_a_bytes * 100
        gain_a_vs_b = (total_a_bytes - total_b_bytes) / total_a_bytes * 100
        gain_b_vs_c = (total_b_bytes - total_c_bytes) / total_b_bytes * 100 if total_b_bytes else 0.0

        alphabet = 2 ** bits
        all_a_symbols = np.concatenate([r["A_real_symbols"] for r in rows])
        all_c_symbols = np.concatenate([r["C_full_oracle_symbols"] for r in rows])
        divergence = _histogram_divergence(all_a_symbols, all_c_symbols, alphabet)

        print(f"    TOTAL VAL-B residual bytes: A(real)={total_a_bytes:,}  "
             f"B(oracle-ref,real-motion)={total_b_bytes:,}  C(full-oracle)={total_c_bytes:,}")
        print(f"    channel gain A->C (full oracle) = {gain_a_vs_c:+.4f}% -> {_verdict(gain_a_vs_c)}")
        print(f"    decomposition: A->B (reference-pixel effect only) = {gain_a_vs_b:+.4f}%  "
             f"B->C (motion-vector effect on top of oracle ref) = {gain_b_vs_c:+.4f}%")
        print(f"    symbol histogram divergence (A vs C, symmetric-KL): {divergence:.6f} bits", flush=True)

        report["rate_points"].append({
            "bits": bits, "residual_arm_identity": m13_identity.hex(),
            "total_val_b_bytes": {"A_real": total_a_bytes, "B_oracle_ref_real_motion": total_b_bytes,
                                  "C_full_oracle": total_c_bytes},
            "channel_gain_percent": {"A_vs_C_full_oracle": gain_a_vs_c,
                                     "A_vs_B_reference_pixel_effect": gain_a_vs_b,
                                     "B_vs_C_motion_vector_effect": gain_b_vs_c},
            "verdict_A_vs_C": _verdict(gain_a_vs_c),
            "symbol_histogram_divergence_bits": divergence,
            "summary": summary,
        })

    path = args.output_dir / "m17_residual_diagnostic.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
