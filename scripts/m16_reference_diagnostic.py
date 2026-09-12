"""M16 Phase B/C/D/E - the central diagnostic. At every P-frame, compute
motion estimation TWICE against two different references, without ever
feeding the second one back into what actually gets encoded:

  A - REAL DEPLOYED REFERENCE: `previous`, exactly as
      `m13_closed_loop.encode_multi` advances it (model.decode of the real,
      quantized `reconstructed_latent` - intra-quantized at frame 0,
      residual-quantized at every frame since, cumulative). This IS the
      reference actually used to produce the encoded stream.

  B - IDEAL LATENT REFERENCE (oracle, diagnostic only): model.decode(
      model.encode(raw_previous_frame)) - the autoencoder round-trip of the
      TRUE, un-quantized previous frame, freshly computed from ground truth
      every step (no error ever accumulates, by construction - the same
      idealization M14's collect_motion_symbols already uses, now compared
      against the REAL coder instead of against calibrate_grids).

  C - DECODED-LATENT REFERENCE: per Phase A's trace, no distinct
      realization of this exists in the current architecture - motion
      estimation is defined only in pixel space (`estimate_block_motion`/
      `warp_blocks` never accept a latent), so "the decoded latent, used
      directly" is not an available alternative to A. Not implemented;
      documented as such rather than forced.

The REAL chain (A) is what actually advances frame-to-frame (identical to
encode_multi); the ORACLE (B) is a parallel, read-only side computation.
Nothing here ever writes a stream or changes what gets encoded.

Run:
  ./.venv/Scripts/python.exe scripts/m16_reference_diagnostic.py
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
import torch.nn.functional as F

from nvc.compression.codec import (
    decode_payload_to_latent, encode_latent_to_payload, latent_to_symbols, symbols_to_latent,
)
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m16_gop_audit")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
RATE_POINTS = (5, 4, 3)

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


def _block_sad(current: torch.Tensor, reference: torch.Tensor, block_size: int) -> torch.Tensor:
    """Same per-block SAD formula `estimate_block_motion` uses internally,
    applied post-hoc to a chosen (current, warped-reference) pair."""
    absdiff = (current - reference).abs().sum(dim=1, keepdim=True)
    return F.avg_pool2d(absdiff, block_size)[0, 0] * (block_size * block_size)


@torch.no_grad()
def diagnose_sequence(mc, model, sequence, *, bits, intra_params, intra_entropy_model,
                      residual_params, gop_size, block_size, search_range, device):
    """Runs the REAL closed loop (identical to encode_multi) frame by frame;
    at every P-frame, ALSO computes the oracle comparison as a read-only
    side channel. Returns one row per P-frame."""
    frames = sequence.load_frames()
    frame_count = frames.shape[0]
    types = mc.gop_frame_types(frame_count, gop_size)

    real_previous = None
    oracle_previous_raw = None
    latent_shape: tuple[int, ...] | None = None
    rows: list[dict[str, Any]] = []
    real_reconstructions: list[torch.Tensor] = []

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
                    shape=tuple(latent.shape[1:]))
                real_previous = model.decode(decoded.to(device))
                real_reconstructions.append(real_previous.detach().cpu())
                oracle_previous_raw = frame
                continue

            is_boundary = (index - 1) % gop_size == 0
            gop_position = index % gop_size

            oracle_previous = model.decode(model.encode(oracle_previous_raw))

            mv_real = mc.estimate_block_motion(real_previous, frame, block_size=block_size,
                                               search_range=search_range)
            mv_oracle = mc.estimate_block_motion(oracle_previous, frame, block_size=block_size,
                                                 search_range=search_range)

            warped_real = mc.warp_blocks(real_previous, mv_real, block_size=block_size)
            warped_oracle = mc.warp_blocks(oracle_previous, mv_oracle, block_size=block_size)
            sad_real = _block_sad(frame, warped_real, block_size)
            sad_oracle = _block_sad(frame, warped_oracle, block_size)

            reference_latent_real = model.encode(warped_real)
            reference_latent_oracle = model.encode(warped_oracle)
            delta_real = latent - reference_latent_real
            delta_oracle = latent - reference_latent_oracle
            symbols_real = latent_to_symbols(delta_real, residual_params)
            symbols_oracle = latent_to_symbols(delta_oracle, residual_params)

            fraction_changed = float((mv_real != mv_oracle).float().mean())
            rows.append({
                "sequence_id": sequence.sequence_id, "index": index, "is_boundary": is_boundary,
                "gop_position": gop_position,
                "fraction_motion_vectors_changed": fraction_changed,
                "sad_real_mean": float(sad_real.mean()), "sad_oracle_mean": float(sad_oracle.mean()),
                "sad_real_median": float(sad_real.median()), "sad_oracle_median": float(sad_oracle.median()),
                "residual_energy_real": float(delta_real.abs().mean()),
                "residual_energy_oracle": float(delta_oracle.abs().mean()),
                "motion_symbols_real": mc.motion_to_symbols(mv_real, search_range=search_range),
                "motion_symbols_oracle": mc.motion_to_symbols(mv_oracle, search_range=search_range),
                "residual_symbols_real": symbols_real.reshape(-1),
                "residual_symbols_oracle": symbols_oracle.reshape(-1),
                "residual_channels": latent_shape[0],
            })

            # Advance the REAL chain EXACTLY as encode_multi does (bit-depth
            # -dependent: reference_latent_real + the DEQUANTIZED residual) -
            # the oracle side above is a read-only side channel, never fed
            # back into what actually advances the real reference.
            reconstructed_latent = reference_latent_real + symbols_to_latent(
                symbols_real, latent_shape, residual_params).to(device)
            real_previous = model.decode(reconstructed_latent)
            real_reconstructions.append(real_previous.detach().cpu())
            oracle_previous_raw = frame

    return rows, real_reconstructions


def _summarize(rows: list[dict[str, Any]], key_prefix: str) -> dict[str, Any]:
    def _agg(subset, field):
        values = [r[field] for r in subset]
        return statistics.fmean(values) if values else None

    groups = {
        "boundary": [r for r in rows if r["is_boundary"]],
        "ordinary": [r for r in rows if not r["is_boundary"]],
        "all": rows,
    }
    out = {}
    for name, subset in groups.items():
        out[name] = {
            "count": len(subset),
            "mean_fraction_motion_vectors_changed": _agg(subset, "fraction_motion_vectors_changed"),
            "mean_sad_real": _agg(subset, "sad_real_mean"),
            "mean_sad_oracle": _agg(subset, "sad_oracle_mean"),
            "median_sad_real": statistics.median(r["sad_real_median"] for r in subset) if subset else None,
            "median_sad_oracle": statistics.median(r["sad_oracle_median"] for r in subset) if subset else None,
            "mean_residual_energy_real": _agg(subset, "residual_energy_real"),
            "mean_residual_energy_oracle": _agg(subset, "residual_energy_oracle"),
        }
    # per-GOP-position trend (not just the binary split) - tests whether the
    # discrepancy is concentrated at the boundary or grows across the GOP.
    by_position: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_position.setdefault(r["gop_position"], []).append(r)
    out["by_gop_position"] = {
        str(pos): {"count": len(subset), "mean_sad_real": _agg(subset, "sad_real_mean"),
                  "mean_sad_oracle": _agg(subset, "sad_oracle_mean"),
                  "mean_fraction_changed": _agg(subset, "fraction_motion_vectors_changed")}
        for pos, subset in sorted(by_position.items())
    }
    return out


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M16 Phase B/C/D/E: real-vs-oracle reference diagnostic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--train-frames-per-sequence", type=int, default=8,
                        help="M15's own broad-coverage recipe, reused here for the "
                             "diagnostic's TRAIN entropy-fitting sample")
    parser.add_argument("--val-frames-per-sequence", type=int, default=40)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
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
    m15cal = _load_script("m15_calibration_policy")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    train_sequences_full = discover_sequences(args.manifest, split="train")
    # M15's own finding: sample TRAIN broadly, not sequentially, for any NEW
    # fitting this milestone does (the deployed quantizer itself still comes
    # from calibrate_grids's real, unmodified, sequential recipe below).
    train_sample = m15cal.build_policy("C_broad_576", train_sequences_full, seed=args.seed)
    val_sequences = discover_sequences(args.manifest, split="val",
                                       max_frames_per_sequence=args.val_frames_per_sequence)
    val_a, val_b = val_sequences[0::2], val_sequences[1::2]

    print("=" * 120, flush=True)
    print("M16 PHASE B/C/D/E - REAL vs ORACLE REFERENCE DIAGNOSTIC")
    print("=" * 120)
    print(f"  TRAIN sample (M15 broad policy C): {len(train_sample)} sequences, "
         f"{sum(s.frame_count for s in train_sample)} frames")
    print(f"  VAL-A: {[s.sequence_id for s in val_a]}")
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"phase": "M16 Phase B/C/D/E diagnostic", "rate_points": []}

    for bits in args.rate_points:
        print(f"\n  ---- {bits}-bit ----", flush=True)
        started = time.perf_counter()
        calibration = mc.calibrate_grids(
            model, train_sequences_full, bits=bits, mode="per_channel", gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, reference_mode="mc",
            max_frames=args.calibration_frames)
        intra_params = calibration["intra_params"]
        intra_entropy_model = calibration["intra_entropy_model"]
        residual_params = calibration["residual_params"]
        motion_bits = mc.motion_alphabet_bits(args.search_range)

        def _run(sequences, label):
            rows = []
            for sequence in sequences:
                sequence_rows, _ = diagnose_sequence(
                    mc, model, sequence, bits=bits, intra_params=intra_params,
                    intra_entropy_model=intra_entropy_model, residual_params=residual_params,
                    gop_size=args.gop, block_size=args.block_size, search_range=args.search_range,
                    device=device)
                rows.extend(sequence_rows)
            print(f"    {label}: {len(rows)} P-frames diagnosed "
                 f"({time.perf_counter() - started:.1f}s elapsed)", flush=True)
            return rows

        train_rows = _run(train_sample, "TRAIN")
        val_a_rows = _run(val_a, "VAL-A")
        val_b_rows = _run(val_b, "VAL-B")

        summary_val_b = _summarize(val_b_rows, "val_b")
        print(f"    VAL-B all P-frames: mean SAD real={summary_val_b['all']['mean_sad_real']:.4f} "
             f"oracle={summary_val_b['all']['mean_sad_oracle']:.4f}  "
             f"fraction MV changed={summary_val_b['all']['mean_fraction_motion_vectors_changed']:.4f}")
        print(f"      boundary: SAD real={summary_val_b['boundary']['mean_sad_real']:.4f} "
             f"oracle={summary_val_b['boundary']['mean_sad_oracle']:.4f}")
        print(f"      ordinary: SAD real={summary_val_b['ordinary']['mean_sad_real']:.4f} "
             f"oracle={summary_val_b['ordinary']['mean_sad_oracle']:.4f}", flush=True)

        # --- Phase D: offline entropy impact, TRAIN-fit / VAL-B held-out ------------------
        def _flat_motion(rows, key):
            return np.stack([r[key] for r in rows]) if rows else np.zeros((0, 1), dtype=np.int64)

        def _flat_residual(rows, key, channels):
            return np.stack([r[key].reshape(channels, -1) for r in rows]) if rows else \
                np.zeros((0, channels, 1), dtype=np.int64)

        channels = train_rows[0]["residual_channels"] if train_rows else 1

        train_motion_real = _flat_motion(train_rows, "motion_symbols_real")
        train_motion_oracle = _flat_motion(train_rows, "motion_symbols_oracle")
        val_b_motion_real = _flat_motion(val_b_rows, "motion_symbols_real")
        val_b_motion_oracle = _flat_motion(val_b_rows, "motion_symbols_oracle")

        motion_table_real = EmpiricalEntropyModel.from_symbols(
            train_motion_real.reshape(train_motion_real.shape[0], 2, -1),
            bits=motion_bits, num_tables=2)
        motion_table_oracle = EmpiricalEntropyModel.from_symbols(
            train_motion_oracle.reshape(train_motion_oracle.shape[0], 2, -1),
            bits=motion_bits, num_tables=2)

        def _motion_bits_per_symbol(model_, symbols):
            probabilities = model_.probabilities()
            reshaped = symbols.reshape(symbols.shape[0], 2, -1)  # [N, {dy,dx}, blocks]
            flat_symbols = reshaped.transpose(1, 0, 2).reshape(2, -1)  # table-major
            total, count = 0.0, 0
            for t in range(2):
                p = probabilities[t][flat_symbols[t]]
                total += float(-np.sum(np.log2(np.maximum(p, 1e-300))))
                count += flat_symbols[t].size
            return total / count

        motion_h_real = _motion_bits_per_symbol(motion_table_real, val_b_motion_real)
        motion_h_oracle_under_oracle_table = _motion_bits_per_symbol(motion_table_oracle, val_b_motion_oracle)
        motion_gain_percent = (motion_h_real - motion_h_oracle_under_oracle_table) / motion_h_real * 100

        train_residual_real = _flat_residual(train_rows, "residual_symbols_real", channels)
        train_residual_oracle = _flat_residual(train_rows, "residual_symbols_oracle", channels)
        val_b_residual_real = _flat_residual(val_b_rows, "residual_symbols_real", channels)
        val_b_residual_oracle = _flat_residual(val_b_rows, "residual_symbols_oracle", channels)

        residual_table_real = EmpiricalEntropyModel.from_symbols(
            train_residual_real, bits=bits, num_tables=channels)
        residual_table_oracle = EmpiricalEntropyModel.from_symbols(
            train_residual_oracle, bits=bits, num_tables=channels)

        def _residual_bits_per_symbol(model_, symbols):
            # symbols: [N, channels, HW], channel-major within each frame -
            # matches latent_to_symbols' own C-major flattening.
            probabilities = model_.probabilities()
            flat_symbols = symbols.reshape(-1)
            flat_table = np.tile(np.repeat(np.arange(channels), symbols.shape[-1]), symbols.shape[0])
            p = probabilities[flat_table, flat_symbols]
            return float(-np.sum(np.log2(np.maximum(p, 1e-300)))) / flat_symbols.size

        residual_h_real = _residual_bits_per_symbol(residual_table_real, val_b_residual_real)
        residual_h_oracle = _residual_bits_per_symbol(residual_table_oracle, val_b_residual_oracle)
        residual_gain_percent = (residual_h_real - residual_h_oracle) / residual_h_real * 100

        print(f"    Phase D (VAL-B, TRAIN-fit): motion H real={motion_h_real:.5f} "
             f"oracle={motion_h_oracle_under_oracle_table:.5f} gain={motion_gain_percent:+.4f}%  "
             f"[diagnostic proxy] residual H real={residual_h_real:.5f} "
             f"oracle={residual_h_oracle:.5f} gain={residual_gain_percent:+.4f}%", flush=True)

        # --- Phase E: oracle upper bound, channel vs total-stream ------------------------
        blocks_per_frame = val_b_motion_real.shape[1] // 2 if val_b_motion_real.ndim > 1 else 0
        motion_bytes_per_frame_real = motion_h_real * blocks_per_frame * 2 / 8
        motion_bytes_per_frame_oracle = motion_h_oracle_under_oracle_table * blocks_per_frame * 2 / 8

        report["rate_points"].append({
            "bits": bits,
            "reference_summary_val_b": summary_val_b,
            "phase_d_offline_entropy": {
                "motion_h_real_bits_per_symbol": motion_h_real,
                "motion_h_oracle_bits_per_symbol": motion_h_oracle_under_oracle_table,
                "motion_channel_gain_percent": motion_gain_percent,
                "motion_verdict": _verdict(motion_gain_percent),
                "residual_h_real_bits_per_symbol_diagnostic_proxy": residual_h_real,
                "residual_h_oracle_bits_per_symbol_diagnostic_proxy": residual_h_oracle,
                "residual_channel_gain_percent_diagnostic_proxy": residual_gain_percent,
            },
            "phase_e_oracle_bound": {
                "motion_bytes_per_frame_real": motion_bytes_per_frame_real,
                "motion_bytes_per_frame_oracle": motion_bytes_per_frame_oracle,
            },
            "motion_table_real_identity": motion_table_real.model_id().hex(),
            "motion_table_oracle_identity": motion_table_oracle.model_id().hex(),
        })

    path = args.output_dir / "m16_reference_diagnostic.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
