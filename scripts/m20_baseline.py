"""M20 Phase 0 - confirm nothing has drifted since M19, and reproduce M19's
baseline ROUTING statistics (not just its byte totals) from scratch.

Checks, in order:

  1. the three frozen identities per bit depth - the M13 residual arm
     identity, the M10L K=512 assignment-codebook identity and the M13
     recalibrated coding-codebook identity - against M19's own recorded
     `m19_identities.json`;
  2. VAL-B residual bytes for the real and full-oracle arms against M17's
     recorded totals (M19 Phase 0's own check, re-run);
  3. M19's routing statistics: mean assignment churn, the symbol-change x
     assignment-change 2x2 table, and the routing-only share of excess bits.

(3) is the part M20 actually builds on, so it is recomputed here rather than
read out of M19's JSON - if the ~9-12% routing-only share does not reproduce,
M20 has no premise and should stop.

Run:
  ./.venv/Scripts/python.exe scripts/m20_baseline.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
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

DEFAULT_M17_JSON = Path("outputs/m17_residual_oracle_audit/m17_residual_diagnostic.json")
DEFAULT_M19_IDENTITIES = Path("outputs/m19_reference_error_audit/m19_identities.json")
DEFAULT_M19_ANALYSIS = Path("outputs/m19_reference_error_audit/m19_analysis.json")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def baseline_routing(mc, ma, m13, m20, model, rig, sequence, *, bits, gop_size, block_size,
                     search_range, device) -> list[dict[str, Any]]:
    """The real and full-oracle arms of M17/M19, frame by frame, recording the
    per-position routing facts M20 needs (assignment, symbol, coded length).

    The REAL chain is the only one that advances `real_previous` - the oracle
    arm is a read-only diagnostic side channel, exactly as M16-M19 established.
    """
    model11 = rig["model11"]
    assign_codebook, coding_codebook = rig["assign_codebook"], rig["coding_codebook"]
    residual_params, zero = rig["residual_params"], rig["zero"]
    frames = sequence.load_frames()
    types = mc.gop_frame_types(frames.shape[0], gop_size)

    real_previous = None
    oracle_previous_raw = None
    latent_shape: tuple[int, ...] | None = None
    rows_out: list[dict[str, Any]] = []

    with mc.deterministic_kernels():
        for index in range(frames.shape[0]):
            frame = frames[index:index + 1].to(device)
            latent = model.encode(frame)
            latent_shape = tuple(latent.shape[1:])
            if types[index] == mc.FRAME_TYPE_I:
                payload, _ = encode_latent_to_payload(
                    latent, params=rig["intra_params"], entropy_model=rig["intra_entropy_model"])
                decoded, _ = decode_payload_to_latent(
                    payload, entropy_model=rig["intra_entropy_model"], params=rig["intra_params"],
                    shape=latent_shape)
                real_previous = model.decode(decoded.to(device))
                oracle_previous_raw = frame
                continue

            oracle_previous = model.decode(model.encode(oracle_previous_raw))
            mv_real = mc.estimate_block_motion(real_previous, frame, block_size=block_size,
                                               search_range=search_range)
            mv_oracle = mc.estimate_block_motion(oracle_previous, frame, block_size=block_size,
                                                 search_range=search_range)
            reference_real = model.encode(mc.warp_blocks(real_previous, mv_real,
                                                         block_size=block_size))
            reference_oracle = model.encode(mc.warp_blocks(oracle_previous, mv_oracle,
                                                           block_size=block_size))
            symbols_real = latent_to_symbols(latent - reference_real,
                                             residual_params).reshape(latent_shape)
            symbols_oracle = latent_to_symbols(latent - reference_oracle,
                                               residual_params).reshape(latent_shape)

            row: dict[str, Any] = {
                "sequence_id": sequence.sequence_id, "index": index,
                "is_boundary": (index - 1) % gop_size == 0, "gop_position": index % gop_size}
            for name, (reference, symbols) in (("real", (reference_real, symbols_real)),
                                               ("oracle", (reference_oracle, symbols_oracle))):
                # `m13.encode_frame_recalibrated` is the deployed call; running it
                # here (rather than only M20's own path) is what makes the byte
                # totals directly comparable to M17's recorded ones.
                payload, ideal = m13.encode_frame_recalibrated(
                    model11, assign_codebook, coding_codebook, reference, symbols, zero, bits=bits)
                table_index = assign_codebook.assign_tensor(
                    m20.frame_rows(model11, reference, symbols, zero))
                flat = np.asarray(symbols, dtype=np.int64).reshape(-1)
                row[f"{name}_bytes"] = len(payload)
                row[f"{name}_ideal_bits"] = ideal
                row[f"{name}_table_index"] = table_index
                row[f"{name}_symbols"] = flat
                row[f"{name}_code_len"] = -np.log2(np.maximum(
                    coding_codebook.probabilities[table_index, flat], 1e-300))

            reconstructed_latent = reference_real + symbols_to_latent(
                symbols_real.reshape(-1), latent_shape, residual_params).to(device)
            real_previous = model.decode(reconstructed_latent)
            oracle_previous_raw = frame
            rows_out.append(row)
    del ma
    return rows_out


def routing_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """M19's Phase D/E numbers: churn, the 2x2 table, and the routing-only
    share of excess bits. Computed over every VAL-B position, not sampled."""
    assignment_changed = np.concatenate(
        [r["real_table_index"] != r["oracle_table_index"] for r in rows])
    symbol_changed = np.concatenate([r["real_symbols"] != r["oracle_symbols"] for r in rows])
    delta = np.concatenate([r["real_code_len"] - r["oracle_code_len"] for r in rows])
    total_excess = float(delta.sum())

    cells = {}
    for sym in (False, True):
        for assign in (False, True):
            mask = (symbol_changed == sym) & (assignment_changed == assign)
            cell_sum = float(delta[mask].sum())
            cells[f"symbol_changed={sym}_assignment_changed={assign}"] = {
                "fraction_of_positions": float(mask.mean()),
                "mean_delta_code_len_bits": float(delta[mask].mean()) if mask.any() else 0.0,
                "sum_delta_code_len_bits": cell_sum,
                "share_of_total_excess_bits": cell_sum / total_excess if total_excess else 0.0,
            }
    routing_only = cells["symbol_changed=False_assignment_changed=True"]
    return {
        "positions": int(assignment_changed.size),
        "mean_assignment_churn": float(assignment_changed.mean()),
        "mean_symbol_change_rate": float(symbol_changed.mean()),
        "total_excess_bits_real_minus_oracle": total_excess,
        "decomposition_2x2": cells,
        "routing_only_share_of_excess_bits": routing_only["share_of_total_excess_bits"],
        "routing_only_excess_bits": routing_only["sum_delta_code_len_bits"],
        "symbol_change_share_of_excess_bits": (
            cells["symbol_changed=True_assignment_changed=False"]["share_of_total_excess_bits"]
            + cells["symbol_changed=True_assignment_changed=True"]["share_of_total_excess_bits"]),
    }


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M20 Phase 0: baseline + M19 routing-statistic reproduction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt"))
    parser.add_argument("--m11-dir", type=Path, default=Path("outputs/m11_autoregressive_entropy"))
    parser.add_argument("--m10k-dir", type=Path, default=Path("outputs/m10k_learned_entropy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m20_codebook_hysteresis"))
    parser.add_argument("--m17-json", type=Path, default=DEFAULT_M17_JSON)
    parser.add_argument("--m19-identities", type=Path, default=DEFAULT_M19_IDENTITIES)
    parser.add_argument("--rate-points", type=int, nargs="+", default=[5, 4, 3])
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--val-sequences", type=int, default=4)
    parser.add_argument("--val-frames-per-sequence", type=int, default=None)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--baseline-test-log", type=Path, default=None,
                        help="Optional pytest log to record the pre-M20 suite result from.")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint),
                        ("--m17-json", args.m17_json), ("--m19-identities", args.m19_identities)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    mc = _load_script("m10h_motion_compensation")
    ma = _load_script("m11_ar_entropy")
    m13 = _load_script("m13_recalibration")
    m20 = _load_script("m20_hysteresis")
    md = _load_script("m11_data")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    val_b = m20.val_b_sequences(args.manifest, count=args.val_sequences,
                               max_frames=args.val_frames_per_sequence)
    train_full = discover_sequences(args.manifest, split="train")
    recorded_m17 = {rp["bits"]: rp for rp in
                    json.loads(args.m17_json.read_text(encoding="utf-8"))["rate_points"]}
    recorded_identities = json.loads(args.m19_identities.read_text(encoding="utf-8"))

    git_status = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, check=False).stdout.strip()
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False).stdout.strip()

    print("=" * 118)
    print("M20 PHASE 0 - BASELINE + M19 ROUTING REPRODUCTION")
    print("=" * 118)
    print(f"  git HEAD: {git_head}   working tree: "
          f"{'clean' if not git_status else git_status.replace(chr(10), ' | ')}")
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "M20 Phase 0", "git_head": git_head,
        "git_status_porcelain": git_status, "git_clean": not git_status,
        "baseline_test_suite": None,
        "val_b_sequence_ids": [s.sequence_id for s in val_b],
        "declared_margin_sweep": list(m20.MARGIN_SWEEP),
        "declared_hysteresis_states": list(m20.STATES),
        "rate_points": [], "all_identities_match": True, "all_bytes_match": True,
    }
    if args.baseline_test_log and args.baseline_test_log.is_file():
        tail = args.baseline_test_log.read_text(encoding="utf-8", errors="replace").strip()
        report["baseline_test_suite"] = [line for line in tail.splitlines()
                                         if " passed" in line or " failed" in line][-1:]

    for bits in args.rate_points:
        rig = m20.prepare_rate_point(
            model, bits=bits, manifest=args.manifest, checkpoint=args.checkpoint,
            m11_dir=args.m11_dir, m10k_dir=args.m10k_dir, device=device, cache_dir=cache_dir,
            train_full=train_full, calibration_frames=args.calibration_frames,
            gop_size=args.gop, block_size=args.block_size, search_range=args.search_range)

        rows: list[dict[str, Any]] = []
        for sequence in val_b:
            rows.extend(baseline_routing(mc, ma, m13, m20, model, rig, sequence, bits=bits,
                                         gop_size=args.gop, block_size=args.block_size,
                                         search_range=args.search_range, device=device))
        real_bytes = sum(r["real_bytes"] for r in rows)
        oracle_bytes = sum(r["oracle_bytes"] for r in rows)
        stats = routing_statistics(rows)

        expected = recorded_identities[str(bits)]
        identity_match = (rig["residual_identity"] == expected["residual_identity"]
                          and rig["assign_codebook_id"] == expected["assign_codebook_id"]
                          and rig["coding_codebook_id"] == expected["coding_codebook_id"])
        recorded_bytes = recorded_m17[bits]["total_val_b_bytes"]
        bytes_match = (real_bytes == recorded_bytes["A_real"]
                       and oracle_bytes == recorded_bytes["C_full_oracle"])
        report["all_identities_match"] &= identity_match
        report["all_bytes_match"] &= bytes_match

        print(f"\n  ---- {bits}-bit ----")
        print(f"    residual={rig['residual_identity']}  assign_cb={rig['assign_codebook_id']}  "
              f"coding_cb={rig['coding_codebook_id']}  identities_match={identity_match}")
        print(f"    P-frames={len(rows)}  real={real_bytes:,} (recorded {recorded_bytes['A_real']:,}) "
              f" oracle={oracle_bytes:,} (recorded {recorded_bytes['C_full_oracle']:,})  "
              f"bytes_match={bytes_match}")
        print(f"    churn={stats['mean_assignment_churn'] * 100:.2f}%  "
              f"symbol-change={stats['mean_symbol_change_rate'] * 100:.2f}%  "
              f"routing-only share of excess bits="
              f"{stats['routing_only_share_of_excess_bits'] * 100:.2f}%  "
              f"(symbol-change share={stats['symbol_change_share_of_excess_bits'] * 100:.2f}%)",
              flush=True)

        report["rate_points"].append({
            "bits": bits, "p_frames": len(rows),
            "residual_identity": rig["residual_identity"],
            "assign_codebook_id": rig["assign_codebook_id"],
            "coding_codebook_id": rig["coding_codebook_id"],
            "m13_smoothing_strength": rig["m13_strength"],
            "expected_identities": expected, "identities_match": identity_match,
            "real_bytes": real_bytes, "oracle_bytes": oracle_bytes,
            "recorded_real_bytes": recorded_bytes["A_real"],
            "recorded_oracle_bytes": recorded_bytes["C_full_oracle"], "bytes_match": bytes_match,
            "routing": stats,
        })

    ok = report["all_identities_match"] and report["all_bytes_match"]
    report["status"] = "CONFIRMED" if ok else "MISMATCH - STOP"
    print(f"\nBASELINE STATUS: {report['status']}")
    path = args.output_dir / "m20_baseline.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Report: {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
