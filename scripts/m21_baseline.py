"""M21 Phase 0 + Phase 1 - reproduce the frozen deployed baseline on VAL-B, and
trace the exact reference path by MEASURING it.

Phase 0 records, per bit depth: the five frozen entropy identities, residual
bytes (I and P separately), motion bytes, container bytes, BPP, PSNR, MS-SSIM,
residual symbol count, and the per-GOP-position byte breakdown - then checks the
P-frame residual totals against M17/M19/M20's own recorded VAL-B numbers. Those
were produced by a frame-level loop with no container and no motion coding, so
agreement is a real cross-check of two independently written pipelines, not a
tautology.

Phase 1 walks one P-frame and records every intermediate tensor's shape, dtype,
device and value range, plus a probe establishing which quantities are available
identically at encoder and decoder, and where a refinement can therefore be
inserted.

Run:
  ./.venv/Scripts/python.exe scripts/m21_baseline.py
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
    decode_payload_to_latent, encode_latent_to_payload, latent_to_symbols,
)
from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

# M17's recorded VAL-B P-frame residual bytes for the deployed M13 arm, which
# M19 and M20 both reproduced byte-for-byte. The number M21 must land on.
RECORDED_VAL_B_P_RESIDUAL_BYTES = {5: 1_333_275, 4: 908_404, 3: 549_083}
DEFAULT_M19_IDENTITIES = Path("outputs/m19_reference_error_audit/m19_identities.json")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _describe(name: str, tensor: torch.Tensor, note: str) -> dict[str, Any]:
    values = tensor.detach().float()
    return {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "device": str(tensor.device), "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean()), "std": float(values.std()), "note": note}


@torch.no_grad()
def trace_reference_path(mc, m13, m21, model, rig, sequence, *, gop_size, block_size,
                         search_range, device) -> dict[str, Any]:
    """Walk the deployed loop to its first P-frame and record every tensor.

    Also probes the two claims the whole milestone rests on: that the motion
    payload round trip is lossless (so the encoder warps with exactly the
    vectors the decoder reads), and that `previous`/`warped`/`reference` are
    the only tensors a refinement could touch without seeing the current frame.
    """
    spec = rig["spec"]
    frames = sequence.load_frames()
    types = mc.gop_frame_types(frames.shape[0], gop_size)
    previous = None
    stages: list[dict[str, Any]] = []

    with mc.deterministic_kernels():
        for index in range(frames.shape[0]):
            frame = frames[index:index + 1].to(device)
            latent = model.encode(frame)
            latent_shape = tuple(latent.shape[1:])
            if types[index] == mc.FRAME_TYPE_I:
                payload, _ = encode_latent_to_payload(
                    latent, params=rig["intra_params"],
                    entropy_model=rig["intra_entropy_model"])
                decoded, _ = decode_payload_to_latent(
                    payload, entropy_model=rig["intra_entropy_model"],
                    params=rig["intra_params"], shape=latent_shape)
                previous = model.decode(decoded.to(device))
                stages.append(_describe("I_frame_reconstruction", previous,
                                        "model.decode(decoded intra latent) - seeds `previous`"))
                continue

            stages.append(_describe("previous", previous,
                                    "decoded previous reconstruction; PIXEL insertion point - "
                                    "available identically at encoder and decoder, upstream of "
                                    "both motion estimation and warping"))
            stages.append(_describe("current_frame", frame,
                                    "ENCODER ONLY - a refinement must never read this"))
            stages.append(_describe("current_latent", latent,
                                    "ENCODER ONLY - model.encode(current frame)"))

            motion = mc.estimate_block_motion(previous, frame, block_size=block_size,
                                              search_range=search_range)
            motion_payload = mc.encode_motion_payload(
                motion, search_range=search_range,
                entropy_model=rig["motion_entropy_model"])
            decoded_motion = mc.decode_motion_payload(
                motion_payload, (frames.shape[2] // block_size, frames.shape[3] // block_size),
                search_range=search_range, entropy_model=rig["motion_entropy_model"])
            motion_roundtrip_lossless = bool(torch.equal(motion.cpu(), decoded_motion.cpu()))
            stages.append(_describe("motion", motion.float(),
                                    "ENCODER ONLY before coding; the DECODED field below is what "
                                    "both sides share"))
            stages.append(_describe("decoded_motion", decoded_motion.float(),
                                    "available identically at encoder and decoder"))

            warped = mc.warp_blocks(previous, decoded_motion, block_size=block_size)
            stages.append(_describe("warped", warped,
                                    "warp_blocks(previous, decoded_motion) - integer indexing, "
                                    "no interpolation; available identically at both sides"))
            reference_latent = model.encode(warped)
            stages.append(_describe("reference_latent", reference_latent,
                                    "model.encode(warped); LATENT insertion point - available "
                                    "identically at both sides, downstream of motion"))
            delta = latent - reference_latent
            stages.append(_describe("delta", delta, "ENCODER ONLY - latent minus reference"))
            symbols = latent_to_symbols(delta, rig["residual_params"])
            rows = m21.CANDIDATES[0]  # touch the frozen family so a rename breaks loudly
            del rows

            g16_rows = m13._load_script("m11_ar_entropy")._rows(
                spec["model"].log_probabilities(
                    reference_latent,
                    spec["model"].planes(
                        torch.from_numpy(symbols).reshape(1, *latent_shape).to(device),
                        spec["zero"].to(device))))
            table_index = spec["assign_codebook"].assign_tensor(g16_rows)
            payload, ideal = m13.encode_frame_recalibrated(
                spec["model"], spec["assign_codebook"], spec["coding_codebook"],
                reference_latent, symbols.reshape(latent_shape), spec["zero"],
                bits=rig["bits"])

            return {
                "sequence": sequence.sequence_id, "p_frame_index": index,
                "stages": stages,
                "motion_roundtrip_lossless": motion_roundtrip_lossless,
                "symbols": {"count": int(symbols.size), "dtype": str(symbols.dtype),
                            "min": int(symbols.min()), "max": int(symbols.max()),
                            "alphabet": 2 ** rig["bits"]},
                "g16_rows": {"shape": list(g16_rows.shape), "dtype": str(g16_rows.dtype),
                             "device": str(g16_rows.device)},
                "table_index": {"count": int(table_index.size), "distinct": int(
                    np.unique(table_index).size), "K": int(spec["assign_codebook"].size)},
                "payload_bytes": len(payload), "ideal_bits": ideal,
                "insertion_points": [
                    {"name": "pixel", "tensor": "previous",
                     "available_at_decoder": True, "upstream_of_motion": True,
                     "reads_current_frame": False},
                    {"name": "latent", "tensor": "reference_latent",
                     "available_at_decoder": True, "upstream_of_motion": False,
                     "reads_current_frame": False},
                ],
                "forbidden_inputs": ["current_frame", "current_latent", "delta", "symbols",
                                     "future frames", "pre-coding motion vectors"],
            }
    raise RuntimeError(f"{sequence.sequence_id} has no P-frame at GOP {gop_size}")


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M21 Phase 0/1: frozen baseline + exact reference-path trace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt"))
    parser.add_argument("--m11-dir", type=Path, default=Path("outputs/m11_autoregressive_entropy"))
    parser.add_argument("--m10k-dir", type=Path, default=Path("outputs/m10k_learned_entropy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m21_reference_refinement"))
    parser.add_argument("--m19-identities", type=Path, default=DEFAULT_M19_IDENTITIES)
    parser.add_argument("--rate-points", type=int, nargs="+", default=[5, 4, 3])
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
    parser.add_argument("--baseline-test-log", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

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
    recorded = json.loads(args.m19_identities.read_text(encoding="utf-8"))

    git_status = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, check=False).stdout.strip()
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False).stdout.strip()
    unexpected = [line for line in git_status.splitlines()
                  if "m21" not in line.lower() and "CHANGELOG" not in line]

    print("=" * 124)
    print("M21 PHASE 0/1 - FROZEN BASELINE + EXACT REFERENCE-PATH TRACE")
    print("=" * 124)
    print(f"  git HEAD: {git_head}")
    print(f"  non-M21 working-tree changes: {unexpected if unexpected else 'none'}")
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}")
    print(f"  motion-table TRAIN allocation: {len(motion_train)} sequences "
          f"@ {args.train_frames_per_sequence} frames", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = args.output_dir / "baseline_streams"
    report: dict[str, Any] = {
        "phase": "M21 Phase 0/1", "git_head": git_head,
        "git_status_porcelain": git_status,
        "non_m21_working_tree_changes": unexpected,
        "clean_except_m21": not unexpected,
        "baseline_test_suite": None,
        "val_b_sequence_ids": [s.sequence_id for s in val_b],
        "declared_candidates": [c.name for c in m21.CANDIDATES],
        "rate_points": [], "all_identities_match": True, "all_bytes_match": True,
    }
    if args.baseline_test_log and args.baseline_test_log.is_file():
        lines = args.baseline_test_log.read_text(encoding="utf-8", errors="replace").splitlines()
        report["baseline_test_suite"] = [ln for ln in lines
                                         if " passed" in ln or " failed" in ln][-1:]

    trace: dict[str, Any] = {"phase": "M21 Phase 1", "rate_points": {}}

    for bits in args.rate_points:
        rig = m21.prepare_rate_point(
            model, bits=bits, manifest=args.manifest, checkpoint=args.checkpoint,
            m11_dir=args.m11_dir, m10k_dir=args.m10k_dir, device=device, cache_dir=cache_dir,
            train_full=train_full, motion_train=motion_train,
            calibration_frames=args.calibration_frames, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            motion_cache=args.output_dir / "m21_deployed_motion_table.json")

        expected = recorded[str(bits)]
        identity_match = (rig["residual_identity"] == expected["residual_identity"]
                          and rig["assign_codebook_id"] == expected["assign_codebook_id"]
                          and rig["coding_codebook_id"] == expected["coding_codebook_id"]
                          and rig["motion_identity"] == m21.DEPLOYED_MOTION_IDENTITY)
        report["all_identities_match"] &= identity_match

        run = m21.run_candidate(
            mc, m13, model, val_b, rig["spec"], m21.IDENTITY, stream_dir,
            intra_params=rig["intra_params"], intra_entropy_model=rig["intra_entropy_model"],
            residual_params=rig["residual_params"],
            motion_entropy_model=rig["motion_entropy_model"], bits=bits, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range)
        summary = run["aggregate"]
        bytes_match = (summary["total_p_frame_residual_bytes"]
                       == RECORDED_VAL_B_P_RESIDUAL_BYTES.get(bits))
        report["all_bytes_match"] &= bytes_match

        if bits == args.rate_points[0]:
            trace["rate_points"][str(bits)] = trace_reference_path(
                mc, m13, m21, model, rig, val_b[0], gop_size=args.gop,
                block_size=args.block_size, search_range=args.search_range, device=device)

        print(f"\n  ---- {bits}-bit ----")
        print(f"    residual={rig['residual_identity']}  intra={rig['intra_identity']}  "
              f"motion={rig['motion_identity']}  identities_match={identity_match}")
        print(f"    P-residual={summary['total_p_frame_residual_bytes']:,} "
              f"(recorded {RECORDED_VAL_B_P_RESIDUAL_BYTES.get(bits):,})  match={bytes_match}")
        print(f"    I-residual={summary['total_i_frame_residual_bytes']:,}  "
              f"motion={summary['total_motion_bytes']:,}  overhead="
              f"{summary['total_container_overhead_bytes']:,}  "
              f"container={summary['total_container_bytes']:,}")
        print(f"    BPP={summary['stream_bpp']:.6f}  PSNR={summary['mean_psnr_db']:.4f} dB  "
              f"MS-SSIM={summary['mean_msssim']:.6f}  P-frames={summary['p_frames']}  "
              f"symbols/frame={16384}")
        print(f"    byte accounting closes: {summary['byte_accounting_closes']}   "
              f"decode checks: {run['decode_checks']}", flush=True)

        report["rate_points"].append({
            "bits": bits, "identities": {
                "residual": rig["residual_identity"], "intra": rig["intra_identity"],
                "motion": rig["motion_identity"], "assign_codebook": rig["assign_codebook_id"],
                "coding_codebook": rig["coding_codebook_id"],
                "m10k": rig["m10k_identity"], "calibration_signature": rig["signature"]},
            "expected_identities": expected, "identities_match": identity_match,
            "m13_smoothing_strength": rig["m13_strength"],
            "baseline": summary, "per_sequence": run["per_sequence"],
            "gop_position": run["gop_position"], "decode_checks": run["decode_checks"],
            "recorded_p_residual_bytes": RECORDED_VAL_B_P_RESIDUAL_BYTES.get(bits),
            "bytes_match": bytes_match,
            "residual_symbols_total": summary["p_frames"] * 16384,
        })

    ok = (report["all_identities_match"] and report["all_bytes_match"]
          and report["clean_except_m21"])
    report["status"] = "CONFIRMED" if ok else "MISMATCH - STOP"
    (args.output_dir / "m21_baseline.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "m21_reference_path.json").write_text(
        json.dumps(trace, indent=2, default=str), encoding="utf-8")
    print(f"\nBASELINE STATUS: {report['status']}")
    print(f"Reports: {args.output_dir / 'm21_baseline.json'}, "
          f"{args.output_dir / 'm21_reference_path.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
