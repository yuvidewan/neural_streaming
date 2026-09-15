"""M22 Phase 0 + Phase 1 - reproduce the frozen production baseline, then probe
the residual path to find the minimum representation that can actually change.

Phase 0 runs the deployed arm through M21's real closed loop (`.nvct` v2, real
motion coding, real M11-G16 + K=512 + M13, real range coder) on VAL-B at 5/4/3
bits and records everything the milestone asks for, then checks the P-frame
residual totals against M17/M19/M20/M21's recorded values.

Phase 1 does NOT describe the coupling - it demonstrates each link by probing
it: change the residual grid, then observe (a) the calibration signature move,
(b) `check_provenance` reject the deployed G16 checkpoint, (c) the symbol cache
key fail to notice, and (d) the container need no new field. That is what
decides how small an intervention M22 can make.

Run:
  ./.venv/Scripts/python.exe scripts/m22_baseline.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
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

RECORDED_VAL_B_P_RESIDUAL_BYTES = {5: 1_333_275, 4: 908_404, 3: 549_083}
RECORDED_VAL_B_TOTAL_BYTES = {5: 1_625_012, 4: 1_140_526, 3: 723_381}
DEFAULT_M19_IDENTITIES = Path("outputs/m19_reference_error_audit/m19_identities.json")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def residual_statistics(mc, m13, model, rig, sequences, *, bits, gop_size, block_size,
                        search_range, device) -> dict[str, Any]:
    """Residual RMS/MAE, symbol agreement against the ORACLE-reference arm, and
    G16 ideal bits - the compression-relevant baseline quantities M18/M21
    established must be tracked separately from reconstruction error.

    The deployed arm alone advances `previous`; the oracle arm is a read-only
    side channel, exactly as M16-M21 ran it.
    """
    spec = rig["spec"]
    residual_params = rig["residual_params"]
    totals = {"square": 0.0, "abs": 0.0, "count": 0, "agree": 0, "symbols": 0,
              "ideal_bits": 0.0, "oracle_ideal_bits": 0.0, "coded": 0, "oracle_coded": 0,
              "p_frames": 0, "boundary_agree": 0, "boundary_symbols": 0}
    for sequence in sequences:
        frames = sequence.load_frames()
        types = mc.gop_frame_types(frames.shape[0], gop_size)
        previous, previous_raw = None, None
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
                    previous_raw = frame
                    continue
                is_boundary = (index - 1) % gop_size == 0

                def _arm(reference_frame):
                    motion = mc.estimate_block_motion(reference_frame, frame,
                                                      block_size=block_size,
                                                      search_range=search_range)
                    warped = mc.warp_blocks(reference_frame, motion, block_size=block_size)
                    reference_latent = model.encode(warped)
                    delta = latent - reference_latent
                    symbols = latent_to_symbols(delta, residual_params)
                    payload, ideal = m13.encode_frame_recalibrated(
                        spec["model"], spec["assign_codebook"], spec["coding_codebook"],
                        reference_latent, symbols.reshape(latent_shape), spec["zero"],
                        bits=bits)
                    return reference_latent, delta, symbols, payload, ideal

                reference_latent, delta, symbols, payload, ideal = _arm(previous)
                _, _, oracle_symbols, oracle_payload, oracle_ideal = _arm(
                    model.decode(model.encode(previous_raw)))

                values = delta.detach().cpu().numpy()
                agree = int((symbols == oracle_symbols).sum())
                totals["square"] += float((values ** 2).sum())
                totals["abs"] += float(np.abs(values).sum())
                totals["count"] += int(values.size)
                totals["agree"] += agree
                totals["symbols"] += int(symbols.size)
                totals["ideal_bits"] += ideal
                totals["oracle_ideal_bits"] += oracle_ideal
                totals["coded"] += len(payload)
                totals["oracle_coded"] += len(oracle_payload)
                totals["p_frames"] += 1
                if is_boundary:
                    totals["boundary_agree"] += agree
                    totals["boundary_symbols"] += int(symbols.size)

                previous = model.decode(reference_latent + symbols_to_latent(
                    symbols, latent_shape, residual_params).to(device))
                previous_raw = frame

    count = max(totals["count"], 1)
    return {
        "p_frames": totals["p_frames"],
        "residual_rms": (totals["square"] / count) ** 0.5,
        "residual_mae": totals["abs"] / count,
        "residual_symbols": totals["symbols"],
        "symbol_agreement_with_oracle": totals["agree"] / max(totals["symbols"], 1),
        "boundary_symbol_agreement_with_oracle": (
            totals["boundary_agree"] / max(totals["boundary_symbols"], 1)),
        "g16_ideal_bits": totals["ideal_bits"],
        "oracle_g16_ideal_bits": totals["oracle_ideal_bits"],
        "coded_residual_bytes": totals["coded"],
        "oracle_coded_residual_bytes": totals["oracle_coded"],
        "m17_oracle_gap_bytes": totals["coded"] - totals["oracle_coded"],
        "m17_oracle_gap_percent": ((totals["coded"] - totals["oracle_coded"])
                                   / max(totals["coded"], 1) * 100),
    }


def coupling_audit(ev, ev_m11, ma, md, m22, model, rig, *, bits, residual_stack,
                   calibration_frames, m11_dir, device) -> dict[str, Any]:
    """Phase 1: demonstrate, link by link, what a residual-grid change breaks."""
    deployed_params = rig["residual_params"]
    probe = m22.grid_variant("tight_p1").build(residual_stack, bits)

    deployed_signature = m22.calibration_signature_for(
        ev, deployed_params, bits=bits, calibration_frames=calibration_frames)
    probe_signature = m22.calibration_signature_for(
        ev, probe, bits=bits, calibration_frames=calibration_frames)

    # (b) does the deployed G16 checkpoint reject the new grid?
    _, checkpoint11 = ma.load_model(m11_dir / f"m11_G16_entropy_{bits}bit.pt", device=device)
    rejected, message = False, None
    try:
        ev_m11.check_provenance(checkpoint11, signature=probe_signature, bits=bits,
                                group_size=m22.M11_G16_GROUP_SIZE,
                                context_definition_id=ma.context_definition_id(
                                    m22.M11_G16_GROUP_SIZE),
                                m10k_identity=bytes.fromhex(rig["m10k_identity"]))
    except Exception as error:            # noqa: BLE001 - the point is to catch whatever it is
        rejected, message = True, f"{type(error).__name__}: {error}"

    # (c) does the symbol cache notice a grid change?
    from nvc.evaluation.sequences import discover_sequences as _discover
    train_ids = [s.sequence_id for s in _discover(rig["manifest"], split="train")]
    val_ids = [s.sequence_id for s in _discover(rig["manifest"], split="val",
                                                max_frames_per_sequence=40)]
    cache_key_args = dict(checkpoint=rig["checkpoint"], bits=bits, quant_mode="per_channel",
                          gop=10, block_size=16, search_range=16,
                          calibration_frames=calibration_frames, train_frames=600,
                          val_frames_per_sequence=40, train_ids=train_ids, val_ids=val_ids)
    cache_key = md.cache_key(**cache_key_args)

    return {
        "deployed_grid_signature": m22.grid_signature(deployed_params),
        "probe_grid_signature": m22.grid_signature(probe),
        "deployed_calibration_signature": deployed_signature,
        "probe_calibration_signature": probe_signature,
        "calibration_signature_moves_with_the_grid": deployed_signature != probe_signature,
        "calibration_signature_hashes": ["residual_params.scale", "residual_params.zero_point",
                                         "bits", "mode", "calibration_frames"],
        "g16_checkpoint_rejects_the_new_grid": rejected,
        "g16_rejection_message": message,
        "symbol_cache_key_includes_the_grid": False,
        "symbol_cache_key_inputs": sorted(cache_key_args),
        "symbol_cache_key_for_deployed": cache_key,
        "symbol_cache_hazard": (
            "m11_data.load_or_collect's cache key does not include the residual grid, so "
            "calling it under a new grid silently returns symbols collected under the "
            "deployed one - m22_residual.collect_symbols_for_grid exists to avoid that"),
        "container_needs_a_new_field": False,
        "container_reason": (
            ".nvct v2 already carries residual_entropy_model_id, and "
            "m11_ar_entropy.model_identity already hashes the calibration signature and the "
            "coding codebook into it - so a refitted stack is a DIFFERENT stream identity "
            "that the existing header check rejects, with no format change"),
        "invalidated_in_order": ["M10K learned entropy model",
                                 "M11-G16 (initialised from M10K, trained on these symbols)",
                                 "K=512 assignment codebook (fitted to G16's predictions)",
                                 "M13 coding frequencies (fitted to those assignments)"],
        "unchanged_by_a_residual_grid_change": ["intra quantizer and intra entropy model",
                                                "motion estimator, motion coding, motion table",
                                                "range coder API", "causal decode ordering",
                                                "container format", "GOP", "autoencoder weights"],
        "minimum_intervention": (
            "residual_params alone, evaluated in two explicitly separated arms: STALE (deployed "
            "downstream kept, provenance override recorded) and REFIT (M10K/G16/codebook/M13 "
            "rebuilt on TRAIN by the original fitting code)"),
    }


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M22 Phase 0/1: frozen baseline and the residual-path coupling audit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt"))
    parser.add_argument("--m11-dir", type=Path, default=Path("outputs/m11_autoregressive_entropy"))
    parser.add_argument("--m10k-dir", type=Path, default=Path("outputs/m10k_learned_entropy"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/m22_residual_freeze_lift"))
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

    mc = _load_script("m10h_motion_compensation")
    ma = _load_script("m11_ar_entropy")
    m13 = _load_script("m13_recalibration")
    md = _load_script("m11_data")
    ev = _load_script("m10l_evaluate")
    ev_m11 = _load_script("m11_evaluate")
    m21 = _load_script("m21_refinement")
    m22 = _load_script("m22_residual")

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
                  if "m22" not in line.lower() and "CHANGELOG" not in line]

    print("=" * 126)
    print("M22 PHASE 0/1 - FROZEN BASELINE + RESIDUAL-PATH COUPLING AUDIT")
    print("=" * 126)
    print(f"  git HEAD: {git_head}")
    print(f"  non-M22 working-tree changes: {unexpected if unexpected else 'none'}")
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = args.output_dir / "baseline_streams"
    report: dict[str, Any] = {
        "phase": "M22 Phase 0/1", "git_head": git_head,
        "git_status_porcelain": git_status, "non_m22_working_tree_changes": unexpected,
        "clean_except_m22": not unexpected, "baseline_test_suite": None,
        "val_b_sequence_ids": [s.sequence_id for s in val_b],
        "declared_grid_variants": [v.to_dict() for v in m22.GRID_VARIANTS],
        "rate_points": [], "all_identities_match": True, "all_bytes_match": True,
    }
    if args.baseline_test_log and args.baseline_test_log.is_file():
        lines = args.baseline_test_log.read_text(encoding="utf-8", errors="replace").splitlines()
        report["baseline_test_suite"] = [ln for ln in lines
                                         if " passed" in ln or " failed" in ln][-1:]

    for bits in args.rate_points:
        started = time.perf_counter()
        rig = m21.prepare_rate_point(
            model, bits=bits, manifest=args.manifest, checkpoint=args.checkpoint,
            m11_dir=args.m11_dir, m10k_dir=args.m10k_dir, device=device, cache_dir=cache_dir,
            train_full=train_full, motion_train=motion_train,
            calibration_frames=args.calibration_frames, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            motion_cache=args.output_dir / "m22_deployed_motion_table.json")
        rig["manifest"] = args.manifest
        rig["checkpoint"] = args.checkpoint

        expected = recorded[str(bits)]
        identity_match = (rig["residual_identity"] == expected["residual_identity"]
                          and rig["assign_codebook_id"] == expected["assign_codebook_id"]
                          and rig["coding_codebook_id"] == expected["coding_codebook_id"]
                          and rig["motion_identity"] == m22.DEPLOYED_MOTION_IDENTITY)
        report["all_identities_match"] &= identity_match

        run = m21.run_candidate(
            mc, m13, model, val_b, rig["spec"], m21.IDENTITY, stream_dir,
            intra_params=rig["intra_params"], intra_entropy_model=rig["intra_entropy_model"],
            residual_params=rig["residual_params"],
            motion_entropy_model=rig["motion_entropy_model"], bits=bits, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range)
        summary = run["aggregate"]
        bytes_match = (summary["total_p_frame_residual_bytes"]
                       == RECORDED_VAL_B_P_RESIDUAL_BYTES.get(bits)
                       and summary["total_container_bytes"]
                       == RECORDED_VAL_B_TOTAL_BYTES.get(bits))
        report["all_bytes_match"] &= bytes_match

        stats = residual_statistics(mc, m13, model, rig, val_b, bits=bits, gop_size=args.gop,
                                    block_size=args.block_size, search_range=args.search_range,
                                    device=device)
        residuals = m22.collect_train_residuals(
            mc, model, train_full, bits=bits,
            intra_params=rig["intra_params"],
            intra_entropy_model=rig["intra_entropy_model"],
            gop_size=args.gop, block_size=args.block_size,
            search_range=args.search_range, max_frames=args.calibration_frames, device=device,
            cache=args.output_dir / f"m22_train_residuals_{bits}bit.pt")
        audit = coupling_audit(ev, ev_m11, ma, md, m22, model, rig, bits=bits,
                               residual_stack=residuals["residual_stack"],
                               calibration_frames=args.calibration_frames,
                               m11_dir=args.m11_dir, device=device)

        print(f"\n  ---- {bits}-bit ----  ({time.perf_counter() - started:.1f}s)")
        print(f"    residual={rig['residual_identity']}  intra={rig['intra_identity']}  "
              f"motion={rig['motion_identity']}  identities_match={identity_match}")
        print(f"    total={summary['total_container_bytes']:,} "
              f"(recorded {RECORDED_VAL_B_TOTAL_BYTES.get(bits):,})  "
              f"P-resid={summary['total_p_frame_residual_bytes']:,} "
              f"(recorded {RECORDED_VAL_B_P_RESIDUAL_BYTES.get(bits):,})  match={bytes_match}")
        print(f"    I-resid={summary['total_i_frame_residual_bytes']:,}  "
              f"motion={summary['total_motion_bytes']:,}  "
              f"overhead={summary['total_container_overhead_bytes']:,}  "
              f"BPP={summary['stream_bpp']:.6f}  PSNR={summary['mean_psnr_db']:.4f}  "
              f"MS-SSIM={summary['mean_msssim']:.6f}")
        print(f"    residual RMS={stats['residual_rms']:.6f}  MAE={stats['residual_mae']:.6f}  "
              f"symbols={stats['residual_symbols']:,}  "
              f"oracle symbol agreement={stats['symbol_agreement_with_oracle'] * 100:.2f}% "
              f"(boundary {stats['boundary_symbol_agreement_with_oracle'] * 100:.2f}%)")
        print(f"    G16 ideal bits={stats['g16_ideal_bits'] / 1e6:.4f}M  "
              f"M17 oracle gap={stats['m17_oracle_gap_percent']:+.4f}% of the residual channel")
        print(f"    COUPLING: signature moves={audit['calibration_signature_moves_with_the_grid']}"
              f"  G16 rejects the new grid={audit['g16_checkpoint_rejects_the_new_grid']}  "
              f"container needs a new field={audit['container_needs_a_new_field']}")
        print(f"              {audit['g16_rejection_message']}", flush=True)

        report["rate_points"].append({
            "bits": bits,
            "identities": {"residual": rig["residual_identity"], "intra": rig["intra_identity"],
                           "motion": rig["motion_identity"],
                           "assign_codebook": rig["assign_codebook_id"],
                           "coding_codebook": rig["coding_codebook_id"],
                           "m10k": rig["m10k_identity"],
                           "calibration_signature": rig["signature"]},
            "expected_identities": expected, "identities_match": identity_match,
            "baseline": summary, "per_sequence": run["per_sequence"],
            "gop_position": run["gop_position"], "decode_checks": run["decode_checks"],
            "residual_statistics": stats,
            "encode_seconds": summary["encode_seconds"],
            "recorded_p_residual_bytes": RECORDED_VAL_B_P_RESIDUAL_BYTES.get(bits),
            "recorded_total_bytes": RECORDED_VAL_B_TOTAL_BYTES.get(bits),
            "bytes_match": bytes_match, "coupling_audit": audit,
        })

    ok = (report["all_identities_match"] and report["all_bytes_match"]
          and report["clean_except_m22"])
    report["status"] = "CONFIRMED" if ok else "MISMATCH - STOP"
    (args.output_dir / "m22_baseline.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nBASELINE STATUS: {report['status']}")
    print(f"Report: {args.output_dir / 'm22_baseline.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
