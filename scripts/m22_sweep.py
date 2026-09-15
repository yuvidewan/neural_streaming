"""M22 Phases 3-5 - the residual-grid sweep, in two explicitly separated arms.

For every pre-registered grid variant:

  STALE   the new grid with the deployed M10K/G16/K=512/M13 kept exactly as they
          are. The M11-G16 provenance guard is bypassed by an EXPLICIT, recorded
          override - this arm exists precisely to price what that guard is
          protecting, so silently satisfying it would defeat the measurement.
  REFIT   the new grid with M10K, G16, the K=512 codebook and the M13 frequencies
          rebuilt on TRAIN by the same unmodified fitting code that produced the
          deployed stack. Every fitted component gets a new identity, which the
          `.nvct` v2 header already carries - no format change.

Both arms run the REAL deployed closed loop and are decoded back from their
containers. Actual coded total-stream bytes are the primary metric; PSNR and
MS-SSIM are reported alongside and gated by the pre-declared distortion guard,
because changing a quantizer grid is a rate/distortion move by construction.

Per the milestone's codebook-compatibility rule, the STALE arm also QUANTIFIES
the incompatibility (how far the deployed codebook's assignments and coding
tables drift under the new grid) rather than silently recalibrating.

DAVIS TEST is never opened by this script.

Run:
  ./.venv/Scripts/python.exe scripts/m22_sweep.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Deltas against the deployed control. Positive percentages mean FEWER
    bytes; PSNR/MS-SSIM deltas keep their natural sign."""
    def _gain(field):
        base = baseline[field]
        return (base - candidate[field]) / base * 100 if base else 0.0

    return {
        "delta_total_bytes": candidate["total_container_bytes"]
        - baseline["total_container_bytes"],
        "delta_residual_bytes": candidate["total_residual_bytes"]
        - baseline["total_residual_bytes"],
        "delta_p_residual_bytes": candidate["total_p_frame_residual_bytes"]
        - baseline["total_p_frame_residual_bytes"],
        "delta_i_residual_bytes": candidate["total_i_frame_residual_bytes"]
        - baseline["total_i_frame_residual_bytes"],
        "delta_motion_bytes": candidate["total_motion_bytes"] - baseline["total_motion_bytes"],
        "total_stream_gain_percent": _gain("total_container_bytes"),
        "residual_gain_percent": _gain("total_residual_bytes"),
        "p_residual_gain_percent": _gain("total_p_frame_residual_bytes"),
        "ideal_bits_gain_percent": _gain("p_frame_ideal_bits"),
        "delta_bpp": candidate["stream_bpp"] - baseline["stream_bpp"],
        "delta_psnr_db": candidate["mean_psnr_db"] - baseline["mean_psnr_db"],
        "delta_msssim": candidate["mean_msssim"] - baseline["mean_msssim"],
    }


def resume_refit(path: Path, *, m22, ma, ml, cx, device, grid_sig: str, signature: str):
    """Reuse a refitted stack already on disk instead of retraining it.

    A refit takes ~9 minutes, so a crashed sweep should not throw away the ones
    that finished. But reuse is only legitimate if the artifact on disk was
    fitted to THIS grid by THIS code: the digest, the grid signature, the
    calibration signature and the code identity are all checked, and any
    mismatch RAISES rather than quietly evaluating a checkpoint against a
    quantizer it was never fitted to. Pass --no-resume to refit regardless.
    """
    m22m = _load_script("m22_mechanism")
    loaded = m22m.load_refit_checkpoint(path, device=device, m22=m22, ma=ma, ml=ml, cx=cx)
    record = loaded["record"]
    for field, found, expected in (
            ("grid signature", record["quantizer_identity"]["grid_signature"], grid_sig),
            ("calibration signature", record["quantizer_identity"]["calibration_signature"],
             signature)):
        if found != expected:
            raise ValueError(
                f"{path.name} records {field} {found}, but this sweep built {expected}. "
                "Delete the checkpoint or pass --no-resume; do NOT evaluate it as is.")
    if record["code_identity"] != m22.code_identity():
        # Deliberately NOT an error. A stale artifact is only a hazard if it is
        # USED; rebuilding it is always safe, so the correct response is to
        # refit rather than to abort a multi-hour sweep. Contrast the grid and
        # calibration checks above, which raise because reusing THAT checkpoint
        # would evaluate a stack under a quantizer it was never fitted to.
        changed = sorted(name for name, digest in m22.code_identity().items()
                         if record["code_identity"].get(name) != digest)
        print(f"    [resume] {path.name} was fitted by different code ({', '.join(changed)}); "
              "refitting instead of reusing it", flush=True)
        return None
    return loaded


@torch.no_grad()
def codebook_incompatibility(mc, m13, m21, model, spec, residual_params, sequences, *, bits,
                             gop_size, block_size, search_range, device,
                             deployed_spec) -> dict[str, Any]:
    """Phase 15: how badly does the DEPLOYED codebook fit the new grid's symbols?

    Recorded rather than repaired - the milestone forbids silently recalibrating
    it. Measures assignment drift and the coding tables' cross-entropy against
    the symbols the new grid actually produces.
    """
    from nvc.compression.codec import (
        decode_payload_to_latent, encode_latent_to_payload, latent_to_symbols,
        symbols_to_latent)

    ma = _load_script("m11_ar_entropy")
    totals = {"positions": 0, "assignment_changed": 0, "code_len": 0.0,
              "deployed_code_len": 0.0, "symbols": np.zeros(2 ** bits, dtype=np.int64)}
    for sequence in sequences:
        frames = sequence.load_frames()
        types = mc.gop_frame_types(frames.shape[0], gop_size)
        previous = None
        with mc.deterministic_kernels():
            for index in range(frames.shape[0]):
                frame = frames[index:index + 1].to(device)
                latent = model.encode(frame)
                latent_shape = tuple(latent.shape[1:])
                if types[index] == mc.FRAME_TYPE_I:
                    payload, _ = encode_latent_to_payload(
                        latent, params=spec["intra_params"],
                        entropy_model=spec["intra_entropy_model"])
                    decoded, _ = decode_payload_to_latent(
                        payload, entropy_model=spec["intra_entropy_model"],
                        params=spec["intra_params"], shape=latent_shape)
                    previous = model.decode(decoded.to(device))
                    continue
                motion = mc.estimate_block_motion(previous, frame, block_size=block_size,
                                                  search_range=search_range)
                warped = mc.warp_blocks(previous, motion, block_size=block_size)
                reference = model.encode(warped)
                symbols = latent_to_symbols(latent - reference, residual_params)
                target = torch.from_numpy(symbols).reshape(1, *latent_shape).to(device)

                new_rows = ma._rows(spec["model"].log_probabilities(
                    reference, spec["model"].planes(target, spec["zero"].to(device))))
                new_index = spec["assign_codebook"].assign_tensor(new_rows)
                old_rows = ma._rows(deployed_spec["model"].log_probabilities(
                    reference, deployed_spec["model"].planes(
                        target, deployed_spec["zero"].to(device))))
                old_index = deployed_spec["assign_codebook"].assign_tensor(old_rows)

                totals["positions"] += symbols.size
                totals["assignment_changed"] += int((new_index != old_index).sum())
                totals["code_len"] += float(-np.log2(np.maximum(
                    spec["coding_codebook"].probabilities[new_index, symbols], 1e-300)).sum())
                totals["deployed_code_len"] += float(-np.log2(np.maximum(
                    deployed_spec["coding_codebook"].probabilities[old_index, symbols],
                    1e-300)).sum())
                totals["symbols"] += np.bincount(symbols, minlength=2 ** bits)

                previous = model.decode(reference + symbols_to_latent(
                    symbols, latent_shape, residual_params).to(device))
    positions = max(totals["positions"], 1)
    probability = totals["symbols"] / max(totals["symbols"].sum(), 1)
    nonzero = probability[probability > 0]
    del m13, m21
    return {
        "positions": totals["positions"],
        "assignment_drift": totals["assignment_changed"] / positions,
        "refit_bits_per_symbol": totals["code_len"] / positions,
        "deployed_bits_per_symbol": totals["deployed_code_len"] / positions,
        "deployed_table_penalty_percent": (
            (totals["deployed_code_len"] - totals["code_len"]) / max(totals["code_len"], 1e-9)
            * 100),
        "symbol_entropy_bits": float(-(nonzero * np.log2(nonzero)).sum()),
    }


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M22 Phases 3-5: the residual-grid sweep (VAL-B; TEST never opened).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt"))
    parser.add_argument("--m11-dir", type=Path, default=Path("outputs/m11_autoregressive_entropy"))
    parser.add_argument("--m10k-dir", type=Path, default=Path("outputs/m10k_learned_entropy"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/m22_residual_freeze_lift"))
    parser.add_argument("--output-name", type=str, default="m22_sweep.json")
    parser.add_argument("--variants", type=str, nargs="*", default=None)
    parser.add_argument("--arms", type=str, nargs="+", default=["stale", "refit"],
                        choices=["stale", "refit"])
    parser.add_argument("--rate-points", type=int, nargs="+", default=None)
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--val-sequences", type=int, default=4)
    parser.add_argument("--val-frames-per-sequence", type=int, default=None)
    parser.add_argument("--train-frames-per-sequence", type=int, default=8)
    parser.add_argument("--train-frames", type=int, default=600)
    parser.add_argument("--refit-val-frames-per-sequence", type=int, default=40)
    parser.add_argument("--m10k-epochs", type=int, default=20)
    parser.add_argument("--g16-epochs", type=int, default=20)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--codebook-fit-rows", type=int, default=100_000)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="refit every stack even if a matching checkpoint exists")
    parser.set_defaults(resume=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    mc = _load_script("m10h_motion_compensation")
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    mk = _load_script("m10k_learned_entropy")
    mt = _load_script("m11_train")
    md = _load_script("m11_data")
    cx = _load_script("m11_causal_context")
    ce = _load_script("m10j_conditional_entropy")
    m13 = _load_script("m13_recalibration")
    ev = _load_script("m10l_evaluate")
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
    refit_val = discover_sequences(args.manifest, split="val",
                                   max_frames_per_sequence=args.refit_val_frames_per_sequence)
    motion_train = m21.broad_train_sequences(
        args.manifest, frames_per_sequence=args.train_frames_per_sequence)
    variants = [v for v in m22.GRID_VARIANTS
                if args.variants is None or v.name in args.variants]
    if args.variants:
        missing = set(args.variants) - {v.name for v in m22.GRID_VARIANTS}
        if missing:
            print(f"[ERROR] not pre-registered: {sorted(missing)}", file=sys.stderr)
            return 1
    rate_points = args.rate_points or [m22.SCREEN_BITS]

    print("=" * 142)
    print("M22 PHASES 3-5 - RESIDUAL-GRID SWEEP (VAL-B, held out; DAVIS TEST never opened)")
    print("=" * 142)
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}")
    print(f"  pre-registered grid variants: {[v.name for v in variants]}")
    print(f"  arms: {args.arms}   rate points: {rate_points}")
    print(f"  distortion guard (declared): PSNR >= -{m22.MAX_PSNR_REGRESSION_DB:.2f} dB, "
          f"MS-SSIM >= -{m22.MAX_MSSSIM_REGRESSION:.4f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = args.output_dir / "sweep_streams"
    report: dict[str, Any] = {
        "phase": "M22 Phases 3-5",
        "val_b_sequence_ids": [s.sequence_id for s in val_b],
        "grid_variants": [v.to_dict() for v in variants],
        "arms": list(args.arms), "rate_points": [],
        "declared_screen_bits": m22.SCREEN_BITS,
        "declared_confirm_bits": list(m22.CONFIRM_BITS),
        "declared_stage2_admission_percent": m22.STAGE2_ADMISSION_PERCENT,
        "distortion_guard": {"max_psnr_regression_db": m22.MAX_PSNR_REGRESSION_DB,
                             "max_msssim_regression": m22.MAX_MSSSIM_REGRESSION},
    }

    report_path = args.output_dir / args.output_name

    def persist() -> None:
        """Written after every row, not once at the end: a sweep is hours long and
        a crash in the last variant must not discard the measured ones."""
        report_path.write_text(json.dumps(report, indent=2, default=str),
                               encoding="utf-8")

    for bits in rate_points:
        started = time.perf_counter()
        deployed_rig = m21.prepare_rate_point(
            model, bits=bits, manifest=args.manifest, checkpoint=args.checkpoint,
            m11_dir=args.m11_dir, m10k_dir=args.m10k_dir, device=device, cache_dir=cache_dir,
            train_full=train_full, motion_train=motion_train,
            calibration_frames=args.calibration_frames, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            motion_cache=args.output_dir / "m22_deployed_motion_table.json")
        residuals = m22.collect_train_residuals(
            mc, model, train_full, bits=bits,
            intra_params=deployed_rig["intra_params"],
            intra_entropy_model=deployed_rig["intra_entropy_model"],
            gop_size=args.gop, block_size=args.block_size,
            search_range=args.search_range, max_frames=args.calibration_frames, device=device,
            cache=args.output_dir / f"m22_train_residuals_{bits}bit.pt")
        residual_stack = residuals["residual_stack"]
        # The identity control, enforced: if the TRAIN stack does not rebuild the
        # deployed grid exactly, every delta below is measured from the wrong
        # origin. Raising here is the whole point.
        m22.verify_deployed_grid(residual_stack, deployed_rig["residual_params"], bits=bits)

        rows: list[dict[str, Any]] = []
        baseline_summary = None
        # attached to the report up front, so `persist()` below captures partial
        # progress; `rows` is the same list object the report holds
        report["rate_points"].append({
            "bits": bits, "deployed_identity": deployed_rig["residual_identity"],
            "deployed_motion_identity": deployed_rig["motion_identity"],
            "candidates": rows,
        })
        for variant in variants:
            params = variant.build(residual_stack, bits)
            train_stats = m22.grid_statistics(residual_stack, params, bits=bits)
            signature = m22.calibration_signature_for(
                ev, params, bits=bits, calibration_frames=args.calibration_frames)
            is_control = variant.name == "deployed"

            for arm in args.arms:
                if is_control and arm == "refit" and "stale" in args.arms:
                    continue                    # the control is the same in both arms
                label = f"{variant.name}/{arm}"
                run_started = time.perf_counter()

                reused = None
                if arm == "stale":
                    spec = dict(deployed_rig["spec"])
                    provenance_note = (
                        "deployed M10K/G16/K=512/M13 kept; the M11-G16 calibration-signature "
                        "guard is bypassed by construction (the checkpoint is never re-checked "
                        "against the new grid) - recorded, not satisfied"
                        if not is_control else "control: deployed grid, deployed stack")
                    refit_info = None
                else:
                    ckpt_path = (args.output_dir / "checkpoints"
                                 / f"m22_{variant.name}_{bits}bit.pt")
                    if args.resume and ckpt_path.with_suffix(".provenance.json").is_file():
                        reused = resume_refit(
                            ckpt_path, m22=m22, ma=ma, ml=ml, cx=cx, device=device,
                            grid_sig=m22.grid_signature(params), signature=signature)
                if reused is not None:
                    print(f"    [{label}] reusing the verified refit checkpoint "
                          f"{ckpt_path.name} (grid + code identity match)", flush=True)
                    spec = reused["spec"]
                    record = reused["record"]
                    refit_info = {
                        "resumed_from_checkpoint": True,
                        "m10k_identity": record["entropy_identities"]["m10k"],
                        "residual_entropy_model_id": record["entropy_identities"].get(
                            "residual_entropy_model"),
                        "residual_identity": record["entropy_identities"]["residual"],
                        "assign_codebook_id": record["entropy_identities"]["assign_codebook"],
                        "coding_codebook_id": record["entropy_identities"]["coding_codebook"],
                        "g16_selected_epoch": record["training"]["selected_epoch"],
                        "g16_val_a_bits": record["training"]["best_val_a_bits"],
                        "train_p_frames": record["dataset_identity"]["train_p_frames_collected"],
                        "val_p_frames": record["dataset_identity"]["val_p_frames_collected"],
                        "checkpoint": record,
                    }
                    provenance_note = ("M10K/G16/K=512/M13 rebuilt on TRAIN by the original "
                                       "fitting code; a new residual entropy identity "
                                       "(loaded from the verified checkpoint of an earlier run)")
                elif arm == "refit":
                    print(f"    [{label}] refitting the downstream stack ...", flush=True)
                    refit = m22.refit_downstream(
                        mc, mk, ma, ml, mt, m13, cx, ce, ev, model, bits=bits,
                        residual_params=params,
                        intra_params=deployed_rig["intra_params"],
                        intra_entropy_model=deployed_rig["intra_entropy_model"],
                        motion_entropy_model=deployed_rig["motion_entropy_model"],
                        residual_stack=residual_stack,
                        train_sequences=train_full, val_sequences=refit_val, device=device,
                        gop_size=args.gop, block_size=args.block_size,
                        search_range=args.search_range, train_frames=args.train_frames,
                        val_frames_per_sequence=args.refit_val_frames_per_sequence,
                        seed=args.seed, m10k_epochs=args.m10k_epochs,
                        g16_epochs=args.g16_epochs, codebook_size=args.codebook_size,
                        codebook_rows=args.codebook_fit_rows,
                        log=lambda m: print(m, flush=True))
                    identity = m22.residual_identity_for(
                        ma, refit["model11"], m10k_identity_hex=refit["m10k_identity"],
                        signature=signature, bits=bits,
                        coding_codebook=refit["coding_codebook"])
                    spec = {"model": refit["model11"], "zero": refit["zero"],
                            "assign_codebook": refit["assign_codebook"],
                            "coding_codebook": refit["coding_codebook"],
                            "identity": bytes.fromhex(identity)}
                    refit_info = {k: v for k, v in refit.items()
                                  if k not in ("model11", "assign_codebook", "coding_codebook",
                                               "zero", "g16_history")}
                    refit_info["residual_identity"] = identity
                    refit_info["assign_codebook_id"] = refit[
                        "assign_codebook"].codebook_id().hex()
                    refit_info["coding_codebook_id"] = refit[
                        "coding_codebook"].codebook_id().hex()
                    refit_info["checkpoint"] = m22.save_refit_checkpoint(
                        args.output_dir / "checkpoints"
                        / f"m22_{variant.name}_{bits}bit.pt", refit,
                        variant_name=variant.name, bits=bits, residual_params=params,
                        signature=signature, residual_identity=identity, seed=args.seed,
                        m10k_epochs=args.m10k_epochs, g16_epochs=args.g16_epochs,
                        codebook_size=args.codebook_size, train_sequences=train_full,
                        val_sequences=refit_val)
                    provenance_note = ("M10K/G16/K=512/M13 rebuilt on TRAIN by the original "
                                       "fitting code; a new residual entropy identity")

                run = m21.run_candidate(
                    mc, m13, model, val_b, spec, m21.IDENTITY, stream_dir,
                    intra_params=deployed_rig["intra_params"],
                    intra_entropy_model=deployed_rig["intra_entropy_model"],
                    residual_params=params,
                    motion_entropy_model=deployed_rig["motion_entropy_model"], bits=bits,
                    gop_size=args.gop, block_size=args.block_size,
                    search_range=args.search_range)
                # NOT `path`: that name belongs to the report file `persist()`
                # closes over, and rebinding it here silently redirected every
                # report write into a just-deleted stream file.
                for stream_path in stream_dir.glob(f"identity_{bits}bit_*.nvct"):
                    stream_path.unlink()
                summary = run["aggregate"]
                if is_control and baseline_summary is None:
                    baseline_summary = summary
                delta = compare(summary, baseline_summary) if baseline_summary else {}
                guard = (m22.distortion_regression(delta.get("delta_psnr_db", 0.0),
                                                   delta.get("delta_msssim", 0.0))
                         if delta else None)
                rows.append({
                    "variant": variant.name, "arm": arm, "label": label,
                    "definition": variant.definition, "is_control": is_control,
                    "grid_signature": m22.grid_signature(params),
                    "calibration_signature": signature,
                    "train_grid_statistics": train_stats,
                    "aggregate": summary, "per_sequence": run["per_sequence"],
                    "gop_position": run["gop_position"],
                    "decode_checks": run["decode_checks"],
                    "decoder_compatible": all(run["decode_checks"][k] for k in (
                        "symbols_exact", "reconstruction_exact", "references_exact")),
                    "provenance_note": provenance_note, "refit": refit_info,
                    "seconds": time.perf_counter() - run_started, **delta,
                    "distortion_regression": guard,
                    "verdict": m22.verdict(delta.get("total_stream_gain_percent", 0.0)),
                })
                persist()
                print(f"    [{label}] {rows[-1]['aggregate']['total_container_bytes']:,} bytes "
                      f"({rows[-1].get('total_stream_gain_percent', 0.0):+.4f}% total stream, "
                      f"decode-exact={rows[-1]['decoder_compatible']}) "
                      f"in {rows[-1]['seconds']:.1f}s", flush=True)

        print(f"\n  ================ {bits}-bit  ({time.perf_counter() - started:.1f}s) "
              "================")
        print(f"    {'variant/arm':>24} {'step':>8} {'clip%':>7} {'Hsym':>6} {'total bytes':>12} "
              f"{'d total':>10} {'stream %':>9} {'dPSNR':>8} {'dMS-SSIM':>10} {'dec':>4} "
              f"{'verdict':>9}")
        for row in rows:
            stats = row["train_grid_statistics"]
            print(f"    {row['label']:>24} {stats['step_mean']:>8.4f} "
                  f"{stats['clipping_fraction'] * 100:>7.3f} "
                  f"{stats['symbol_entropy_bits']:>6.3f} "
                  f"{row['aggregate']['total_container_bytes']:>12,} "
                  f"{row.get('delta_total_bytes', 0):>+10,} "
                  f"{row.get('total_stream_gain_percent', 0.0):>+9.4f} "
                  f"{row.get('delta_psnr_db', 0.0):>+8.4f} "
                  f"{row.get('delta_msssim', 0.0):>+10.6f} "
                  f"{str(row['decoder_compatible'])[0]:>4} {row['verdict']:>9}"
                  + ("  [DISTORTION]" if row["distortion_regression"] else ""))
        print(flush=True)
        persist()

    persist()
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
