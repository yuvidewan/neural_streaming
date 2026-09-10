"""M11 - channel-autoregressive entropy coding on the DAVIS test set.

Six arms code the SAME residual symbols of the SAME M10H motion-compensated
stream, in one closed loop, so nothing but the residual entropy model differs:

    marginal          M10H, one table per latent channel
    local_activity4   M10J, hand-designed 4-bucket context
    learned           M10K, one learned table per position, z_ref-conditioned
    codebook          M10L, 512 shared prototypes of M10K's prediction
    m11               M11, G = 1: z_ref + every earlier channel, 64 decode steps,
                      one table per position - the compression ceiling
    m11_op            M11 at the operating point picked on VAL-A, through its own
                      512-entry codebook - the practical configuration

Because the arms share one loop, reconstruction, motion and PSNR/MS-SSIM are
identical by construction; that is asserted per rate point and the run stops if
it ever fails. Every stream is decoded back from disk with only the stream, the
calibration and the arm's model, and the M11 arms decode through the
group-sequential decoder, so encode AND decode time are both measured.

PROVENANCE IS ENFORCED, NOT JUST RECORDED
-----------------------------------------
Each M11 checkpoint stores the calibration signature it was trained under. This
script recomputes the calibration - bit-reproducible across processes since the
M11 phase 0 fix - and refuses to run if a signature differs. A stale model is a
stop, never a silent rate loss.

The TRAIN symbols behind the M10H/M10J tables come from the same cache the
offline gate and training used (identical frames and settings to M10L's
benchmark, and byte-identical to a fresh collection). Nothing in this script
fits anything to test data.

Run (after scripts/m11_offline_gate.py and scripts/m11_train.py):
  ./.venv/Scripts/python.exe scripts/m11_evaluate.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.codec import (
    decode_payload_to_latent,
    encode_latent_to_payload,
    latent_to_symbols,
    symbols_to_latent,
)
from nvc.compression.entropy_model import TOTAL_FREQUENCY
from nvc.compression.range_coder import decode_symbols, encode_symbols
from nvc.evaluation.perceptual_metrics import msssim
from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device

DEFAULT_OUTPUT_DIR = Path("outputs/m11_autoregressive_entropy")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
DEFAULT_M10L_DIR = Path("outputs/m10l_shared_codebook/codebooks")
FROZEN_LAMBDA = 3.0e-4
RATE_POINTS = (5, 4, 3)
ARMS = ("marginal", "local_activity4", "learned", "codebook", "m11", "m11_op")
M11_ARMS = ("m11", "m11_op")
WATCH = ("bmx-bumps", "drone", "cat-girl", "drift-chicane", "gold-fish", "schoolgirls")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProvenanceError(RuntimeError):
    """A model or codebook does not match the calibration it is about to be used with."""


def check_provenance(checkpoint: dict, *, signature: str, bits: int, group_size: int,
                     context_definition_id: str, m10k_identity: bytes) -> None:
    """Refuse a model trained under different conditions. No fallback."""
    problems = []
    if checkpoint.get("calibration_signature") != signature:
        problems.append(f"calibration {checkpoint.get('calibration_signature')} != {signature}")
    if checkpoint.get("bits") != bits:
        problems.append(f"bit depth {checkpoint.get('bits')} != {bits}")
    if checkpoint.get("model_config", {}).get("group_size") != group_size:
        problems.append(f"group size {checkpoint.get('model_config', {}).get('group_size')} "
                        f"!= {group_size}")
    if checkpoint.get("context_definition_id") != context_definition_id:
        problems.append("context definition differs")
    if checkpoint.get("m10k_identity") != bytes(m10k_identity).hex():
        problems.append("warm-started from a different M10K model")
    if problems:
        raise ProvenanceError("M11 model provenance mismatch: " + "; ".join(problems))


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M11: channel-autoregressive entropy coding on the DAVIS test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m10k-dir", type=Path, default=DEFAULT_M10K_DIR)
    parser.add_argument("--m10l-dir", type=Path, default=DEFAULT_M10L_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--training-report", type=Path, default=None)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--train-frames", type=int, default=600)
    parser.add_argument("--val-frames-per-sequence", type=int, default=40)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


# --- coding one frame under one arm ---------------------------------------------------


def residual_payload(arm, spec, symbols, reference_numpy, reference_tensor, *, shape, bits,
                     mk, ml, ma, timings):
    """Encode one frame's residual symbols under one arm; returns (payload, ideal bits)."""
    flat = symbols.reshape(-1)
    if arm in M11_ARMS:
        payload, ideal = ma.encode_frame(spec["model"], reference_tensor,
                                         symbols.reshape(shape), spec["zero"], bits=bits,
                                         codebook=spec.get("codebook"), timings=timings)
        return payload, ideal
    started = time.perf_counter()
    if arm == "learned":
        entropy_model, table = mk.frame_entropy_model(spec["model"], reference_tensor, bits=bits)
        cumulative, frequencies = entropy_model.cumulative, entropy_model.frequencies
    elif arm == "codebook":
        table = ml.frame_table_index(spec["model"], spec["codebook"], reference_tensor)
        cumulative, frequencies = spec["codebook"].cumulative, spec["codebook"].frequencies
    else:
        contexts = spec["context_model"].contexts(reference_numpy)
        table = spec["context_model"].table_index(*shape, contexts)
        cumulative = spec["entropy_model"].cumulative
        frequencies = spec["entropy_model"].frequencies
    timings["tables"] = timings.get("tables", 0.0) + time.perf_counter() - started
    started = time.perf_counter()
    payload = encode_symbols(flat, cumulative, table)
    timings["coder"] = timings.get("coder", 0.0) + time.perf_counter() - started
    probabilities = frequencies.astype(np.float64) / TOTAL_FREQUENCY
    return payload, float(-np.log2(probabilities[table, flat]).sum())


def residual_decode(arm, spec, payload, reference_tensor, *, shape, bits, mk, ml, ma, timings):
    if arm in M11_ARMS:
        return ma.decode_frame(spec["model"], payload, reference_tensor, spec["zero"],
                               bits=bits, shape=shape, codebook=spec.get("codebook"),
                               timings=timings)
    channels, height, width = shape
    started = time.perf_counter()
    if arm == "learned":
        entropy_model, table = mk.frame_entropy_model(spec["model"], reference_tensor, bits=bits)
        cumulative = entropy_model.cumulative
    elif arm == "codebook":
        table = ml.frame_table_index(spec["model"], spec["codebook"], reference_tensor)
        cumulative = spec["codebook"].cumulative
    else:
        contexts = spec["context_model"].contexts(reference_tensor[0].cpu().numpy())
        cumulative = spec["entropy_model"].cumulative
        table = spec["context_model"].table_index(channels, height, width, contexts)
    timings["tables"] = timings.get("tables", 0.0) + time.perf_counter() - started
    started = time.perf_counter()
    symbols = decode_symbols(payload, channels * height * width, cumulative, table)
    timings["coder"] = timings.get("coder", 0.0) + time.perf_counter() - started
    return symbols


# --- the closed loop ------------------------------------------------------------------


@torch.no_grad()
def encode_multi(mc, mk, ml, ma, model, frames, arms, paths, *, intra_params,
                 intra_entropy_model, residual_params, motion_entropy_model, bits, gop_size,
                 block_size, search_range) -> dict[str, Any]:
    """One closed-loop pass; every arm entropy-codes the SAME symbols."""
    model.eval()
    device = next(model.parameters()).device
    frame_count = frames.shape[0]
    types = mc.gop_frame_types(frame_count, gop_size)
    latent_shape = tuple(model.encode(frames[0:1].to(device)).shape[1:])

    writers, records = {}, {arm: [] for arm in arms}
    ideal_bits = {arm: 0.0 for arm in arms}
    timings = {arm: {} for arm in arms}
    for arm, spec in arms.items():
        header = mc.TemporalStreamHeader(
            gop_size=gop_size, quantization_bits=intra_params.bits,
            quantization_mode=intra_params.mode,
            image_width=frames.shape[3], image_height=frames.shape[2],
            image_channels=frames.shape[1], latent_channels=latent_shape[0],
            latent_height=latent_shape[1], latent_width=latent_shape[2],
            frame_count=frame_count,
            num_intra_quantization_params=intra_params.scale.numel(),
            num_residual_quantization_params=residual_params.scale.numel(),
            block_size=block_size, search_range=search_range,
            motion_bits=motion_entropy_model.bits, reference_mode="mc",
            intra_entropy_model_id=intra_entropy_model.model_id(),
            residual_entropy_model_id=spec["identity"],
            motion_entropy_model_id=motion_entropy_model.model_id())
        writers[arm] = mc.TemporalStreamWriter(paths[arm], header, intra_params, residual_params)

    reconstructions, symbol_log, previous = [], [], None
    try:
        with mc.deterministic_kernels():
            for index in range(frame_count):
                frame = frames[index:index + 1].to(device)
                latent = model.encode(frame)
                frame_type = types[index]
                if frame_type == mc.FRAME_TYPE_I:
                    payload, _ = encode_latent_to_payload(
                        latent, params=intra_params, entropy_model=intra_entropy_model)
                    decoded, _ = decode_payload_to_latent(
                        payload, entropy_model=intra_entropy_model, params=intra_params,
                        shape=latent_shape)
                    reconstructed_latent = decoded.to(device)
                    for arm in arms:
                        writers[arm].append_frame(frame_type, b"", payload)
                        records[arm].append({"frame_type": "I", "motion_bytes": 0,
                                             "residual_bytes": len(payload)})
                else:
                    motion = mc.estimate_block_motion(
                        previous, frame, block_size=block_size, search_range=search_range)
                    motion_payload = mc.encode_motion_payload(
                        motion, search_range=search_range, entropy_model=motion_entropy_model)
                    decoded_motion = mc.decode_motion_payload(
                        motion_payload, (frames.shape[2] // block_size,
                                         frames.shape[3] // block_size),
                        search_range=search_range, entropy_model=motion_entropy_model)
                    warped = mc.warp_blocks(previous, decoded_motion, block_size=block_size)
                    reference_latent = model.encode(warped)
                    delta = latent - reference_latent
                    symbols = latent_to_symbols(delta, residual_params)       # ONE quantization
                    symbol_log.append(symbols.reshape(latent_shape))
                    reference_numpy = reference_latent[0].cpu().numpy()
                    for arm, spec in arms.items():
                        payload, ideal = residual_payload(
                            arm, spec, symbols, reference_numpy, reference_latent,
                            shape=latent_shape, bits=bits, mk=mk, ml=ml, ma=ma,
                            timings=timings[arm])
                        ideal_bits[arm] += ideal
                        writers[arm].append_frame(frame_type, motion_payload, payload)
                        records[arm].append({"frame_type": "P",
                                             "motion_bytes": len(motion_payload),
                                             "residual_bytes": len(payload)})
                    reconstructed_latent = reference_latent + symbols_to_latent(
                        symbols, latent_shape, residual_params).to(device)
                reconstruction = model.decode(reconstructed_latent)
                previous = reconstruction
                reconstructions.append(reconstruction.detach().cpu())
    finally:
        for writer in writers.values():
            writer.close()

    result = {"frame_count": frame_count, "symbols": symbol_log,
              "reconstructions": torch.cat(reconstructions, dim=0), "arms": {}}
    for arm in arms:
        rows = records[arm]
        container = paths[arm].stat().st_size
        motion_total = sum(r["motion_bytes"] for r in rows)
        residual_total = sum(r["residual_bytes"] for r in rows)
        result["arms"][arm] = {
            "p_frames": sum(1 for r in rows if r["frame_type"] == "P"),
            "i_frame_residual_bytes": sum(r["residual_bytes"] for r in rows
                                          if r["frame_type"] == "I"),
            "p_frame_residual_bytes": sum(r["residual_bytes"] for r in rows
                                          if r["frame_type"] == "P"),
            "motion_bytes": motion_total, "residual_bytes": residual_total,
            "container_bytes": container,
            "container_overhead_bytes": container - motion_total - residual_total,
            "p_frame_ideal_bits": ideal_bits[arm], "encode_seconds": dict(timings[arm]),
        }
    return result


@torch.no_grad()
def decode_sequence(mc, mk, ml, ma, model, path, arm, spec, *, intra_entropy_model,
                    motion_entropy_model, bits):
    """Decode from the stream, the calibration and the arm's model only."""
    model.eval()
    device = next(model.parameters()).device
    reader = mc.TemporalStreamReader(path)
    header = reader.header
    if spec["identity"] != header.residual_entropy_model_id:
        raise mc.TemporalFormatError(
            f"residual entropy model mismatch: stream declares "
            f"{header.residual_entropy_model_id.hex()}, supplied is {spec['identity'].hex()}")
    reconstructions, symbol_log, previous = [], [], None
    timings: dict[str, float] = {}
    with mc.deterministic_kernels():
        for frame_type, motion_payload, residual_payload_bytes in reader:
            if frame_type == mc.FRAME_TYPE_I:
                latent, _ = decode_payload_to_latent(
                    residual_payload_bytes, entropy_model=intra_entropy_model,
                    params=reader.intra_params, shape=header.latent_shape)
                reconstructed_latent = latent.to(device)
            else:
                motion = mc.decode_motion_payload(
                    motion_payload, header.block_grid, search_range=header.search_range,
                    entropy_model=motion_entropy_model)
                warped = mc.warp_blocks(previous, motion, block_size=header.block_size)
                reference_latent = model.encode(warped)
                symbols = residual_decode(arm, spec, residual_payload_bytes, reference_latent,
                                          shape=header.latent_shape, bits=bits, mk=mk, ml=ml,
                                          ma=ma, timings=timings)
                symbol_log.append(symbols.reshape(header.latent_shape))
                reconstructed_latent = reference_latent + symbols_to_latent(
                    symbols, header.latent_shape, reader.residual_params).to(device)
            reconstruction = model.decode(reconstructed_latent)
            reconstructions.append(reconstruction)
            previous = reconstruction
    return torch.cat(reconstructions, dim=0), symbol_log, timings


def _operating_point(args) -> int:
    path = args.training_report or (args.output_dir / "m11_training.json")
    if not path.is_file():
        raise SystemExit(f"[ERROR] training report not found: {path}. "
                         f"Run scripts/m11_train.py first.")
    return int(json.loads(path.read_text(encoding="utf-8"))["operating_point_group_size"])


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    mc = _load_script("m10h_motion_compensation")
    ce = _load_script("m10j_conditional_entropy")
    mk = _load_script("m10k_learned_entropy")
    ml = _load_script("m10l_shared_codebook")
    ma = _load_script("m11_ar_entropy")
    md = _load_script("m11_data")
    cx = _load_script("m11_causal_context")
    ev = _load_script("m10l_evaluate")
    m10e = _load_script("m10e_evaluate")
    op_group = _operating_point(args)

    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    test_sequences = discover_sequences(
        args.manifest, split="test", max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames_per_sequence)

    print("=" * 136, flush=True)
    print("M11 - CHANNEL-AUTOREGRESSIVE ENTROPY vs M10H/M10J/M10K/M10L (identical symbols, DAVIS test)")
    print("=" * 136)
    print(f"  frozen model : {args.checkpoint}   lambda {FROZEN_LAMBDA:.1e}")
    print(f"  arms         : intra (GOP=1), {', '.join(ARMS)}")
    print(f"  m11          : G=1, per-position tables (64 decode steps)")
    print(f"  m11_op       : G={op_group}, 512-entry codebook ({64 // op_group} decode steps) "
          f"- chosen on VAL-A")
    print(f"  sequences    : {len(test_sequences)}  frames: "
          f"{sum(s.frame_count for s in test_sequences)}")

    stream_dir = args.output_dir / "benchmark_streams"
    stream_dir.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[str, int], dict[str, Any]] = {}
    per_sequence_rows: list[dict[str, Any]] = []
    decode_times: dict[tuple[str, int], dict[str, float]] = {}
    provenance: dict[str, Any] = {}

    for bits in args.rate_points:
        print(f"\n  preparing {bits}-bit ...", flush=True)
        data = md.load_or_collect(
            model, checkpoint=args.checkpoint, manifest=args.manifest, bits=bits,
            device=device, calibration_frames=args.calibration_frames,
            train_frames=args.train_frames,
            val_frames_per_sequence=args.val_frames_per_sequence,
            cache_dir=args.cache_dir or md.DEFAULT_CACHE_DIR,
            log=lambda m: print(m, flush=True))
        cached_calibration = data["calibration"]
        calibration = mc.calibrate_grids(
            model, discover_sequences(args.manifest, split="train"), bits=bits,
            mode="per_channel", gop_size=args.gop, block_size=args.block_size,
            search_range=args.search_range, reference_mode="mc",
            max_frames=args.calibration_frames)
        signature = ev.calibration_signature(calibration, bits=bits,
                                             calibration_frames=args.calibration_frames,
                                             quant_mode="per_channel")
        cached_signature = ev.calibration_signature(
            cached_calibration, bits=bits, calibration_frames=args.calibration_frames,
            quant_mode="per_channel")
        if signature != cached_signature:
            print(f"[ERROR] fresh calibration {signature} differs from the cached one "
                  f"{cached_signature} - the determinism fix has regressed. STOPPING.",
                  file=sys.stderr)
            return 1
        train_symbols = list(data["train_symbols"].astype(np.int64))
        train_references = list(data["train_references"])
        channels = train_symbols[0].shape[0]
        zero = torch.from_numpy(cx.zero_symbols(calibration["residual_params"], channels))

        arms: dict[str, dict[str, Any]] = {}
        for scheme in ("marginal", "local_activity4"):
            context_model = ce.fit_context_model(scheme, train_references)
            built = ce.build_conditional_entropy_model(train_symbols, train_references,
                                                       context_model, bits=bits)
            arms[scheme] = {"context_model": context_model,
                            "entropy_model": built["entropy_model"],
                            "identity": built["entropy_model"].model_id()}
        m10k, m10k_checkpoint = mk.load_entropy_model(
            args.m10k_dir / f"learned_entropy_{bits}bit.pt", device=device)
        m10k_identity = ev._m10k_model_identity(m10k_checkpoint["model_state_dict"])
        arms["learned"] = {"model": m10k, "identity": m10k_identity}
        m10l_codebook = ml.SharedCodebook.from_dict(json.loads(
            (args.m10l_dir / f"codebook_{bits}bit_K512_code_length.json").read_text(
                encoding="utf-8")))
        arms["codebook"] = {"model": m10k, "codebook": m10l_codebook,
                            "identity": m10l_codebook.codebook_id(
                                model_identity=m10k_identity, calibration_signature=signature)}

        for arm, group, use_codebook in (("m11", 1, False), ("m11_op", op_group, True)):
            path = args.output_dir / f"m11_G{group}_entropy_{bits}bit.pt"
            m11, checkpoint = ma.load_model(path, device=device)
            try:
                check_provenance(checkpoint, signature=signature, bits=bits, group_size=group,
                                 context_definition_id=ma.context_definition_id(group),
                                 m10k_identity=m10k_identity)
            except ProvenanceError as error:
                print(f"[ERROR] {arm}: {error}. STOPPING.", file=sys.stderr)
                return 1
            codebook = None
            if use_codebook:
                codebook = ml.SharedCodebook.from_dict(json.loads(
                    (args.output_dir / f"m11_G{group}_codebook_{bits}bit_K512.json").read_text(
                        encoding="utf-8")))
            arms[arm] = {"model": m11, "zero": zero, "codebook": codebook,
                         "identity": ma.model_identity(
                             m11, m10k_identity=m10k_identity, calibration_signature=signature,
                             bits=bits, codebook=codebook),
                         "group_size": group,
                         "parameters": sum(p.numel() for p in m11.parameters()),
                         "checkpoint_bytes": path.stat().st_size}
            provenance[f"{bits}bit_{arm}"] = {
                "group_size": group, "codebook": use_codebook,
                "selected_epoch": checkpoint["selected_epoch"],
                "identity": arms[arm]["identity"].hex(), "calibration_signature": signature,
                "m10k_identity": m10k_identity.hex(), "split": checkpoint["split"]}

        intra_rows = []
        for sequence in test_sequences:
            frames = sequence.load_frames()
            path = stream_dir / f"intra_{bits}bit_{sequence.sequence_id}.nvct"
            encoded = mc.encode_sequence(
                model, frames, path, intra_params=calibration["intra_params"],
                intra_entropy_model=calibration["intra_entropy_model"],
                residual_params=calibration["residual_params"],
                residual_entropy_model=calibration["residual_entropy_model"],
                motion_entropy_model=calibration["motion_entropy_model"],
                mode="prev", gop_size=1, block_size=args.block_size,
                search_range=args.search_range)
            recon = encoded["encoder_reconstructions"]
            mse = torch.mean((recon - frames) ** 2).item()
            intra_rows.append({
                "sequence": sequence.sequence_id, "motion_bytes": 0,
                "residual_bytes": encoded["residual_bytes"],
                "i_frame_residual_bytes": encoded["i_frame_residual_bytes"],
                "p_frame_residual_bytes": 0, "p_frame_ideal_bits": 0.0,
                "container_overhead_bytes": encoded["container_overhead_bytes"],
                "container_bytes": encoded["container_bytes"],
                "total_pixels": sequence.total_pixels, "raw_rgb_bytes": sequence.raw_rgb_bytes(),
                "mean_psnr_db": 10.0 * math.log10(1.0 / mse),
                "mean_msssim": float(msssim(recon.clamp(0, 1), frames).mean()),
                "p_frames": 0, "encode_seconds": {}})

        arm_rows = {arm: [] for arm in ARMS}
        invariants = {"symbols": True, "reconstruction": True, "motion": True, "metrics": True}
        decode_accumulator = {arm: {} for arm in ARMS}
        for sequence in test_sequences:
            frames = sequence.load_frames()
            paths = {arm: stream_dir / f"{arm}_{bits}bit_{sequence.sequence_id}.nvct"
                     for arm in ARMS}
            result = encode_multi(
                mc, mk, ml, ma, model, frames, arms, paths,
                intra_params=calibration["intra_params"],
                intra_entropy_model=calibration["intra_entropy_model"],
                residual_params=calibration["residual_params"],
                motion_entropy_model=calibration["motion_entropy_model"], bits=bits,
                gop_size=args.gop, block_size=args.block_size, search_range=args.search_range)
            recon = result["reconstructions"]
            mse = torch.mean((recon - frames) ** 2).item()
            psnr = 10.0 * math.log10(1.0 / mse)
            quality = float(msssim(recon.clamp(0, 1), frames).mean())
            invariants["motion"] &= len({result["arms"][a]["motion_bytes"] for a in ARMS}) == 1
            record = {"sequence": sequence.sequence_id, "bits": bits,
                      "watched": sequence.sequence_id in WATCH}
            for arm in ARMS:
                stats = result["arms"][arm]
                decoded, decoded_symbols, timing = decode_sequence(
                    mc, mk, ml, ma, model, paths[arm], arm, arms[arm],
                    intra_entropy_model=calibration["intra_entropy_model"],
                    motion_entropy_model=calibration["motion_entropy_model"], bits=bits)
                for key, value in timing.items():
                    decode_accumulator[arm][key] = decode_accumulator[arm].get(key, 0.0) + value
                invariants["symbols"] &= all(
                    np.array_equal(a.reshape(-1), b.reshape(-1))
                    for a, b in zip(result["symbols"], decoded_symbols))
                invariants["reconstruction"] &= torch.equal(decoded.cpu(), recon)
                arm_rows[arm].append({"sequence": sequence.sequence_id, **stats,
                                      "total_pixels": sequence.total_pixels,
                                      "raw_rgb_bytes": sequence.raw_rgb_bytes(),
                                      "mean_psnr_db": psnr, "mean_msssim": quality})
                record[f"{arm}_residual_bytes"] = stats["residual_bytes"]
            per_sequence_rows.append(record)

        def aggregate(name, rows):
            total = sum(r["container_bytes"] for r in rows)
            pixels = sum(r["total_pixels"] for r in rows)
            encode: dict[str, float] = {}
            for r in rows:
                for key, value in r["encode_seconds"].items():
                    encode[key] = encode.get(key, 0.0) + value
            return {
                "arm": name, "bits": bits,
                "total_motion_bytes": sum(r["motion_bytes"] for r in rows),
                "total_residual_bytes": sum(r["residual_bytes"] for r in rows),
                "total_i_frame_residual_bytes": sum(r["i_frame_residual_bytes"] for r in rows),
                "total_p_frame_residual_bytes": sum(r["p_frame_residual_bytes"] for r in rows),
                "total_container_overhead_bytes": sum(r["container_overhead_bytes"] for r in rows),
                "total_container_bytes": total, "total_pixels": pixels,
                "stream_bpp": total * 8 / pixels,
                "compression_ratio": sum(r["raw_rgb_bytes"] for r in rows) / total,
                "mean_psnr_db": statistics.fmean(r["mean_psnr_db"] for r in rows),
                "mean_msssim": statistics.fmean(r["mean_msssim"] for r in rows),
                "p_frame_ideal_bits": sum(r["p_frame_ideal_bits"] for r in rows),
                "p_frames": sum(r["p_frames"] for r in rows),
                "encode_seconds": encode,
                "byte_accounting_closes": sum(
                    r["motion_bytes"] + r["residual_bytes"] + r["container_overhead_bytes"]
                    for r in rows) == total,
            }

        results[("intra", bits)] = aggregate("intra", intra_rows)
        for arm in ARMS:
            results[(arm, bits)] = aggregate(arm, arm_rows[arm])
            decode_times[(arm, bits)] = decode_accumulator[arm]
        metrics = {(round(results[(a, bits)]["mean_psnr_db"], 9),
                    round(results[(a, bits)]["mean_msssim"], 9)) for a in ARMS}
        invariants["metrics"] = len(metrics) == 1
        print(f"    invariants - symbols {invariants['symbols']} | reconstruction "
              f"{invariants['reconstruction']} | motion {invariants['motion']} | "
              f"PSNR/MS-SSIM {invariants['metrics']}", flush=True)
        if not all(invariants.values()):
            print("[ERROR] the arms diverged; rate results are not interpretable. STOPPING.",
                  file=sys.stderr)
            return 1

    # --- report ---------------------------------------------------------------------------
    print()
    print("=" * 136)
    print("FULL BYTE ACCOUNTING")
    print("=" * 136)
    print(f"{'arm':<17} {'bits':>5} {'I bytes':>11} {'P resid':>11} {'motion':>9} "
          f"{'overhead':>9} {'TOTAL':>11} {'BPP':>7} {'PSNR':>7} {'MS-SSIM':>7} "
          f"{'vs M10L':>8} {'vs M10K':>8}")
    for bits in args.rate_points:
        m10l = results[("codebook", bits)]["total_residual_bytes"]
        m10k = results[("learned", bits)]["total_residual_bytes"]
        for arm in ["intra"] + list(ARMS):
            a = results[(arm, bits)]
            vs_l = "" if arm in ("intra", "codebook") else \
                f"{(a['total_residual_bytes'] - m10l) / m10l * 100:+7.2f}%"
            vs_k = "" if arm not in M11_ARMS else \
                f"{(a['total_residual_bytes'] - m10k) / m10k * 100:+7.2f}%"
            print(f"{arm:<17} {bits:>5} {a['total_i_frame_residual_bytes']:>11,} "
                  f"{a['total_p_frame_residual_bytes']:>11,} {a['total_motion_bytes']:>9,} "
                  f"{a['total_container_overhead_bytes']:>9,} {a['total_container_bytes']:>11,} "
                  f"{a['stream_bpp']:>7.4f} {a['mean_psnr_db']:>7.3f} {a['mean_msssim']:>7.4f} "
                  f"{vs_l:>8} {vs_k:>8}")

    print()
    print("=" * 136)
    print("IDEAL BITS vs EMITTED BYTES (P-frame residuals)")
    print("=" * 136)
    theory_rows = []
    for bits in args.rate_points:
        base = results[("codebook", bits)]
        for arm in ARMS:
            a = results[(arm, bits)]
            ideal_gain = (base["p_frame_ideal_bits"] - a["p_frame_ideal_bits"]) \
                / base["p_frame_ideal_bits"] * 100
            byte_gain = (base["total_p_frame_residual_bytes"] - a["total_p_frame_residual_bytes"]) \
                / base["total_p_frame_residual_bytes"] * 100
            overhead = (a["total_p_frame_residual_bytes"] * 8 - a["p_frame_ideal_bits"]) \
                / a["p_frame_ideal_bits"] * 100
            realised = byte_gain / ideal_gain * 100 if abs(ideal_gain) > 1e-9 else float("nan")
            theory_rows.append({"bits": bits, "arm": arm, "ideal_bits": a["p_frame_ideal_bits"],
                                "ideal_gain_vs_m10l_percent": ideal_gain,
                                "byte_gain_vs_m10l_percent": byte_gain,
                                "coder_overhead_percent": overhead,
                                "realised_percent": realised})
            print(f"{bits:>5} {arm:<17} ideal {a['p_frame_ideal_bits']:>14,.0f}  "
                  f"vs M10L {ideal_gain:>+7.2f}%   bytes {a['total_p_frame_residual_bytes']:>11,}  "
                  f"vs M10L {byte_gain:>+7.2f}%   coder overhead {overhead:>+6.3f}%   realised "
                  f"{(f'{realised:.0f}%' if realised == realised else 'n/a'):>6}")

    print()
    print("=" * 136)
    print("BD-RATE over three rate points (piecewise linear, no extrapolation)")
    print("=" * 136)

    def curve(arm, metric="mean_psnr_db"):
        return [(results[(arm, b)]["stream_bpp"], results[(arm, b)][metric])
                for b in args.rate_points]

    bd_rows = []
    for test, base in (("m11", "codebook"), ("m11_op", "codebook"), ("m11", "learned"),
                       ("m11_op", "learned"), ("codebook", "local_activity4"),
                       ("m11", "marginal"), ("m11", "intra"), ("m11_op", "intra"),
                       ("codebook", "intra")):
        psnr_bd = m10e._bd_rate_linear(curve(base), curve(test))
        ms_bd = m10e._bd_rate_linear(curve(base, "mean_msssim"), curve(test, "mean_msssim"))
        bd_rows.append({"test": test, "base": base, "bd_rate_psnr": psnr_bd,
                        "bd_rate_msssim": ms_bd})
        print(f"  {test + ' vs ' + base:<34} PSNR "
              f"{(f'{psnr_bd:+.2f}%' if psnr_bd is not None else 'n/a'):>9}   MS-SSIM "
              f"{(f'{ms_bd:+.2f}%' if ms_bd is not None else 'n/a'):>9}")

    print()
    print("=" * 136)
    print("LATENCY per P-frame - encode and decode, split by stage (ms)")
    print("=" * 136)
    latency_rows = []
    for bits in args.rate_points:
        for arm in ARMS:
            a = results[(arm, bits)]
            frames_p = max(a["p_frames"], 1)
            encode = {k: v / frames_p * 1000 for k, v in a["encode_seconds"].items()}
            decode = {k: v / frames_p * 1000 for k, v in decode_times[(arm, bits)].items()}
            row = {"bits": bits, "arm": arm, "encode_ms": encode, "decode_ms": decode,
                   "encode_total_ms": sum(encode.values()),
                   "decode_total_ms": sum(decode.values())}
            latency_rows.append(row)
            print(f"{bits:>5} {arm:<17} encode {row['encode_total_ms']:>7.2f}  "
                  + " ".join(f"{k} {v:.2f}" for k, v in sorted(encode.items()))
                  + f"   | decode {row['decode_total_ms']:>7.2f}  "
                  + " ".join(f"{k} {v:.2f}" for k, v in sorted(decode.items())))

    reference_bits = args.rate_points[1] if len(args.rate_points) > 1 else args.rate_points[0]
    print()
    print("=" * 136)
    print(f"PER-SEQUENCE residual bytes at {reference_bits}-bit (* = motion-sensitive)")
    print("=" * 136)
    for record in [r for r in per_sequence_rows if r["bits"] == reference_bits]:
        base = record["codebook_residual_bytes"]
        record["m11_vs_m10l_percent"] = (record["m11_residual_bytes"] - base) / base * 100
        record["m11_op_vs_m10l_percent"] = (record["m11_op_residual_bytes"] - base) / base * 100
        print(f"{'*' if record['watched'] else ' '}{record['sequence']:<17} M10L {base:>9,}  "
              f"M11 {record['m11_residual_bytes']:>9,} ({record['m11_vs_m10l_percent']:+6.2f}%)  "
              f"M11-op {record['m11_op_residual_bytes']:>9,} "
              f"({record['m11_op_vs_m10l_percent']:+6.2f}%)")

    closes = all(results[k]["byte_accounting_closes"] for k in results)
    print(f"\n  byte accounting closes everywhere: {closes}")
    report = {
        "phase": "M11 channel-autoregressive entropy benchmark",
        "frozen_lambda": FROZEN_LAMBDA, "checkpoint": str(args.checkpoint),
        "rate_points_bits": list(args.rate_points), "operating_point_group_size": op_group,
        "arms": {f"{k[0]}@{k[1]}bit": v for k, v in results.items()},
        "per_sequence": per_sequence_rows, "ideal_vs_deployed": theory_rows,
        "bd_rate": bd_rows, "latency": latency_rows, "provenance": provenance,
        "invariants": {"symbols_identical": True, "reconstruction_identical": True,
                       "motion_identical": True, "metrics_identical": True,
                       "byte_accounting_closes": closes},
    }
    path = args.output_dir / "m11_benchmark.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
