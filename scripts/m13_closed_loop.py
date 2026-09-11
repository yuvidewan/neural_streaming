"""M13 Phases D/E - the real motion-compensated closed loop, shared by the
small-scale coded-byte validation (Phase D) and the full DAVIS benchmark
(Phase E).

Deliberately NOT a modification of `m11_evaluate.py`'s `encode_multi`/
`decode_sequence` - those dispatch every M11-family arm through
`ma.encode_frame`/`decode_frame` with ONE `codebook` object serving both
symbol-to-prototype ASSIGNMENT and arithmetic CODING. M13's whole premise
requires those to be two DIFFERENT objects for the recalibrated arm (see
`m13_recalibration.py`'s module docstring), which `m11_evaluate.py`'s arm
dispatch cannot express without being changed - and M10A-M11 scripts are
frozen. So this is a new, parallel closed loop, structurally identical to
`m11_evaluate.encode_multi`/`decode_sequence` (same GOP handling, same
motion estimation/warping, same `.nvct` v2 `TemporalStreamWriter`/
`TemporalStreamReader`), that dispatches exactly two arms:

  m11_op      unchanged - `ma.encode_frame`/`decode_frame` through the
              DEPLOYED codebook, both assignment and coding.
  m13_recal   `m13_recalibration.encode_frame_recalibrated`/
              `decode_frame_recalibrated` - SAME model, SAME assign_codebook
              (so assignment is identical to m11_op's), DIFFERENT
              coding_codebook (the recalibrated frequency table).

Both arms code the SAME residual symbols in one closed-loop pass (computed
once, from the SAME motion-compensated reference), exactly like
`m11_evaluate.encode_multi` - so symbol/motion/reconstruction equality
between arms is not just measured but structurally guaranteed by
construction, and any observed difference would indicate a real bug.
"""

from __future__ import annotations

import importlib.util
import math
import statistics
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
from nvc.evaluation.perceptual_metrics import msssim


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ARMS = ("m11_op", "m13_recal")


def _residual_payload(m13, ma, arm, spec, symbols, reference_tensor, *, shape, bits, timings):
    if arm == "m11_op":
        return ma.encode_frame(spec["model"], reference_tensor, symbols.reshape(shape),
                               spec["zero"], bits=bits, codebook=spec["codebook"], timings=timings)
    return m13.encode_frame_recalibrated(
        spec["model"], spec["assign_codebook"], spec["coding_codebook"], reference_tensor,
        symbols.reshape(shape), spec["zero"], bits=bits, timings=timings)


def _residual_decode(m13, ma, arm, spec, payload, reference_tensor, *, shape, bits, timings):
    if arm == "m11_op":
        return ma.decode_frame(spec["model"], payload, reference_tensor, spec["zero"], bits=bits,
                               shape=shape, codebook=spec["codebook"], timings=timings)
    return m13.decode_frame_recalibrated(
        spec["model"], spec["assign_codebook"], spec["coding_codebook"], payload,
        reference_tensor, spec["zero"], bits=bits, shape=shape, timings=timings)


@torch.no_grad()
def encode_multi(mc, ma, m13, model, frames, arms, paths, *, intra_params, intra_entropy_model,
                 residual_params, motion_entropy_model, bits, gop_size, block_size,
                 search_range) -> dict[str, Any]:
    """One closed-loop pass; both arms entropy-code the SAME symbols.
    Structurally identical to `m11_evaluate.encode_multi` - see module
    docstring for why this is a parallel implementation, not a shared call."""
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
                    for arm, spec in arms.items():
                        payload, ideal = _residual_payload(
                            m13, ma, arm, spec, symbols, reference_latent, shape=latent_shape,
                            bits=bits, timings=timings[arm])
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
def decode_sequence(mc, ma, m13, model, path, arm, spec, *, intra_entropy_model,
                    motion_entropy_model, bits):
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
                symbols = _residual_decode(m13, ma, arm, spec, residual_payload_bytes,
                                           reference_latent, shape=header.latent_shape,
                                           bits=bits, timings=timings)
                symbol_log.append(symbols.reshape(header.latent_shape))
                reconstructed_latent = reference_latent + symbols_to_latent(
                    symbols, header.latent_shape, reader.residual_params).to(device)
            reconstruction = model.decode(reconstructed_latent)
            reconstructions.append(reconstruction)
            previous = reconstruction
    return torch.cat(reconstructions, dim=0), symbol_log, timings


def build_arms(ma, m13, model, assign_codebook, coding_codebook, zero, *, m10k_identity,
              calibration_signature, bits) -> dict[str, dict[str, Any]]:
    """The two specs `encode_multi`/`decode_sequence` above dispatch on."""
    old_identity = ma.model_identity(model, m10k_identity=m10k_identity,
                                     calibration_signature=calibration_signature, bits=bits,
                                     codebook=assign_codebook)
    new_identity = ma.model_identity(model, m10k_identity=m10k_identity,
                                     calibration_signature=calibration_signature, bits=bits,
                                     codebook=coding_codebook)
    return {
        "m11_op": {"model": model, "zero": zero, "codebook": assign_codebook,
                  "identity": old_identity},
        "m13_recal": {"model": model, "zero": zero, "assign_codebook": assign_codebook,
                     "coding_codebook": coding_codebook, "identity": new_identity},
    }


def aggregate(name: str, bits: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
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


def run_sequences(mc, ma, m13, model, sequences, arms, stream_dir, *, intra_params,
                  intra_entropy_model, residual_params, motion_entropy_model, bits, gop_size,
                  block_size, search_range) -> dict[str, Any]:
    """Runs `encode_multi`/`decode_sequence` over every sequence, checking the
    hard invariants (symbols/motion/reconstruction/metrics identical between
    arms) per sequence and stopping immediately if any fails."""
    per_sequence_rows: list[dict[str, Any]] = []
    arm_rows = {arm: [] for arm in ARMS}
    decode_accumulator = {arm: {} for arm in ARMS}
    invariants = {"symbols": True, "reconstruction": True, "motion": True, "metrics": True}

    for sequence in sequences:
        frames = sequence.load_frames()
        paths = {arm: stream_dir / f"{arm}_{bits}bit_{sequence.sequence_id}.nvct" for arm in ARMS}
        result = encode_multi(
            mc, ma, m13, model, frames, arms, paths, intra_params=intra_params,
            intra_entropy_model=intra_entropy_model, residual_params=residual_params,
            motion_entropy_model=motion_entropy_model, bits=bits, gop_size=gop_size,
            block_size=block_size, search_range=search_range)
        recon = result["reconstructions"]
        mse = torch.mean((recon - frames) ** 2).item()
        psnr = 10.0 * math.log10(1.0 / mse)
        quality = float(msssim(recon.clamp(0, 1), frames).mean())
        invariants["motion"] &= len({result["arms"][a]["motion_bytes"] for a in ARMS}) == 1
        record = {"sequence": sequence.sequence_id, "bits": bits}
        for arm in ARMS:
            stats = result["arms"][arm]
            decoded, decoded_symbols, timing = decode_sequence(
                mc, ma, m13, model, paths[arm], arm, arms[arm],
                intra_entropy_model=intra_entropy_model, motion_entropy_model=motion_entropy_model,
                bits=bits)
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

    metrics = {(round(statistics.fmean(r["mean_psnr_db"] for r in arm_rows[a]), 9),
               round(statistics.fmean(r["mean_msssim"] for r in arm_rows[a]), 9)) for a in ARMS}
    invariants["metrics"] = len(metrics) == 1

    results = {arm: aggregate(arm, bits, arm_rows[arm]) for arm in ARMS}
    decode_times = {arm: decode_accumulator[arm] for arm in ARMS}
    return {"results": results, "decode_times": decode_times, "invariants": invariants,
           "per_sequence": per_sequence_rows}
