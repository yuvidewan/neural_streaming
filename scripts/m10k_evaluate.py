"""M10K evaluation: does the learned entropy model's gain survive the real coder?

The offline gate measured a held-out cross-entropy improvement over M10J of
+1.19 / +1.44 / +1.81% at 5 / 4 / 3-bit. This script asks the only question that
settles it: how much of that becomes actual arithmetic-coded bytes.

FOUR ARMS, ONE CLOSED LOOP
----------------------------
    intra            the same coder at GOP=1
    marginal         M10H, per-channel tables
    local_activity4  M10J, per-(channel, 4-bucket) tables
    learned          M10K, a distribution predicted per position from z_ref

Motion, warp, z_ref, the residual and its quantized symbols are computed ONCE
per sequence and shared. Each arm only re-codes those same symbols with its own
probability model, so identical symbols / motion / reconstruction is a property
of the code path rather than something checked afterwards. PSNR and MS-SSIM must
come out identical across the three temporal arms; if they do not, the
experiment is invalid and the script says so.

DEPLOYMENT WITHOUT A FORMAT CHANGE
------------------------------------
The learned model's per-position distributions are handed to the EXISTING
arithmetic coder as 16,384 frequency tables with `table_index = arange`. The
`.nvct` v2 header's residual entropy model id distinguishes the arms, so a
decoder handed the wrong model fails loudly. Verified before running: the
network is bit-reproducible under deterministic cuDNN, the float -> integer
conversion is exact, every table totals 65536 with no zero entry, and the round
trip is exact.

Example usage (PowerShell, from the project root, with .venv activated):

    .venv\\Scripts\\python.exe scripts\\m10k_evaluate.py
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
from nvc.compression.range_coder import decode_symbols, encode_symbols
from nvc.evaluation.perceptual_metrics import msssim
from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device

DEFAULT_OUTPUT_DIR = Path("outputs/m10k_learned_entropy")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
FROZEN_LAMBDA = 3.0e-4
RATE_POINTS = (5, 4, 3)
ARMS = ("marginal", "local_activity4", "learned")
WATCH = ("bmx-bumps", "drone", "cat-girl", "drift-chicane", "gold-fish", "schoolgirls")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10K: learned vs hand-designed conditional entropy, identical symbols.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--quant-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--table-frames", type=int, default=600)
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def _residual_payload(arm: str, spec: dict[str, Any], symbols: np.ndarray,
                      reference_numpy: np.ndarray, reference_tensor, *, shape, bits: int,
                      mk) -> tuple[bytes, float]:
    """Code one frame's residual symbols under one arm, and report the ideal cost."""
    if arm == "learned":
        entropy_model, table = mk.frame_entropy_model(
            spec["model"], reference_tensor, bits=bits)
    else:
        context_model = spec["context_model"]
        entropy_model = spec["entropy_model"]
        contexts = context_model.contexts(reference_numpy)
        table = context_model.table_index(*shape, contexts)
    flat = symbols.reshape(-1)
    payload = encode_symbols(flat, entropy_model.cumulative, table)
    frequencies = entropy_model.frequencies.astype(np.float64)
    probabilities = frequencies / frequencies.sum(axis=1, keepdims=True)
    ideal = float(-np.log2(probabilities[table, flat]).sum())
    return payload, ideal


@torch.no_grad()
def encode_multi(mc, mk, model, frames, arms, paths, *, intra_params, intra_entropy_model,
                 residual_params, motion_entropy_model, bits, gop_size, block_size,
                 search_range) -> dict[str, Any]:
    """One closed-loop pass; every arm re-codes the SAME symbols."""
    model.eval()
    device = next(model.parameters()).device
    frame_count = frames.shape[0]
    types = mc.gop_frame_types(frame_count, gop_size)
    latent_shape = tuple(model.encode(frames[0:1].to(device)).shape[1:])
    blocks = (frames.shape[2] // block_size, frames.shape[3] // block_size)

    writers, records = {}, {arm: [] for arm in arms}
    ideal_bits = {arm: 0.0 for arm in arms}
    timings = {arm: 0.0 for arm in arms}
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
                        motion_payload, blocks, search_range=search_range,
                        entropy_model=motion_entropy_model)
                    warped = mc.warp_blocks(previous, decoded_motion, block_size=block_size)
                    reference_latent = model.encode(warped)
                    delta = latent - reference_latent

                    symbols = latent_to_symbols(delta, residual_params)   # ONE quantization
                    symbol_log.append(symbols.reshape(latent_shape))
                    reference_numpy = reference_latent[0].cpu().numpy()

                    for arm, spec in arms.items():
                        started = time.perf_counter()
                        payload, ideal = _residual_payload(
                            arm, spec, symbols, reference_numpy, reference_latent,
                            shape=latent_shape, bits=bits, mk=mk)
                        timings[arm] += time.perf_counter() - started
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
            "i_frames": sum(1 for r in rows if r["frame_type"] == "I"),
            "p_frames": sum(1 for r in rows if r["frame_type"] == "P"),
            "i_frame_residual_bytes": sum(r["residual_bytes"] for r in rows
                                          if r["frame_type"] == "I"),
            "p_frame_residual_bytes": sum(r["residual_bytes"] for r in rows
                                          if r["frame_type"] == "P"),
            "motion_bytes": motion_total, "residual_bytes": residual_total,
            "container_bytes": container,
            "container_overhead_bytes": container - motion_total - residual_total,
            "p_frame_ideal_bits": ideal_bits[arm],
            "residual_coding_seconds": timings[arm],
        }
    return result


@torch.no_grad()
def decode_sequence(mc, mk, model, path, arm, spec, *, intra_entropy_model,
                    motion_entropy_model, bits):
    """Decode using only the stream, the calibration and the arm's model."""
    model.eval()
    device = next(model.parameters()).device
    reader = mc.TemporalStreamReader(path)
    header = reader.header
    if spec["identity"] != header.residual_entropy_model_id:
        raise mc.TemporalFormatError(
            f"residual entropy model mismatch: stream declares "
            f"{header.residual_entropy_model_id.hex()}, supplied is {spec['identity'].hex()}")

    reconstructions, symbol_log, previous = [], [], None
    with mc.deterministic_kernels():
        for index, (frame_type, motion_payload, residual_payload) in enumerate(reader):
            if frame_type == mc.FRAME_TYPE_I:
                latent, _ = decode_payload_to_latent(
                    residual_payload, entropy_model=intra_entropy_model,
                    params=reader.intra_params, shape=header.latent_shape)
                reconstructed_latent = latent.to(device)
            else:
                motion = mc.decode_motion_payload(
                    motion_payload, header.block_grid, search_range=header.search_range,
                    entropy_model=motion_entropy_model)
                warped = mc.warp_blocks(previous, motion, block_size=header.block_size)
                reference_latent = model.encode(warped)
                channels, height, width = header.latent_shape
                if arm == "learned":
                    entropy_model, table = mk.frame_entropy_model(
                        spec["model"], reference_latent, bits=bits)
                else:
                    contexts = spec["context_model"].contexts(reference_latent[0].cpu().numpy())
                    entropy_model = spec["entropy_model"]
                    table = spec["context_model"].table_index(channels, height, width, contexts)
                symbols = decode_symbols(residual_payload, channels * height * width,
                                         entropy_model.cumulative, table)
                symbol_log.append(symbols.reshape(header.latent_shape))
                reconstructed_latent = reference_latent + symbols_to_latent(
                    symbols, header.latent_shape, reader.residual_params).to(device)

            reconstruction = model.decode(reconstructed_latent)
            reconstructions.append(reconstruction)
            previous = reconstruction
    return torch.cat(reconstructions, dim=0), symbol_log


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
    m10e = _load_script("m10e_evaluate")
    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    train_sequences = discover_sequences(args.manifest, split="train")
    test_sequences = discover_sequences(
        args.manifest, split="test", max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames_per_sequence)

    print("=" * 126)
    print("M10K - LEARNED vs HAND-DESIGNED CONDITIONAL ENTROPY (identical symbols, DAVIS test)")
    print("=" * 126)
    print(f"  frozen model : {args.checkpoint}   lambda {FROZEN_LAMBDA:.1e}")
    print(f"  arms         : intra (GOP=1), {', '.join(ARMS)}  (temporal, GOP={args.gop})")
    print(f"  rate points  : {args.rate_points} bit")
    print(f"  sequences    : {len(test_sequences)}  frames: "
          f"{sum(s.frame_count for s in test_sequences)}")

    stream_dir = args.output_dir / "benchmark_streams"
    stream_dir.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[str, int], dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    per_sequence_rows: list[dict[str, Any]] = []
    model_costs: list[dict[str, Any]] = []

    for bits in args.rate_points:
        print(f"\n  preparing {bits}-bit ...", flush=True)
        calibration = mc.calibrate_grids(
            model, train_sequences, bits=bits, mode=args.quant_mode, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            reference_mode="mc", max_frames=args.calibration_frames)
        symbols, references = ce.collect_training_symbols(
            mc, model, train_sequences, calibration, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, device=device,
            max_frames=args.table_frames)

        arms: dict[str, dict[str, Any]] = {}
        for scheme in ("marginal", "local_activity4"):
            context_model = ce.fit_context_model(scheme, references)
            built = ce.build_conditional_entropy_model(symbols, references, context_model,
                                                       bits=bits)
            arms[scheme] = {"context_model": context_model,
                            "entropy_model": built["entropy_model"],
                            "identity": built["entropy_model"].model_id()}
            provenance[f"{bits}bit_{scheme}"] = built["provenance"]

        model_path = args.model_dir / f"learned_entropy_{bits}bit.pt"
        if not model_path.is_file():
            print(f"[ERROR] learned model not found: {model_path}. Run the gate first.",
                  file=sys.stderr)
            return 1
        learned, checkpoint = mk.load_entropy_model(model_path, device=device)
        parameters = sum(p.numel() for p in learned.parameters())
        # Identity: hash the weights, so a different learned model is a different
        # stream and the container's existing check catches a mismatch.
        import hashlib
        digest = hashlib.sha256()
        for key in sorted(checkpoint["model_state_dict"]):
            digest.update(key.encode())
            digest.update(checkpoint["model_state_dict"][key].cpu().numpy().tobytes())
        arms["learned"] = {"model": learned, "identity": digest.digest()[:8]}
        provenance[f"{bits}bit_learned"] = {
            "split": "train", "selected_epoch": checkpoint["selection"]["selected_epoch"],
            "selection_metric": checkpoint["selection"]["selection_metric"],
            "parameters": parameters,
            "checkpoint_bytes": model_path.stat().st_size,
            "model_identity": digest.digest()[:8].hex(),
        }
        model_costs.append({"bits": bits, "parameters": parameters,
                            "checkpoint_bytes": model_path.stat().st_size})

        # intra control
        intra_rows = []
        for sequence in test_sequences:
            frames = sequence.load_frames()
            path = stream_dir / f"intra_{bits}bit_{sequence.sequence_id}.nvct"
            encoded = mc.encode_sequence(
                model, frames, path,
                intra_params=calibration["intra_params"],
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
                "total_pixels": sequence.total_pixels,
                "raw_rgb_bytes": sequence.raw_rgb_bytes(),
                "stream_bpp": encoded["container_bytes"] * 8 / sequence.total_pixels,
                "mean_psnr_db": 10.0 * math.log10(1.0 / mse),
                "mean_msssim": float(msssim(recon.clamp(0, 1), frames).mean()),
                "residual_coding_seconds": 0.0, "p_frames": 0,
            })

        arm_rows = {arm: [] for arm in ARMS}
        invariants = {"symbols": True, "reconstruction": True, "motion": True, "metrics": True}
        for sequence in test_sequences:
            frames = sequence.load_frames()
            paths = {arm: stream_dir / f"{arm}_{bits}bit_{sequence.sequence_id}.nvct"
                     for arm in ARMS}
            result = encode_multi(
                mc, mk, model, frames, arms, paths,
                intra_params=calibration["intra_params"],
                intra_entropy_model=calibration["intra_entropy_model"],
                residual_params=calibration["residual_params"],
                motion_entropy_model=calibration["motion_entropy_model"],
                bits=bits, gop_size=args.gop, block_size=args.block_size,
                search_range=args.search_range)

            recon = result["reconstructions"]
            mse = torch.mean((recon - frames) ** 2).item()
            psnr = 10.0 * math.log10(1.0 / mse)
            quality = float(msssim(recon.clamp(0, 1), frames).mean())
            invariants["motion"] &= len({result["arms"][a]["motion_bytes"] for a in ARMS}) == 1

            record = {"sequence": sequence.sequence_id, "bits": bits,
                      "watched": sequence.sequence_id in WATCH}
            for arm in ARMS:
                stats = result["arms"][arm]
                decoded, decoded_symbols = decode_sequence(
                    mc, mk, model, paths[arm], arm, arms[arm],
                    intra_entropy_model=calibration["intra_entropy_model"],
                    motion_entropy_model=calibration["motion_entropy_model"], bits=bits)
                invariants["symbols"] &= all(
                    np.array_equal(a.reshape(-1), b.reshape(-1))
                    for a, b in zip(result["symbols"], decoded_symbols))
                invariants["reconstruction"] &= torch.equal(decoded.cpu(), recon)
                arm_rows[arm].append({
                    "sequence": sequence.sequence_id, **stats,
                    "total_pixels": sequence.total_pixels,
                    "raw_rgb_bytes": sequence.raw_rgb_bytes(),
                    "stream_bpp": stats["container_bytes"] * 8 / sequence.total_pixels,
                    "mean_psnr_db": psnr, "mean_msssim": quality,
                })
                record[f"{arm}_residual_bytes"] = stats["residual_bytes"]
                record[f"{arm}_bpp"] = stats["container_bytes"] * 8 / sequence.total_pixels
            per_sequence_rows.append(record)

        def aggregate(name, rows):
            total = sum(r["container_bytes"] for r in rows)
            pixels = sum(r["total_pixels"] for r in rows)
            return {
                "arm": name, "bits": bits, "sequences": rows,
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
                "residual_coding_seconds": sum(r["residual_coding_seconds"] for r in rows),
                "p_frames": sum(r["p_frames"] for r in rows),
                "byte_accounting_closes": sum(
                    r["motion_bytes"] + r["residual_bytes"] + r["container_overhead_bytes"]
                    for r in rows) == total,
            }

        results[("intra", bits)] = aggregate("intra", intra_rows)
        for arm in ARMS:
            results[(arm, bits)] = aggregate(arm, arm_rows[arm])
        metrics = {(round(results[(a, bits)]["mean_psnr_db"], 9),
                    round(results[(a, bits)]["mean_msssim"], 9)) for a in ARMS}
        invariants["metrics"] = len(metrics) == 1
        print(f"    invariants - symbols {invariants['symbols']} | reconstruction "
              f"{invariants['reconstruction']} | motion {invariants['motion']} | "
              f"PSNR/MS-SSIM {invariants['metrics']}")
        if not all(invariants.values()):
            print("[ERROR] the arms diverged; rate results are not interpretable",
                  file=sys.stderr)
            return 1

    print()
    print("=" * 126)
    print("FULL BYTE ACCOUNTING")
    print("=" * 126)
    print(f"{'arm':<17} {'bits':>5} {'I bytes':>12} {'P resid':>12} {'motion':>10} "
          f"{'TOTAL':>12} {'BPP':>8} {'PSNR':>8} {'MS-SSIM':>8} {'vs M10H':>9} {'vs M10J':>9}")
    for bits in args.rate_points:
        base = results[("marginal", bits)]["total_residual_bytes"]
        m10j = results[("local_activity4", bits)]["total_residual_bytes"]
        for arm in ["intra"] + list(ARMS):
            a = results[(arm, bits)]
            vs_h = "" if arm in ("intra", "marginal") else \
                f"{(a['total_residual_bytes'] - base) / base * 100:+8.2f}%"
            vs_j = "" if arm != "learned" else \
                f"{(a['total_residual_bytes'] - m10j) / m10j * 100:+8.2f}%"
            print(f"{arm:<17} {bits:>5} {a['total_i_frame_residual_bytes']:>12,} "
                  f"{a['total_p_frame_residual_bytes']:>12,} {a['total_motion_bytes']:>10,} "
                  f"{a['total_container_bytes']:>12,} {a['stream_bpp']:>8.4f} "
                  f"{a['mean_psnr_db']:>8.3f} {a['mean_msssim']:>8.4f} {vs_h:>9} {vs_j:>9}")

    print()
    print("=" * 126)
    print("IDEAL BITS vs EMITTED BYTES - does the coder realise the modelling gain?")
    print("=" * 126)
    print(f"{'bits':>5} {'arm':<17} {'ideal P bits':>16} {'vs M10H':>10} "
          f"{'P resid bytes':>15} {'vs M10H':>10} {'coder overhead':>15} {'realised':>10}")
    theory_rows = []
    for bits in args.rate_points:
        base = results[("marginal", bits)]
        for arm in ARMS:
            a = results[(arm, bits)]
            ideal_gain = (base["p_frame_ideal_bits"] - a["p_frame_ideal_bits"]) \
                / base["p_frame_ideal_bits"] * 100
            byte_gain = (base["total_p_frame_residual_bytes"] - a["total_p_frame_residual_bytes"]) \
                / base["total_p_frame_residual_bytes"] * 100
            overhead = (a["total_p_frame_residual_bytes"] * 8 - a["p_frame_ideal_bits"]) \
                / a["p_frame_ideal_bits"] * 100
            realised = byte_gain / ideal_gain * 100 if abs(ideal_gain) > 1e-9 else float("nan")
            theory_rows.append({"bits": bits, "arm": arm,
                                "ideal_p_frame_bits": a["p_frame_ideal_bits"],
                                "ideal_reduction_percent": ideal_gain,
                                "byte_reduction_percent": byte_gain,
                                "coder_overhead_percent": overhead,
                                "realised_fraction_percent": realised})
            print(f"{bits:>5} {arm:<17} {a['p_frame_ideal_bits']:>16,.0f} {ideal_gain:>+9.2f}% "
                  f"{a['total_p_frame_residual_bytes']:>15,} {byte_gain:>+9.2f}% "
                  f"{overhead:>+14.2f}% "
                  f"{(f'{realised:.0f}%' if realised == realised else 'n/a'):>10}")

    print()
    print("=" * 126)
    print("BD-RATE over three rate points")
    print("=" * 126)

    def curve(arm, metric="mean_psnr_db"):
        return [(results[(arm, b)]["stream_bpp"], results[(arm, b)][metric])
                for b in args.rate_points]

    bd_rows = []
    print(f"{'comparison':<40} {'PSNR BD-rate':>15} {'MS-SSIM BD-rate':>18}")
    for test, base in (("local_activity4", "marginal"), ("learned", "marginal"),
                       ("learned", "local_activity4"), ("marginal", "intra"),
                       ("local_activity4", "intra"), ("learned", "intra")):
        psnr_bd = m10e._bd_rate_linear(curve(base), curve(test))
        ms_bd = m10e._bd_rate_linear(curve(base, "mean_msssim"), curve(test, "mean_msssim"))
        bd_rows.append({"test": test, "base": base, "bd_rate_psnr": psnr_bd,
                        "bd_rate_msssim": ms_bd})
        print(f"{test + ' vs ' + base:<40} "
              f"{(f'{psnr_bd:+.2f}%' if psnr_bd is not None else 'n/a'):>15} "
              f"{(f'{ms_bd:+.2f}%' if ms_bd is not None else 'n/a'):>18}")

    print()
    print("=" * 126)
    print("MODEL SIZE AND CODING COST - is the gain worth what it costs?")
    print("=" * 126)
    print(f"{'bits':>5} {'params':>10} {'ckpt bytes':>12} "
          + " ".join(f"{a + ' ms/P':>18}" for a in ARMS))
    for bits in args.rate_points:
        cost = next(c for c in model_costs if c["bits"] == bits)
        cells = []
        for arm in ARMS:
            a = results[(arm, bits)]
            cells.append(f"{a['residual_coding_seconds'] * 1000 / max(a['p_frames'], 1):>18.2f}")
        print(f"{bits:>5} {cost['parameters']:>10,} {cost['checkpoint_bytes']:>12,} "
              + " ".join(cells))
    print("\n  ms/P is encoder-side residual probability-modelling + entropy coding per P-frame.")
    print("  M10J's tables are a lookup; M10K adds a network forward pass and a 16,384-table build.")

    reference_bits = args.rate_points[1] if len(args.rate_points) > 1 else args.rate_points[0]
    print()
    print("=" * 126)
    print(f"PER-SEQUENCE residual bytes at {reference_bits}-bit (* = motion-sensitive)")
    print("=" * 126)
    print(f"{'sequence':<18} " + " ".join(f"{a:>18}" for a in ARMS)
          + f" {'K vs J':>9}")
    for record in [r for r in per_sequence_rows if r["bits"] == reference_bits]:
        delta = (record["learned_residual_bytes"] - record["local_activity4_residual_bytes"]) \
            / record["local_activity4_residual_bytes"] * 100
        record["learned_vs_m10j_percent"] = delta
        print(f"{'*' if record['watched'] else ' '}{record['sequence']:<17} "
              + " ".join(f"{record[f'{a}_residual_bytes']:>18,}" for a in ARMS)
              + f" {delta:>+8.2f}%")
    losses = [r for r in per_sequence_rows if r["bits"] == reference_bits
              and r["learned_vs_m10j_percent"] > 0]
    print(f"\n  M10K loses to M10J on {len(losses)}/9 sequences at {reference_bits}-bit"
          + (f": {', '.join(r['sequence'] for r in losses)}" if losses else ""))

    closes = all(results[k]["byte_accounting_closes"] for k in results)
    print(f"\n  byte accounting closes everywhere: {closes}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "M10K learned conditional entropy benchmark",
        "frozen_lambda": FROZEN_LAMBDA, "checkpoint": str(args.checkpoint),
        "rate_points_bits": list(args.rate_points),
        "arms": {f"{k[0]}@{k[1]}bit": {kk: vv for kk, vv in v.items() if kk != "sequences"}
                 for k, v in results.items()},
        "per_sequence": {f"{k[0]}@{k[1]}bit": v["sequences"] for k, v in results.items()},
        "ideal_vs_deployed": theory_rows, "bd_rate": bd_rows,
        "per_sequence_summary": per_sequence_rows,
        "model_costs": model_costs, "calibration_provenance": provenance,
        "invariants": {"symbols_identical": True, "reconstruction_identical": True,
                       "motion_identical": True, "metrics_identical": True,
                       "byte_accounting_closes": closes},
    }
    path = args.output_dir / "learned_entropy_benchmark.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
