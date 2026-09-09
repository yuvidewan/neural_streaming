"""M10L - shared-codebook entropy coding on the DAVIS test set.

Four arms code the SAME residual symbols of the SAME M10H motion-compensated
stream, in one closed loop, so nothing but the entropy representation differs:

    marginal          M10H, one table per latent channel
    local_activity4   M10J, hand-designed 4-bucket context
    learned           M10K, one learned table per symbol position (16,384)
    codebook          M10L, K shared prototype tables

Because the arms share the loop, reconstruction, motion and PSNR/MS-SSIM are
IDENTICAL by construction, and that is asserted per rate point rather than
assumed. If they ever diverge the run stops: a bitrate comparison between arms
that decoded different pictures would be meaningless.

The codebook K used here comes from the offline gate, which selected it on
VALIDATION. Nothing in this script re-selects K, re-fits a codebook or looks at
test data before coding it.

Run (after scripts/m10l_offline_gate.py):
  ./.venv/Scripts/python.exe scripts/m10l_evaluate.py
"""

from __future__ import annotations

import argparse
import hashlib
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

DEFAULT_OUTPUT_DIR = Path("outputs/m10l_shared_codebook")
DEFAULT_MODEL_DIR = Path("outputs/m10k_learned_entropy")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
FROZEN_LAMBDA = 3.0e-4
RATE_POINTS = (5, 4, 3)
ARMS = ("marginal", "local_activity4", "learned", "codebook")
WATCH = ("bmx-bumps", "drone", "cat-girl", "drift-chicane", "gold-fish", "schoolgirls")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _m10k_model_identity(state_dict) -> bytes:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        digest.update(key.encode())
        digest.update(state_dict[key].cpu().numpy().tobytes())
    return digest.digest()[:8]


def calibration_signature(calibration, *, bits: int, calibration_frames: int,
                          quant_mode: str) -> str:
    """A stable fingerprint of the quantization grid a model/codebook was fitted to.

    M10K's deployment lesson was that a learned entropy model scored against a
    different grid silently costs bits instead of failing. A codebook inherits
    that coupling and adds one of its own, so the grid itself is hashed into the
    stream identity - a stale pairing then fails the container's existing check.
    """
    digest = hashlib.sha256()
    parameters = calibration["residual_params"]
    digest.update(np.asarray(parameters.scale.cpu().numpy(), dtype=np.float64).tobytes())
    digest.update(np.asarray(parameters.zero_point.cpu().numpy(),
                             dtype=np.float64).tobytes())
    digest.update(json.dumps({"bits": bits, "mode": quant_mode,
                              "calibration_frames": calibration_frames},
                             sort_keys=True).encode())
    return digest.hexdigest()[:16]


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10L: shared-codebook entropy coding on the DAVIS test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gate-report", type=Path, default=None,
                        help="offline gate report that selected K (default: output-dir)")
    parser.add_argument("--codebook-size", type=int, default=None,
                        help="override the gate's K; recorded in the report if used")
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--quant-mode", choices=["global", "per_channel"],
                        default="per_channel")
    parser.add_argument("--calibration-frames", type=int, default=400,
                        help="MUST match the value the M10K models were fitted under")
    parser.add_argument("--table-frames", type=int, default=600)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def _residual_payload(arm, spec, symbols, reference_numpy, reference_tensor, *,
                      shape, bits, mk, ml):
    """Code one frame's residual symbols under one arm, and report its ideal cost.

    Returns the payload, the ideal code length, and a split of the modelling
    time into (table work, arithmetic coding) - the two numbers M10L is trading
    against each other.
    """
    started = time.perf_counter()
    if arm == "learned":
        entropy_model, table = mk.frame_entropy_model(spec["model"], reference_tensor,
                                                      bits=bits)
        cumulative, probabilities = entropy_model.cumulative, entropy_model.frequencies
    elif arm == "codebook":
        codebook = spec["codebook"]
        table = ml.frame_table_index(spec["model"], codebook, reference_tensor)
        cumulative, probabilities = codebook.cumulative, codebook.frequencies
    else:
        context_model, entropy_model = spec["context_model"], spec["entropy_model"]
        contexts = context_model.contexts(reference_numpy)
        table = context_model.table_index(*shape, contexts)
        cumulative, probabilities = entropy_model.cumulative, entropy_model.frequencies
    table_seconds = time.perf_counter() - started

    flat = symbols.reshape(-1)
    started = time.perf_counter()
    payload = encode_symbols(flat, cumulative, table)
    encode_seconds = time.perf_counter() - started

    floats = probabilities.astype(np.float64) / TOTAL_FREQUENCY
    ideal = float(-np.log2(floats[table, flat]).sum())
    return payload, ideal, table_seconds, encode_seconds


@torch.no_grad()
def encode_multi(mc, mk, ml, model, frames, arms, paths, *, intra_params,
                 intra_entropy_model, residual_params, motion_entropy_model, bits,
                 gop_size, block_size, search_range) -> dict[str, Any]:
    """One closed-loop pass; every arm re-codes the SAME symbols."""
    model.eval()
    device = next(model.parameters()).device
    frame_count = frames.shape[0]
    types = mc.gop_frame_types(frame_count, gop_size)
    latent_shape = tuple(model.encode(frames[0:1].to(device)).shape[1:])

    writers, records = {}, {arm: [] for arm in arms}
    ideal_bits = {arm: 0.0 for arm in arms}
    table_seconds = {arm: 0.0 for arm in arms}
    encode_seconds = {arm: 0.0 for arm in arms}
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
        writers[arm] = mc.TemporalStreamWriter(paths[arm], header, intra_params,
                                               residual_params)

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
                        motion, search_range=search_range,
                        entropy_model=motion_entropy_model)
                    decoded_motion = mc.decode_motion_payload(
                        motion_payload, (frames.shape[2] // block_size,
                                         frames.shape[3] // block_size),
                        search_range=search_range, entropy_model=motion_entropy_model)
                    warped = mc.warp_blocks(previous, decoded_motion, block_size=block_size)
                    reference_latent = model.encode(warped)
                    delta = latent - reference_latent

                    symbols = latent_to_symbols(delta, residual_params)   # ONE quantization
                    symbol_log.append(symbols.reshape(latent_shape))
                    reference_numpy = reference_latent[0].cpu().numpy()

                    for arm, spec in arms.items():
                        payload, ideal, tables, coding = _residual_payload(
                            arm, spec, symbols, reference_numpy, reference_latent,
                            shape=latent_shape, bits=bits, mk=mk, ml=ml)
                        ideal_bits[arm] += ideal
                        table_seconds[arm] += tables
                        encode_seconds[arm] += coding
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
            "table_seconds": table_seconds[arm],
            "encode_seconds": encode_seconds[arm],
            "residual_coding_seconds": table_seconds[arm] + encode_seconds[arm],
        }
    return result


@torch.no_grad()
def decode_sequence(mc, mk, ml, model, path, arm, spec, *, intra_entropy_model,
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
        for frame_type, motion_payload, residual_payload in reader:
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
                    cumulative = entropy_model.cumulative
                elif arm == "codebook":
                    table = ml.frame_table_index(spec["model"], spec["codebook"],
                                                 reference_latent)
                    cumulative = spec["codebook"].cumulative
                else:
                    contexts = spec["context_model"].contexts(
                        reference_latent[0].cpu().numpy())
                    cumulative = spec["entropy_model"].cumulative
                    table = spec["context_model"].table_index(
                        channels, height, width, contexts)
                symbols = decode_symbols(residual_payload, channels * height * width,
                                         cumulative, table)
                symbol_log.append(symbols.reshape(header.latent_shape))
                reconstructed_latent = reference_latent + symbols_to_latent(
                    symbols, header.latent_shape, reader.residual_params).to(device)

            reconstruction = model.decode(reconstructed_latent)
            reconstructions.append(reconstruction)
            previous = reconstruction
    return torch.cat(reconstructions, dim=0), symbol_log


def _selection(args, ml) -> tuple[dict[int, int], str]:
    """K per rate point and the assignment metric, both decided by the offline gate.

    Nothing here re-selects anything: the gate chose on VALIDATION, before test
    data existed to this milestone, and this reads its answer. An explicit
    --codebook-size override is honoured but recorded in the report, so a
    hand-picked K can never be mistaken for a selected one.
    """
    path = args.gate_report or (args.output_dir / "offline_gate.json")
    if not path.is_file():
        raise SystemExit(f"[ERROR] gate report not found: {path}. "
                         f"Run scripts/m10l_offline_gate.py first.")
    report = json.loads(path.read_text(encoding="utf-8"))
    metric = report.get("selected_metric", ml.DEFAULT_METRIC)
    if args.codebook_size is not None:
        return {bits: args.codebook_size for bits in args.rate_points}, metric
    if not report.get("gate_passed"):
        raise SystemExit("[ERROR] the offline gate did not pass; there is no K to "
                         "benchmark. That result stands on its own - see the gate report.")
    return ({int(bits): row["codebook_size"] for bits, row in report["selected"].items()},
            metric)


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
    m10e = _load_script("m10e_evaluate")
    sizes, metric = _selection(args, ml)

    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    train_sequences = discover_sequences(args.manifest, split="train")
    test_sequences = discover_sequences(
        args.manifest, split="test", max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames_per_sequence)

    print("=" * 132)
    print("M10L - SHARED ENTROPY-TABLE CODEBOOK vs M10K (identical symbols, DAVIS test)")
    print("=" * 132)
    print(f"  frozen model : {args.checkpoint}   lambda {FROZEN_LAMBDA:.1e}")
    print(f"  arms         : intra (GOP=1), {', '.join(ARMS)}  (temporal, GOP={args.gop})")
    print(f"  codebook K   : {sizes}  metric: {metric}  "
          f"(both selected on VALIDATION by the offline gate)")
    print(f"  sequences    : {len(test_sequences)}  frames: "
          f"{sum(s.frame_count for s in test_sequences)}")

    stream_dir = args.output_dir / "benchmark_streams"
    stream_dir.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[str, int], dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    table_counts: dict[tuple[str, int], int] = {}
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
        signature = calibration_signature(
            calibration, bits=bits, calibration_frames=args.calibration_frames,
            quant_mode=args.quant_mode)

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
            print(f"[ERROR] M10K model not found: {model_path}", file=sys.stderr)
            return 1
        learned, checkpoint = mk.load_entropy_model(model_path, device=device)
        parameters = sum(p.numel() for p in learned.parameters())
        learned_identity = _m10k_model_identity(checkpoint["model_state_dict"])
        arms["learned"] = {"model": learned, "identity": learned_identity}
        provenance[f"{bits}bit_learned"] = {
            "split": "train", "selected_epoch": checkpoint["selection"]["selected_epoch"],
            "parameters": parameters, "checkpoint_bytes": model_path.stat().st_size,
            "model_identity": learned_identity.hex(),
            "calibration_signature": signature,
        }

        size = sizes[bits]
        codebook_path = (args.output_dir / "codebooks"
                         / f"codebook_{bits}bit_K{size}_{metric}.json")
        if not codebook_path.is_file():
            print(f"[ERROR] codebook not found: {codebook_path}. Run the gate first.",
                  file=sys.stderr)
            return 1
        codebook = ml.SharedCodebook.from_dict(
            json.loads(codebook_path.read_text(encoding="utf-8")))
        codebook_identity = codebook.codebook_id(
            model_identity=learned_identity, calibration_signature=signature)
        arms["codebook"] = {"model": learned, "codebook": codebook,
                            "identity": codebook_identity}
        provenance[f"{bits}bit_codebook"] = {
            "split": "train", "codebook_size": codebook.size,
            "metric": codebook.metric,
            "codebook_bytes_on_disk": codebook_path.stat().st_size,
            "codebook_table_memory_bytes": codebook.table_memory_bytes(),
            "codebook_identity": codebook_identity.hex(),
            "m10k_model_identity": learned_identity.hex(),
            "calibration_signature": signature,
            "fit_provenance": codebook.provenance,
        }
        model_costs.append({
            "bits": bits, "parameters": parameters,
            "checkpoint_bytes": model_path.stat().st_size,
            "codebook_size": codebook.size,
            "m10k_tables": 16384, "m10k_table_memory_bytes": 16384 * (2 ** bits + 1) * 8
                                                             + 16384 * (2 ** bits) * 8,
            "m10l_table_memory_bytes": codebook.table_memory_bytes(),
            "m10l_index_memory_bytes": 16384 * 8,
        })

        # intra control (GOP=1, no temporal prediction at all)
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
                "table_seconds": 0.0, "encode_seconds": 0.0,
                "residual_coding_seconds": 0.0, "p_frames": 0,
            })

        table_counts[("marginal", bits)] = int(
            arms["marginal"]["entropy_model"].num_tables)
        table_counts[("local_activity4", bits)] = int(
            arms["local_activity4"]["entropy_model"].num_tables)
        table_counts[("learned", bits)] = 16384
        table_counts[("codebook", bits)] = codebook.size

        arm_rows = {arm: [] for arm in ARMS}
        invariants = {"symbols": True, "reconstruction": True, "motion": True,
                      "metrics": True}
        for sequence in test_sequences:
            frames = sequence.load_frames()
            paths = {arm: stream_dir / f"{arm}_{bits}bit_{sequence.sequence_id}.nvct"
                     for arm in ARMS}
            result = encode_multi(
                mc, mk, ml, model, frames, arms, paths,
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
            invariants["motion"] &= len(
                {result["arms"][a]["motion_bytes"] for a in ARMS}) == 1

            record = {"sequence": sequence.sequence_id, "bits": bits,
                      "watched": sequence.sequence_id in WATCH}
            for arm in ARMS:
                stats = result["arms"][arm]
                decoded, decoded_symbols = decode_sequence(
                    mc, mk, ml, model, paths[arm], arm, arms[arm],
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
                "total_i_frame_residual_bytes": sum(r["i_frame_residual_bytes"]
                                                    for r in rows),
                "total_p_frame_residual_bytes": sum(r["p_frame_residual_bytes"]
                                                    for r in rows),
                "total_container_overhead_bytes": sum(r["container_overhead_bytes"]
                                                      for r in rows),
                "total_container_bytes": total, "total_pixels": pixels,
                "stream_bpp": total * 8 / pixels,
                "compression_ratio": sum(r["raw_rgb_bytes"] for r in rows) / total,
                "mean_psnr_db": statistics.fmean(r["mean_psnr_db"] for r in rows),
                "mean_msssim": statistics.fmean(r["mean_msssim"] for r in rows),
                "p_frame_ideal_bits": sum(r["p_frame_ideal_bits"] for r in rows),
                "table_seconds": sum(r["table_seconds"] for r in rows),
                "encode_seconds": sum(r["encode_seconds"] for r in rows),
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
    print("=" * 132)
    print("FULL BYTE ACCOUNTING")
    print("=" * 132)
    print(f"{'arm':<17} {'bits':>5} {'I bytes':>12} {'P resid':>12} {'motion':>10} "
          f"{'TOTAL':>12} {'BPP':>8} {'PSNR':>8} {'MS-SSIM':>8} {'vs M10J':>9} "
          f"{'vs M10K':>9}")
    for bits in args.rate_points:
        m10j = results[("local_activity4", bits)]["total_residual_bytes"]
        m10k = results[("learned", bits)]["total_residual_bytes"]
        for arm in ["intra"] + list(ARMS):
            a = results[(arm, bits)]
            vs_j = "" if arm in ("intra", "marginal", "local_activity4") else \
                f"{(a['total_residual_bytes'] - m10j) / m10j * 100:+8.2f}%"
            vs_k = "" if arm != "codebook" else \
                f"{(a['total_residual_bytes'] - m10k) / m10k * 100:+8.2f}%"
            print(f"{arm:<17} {bits:>5} {a['total_i_frame_residual_bytes']:>12,} "
                  f"{a['total_p_frame_residual_bytes']:>12,} {a['total_motion_bytes']:>10,} "
                  f"{a['total_container_bytes']:>12,} {a['stream_bpp']:>8.4f} "
                  f"{a['mean_psnr_db']:>8.3f} {a['mean_msssim']:>8.4f} {vs_j:>9} {vs_k:>9}")

    print()
    print("=" * 132)
    print("IDEAL BITS vs EMITTED BYTES - what does the codebook actually cost in rate?")
    print("=" * 132)
    print(f"{'bits':>5} {'arm':<17} {'ideal P bits':>16} {'vs M10K':>10} "
          f"{'P resid bytes':>15} {'vs M10K':>10} {'coder overhead':>15}")
    theory_rows = []
    for bits in args.rate_points:
        base = results[("learned", bits)]
        for arm in ARMS:
            a = results[(arm, bits)]
            ideal_delta = (a["p_frame_ideal_bits"] - base["p_frame_ideal_bits"]) \
                / base["p_frame_ideal_bits"] * 100
            byte_delta = (a["total_p_frame_residual_bytes"]
                          - base["total_p_frame_residual_bytes"]) \
                / base["total_p_frame_residual_bytes"] * 100
            overhead = (a["total_p_frame_residual_bytes"] * 8 - a["p_frame_ideal_bits"]) \
                / a["p_frame_ideal_bits"] * 100
            theory_rows.append({"bits": bits, "arm": arm,
                                "ideal_p_frame_bits": a["p_frame_ideal_bits"],
                                "ideal_delta_vs_m10k_percent": ideal_delta,
                                "byte_delta_vs_m10k_percent": byte_delta,
                                "coder_overhead_percent": overhead})
            print(f"{bits:>5} {arm:<17} {a['p_frame_ideal_bits']:>16,.0f} "
                  f"{ideal_delta:>+9.2f}% {a['total_p_frame_residual_bytes']:>15,} "
                  f"{byte_delta:>+9.2f}% {overhead:>+14.2f}%")

    print()
    print("=" * 132)
    print("BD-RATE over three rate points")
    print("=" * 132)

    def curve(arm, metric="mean_psnr_db"):
        return [(results[(arm, b)]["stream_bpp"], results[(arm, b)][metric])
                for b in args.rate_points]

    bd_rows = []
    print(f"{'comparison':<44} {'PSNR BD-rate':>15} {'MS-SSIM BD-rate':>18}")
    for test, base in (("learned", "local_activity4"), ("codebook", "local_activity4"),
                       ("codebook", "learned"), ("codebook", "marginal"),
                       ("codebook", "intra"), ("learned", "intra")):
        psnr_bd = m10e._bd_rate_linear(curve(base), curve(test))
        ms_bd = m10e._bd_rate_linear(curve(base, "mean_msssim"),
                                     curve(test, "mean_msssim"))
        bd_rows.append({"test": test, "base": base, "bd_rate_psnr": psnr_bd,
                        "bd_rate_msssim": ms_bd})
        print(f"{test + ' vs ' + base:<44} "
              f"{(f'{psnr_bd:+.2f}%' if psnr_bd is not None else 'n/a'):>15} "
              f"{(f'{ms_bd:+.2f}%' if ms_bd is not None else 'n/a'):>18}")

    print()
    print("=" * 132)
    print("LATENCY - probability modelling vs arithmetic coding, per P-frame (encoder side)")
    print("=" * 132)
    print(f"{'bits':>5} {'arm':<17} {'K':>7} {'model ms':>10} {'coder ms':>10} "
          f"{'total ms':>10} {'vs M10K model':>15} {'vs M10K total':>15}")
    print("  model ms = everything before the coder. For learned and codebook that")
    print("  INCLUDES the shared network forward; the offline gate reports the split.")
    latency_rows = []
    for bits in args.rate_points:
        base = results[("learned", bits)]
        base_p = max(base["p_frames"], 1)
        base_table = base["table_seconds"] * 1000 / base_p
        base_total = base["residual_coding_seconds"] * 1000 / base_p
        for arm in ARMS:
            a = results[(arm, bits)]
            frames_p = max(a["p_frames"], 1)
            table_ms = a["table_seconds"] * 1000 / frames_p
            coder_ms = a["encode_seconds"] * 1000 / frames_p
            total_ms = a["residual_coding_seconds"] * 1000 / frames_p
            tables = table_counts[(arm, bits)]
            latency_rows.append({
                "bits": bits, "arm": arm, "tables": tables,
                "modelling_ms_per_p_frame": table_ms, "coder_ms_per_p_frame": coder_ms,
                "total_ms_per_p_frame": total_ms,
                "modelling_reduction_vs_m10k_percent": (base_table - table_ms)
                                                   / base_table * 100,
                "total_reduction_vs_m10k_percent": (base_total - total_ms)
                                                   / base_total * 100,
            })
            print(f"{bits:>5} {arm:<17} {tables:>7,} {table_ms:>10.3f} {coder_ms:>10.3f} "
                  f"{total_ms:>10.3f} "
                  f"{(base_table - table_ms) / base_table * 100:>14.1f}% "
                  f"{(base_total - total_ms) / base_total * 100:>14.1f}%")

    print()
    print("=" * 132)
    print("MEMORY - coder-facing probability tables")
    print("=" * 132)
    print(f"{'bits':>5} {'M10K tables':>13} {'M10K memory':>14} {'M10L K':>8} "
          f"{'M10L memory':>14} {'M10L index':>12} {'reduction':>11}")
    for cost in model_costs:
        reduction = (cost["m10k_table_memory_bytes"]
                     - cost["m10l_table_memory_bytes"] - cost["m10l_index_memory_bytes"]) \
            / cost["m10k_table_memory_bytes"] * 100
        cost["table_memory_reduction_percent"] = reduction
        print(f"{cost['bits']:>5} {cost['m10k_tables']:>13,} "
              f"{cost['m10k_table_memory_bytes'] / 1e6:>13.2f}M {cost['codebook_size']:>8} "
              f"{cost['m10l_table_memory_bytes'] / 1e3:>13.1f}K "
              f"{cost['m10l_index_memory_bytes'] / 1e3:>11.1f}K {reduction:>10.2f}%")

    reference_bits = args.rate_points[1] if len(args.rate_points) > 1 else args.rate_points[0]
    print()
    print("=" * 132)
    print(f"PER-SEQUENCE residual bytes at {reference_bits}-bit (* = motion-sensitive)")
    print("=" * 132)
    print(f"{'sequence':<18} " + " ".join(f"{a:>17}" for a in ARMS) + f" {'L vs K':>9}")
    for record in [r for r in per_sequence_rows if r["bits"] == reference_bits]:
        delta = (record["codebook_residual_bytes"] - record["learned_residual_bytes"]) \
            / record["learned_residual_bytes"] * 100
        record["codebook_vs_m10k_percent"] = delta
        print(f"{'*' if record['watched'] else ' '}{record['sequence']:<17} "
              + " ".join(f"{record[f'{a}_residual_bytes']:>17,}" for a in ARMS)
              + f" {delta:>+8.2f}%")
    worse = [r for r in per_sequence_rows if r["bits"] == reference_bits
             and r["codebook_vs_m10k_percent"] > 0]
    print(f"\n  M10L costs more than M10K on {len(worse)}/{len(test_sequences)} sequences "
          f"at {reference_bits}-bit"
          + (f": {', '.join(r['sequence'] for r in worse)}" if worse else ""))

    closes = all(results[k]["byte_accounting_closes"] for k in results)
    print(f"  byte accounting closes everywhere: {closes}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "M10L shared-codebook entropy benchmark",
        "frozen_lambda": FROZEN_LAMBDA, "checkpoint": str(args.checkpoint),
        "rate_points_bits": list(args.rate_points),
        "codebook_sizes": {str(k): v for k, v in sizes.items()},
        "codebook_size_override": args.codebook_size,
        "assignment_metric": metric,
        "arms": {f"{k[0]}@{k[1]}bit": {kk: vv for kk, vv in v.items() if kk != "sequences"}
                 for k, v in results.items()},
        "per_sequence": {f"{k[0]}@{k[1]}bit": v["sequences"] for k, v in results.items()},
        "ideal_vs_deployed": theory_rows, "bd_rate": bd_rows,
        "latency": latency_rows, "per_sequence_summary": per_sequence_rows,
        "model_costs": model_costs, "calibration_provenance": provenance,
        "invariants": {"symbols_identical": True, "reconstruction_identical": True,
                       "motion_identical": True, "metrics_identical": True,
                       "byte_accounting_closes": closes},
    }
    path = args.output_dir / "shared_codebook_benchmark.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
