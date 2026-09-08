"""M10I evaluation: does conditioning the residual on the reference reduce residual rate?

THE ABLATION
-------------
    intra   every frame independent (GOP=1)
    m10h    motion compensation + MARGINAL residual coding      (the baseline)
    m10i    motion compensation + CONDITIONAL residual coding   (the hypothesis)

`m10h` and `m10i` share the same frozen encoder/decoder, the same block-matching
motion estimator, the same motion quantization and the same motion entropy
model, and both write the same `.nvct` v2 container. The ONLY difference is
whether the tensor handed to the quantizer is the raw residual `r` or the
conditioned `w = f_a(r, z_ref)`.

That matters for the headline claim: because motion is produced by identical
code from identical inputs, **motion bytes must come out identical between the
two arms**, and the evaluator asserts it. Any change in total bitrate is
therefore attributable to the residual, which is exactly what the hypothesis is
about. The stated success signature is: motion bits unchanged, residual bits
down, quality maintained or improved.

THREE RATE POINTS, CHOSEN FROM MEASUREMENT
--------------------------------------------
A bit-depth probe of the M10H MC path found the RD curve is essentially VERTICAL
above 5-bit: 8 -> 5 bit halves the rate while costing 0.08 dB. 8/6/5 would
therefore produce three points with no quality separation and nothing for a
BD-rate integration to span. 5/4/3 spans 29.710 -> 28.289 dB (1.42 dB) over a
2.1x rate range, which is a real curve.

Example usage (PowerShell, from the project root, with .venv activated):

    .venv\\Scripts\\python.exe scripts\\m10i_evaluate.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.codec import (
    decode_payload_to_latent,
    encode_latent_to_payload,
    latent_to_symbols,
)
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.evaluation.perceptual_metrics import msssim
from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device

DEFAULT_OUTPUT_DIR = Path("outputs/m10i_temporal_residual")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_CODEC = Path("outputs/m10i_temporal_residual/conditional_codec.pt")
FROZEN_LAMBDA = 3.0e-4
RATE_POINTS = (5, 4, 3)
WATCH = ("bmx-bumps", "drone", "cat-girl", "drift-chicane", "gold-fish", "schoolgirls")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10I: marginal vs conditional residual coding under matched motion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--codec", type=Path, default=DEFAULT_CODEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--quant-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--arms", nargs="+", default=["intra", "m10h", "m10i"])
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


@torch.no_grad()
def calibrate_conditional_residual(m10i, mc, model, codec, sequences, calibration, *,
                                   bits: int, mode: str, gop_size: int, block_size: int,
                                   search_range: int, device, max_frames: int):
    """Fit the residual grid and tables to `w`, on TRAIN sequences only.

    Runs the conditional coder closed-loop so the grid sees exactly the
    distribution it will code - not the raw residual, which is a different
    tensor once the analysis transform is trained.
    """
    model.eval()
    codec.eval()
    coded_samples = []
    seen = 0
    with mc.deterministic_kernels():
        for sequence in sequences:
            frames = sequence.load_frames()
            types = mc.gop_frame_types(frames.shape[0], gop_size)
            latent_shape = None
            previous = None
            for index in range(frames.shape[0]):
                if seen >= max_frames:
                    break
                frame = frames[index:index + 1].to(device)
                latent = model.encode(frame)
                if latent_shape is None:
                    latent_shape = tuple(latent.shape[1:])
                if types[index] == mc.FRAME_TYPE_I:
                    payload, _ = encode_latent_to_payload(
                        latent, params=calibration["intra_params"],
                        entropy_model=calibration["intra_entropy_model"])
                    decoded, _ = decode_payload_to_latent(
                        payload, entropy_model=calibration["intra_entropy_model"],
                        params=calibration["intra_params"], shape=latent_shape)
                    previous = model.decode(decoded.to(device))
                elif previous is not None:
                    motion = mc.estimate_block_motion(
                        previous, frame, block_size=block_size, search_range=search_range)
                    warped = mc.warp_blocks(previous, motion, block_size=block_size)
                    reference = model.encode(warped)
                    coded = codec.encode_residual(latent - reference, reference)
                    coded_samples.append(coded.cpu())
                    previous = model.decode(latent)
                seen += 1
            if seen >= max_frames:
                break

    stack = torch.cat(coded_samples, dim=0)
    params = calibrate_quantization_params(stack, bits=bits, mode=mode)
    symbols = np.stack([
        latent_to_symbols(stack[i:i + 1], params).reshape(stack.shape[1], -1)
        for i in range(stack.shape[0])])
    entropy_model = EmpiricalEntropyModel.from_symbols(
        symbols, bits=bits, num_tables=stack.shape[1])
    return params, entropy_model, int(stack.shape[0])


@torch.no_grad()
def run_arm(m10i, mc, model, codec, sequences, calibration, *, arm: str, gop: int, bits: int,
            block_size: int, search_range: int, stream_dir: Path) -> dict[str, Any]:
    rows = []
    for sequence in sequences:
        frames = sequence.load_frames()
        path = stream_dir / f"{arm}_{bits}bit_{sequence.sequence_id}.nvct"
        common = dict(
            intra_params=calibration["intra_params"],
            intra_entropy_model=calibration["intra_entropy_model"],
            residual_params=calibration["residual_params"],
            residual_entropy_model=calibration["residual_entropy_model"],
            motion_entropy_model=calibration["motion_entropy_model"],
            gop_size=gop, block_size=block_size, search_range=search_range)

        if arm == "m10i":
            encoded = m10i.encode_sequence_conditional(mc, model, codec, frames, path, **common)
            _, decoded_latents = m10i.decode_sequence_conditional(
                mc, model, codec, path,
                intra_entropy_model=calibration["intra_entropy_model"],
                residual_entropy_model=calibration["residual_entropy_model"],
                motion_entropy_model=calibration["motion_entropy_model"],
                return_latents=True)
        else:
            encoded = mc.encode_sequence(model, frames, path, mode="prev" if arm == "intra"
                                         else "mc", **common)
            _, decoded_latents = mc.decode_sequence(
                model, path,
                intra_entropy_model=calibration["intra_entropy_model"],
                residual_entropy_model=calibration["residual_entropy_model"],
                motion_entropy_model=calibration["motion_entropy_model"],
                return_latents=True)

        symmetric = torch.equal(encoded["encoder_latents"], decoded_latents)
        recon = encoded["encoder_reconstructions"]
        mse = torch.mean((recon - frames) ** 2).item()
        psnr = float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)
        quality = float(msssim(recon.clamp(0, 1), frames).mean())
        total_bytes = encoded["container_bytes"]
        rows.append({
            "sequence": sequence.sequence_id, "frames": encoded["frame_count"],
            "i_frames": encoded["i_frames"], "p_frames": encoded["p_frames"],
            "motion_bytes": encoded["motion_bytes"],
            "residual_bytes": encoded["residual_bytes"],
            "i_frame_residual_bytes": encoded["i_frame_residual_bytes"],
            "p_frame_residual_bytes": encoded["p_frame_residual_bytes"],
            "container_overhead_bytes": encoded["container_overhead_bytes"],
            "container_bytes": total_bytes, "total_pixels": sequence.total_pixels,
            "raw_rgb_bytes": sequence.raw_rgb_bytes(),
            "stream_bpp": total_bytes * 8 / sequence.total_pixels,
            "compression_ratio": sequence.raw_rgb_bytes() / total_bytes,
            "mean_psnr_db": psnr, "mean_msssim": quality,
            "encoder_decoder_latents_identical": symmetric,
        })

    total_bytes = sum(r["container_bytes"] for r in rows)
    total_pixels = sum(r["total_pixels"] for r in rows)
    p_frames = sum(r["p_frames"] for r in rows)
    return {
        "arm": arm, "gop": gop, "bits": bits, "sequences": rows,
        "total_frames": sum(r["frames"] for r in rows),
        "total_i_frames": sum(r["i_frames"] for r in rows), "total_p_frames": p_frames,
        "total_motion_bytes": sum(r["motion_bytes"] for r in rows),
        "total_residual_bytes": sum(r["residual_bytes"] for r in rows),
        "total_i_frame_residual_bytes": sum(r["i_frame_residual_bytes"] for r in rows),
        "total_p_frame_residual_bytes": sum(r["p_frame_residual_bytes"] for r in rows),
        "total_container_overhead_bytes": sum(r["container_overhead_bytes"] for r in rows),
        "total_container_bytes": total_bytes,
        "byte_accounting_closes": sum(
            r["motion_bytes"] + r["residual_bytes"] + r["container_overhead_bytes"]
            for r in rows) == total_bytes,
        "total_pixels": total_pixels,
        "stream_bpp": total_bytes * 8 / total_pixels,
        "compression_ratio": sum(r["raw_rgb_bytes"] for r in rows) / total_bytes,
        "mean_psnr_db": statistics.fmean(r["mean_psnr_db"] for r in rows),
        "mean_msssim": statistics.fmean(r["mean_msssim"] for r in rows),
        "motion_bits_per_p_frame": (sum(r["motion_bytes"] for r in rows) * 8 / p_frames
                                    if p_frames else 0.0),
        "residual_bits_per_p_frame": (sum(r["p_frame_residual_bytes"] for r in rows) * 8 / p_frames
                                      if p_frames else 0.0),
        "total_bits_per_p_frame": ((sum(r["motion_bytes"] for r in rows)
                                    + sum(r["p_frame_residual_bytes"] for r in rows)) * 8 / p_frames
                                   if p_frames else 0.0),
        "all_encoder_decoder_symmetric": all(r["encoder_decoder_latents_identical"] for r in rows),
    }


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1
    if "m10i" in args.arms and not args.codec.is_file():
        print(f"[ERROR] --codec not found: {args.codec}. Train it first with "
              f"scripts/m10i_conditional_residual.py --stage train", file=sys.stderr)
        return 1

    mc = _load_script("m10h_motion_compensation")
    m10i = _load_script("m10i_conditional_residual")
    m10e = _load_script("m10e_evaluate")
    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    codec = None
    codec_meta = {}
    if "m10i" in args.arms:
        codec, checkpoint = m10i.load_codec(args.codec, device=device)
        codec_meta = {
            "codec_config": checkpoint["codec_config"],
            "selection": checkpoint.get("selection"),
            "trainable_parameters": sum(p.numel() for p in codec.parameters()),
            "lambda_rate": checkpoint.get("lambda_rate"),
            "train_bits": checkpoint.get("train_bits"),
        }

    train_sequences = discover_sequences(args.manifest, split="train")
    test_sequences = discover_sequences(
        args.manifest, split="test", max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames_per_sequence)

    print("=" * 122)
    print("M10I - MARGINAL vs CONDITIONAL RESIDUAL CODING (matched motion, DAVIS test)")
    print("=" * 122)
    print(f"  frozen model : {args.checkpoint}")
    print(f"  conditional  : {args.codec if codec is not None else '(not evaluated)'}")
    if codec_meta.get("selection"):
        s = codec_meta["selection"]
        print(f"  selected     : epoch {s['selected_epoch']} by {s['selection_metric']} "
              f"(validation only); final epoch {s['final_epoch']}")
    print(f"  rate points  : {args.rate_points} bit  (data-driven; 8/6/5 are vertically aligned)")
    print(f"  sequences    : {len(test_sequences)}  frames: "
          f"{sum(s.frame_count for s in test_sequences)}")

    stream_dir = args.output_dir / "benchmark_streams"
    results: dict[tuple[str, int], dict[str, Any]] = {}
    provenance: dict[str, Any] = {}

    for bits in args.rate_points:
        base = mc.calibrate_grids(
            model, train_sequences, bits=bits, mode=args.quant_mode, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            reference_mode="mc", max_frames=args.calibration_frames)
        provenance[f"{bits}bit_marginal"] = base["provenance"]

        conditional = None
        if codec is not None:
            params, entropy_model, samples = calibrate_conditional_residual(
                m10i, mc, model, codec, train_sequences, base, bits=bits, mode=args.quant_mode,
                gop_size=args.gop, block_size=args.block_size, search_range=args.search_range,
                device=device, max_frames=args.calibration_frames)
            conditional = dict(base)
            conditional["residual_params"] = params
            conditional["residual_entropy_model"] = entropy_model
            provenance[f"{bits}bit_conditional"] = {
                **base["provenance"], "conditional_residual_samples": samples,
                "conditional_residual_entropy_model_id": entropy_model.model_id().hex(),
                "note": ("The conditional arm's residual grid is fitted to w = f_a(r, z_ref), the "
                         "tensor it actually codes, on the TRAIN split only."),
            }

        for arm in args.arms:
            print(f"\n  running {arm:<5} @ {bits}-bit ...", flush=True)
            results[(arm, bits)] = run_arm(
                m10i, mc, model, codec, test_sequences,
                conditional if arm == "m10i" else base,
                arm=arm, gop=1 if arm == "intra" else args.gop, bits=bits,
                block_size=args.block_size, search_range=args.search_range,
                stream_dir=stream_dir)

    print()
    print("=" * 122)
    print("FULL BYTE ACCOUNTING")
    print("=" * 122)
    print(f"{'arm':<6} {'bits':>5} {'I bytes':>12} {'P resid':>12} {'motion':>10} "
          f"{'overhead':>9} {'TOTAL':>12} {'BPP':>8} {'PSNR':>8} {'MS-SSIM':>8} "
          f"{'resid b/P':>10} {'mot b/P':>8}")
    for bits in args.rate_points:
        for arm in args.arms:
            a = results.get((arm, bits))
            if a is None:
                continue
            print(f"{arm:<6} {bits:>5} {a['total_i_frame_residual_bytes']:>12,} "
                  f"{a['total_p_frame_residual_bytes']:>12,} {a['total_motion_bytes']:>10,} "
                  f"{a['total_container_overhead_bytes']:>9,} {a['total_container_bytes']:>12,} "
                  f"{a['stream_bpp']:>8.4f} {a['mean_psnr_db']:>8.3f} {a['mean_msssim']:>8.4f} "
                  f"{a['residual_bits_per_p_frame']:>10,.0f} {a['motion_bits_per_p_frame']:>8,.0f}")

    # --- the ablation: motion must be identical -------------------------------
    print()
    print("=" * 122)
    print("CONDITIONING ABLATION - m10i vs m10h, motion held identical")
    print("=" * 122)
    print(f"{'bits':>5} {'d motion':>12} {'d residual':>14} {'d total':>14} {'d BPP':>10} "
          f"{'d PSNR':>9} {'d MS-SSIM':>11}  motion identical?")
    ablation = []
    for bits in args.rate_points:
        a, b = results.get(("m10h", bits)), results.get(("m10i", bits))
        if a is None or b is None:
            continue
        d_motion = b["total_motion_bytes"] - a["total_motion_bytes"]
        d_residual = b["total_residual_bytes"] - a["total_residual_bytes"]
        d_total = b["total_container_bytes"] - a["total_container_bytes"]
        identical = d_motion == 0
        entry = {
            "bits": bits, "delta_motion_bytes": d_motion,
            "delta_residual_bytes": d_residual,
            "delta_residual_percent": d_residual / a["total_residual_bytes"] * 100,
            "delta_total_bytes": d_total,
            "delta_bpp_percent": (b["stream_bpp"] - a["stream_bpp"]) / a["stream_bpp"] * 100,
            "delta_psnr_db": b["mean_psnr_db"] - a["mean_psnr_db"],
            "delta_msssim": b["mean_msssim"] - a["mean_msssim"],
            "motion_identical": identical,
        }
        ablation.append(entry)
        print(f"{bits:>5} {d_motion:>+12,} {d_residual:>+14,} {d_total:>+14,} "
              f"{entry['delta_bpp_percent']:>+9.2f}% {entry['delta_psnr_db']:>+9.3f} "
              f"{entry['delta_msssim']:>+11.4f}  {'YES' if identical else 'NO - CONFOUNDED'}")
    print()
    print("  Success signature: motion unchanged, residual DOWN, quality held or improved.")
    if ablation and all(e["motion_identical"] for e in ablation):
        print("  Motion is byte-identical at every rate point, so every difference above is")
        print("  attributable to the residual model - the experiment is not confounded.")

    # --- RD curves and BD-rate -------------------------------------------------
    def curve(arm, metric="mean_psnr_db"):
        return [(results[(arm, b)]["stream_bpp"], results[(arm, b)][metric])
                for b in args.rate_points if (arm, b) in results]

    print()
    print("=" * 122)
    print("BD-RATE over three rate points")
    print("=" * 122)
    bd_rows = []
    pairs = [("m10h", "intra"), ("m10i", "intra"), ("m10i", "m10h")]
    print(f"{'comparison':<22} {'PSNR BD-rate':>16} {'MS-SSIM BD-rate':>18}")
    for test, base in pairs:
        if test not in args.arms or base not in args.arms:
            continue
        psnr_bd = m10e._bd_rate_linear(curve(base), curve(test))
        ms_bd = m10e._bd_rate_linear(curve(base, "mean_msssim"), curve(test, "mean_msssim"))
        bd_rows.append({"test": test, "base": base,
                        "bd_rate_psnr": psnr_bd, "bd_rate_msssim": ms_bd})
        print(f"{test + ' vs ' + base:<22} "
              f"{(f'{psnr_bd:+.2f}%' if psnr_bd is not None else 'n/a'):>16} "
              f"{(f'{ms_bd:+.2f}%' if ms_bd is not None else 'n/a'):>18}")
    print()
    for arm in args.arms:
        points = curve(arm)
        span = max(p[1] for p in points) - min(p[1] for p in points)
        print(f"  {arm:<6} quality span across its three points: {span:.3f} dB")
    print("  BD-rate is 'n/a' when two curves share no overlapping quality range;")
    print("  that is reported rather than extrapolated beyond what was measured.")

    # --- per sequence ----------------------------------------------------------
    reference_bits = args.rate_points[1] if len(args.rate_points) > 1 else args.rate_points[0]
    print()
    print("=" * 122)
    print(f"PER-SEQUENCE at the {reference_bits}-bit rate point "
          f"(* = sequence M10G/M10H flagged as motion-sensitive)")
    print("=" * 122)
    per_sequence = []
    if all((arm, reference_bits) in results for arm in args.arms):
        print(f"{'sequence':<16} " + " ".join(f"{a + ' BPP':>11} {a + ' PSNR':>11}"
                                              for a in args.arms) +
              f" {'d resid %':>10} {'d PSNR':>8}")
        for index, sequence in enumerate(test_sequences):
            cells, record = [], {"sequence": sequence.sequence_id,
                                 "watched": sequence.sequence_id in WATCH}
            for arm in args.arms:
                row = results[(arm, reference_bits)]["sequences"][index]
                cells.append(f"{row['stream_bpp']:>11.4f} {row['mean_psnr_db']:>11.2f}")
                record[f"{arm}_bpp"] = row["stream_bpp"]
                record[f"{arm}_psnr"] = row["mean_psnr_db"]
                record[f"{arm}_residual_bytes"] = row["residual_bytes"]
                record[f"{arm}_motion_bytes"] = row["motion_bytes"]
            if "m10h" in args.arms and "m10i" in args.arms:
                record["delta_residual_percent"] = (
                    (record["m10i_residual_bytes"] - record["m10h_residual_bytes"])
                    / record["m10h_residual_bytes"] * 100)
                record["delta_psnr_db"] = record["m10i_psnr"] - record["m10h_psnr"]
                cells.append(f"{record['delta_residual_percent']:>+9.2f}% "
                             f"{record['delta_psnr_db']:>+8.2f}")
            per_sequence.append(record)
            print(f"{'*' if record['watched'] else ' '}{sequence.sequence_id:<15} "
                  + " ".join(cells))

    symmetry = {f"{arm}@{bits}": results[(arm, bits)]["all_encoder_decoder_symmetric"]
                for bits in args.rate_points for arm in args.arms if (arm, bits) in results}
    closes = all(results[k]["byte_accounting_closes"] for k in results)
    print()
    print(f"  byte accounting closes everywhere: {closes}")
    print(f"  encoder/decoder reference symmetry: "
          f"{'PASS' if all(symmetry.values()) else 'FAIL -> ' + str(symmetry)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": "M10I conditional residual benchmark",
        "frozen_lambda": FROZEN_LAMBDA, "checkpoint": str(args.checkpoint),
        "conditional_codec": str(args.codec), "codec_meta": codec_meta,
        "rate_points_bits": list(args.rate_points),
        "rate_point_rationale": (
            "Data-driven. A bit-depth probe found the M10H MC RD curve is vertical above 5-bit "
            "(8 -> 5 bit halves rate for 0.08 dB), so 8/6/5 could not support BD-rate. 5/4/3 "
            "spans 1.42 dB over a 2.1x rate range."),
        "calibration_provenance": provenance,
        "arms": {f"{k[0]}@{k[1]}bit": v for k, v in results.items()},
        "conditioning_ablation": ablation,
        "bd_rate": bd_rows,
        "per_sequence_at_reference_rate_point": per_sequence,
        "encoder_decoder_symmetry": symmetry,
        "byte_accounting_closes": closes,
    }
    path = args.output_dir / "conditional_benchmark.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
