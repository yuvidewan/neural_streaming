"""M22 Phase 13 - the mechanism test: how much of M17's oracle gap does a
candidate actually close, and at which level?

For a refitted candidate (loaded from its saved checkpoint, so the stack that is
measured is provably the stack that was fitted), this runs the REAL and ORACLE
reference arms side by side and reports the gap at four separate levels:

  1 latent      MSE of the reference latent against the oracle reference latent
  2 symbol      agreement between the real-arm and oracle-arm residual symbols
  3 entropy     G16 ideal bits
  4 coded       ACTUAL arithmetic-coded residual bytes

and, for each, the fraction of the DEPLOYED grid's own gap that the candidate
closes. The four are kept apart because M18 established they do not move
together, and M21 found a candidate that improved bytes while making pixel error
worse - so "closed the gap" has to be said at a specific level or not at all.

A candidate that improves reconstruction while leaving symbol behaviour
unchanged is exactly what this is built to detect and say out loud.

Run:
  ./.venv/Scripts/python.exe scripts/m22_mechanism.py --variant broad_p001 --bits 3
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

from nvc.compression.codec import (
    decode_payload_to_latent, encode_latent_to_payload, latent_to_symbols, symbols_to_latent,
)
from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_refit_checkpoint(path: Path, *, device, m22, ma, ml, cx):
    """Rebuild a refitted stack from disk, verifying the recorded digest first.

    The digest check is the point: a checkpoint must never be evaluated under a
    different quantizer than the one it was fitted to, and the recorded
    `quantizer_identity` is what makes that checkable rather than assumed.
    """
    import hashlib

    from nvc.compression.quantization import QuantizationParams

    record_path = path.with_suffix(".provenance.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise ValueError(f"{path.name} hashes to {digest[:16]}, recorded {record['sha256'][:16]}")

    payload = torch.load(path, weights_only=False)
    model11 = ma.ChannelContextEntropyModel(**payload["model_config"]).to(device)
    model11.load_state_dict(payload["model_state_dict"])
    model11.eval()
    assign_codebook = ml.SharedCodebook.from_dict(payload["assign_codebook"])
    coding_codebook = ml.SharedCodebook.from_dict(payload["coding_codebook"])
    residual_params = QuantizationParams(
        scale=payload["residual_scale"], zero_point=payload["residual_zero_point"],
        bits=payload["bits"], mode=payload["quantizer_identity"]["mode"])
    if m22.grid_signature(residual_params) != payload["quantizer_identity"]["grid_signature"]:
        raise ValueError("the checkpoint's residual grid does not match its recorded signature")
    channels = int(residual_params.scale.numel())
    zero = torch.from_numpy(cx.zero_symbols(residual_params, channels))
    return {"spec": {"model": model11, "zero": zero, "assign_codebook": assign_codebook,
                     "coding_codebook": coding_codebook,
                     "identity": bytes.fromhex(payload["entropy_identities"]["residual"])},
            "residual_params": residual_params, "record": record, "payload_keys": list(payload)}


class GapAccumulator:
    """Real vs oracle at four levels, over every VAL-B P-frame, split by GOP
    position because M16-M21 all found position 1 anomalous."""

    def __init__(self) -> None:
        self.groups = {name: {
            "frames": 0, "latent_se": 0.0, "latent_n": 0, "agree": 0, "symbols": 0,
            "ideal_real": 0.0, "ideal_oracle": 0.0, "bytes_real": 0, "bytes_oracle": 0}
            for name in ("all", "boundary", "ordinary")}

    def add(self, *, is_boundary, latent_se, latent_n, agree, symbols, ideal_real, ideal_oracle,
            bytes_real, bytes_oracle) -> None:
        for name in ("all", "boundary" if is_boundary else "ordinary"):
            bucket = self.groups[name]
            bucket["frames"] += 1
            bucket["latent_se"] += latent_se
            bucket["latent_n"] += latent_n
            bucket["agree"] += agree
            bucket["symbols"] += symbols
            bucket["ideal_real"] += ideal_real
            bucket["ideal_oracle"] += ideal_oracle
            bucket["bytes_real"] += bytes_real
            bucket["bytes_oracle"] += bytes_oracle

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for name, bucket in self.groups.items():
            real = max(bucket["bytes_real"], 1)
            out[name] = {
                "p_frames": bucket["frames"],
                "latent_mse_vs_oracle": bucket["latent_se"] / max(bucket["latent_n"], 1),
                "symbol_agreement": bucket["agree"] / max(bucket["symbols"], 1),
                "ideal_bits_real": bucket["ideal_real"],
                "ideal_bits_oracle": bucket["ideal_oracle"],
                "ideal_bits_gap_percent": ((bucket["ideal_real"] - bucket["ideal_oracle"])
                                           / max(bucket["ideal_real"], 1e-9) * 100),
                "coded_bytes_real": bucket["bytes_real"],
                "coded_bytes_oracle": bucket["bytes_oracle"],
                "coded_gap_bytes": bucket["bytes_real"] - bucket["bytes_oracle"],
                "coded_gap_percent": (bucket["bytes_real"] - bucket["bytes_oracle"]) / real * 100,
            }
        return out


@torch.no_grad()
def measure(mc, m13, model, spec, residual_params, intra_params, intra_entropy_model, sequence,
            accumulator, *, bits, gop_size, block_size, search_range, device) -> int:
    """The real arm advances the chain; the oracle arm is a read-only side
    channel, exactly as M16-M21 ran it."""
    frames = sequence.load_frames()
    types = mc.gop_frame_types(frames.shape[0], gop_size)
    previous, previous_raw = None, None
    p_frames = 0

    with mc.deterministic_kernels():
        for index in range(frames.shape[0]):
            frame = frames[index:index + 1].to(device)
            latent = model.encode(frame)
            latent_shape = tuple(latent.shape[1:])
            if types[index] == mc.FRAME_TYPE_I:
                payload, _ = encode_latent_to_payload(
                    latent, params=intra_params, entropy_model=intra_entropy_model)
                decoded, _ = decode_payload_to_latent(
                    payload, entropy_model=intra_entropy_model, params=intra_params,
                    shape=latent_shape)
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
                symbols = latent_to_symbols(latent - reference_latent, residual_params)
                payload, ideal = m13.encode_frame_recalibrated(
                    spec["model"], spec["assign_codebook"], spec["coding_codebook"],
                    reference_latent, symbols.reshape(latent_shape), spec["zero"], bits=bits)
                return reference_latent, symbols, payload, ideal

            reference_real, symbols_real, payload_real, ideal_real = _arm(previous)
            reference_oracle, symbols_oracle, payload_oracle, ideal_oracle = _arm(
                model.decode(model.encode(previous_raw)))

            error = (reference_real - reference_oracle)
            accumulator.add(
                is_boundary=is_boundary, latent_se=float((error ** 2).sum()),
                latent_n=int(error.numel()),
                agree=int((symbols_real == symbols_oracle).sum()),
                symbols=int(symbols_real.size), ideal_real=ideal_real,
                ideal_oracle=ideal_oracle, bytes_real=len(payload_real),
                bytes_oracle=len(payload_oracle))

            previous = model.decode(reference_real + symbols_to_latent(
                symbols_real, latent_shape, residual_params).to(device))
            previous_raw = frame
            p_frames += 1
    return p_frames


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M22 Phase 13: mechanism decomposition for a refitted candidate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--bits", type=int, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt"))
    parser.add_argument("--m11-dir", type=Path, default=Path("outputs/m11_autoregressive_entropy"))
    parser.add_argument("--m10k-dir", type=Path, default=Path("outputs/m10k_learned_entropy"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/m22_residual_freeze_lift"))
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--val-sequences", type=int, default=4)
    parser.add_argument("--train-frames-per-sequence", type=int, default=8)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    mc = _load_script("m10h_motion_compensation")
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    cx = _load_script("m11_causal_context")
    m13 = _load_script("m13_recalibration")
    md = _load_script("m11_data")
    m21 = _load_script("m21_refinement")
    m22 = _load_script("m22_residual")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    val_b = m21.val_b_sequences(args.manifest, count=args.val_sequences)
    train_full = discover_sequences(args.manifest, split="train")
    motion_train = m21.broad_train_sequences(
        args.manifest, frames_per_sequence=args.train_frames_per_sequence)

    print("=" * 128)
    print(f"M22 PHASE 13 - MECHANISM DECOMPOSITION, candidate '{args.variant}'")
    print("=" * 128)
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}", flush=True)

    report: dict[str, Any] = {"phase": "M22 Phase 13", "variant": args.variant,
                              "val_b_sequence_ids": [s.sequence_id for s in val_b],
                              "rate_points": []}

    for bits in args.bits:
        started = time.perf_counter()
        deployed_rig = m21.prepare_rate_point(
            model, bits=bits, manifest=args.manifest, checkpoint=args.checkpoint,
            m11_dir=args.m11_dir, m10k_dir=args.m10k_dir, device=device, cache_dir=cache_dir,
            train_full=train_full, motion_train=motion_train,
            calibration_frames=args.calibration_frames, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            motion_cache=args.output_dir / "m22_deployed_motion_table.json")

        arms = {"deployed": (deployed_rig["spec"], deployed_rig["residual_params"], None)}
        path = args.output_dir / "checkpoints" / f"m22_{args.variant}_{bits}bit.pt"
        if not path.is_file():
            print(f"[ERROR] no refit checkpoint at {path}", file=sys.stderr)
            return 1
        loaded = load_refit_checkpoint(path, device=device, m22=m22, ma=ma, ml=ml, cx=cx)
        arms[args.variant] = (loaded["spec"], loaded["residual_params"], loaded["record"])

        results = {}
        for name, (spec, params, record) in arms.items():
            accumulator = GapAccumulator()
            p_frames = 0
            for sequence in val_b:
                p_frames += measure(
                    mc, m13, model, spec, params, deployed_rig["intra_params"],
                    deployed_rig["intra_entropy_model"], sequence, accumulator, bits=bits,
                    gop_size=args.gop, block_size=args.block_size,
                    search_range=args.search_range, device=device)
            results[name] = {"p_frames": p_frames, "gap": accumulator.to_dict(),
                             "provenance": record}

        deployed = results["deployed"]["gap"]["all"]
        candidate = results[args.variant]["gap"]["all"]

        def _closed(field, lower_is_better=True):
            base, other = deployed[field], candidate[field]
            if base == 0:
                return 0.0
            return (base - other) / base * 100 if lower_is_better else (other - base) / base * 100

        closure = {
            "latent_mse_gap_closed_percent": _closed("latent_mse_vs_oracle"),
            "symbol_disagreement_closed_percent": (
                ((1 - deployed["symbol_agreement"]) - (1 - candidate["symbol_agreement"]))
                / max(1 - deployed["symbol_agreement"], 1e-9) * 100),
            "ideal_bits_gap_closed_percent": (
                (deployed["ideal_bits_real"] - deployed["ideal_bits_oracle"]
                 - (candidate["ideal_bits_real"] - candidate["ideal_bits_oracle"]))
                / max(deployed["ideal_bits_real"] - deployed["ideal_bits_oracle"], 1e-9) * 100),
            "coded_gap_closed_percent": (
                (deployed["coded_gap_bytes"] - candidate["coded_gap_bytes"])
                / max(deployed["coded_gap_bytes"], 1) * 100),
        }

        print(f"\n  ---- {bits}-bit ----  ({time.perf_counter() - started:.1f}s)")
        print(f"    {'level':>28} {'deployed':>16} {args.variant:>16} {'gap closed':>12}")
        print(f"    {'1 latent MSE vs oracle':>28} {deployed['latent_mse_vs_oracle']:>16.6e} "
              f"{candidate['latent_mse_vs_oracle']:>16.6e} "
              f"{closure['latent_mse_gap_closed_percent']:>+11.2f}%")
        print(f"    {'2 symbol agreement':>28} {deployed['symbol_agreement'] * 100:>15.2f}% "
              f"{candidate['symbol_agreement'] * 100:>15.2f}% "
              f"{closure['symbol_disagreement_closed_percent']:>+11.2f}%")
        print(f"    {'3 G16 ideal-bit gap':>28} {deployed['ideal_bits_gap_percent']:>15.4f}% "
              f"{candidate['ideal_bits_gap_percent']:>15.4f}% "
              f"{closure['ideal_bits_gap_closed_percent']:>+11.2f}%")
        print(f"    {'4 coded residual-byte gap':>28} {deployed['coded_gap_percent']:>15.4f}% "
              f"{candidate['coded_gap_percent']:>15.4f}% "
              f"{closure['coded_gap_closed_percent']:>+11.2f}%")
        print(f"    real coded residual bytes: deployed {deployed['coded_bytes_real']:,} -> "
              f"{args.variant} {candidate['coded_bytes_real']:,} "
              f"({(deployed['coded_bytes_real'] - candidate['coded_bytes_real']) / deployed['coded_bytes_real'] * 100:+.4f}%)")
        for group in ("boundary", "ordinary"):
            base, other = results["deployed"]["gap"][group], results[args.variant]["gap"][group]
            print(f"    GOP {group:<9} frames={base['p_frames']:>4}  coded gap "
                  f"{base['coded_gap_percent']:>8.4f}% -> {other['coded_gap_percent']:>8.4f}%  "
                  f"real bytes {base['coded_bytes_real']:,} -> {other['coded_bytes_real']:,}")
        print(flush=True)

        report["rate_points"].append({
            "bits": bits, "deployed": results["deployed"], "candidate": results[args.variant],
            "gap_closure": closure,
            "candidate_checkpoint": str(path).replace("\\", "/"),
        })

    path = args.output_dir / f"m22_mechanism_{args.variant}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
