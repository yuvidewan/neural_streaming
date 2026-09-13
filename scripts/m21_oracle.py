"""M21 Phase 2 - the oracle diagnostic: is causal reference refinement even
plausible before any candidate is built?

For every VAL-B P-frame this measures, side by side:

  A  deployed   `previous` = model.decode(reference + dequant(residual))
  B  oracle     `model.decode(model.encode(raw previous frame))` - M16/M17's own
                idealization, reused unmodified. NOT causal; an upper bound.
  C  every pre-registered causal candidate from `m21_refinement.CANDIDATES`.

and reports, for each, the four quantities the milestone insists on keeping
separate because M18 proved they do not move together:

  1 pixel MSE vs the oracle reference
  2 latent MSE vs the oracle reference latent
  3 motion SAD against the current target
  4 residual magnitude statistics
  5 residual symbol agreement with the oracle-reference pipeline
  6 M11-G16 predicted cross-entropy (ideal bits)
  7 ACTUAL coded residual bytes from the deployed arithmetic coder

The point of running this BEFORE the closed-loop sweep is to know whether any
candidate moves the reference *toward* the oracle at all. A candidate that
increases pixel MSE against the oracle cannot plausibly be recovering M17's gap,
whatever its byte count does; and a candidate that decreases pixel MSE but not
coded bytes is exactly the M18 failure mode, which this milestone must be able
to recognise rather than re-discover.

This is an OPEN-LOOP probe: every candidate is measured against the SAME
deployed chain (the real arm alone advances `previous`), so the candidates are
compared on identical inputs. The closed-loop, drift-carrying measurement is
`m21_sweep.py`'s job.

Run:
  ./.venv/Scripts/python.exe scripts/m21_oracle.py
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


class Accumulator:
    """Running sums for one reference variant across every VAL-B P-frame."""

    def __init__(self) -> None:
        self.frames = 0
        self.pixel_se_vs_oracle = 0.0
        self.pixel_elements = 0
        self.latent_se_vs_oracle = 0.0
        self.latent_elements = 0
        self.motion_sad = 0.0
        self.residual_abs = 0.0
        self.residual_sq = 0.0
        self.symbols_equal_oracle = 0
        self.symbols_total = 0
        self.ideal_bits = 0.0
        self.coded_bytes = 0
        self.boundary_bytes = 0
        self.ordinary_bytes = 0
        self.boundary_frames = 0

    def add(self, *, pixel_se, pixel_n, latent_se, latent_n, sad, residual, symbols,
            oracle_symbols, ideal, payload_bytes, is_boundary) -> None:
        self.frames += 1
        self.pixel_se_vs_oracle += pixel_se
        self.pixel_elements += pixel_n
        self.latent_se_vs_oracle += latent_se
        self.latent_elements += latent_n
        self.motion_sad += sad
        self.residual_abs += float(np.abs(residual).sum())
        self.residual_sq += float((residual ** 2).sum())
        self.symbols_equal_oracle += int((symbols == oracle_symbols).sum())
        self.symbols_total += int(symbols.size)
        self.ideal_bits += ideal
        self.coded_bytes += payload_bytes
        if is_boundary:
            self.boundary_bytes += payload_bytes
            self.boundary_frames += 1
        else:
            self.ordinary_bytes += payload_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_frames": self.frames,
            "pixel_mse_vs_oracle": self.pixel_se_vs_oracle / max(self.pixel_elements, 1),
            "latent_mse_vs_oracle": self.latent_se_vs_oracle / max(self.latent_elements, 1),
            "mean_motion_sad": self.motion_sad / max(self.frames, 1),
            "residual_mean_abs": self.residual_abs / max(self.latent_elements, 1),
            "residual_rms": (self.residual_sq / max(self.latent_elements, 1)) ** 0.5,
            "symbol_agreement_with_oracle": self.symbols_equal_oracle
            / max(self.symbols_total, 1),
            "ideal_bits": self.ideal_bits,
            "coded_residual_bytes": self.coded_bytes,
            "boundary_bytes": self.boundary_bytes, "ordinary_bytes": self.ordinary_bytes,
            "boundary_frames": self.boundary_frames,
        }


@torch.no_grad()
def diagnose(mc, m13, m21, model, rig, sequence, candidates, accumulators, *, bits, gop_size,
             block_size, search_range, device) -> int:
    """One VAL-B sequence. The DEPLOYED arm alone advances `previous`; every
    other variant is a read-only side channel measured on the same inputs, the
    discipline M16-M20 all used."""
    spec = rig["spec"]
    residual_params, zero = rig["residual_params"], rig["zero"]
    frames = sequence.load_frames()
    types = mc.gop_frame_types(frames.shape[0], gop_size)
    previous = None
    previous_raw = None
    p_frames = 0

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
            oracle_previous = model.decode(model.encode(previous_raw))

            def _variant(reference_frame, *, latent_refinement=None):
                motion = mc.estimate_block_motion(reference_frame, frame,
                                                  block_size=block_size,
                                                  search_range=search_range)
                warped = mc.warp_blocks(reference_frame, motion, block_size=block_size)
                reference_latent = model.encode(warped)
                if latent_refinement is not None:
                    reference_latent = latent_refinement(reference_latent, model)
                delta = latent - reference_latent
                symbols = latent_to_symbols(delta, residual_params)
                payload, ideal = m13.encode_frame_recalibrated(
                    spec["model"], spec["assign_codebook"], spec["coding_codebook"],
                    reference_latent, symbols.reshape(latent_shape), spec["zero"], bits=bits)
                sad = float((frame - warped).abs().mean())
                return {"reference_frame": reference_frame, "reference_latent": reference_latent,
                        "delta": delta, "symbols": symbols, "payload": payload, "ideal": ideal,
                        "sad": sad}

            oracle = _variant(oracle_previous)
            oracle_symbols = oracle["symbols"]
            oracle_latent = oracle["reference_latent"]

            def _record(key, refined_frame, data):
                pixel_error = refined_frame - oracle_previous
                latent_error = data["reference_latent"] - oracle_latent
                accumulators[key].add(
                    pixel_se=float((pixel_error ** 2).sum()),
                    pixel_n=int(pixel_error.numel()),
                    latent_se=float((latent_error ** 2).sum()),
                    latent_n=int(latent_error.numel()),
                    sad=data["sad"],
                    residual=data["delta"].detach().cpu().numpy(),
                    symbols=data["symbols"], oracle_symbols=oracle_symbols,
                    ideal=data["ideal"], payload_bytes=len(data["payload"]),
                    is_boundary=is_boundary)

            _record("__oracle__", oracle_previous, oracle)
            for refinement in candidates:
                if refinement.domain == "pixel":
                    refined = refinement(previous, model)
                    data = _variant(refined)
                elif refinement.domain == "latent":
                    refined = previous
                    data = _variant(previous, latent_refinement=refinement)
                else:
                    refined = previous
                    data = _variant(previous)
                _record(refinement.name, refined, data)
                if refinement.name == "identity":
                    deployed = data

            # Only the deployed arm advances the chain.
            reconstructed_latent = deployed["reference_latent"] + symbols_to_latent(
                deployed["symbols"], latent_shape, residual_params).to(device)
            previous = model.decode(reconstructed_latent)
            previous_raw = frame
            p_frames += 1
    return p_frames


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M21 Phase 2: oracle diagnostic over the pre-registered candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt"))
    parser.add_argument("--m11-dir", type=Path, default=Path("outputs/m11_autoregressive_entropy"))
    parser.add_argument("--m10k-dir", type=Path, default=Path("outputs/m10k_learned_entropy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m21_reference_refinement"))
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
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

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
    candidates = list(m21.CANDIDATES)

    print("=" * 132)
    print("M21 PHASE 2 - ORACLE DIAGNOSTIC (open loop; VAL-B, held out; never TEST)")
    print("=" * 132)
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}")
    print(f"  candidates: {len(candidates)} pre-registered + the non-causal oracle upper bound",
          flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "M21 Phase 2",
        "val_b_sequence_ids": [s.sequence_id for s in val_b],
        "candidate_definitions": [c.to_dict() for c in candidates],
        "rate_points": [],
    }

    for bits in args.rate_points:
        started = time.perf_counter()
        rig = m21.prepare_rate_point(
            model, bits=bits, manifest=args.manifest, checkpoint=args.checkpoint,
            m11_dir=args.m11_dir, m10k_dir=args.m10k_dir, device=device, cache_dir=cache_dir,
            train_full=train_full, motion_train=motion_train,
            calibration_frames=args.calibration_frames, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            motion_cache=args.output_dir / "m21_deployed_motion_table.json")
        accumulators = {"__oracle__": Accumulator()}
        accumulators.update({c.name: Accumulator() for c in candidates})
        p_frames = 0
        for sequence in val_b:
            p_frames += diagnose(mc, m13, m21, model, rig, sequence, candidates, accumulators,
                                 bits=bits, gop_size=args.gop, block_size=args.block_size,
                                 search_range=args.search_range, device=device)

        rows = {name: acc.to_dict() for name, acc in accumulators.items()}
        deployed = rows["identity"]
        oracle = rows["__oracle__"]
        base_bytes = deployed["coded_residual_bytes"]
        gap_bytes = base_bytes - oracle["coded_residual_bytes"]

        print(f"\n  ---- {bits}-bit ----  {p_frames} P-frames, "
              f"{time.perf_counter() - started:.1f}s, residual={rig['residual_identity']}")
        print(f"    deployed coded residual bytes : {base_bytes:,}")
        print(f"    ORACLE coded residual bytes   : {oracle['coded_residual_bytes']:,}  "
              f"(gap {gap_bytes:+,} = {gap_bytes / base_bytes * 100:+.3f}% - M17's bound)")
        header = (f"    {'candidate':>22} {'pixMSE/oracle':>14} {'latMSE/oracle':>14} "
                  f"{'SAD':>8} {'|res|':>8} {'sym agree':>10} {'ideal Mbit':>11} "
                  f"{'bytes':>10} {'d bytes':>10} {'d %':>8}")
        print(header)
        for name in ["__oracle__"] + [c.name for c in candidates]:
            row = rows[name]
            delta = row["coded_residual_bytes"] - base_bytes
            print(f"    {name:>22} {row['pixel_mse_vs_oracle']:>14.6e} "
                  f"{row['latent_mse_vs_oracle']:>14.6e} {row['mean_motion_sad']:>8.5f} "
                  f"{row['residual_mean_abs']:>8.4f} "
                  f"{row['symbol_agreement_with_oracle'] * 100:>9.2f}% "
                  f"{row['ideal_bits'] / 1e6:>11.4f} {row['coded_residual_bytes']:>10,} "
                  f"{delta:>+10,} {-delta / base_bytes * 100:>+8.3f}")
        print(flush=True)

        report["rate_points"].append({
            "bits": bits, "p_frames": p_frames,
            "residual_identity": rig["residual_identity"],
            "deployed_coded_residual_bytes": base_bytes,
            "oracle_coded_residual_bytes": oracle["coded_residual_bytes"],
            "oracle_gap_bytes": gap_bytes,
            "oracle_gap_percent": gap_bytes / base_bytes * 100,
            "variants": rows,
        })

    path = args.output_dir / "m21_oracle.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
