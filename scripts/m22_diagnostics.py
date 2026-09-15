"""M22 Phase 2 - residual-space diagnostics: WHY is the residual symbol unstable
under decoded-reference error?

The milestone lists five candidate explanations and forbids inferring the answer
from M17/M21. This measures them, on TRAIN and VAL-B only:

  A distribution            per-channel std, MAE, RMS, dynamic range
  B per-channel variance    concentration across the 64 latent channels
  C per-channel range       and how the grid divides it
  D quantizer step          step, step/std, and whether zero is representable
  E clipping frequency      per channel and overall
  F symbol entropy          and its efficiency against the fixed alphabet
  G symbol agreement        real-reference vs oracle-reference, per bit depth
  H symbol sensitivity      to REAL decoded-reference perturbations, by size
  I level distance          |offset - round(offset)|, the distance from a value to
                            ITS OWN reconstruction level, in steps: 0 means exactly
                            on a level (maximally stable), 0.5 means exactly on a
                            decision boundary (maximally fragile)
  J level population        fraction within 0.05 / 0.10 / 0.25 / 0.50 of a level
  K level vs cost           do fragile positions carry the excess bits?

The decisive question is H/I/J/K together, and the two quantities are put in the
SAME units (steps) on purpose so they can be compared directly. If the reference
perturbation is small against a step, only positions near a decision boundary
flip, symbols move by at most +/-1, and the STEP SIZE is the lever. If the
perturbation is comparable to or larger than a step, symbols move wholesale and
no re-placement of a uniform grid can help - only a different representation
could.

Run:
  ./.venv/Scripts/python.exe scripts/m22_diagnostics.py
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

LEVEL_BANDS = (0.05, 0.10, 0.25, 0.50)
DISPLACEMENT_BINS = (0, 1, 2, 3, 5, 9, 10 ** 9)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SensitivityAccumulator:
    """Phase 2 G/H/I/J/K, accumulated over every VAL-B P-frame.

    Everything is a sum or a count so the result is order-independent and
    identical across processes.
    """

    def __init__(self, bits: int) -> None:
        self.bits = bits
        self.positions = 0
        self.changed = 0
        self.displacement_histogram = np.zeros(len(DISPLACEMENT_BINS) - 1, dtype=np.int64)
        self.excess_bits_by_displacement = np.zeros(len(DISPLACEMENT_BINS) - 1, dtype=np.float64)
        self.level_band_positions = np.zeros(len(LEVEL_BANDS), dtype=np.int64)
        self.level_band_changed = np.zeros(len(LEVEL_BANDS), dtype=np.int64)
        self.level_band_excess_bits = np.zeros(len(LEVEL_BANDS), dtype=np.float64)
        self.level_distance_sum = 0.0
        self.total_excess_bits = 0.0
        self.reference_shift_sum = 0.0
        self.reference_shift_square = 0.0
        self.decile_edges: np.ndarray | None = None
        self.decile_positions = np.zeros(10, dtype=np.int64)
        self.decile_changed = np.zeros(10, dtype=np.int64)

    def add(self, *, real_symbols, oracle_symbols, real_offset, reference_shift_steps,
            code_len_real, code_len_oracle) -> None:
        changed = real_symbols != oracle_symbols
        displacement = np.abs(real_symbols - oracle_symbols)
        excess = code_len_real - code_len_oracle
        level_distance = np.abs(real_offset - np.round(real_offset))

        self.positions += real_symbols.size
        self.changed += int(changed.sum())
        self.level_distance_sum += float(level_distance.sum())
        self.total_excess_bits += float(excess.sum())
        self.reference_shift_sum += float(np.abs(reference_shift_steps).sum())
        self.reference_shift_square += float((reference_shift_steps ** 2).sum())

        index = np.digitize(displacement, DISPLACEMENT_BINS[1:-1], right=False)
        for bucket in range(self.displacement_histogram.size):
            mask = index == bucket
            self.displacement_histogram[bucket] += int(mask.sum())
            self.excess_bits_by_displacement[bucket] += float(excess[mask].sum())

        for position, band in enumerate(LEVEL_BANDS):
            mask = level_distance < band
            self.level_band_positions[position] += int(mask.sum())
            self.level_band_changed[position] += int((mask & changed).sum())
            self.level_band_excess_bits[position] += float(excess[mask].sum())

        # Reference-shift deciles: does a bigger reference perturbation flip more
        # symbols? The edges are fixed from the FIRST frame so every frame after
        # is binned identically.
        if self.decile_edges is None:
            self.decile_edges = np.quantile(np.abs(reference_shift_steps),
                                            np.linspace(0.1, 0.9, 9))
        decile = np.digitize(np.abs(reference_shift_steps), self.decile_edges, right=False)
        for bucket in range(10):
            mask = decile == bucket
            self.decile_positions[bucket] += int(mask.sum())
            self.decile_changed[bucket] += int((mask & changed).sum())

    def to_dict(self) -> dict[str, Any]:
        positions = max(self.positions, 1)
        labels = [f"{DISPLACEMENT_BINS[i]}" if DISPLACEMENT_BINS[i + 1] - DISPLACEMENT_BINS[i] == 1
                  else f"{DISPLACEMENT_BINS[i]}-{DISPLACEMENT_BINS[i + 1] - 1}"
                  for i in range(len(DISPLACEMENT_BINS) - 1)]
        labels[-1] = f">={DISPLACEMENT_BINS[-2]}"
        total_excess = self.total_excess_bits or 1.0
        return {
            "positions": self.positions,
            "symbol_change_rate": self.changed / positions,
            "symbol_agreement": 1.0 - self.changed / positions,
            "mean_level_distance_steps": self.level_distance_sum / positions,
            "total_excess_bits": self.total_excess_bits,
            "mean_reference_shift_steps": self.reference_shift_sum / positions,
            "rms_reference_shift_steps": (self.reference_shift_square / positions) ** 0.5,
            "symbol_displacement": {
                "labels": labels,
                "fraction_of_positions": (self.displacement_histogram / positions).tolist(),
                "share_of_excess_bits": (self.excess_bits_by_displacement
                                         / total_excess).tolist(),
                "excess_bits": self.excess_bits_by_displacement.tolist(),
            },
            "level_bands": {
                "bands": list(LEVEL_BANDS),
                "fraction_of_positions": (self.level_band_positions / positions).tolist(),
                "change_rate_within_band": (
                    self.level_band_changed
                    / np.maximum(self.level_band_positions, 1)).tolist(),
                "share_of_excess_bits": (self.level_band_excess_bits
                                         / total_excess).tolist(),
            },
            "change_rate_by_reference_shift_decile": (
                self.decile_changed / np.maximum(self.decile_positions, 1)).tolist(),
        }


@torch.no_grad()
def diagnose_sequence(mc, m13, model, rig, sequence, accumulator, *, bits, gop_size, block_size,
                      search_range, device) -> int:
    """One VAL-B sequence. The real arm advances the chain; the oracle arm is a
    read-only side channel, and the per-position reference shift is measured in
    units of the residual quantizer's own step so it is directly comparable to
    the level distance."""
    spec = rig["spec"]
    residual_params = rig["residual_params"]
    coding_probabilities = spec["coding_codebook"].probabilities
    scale = residual_params.scale.reshape(-1).to(device)
    zero_point = residual_params.zero_point.reshape(-1).to(device)
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
                    latent, params=rig["intra_params"],
                    entropy_model=rig["intra_entropy_model"])
                decoded, _ = decode_payload_to_latent(
                    payload, entropy_model=rig["intra_entropy_model"],
                    params=rig["intra_params"], shape=latent_shape)
                previous = model.decode(decoded.to(device))
                previous_raw = frame
                continue

            def _arm(reference_frame):
                motion = mc.estimate_block_motion(reference_frame, frame,
                                                  block_size=block_size,
                                                  search_range=search_range)
                warped = mc.warp_blocks(reference_frame, motion, block_size=block_size)
                reference_latent = model.encode(warped)
                delta = latent - reference_latent
                symbols = latent_to_symbols(delta, residual_params)
                table_index = spec["assign_codebook"].assign_tensor(
                    _rows(spec["model"], reference_latent, symbols, latent_shape,
                          spec["zero"], device))
                code_len = -np.log2(np.maximum(
                    coding_probabilities[table_index, symbols], 1e-300))
                return reference_latent, delta, symbols, code_len

            reference_real, delta_real, symbols_real, code_len_real = _arm(previous)
            reference_oracle, _, symbols_oracle, code_len_oracle = _arm(
                model.decode(model.encode(previous_raw)))

            # The quantizer's own view of the position: offset = v/scale + zp,
            # so |offset - round(offset)| is the distance to its own reconstruction
            # level IN STEPS, and the reference shift measured the same way is
            # directly comparable to it.
            broadcast = scale.view(1, -1, 1, 1)
            offset = (delta_real / broadcast + zero_point.view(1, -1, 1, 1))
            shift = ((reference_oracle - reference_real) / broadcast)

            accumulator.add(
                real_symbols=symbols_real, oracle_symbols=symbols_oracle,
                real_offset=offset.reshape(-1).cpu().numpy().astype(np.float64),
                reference_shift_steps=shift.reshape(-1).cpu().numpy().astype(np.float64),
                code_len_real=code_len_real, code_len_oracle=code_len_oracle)

            previous = model.decode(reference_real + symbols_to_latent(
                symbols_real, latent_shape, residual_params).to(device))
            previous_raw = frame
            p_frames += 1
    del m13
    return p_frames


def _rows(model11, reference, symbols, latent_shape, zero, device):
    ma = _load_script("m11_ar_entropy")
    target = torch.from_numpy(np.asarray(symbols, dtype=np.int64)).reshape(
        1, *latent_shape).to(device)
    return ma._rows(model11.log_probabilities(reference, model11.planes(target,
                                                                       zero.to(device))))


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M22 Phase 2: residual-space diagnostics (TRAIN + VAL-B only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt"))
    parser.add_argument("--m11-dir", type=Path, default=Path("outputs/m11_autoregressive_entropy"))
    parser.add_argument("--m10k-dir", type=Path, default=Path("outputs/m10k_learned_entropy"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/m22_residual_freeze_lift"))
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

    print("=" * 132)
    print("M22 PHASE 2 - RESIDUAL-SPACE DIAGNOSTICS (TRAIN + VAL-B only; TEST never opened)")
    print("=" * 132)
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "M22 Phase 2", "val_b_sequence_ids": [s.sequence_id for s in val_b],
        "level_bands": list(LEVEL_BANDS),
        "displacement_bins": list(DISPLACEMENT_BINS[:-1]) + ["inf"],
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
            motion_cache=args.output_dir / "m22_deployed_motion_table.json")

        residuals = m22.collect_train_residuals(
            mc, model, train_full, bits=bits,
            intra_params=rig["intra_params"],
            intra_entropy_model=rig["intra_entropy_model"],
            gop_size=args.gop, block_size=args.block_size,
            search_range=args.search_range, max_frames=args.calibration_frames, device=device,
            cache=args.output_dir / f"m22_train_residuals_{bits}bit.pt")
        m22.verify_deployed_grid(residuals["residual_stack"], rig["residual_params"], bits=bits)
        train_stats = m22.grid_statistics(residuals["residual_stack"],
                                          rig["residual_params"], bits=bits)

        accumulator = SensitivityAccumulator(bits)
        p_frames = 0
        for sequence in val_b:
            p_frames += diagnose_sequence(mc, m13, model, rig, sequence, accumulator, bits=bits,
                                          gop_size=args.gop, block_size=args.block_size,
                                          search_range=args.search_range, device=device)
        sensitivity = accumulator.to_dict()

        variance = np.asarray(train_stats["residual_std_per_channel"]) ** 2
        order = np.sort(variance)[::-1]
        concentration = {
            "top10pct_share": float(order[:max(1, len(order) // 10)].sum() / order.sum()),
            "top25pct_share": float(order[:max(1, len(order) // 4)].sum() / order.sum()),
            "max_over_median": float(order[0] / np.median(order)),
        }

        print(f"\n  ================ {bits}-bit  ({time.perf_counter() - started:.1f}s, "
              f"{p_frames} VAL-B P-frames, residual={rig['residual_identity']}) ================")
        print(f"    [A-C] residual std (mean over channels) {train_stats['residual_std_mean']:.4f}"
              f"  MAE {train_stats['residual_abs_mean']:.4f}  RMS {train_stats['residual_rms']:.4f}"
              f"  dynamic range {train_stats['dynamic_range_mean']:.3f}")
        print(f"          variance concentration: top10% of channels carry "
              f"{concentration['top10pct_share'] * 100:.1f}%, top25% "
              f"{concentration['top25pct_share'] * 100:.1f}%, max/median "
              f"{concentration['max_over_median']:.1f}x")
        print(f"    [D]   step mean {train_stats['step_mean']:.5f} "
              f"(min {train_stats['step_min']:.5f}, max {train_stats['step_max']:.5f})  "
              f"step/std {train_stats['step_over_std_mean']:.4f}  "
              f"zero exactly representable: {train_stats['zero_exactly_representable']}")
        print(f"    [E]   clipping {train_stats['clipping_fraction'] * 100:.4f}%  "
              f"quantization SNR {train_stats['quantization_snr_db']:.2f} dB")
        print(f"    [F]   symbol entropy {train_stats['symbol_entropy_bits']:.4f} bits of "
              f"{bits} ({train_stats['symbol_entropy_efficiency'] * 100:.1f}% efficient), "
              f"{train_stats['distinct_symbols_used']}/{train_stats['alphabet']} symbols used")
        print(f"    [G]   VAL-B symbol agreement with the oracle arm "
              f"{sensitivity['symbol_agreement'] * 100:.2f}%  "
              f"(change rate {sensitivity['symbol_change_rate'] * 100:.2f}%)")
        print(f"    [H]   reference shift: mean {sensitivity['mean_reference_shift_steps']:.4f} "
              f"steps, RMS {sensitivity['rms_reference_shift_steps']:.4f} steps")
        print(f"          change rate by reference-shift decile: "
              + " ".join(f"{r * 100:.0f}%" for r in
                         sensitivity["change_rate_by_reference_shift_decile"]))
        print(f"    [I-J] mean distance to its own reconstruction level "
              f"{sensitivity['mean_level_distance_steps']:.4f} steps "
              f"(uniform would be 0.25; 0.5 = on a decision boundary)")
        band = sensitivity["level_bands"]
        print(f"          {'band':>8} {'positions':>11} {'change rate':>12} {'excess-bit share':>17}")
        for position, value in enumerate(band["bands"]):
            print(f"          <{value:>7.2f} {band['fraction_of_positions'][position] * 100:>10.2f}%"
                  f" {band['change_rate_within_band'][position] * 100:>11.2f}%"
                  f" {band['share_of_excess_bits'][position] * 100:>16.2f}%")
        print(f"    [K]   symbol displacement |real - oracle|, and where the excess bits are:")
        displacement = sensitivity["symbol_displacement"]
        print(f"          {'steps':>8} {'positions':>11} {'excess-bit share':>17}")
        for position, label in enumerate(displacement["labels"]):
            print(f"          {label:>8} "
                  f"{displacement['fraction_of_positions'][position] * 100:>10.2f}%"
                  f" {displacement['share_of_excess_bits'][position] * 100:>16.2f}%")
        print(flush=True)

        report["rate_points"].append({
            "bits": bits, "p_frames": p_frames,
            "residual_identity": rig["residual_identity"],
            "train_grid_statistics": train_stats,
            "variance_concentration": concentration,
            "val_b_sensitivity": sensitivity,
        })

    path = args.output_dir / "m22_diagnostics.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
