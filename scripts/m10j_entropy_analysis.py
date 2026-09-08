"""M10J step 1: does the warped reference actually predict the residual symbols?

WHY THIS RUNS BEFORE ANY IMPLEMENTATION
-----------------------------------------
M10I spent a milestone on a learned conditional transform and found it improved
distortion rather than rate. Before building a conditional ENTROPY model, this
script asks the prior question directly and cheaply: is there predictive
information about the residual symbol in the reference at all? If there is not,
no amount of modelling machinery will find it, and the correct outcome is to
stop.

THE MEASUREMENT, AND THE TRAP IT AVOIDS
-----------------------------------------
The deployed arithmetic coder ALREADY uses one frequency table per latent
channel. So the honest baseline is not H(R) - it is

    H(R | channel)

and the quantity that matters is the ADDITIONAL reduction from reference
context:

    H(R | channel, C)   versus   H(R | channel)

Comparing H(R|C) against a channel-blind H(R) would credit the reference with
the gain the per-channel model already captures, and would manufacture an
improvement out of nothing. Everything below is reported against the
channel-conditional baseline, with the channel-blind figure shown once purely to
make that gap visible.

WHAT COUNTS AS A REAL GAIN
----------------------------
Conditioning can only ever reduce empirical entropy on the data it was measured
on - adding contexts never increases a plug-in entropy estimate, even for a
random context. Two guards are therefore applied:

  * a RANDOM context of the same cardinality is measured alongside every real
    one, giving the spurious reduction that context count alone buys;
  * entropies are estimated on TRAIN symbols and re-scored on a held-out
    VALIDATION split, so a context that only memorises the training histogram
    shows up as a validation regression.

Test frames are never touched.

Example usage (PowerShell, from the project root, with .venv activated):

    .venv\\Scripts\\python.exe scripts\\m10j_entropy_analysis.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.codec import (
    decode_payload_to_latent,
    encode_latent_to_payload,
    latent_to_symbols,
)
from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device

DEFAULT_OUTPUT_DIR = Path("outputs/m10j_conditional_entropy")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
RATE_POINTS = (5, 4, 3)
DEFAULT_GOP = 10
DEFAULT_BLOCK_SIZE = 16
DEFAULT_SEARCH_RANGE = 16
# Laplace smoothing, matching the project's established entropy-table practice.
SMOOTHING = 1.0


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- context functions (deterministic, causal, decoder-computable) ------------


def magnitude_thresholds(reference: np.ndarray, buckets: int) -> np.ndarray:
    """Per-channel |z_ref| quantile edges, fitted on TRAIN references only.

    Per-channel rather than global because latent channels have very different
    scales; a global threshold would put whole channels in one bucket and make
    the context inert for them.
    """
    channels = reference.shape[0]
    quantiles = np.linspace(0, 100, buckets + 1)[1:-1]
    return np.stack([np.percentile(np.abs(reference[c]), quantiles) for c in range(channels)])


def context_none(reference: np.ndarray, meta: dict) -> np.ndarray:
    return np.zeros(reference.shape, dtype=np.int64)


def context_sign(reference: np.ndarray, meta: dict) -> np.ndarray:
    """2 contexts: is the reference latent negative or not."""
    return (reference >= 0).astype(np.int64)


def context_magnitude(reference: np.ndarray, meta: dict) -> np.ndarray:
    """`buckets` contexts: which per-channel |z_ref| quantile band."""
    edges = meta["thresholds"]
    out = np.zeros(reference.shape, dtype=np.int64)
    for c in range(reference.shape[0]):
        out[c] = np.searchsorted(edges[c], np.abs(reference[c]))
    return out


def context_sign_magnitude(reference: np.ndarray, meta: dict) -> np.ndarray:
    """sign x magnitude, the natural product of the two simplest signals."""
    buckets = meta["buckets"]
    return context_magnitude(reference, meta) * 2 + context_sign(reference, meta)


def context_local_activity(reference: np.ndarray, meta: dict) -> np.ndarray:
    """How much the reference varies in a 3x3 neighbourhood, bucketed.

    The intuition worth testing: the residual should be large exactly where the
    warped reference is unreliable, and local high-frequency content is a proxy
    for that.
    """
    edges = meta["thresholds"]
    tensor = torch.from_numpy(reference).unsqueeze(0)
    mean = torch.nn.functional.avg_pool2d(tensor, 3, stride=1, padding=1)
    activity = torch.nn.functional.avg_pool2d((tensor - mean).abs(), 3, stride=1, padding=1)
    activity = activity[0].numpy()
    out = np.zeros(reference.shape, dtype=np.int64)
    for c in range(reference.shape[0]):
        out[c] = np.searchsorted(edges[c], activity[c])
    return out


CONTEXTS: dict[str, dict[str, Any]] = {
    "marginal": {"fn": context_none, "cardinality": 1, "needs": None},
    "sign": {"fn": context_sign, "cardinality": 2, "needs": None},
    "magnitude4": {"fn": context_magnitude, "cardinality": 4, "needs": "magnitude"},
    "magnitude8": {"fn": context_magnitude, "cardinality": 8, "needs": "magnitude"},
    "sign_x_magnitude4": {"fn": context_sign_magnitude, "cardinality": 8, "needs": "magnitude"},
    "local_activity4": {"fn": context_local_activity, "cardinality": 4, "needs": "activity"},
}


# --- entropy bookkeeping ------------------------------------------------------


class ContextCounts:
    """counts[channel, context, symbol], accumulated streaming."""

    def __init__(self, channels: int, cardinality: int, symbols: int) -> None:
        self.counts = np.zeros((channels, cardinality, symbols), dtype=np.int64)
        self.channels, self.cardinality, self.symbols = channels, cardinality, symbols

    def add(self, symbols: np.ndarray, contexts: np.ndarray) -> None:
        channels = self.channels
        flat_symbol = symbols.reshape(channels, -1)
        flat_context = contexts.reshape(channels, -1)
        for c in range(channels):
            index = flat_context[c] * self.symbols + flat_symbol[c]
            self.counts[c] += np.bincount(
                index, minlength=self.cardinality * self.symbols
            ).reshape(self.cardinality, self.symbols)

    def entropy_bits_per_symbol(self, smoothing: float = SMOOTHING) -> float:
        """Plug-in conditional entropy, weighted by how often each context occurs."""
        smoothed = self.counts + smoothing
        probabilities = smoothed / smoothed.sum(axis=2, keepdims=True)
        per_cell = -(probabilities * np.log2(probabilities)).sum(axis=2)
        weights = self.counts.sum(axis=2)
        total = weights.sum()
        return float((per_cell * weights).sum() / total) if total else 0.0

    def cross_entropy_bits_per_symbol(self, other: "ContextCounts",
                                      smoothing: float = SMOOTHING) -> float:
        """Cost of coding `other`'s symbols with THIS model's tables.

        This is the honest held-out number: a context that merely memorised the
        training histogram pays for it here.
        """
        smoothed = self.counts + smoothing
        probabilities = smoothed / smoothed.sum(axis=2, keepdims=True)
        bits = -np.log2(probabilities)
        total = other.counts.sum()
        return float((bits * other.counts).sum() / total) if total else 0.0

    def occupancy(self, minimum: int) -> dict[str, Any]:
        weights = self.counts.sum(axis=2)
        return {
            "cells": int(weights.size),
            "occupied_cells": int((weights > 0).sum()),
            "sparse_cells": int(((weights > 0) & (weights < minimum)).sum()),
            "min_samples": int(weights.min()), "max_samples": int(weights.max()),
            "median_samples": float(np.median(weights)),
        }


# --- symbol collection --------------------------------------------------------


@torch.no_grad()
def collect_symbols(mc, model, sequences, calibration, *, gop_size: int, block_size: int,
                    search_range: int, device, max_frames: int):
    """Run the M10H codec closed-loop and record each P-frame's residual symbols
    alongside the reference latent that produced them.

    These are the EXACT symbols M10H entropy-codes - not a re-derivation - so
    any entropy measured here is the entropy of the real bitstream's content.
    """
    model.eval()
    symbol_frames, reference_frames = [], []
    seen = 0
    with mc.deterministic_kernels():
        for sequence in sequences:
            frames = sequence.load_frames()
            types = mc.gop_frame_types(frames.shape[0], gop_size)
            latent_shape, previous = None, None
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
                    reconstructed = decoded.to(device)
                else:
                    motion = mc.estimate_block_motion(
                        previous, frame, block_size=block_size, search_range=search_range)
                    payload = mc.encode_motion_payload(
                        motion, search_range=search_range,
                        entropy_model=calibration["motion_entropy_model"])
                    decoded_motion = mc.decode_motion_payload(
                        payload, (frames.shape[2] // block_size, frames.shape[3] // block_size),
                        search_range=search_range,
                        entropy_model=calibration["motion_entropy_model"])
                    warped = mc.warp_blocks(previous, decoded_motion, block_size=block_size)
                    reference = model.encode(warped)
                    delta = latent - reference

                    symbols = latent_to_symbols(delta, calibration["residual_params"])
                    symbol_frames.append(symbols.reshape(latent_shape).astype(np.int64))
                    reference_frames.append(reference[0].cpu().numpy())

                    residual_payload, _ = encode_latent_to_payload(
                        delta, params=calibration["residual_params"],
                        entropy_model=calibration["residual_entropy_model"])
                    decoded_delta, _ = decode_payload_to_latent(
                        residual_payload, entropy_model=calibration["residual_entropy_model"],
                        params=calibration["residual_params"], shape=latent_shape)
                    reconstructed = reference + decoded_delta.to(device)
                previous = model.decode(reconstructed)
                seen += 1
            if seen >= max_frames:
                break
    return symbol_frames, reference_frames


def analyse(symbol_frames, reference_frames, holdout_symbols, holdout_references, *,
            bits: int, channels: int, seed: int = 42) -> list[dict[str, Any]]:
    """Measure every context scheme on train, then re-score it on held-out data."""
    symbols_count = 2 ** bits
    rng = np.random.default_rng(seed)
    stacked_reference = np.concatenate([r.reshape(channels, -1) for r in reference_frames], axis=1)

    results = []
    for name, spec in CONTEXTS.items():
        cardinality = spec["cardinality"]
        meta: dict[str, Any] = {"buckets": cardinality}
        if spec["needs"] == "magnitude":
            buckets = 4 if name == "sign_x_magnitude4" else cardinality
            meta = {"buckets": buckets,
                    "thresholds": magnitude_thresholds(stacked_reference, buckets)}
        elif spec["needs"] == "activity":
            tensor = torch.from_numpy(np.stack(reference_frames))
            mean = torch.nn.functional.avg_pool2d(tensor, 3, stride=1, padding=1)
            activity = torch.nn.functional.avg_pool2d((tensor - mean).abs(), 3, stride=1,
                                                      padding=1).numpy()
            meta = {"buckets": cardinality,
                    "thresholds": magnitude_thresholds(
                        activity.transpose(1, 0, 2, 3).reshape(channels, -1), cardinality)}

        train = ContextCounts(channels, cardinality, symbols_count)
        for symbols, reference in zip(symbol_frames, reference_frames):
            train.add(symbols, spec["fn"](reference, meta))
        held = ContextCounts(channels, cardinality, symbols_count)
        for symbols, reference in zip(holdout_symbols, holdout_references):
            held.add(symbols, spec["fn"](reference, meta))

        # A random context of the same cardinality: the spurious reduction that
        # context count alone buys on a plug-in estimate.
        random_train = ContextCounts(channels, cardinality, symbols_count)
        random_held = ContextCounts(channels, cardinality, symbols_count)
        for symbols, _ in zip(symbol_frames, reference_frames):
            random_train.add(symbols, rng.integers(0, cardinality, size=symbols.shape))
        for symbols, _ in zip(holdout_symbols, holdout_references):
            random_held.add(symbols, rng.integers(0, cardinality, size=symbols.shape))

        results.append({
            "context": name, "cardinality": cardinality,
            "train_entropy_bits": train.entropy_bits_per_symbol(),
            "holdout_cross_entropy_bits": train.cross_entropy_bits_per_symbol(held),
            "random_train_entropy_bits": random_train.entropy_bits_per_symbol(),
            "random_holdout_cross_entropy_bits":
                random_train.cross_entropy_bits_per_symbol(random_held),
            "occupancy": train.occupancy(minimum=1000),
            "thresholds": (meta["thresholds"].tolist()
                           if isinstance(meta.get("thresholds"), np.ndarray) else None),
        })
    return results


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10J: offline conditional-entropy analysis of M10H residual symbols.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--gop", type=int, default=DEFAULT_GOP)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--search-range", type=int, default=DEFAULT_SEARCH_RANGE)
    parser.add_argument("--quant-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--train-frames", type=int, default=600)
    parser.add_argument("--holdout-frames", type=int, default=200)
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    mc = _load_script("m10h_motion_compensation")
    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    train_sequences = discover_sequences(args.manifest, split="train")
    val_sequences = discover_sequences(args.manifest, split="val")

    print("=" * 112)
    print("M10J STEP 1 - IS THERE PREDICTIVE INFORMATION IN THE REFERENCE?")
    print("=" * 112)
    print("  Baseline is H(R | channel), NOT H(R): the deployed coder already uses one")
    print("  frequency table per channel, so only the ADDITIONAL reduction counts.")
    print("  Every context is measured against a RANDOM context of equal cardinality and")
    print("  re-scored on held-out validation symbols. TEST frames are never touched.")

    report: dict[str, Any] = {
        "phase": "M10J offline conditional-entropy analysis",
        "baseline_definition": "H(R | channel) - the deployed per-channel table model",
        "guards": [
            "random context of equal cardinality, to expose plug-in bias",
            "held-out cross-entropy on the validation split",
            "no test frames used at any point",
        ],
        "smoothing": SMOOTHING, "rate_points": [], }

    for bits in args.rate_points:
        print(f"\n  collecting M10H residual symbols at {bits}-bit ...", flush=True)
        calibration = mc.calibrate_grids(
            model, train_sequences, bits=bits, mode=args.quant_mode, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            reference_mode="mc", max_frames=args.calibration_frames)
        train_symbols, train_references = collect_symbols(
            mc, model, train_sequences, calibration, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            device=device, max_frames=args.train_frames)
        held_symbols, held_references = collect_symbols(
            mc, model, val_sequences, calibration, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            device=device, max_frames=args.holdout_frames)

        channels = train_symbols[0].shape[0]
        total_symbols = sum(s.size for s in train_symbols)
        print(f"    {len(train_symbols)} train P-frames ({total_symbols:,} symbols), "
              f"{len(held_symbols)} held-out P-frames")

        rows = analyse(train_symbols, train_references, held_symbols, held_references,
                       bits=bits, channels=channels)
        baseline = next(r for r in rows if r["context"] == "marginal")

        print(f"\n  {bits}-bit  (baseline H(R|channel) = {baseline['train_entropy_bits']:.4f} "
              f"bits/symbol, held-out {baseline['holdout_cross_entropy_bits']:.4f})")
        print(f"    {'context':<20} {'|C|':>4} {'H train':>9} {'reduction':>10} "
              f"{'H held-out':>11} {'reduction':>10} {'random ctrl':>12} {'NET':>8}")
        for row in rows:
            train_gain = (baseline["train_entropy_bits"] - row["train_entropy_bits"]) \
                / baseline["train_entropy_bits"] * 100
            held_gain = (baseline["holdout_cross_entropy_bits"]
                         - row["holdout_cross_entropy_bits"]) \
                / baseline["holdout_cross_entropy_bits"] * 100
            random_gain = (baseline["holdout_cross_entropy_bits"]
                           - row["random_holdout_cross_entropy_bits"]) \
                / baseline["holdout_cross_entropy_bits"] * 100
            net = held_gain - random_gain
            row.update({"train_reduction_percent": train_gain,
                        "holdout_reduction_percent": held_gain,
                        "random_holdout_reduction_percent": random_gain,
                        "net_reduction_percent": net})
            print(f"    {row['context']:<20} {row['cardinality']:>4} "
                  f"{row['train_entropy_bits']:>9.4f} {train_gain:>+9.2f}% "
                  f"{row['holdout_cross_entropy_bits']:>11.4f} {held_gain:>+9.2f}% "
                  f"{random_gain:>+11.2f}% {net:>+7.2f}%")

        best = max((r for r in rows if r["context"] != "marginal"),
                   key=lambda r: r["net_reduction_percent"])
        print(f"    -> best net (held-out, random-corrected): {best['context']} "
              f"{best['net_reduction_percent']:+.2f}%")
        report["rate_points"].append({"bits": bits, "channels": channels,
                                      "train_p_frames": len(train_symbols),
                                      "train_symbols": total_symbols,
                                      "holdout_p_frames": len(held_symbols),
                                      "contexts": rows, "best_context": best["context"],
                                      "best_net_reduction_percent":
                                          best["net_reduction_percent"]})

    print()
    print("=" * 112)
    print("GATE")
    print("=" * 112)
    best_overall = max(p["best_net_reduction_percent"] for p in report["rate_points"])
    report["best_net_reduction_percent"] = best_overall
    print(f"  best net held-out entropy reduction across all rate points: {best_overall:+.2f}%")
    print("  A conditional entropy coder can only ever deliver LESS than the entropy")
    print("  reduction measured here, because the arithmetic coder adds its own overhead")
    print("  and the context tables have to be shipped or re-derived.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "entropy_analysis.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
