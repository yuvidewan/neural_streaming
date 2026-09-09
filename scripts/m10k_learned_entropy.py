"""M10K: a small LEARNED conditional entropy model for the M10H residual symbols.

WHAT THIS CHANGES
-------------------
    M10H:  symbol            ->  per-CHANNEL table                    (64 tables)
    M10J:  symbol + context  ->  per-(CHANNEL, 4-BUCKET) table        (256 tables)
    M10K:  symbol            ->  a distribution predicted PER POSITION from z_ref

Nothing else moves: same encoder/decoder, motion, warp, residual, quantization,
GOP, container, and the same arithmetic coder. The residual symbols and the
reconstruction are bit-identical to M10H's; only the probability model differs.

THE AUDIT FINDING THAT MAKES THIS DEPLOYABLE UNCHANGED
--------------------------------------------------------
The project's arithmetic coder already selects a frequency table per symbol via
`table_index`. Nothing constrains how many tables there are, so a model that
predicts a DIFFERENT distribution at every one of the 64x16x16 = 16,384
positions is expressible as 16,384 tables with `table_index = arange(16384)`.
Measured: exact round trip, 4 ms encode / 2 ms decode per frame, 4.1 MB of
cumulative array. No coder change, no `.nvct` change, no new format.

THE MODEL
-----------
Deliberately tiny - the question is whether learned probability modelling adds
anything over M10J's 4-bucket lookup, not how large a network can be trained.

    z_ref [B, C, H, W]  ->  treat each latent channel as its own sample
    conv 1->32 (3x3) -> ReLU -> conv 32->32 (3x3) -> ReLU
    + a learned per-channel embedding (so channel identity is a real input,
      matching what M10H/M10J get from having one table per channel)
    -> conv 32->alphabet (1x1)
    -> logits over the residual symbol alphabet at every position

A 3x3 stack gives each prediction a 5x5 receptive field on z_ref, which is the
same kind of local information M10J's `local_activity4` bucketed by hand - only
here the network learns what to extract instead of being told.

CAUSALITY
-----------
The predictor's ONLY input is `z_ref` (plus the channel index and position).
`z_ref = Encoder(Warp(x_hat_{t-1}, decoded motion))` is fully reconstructed by
the decoder before it touches the residual payload. There is deliberately NO
autoregressive dependence on previously decoded residual symbols: that would add
a sequential dependency to the coder and is not needed to answer this
milestone's question.

TRAINING
----------
Pure rate: `L = -log2 P(R | z_ref, channel)` on the EXISTING M10H symbols. No
reconstruction loss, no transform training, no lambda - M10I already showed what
happens when a rate question is asked of a distortion-dominated objective.
Fitted on TRAIN, selected on VALIDATION, measured on TEST once.

Example usage (PowerShell, from the project root, with .venv activated):

    .venv\\Scripts\\python.exe scripts\\m10k_learned_entropy.py --stage gate
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nvc.compression.entropy_model import (
    MIN_FREQUENCY,
    TOTAL_FREQUENCY,
    EmpiricalEntropyModel,
)
from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m10k_learned_entropy")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
FROZEN_LAMBDA = 3.0e-4
RATE_POINTS = (5, 4, 3)
DEFAULT_GOP = 10
DEFAULT_BLOCK_SIZE = 16
DEFAULT_SEARCH_RANGE = 16
# Numerical floor before the float -> integer frequency conversion. Chosen so
# that even the least likely symbol survives the conversion with frequency >= 1.
PROBABILITY_FLOOR = 1.0 / TOTAL_FREQUENCY


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the model ----------------------------------------------------------------


class LearnedConditionalEntropyModel(nn.Module):
    """Predicts P(residual symbol | z_ref neighbourhood, channel) per position.

    Each latent channel is processed by the SAME small convolutional stack, with
    a learned per-channel embedding added before the output head. That keeps the
    parameter count independent of the channel count while still letting channel
    identity shape the prediction - which is exactly what M10H and M10J get from
    having one table per channel.
    """

    def __init__(self, latent_channels: int = 64, alphabet: int = 16,
                 hidden: int = 32) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.alphabet = alphabet
        self.hidden = hidden
        self.features = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.channel_embedding = nn.Embedding(latent_channels, hidden)
        self.head = nn.Conv2d(hidden, alphabet, 1)
        nn.init.zeros_(self.channel_embedding.weight)

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] reference latent -> [B, C, alphabet, H, W] logits."""
        batch, channels, height, width = reference.shape
        if channels != self.latent_channels:
            raise ValueError(
                f"expected {self.latent_channels} latent channels, got {channels}")
        stacked = reference.reshape(batch * channels, 1, height, width)
        features = self.features(stacked)
        embedding = self.channel_embedding.weight.repeat(batch, 1)
        features = features + embedding[:, :, None, None]
        logits = self.head(features)
        return logits.reshape(batch, channels, self.alphabet, height, width)

    def log_probabilities(self, reference: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.forward(reference), dim=2)

    def config_dict(self) -> dict[str, int]:
        return {"latent_channels": self.latent_channels, "alphabet": self.alphabet,
                "hidden": self.hidden}


def build_model(config: dict[str, int] | None = None) -> LearnedConditionalEntropyModel:
    return LearnedConditionalEntropyModel(**(config or {}))


def load_entropy_model(path, *, device=None):
    checkpoint = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, checkpoint


# --- float probabilities -> coder frequencies ---------------------------------


def probabilities_to_frequencies(probabilities: np.ndarray) -> np.ndarray:
    """Deterministic float -> integer frequency conversion for the coder.

    The coder requires every table to total EXACTLY 65536 with no zero entry, so
    the conversion has to be exact and reproducible, not merely close:

      1. clamp away underflow, then renormalise;
      2. floor(p * 65536), then raise every entry to at least MIN_FREQUENCY;
      3. distribute the remaining residual deterministically - largest
         frequencies first, ties broken by index - so the same probabilities
         always produce the same table on any machine;
      4. when flooring has overshot the total, take back from the largest
         entries first, never below MIN_FREQUENCY.

    Step 3's tie-break is what makes encoder and decoder agree: `np.argsort` with
    a stable kind gives a fixed order for equal frequencies.
    """
    if probabilities.ndim != 2:
        raise ValueError(f"expected [tables, alphabet], got {probabilities.shape}")
    clamped = np.maximum(probabilities.astype(np.float64), PROBABILITY_FLOOR)
    clamped = clamped / clamped.sum(axis=1, keepdims=True)

    frequencies = np.maximum(
        np.floor(clamped * TOTAL_FREQUENCY).astype(np.int64), MIN_FREQUENCY)
    alphabet = frequencies.shape[1]
    # Rank of each entry within its own row, largest first, ties by index. This is
    # the deterministic order the residual is distributed along.
    order = np.argsort(-frequencies, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(alphabet)[None, :].repeat(
        frequencies.shape[0], axis=0), axis=1)

    residual = TOTAL_FREQUENCY - frequencies.sum(axis=1)

    # Surplus: hand out floor(r / alphabet) to everyone, then one more to the top
    # (r mod alphabet) ranks - exactly what walking the ranking r times would do.
    surplus = np.maximum(residual, 0)
    frequencies += (surplus // alphabet)[:, None]
    frequencies += (rank < (surplus % alphabet)[:, None]).astype(np.int64)

    # Deficit: take back one at a time from the largest entries, never below
    # MIN_FREQUENCY. Rows needing this are rare, so the loop stays exact without
    # costing anything in the common case.
    for row in np.nonzero(residual < 0)[0]:
        deficit, ranking, index = int(-residual[row]), order[row], 0
        while deficit > 0:
            target = ranking[index % alphabet]
            if frequencies[row, target] > MIN_FREQUENCY:
                frequencies[row, target] -= 1
                deficit -= 1
            index += 1
    return frequencies


@torch.no_grad()
def frame_entropy_model(model, reference: torch.Tensor, *, bits: int):
    """One `EmpiricalEntropyModel` holding a distribution per symbol position.

    Returns (entropy_model, table_index) ready for the existing coder. The table
    order is C-major to match `latent_to_symbols`, so table i belongs to flat
    symbol i and `table_index` is simply arange.
    """
    log_probabilities = model.log_probabilities(reference)  # [1, C, A, H, W]
    probabilities = log_probabilities.exp()[0]              # [C, A, H, W]
    channels, alphabet, height, width = probabilities.shape
    flat = probabilities.permute(0, 2, 3, 1).reshape(channels * height * width, alphabet)
    frequencies = probabilities_to_frequencies(flat.double().cpu().numpy())
    return (EmpiricalEntropyModel(frequencies, bits=bits),
            np.arange(channels * height * width, dtype=np.int64))


# --- training ------------------------------------------------------------------


def train_entropy_model(model, train_pairs, val_pairs, *, epochs: int, batch_size: int,
                        learning_rate: float, device, seed: int = 42, log=print):
    """Minimise -log2 P(R | z_ref, channel). Pure rate, nothing else."""
    seed_everything(seed)
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_symbols, train_references = train_pairs
    val_symbols, val_references = val_pairs
    train_symbols = torch.from_numpy(np.stack(train_symbols)).long()
    train_references = torch.from_numpy(np.stack(train_references)).float()
    val_symbols = torch.from_numpy(np.stack(val_symbols)).long()
    val_references = torch.from_numpy(np.stack(val_references)).float()

    def nll_bits(symbols, references, *, training: bool) -> float:
        total, count = 0.0, 0
        order = torch.randperm(symbols.shape[0]) if training else torch.arange(symbols.shape[0])
        for start in range(0, symbols.shape[0], batch_size):
            index = order[start:start + batch_size]
            reference = references[index].to(device)
            target = symbols[index].to(device)
            log_probabilities = model.log_probabilities(reference)
            # [B, C, A, H, W] -> gather the log-probability of the true symbol
            picked = log_probabilities.gather(2, target.unsqueeze(2)).squeeze(2)
            loss = -picked.mean() / math.log(2.0)
            if training:
                optimizer.zero_grad(set_to_none=True)
                (loss * math.log(2.0)).backward()
                optimizer.step()
            total += float(loss.detach()) * index.numel()
            count += index.numel()
        return total / count

    history = []
    # The M10G convention deploys the BEST-VALIDATION model, so its weights have
    # to be kept as training continues past it. Reporting a selected epoch while
    # measuring the final weights is the exact defect M10I hit.
    best_state, best_bits = None, float("inf")
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        train_bits = nll_bits(train_symbols, train_references, training=True)
        model.eval()
        with torch.no_grad():
            val_bits = nll_bits(val_symbols, val_references, training=False)
        history.append({"epoch": epoch, "train_nll_bits": train_bits,
                        "val_nll_bits": val_bits, "rate_enabled": True,
                        "val_loss": val_bits,
                        "elapsed_seconds": time.time() - started})
        if val_bits < best_bits:
            best_bits = val_bits
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"[EPOCH {epoch:>3}] train {train_bits:.5f}  val {val_bits:.5f} bits/symbol  "
            f"{history[-1]['elapsed_seconds']:.0f}s")
    # Restore the selected weights so every downstream measurement uses them.
    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def cross_entropy_bits(model, symbols, references, *, device, batch_size: int = 16) -> float:
    """Mean -log2 P(R) over a symbol set, under the learned model."""
    model.eval()
    stacked_symbols = torch.from_numpy(np.stack(symbols)).long()
    stacked_references = torch.from_numpy(np.stack(references)).float()
    total, count = 0.0, 0
    for start in range(0, stacked_symbols.shape[0], batch_size):
        reference = stacked_references[start:start + batch_size].to(device)
        target = stacked_symbols[start:start + batch_size].to(device)
        log_probabilities = model.log_probabilities(reference)
        picked = log_probabilities.gather(2, target.unsqueeze(2)).squeeze(2)
        total += float(-picked.sum()) / math.log(2.0)
        count += target.numel()
    return total / count


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10K: a small learned conditional entropy model for M10H residuals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage", choices=["gate", "smoke"], default="gate")
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--gop", type=int, default=DEFAULT_GOP)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--search-range", type=int, default=DEFAULT_SEARCH_RANGE)
    parser.add_argument("--quant-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--train-frames", type=int, default=600)
    parser.add_argument("--val-frames", type=int, default=200)
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
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
    ce = _load_script("m10j_conditional_entropy")
    convention = _load_script("m10g_evaluation_convention")
    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    train_sequences = discover_sequences(args.manifest, split="train")
    val_sequences = discover_sequences(args.manifest, split="val")
    smoke = args.stage == "smoke"
    train_frames = 40 if smoke else args.train_frames
    val_frames = 20 if smoke else args.val_frames
    epochs = 2 if smoke else args.epochs

    print("=" * 112)
    print("M10K - LEARNED CONDITIONAL ENTROPY MODEL: OFFLINE GATE")
    print("=" * 112)
    print("  Deployed through the EXISTING arithmetic coder as one table per symbol position")
    print("  (audit: 16,384 tables round-trip exactly at ~4 ms encode / 2 ms decode per frame).")
    print("  Objective is pure rate: -log2 P(R | z_ref, channel). No reconstruction loss.")
    print("  Fitted on TRAIN, selected on VALIDATION. Test data is not touched here.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "M10K offline gate", "frozen_lambda": FROZEN_LAMBDA,
        "checkpoint": str(args.checkpoint), "rate_points": [],
        "note": ("Held-out validation cross-entropy is the headline. Training NLL is "
                 "diagnostic only - a more expressive model can always look better on the "
                 "data it was fitted to."),
    }

    for bits in args.rate_points:
        alphabet = 2 ** bits
        print(f"\n  collecting M10H symbols at {bits}-bit ...", flush=True)
        calibration = mc.calibrate_grids(
            model, train_sequences, bits=bits, mode=args.quant_mode, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            reference_mode="mc", max_frames=args.calibration_frames)
        train_symbols, train_references = ce.collect_training_symbols(
            mc, model, train_sequences, calibration, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, device=device,
            max_frames=train_frames)
        val_symbols, val_references = ce.collect_training_symbols(
            mc, model, val_sequences, calibration, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, device=device,
            max_frames=val_frames)
        channels = train_symbols[0].shape[0]
        print(f"    {len(train_symbols)} train P-frames, {len(val_symbols)} validation P-frames")

        # --- the two existing baselines, scored on the SAME held-out symbols ---
        baselines = {}
        for scheme in ("marginal", "local_activity4"):
            context_model = ce.fit_context_model(scheme, train_references)
            built = ce.build_conditional_entropy_model(
                train_symbols, train_references, context_model, bits=bits)
            frequencies = built["entropy_model"].frequencies.astype(np.float64)
            probabilities = frequencies / frequencies.sum(axis=1, keepdims=True)
            total, count = 0.0, 0
            for symbol_frame, reference in zip(val_symbols, val_references):
                contexts = context_model.contexts(reference)
                table = context_model.table_index(*symbol_frame.shape, contexts)
                total += float(-np.log2(probabilities[table, symbol_frame.reshape(-1)]).sum())
                count += symbol_frame.size
            baselines[scheme] = total / count

        # --- the learned model ------------------------------------------------
        learned = build_model({"latent_channels": channels, "alphabet": alphabet,
                               "hidden": args.hidden})
        parameters = sum(p.numel() for p in learned.parameters())
        print(f"    learned model: {parameters:,} trainable parameters "
              f"(hidden={args.hidden}, alphabet={alphabet})")
        history = train_entropy_model(
            learned, (train_symbols, train_references), (val_symbols, val_references),
            epochs=epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
            device=device, seed=args.seed)
        # `train_entropy_model` has already restored the best-validation weights.
        selection = convention.select_checkpoint(history, objective_key="val_loss")
        learned_bits = cross_entropy_bits(
            learned, val_symbols, val_references, device=device)

        marginal, conditional = baselines["marginal"], baselines["local_activity4"]
        row = {
            "bits": bits, "alphabet": alphabet, "channels": channels,
            "train_p_frames": len(train_symbols), "val_p_frames": len(val_symbols),
            "parameters": parameters,
            "m10h_marginal_val_bits": marginal,
            "m10j_conditional_val_bits": conditional,
            "m10k_learned_val_bits": learned_bits,
            "m10k_vs_m10h_percent": (marginal - learned_bits) / marginal * 100,
            "m10k_vs_m10j_percent": (conditional - learned_bits) / conditional * 100,
            "m10j_vs_m10h_percent": (marginal - conditional) / marginal * 100,
            "checkpoint_selection": selection,
            "history": history,
        }
        report["rate_points"].append(row)
        print(f"\n    HELD-OUT validation cross-entropy (bits/symbol):")
        print(f"      M10H marginal        {marginal:.5f}")
        print(f"      M10J local_activity4 {conditional:.5f}   "
              f"({row['m10j_vs_m10h_percent']:+.2f}% vs M10H)")
        print(f"      M10K learned         {learned_bits:.5f}   "
              f"({row['m10k_vs_m10h_percent']:+.2f}% vs M10H, "
              f"{row['m10k_vs_m10j_percent']:+.2f}% vs M10J)")

        torch.save({"model_state_dict": learned.state_dict(),
                    "model_config": learned.config_dict(),
                    "history": history, "selection": selection, "bits": bits},
                   args.output_dir / f"learned_entropy_{bits}bit.pt")

    print()
    print("=" * 112)
    print("GATE: does the learned model beat M10J on held-out data?")
    print("=" * 112)
    print(f"{'bits':>5} {'M10H':>10} {'M10J':>10} {'M10K':>10} {'K vs J':>10}   verdict")
    beats = []
    for row in report["rate_points"]:
        gain = row["m10k_vs_m10j_percent"]
        beats.append(gain)
        verdict = ("beats M10J" if gain > 0.5 else
                   "matches M10J" if gain > -0.5 else "WORSE than M10J")
        print(f"{row['bits']:>5} {row['m10h_marginal_val_bits']:>10.5f} "
              f"{row['m10j_conditional_val_bits']:>10.5f} "
              f"{row['m10k_learned_val_bits']:>10.5f} {gain:>+9.2f}%   {verdict}")
    report["best_vs_m10j_percent"] = max(beats)
    report["gate_passed"] = max(beats) > 0.5
    print()
    print(f"  best held-out improvement over M10J: {max(beats):+.2f}%")
    print(f"  GATE {'PASSED' if report['gate_passed'] else 'NOT PASSED'} "
          f"(threshold: >0.5% on held-out validation)")

    path = args.output_dir / ("smoke_gate.json" if smoke else "offline_gate.json")
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
