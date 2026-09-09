"""M10L offline gate - does a shared codebook keep M10K's rate gain more cheaply?

The gate deliberately has TWO criteria, because either alone is easy to pass and
useless:

  RATE    the held-out ideal code length under the codebook must stay within a
          small penalty of M10K's, measured on VALIDATION P-frames the codebook
          was never fitted to;
  COST    per-frame entropy work must actually get materially cheaper.

A codebook that is fast but throws away the gain is failure mode C; one that
keeps the gain but is no faster is failure mode B. The gate reports the whole
K-vs-penalty-vs-time frontier rather than only the K that happens to win, so the
tradeoff is visible even if no K clears the pre-registered thresholds.

BEFORE ANY OF THAT: the zero-loss reference. A codebook holding each frame's own
16,384 distributions, mapped to itself, must reproduce M10K's frequencies,
table_index and payload BYTE-FOR-BYTE. If it does not, the codebook path is
doing something different from M10K and no penalty measured through it means
anything - the gate stops there.

Everything here is fitted on TRAIN and scored on VALIDATION. Test sequences are
not opened by this script.

Run:
  ./.venv/Scripts/python.exe scripts/m10l_offline_gate.py
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

from nvc.compression.entropy_model import TOTAL_FREQUENCY, EmpiricalEntropyModel
from nvc.compression.range_coder import decode_symbols, encode_symbols
from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m10l_shared_codebook")
DEFAULT_MODEL_DIR = Path("outputs/m10k_learned_entropy")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
RATE_POINTS = (5, 4, 3)

# Pre-registered, before any K was measured. Taken from the milestone brief's
# practical success criterion.
MAX_RATE_PENALTY_PERCENT = 0.5
MIN_TABLE_TIME_REDUCTION_PERCENT = 50.0


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def zero_loss_reference(mk, ml, learned, reference, symbols, *, bits) -> dict[str, Any]:
    """K = 16,384 identity: the codebook path must BE the M10K path.

    Two checks, and the second is the stronger one:
      1. the identity mapping (`table_index = arange`) reproduces M10K exactly;
      2. running the real nearest-prototype SEARCH against that same codebook
         also reproduces M10K's payload - which validates the assignment code,
         not just the table plumbing. Index ties are permitted here (two
         identical distributions may both resolve to the lower index) because
         identical tables produce identical bytes either way; the payload is
         what must match.
    """
    entropy_model, m10k_index = mk.frame_entropy_model(learned, reference, bits=bits)
    m10k_payload = encode_symbols(symbols.reshape(-1), entropy_model.cumulative, m10k_index)

    probabilities = ml.frame_probabilities(learned, reference).double().cpu().numpy()
    codebook, identity_index = ml.identity_codebook(probabilities, bits=bits)
    identity_payload = encode_symbols(
        symbols.reshape(-1), codebook.cumulative, identity_index)

    searched_index = codebook.assign(probabilities)
    searched_payload = encode_symbols(
        symbols.reshape(-1), codebook.cumulative, searched_index)

    round_trip = decode_symbols(identity_payload, symbols.size, codebook.cumulative,
                                identity_index)
    return {
        "num_tables": int(codebook.size),
        "frequencies_identical": bool(np.array_equal(
            entropy_model.frequencies, codebook.frequencies)),
        "table_index_identical": bool(np.array_equal(m10k_index, identity_index)),
        "payload_identical": bool(m10k_payload == identity_payload),
        "search_payload_identical": bool(m10k_payload == searched_payload),
        "search_index_identical": bool(np.array_equal(m10k_index, searched_index)),
        "round_trip_exact": bool(np.array_equal(round_trip, symbols.reshape(-1))),
        "payload_bytes": len(m10k_payload),
    }


def held_out_bits(codebook, model, symbol_frames, reference_frames, *, device) -> float:
    """Ideal code length per symbol under the codebook, on frames it never saw."""
    total, count = 0.0, 0
    ml = held_out_bits.ml
    for symbols, reference in zip(symbol_frames, reference_frames):
        tensor = torch.from_numpy(reference).float()[None].to(device)
        table_index = ml.frame_table_index(model, codebook, tensor)
        total += codebook.expected_bits(symbols.reshape(-1), table_index)
        count += symbols.size
    return total / count


def m10k_held_out_bits(mk, model, symbol_frames, reference_frames, *, bits,
                       device) -> float:
    """M10K's own held-out ideal bits, measured through its DEPLOYED integer
    tables rather than its float probabilities - the codebook is compared
    against the tables M10K actually codes with, not the ones it would like to.
    """
    total, count = 0.0, 0
    for symbols, reference in zip(symbol_frames, reference_frames):
        tensor = torch.from_numpy(reference).float()[None].to(device)
        entropy_model, table_index = mk.frame_entropy_model(model, tensor, bits=bits)
        probabilities = entropy_model.frequencies / float(TOTAL_FREQUENCY)
        total += float(-np.log2(probabilities[table_index, symbols.reshape(-1)]).sum())
        count += symbols.size
    return total / count


def context_held_out_bits(context_model, entropy_model, symbol_frames, reference_frames):
    probabilities = entropy_model.frequencies / float(TOTAL_FREQUENCY)
    total, count = 0.0, 0
    for symbols, reference in zip(symbol_frames, reference_frames):
        contexts = context_model.contexts(reference)
        table = context_model.table_index(*symbols.shape, contexts)
        total += float(-np.log2(probabilities[table, symbols.reshape(-1)]).sum())
        count += symbols.size
    return total / count


def time_entropy_step(mk, ml, model, codebook, reference_frames, symbol_frames, *,
                      bits, device, repeats: int) -> dict[str, float]:
    """Per-frame wall time of the M10K and M10L entropy steps, on the same frames.

    Both arms run the identical network forward, so that is timed once and the
    two arms are timed for everything AFTER it: for M10K, moving [16384, A]
    probabilities to the host, converting them to integer frequencies and
    building the cumulative array; for M10L, one assignment against K fixed
    prototypes. Timing each directly rather than subtracting the network from an
    end-to-end figure keeps a noisy network measurement from turning into a
    negative table time.
    """
    frames = [(torch.from_numpy(r).float()[None].to(device), s.reshape(-1))
              for r, s in zip(reference_frames[:repeats], symbol_frames[:repeats])]
    result: dict[str, float] = {}

    def m10k_tables(probabilities):
        host = probabilities.double().cpu().numpy()
        frequencies = mk.probabilities_to_frequencies(host)
        return (EmpiricalEntropyModel(frequencies, bits=bits),
                np.arange(frequencies.shape[0], dtype=np.int64))

    for reference, _ in frames[:2]:                      # warm-up, discarded
        probabilities = ml.frame_probabilities(model, reference)
        m10k_tables(probabilities)
        codebook.assign_tensor(probabilities)

    _synchronize(device)
    started = time.perf_counter()
    predicted = [ml.frame_probabilities(model, reference) for reference, _ in frames]
    _synchronize(device)
    result["network_ms"] = (time.perf_counter() - started) / len(frames) * 1000.0

    _synchronize(device)
    started = time.perf_counter()
    built = [m10k_tables(probabilities) for probabilities in predicted]
    _synchronize(device)
    result["m10k_table_ms"] = (time.perf_counter() - started) / len(frames) * 1000.0

    _synchronize(device)
    started = time.perf_counter()
    indices = [codebook.assign_tensor(probabilities) for probabilities in predicted]
    _synchronize(device)
    result["m10l_table_ms"] = (time.perf_counter() - started) / len(frames) * 1000.0

    started = time.perf_counter()
    for (entropy_model, table_index), (_, symbols) in zip(built, frames):
        encode_symbols(symbols, entropy_model.cumulative, table_index)
    result["m10k_encode_ms"] = (time.perf_counter() - started) / len(frames) * 1000.0

    started = time.perf_counter()
    for table_index, (_, symbols) in zip(indices, frames):
        encode_symbols(symbols, codebook.cumulative, table_index)
    result["m10l_encode_ms"] = (time.perf_counter() - started) / len(frames) * 1000.0

    result["m10k_total_ms"] = result["network_ms"] + result["m10k_table_ms"] \
        + result["m10k_encode_ms"]
    result["m10l_total_ms"] = result["network_ms"] + result["m10l_table_ms"] \
        + result["m10l_encode_ms"]
    result["unique_tables_used"] = float(np.mean(
        [ml.unique_tables_used(index) for index in indices]))
    return result


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10L: offline gate for the shared entropy-table codebook.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage", choices=["full", "smoke"], default="full")
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--codebook-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--metric-comparison-bits", type=int, default=4,
                        help="rate point at which L1 is compared against code length")
    parser.add_argument("--fit-rows", type=int, default=100_000)
    parser.add_argument("--fit-iterations", type=int, default=15)
    parser.add_argument("--train-frames", type=int, default=600)
    parser.add_argument("--val-frames", type=int, default=200)
    parser.add_argument("--timing-frames", type=int, default=20)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--quant-mode", choices=["global", "per_channel"],
                        default="per_channel")
    parser.add_argument("--calibration-frames", type=int, default=400,
                        help="MUST match the value the M10K models were fitted under")
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
    mk = _load_script("m10k_learned_entropy")
    ml = _load_script("m10l_shared_codebook")
    held_out_bits.ml = ml

    smoke = args.stage == "smoke"
    sizes = tuple(args.codebook_sizes) if args.codebook_sizes else (
        (16, 64) if smoke else ml.CANDIDATE_K)
    train_frames = 40 if smoke else args.train_frames
    val_frames = 20 if smoke else args.val_frames
    fit_rows = 8_000 if smoke else args.fit_rows
    timing_frames = 5 if smoke else args.timing_frames

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    train_sequences = discover_sequences(args.manifest, split="train")
    val_sequences = discover_sequences(args.manifest, split="val")

    print("=" * 118)
    print("M10L - SHARED ENTROPY-TABLE CODEBOOK: OFFLINE GATE")
    print("=" * 118)
    print(f"  candidate K   : {list(sizes)}")
    print(f"  gate criteria : rate penalty <= {MAX_RATE_PENALTY_PERCENT}% of M10K "
          f"AND table time reduced >= {MIN_TABLE_TIME_REDUCTION_PERCENT}%")
    print("  codebooks fitted on TRAIN, scored on VALIDATION. Test is not opened here.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    codebook_dir = args.output_dir / "codebooks"
    codebook_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "M10L offline gate", "candidate_k": list(sizes),
        "max_rate_penalty_percent": MAX_RATE_PENALTY_PERCENT,
        "min_table_time_reduction_percent": MIN_TABLE_TIME_REDUCTION_PERCENT,
        "checkpoint": str(args.checkpoint), "rate_points": [],
    }

    for bits in args.rate_points:
        model_path = args.model_dir / f"learned_entropy_{bits}bit.pt"
        if not model_path.is_file():
            print(f"[ERROR] M10K model not found: {model_path}. Run M10K first.",
                  file=sys.stderr)
            return 1
        learned, checkpoint = mk.load_entropy_model(model_path, device=device)

        print(f"\n  collecting {bits}-bit symbols ...", flush=True)
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
        print(f"    {len(train_symbols)} train P-frames, {len(val_symbols)} validation")

        # --- STOP CONDITION: the codebook path must be able to BE M10K --------
        probe = torch.from_numpy(val_references[0]).float()[None].to(device)
        with mc.deterministic_kernels():
            reference_check = zero_loss_reference(
                mk, ml, learned, probe, val_symbols[0], bits=bits)
        ok = (reference_check["frequencies_identical"]
              and reference_check["table_index_identical"]
              and reference_check["payload_identical"]
              and reference_check["search_payload_identical"]
              and reference_check["round_trip_exact"])
        print(f"    zero-loss K={reference_check['num_tables']:,} reference: "
              f"frequencies {reference_check['frequencies_identical']} | "
              f"table_index {reference_check['table_index_identical']} | "
              f"payload {reference_check['payload_identical']} | "
              f"search payload {reference_check['search_payload_identical']} | "
              f"round trip {reference_check['round_trip_exact']}")
        if not ok:
            print("\n[ERROR] the codebook path does not reproduce M10K exactly. "
                  "No penalty measured through it would be meaningful. STOPPING.",
                  file=sys.stderr)
            report["stopped"] = {"bits": bits, "zero_loss_reference": reference_check}
            (args.output_dir / "offline_gate.json").write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8")
            return 1

        # --- the three existing arms, on the SAME held-out frames -------------
        baselines = {}
        for scheme in ("marginal", "local_activity4"):
            context_model = ce.fit_context_model(scheme, train_references)
            built = ce.build_conditional_entropy_model(
                train_symbols, train_references, context_model, bits=bits)
            baselines[scheme] = context_held_out_bits(
                context_model, built["entropy_model"], val_symbols, val_references)
        m10k_bits = m10k_held_out_bits(mk, learned, val_symbols, val_references,
                                       bits=bits, device=device)
        print(f"    held-out ideal bits/symbol - M10H {baselines['marginal']:.5f}  "
              f"M10J {baselines['local_activity4']:.5f}  M10K {m10k_bits:.5f}")

        # --- fit codebooks on TRAIN -------------------------------------------
        print(f"    sampling TRAIN distributions for codebook fitting ...", flush=True)
        samples = ml.sample_training_distributions(
            learned, train_references, device=device, max_rows=fit_rows, seed=args.seed)
        print(f"      {samples.shape[0]:,} rows x {samples.shape[1]} symbols")

        metrics = ["code_length"]
        if bits == args.metric_comparison_bits:
            metrics.append("l1")

        candidates = []
        for metric in metrics:
            for size in sizes:
                started = time.perf_counter()
                codebook = ml.fit_codebook(
                    samples, size, bits=bits, metric=metric, seed=args.seed,
                    max_iterations=args.fit_iterations,
                    provenance={"calibration_frames": args.calibration_frames,
                                "train_frames": len(train_symbols),
                                "m10k_model": str(model_path)})
                fit_seconds = time.perf_counter() - started
                with mc.deterministic_kernels():
                    codebook_bits = held_out_bits(
                        codebook, learned, val_symbols, val_references, device=device)
                    timing = time_entropy_step(
                        mk, ml, learned, codebook, val_references, val_symbols,
                        bits=bits, device=device, repeats=timing_frames)
                penalty = (codebook_bits - m10k_bits) / m10k_bits * 100.0
                row = {
                    "codebook_size": size, "metric": metric,
                    "held_out_bits_per_symbol": codebook_bits,
                    "rate_penalty_vs_m10k_percent": penalty,
                    "vs_m10j_percent": (baselines["local_activity4"] - codebook_bits)
                                       / baselines["local_activity4"] * 100.0,
                    "vs_m10h_percent": (baselines["marginal"] - codebook_bits)
                                       / baselines["marginal"] * 100.0,
                    "fit_seconds": fit_seconds,
                    "occupied_clusters": codebook.provenance["occupied_clusters"],
                    "lloyd_iterations": codebook.provenance["iterations"],
                    "codebook_memory_bytes": codebook.table_memory_bytes(),
                    "codebook_id": codebook.codebook_id(
                        model_identity=b"", calibration_signature="").hex(),
                    **{k: float(v) for k, v in timing.items()},
                }
                row["table_time_reduction_percent"] = (
                    (timing["m10k_table_ms"] - timing["m10l_table_ms"])
                    / timing["m10k_table_ms"] * 100.0)
                row["total_time_reduction_percent"] = (
                    (timing["m10k_total_ms"] - timing["m10l_total_ms"])
                    / timing["m10k_total_ms"] * 100.0)
                row["passes_gate"] = bool(
                    penalty <= MAX_RATE_PENALTY_PERCENT
                    and row["table_time_reduction_percent"]
                    >= MIN_TABLE_TIME_REDUCTION_PERCENT)
                candidates.append(row)
                (codebook_dir / f"codebook_{bits}bit_K{size}_{metric}.json").write_text(
                    json.dumps(codebook.to_dict()), encoding="utf-8")

        m10k_table_ms = candidates[0]["m10k_table_ms"]
        m10k_memory = 16384 * ((2 ** bits) + (2 ** bits + 1)) * 8
        report["rate_points"].append({
            "bits": bits, "alphabet": 2 ** bits,
            "train_p_frames": len(train_symbols), "val_p_frames": len(val_symbols),
            "m10h_marginal_val_bits": baselines["marginal"],
            "m10j_conditional_val_bits": baselines["local_activity4"],
            "m10k_learned_val_bits": m10k_bits,
            "m10k_table_ms": m10k_table_ms,
            "m10k_num_tables": 16384,
            "m10k_table_memory_bytes": m10k_memory,
            "zero_loss_reference": reference_check,
            "fit_rows": int(samples.shape[0]),
            "candidates": candidates,
            "m10k_selection_epoch": checkpoint["selection"]["selected_epoch"],
        })

        print(f"\n    {'K':>6} {'metric':<12} {'bits/sym':>10} {'vs M10K':>9} "
              f"{'vs M10J':>9} {'tbl ms':>8} {'vs M10K tbl':>12} {'used':>7}  gate")
        for row in candidates:
            print(f"    {row['codebook_size']:>6} {row['metric']:<12} "
                  f"{row['held_out_bits_per_symbol']:>10.5f} "
                  f"{row['rate_penalty_vs_m10k_percent']:>+8.2f}% "
                  f"{row['vs_m10j_percent']:>+8.2f}% "
                  f"{row['m10l_table_ms']:>8.3f} "
                  f"{row['table_time_reduction_percent']:>11.1f}% "
                  f"{row['unique_tables_used']:>7.0f}  "
                  f"{'PASS' if row['passes_gate'] else '-'}")
        print(f"    (M10K reference: {m10k_table_ms:.3f} ms of table work per P-frame, "
              f"16,384 tables)")

    print()
    print("=" * 118)
    print("METRIC SELECTION (validation only)")
    print("=" * 118)
    comparison = next((r for r in report["rate_points"]
                       if r["bits"] == args.metric_comparison_bits), None)
    if comparison is None or len({c["metric"] for c in comparison["candidates"]}) < 2:
        chosen_metric = ml.DEFAULT_METRIC
        print(f"  only one metric evaluated; using {chosen_metric}")
    else:
        chosen_metric = ml.select_metric(comparison["candidates"])
        for metric in sorted({c["metric"] for c in comparison["candidates"]}):
            best = ml.select_codebook_size(
                [c for c in comparison["candidates"] if c["metric"] == metric])
            summary = ("no passing candidate" if best is None else
                       f"best K={best['codebook_size']} at "
                       f"{best['held_out_bits_per_symbol']:.5f} bits/symbol "
                       f"({best['rate_penalty_vs_m10k_percent']:+.2f}% vs M10K)")
            print(f"    {metric:<12} {summary}")
        print(f"  selected: {chosen_metric} "
              f"(on {args.metric_comparison_bits}-bit validation, before any test data)")
    report["selected_metric"] = chosen_metric

    print()
    print("=" * 118)
    print("GATE")
    print("=" * 118)
    selected: dict[int, Any] = {}
    for rate_point in report["rate_points"]:
        best = ml.select_codebook_size(
            [c for c in rate_point["candidates"] if c["metric"] == chosen_metric])
        if best is not None:
            selected[rate_point["bits"]] = best
    report["gate_passed"] = len(selected) == len(report["rate_points"])
    report["selected"] = {str(k): v for k, v in selected.items()}
    if selected:
        print("  Selected K per rate point - best held-out rate among candidates that")
        print("  clear the pre-registered runtime bar (ties to the smaller codebook):")
        for bits in sorted(selected, reverse=True):
            row = selected[bits]
            print(f"    {bits}-bit: K={row['codebook_size']:<4} "
                  f"rate {row['rate_penalty_vs_m10k_percent']:+.2f}% vs M10K, "
                  f"table time {row['table_time_reduction_percent']:.1f}% lower, "
                  f"still {row['vs_m10j_percent']:+.2f}% better than M10J")
    missing = [r["bits"] for r in report["rate_points"] if r["bits"] not in selected]
    if missing:
        print(f"  NOT PASSED at {missing} - no K met both the rate and the runtime "
              f"criterion there.")
        print("  The full frontier is in the report; that is the result.")
    else:
        print("  GATE PASSED at every rate point.")

    path = args.output_dir / ("smoke_gate.json" if smoke else "offline_gate.json")
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
