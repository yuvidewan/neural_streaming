"""M10L step 1 - profile the existing M10K P-frame entropy path, stage by stage.

The M10K report already said "table construction is 97-99% of the cost", but
that came from two coarse timers. Before optimising anything, this script
decomposes the path into the eight stages the milestone brief names and times
each one on real DAVIS data, so the optimisation targets a measured bottleneck
rather than a remembered one.

The decomposition is verified against `m10k_learned_entropy.frame_entropy_model`
on every frame: same frequencies, same table_index, same payload. A profile of
a path that is not the real path would be worthless.

Two things this measures that the M10K benchmark's `residual_coding_seconds`
conflated:

  * the ideal-bits diagnostic (a [16384, A] float64 log2 over the whole table
    array) was inside the timed region, and is not deployment work;
  * the M10K benchmark JSON was written BEFORE the table-build vectorisation,
    so its numbers describe code that no longer exists.

Both are reported here separately and honestly.

Run:
  ./.venv/Scripts/python.exe scripts/m10l_profile.py --frames 24
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.entropy_model import TOTAL_FREQUENCY, EmpiricalEntropyModel
from nvc.compression.range_coder import encode_symbols
from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device

DEFAULT_OUTPUT_DIR = Path("outputs/m10l_shared_codebook")
DEFAULT_MODEL_DIR = Path("outputs/m10k_learned_entropy")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
RATE_POINTS = (5, 4, 3)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stopwatch:
    """Accumulates per-stage wall time, synchronising CUDA before each reading.

    Without the synchronise, every GPU stage would appear free and its real
    cost would land on whichever later stage first touched the result.
    """

    def __init__(self, device) -> None:
        self.device = device
        self.totals: dict[str, float] = {}
        self._mark = self._now()

    def _now(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def restart(self) -> None:
        self._mark = self._now()

    def lap(self, stage: str) -> None:
        now = self._now()
        self.totals[stage] = self.totals.get(stage, 0.0) + (now - self._mark)
        self._mark = now


@torch.no_grad()
def profile_frame(model, entropy_model_net, reference, symbols, *, bits, watch):
    """One P-frame through the M10K path, timed stage by stage.

    Returns the same (frequencies, table_index, payload) the production path
    produces, so the caller can assert this decomposition IS that path.
    """
    watch.restart()

    logits = entropy_model_net.forward(reference)
    watch.lap("1_network_forward")

    log_probabilities = torch.log_softmax(logits, dim=2)
    probabilities = log_probabilities.exp()[0]
    channels, alphabet, height, width = probabilities.shape
    flat = probabilities.permute(0, 2, 3, 1).reshape(channels * height * width, alphabet)
    watch.lap("2_probability_normalization")

    host = flat.double().cpu().numpy()
    watch.lap("8_device_to_host_transfer")

    mk = profile_frame.mk
    frequencies = mk.probabilities_to_frequencies(host)
    watch.lap("3_float_to_int_frequencies")

    built = EmpiricalEntropyModel(frequencies, bits=bits)
    watch.lap("4_table_validation_and_alloc")

    cumulative = built.cumulative
    watch.lap("5_cumulative_materialization")

    table_index = np.arange(frequencies.shape[0], dtype=np.int64)
    watch.lap("6_table_index")

    payload = encode_symbols(symbols.reshape(-1), cumulative, table_index)
    watch.lap("7_arithmetic_encoding")

    # Diagnostic only - present in the M10K benchmark's timing region but not
    # deployment work. Timed separately so the deployed cost is not overstated.
    probabilities_host = frequencies.astype(np.float64) / TOTAL_FREQUENCY
    ideal = float(-np.log2(probabilities_host[table_index, symbols.reshape(-1)]).sum())
    watch.lap("d_ideal_bits_diagnostic")

    return frequencies, table_index, payload, ideal


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10L: stage-by-stage profile of the M10K entropy path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--frames", type=int, default=24,
                        help="P-frames to profile per rate point")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--quant-mode", choices=["global", "per_channel"],
                        default="per_channel")
    parser.add_argument("--calibration-frames", type=int, default=400,
                        help="MUST match the value the M10K models were fitted under")
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
    profile_frame.mk = mk

    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    train_sequences = discover_sequences(args.manifest, split="train")

    print("=" * 108)
    print("M10L STEP 1 - PROFILE OF THE EXISTING M10K P-FRAME ENTROPY PATH")
    print("=" * 108)
    print(f"  device: {device}   frames per rate point: {args.frames} "
          f"(+{args.warmup} warm-up, discarded)")
    print("  every stage verified against m10k_learned_entropy.frame_entropy_model")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"phase": "M10L M10K path profile", "rate_points": []}

    for bits in args.rate_points:
        model_path = args.model_dir / f"learned_entropy_{bits}bit.pt"
        if not model_path.is_file():
            print(f"[ERROR] M10K model not found: {model_path}", file=sys.stderr)
            return 1
        learned, _ = mk.load_entropy_model(model_path, device=device)

        calibration = mc.calibrate_grids(
            model, train_sequences, bits=bits, mode=args.quant_mode, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            reference_mode="mc", max_frames=args.calibration_frames)
        symbols, references = ce.collect_training_symbols(
            mc, model, train_sequences, calibration, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, device=device,
            max_frames=args.frames + args.warmup + 20)
        usable = min(len(symbols), args.frames + args.warmup)
        if usable <= args.warmup:
            print(f"[ERROR] only {usable} P-frames collected at {bits}-bit",
                  file=sys.stderr)
            return 1

        watch = Stopwatch(device)
        matched, table_counts, payload_bytes = True, set(), 0
        peak_bytes = 0
        for index in range(usable):
            reference = torch.from_numpy(references[index]).float()[None].to(device)
            frame_symbols = symbols[index]
            if index == args.warmup:
                watch.totals.clear()
            trace = index == usable - 1
            if trace:
                tracemalloc.start()
            frequencies, table_index, payload, _ = profile_frame(
                model, learned, reference, frame_symbols, bits=bits, watch=watch)
            if trace:
                peak_bytes = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
            if index >= args.warmup:
                reference_model, reference_index = mk.frame_entropy_model(
                    learned, reference, bits=bits)
                matched &= bool(np.array_equal(
                    reference_model.frequencies, frequencies))
                matched &= bool(np.array_equal(reference_index, table_index))
                matched &= payload == encode_symbols(
                    frame_symbols.reshape(-1), reference_model.cumulative, reference_index)
                table_counts.add(int(frequencies.shape[0]))
                payload_bytes += len(payload)

        counted = usable - args.warmup
        deployed = {k: v for k, v in watch.totals.items() if not k.startswith("d_")}
        total = sum(deployed.values())
        stages = []
        for stage in sorted(watch.totals):
            milliseconds = watch.totals[stage] / counted * 1000.0
            stages.append({
                "stage": stage, "ms_per_p_frame": milliseconds,
                "percent_of_deployed": (watch.totals[stage] / total * 100.0
                                        if not stage.startswith("d_") else None),
                "diagnostic_only": stage.startswith("d_"),
            })

        num_tables = table_counts.pop() if len(table_counts) == 1 else -1
        model_object = EmpiricalEntropyModel(
            np.full((num_tables, 2 ** bits), TOTAL_FREQUENCY // (2 ** bits),
                    dtype=np.int64), bits=bits)
        row = {
            "bits": bits, "alphabet": 2 ** bits, "p_frames_timed": counted,
            "matches_m10k_production_path": matched,
            "num_probability_tables": num_tables,
            "frequencies_per_table": 2 ** bits,
            "probability_tensor_shape": [num_tables, 2 ** bits],
            "frequency_bytes": int(model_object.frequencies.nbytes),
            "cumulative_bytes": int(model_object.cumulative.nbytes),
            "python_peak_alloc_bytes": int(peak_bytes),
            "total_deployed_ms_per_p_frame": total / counted * 1000.0,
            "residual_bytes_per_frame": payload_bytes / counted,
            "stages": stages,
        }
        report["rate_points"].append(row)

        print(f"\n  {bits}-bit  (alphabet {2 ** bits}, {num_tables:,} tables, "
              f"{counted} P-frames)   production path match: {matched}")
        print(f"    {'stage':<32} {'ms/P-frame':>12} {'% deployed':>12}")
        for stage in stages:
            share = ("     n/a" if stage["percent_of_deployed"] is None
                     else f"{stage['percent_of_deployed']:>11.1f}%")
            print(f"    {stage['stage']:<32} {stage['ms_per_p_frame']:>12.3f} {share}")
        print(f"    {'TOTAL (deployed)':<32} {row['total_deployed_ms_per_p_frame']:>12.3f}"
              f" {'100.0%':>12}")
        print(f"    tables: {num_tables:,} x {2 ** bits}   "
              f"frequencies {row['frequency_bytes'] / 1e6:.2f} MB   "
              f"cumulative {row['cumulative_bytes'] / 1e6:.2f} MB   "
              f"peak python alloc {peak_bytes / 1e6:.2f} MB")

    if not all(r["matches_m10k_production_path"] for r in report["rate_points"]):
        print("\n[ERROR] the profiled decomposition does not reproduce the M10K path; "
              "the profile cannot be trusted. STOPPING.", file=sys.stderr)
        return 1

    print()
    print("=" * 108)
    print("BOTTLENECK")
    print("=" * 108)
    for row in report["rate_points"]:
        deployed = [s for s in row["stages"] if not s["diagnostic_only"]]
        worst = max(deployed, key=lambda s: s["ms_per_p_frame"])
        table_stages = [s for s in deployed
                        if s["stage"].startswith(("3_", "4_", "5_"))]
        table_share = sum(s["percent_of_deployed"] for s in table_stages)
        row["table_construction_percent"] = table_share
        row["table_construction_ms"] = sum(s["ms_per_p_frame"] for s in table_stages)
        print(f"  {row['bits']}-bit: largest stage {worst['stage']} "
              f"({worst['ms_per_p_frame']:.2f} ms, {worst['percent_of_deployed']:.1f}%); "
              f"table construction (3+4+5) {row['table_construction_ms']:.2f} ms "
              f"= {table_share:.1f}% of deployed cost")

    path = args.output_dir / "m10k_path_profile.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
