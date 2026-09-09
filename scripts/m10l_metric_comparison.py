"""M10L - how should a predicted distribution be assigned to a prototype?

The milestone brief asks for L1 to be considered alongside the information-
theoretic metrics, and for the choice to be made on VALIDATION. Comparing two
separately-fitted codebooks would confound two variables at once (which
prototypes exist, and how rows are routed to them), so this compares three
things instead:

  A  code-length assignment against the code-length-fitted codebook  [deployed]
  B  L1 assignment against the SAME codebook  - isolates the routing rule alone
  C  L1 assignment against an L1-fitted codebook - L1 done properly end to end,
     with the component-wise median centroid its objective actually requires

KL is not a fourth arm: KL(p||q) = H(p,q) - H(p) and H(p) does not depend on q,
so it selects exactly the prototype code length does. That equivalence is
asserted here on the real distributions rather than only argued, and pinned by
a unit test.

All three are scored by the quantity that turns into bytes - the ideal code
length of the REAL M10H residual symbols under the assigned tables - on
validation P-frames the codebooks were never fitted to. Test data is not opened.

Run:
  ./.venv/Scripts/python.exe scripts/m10l_metric_comparison.py --bits 4
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

from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m10l_shared_codebook")
DEFAULT_MODEL_DIR = Path("outputs/m10k_learned_entropy")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(ml, codebook, model, symbol_frames, reference_frames, *, metric, device):
    """Ideal bits/symbol when rows are routed by `metric` into `codebook`."""
    total, count = 0.0, 0
    for symbols, reference in zip(symbol_frames, reference_frames):
        tensor = torch.from_numpy(reference).float()[None].to(device)
        probabilities = ml.frame_probabilities(model, tensor).double().cpu().numpy()
        costs = ml.prototype_costs(probabilities, codebook.probabilities, metric=metric)
        table_index = ml.argmin_lowest_index(costs)
        total += codebook.expected_bits(symbols.reshape(-1), table_index)
        count += symbols.size
    return total / count


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10L: assignment-metric comparison, on validation only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=None,
                        help="default: the K the offline gate selected at this rate point")
    parser.add_argument("--fit-rows", type=int, default=100_000)
    parser.add_argument("--fit-iterations", type=int, default=15)
    parser.add_argument("--train-frames", type=int, default=600)
    parser.add_argument("--val-frames", type=int, default=200)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--quant-mode", choices=["global", "per_channel"],
                        default="per_channel")
    parser.add_argument("--calibration-frames", type=int, default=400)
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

    size = args.codebook_size
    gate_path = args.output_dir / "offline_gate.json"
    if size is None:
        if not gate_path.is_file():
            print(f"[ERROR] no --codebook-size and no gate report at {gate_path}",
                  file=sys.stderr)
            return 1
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        selected = gate.get("selected", {}).get(str(args.bits))
        if selected is None:
            print(f"[ERROR] the gate selected no K at {args.bits}-bit", file=sys.stderr)
            return 1
        size = selected["codebook_size"]

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    learned, _ = mk.load_entropy_model(
        args.model_dir / f"learned_entropy_{args.bits}bit.pt", device=device)

    print("=" * 104)
    print("M10L - ASSIGNMENT METRIC COMPARISON (validation only)")
    print("=" * 104)
    print(f"  rate point {args.bits}-bit, K={size}")
    print("  scored by ideal code length of the real M10H residual symbols")

    print("\n  collecting symbols ...", flush=True)
    calibration = mc.calibrate_grids(
        model, discover_sequences(args.manifest, split="train"), bits=args.bits,
        mode=args.quant_mode, gop_size=args.gop, block_size=args.block_size,
        search_range=args.search_range, reference_mode="mc",
        max_frames=args.calibration_frames)
    train_symbols, train_references = ce.collect_training_symbols(
        mc, model, discover_sequences(args.manifest, split="train"), calibration,
        gop_size=args.gop, block_size=args.block_size, search_range=args.search_range,
        device=device, max_frames=args.train_frames)
    val_symbols, val_references = ce.collect_training_symbols(
        mc, model, discover_sequences(args.manifest, split="val"), calibration,
        gop_size=args.gop, block_size=args.block_size, search_range=args.search_range,
        device=device, max_frames=args.val_frames)
    print(f"    {len(train_symbols)} train P-frames, {len(val_symbols)} validation")

    samples = ml.sample_training_distributions(
        learned, train_references, device=device, max_rows=args.fit_rows, seed=args.seed)
    codebooks = {}
    for metric in ("code_length", "l1"):
        started = time.perf_counter()
        codebooks[metric] = ml.fit_codebook(
            samples, size, bits=args.bits, metric=metric, seed=args.seed,
            max_iterations=args.fit_iterations)
        print(f"    fitted {metric:<12} codebook in {time.perf_counter() - started:6.1f}s "
              f"({codebooks[metric].provenance['occupied_clusters']} clusters occupied)")

    with mc.deterministic_kernels():
        arms = [
            ("A  code-length fit, code-length assignment  [deployed]",
             "code_length", "code_length"),
            ("B  code-length fit, L1 assignment",
             "code_length", "l1"),
            ("C  L1 fit (median centroid), L1 assignment",
             "l1", "l1"),
        ]
        rows = []
        for label, fitted, assigned in arms:
            bits_per_symbol = score(ml, codebooks[fitted], learned, val_symbols,
                                    val_references, metric=assigned, device=device)
            rows.append({"label": label, "fitted_metric": fitted,
                         "assignment_metric": assigned,
                         "held_out_bits_per_symbol": bits_per_symbol})

        # KL is not a separate arm: it selects the same prototype code length
        # does. Asserted here on the real distributions, not only argued.
        probe = ml.frame_probabilities(
            learned, torch.from_numpy(val_references[0]).float()[None].to(device)
        ).double().cpu().numpy()
        prototypes = codebooks["code_length"].probabilities
        kl_matches = bool(np.array_equal(
            ml.argmin_lowest_index(ml.prototype_costs(probe, prototypes, metric="kl")),
            ml.argmin_lowest_index(ml.prototype_costs(probe, prototypes,
                                                      metric="code_length"))))

    baseline = rows[0]["held_out_bits_per_symbol"]
    print(f"\n  {'arm':<52} {'bits/symbol':>13} {'vs A':>9}")
    for row in rows:
        row["vs_deployed_percent"] = (
            (row["held_out_bits_per_symbol"] - baseline) / baseline * 100)
        print(f"  {row['label']:<52} {row['held_out_bits_per_symbol']:>13.5f} "
              f"{row['vs_deployed_percent']:>+8.2f}%")
    print(f"\n  KL and code-length assignment agree on every position: {kl_matches}")

    best = min(rows, key=lambda r: r["held_out_bits_per_symbol"])
    print(f"  selected on validation: {best['assignment_metric']} assignment "
          f"({best['label'].split()[0]})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "M10L assignment metric comparison", "bits": args.bits,
        "codebook_size": size, "val_p_frames": len(val_symbols),
        "train_p_frames": len(train_symbols), "fit_rows": int(samples.shape[0]),
        "kl_matches_code_length": kl_matches, "arms": rows,
        "selected_metric": best["assignment_metric"],
    }
    path = args.output_dir / f"metric_comparison_{args.bits}bit.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
