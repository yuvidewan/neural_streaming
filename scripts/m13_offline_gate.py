"""M13 Phase C - independently reproduce M12's recalibration-gain estimate,
through the ACTUAL integer-quantized frequency table this time (not the
float `cx.parent_bits` estimate M12's gate used for a quick diagnostic).

M12's offline gate reported a "m11_g16_recalibrated" baseline (+1.604% /
+2.916% / +5.077% at 5/4/3-bit) computed directly from FLOAT smoothed
probabilities (`cx.smoothed_parent`), never quantized to the coder's integer
frequency format. Before trusting that number as a real lever, M13 rebuilds
it independently, end to end, through `m13_recalibration.fit_recalibrated_frequencies`
-> `m10k_learned_entropy.probabilities_to_frequencies` -> `SharedCodebook` -
i.e. the EXACT table the arithmetic coder would actually use - and measures
held-out bits/symbol from THAT integer table, on VAL-B, having chosen the
smoothing strength on VAL-A and fit counts on TRAIN only.

If this number differs materially from M12's float estimate, that gap is
integer-quantization overhead and must be reported as such, not folded
silently into the headline. Phase D then goes one step further, through
the real arithmetic coder.

Run:
  ./.venv/Scripts/python.exe scripts/m13_offline_gate.py
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

from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m13_recalibration")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M11_DIR = Path("outputs/m11_autoregressive_entropy")
RATE_POINTS = (5, 4, 3)

WEAK_BELOW_PERCENT = 0.5
MEANINGFUL_ABOVE_PERCENT = 1.0


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percent(baseline: float, value: float) -> float:
    return (baseline - value) / baseline * 100.0


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M13 Phase C: independent, integer-quantized recalibration gate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=DEFAULT_M11_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--cache-dir", type=Path, default=None)
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

    md = _load_script("m11_data")
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    m13 = _load_script("m13_recalibration")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    print("=" * 110, flush=True)
    print("M13 PHASE C - INDEPENDENT, INTEGER-QUANTIZED RECALIBRATION GATE")
    print("=" * 110)
    print("  TRAIN fits counts, VAL-A selects smoothing strength, VAL-B reports. TEST untouched.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"phase": "M13 Phase C offline gate", "checkpoint": str(args.checkpoint),
                              "weak_below_percent": WEAK_BELOW_PERCENT,
                              "meaningful_above_percent": MEANINGFUL_ABOVE_PERCENT,
                              "rate_points": []}

    from nvc.training.checkpoint import load_model_from_checkpoint

    for bits in args.rate_points:
        print(f"\n  ---- {bits}-bit ----", flush=True)
        base_model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
        base_model.eval()
        data = md.load_or_collect(
            base_model, checkpoint=args.checkpoint, manifest=args.manifest, bits=bits,
            device=device, cache_dir=cache_dir, log=lambda m: print(m, flush=True))
        train_symbols = data["train_symbols"].astype(np.int64)
        val_symbols = data["val_symbols"].astype(np.int64)
        select_mask, report_mask = md.split_validation(data)
        print(f"    {len(train_symbols)} train P-frames | validation: "
             f"{int(select_mask.sum())} VAL-A + {int(report_mask.sum())} VAL-B P-frames "
             f"from {len(data['val_sequence_ids'])} sequences", flush=True)

        model, checkpoint = ma.load_model(args.m11_dir / f"m11_G16_entropy_{bits}bit.pt",
                                          device=device)
        codebook = ml.SharedCodebook.from_dict(json.loads(
            (args.m11_dir / f"m11_G16_codebook_{bits}bit_K512.json").read_text(encoding="utf-8")))

        cx = _load_script("m11_causal_context")
        channels, height, width = train_symbols.shape[1:]
        zero = torch.from_numpy(cx.zero_symbols(data["calibration"]["residual_params"], channels))

        started = time.perf_counter()
        train_k = m13.m11_g16_prototype_indices(model, codebook, train_symbols,
                                                data["train_references"], zero, device=device)
        val_k = m13.m11_g16_prototype_indices(model, codebook, val_symbols,
                                              data["val_references"], zero, device=device)
        print(f"    M11-G16 prototype indices for {len(train_k) + len(val_k)} frames in "
             f"{time.perf_counter() - started:.1f}s", flush=True)

        flat = lambda a: a.reshape(-1)
        t_sym, t_k = flat(train_symbols), flat(train_k)
        a_sym, a_k = flat(val_symbols[select_mask]), flat(val_k[select_mask])
        b_sym, b_k = flat(val_symbols[report_mask]), flat(val_k[report_mask])

        frequencies, strength, val_a_bits, val_a_baseline_bits = m13.fit_recalibrated_frequencies(
            codebook, t_sym, t_k, a_sym, a_k, alphabet=2 ** bits)
        coding_codebook = m13.build_recalibrated_codebook(
            codebook, frequencies, provenance={"bits": bits, "strength": strength})

        h_old = -np.sum(np.log2(codebook.probabilities[b_k, b_sym])) / b_sym.size
        h_new = -np.sum(np.log2(coding_codebook.probabilities[b_k, b_sym])) / b_sym.size
        gain = _percent(h_old, h_new)
        verdict = ("meaningful" if gain >= MEANINGFUL_ABOVE_PERCENT else
                  "marginal" if gain >= WEAK_BELOW_PERCENT else "weak")

        print(f"    VAL-A: baseline {val_a_baseline_bits:.5f}  selected strength {strength:g}  "
             f"recalibrated {val_a_bits:.5f} bits/symbol", flush=True)
        print(f"    VAL-B: H_old (deployed) {h_old:.5f}   H_new (recalibrated, INTEGER table) "
             f"{h_new:.5f}   gain {gain:+.3f}%   -> {verdict}", flush=True)

        report["rate_points"].append({
            "bits": bits, "alphabet": 2 ** bits, "codebook_size": codebook.size,
            "train_p_frames": int(len(train_symbols)), "val_a_p_frames": int(select_mask.sum()),
            "val_b_p_frames": int(report_mask.sum()), "val_sequence_ids": data["val_sequence_ids"],
            "smoothing_strength": strength, "val_a_baseline_bits_per_symbol": val_a_baseline_bits,
            "val_a_recalibrated_bits_per_symbol": val_a_bits,
            "val_b_h_old_bits_per_symbol": h_old, "val_b_h_new_bits_per_symbol": h_new,
            "val_b_gain_percent": gain, "verdict": verdict,
            "old_codebook_id": codebook.codebook_id().hex(),
            "new_codebook_id": coding_codebook.codebook_id().hex(),
        })

    print()
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)
    for rp in report["rate_points"]:
        print(f"  {rp['bits']}-bit  H_old {rp['val_b_h_old_bits_per_symbol']:.5f}  "
             f"H_new {rp['val_b_h_new_bits_per_symbol']:.5f}  "
             f"gain {rp['val_b_gain_percent']:+.3f}%  -> {rp['verdict']}")
    all_meaningful = all(rp["verdict"] == "meaningful" for rp in report["rate_points"])
    report["all_meaningful"] = all_meaningful
    print(f"\n  ALL RATE POINTS MEANINGFUL (>= {MEANINGFUL_ABOVE_PERCENT}%): {all_meaningful}")

    path = args.output_dir / "m13_offline_gate.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
