"""M12 Phase B1-B3 - does spatial causal context add entropy reduction over
the ACTUAL deployed M11-G16 model, once decoding it is resumable?

M11's own offline gate (scripts/m11_offline_gate.py) already measured spatial
contexts (left/up/left_up/...) and channel contexts (prev_channel,
channel_activity) against M10L - the z_ref-only baseline - and found both
families informative, with a context combining both beating channel alone by
~0.4-1.4 net percentage points (outputs/m11_autoregressive_entropy/
offline_gate.json). But M11-G16 - the model actually deployed - is a MUCH
richer channel predictor than that gate's simple "prev_channel"/
"channel_activity" counting contexts: it is a trained network conditioned on
every earlier channel's full spatial layout, group size 16. The question this
script answers is the one M12 actually needs: does spatial context still add
anything once the parent model IS M11-G16, not a simple channel count?

METHODOLOGY (identical to m11_offline_gate.py, parent swapped)
----------------------------------------------------------------
M11-G16 assigns every position to one of its own 512 codebook prototypes k
(scripts/m11_ar_entropy.py + m10l_shared_codebook.py's assignment, run here
with TRUE previously-decoded symbols - exactly what the encoder does, and
what a lossless decoder reconstructs bit-for-bit). Three arms, on VAL-B,
tuned on VAL-A, TRAIN never touching VAL-A/VAL-B/TEST:

  m11_g16_deployed        the actual shipped probabilities: codebook.probabilities[k, s]
  m11_g16_recalibrated    P(s | k) refit on TRAIN's true symbols - NOT spatial
                          context, reported separately so it cannot be confused
                          for one (M12 spec section 12)
  + spatial context       P(s | k, ctx) vs P(s | k), both fit the same way,
                          net of a context-shuffled random control (same
                          whole-split permutation M11 used, not a per-frame
                          shuffle - see m11_causal_context.permuted_context's
                          docstring for why that distinction matters)

Gate thresholds are M11's pre-registered ones: <0.5% net = weak, 0.5-1.0% =
marginal, >1.0% = meaningful (scripts/m11_offline_gate.py).

Run:
  ./.venv/Scripts/python.exe scripts/m12_spatial_offline_gate.py
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

DEFAULT_OUTPUT_DIR = Path("outputs/m12_spatial_offline_gate")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M11_DIR = Path("outputs/m11_autoregressive_entropy")
RATE_POINTS = (5, 4, 3)
DEFAULT_CONTEXTS = ("left", "up", "left_up", "left_up_upleft", "neighbourhood")

WEAK_BELOW_PERCENT = 0.5
MEANINGFUL_ABOVE_PERCENT = 1.0
CONTROL_TOLERANCE_PERCENT = 0.1


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percent(baseline: float, value: float) -> float:
    return (baseline - value) / baseline * 100.0


@torch.no_grad()
def m11_g16_prototype_indices(model, codebook, symbols: np.ndarray, references: np.ndarray,
                              zero: torch.Tensor, *, device, batch_size: int = 8) -> np.ndarray:
    """M11-G16's own codebook prototype index k at every position of every
    frame, using the TRUE previously-decoded symbols as context - exactly the
    ONE parallel forward pass `nll_bits`/`encode_frame` already use, valid
    because coding is lossless and the decoder ends up holding these exact
    symbols. Returns [N, C*H*W] indices in the coder's C-major flat order.
    """
    zero_d = zero.to(device)
    out = []
    for start in range(0, len(symbols), batch_size):
        batch_symbols = torch.from_numpy(
            symbols[start:start + batch_size].astype(np.int64)).to(device)
        batch_references = torch.from_numpy(
            references[start:start + batch_size].astype(np.float32)).to(device)
        log_probabilities = model.log_probabilities(
            batch_references, model.planes(batch_symbols, zero_d))  # [B, C, A, H, W]
        probabilities = log_probabilities.exp()
        b, c, a, h, w = probabilities.shape
        rows = probabilities.permute(0, 1, 3, 4, 2).reshape(b * c * h * w, a)
        assigned = codebook.assign_tensor(rows)
        out.append(assigned.reshape(b, c * h * w))
    return np.concatenate(out, axis=0)


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M12 Phase B1-B3: spatial context gate against the deployed M11-G16 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=DEFAULT_M11_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--contexts", nargs="+", default=list(DEFAULT_CONTEXTS))
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
    cx = _load_script("m11_causal_context")
    contexts = args.contexts

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    print("=" * 116, flush=True)
    print("M12 PHASE B1-B3 - SPATIAL CAUSAL CONTEXT vs THE DEPLOYED M11-G16 MODEL")
    print("=" * 116)
    print("  baseline: M11-G16 (channel-autoregressive, group 16, 512-entry codebook) - "
         "the strongest DEPLOYED model")
    print("  tuning on VAL-A, every reported number on VAL-B (disjoint validation sequences)")
    print(f"  contexts: {contexts}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "M12 Phase B1-B3 offline gate", "checkpoint": str(args.checkpoint),
        "parent_model": "m11_G16 (group_size=16, K=512 codebook)",
        "weak_below_percent": WEAK_BELOW_PERCENT,
        "meaningful_above_percent": MEANINGFUL_ABOVE_PERCENT,
        "control_tolerance_percent": CONTROL_TOLERANCE_PERCENT,
        "scan_order": "C-major raster: i = c*H*W + y*W + x",
        "contexts": {name: {"family": cx.CONTEXTS[name][2], "description": cx.CONTEXTS[name][3],
                            "definition_id": cx.context_definition_id(name)}
                    for name in contexts},
        "rate_points": [],
    }

    from nvc.training.checkpoint import load_model_from_checkpoint

    for bits in args.rate_points:
        alphabet = 2 ** bits
        print(f"\n  ---- {bits}-bit ----", flush=True)
        base_model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
        base_model.eval()
        data = md.load_or_collect(
            base_model, checkpoint=args.checkpoint, manifest=args.manifest, bits=bits,
            device=device, cache_dir=cache_dir, log=lambda m: print(m, flush=True))
        train_symbols = data["train_symbols"].astype(np.int64)
        val_symbols = data["val_symbols"].astype(np.int64)
        select_mask, report_mask = md.split_validation(data)
        channels, height, width = train_symbols.shape[1:]
        zero_np = cx.zero_symbols(data["calibration"]["residual_params"], channels)
        zero = torch.from_numpy(zero_np)
        print(f"    {len(train_symbols)} train P-frames | validation: "
             f"{int(select_mask.sum())} VAL-A + {int(report_mask.sum())} VAL-B P-frames "
             f"from {len(data['val_sequence_ids'])} sequences", flush=True)

        model, _ = ma.load_model(args.m11_dir / f"m11_G16_entropy_{bits}bit.pt", device=device)
        codebook = ml.SharedCodebook.from_dict(json.loads(
            (args.m11_dir / f"m11_G16_codebook_{bits}bit_K512.json").read_text(encoding="utf-8")))

        started = time.perf_counter()
        train_k = m11_g16_prototype_indices(model, codebook, train_symbols, data["train_references"],
                                            zero, device=device)
        val_k = m11_g16_prototype_indices(model, codebook, val_symbols, data["val_references"],
                                          zero, device=device)
        print(f"    M11-G16 prototype indices for {len(train_k) + len(val_k)} frames in "
             f"{time.perf_counter() - started:.1f}s", flush=True)

        flat = lambda a: a.reshape(-1)
        t_sym, t_k = flat(train_symbols), flat(train_k)
        a_sym, a_k = flat(val_symbols[select_mask]), flat(val_k[select_mask])
        b_sym, b_k = flat(val_symbols[report_mask]), flat(val_k[report_mask])

        # --- baseline arms: deployed M11-G16, and recalibration-only (not context) --
        baselines = {}
        m11g16_q = codebook.probabilities
        baselines["m11_g16_deployed"] = float(-np.sum(np.log2(m11g16_q[b_k, b_sym]))) / b_sym.size

        groups = codebook.size
        parent_counts = cx.fit_parent(t_sym, t_k, groups, alphabet)
        strength_parent, _ = cx.select_strength(lambda s: cx.parent_bits(
            cx.smoothed_parent(parent_counts, m11g16_q, s), a_sym, a_k))
        parent = cx.smoothed_parent(parent_counts, m11g16_q, strength_parent)
        baselines["m11_g16_recalibrated"] = cx.parent_bits(parent, b_sym, b_k) / b_sym.size

        recal_gain = _percent(baselines["m11_g16_deployed"], baselines["m11_g16_recalibrated"])
        print(f"\n    VAL-B ideal bits/symbol:  M11-G16 deployed {baselines['m11_g16_deployed']:.5f}   "
             f"M11-G16 recalibrated {baselines['m11_g16_recalibrated']:.5f}", flush=True)
        print(f"    recalibration alone (NOT context): {recal_gain:+.3f}% vs deployed M11-G16 "
             f"(strength {strength_parent:g})", flush=True)

        # --- spatial context candidates, net of a whole-split-permuted control ------
        rng = np.random.default_rng(args.seed)
        rows = []
        for name in contexts:
            function, _, family, _ = cx.CONTEXTS[name]
            causality = cx.check_causality(function, train_symbols[0], zero_np, alphabet,
                                           probes=48, seed=args.seed)
            if not causality["causal"]:
                print(f"[ERROR] context {name} leaks future symbols at "
                     f"{causality['leaking_positions'][:5]} - STOPPING", file=sys.stderr)
                return 1

            def contexts_of(frames):
                return np.stack([cx.compute_context(name, f, zero_np, alphabet)[0]
                                 for f in frames])
            _, card = cx.compute_context(name, train_symbols[0], zero_np, alphabet)
            t_ctx = contexts_of(train_symbols)
            v_ctx = contexts_of(val_symbols)
            t_perm = cx.permuted_context(t_ctx, rng)
            v_perm = np.empty_like(v_ctx)
            v_perm[select_mask] = cx.permuted_context(v_ctx[select_mask], rng)
            v_perm[report_mask] = cx.permuted_context(v_ctx[report_mask], rng)

            result = {"context": name, "family": family, "cardinality": card,
                     "causality": causality, "joint_contexts": groups * card}
            for label, train_c, val_c in (("real", t_ctx, v_ctx), ("control", t_perm, v_perm)):
                tc = flat(train_c)
                ac, bc = flat(val_c[select_mask]), flat(val_c[report_mask])
                child = cx.fit_child(t_sym, t_k, tc, card, groups, alphabet)
                strength, _ = cx.select_strength(lambda s: cx.child_bits(
                    child, parent, a_sym, a_k, ac, card, s))
                bits_b = cx.child_bits(child, parent, b_sym, b_k, bc, card, strength) / b_sym.size
                selection_bits = cx.child_bits(child, parent, a_sym, a_k, ac, card,
                                               strength) / a_sym.size
                result[label] = {"strength": strength, "val_b_bits": bits_b,
                                 "val_a_bits": selection_bits}
            recal = baselines["m11_g16_recalibrated"]
            result["context_gain_percent"] = _percent(recal, result["real"]["val_b_bits"])
            result["control_gain_percent"] = _percent(recal, result["control"]["val_b_bits"])
            result["net_gain_percent"] = (result["context_gain_percent"]
                                          - max(result["control_gain_percent"], 0.0))
            result["vs_m11_g16_deployed_percent"] = _percent(baselines["m11_g16_deployed"],
                                                              result["real"]["val_b_bits"])
            parent_a = cx.parent_bits(parent, a_sym, a_k) / a_sym.size
            result["selection_net_percent"] = (
                _percent(parent_a, result["real"]["val_a_bits"])
                - max(_percent(parent_a, result["control"]["val_a_bits"]), 0.0))
            rows.append(result)
            print(f"    {name:<18} {family:<8} card {card:>4}   "
                 f"ctx {result['context_gain_percent']:+.3f}%  "
                 f"ctrl {result['control_gain_percent']:+.3f}%  "
                 f"NET {result['net_gain_percent']:+.3f}%  "
                 f"vs M11-G16 {result['vs_m11_g16_deployed_percent']:+.3f}%", flush=True)

        best = max(rows, key=lambda r: r["selection_net_percent"])
        report["rate_points"].append({
            "bits": bits, "alphabet": alphabet, "channels": channels, "height": height,
            "width": width, "train_p_frames": int(len(train_symbols)),
            "val_a_p_frames": int(select_mask.sum()), "val_b_p_frames": int(report_mask.sum()),
            "val_sequence_ids": data["val_sequence_ids"], "data_from_cache": data["from_cache"],
            "baselines_val_b_bits": baselines, "recalibration_gain_percent": recal_gain,
            "recalibration_strength": strength_parent, "zero_symbols": zero_np.tolist(),
            "candidates": rows, "best_by_val_a": best["context"],
        })

    print()
    print("=" * 116)
    print("GATE - best context chosen on VAL-A, judged on VAL-B, net of the random control")
    print("=" * 116)
    names = [r["context"] for r in report["rate_points"][0]["candidates"]]
    mean_net = {n: float(np.mean([next(c for c in rp["candidates"] if c["context"] == n)
                                  ["selection_net_percent"] for rp in report["rate_points"]]))
               for n in names}
    chosen = max(mean_net, key=mean_net.get)
    verdicts = []
    for rp in report["rate_points"]:
        row = next(c for c in rp["candidates"] if c["context"] == chosen)
        control_ok = row["control_gain_percent"] <= CONTROL_TOLERANCE_PERCENT
        net = row["net_gain_percent"]
        verdict = ("meaningful" if net > MEANINGFUL_ABOVE_PERCENT else
                  "marginal" if net >= WEAK_BELOW_PERCENT else "weak")
        verdicts.append((rp["bits"], net, row["control_gain_percent"], control_ok, verdict,
                         row["vs_m11_g16_deployed_percent"]))
        print(f"  {rp['bits']}-bit  {chosen:<18} NET {net:+.3f}%   control "
             f"{row['control_gain_percent']:+.3f}% ({'ok' if control_ok else 'NOT ~0'})   "
             f"vs M11-G16 deployed {row['vs_m11_g16_deployed_percent']:+.3f}%   -> {verdict}")
    all_controls_ok = all(c["control_gain_percent"] <= CONTROL_TOLERANCE_PERCENT
                          for rp in report["rate_points"] for c in rp["candidates"])
    passed = bool(all(v[3] for v in verdicts) and min(v[1] for v in verdicts) >= WEAK_BELOW_PERCENT)
    report.update({"chosen_context": chosen, "chosen_family": cx.CONTEXTS[chosen][2],
                   "mean_selection_net_percent": mean_net,
                   "all_controls_within_tolerance": all_controls_ok, "gate_passed": passed,
                   "verdicts": [{"bits": v[0], "net_percent": v[1], "control_percent": v[2],
                                "control_ok": v[3], "verdict": v[4],
                                "vs_m11_g16_deployed_percent": v[5]} for v in verdicts]})
    print(f"\n  random controls within +{CONTROL_TOLERANCE_PERCENT}% for every candidate: "
         f"{all_controls_ok}")
    strongest = max(v[1] for v in verdicts)
    print(f"  GATE {'PASSED' if passed else 'NOT PASSED'}: chosen context {chosen} "
         f"({cx.CONTEXTS[chosen][2]}), net gain {min(v[1] for v in verdicts):+.3f}% to "
         f"{strongest:+.3f}% over deployed M11-G16 across rate points "
         f"(threshold {WEAK_BELOW_PERCENT}% at every point)")

    path = args.output_dir / "m12_spatial_offline_gate.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
