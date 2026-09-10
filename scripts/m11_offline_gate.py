"""M11 offline gate - do previously decoded residual symbols add information over M10L?

The baseline is the STRONGEST current model, not the weakest. M10L assigns every
position to one of 512 prototype distributions derived from M10K's
z_ref-conditioned prediction, and its tables are what the codec deploys. So the
question is asked conditional on that prototype index k:

    does P(R | k, causal residual context) beat P(R | k) on held-out data?

Conditioning count tables on k lets a discrete, fully interpretable estimator
carry all of z_ref's information (M10L is within 0.04% of M10K), so any gain a
context shows is information z_ref did NOT already provide. The alternative -
measuring contexts against the per-channel marginal, as M10J did - would credit
context with everything z_ref already knows and manufacture an apparent gain.

Three effects are separated rather than blended:

  recalibration   P(R | k) refitted on TRAIN symbols vs M10L's prototypes q_k.
                  Not causal context at all - reported so it cannot be
                  mistaken for one.
  context         P(R | k, ctx) vs P(R | k), both fitted the same way.
  control         the same with ctx PERMUTED within each frame: identical
                  histogram and sparsity, no alignment. Its "gain" is estimator
                  bias, and the context gain is reported net of it.

Tuning (smoothing strength, choice of context) uses VAL-A; every reported number
comes from VAL-B, a disjoint set of validation sequences. TEST is never read.

Run:
  ./.venv/Scripts/python.exe scripts/m11_offline_gate.py
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

from nvc.compression.entropy_model import TOTAL_FREQUENCY
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m11_autoregressive_entropy")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
DEFAULT_M10L_DIR = Path("outputs/m10l_shared_codebook/codebooks")
RATE_POINTS = (5, 4, 3)

# Pre-registered decision guidelines, from the milestone brief, fixed before any
# context was measured.
WEAK_BELOW_PERCENT = 0.5
MEANINGFUL_ABOVE_PERCENT = 1.0
# A random control is "approximately zero" if it gains at most this much.
CONTROL_TOLERANCE_PERCENT = 0.1


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percent(baseline: float, value: float) -> float:
    """Reduction relative to baseline, in percent (positive = better)."""
    return (baseline - value) / baseline * 100.0


@torch.no_grad()
def prototype_indices(ml, mk, mc, learned, codebook, references, *, bits, device,
                      with_m10k_bits_for=None):
    """M10L table index for every position of every frame; optionally M10K's
    deployed integer-table cost on the same frames."""
    indices, m10k_bits = [], []
    with mc.deterministic_kernels():
        for i, reference in enumerate(references):
            tensor = torch.from_numpy(np.asarray(reference, dtype=np.float32))[None].to(device)
            indices.append(ml.frame_table_index(learned, codebook, tensor).astype(np.int64))
            if with_m10k_bits_for is not None:
                entropy_model, table = mk.frame_entropy_model(learned, tensor, bits=bits)
                probabilities = entropy_model.frequencies / float(TOTAL_FREQUENCY)
                symbols = with_m10k_bits_for[i].reshape(-1).astype(np.int64)
                m10k_bits.append(float(-np.log2(probabilities[table, symbols]).sum()))
    return np.stack(indices), np.asarray(m10k_bits)


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M11: offline gate for causal residual context.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m10k-dir", type=Path, default=DEFAULT_M10K_DIR)
    parser.add_argument("--m10l-dir", type=Path, default=DEFAULT_M10L_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--contexts", nargs="+", default=None)
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--train-frames", type=int, default=600)
    parser.add_argument("--val-frames-per-sequence", type=int, default=40)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
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
    cx = _load_script("m11_causal_context")
    md = _load_script("m11_data")
    contexts = args.contexts or list(cx.CONTEXTS)

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    cache_dir = None if args.no_cache else (args.cache_dir or md.DEFAULT_CACHE_DIR)

    print("=" * 116, flush=True)
    print("M11 - CAUSAL RESIDUAL CONTEXT: OFFLINE GATE")
    print("=" * 116)
    print("  baseline: M10L (512 prototypes of M10K's z_ref prediction) - the strongest current model")
    print("  tuning on VAL-A, every reported number on VAL-B (disjoint validation sequences)")
    print(f"  contexts: {contexts}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "M11 offline gate", "checkpoint": str(args.checkpoint),
        "weak_below_percent": WEAK_BELOW_PERCENT,
        "meaningful_above_percent": MEANINGFUL_ABOVE_PERCENT,
        "control_tolerance_percent": CONTROL_TOLERANCE_PERCENT,
        "scan_order": "C-major raster: i = c*H*W + y*W + x",
        "contexts": {name: {"family": cx.CONTEXTS[name][2],
                            "description": cx.CONTEXTS[name][3],
                            "definition_id": cx.context_definition_id(name)}
                     for name in contexts},
        "rate_points": [],
    }

    for bits in args.rate_points:
        alphabet = 2 ** bits
        print(f"\n  ---- {bits}-bit ----", flush=True)
        data = md.load_or_collect(
            model, checkpoint=args.checkpoint, manifest=args.manifest, bits=bits,
            device=device, calibration_frames=args.calibration_frames,
            train_frames=args.train_frames,
            val_frames_per_sequence=args.val_frames_per_sequence, cache_dir=cache_dir,
            log=lambda m: print(m, flush=True))
        calibration = data["calibration"]
        train_symbols = data["train_symbols"].astype(np.int64)
        val_symbols = data["val_symbols"].astype(np.int64)
        select_mask, report_mask = md.split_validation(data)
        channels, height, width = train_symbols.shape[1:]
        zero = cx.zero_symbols(calibration["residual_params"], channels)
        print(f"    {len(train_symbols)} train P-frames | validation: "
              f"{int(select_mask.sum())} VAL-A + {int(report_mask.sum())} VAL-B P-frames "
              f"from {len(data['val_sequence_ids'])} sequences", flush=True)

        learned, _ = mk.load_entropy_model(args.m10k_dir / f"learned_entropy_{bits}bit.pt",
                                           device=device)
        codebook = ml.SharedCodebook.from_dict(json.loads(
            (args.m10l_dir / f"codebook_{bits}bit_K512_code_length.json").read_text(
                encoding="utf-8")))
        started = time.perf_counter()
        train_k, _ = prototype_indices(ml, mk, mc, learned, codebook,
                                       data["train_references"], bits=bits, device=device)
        val_k, m10k_frame_bits = prototype_indices(
            ml, mk, mc, learned, codebook, data["val_references"], bits=bits,
            device=device, with_m10k_bits_for=val_symbols)
        print(f"    M10L prototype indices for {len(train_k) + len(val_k)} frames in "
              f"{time.perf_counter() - started:.1f}s", flush=True)

        flat = lambda a: a.reshape(-1)
        t_sym, t_k = flat(train_symbols), flat(train_k)
        a_sym, a_k = flat(val_symbols[select_mask]), flat(val_k[select_mask])
        b_sym, b_k = flat(val_symbols[report_mask]), flat(val_k[report_mask])
        channel_of = np.repeat(np.arange(channels), height * width)
        t_ch = np.tile(channel_of, len(train_symbols))
        a_ch = np.tile(channel_of, int(select_mask.sum()))
        b_ch = np.tile(channel_of, int(report_mask.sum()))
        per = lambda total, n: total / n

        # --- the existing arms, all on VAL-B ------------------------------------
        baselines = {}
        train_list = [s for s in train_symbols]
        train_refs = [r for r in data["train_references"]]
        b_frames = np.nonzero(report_mask)[0]
        for scheme in ("marginal", "local_activity4"):
            context_model = ce.fit_context_model(scheme, train_refs)
            built = ce.build_conditional_entropy_model(train_list, train_refs, context_model,
                                                       bits=bits)
            probabilities = built["entropy_model"].frequencies / float(TOTAL_FREQUENCY)
            total = 0.0
            for i in b_frames:
                ctx = context_model.contexts(data["val_references"][i])
                table = context_model.table_index(channels, height, width, ctx)
                total += float(-np.log2(probabilities[table, val_symbols[i].reshape(-1)]).sum())
            baselines[scheme] = per(total, b_sym.size)
        m10l_q = codebook.probabilities
        baselines["m10l"] = per(cx.parent_bits(m10l_q, b_sym, b_k), b_sym.size)
        baselines["m10k"] = per(float(m10k_frame_bits[report_mask].sum()), b_sym.size)

        # --- recalibrated M10L: P(R | k) refitted on TRAIN symbols -------------
        groups = codebook.size
        parent_counts = cx.fit_parent(t_sym, t_k, groups, alphabet)
        strength_parent, _ = cx.select_strength(lambda s: cx.parent_bits(
            cx.smoothed_parent(parent_counts, m10l_q, s), a_sym, a_k))
        parent = cx.smoothed_parent(parent_counts, m10l_q, strength_parent)
        baselines["m10l_recalibrated"] = per(cx.parent_bits(parent, b_sym, b_k), b_sym.size)

        # channel-only parent (no z_ref) for the raw-signal diagnostic
        ch_counts = cx.fit_parent(t_sym, t_ch, channels, alphabet)
        uniform = np.full((channels, alphabet), 1.0 / alphabet)
        ch_parent = cx.smoothed_parent(ch_counts, uniform, 1.0)
        channel_only = per(cx.parent_bits(ch_parent, b_sym, b_ch), b_sym.size)

        print(f"\n    VAL-B ideal bits/symbol:  M10H {baselines['marginal']:.5f}   "
              f"M10J {baselines['local_activity4']:.5f}   M10K {baselines['m10k']:.5f}   "
              f"M10L {baselines['m10l']:.5f}   M10L-recal {baselines['m10l_recalibrated']:.5f}",
              flush=True)
        recal_gain = _percent(baselines["m10l"], baselines["m10l_recalibrated"])
        print(f"    recalibration alone (not context): {recal_gain:+.3f}% vs M10L "
              f"(strength {strength_parent:g})")

        # --- the candidates --------------------------------------------------------
        rng = np.random.default_rng(args.seed)
        rows = []
        for name in contexts:
            function, _, family, _ = cx.CONTEXTS[name]
            causality = cx.check_causality(function, train_symbols[0], zero, alphabet,
                                           probes=48, seed=args.seed)
            if not causality["causal"]:
                print(f"[ERROR] context {name} leaks future symbols at "
                      f"{causality['leaking_positions'][:5]} - STOPPING", file=sys.stderr)
                return 1

            def contexts_of(frames):
                return np.stack([cx.compute_context(name, f, zero, alphabet)[0]
                                 for f in frames])
            _, card = cx.compute_context(name, train_symbols[0], zero, alphabet)
            t_ctx = contexts_of(train_symbols)
            v_ctx = contexts_of(val_symbols)
            # Controls are shuffled across a WHOLE split at once - see
            # permuted_context for why a per-frame shuffle is not a valid control.
            # VAL-A and VAL-B are shuffled separately so neither borrows the other.
            t_perm = cx.permuted_context(t_ctx, rng)
            v_perm = np.empty_like(v_ctx)
            v_perm[select_mask] = cx.permuted_context(v_ctx[select_mask], rng)
            v_perm[report_mask] = cx.permuted_context(v_ctx[report_mask], rng)

            result = {"context": name, "family": family, "cardinality": card,
                      "causality": causality,
                      "joint_contexts": groups * card}
            for label, train_c, val_c in (("real", t_ctx, v_ctx), ("control", t_perm, v_perm)):
                tc = flat(train_c)
                ac, bc = flat(val_c[select_mask]), flat(val_c[report_mask])
                child = cx.fit_child(t_sym, t_k, tc, card, groups, alphabet)
                strength, _ = cx.select_strength(lambda s: cx.child_bits(
                    child, parent, a_sym, a_k, ac, card, s))
                bits_b = per(cx.child_bits(child, parent, b_sym, b_k, bc, card, strength),
                             b_sym.size)
                # raw signal without z_ref: channel-conditioned, same estimator
                ch_child = cx.fit_child(t_sym, t_ch, tc, card, channels, alphabet)
                ch_strength, _ = cx.select_strength(lambda s: cx.child_bits(
                    ch_child, ch_parent, a_sym, a_ch, ac, card, s))
                ch_bits = per(cx.child_bits(ch_child, ch_parent, b_sym, b_ch, bc, card,
                                            ch_strength), b_sym.size)
                selection_bits = per(cx.child_bits(child, parent, a_sym, a_k, ac, card,
                                                   strength), a_sym.size)
                result[label] = {"strength": strength, "val_b_bits": bits_b,
                                 "val_a_bits": selection_bits,
                                 "channel_only_val_b_bits": ch_bits}
            recal = baselines["m10l_recalibrated"]
            result["context_gain_percent"] = _percent(recal, result["real"]["val_b_bits"])
            result["control_gain_percent"] = _percent(recal, result["control"]["val_b_bits"])
            # A negative control (the cost of a useless context) is NOT added back:
            # that cost is real and would be paid in deployment too.
            result["net_gain_percent"] = (result["context_gain_percent"]
                                          - max(result["control_gain_percent"], 0.0))
            result["vs_m10l_percent"] = _percent(baselines["m10l"],
                                                 result["real"]["val_b_bits"])
            result["channel_only_gain_percent"] = _percent(
                channel_only, result["real"]["channel_only_val_b_bits"])
            parent_a = per(cx.parent_bits(parent, a_sym, a_k), a_sym.size)
            result["selection_net_percent"] = (
                _percent(parent_a, result["real"]["val_a_bits"])
                - max(_percent(parent_a, result["control"]["val_a_bits"]), 0.0))
            rows.append(result)
            print(f"    {name:<24} {family:<8} card {card:>4}   "
                  f"ctx {result['context_gain_percent']:+.3f}%  "
                  f"ctrl {result['control_gain_percent']:+.3f}%  "
                  f"NET {result['net_gain_percent']:+.3f}%  "
                  f"vs M10L {result['vs_m10l_percent']:+.3f}%   "
                  f"[no-z_ref signal {result['channel_only_gain_percent']:+.2f}%]",
                  flush=True)

        best = max(rows, key=lambda r: r["selection_net_percent"])
        report["rate_points"].append({
            "bits": bits, "alphabet": alphabet, "channels": channels,
            "height": height, "width": width,
            "train_p_frames": int(len(train_symbols)),
            "val_a_p_frames": int(select_mask.sum()), "val_b_p_frames": int(report_mask.sum()),
            "val_sequence_ids": data["val_sequence_ids"], "data_from_cache": data["from_cache"],
            "baselines_val_b_bits": baselines, "channel_only_val_b_bits": channel_only,
            "recalibration_gain_percent": recal_gain, "recalibration_strength": strength_parent,
            "zero_symbols": zero.tolist(), "candidates": rows,
            "best_by_val_a": best["context"],
        })

    # --- the decision ----------------------------------------------------------------
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
                         row["vs_m10l_percent"]))
        print(f"  {rp['bits']}-bit  {chosen:<24} NET {net:+.3f}%   control "
              f"{row['control_gain_percent']:+.3f}% ({'ok' if control_ok else 'NOT ~0'})   "
              f"vs M10L {row['vs_m10l_percent']:+.3f}%   -> {verdict}")
    all_controls_ok = all(c["control_gain_percent"] <= CONTROL_TOLERANCE_PERCENT
                          for rp in report["rate_points"] for c in rp["candidates"])
    strongest = max(v[1] for v in verdicts)
    passed = bool(all(v[3] for v in verdicts) and min(v[1] for v in verdicts)
                  >= WEAK_BELOW_PERCENT)
    report.update({"chosen_context": chosen, "chosen_family": cx.CONTEXTS[chosen][2],
                   "mean_selection_net_percent": mean_net,
                   "all_controls_within_tolerance": all_controls_ok,
                   "gate_passed": passed,
                   "verdicts": [{"bits": v[0], "net_percent": v[1], "control_percent": v[2],
                                 "control_ok": v[3], "verdict": v[4],
                                 "vs_m10l_percent": v[5]} for v in verdicts]})
    print(f"\n  random controls within +{CONTROL_TOLERANCE_PERCENT}% for every candidate: "
          f"{all_controls_ok}")
    print(f"  GATE {'PASSED' if passed else 'NOT PASSED'}: chosen context {chosen} "
          f"({cx.CONTEXTS[chosen][2]}), net gain {min(v[1] for v in verdicts):+.3f}% to "
          f"{strongest:+.3f}% over M10L across rate points "
          f"(threshold {WEAK_BELOW_PERCENT}% at every point)")

    path = args.output_dir / "offline_gate.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
