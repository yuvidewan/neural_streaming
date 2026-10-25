"""M23 Phase B1 - is the hyperprior's information ADDITIVE to the full deployed
stack, or is it already inside M11-G16's causal context?

Phase A measured +1.5734 / +2.4588 / +2.1599% at 5/4/3-bit against `zref_only`
(M10K's family). The shipped P-residual path is richer than that: M11-G16's
channel-autoregressive context plus the K=512 codebook plus M13's recalibrated
frequencies, worth a further +5.3-6.0% of coded bytes on this transform. A
transmitted scale map and a causal context model could easily be describing the
same local activity, in which case Phase A's gain is not additive and there is
nothing to carry forward.

This is structurally the question M12 asked about spatial context - "does it
still add anything once the parent model IS M11-G16?" - so the methodology is
M12's, with the context swapped. Baseline and estimator are M12's too, which is
what makes the two milestones' numbers comparable.

THE ONE WAY THIS DIFFERS FROM EVERY PRIOR CONTEXT GATE
------------------------------------------------------
M10J, M11 and M12 all tested CAUSAL contexts: functions of already-decoded
symbols, free to the decoder, which is why those gates check causality and charge
nothing. A hyperprior is the opposite - it is TRANSMITTED, so it is available
before the first symbol is decoded and needs no causality check, but it costs
bits and the gate is worthless unless they are charged.

So this gate deliberately does NOT call `check_causality`, and it DOES add the
context map's own coded cost to every candidate arm, including the control (which
transmits exactly as much). `side_bits_per_symbol` is reported separately at
every rate point so the charge can be seen rather than trusted.

THE ARMS
--------
  scale_map   the hyperprior stripped to its mechanism: a per-position magnitude
              summary of the residual, coarsened, binned on TRAIN quantiles,
              transmitted and charged. No network, nothing learned, fully
              interpretable - and a clean reference for what the mechanism is
              worth before any analysis transform is credited for it.
  learned     the same pipeline fed by Phase A's own trained h_a, so the arm that
              scored +2.46% there is the arm tested here. Retrained in-process
              from the same config and seed rather than loaded, so this gate
              depends on no saved artefact.

Both transmit exactly the context they use - the binned, coarsened map - not the
full hyper-latent, so the charge matches what the arm consumes.

BASELINES (VAL-B ideal bits/symbol, M12's two)
----------------------------------------------
  m11_g16_deployed       the shipped codebook probabilities q[k, s]
  m11_g16_recalibrated   P(s | k) refit on TRAIN, i.e. M13's own mechanism.
                         **This is the full-stack baseline**, and the one every
                         gain below is measured against.

Gains are reported net of a whole-split permuted control under M11's rule,
`gain - max(control, 0)`: a negative control is the cost of a useless signal and
is not added back. Thresholds are the project's pre-registered ones.

HARD SCOPE
----------
Frozen: the transform, the quantizer grids, M10K, the G16 network and its K=512
codebook, lambda, GOP, motion, the `.nvct` format, the splits. Only the
transmitted context moves. Nothing under `src/nvc/` is touched and no earlier
script is rewritten - M12's prototype-index helper and M11's context fitting are
imported read-only.

TRAIN fits, VAL-A selects the smoothing strength, VAL-B reports. TEST is never
opened.

Run:
  ./.venv/Scripts/python.exe scripts/m23_phaseb_gate.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

ARMS = ("scale_map", "learned")
DEFAULT_CONTEXT_BINS = 8
DEFAULT_CONTEXT_STRIDE = 2


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percent(baseline: float, value: float) -> float:
    return 100.0 * (baseline - value) / baseline


# --------------------------------------------------------------------------------------------
# Turning a per-position signal into a transmitted context, and charging it
# --------------------------------------------------------------------------------------------


def coarsen(values: torch.Tensor, *, stride: int) -> torch.Tensor:
    """Average-pool a [N, 1, H, W] map by `stride`.

    Coarsening is what makes a transmitted map affordable: at stride 2 on a 16x16
    latent it is 64 values per frame against 16,384 symbols, so the charge is
    fractions of a percent rather than the 1/C the full-resolution map would
    cost.
    """
    if stride == 1:
        return values
    return torch.nn.functional.avg_pool2d(values, kernel_size=stride, stride=stride,
                                          ceil_mode=True)


def train_bin_edges(values: torch.Tensor, *, bins: int) -> np.ndarray:
    """Quantile edges fitted on TRAIN only.

    Quantiles rather than a uniform grid so every bin is populated, and TRAIN
    only because edges fitted on the split they are scored against would make the
    context look more informative than it is.

    Note what quantile binning costs: it makes the map's marginal UNIFORM, the
    most expensive distribution it could have - an 8-bin map then costs a full 3
    bits per value with nothing left for the entropy coder to exploit. A
    uniform-grid binning would transmit more cheaply and discriminate less. So
    this is the pessimistic choice on the charge, consistent with the gate's other
    conservative choices, and `side_bits_per_symbol_val_b` is reported at every
    rate point so the trade-off is visible rather than buried.
    """
    if bins & (bins - 1):
        raise ValueError(f"bins must be a power of two so the map has an integer "
                         f"bit width, got {bins}")
    flat = values.detach().reshape(-1).cpu().numpy()
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    return np.quantile(flat, quantiles)


def to_bins(values: torch.Tensor, edges: np.ndarray) -> np.ndarray:
    """-> [N, h, w] integer bin indices in [0, len(edges)]."""
    flat = values.detach().reshape(values.shape[0], -1).cpu().numpy()
    binned = np.digitize(flat, edges)
    return binned.reshape(values.shape[0], *values.shape[-2:]).astype(np.int64)


def broadcast_to_symbol_positions(binned: np.ndarray, *, channels: int, height: int,
                                  width: int) -> np.ndarray:
    """[N, h, w] coarse context -> [N, C*H*W] in the coder's C-major flat order.

    The map is spatial, so every channel at a position gets the same context -
    exactly how Phase A's `hyper_synthesis` broadcast its features. Nearest
    upsampling, because the decoder reconstructs it the same way from the coarse
    values it was sent.
    """
    tensor = torch.from_numpy(binned).unsqueeze(1).float()
    upsampled = torch.nn.functional.interpolate(tensor, size=(height, width),
                                                mode="nearest")
    per_position = upsampled.squeeze(1).to(torch.int64).numpy()        # [N, H, W]
    return np.repeat(per_position.reshape(binned.shape[0], 1, height * width),
                     channels, axis=1).reshape(binned.shape[0], channels * height * width)


def context_map_bits(train_binned: np.ndarray, binned: np.ndarray, *, bins: int) -> float:
    """Total REAL bits to transmit the coarse context maps of `binned`.

    One table, fitted on TRAIN by the same `EmpiricalEntropyModel.from_symbols`
    and the same Laplace smoothing every other table in this codec uses. This is
    what the range coder would pay, not a density estimate.
    """
    bit_width = int(np.log2(bins))
    table = EmpiricalEntropyModel.from_symbols(
        train_binned.reshape(train_binned.shape[0], 1, -1), bits=bit_width, num_tables=1)
    probabilities = table.probabilities()[0]
    return float(-np.log2(probabilities[binned.reshape(-1)]).sum())


# --------------------------------------------------------------------------------------------
# The two arms' signals
# --------------------------------------------------------------------------------------------


def scale_map_signal(residual: torch.Tensor) -> torch.Tensor:
    """The hyperprior's mechanism with nothing learned: mean |residual| per
    position, across channels.

    This is what a scale map is FOR - telling the decoder how big the residual is
    about to be at this position - so measuring it directly separates the
    mechanism from the analysis transform that might later approximate it.
    """
    return residual.abs().mean(dim=1, keepdim=True)


def learned_signal(m23, model, signals: torch.Tensor, *, device, hyper_bits: int,
                   batch_size: int = 16) -> torch.Tensor:
    """Phase A's trained h_a, reduced to one value per coarse position.

    The hyper-latent is quantized exactly as Phase A charged it, then averaged
    over its channels: the arm transmits a scalar map, so it is charged for a
    scalar map. Averaging after quantization, not before, because the decoder
    only ever has the quantized values.
    """
    out = []
    model = model.to(device).eval()
    with torch.no_grad():
        for start in range(0, signals.shape[0], batch_size):
            batch = signals[start:start + batch_size].to(device)
            raw = model.analysis(batch)
            _, dequantized, _ = m23.quantize_hyper_latent(raw, bits=hyper_bits)
            out.append(dequantized.mean(dim=1, keepdim=True).cpu())
    return torch.cat(out, dim=0)


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    m23 = _load_script("m23_hyperprior")
    parser = argparse.ArgumentParser(
        description="M23 Phase B1: is the hyperprior additive to the full deployed stack?",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=m23.DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=m23.DEFAULT_M11_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m23_hyperprior"))
    parser.add_argument("--output-name", type=str, default="m23_phaseb_gate.json")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(m23.RATE_POINTS))
    parser.add_argument("--context-bins", type=int, default=DEFAULT_CONTEXT_BINS)
    parser.add_argument("--context-stride", type=int, default=DEFAULT_CONTEXT_STRIDE)
    parser.add_argument("--hyper-bits", type=int, default=m23.DEFAULT_HYPER_BITS)
    parser.add_argument("--hyper-channels", type=int, default=m23.DEFAULT_HYPER_CHANNELS)
    parser.add_argument("--hyper-hidden", type=int, default=m23.DEFAULT_HYPER_HIDDEN)
    parser.add_argument("--hidden", type=int, default=m23.DEFAULT_HIDDEN)
    parser.add_argument("--epochs", type=int, default=20,
                        help="h_a training for the learned arm; Phase A's value")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    m23 = _load_script("m23_hyperprior")
    gate_a = _load_script("m23_offline_gate")
    md = _load_script("m11_data")
    cx = _load_script("m11_causal_context")
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    m12 = _load_script("m12_spatial_offline_gate")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / args.output_name
    configuration = {
        "checkpoint": str(args.checkpoint), "m11_dir": str(args.m11_dir),
        "rate_points": list(args.rate_points), "arms": list(args.arms),
        "context_bins": args.context_bins, "context_stride": args.context_stride,
        "hyper_bits": args.hyper_bits, "hyper_channels": args.hyper_channels,
        "hyper_hidden": args.hyper_hidden, "hidden": args.hidden, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "calibration_frames": args.calibration_frames, "seed": args.seed,
    }
    report: dict[str, Any] = {
        "phase": "M23 Phase B1 - is the hyperprior additive to the full deployed stack?",
        "question": ("Phase A measured +1.57/+2.46/+2.16% against zref_only (M10K's "
                     "family). Does any of it survive once the parent is M11-G16 + K=512 "
                     "+ M13, which adds a further +5.3-6.0%?"),
        "methodology": ("M12's, with the context swapped: same prototype-index helper, "
                        "same hierarchical P(s|k,ctx) fitting, same two baselines, so the "
                        "numbers are comparable to M12's spatial-context result"),
        "differs_from_prior_context_gates": (
            "every earlier context gate (M10J/M11/M12) tested CAUSAL contexts - free to "
            "the decoder, hence a causality check and no charge. A hyperprior is "
            "TRANSMITTED: no causality check is needed, and the gate is worthless unless "
            "its bits are charged. Both are reflected here - no check_causality call, and "
            "side_bits_per_symbol added to every arm INCLUDING the control."),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "pre_registered": {
            "arms": list(ARMS),
            "baseline": "m11_g16_recalibrated (the full deployed stack: G16 + K=512 + M13)",
            "weak_below_percent": m23.WEAK_BELOW_PERCENT,
            "meaningful_above_percent": m23.MEANINGFUL_ABOVE_PERCENT,
            "control_tolerance_percent": gate_a.CONTROL_TOLERANCE_PERCENT,
            "net_of_control_rule": "gain - max(control, 0) (M11's rule)",
            "split_discipline": ("TRAIN fits tables and h_a, VAL-A selects the smoothing "
                                 "strength, VAL-B reports, TEST never opened"),
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": str(device), "platform": platform.platform()},
        "rate_points": [],
    }
    if report_path.is_file() and args.resume:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("configuration") == configuration:
            report["rate_points"] = previous.get("rate_points", [])
            print(f"[resume] {len(report['rate_points'])} rate points already measured",
                  flush=True)

    def persist() -> None:
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    done = {entry["bits"] for entry in report["rate_points"]}

    from nvc.training.checkpoint import load_model_from_checkpoint

    print("=" * 118)
    print("M23 PHASE B1 - IS THE HYPERPRIOR ADDITIVE TO THE FULL STACK? (VAL-B; TEST never "
          "opened)")
    print(f"  parent: M11-G16 + K=512 + M13   arms: {args.arms}   "
          f"context: {args.context_bins} bins at stride {args.context_stride}")
    print("=" * 118, flush=True)

    for bits in args.rate_points:
        if bits in done:
            print(f"  [resume] {bits}-bit already measured", flush=True)
            continue
        alphabet = 2 ** bits
        print(f"\n  ---- {bits}-bit ----", flush=True)
        base_model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
        base_model.eval()
        data = md.load_or_collect(
            base_model, checkpoint=args.checkpoint, manifest=args.manifest, bits=bits,
            device=device, calibration_frames=args.calibration_frames, cache_dir=cache_dir,
            log=lambda m: print(m, flush=True))
        train_symbols = data["train_symbols"].astype(np.int64)
        val_symbols = data["val_symbols"].astype(np.int64)
        select_mask, report_mask = md.split_validation(data)
        channels, height, width = train_symbols.shape[1:]
        residual_params = data["calibration"]["residual_params"]
        zero_np = cx.zero_symbols(residual_params, channels)
        zero = torch.from_numpy(zero_np)
        print(f"    {len(train_symbols)} TRAIN P-frames | VAL-A {int(select_mask.sum())} + "
              f"VAL-B {int(report_mask.sum())}", flush=True)

        model, _ = ma.load_model(args.m11_dir / f"m11_G16_entropy_{bits}bit.pt", device=device)
        codebook = ml.SharedCodebook.from_dict(json.loads(
            (args.m11_dir / f"m11_G16_codebook_{bits}bit_K512.json").read_text(
                encoding="utf-8")))

        started = time.perf_counter()
        train_k = m12.m11_g16_prototype_indices(model, codebook, train_symbols,
                                                data["train_references"], zero, device=device)
        val_k = m12.m11_g16_prototype_indices(model, codebook, val_symbols,
                                              data["val_references"], zero, device=device)
        print(f"    prototype indices in {time.perf_counter() - started:.1f}s", flush=True)

        flat = lambda a: a.reshape(-1)
        t_sym, t_k = flat(train_symbols), flat(train_k)
        a_sym, a_k = flat(val_symbols[select_mask]), flat(val_k[select_mask])
        b_sym, b_k = flat(val_symbols[report_mask]), flat(val_k[report_mask])

        # ---- the two baselines, M12's ------------------------------------------------
        baselines: dict[str, float] = {}
        q = codebook.probabilities
        baselines["m11_g16_deployed"] = float(-np.sum(np.log2(q[b_k, b_sym]))) / b_sym.size
        groups = codebook.size
        parent_counts = cx.fit_parent(t_sym, t_k, groups, alphabet)
        strength_parent, _ = cx.select_strength(lambda s: cx.parent_bits(
            cx.smoothed_parent(parent_counts, q, s), a_sym, a_k))
        parent = cx.smoothed_parent(parent_counts, q, strength_parent)
        baselines["m11_g16_recalibrated"] = cx.parent_bits(parent, b_sym, b_k) / b_sym.size
        parent_a_bits = cx.parent_bits(parent, a_sym, a_k) / a_sym.size
        print(f"    VAL-B baseline bits/symbol:  deployed "
              f"{baselines['m11_g16_deployed']:.5f}   FULL STACK (recalibrated) "
              f"{baselines['m11_g16_recalibrated']:.5f}", flush=True)

        # ---- the signals each arm transmits ------------------------------------------
        train_residual = m23.dequantized_residual(
            train_symbols, residual_params, shape=(channels, height, width))
        val_residual = m23.dequantized_residual(
            val_symbols, residual_params, shape=(channels, height, width))

        signals: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if "scale_map" in args.arms:
            signals["scale_map"] = (scale_map_signal(train_residual),
                                    scale_map_signal(val_residual))
        learned_history = None
        if "learned" in args.arms:
            print("    training Phase A's h_a for the learned arm ...", flush=True)
            config = {"latent_channels": channels, "alphabet": alphabet,
                      "hidden": args.hidden, "hyper_channels": args.hyper_channels,
                      "hyper_hidden": args.hyper_hidden}
            hyper_model = m23.build_model("hyper_plus_zref", config)
            train_refs = torch.from_numpy(
                np.asarray(data["train_references"], dtype=np.float32))
            val_refs = torch.from_numpy(np.asarray(data["val_references"], dtype=np.float32))
            learned_history = m23.train_arm(
                hyper_model,
                (torch.from_numpy(train_symbols).long(), train_refs, train_residual),
                (torch.from_numpy(val_symbols[select_mask]).long(),
                 val_refs[select_mask], val_residual[select_mask]),
                epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.learning_rate, device=device,
                hyper_bits=args.hyper_bits, seed=args.seed,
                log=lambda m: print(m, flush=True))
            signals["learned"] = (
                learned_signal(m23, hyper_model, train_residual, device=device,
                               hyper_bits=args.hyper_bits),
                learned_signal(m23, hyper_model, val_residual, device=device,
                               hyper_bits=args.hyper_bits))

        # ---- each arm: bin, charge, fit, control ------------------------------------
        rng = np.random.default_rng(args.seed)
        rows = []
        for arm in args.arms:
            train_signal, val_signal = signals[arm]
            train_coarse = coarsen(train_signal, stride=args.context_stride)
            val_coarse = coarsen(val_signal, stride=args.context_stride)
            edges = train_bin_edges(train_coarse, bins=args.context_bins)
            train_binned = to_bins(train_coarse, edges)
            val_binned = to_bins(val_coarse, edges)

            # What the arm transmits is the coarse binned map, so that is what it
            # is charged for - not the full hyper-latent it may have come from.
            side_b = context_map_bits(train_binned, val_binned[report_mask],
                                      bins=args.context_bins)
            side_a = context_map_bits(train_binned, val_binned[select_mask],
                                      bins=args.context_bins)
            side_b_per_symbol = side_b / b_sym.size
            side_a_per_symbol = side_a / a_sym.size

            t_ctx = broadcast_to_symbol_positions(train_binned, channels=channels,
                                                  height=height, width=width)
            v_ctx = broadcast_to_symbol_positions(val_binned, channels=channels,
                                                  height=height, width=width)
            t_perm = cx.permuted_context(t_ctx, rng)
            v_perm = np.empty_like(v_ctx)
            v_perm[select_mask] = cx.permuted_context(v_ctx[select_mask], rng)
            v_perm[report_mask] = cx.permuted_context(v_ctx[report_mask], rng)

            card = args.context_bins
            result: dict[str, Any] = {
                "arm": arm, "cardinality": card, "joint_contexts": groups * card,
                "coarse_values_per_frame": int(train_binned[0].size),
                "side_bits_per_symbol_val_b": side_b_per_symbol,
                "side_share_of_net": None,
                "transmitted_not_causal": True,
            }
            for label, train_c, val_c in (("real", t_ctx, v_ctx),
                                          ("control", t_perm, v_perm)):
                tc = flat(train_c)
                ac, bc = flat(val_c[select_mask]), flat(val_c[report_mask])
                child = cx.fit_child(t_sym, t_k, tc, card, groups, alphabet)
                strength, _ = cx.select_strength(lambda s: cx.child_bits(
                    child, parent, a_sym, a_k, ac, card, s) / a_sym.size
                    + side_a_per_symbol)
                bits_b = cx.child_bits(child, parent, b_sym, b_k, bc, card,
                                       strength) / b_sym.size
                bits_a = cx.child_bits(child, parent, a_sym, a_k, ac, card,
                                       strength) / a_sym.size
                result[label] = {
                    "strength": strength,
                    "symbol_bits_val_b": bits_b,
                    "net_bits_val_b": bits_b + side_b_per_symbol,
                    "net_bits_val_a": bits_a + side_a_per_symbol,
                }
            baseline = baselines["m11_g16_recalibrated"]
            result["side_share_of_net"] = side_b_per_symbol / result["real"]["net_bits_val_b"]
            result["gain_percent"] = _percent(baseline, result["real"]["net_bits_val_b"])
            result["control_gain_percent"] = _percent(baseline,
                                                      result["control"]["net_bits_val_b"])
            result["net_gain_percent"] = gate_a.net_of_control(
                result["gain_percent"], result["control_gain_percent"])
            result["control_interpretation"] = gate_a.control_interpretation(
                result["control_gain_percent"])
            result["verdict"] = m23.verdict(result["net_gain_percent"])
            result["selection_net_percent"] = gate_a.net_of_control(
                _percent(parent_a_bits, result["real"]["net_bits_val_a"]),
                _percent(parent_a_bits, result["control"]["net_bits_val_a"]))
            # The symbol-only gain, with the charge removed, so a gain that exists
            # but cannot pay for itself is visible as exactly that.
            result["symbol_only_gain_percent"] = _percent(
                baseline, result["real"]["symbol_bits_val_b"])
            rows.append(result)
            print(f"    {arm:<10} symbols {result['real']['symbol_bits_val_b']:.5f} "
                  f"+ side {side_b_per_symbol:.5f} = {result['real']['net_bits_val_b']:.5f}"
                  f"   gain {result['gain_percent']:+.4f}%  ctrl "
                  f"{result['control_gain_percent']:+.4f}%  NET "
                  f"{result['net_gain_percent']:+.4f}%  {result['verdict']}", flush=True)
            print(f"               (before the charge: "
                  f"{result['symbol_only_gain_percent']:+.4f}%)", flush=True)

        report["rate_points"].append({
            "bits": bits, "alphabet": alphabet, "channels": channels, "height": height,
            "width": width, "groups": groups,
            "train_p_frames": int(len(train_symbols)),
            "val_a_p_frames": int(select_mask.sum()),
            "val_b_p_frames": int(report_mask.sum()),
            "baselines_val_b_bits": baselines,
            "recalibration_strength": strength_parent,
            "learned_h_a_history": learned_history,
            "candidates": rows,
        })
        persist()

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    persist()

    print("\n" + "=" * 118)
    print("PHASE B1 GATE - against the FULL deployed stack, net of control, charge included")
    print("=" * 118)
    print(f"{'bits':>5} {'arm':>10} {'full stack':>11} {'net b/s':>10} {'side%':>7} "
          f"{'gain':>10} {'control':>10} {'NET':>10} {'verdict':>11}")
    for entry in sorted(report["rate_points"], key=lambda e: -e["bits"]):
        base = entry["baselines_val_b_bits"]["m11_g16_recalibrated"]
        for row in entry["candidates"]:
            print(f"{entry['bits']:>5} {row['arm']:>10} {base:>11.5f} "
                  f"{row['real']['net_bits_val_b']:>10.5f} "
                  f"{100 * row['side_share_of_net']:>6.2f}% "
                  f"{row['gain_percent']:>+9.4f}% {row['control_gain_percent']:>+9.4f}% "
                  f"{row['net_gain_percent']:>+9.4f}% {row['verdict']:>11}")
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
