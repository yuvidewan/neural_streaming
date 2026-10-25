"""M23 Phase A - the hyperprior offline gate.

Does a TRANSMITTED description of the current content buy more bits than it
costs, on top of what the decoder already gets free? Measured on held-out data,
at frozen transform weights, before anything is built.

See `scripts/m23_hyperprior.py` for why this is the gap worth gating and what the
measurement can and cannot say. In short: nothing in this codec can make a bit's
cost depend on what is at that position in this frame, which is the mechanism
behind the 192-channel rejection and behind reverse water-filling recovering
0.00%.

THE TWO CHANNELS OF THE STREAM ARE GATED SEPARATELY
---------------------------------------------------
They have different baselines and different amounts of headroom, and averaging
them would hide both:

  intra      11.9-12.2% of the stream. Baseline: the DEPLOYED static
             per-channel histogram - the one part of the codec with no content
             adaptation of any kind. There is no `z_ref` for an intra frame, so
             the only arm is `hyper_only`.
  residual   79.6-84.2% of the stream. Baseline: `zref_only`, i.e. M10K's own
             family trained here under identical conditions, which is the
             information the decoder gets FREE. Arms: `hyper_only` (can
             transmitted side info stand alone?) and `hyper_plus_zref` (the
             deployable question).

SPLIT DISCIPLINE
----------------
TRAIN fits every model and every coding table. VAL-A selects the epoch. VAL-B
reports, and is read once per arm. TEST is never opened by this script. The
hyper-latent's coding table is fitted on TRAIN only - a table fitted on the
split it is scored against would make the side channel look free.

EVERY HYPER ARM IS CHARGED FOR ITS SIDE CHANNEL, and reported net of a
whole-split permutation control. Thresholds are the project's pre-registered
ones: <0.5% net = weak, 0.5-1.0% = marginal, >1.0% = meaningful.

Run:
  ./.venv/Scripts/python.exe scripts/m23_offline_gate.py
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

from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

CHANNEL_ARMS = {
    "intra": ("hyper_only",),
    "residual": ("hyper_only", "hyper_plus_zref"),
}
# The arm each channel's gain is measured against, and where it comes from.
CHANNEL_BASELINE = {
    "intra": "deployed_static_table",
    "residual": "zref_only",
}
# The arm whose control decides whether a gain is content adaptation at all.
CHANNEL_CONTROL_ARM = {"intra": "hyper_only", "residual": "hyper_plus_zref"}

# M11's rule, and its comment is the reason: "A negative control (the cost of a
# useless context) is NOT added back". A control that LOSES is a harmed model,
# not a null one - a mismatched hyper-latent is actively misleading rather than
# merely uninformative - so it confirms the signal is content-specific and must
# not be subtracted, or an arm would be credited for its control being bad.
# Only a POSITIVE control is spurious gain to remove.
# `CONTROL_TOLERANCE_PERCENT` is M11's too: a control is "approximately zero" if
# it gains at most this much.
CONTROL_TOLERANCE_PERCENT = 0.1


def net_of_control(gain_percent: float, control_percent: float) -> float:
    return gain_percent - max(control_percent, 0.0)


def control_interpretation(control_percent: float) -> str:
    if control_percent > CONTROL_TOLERANCE_PERCENT:
        return ("control gains: that much of the arm's gain is capacity or a better "
                "marginal, not content adaptation, and is subtracted")
    if control_percent < -CONTROL_TOLERANCE_PERCENT:
        return ("control LOSES: a mismatched hyper-latent actively misleads the "
                "predictor, which confirms the arm's gain is content-specific. Not "
                "added back - the arm is not credited for its control being bad")
    return "control is approximately zero: the arm's gain stands as measured"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    m23 = _load_script("m23_hyperprior")
    parser = argparse.ArgumentParser(
        description="M23 Phase A: does a transmitted hyperprior beat free conditioning?",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=m23.DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=m23.DEFAULT_M11_DIR,
                        help="only read for the deployed stack's VAL-B figure, as context")
    parser.add_argument("--output-dir", type=Path, default=m23.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default="m23_offline_gate.json")
    parser.add_argument("--channels", nargs="+", choices=list(CHANNEL_ARMS),
                        default=list(CHANNEL_ARMS))
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(m23.RATE_POINTS))
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--intra-train-frames", type=int, default=400)
    parser.add_argument("--intra-val-frames-per-sequence", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hyper-bits", type=int, default=m23.DEFAULT_HYPER_BITS)
    parser.add_argument("--hyper-channels", type=int, default=m23.DEFAULT_HYPER_CHANNELS)
    parser.add_argument("--hyper-hidden", type=int, default=m23.DEFAULT_HYPER_HIDDEN)
    parser.add_argument("--hidden", type=int, default=m23.DEFAULT_HIDDEN)
    parser.add_argument("--gop", type=int, default=m23.DEFAULT_GOP)
    parser.add_argument("--block-size", type=int, default=m23.DEFAULT_BLOCK_SIZE)
    parser.add_argument("--search-range", type=int, default=m23.DEFAULT_SEARCH_RANGE)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


# --------------------------------------------------------------------------------------------
# Data, per channel of the stream
# --------------------------------------------------------------------------------------------


def intra_data(m23, model, args, calibration, device, log) -> dict[str, Any]:
    """TRAIN / VAL-A / VAL-B latents for the intra path.

    VAL is split by SEQUENCE index, the same even/odd split M13-M22 used
    (`m21_refinement.val_b_sequences` takes `sequences[1::2]`), so VAL-A and
    VAL-B never share a sequence and the selection cannot leak into the report.
    """
    train_sequences = discover_sequences(args.manifest, split="train")
    val_sequences = discover_sequences(
        args.manifest, split="val",
        max_frames_per_sequence=args.intra_val_frames_per_sequence)
    log(f"      intra: {len(train_sequences)} TRAIN sequences, "
        f"{len(val_sequences)} VAL sequences")

    train_symbols, train_signals = m23.intra_signal_and_symbols(
        model, train_sequences, intra_params=calibration["intra_params"],
        max_frames=args.intra_train_frames, device=device)

    halves = {}
    for name, subset in (("val_a", val_sequences[0::2]), ("val_b", val_sequences[1::2])):
        symbols, signals = m23.intra_signal_and_symbols(
            model, subset, intra_params=calibration["intra_params"],
            max_frames=10 ** 9, device=device)
        halves[name] = {"symbols": symbols, "signals": signals,
                        "sequences": [s.sequence_id for s in subset]}
    log(f"      intra: TRAIN {train_symbols.shape[0]} frames, "
        f"VAL-A {halves['val_a']['symbols'].shape[0]}, "
        f"VAL-B {halves['val_b']['symbols'].shape[0]}")
    return {"train": {"symbols": train_symbols, "signals": train_signals},
            **halves}


def residual_data(m23, md, model, args, bits, device, log) -> dict[str, Any]:
    """TRAIN / VAL-A / VAL-B residual symbols, z_ref references, and the
    dequantized residual h_a sees - all from M11's own cached collection, so this
    gate is measured on exactly the symbols M10K/M10L/M11 were fitted on."""
    # `calibration_frames` is passed through rather than left to load_or_collect's
    # own default: it is part of that cache's key, and a gate whose calibration
    # and whose symbols were collected under different frame counts would be
    # measuring two different codecs.
    data = md.load_or_collect(
        model, checkpoint=args.checkpoint, manifest=args.manifest, bits=bits, device=device,
        calibration_frames=args.calibration_frames,
        cache_dir=args.cache_dir or md.DEFAULT_CACHE_DIR, log=lambda m: log("      " + m))
    select_mask, _ = md.split_validation(data)
    residual_params = data["calibration"]["residual_params"]

    train_symbols = data["train_symbols"].astype(np.int64)
    val_symbols = data["val_symbols"].astype(np.int64)
    train_references = torch.from_numpy(
        np.asarray(data["train_references"], dtype=np.float32))
    val_references = torch.from_numpy(np.asarray(data["val_references"], dtype=np.float32))
    shape = tuple(train_references.shape[1:])

    train_signals = m23.dequantized_residual(train_symbols, residual_params, shape=shape)
    val_signals = m23.dequantized_residual(val_symbols, residual_params, shape=shape)

    val_b_mask = ~select_mask
    log(f"      residual: TRAIN {train_symbols.shape[0]} P-frames, "
        f"VAL-A {int(select_mask.sum())}, VAL-B {int(val_b_mask.sum())}")
    return {
        "train": {"symbols": train_symbols, "references": train_references,
                  "signals": train_signals},
        "val_a": {"symbols": val_symbols[select_mask],
                  "references": val_references[select_mask],
                  "signals": val_signals[select_mask]},
        "val_b": {"symbols": val_symbols[val_b_mask],
                  "references": val_references[val_b_mask],
                  "signals": val_signals[val_b_mask]},
        "calibration": data["calibration"],
    }


# --------------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------------


def static_table_bits(m14, entropy_model, symbols: np.ndarray) -> float:
    """The DEPLOYED intra coder's held-out bits/symbol: -log2 of the per-channel
    table probability of each true symbol, by M14's own estimator so the number
    is comparable to every intra figure the project has reported."""
    channels = symbols.shape[1]
    table_index = np.repeat(np.arange(channels)[None, :, None],
                            symbols.shape[0], axis=0)
    table_index = np.broadcast_to(table_index, symbols.shape)
    return m14.held_out_bits_per_symbol(entropy_model, symbols, table_index)


def deployed_stack_context(m11_dir: Path, bits: int) -> dict[str, Any] | None:
    """The refitted stack's own VAL-B figure, for context only.

    Reported beside the arms and never subtracted from them: the deployed stack
    is M10K PLUS M11-G16's causal context PLUS M13, a different and richer
    estimator than this gate's in-family baseline. Treating the two as
    interchangeable would overstate any gain found here.
    """
    path = Path(m11_dir) / "m11_training.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    for entry in report.get("rate_points", report if isinstance(report, list) else []):
        if entry.get("bits") == bits:
            return {"source": str(path), "entry": {k: v for k, v in entry.items()
                                                   if not isinstance(v, (dict, list))}}
    return {"source": str(path), "entry": None}


# --------------------------------------------------------------------------------------------
# One arm
# --------------------------------------------------------------------------------------------


def run_arm(m23, *, arm: str, channel: str, data: dict[str, Any], args, bits: int,
            device, control: bool, log) -> dict[str, Any]:
    """Train one arm on TRAIN, select on VAL-A, report on VAL-B.

    `control=True` permutes the frames h_a sees, across the WHOLE split, in TRAIN
    and in VAL alike: the control has to be a model that was fitted under the
    same conditions it is scored under, or it measures the permutation's effect
    on an unfitted model instead of the signal's real contribution.
    """
    started = time.perf_counter()
    channels = int(data["train"]["symbols"].shape[1])
    config = {"latent_channels": channels, "alphabet": 2 ** bits, "hidden": args.hidden,
              "hyper_channels": args.hyper_channels, "hyper_hidden": args.hyper_hidden}
    model = m23.build_model(arm, config)

    def batch_for(split: str, seed_offset: int):
        rows = data[split]
        symbols = torch.from_numpy(np.asarray(rows["symbols"])).long()
        references = rows.get("references")
        signals = rows["signals"]
        if control and model.uses_hyper:
            order = m23.permuted_hyper_signal(signals.shape[0],
                                              seed=args.seed + seed_offset)
            signals = signals[torch.from_numpy(order).long()]
        return symbols, references, signals

    train_batch = batch_for("train", 0)
    select_batch = batch_for("val_a", 1)
    report_batch = batch_for("val_b", 2)

    history = m23.train_arm(
        model, train_batch, select_batch, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, device=device, hyper_bits=args.hyper_bits,
        seed=args.seed, log=log)

    hyper_table, train_clamped = (None, 0.0)
    if model.uses_hyper:
        train_symbols, train_clamped = m23.collect_hyper_symbols(
            model, train_batch[2], device=device, hyper_bits=args.hyper_bits)
        hyper_table = m23.fit_hyper_entropy_model(
            train_symbols, bits=args.hyper_bits, hyper_channels=args.hyper_channels)

    evaluation = m23.evaluate_arm(
        model, report_batch[0], report_batch[1], report_batch[2], device=device,
        hyper_bits=args.hyper_bits, hyper_entropy_model=hyper_table)
    evaluation.update({
        "channel": channel,
        "is_control": control,
        "parameters": model.parameter_count(),
        "selected_epoch": min(history, key=lambda h: (h["val_loss"], h["epoch"]))["epoch"],
        "val_a_bits_at_selection": min(h["val_loss"] for h in history),
        "train_hyper_clamped_fraction": train_clamped,
        "hyper_table_id": (hyper_table.model_id().hex() if hyper_table is not None else None),
        "seconds": time.perf_counter() - started,
        "history": history,
    })
    return evaluation


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    m23 = _load_script("m23_hyperprior")
    mc = _load_script("m10h_motion_compensation")
    md = _load_script("m11_data")
    m14 = _load_script("m14_recalibration")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / args.output_name
    configuration = {
        "checkpoint": str(args.checkpoint), "rate_points": list(args.rate_points),
        "channels": list(args.channels), "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "hyper_bits": args.hyper_bits, "hyper_channels": args.hyper_channels,
        "hyper_hidden": args.hyper_hidden, "hidden": args.hidden,
        "calibration_frames": args.calibration_frames,
        "intra_train_frames": args.intra_train_frames,
        "intra_val_frames_per_sequence": args.intra_val_frames_per_sequence,
        "gop": args.gop, "block_size": args.block_size, "search_range": args.search_range,
        "seed": args.seed,
    }
    report: dict[str, Any] = {
        "phase": "M23 Phase A - hyperprior offline gate",
        "question": ("does a TRANSMITTED description of the current content buy more bits "
                     "than it costs, on top of what the decoder already gets free?"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "pre_registered": {
            "arms_per_channel": {k: list(v) for k, v in CHANNEL_ARMS.items()},
            "baseline_per_channel": dict(CHANNEL_BASELINE),
            "control_arm_per_channel": dict(CHANNEL_CONTROL_ARM),
            "weak_below_percent": m23.WEAK_BELOW_PERCENT,
            "meaningful_above_percent": m23.MEANINGFUL_ABOVE_PERCENT,
            "control_tolerance_percent": CONTROL_TOLERANCE_PERCENT,
            "net_of_control_rule": ("gain - max(control, 0), M11's rule: a negative "
                                    "control is the cost of a useless signal and is "
                                    "not added back"),
            "side_channel_is_charged": True,
            "split_discipline": ("TRAIN fits models and tables, VAL-A selects the epoch, "
                                 "VAL-B reports once per arm, TEST never opened"),
        },
        "limitations": [
            ("the residual baseline is zref_only (M10K's family), NOT the deployed stack, "
             "which adds M11-G16's causal context and M13 on top - worth a further "
             "+5.3-6.0% of coded bytes on this transform. A gain here is a gain over z_ref "
             "conditioning, and whether it survives alongside causal context is Phase B."),
            ("measured at FROZEN transform weights, which were fitted under a per-channel "
             "Laplace rate term, so this is a LOWER bound on what joint training would give."),
            ("h_a is given the DEQUANTIZED signal, not the continuous one, which is the "
             "conservative choice and biases the gate down."),
        ],
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
        else:
            print("[resume] configuration changed; starting fresh", flush=True)

    def persist() -> None:
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    done = {entry["bits"] for entry in report["rate_points"]}
    train_full = discover_sequences(args.manifest, split="train")

    print("=" * 110)
    print("M23 PHASE A - HYPERPRIOR OFFLINE GATE (VAL-B, held out; TEST never opened)")
    print(f"  transform: {args.checkpoint}")
    print(f"  channels: {args.channels}   rate points: {args.rate_points}")
    print(f"  bars: weak < {m23.WEAK_BELOW_PERCENT}%, "
          f"meaningful > {m23.MEANINGFUL_ABOVE_PERCENT}%   "
          f"side channel charged: yes")
    print("=" * 110, flush=True)

    for bits in args.rate_points:
        if bits in done:
            print(f"  [resume] {bits}-bit already measured", flush=True)
            continue
        print(f"\n  {bits}-bit", flush=True)
        started = time.perf_counter()
        calibration = mc.calibrate_grids(
            model, train_full, bits=bits, mode="per_channel", gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            reference_mode="mc", max_frames=args.calibration_frames)

        entry: dict[str, Any] = {"bits": bits, "channels": {}}
        for channel in args.channels:
            print(f"    channel: {channel}", flush=True)
            if channel == "intra":
                data = intra_data(m23, model, args, calibration, device,
                                  log=lambda m: print(m, flush=True))
                baseline_bits = static_table_bits(
                    m14, calibration["intra_entropy_model"], data["val_b"]["symbols"])
                baseline = {"name": "deployed_static_table",
                            "net_bits_per_symbol": baseline_bits,
                            "y_bits_per_symbol": baseline_bits,
                            "side_bits_per_symbol": 0.0,
                            "note": ("the deployed per-channel histogram - the only entropy "
                                     "model the intra path has")}
            else:
                data = residual_data(m23, md, model, args, bits, device,
                                     log=lambda m: print(m, flush=True))
                print(f"      arm: zref_only (the in-family baseline)", flush=True)
                baseline = run_arm(m23, arm="zref_only", channel=channel, data=data,
                                   args=args, bits=bits, device=device, control=False,
                                   log=lambda m: print(m, flush=True))
                baseline["name"] = "zref_only"
                baseline["note"] = ("M10K's family, trained here under identical "
                                    "conditions - the information the decoder gets free")
            print(f"      baseline {baseline['name']}: "
                  f"{baseline['net_bits_per_symbol']:.5f} bits/symbol", flush=True)

            arms: dict[str, Any] = {}
            for arm in CHANNEL_ARMS[channel]:
                print(f"      arm: {arm}", flush=True)
                result = run_arm(m23, arm=arm, channel=channel, data=data, args=args,
                                 bits=bits, device=device, control=False,
                                 log=lambda m: print(m, flush=True))
                gain = 100.0 * (baseline["net_bits_per_symbol"]
                                - result["net_bits_per_symbol"]) / baseline[
                                    "net_bits_per_symbol"]
                result["gain_percent_vs_baseline"] = gain
                result["verdict"] = m23.verdict(gain)
                arms[arm] = result
                print(f"        -> {result['net_bits_per_symbol']:.5f} bits/symbol "
                      f"(y {result['y_bits_per_symbol']:.5f} + side "
                      f"{result['side_bits_per_symbol']:.5f})   "
                      f"gain {gain:+.4f}%  {result['verdict']}", flush=True)

            control_arm = CHANNEL_CONTROL_ARM[channel]
            print(f"      control: {control_arm} with a whole-split permuted signal",
                  flush=True)
            control = run_arm(m23, arm=control_arm, channel=channel, data=data, args=args,
                              bits=bits, device=device, control=True,
                              log=lambda m: print(m, flush=True))
            control_gain = 100.0 * (baseline["net_bits_per_symbol"]
                                    - control["net_bits_per_symbol"]) / baseline[
                                        "net_bits_per_symbol"]
            control["gain_percent_vs_baseline"] = control_gain
            net = net_of_control(arms[control_arm]["gain_percent_vs_baseline"],
                                 control_gain)
            print(f"        -> control gain {control_gain:+.4f}%   "
                  f"NET OF CONTROL {net:+.4f}%  {m23.verdict(net)}", flush=True)
            print(f"           {control_interpretation(control_gain)}", flush=True)

            entry["channels"][channel] = {
                "baseline": {k: v for k, v in baseline.items() if k != "history"},
                "arms": {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                         for k, v in arms.items()},
                "control": {k: v for k, v in control.items() if k != "history"},
                "control_arm": control_arm,
                "control_gain_percent": control_gain,
                "control_within_tolerance": abs(control_gain) <= CONTROL_TOLERANCE_PERCENT,
                "control_interpretation": control_interpretation(control_gain),
                "net_of_control_percent": net,
                "net_of_control_verdict": m23.verdict(net),
                "net_of_control_rule": ("gain - max(control, 0); a negative control is "
                                        "not added back (M11's rule)"),
                "histories": {k: v["history"] for k, v in arms.items()},
            }
            if channel == "residual":
                entry["channels"][channel]["deployed_stack_context"] = \
                    deployed_stack_context(args.m11_dir, bits)
            persist()

        entry["seconds"] = time.perf_counter() - started
        report["rate_points"].append(entry)
        persist()

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    persist()

    print("\n" + "=" * 110)
    print("SUMMARY - net of control, on VAL-B")
    print("=" * 110)
    print(f"{'bits':>5} {'channel':>10} {'arm':>18} {'baseline':>10} {'net':>10} "
          f"{'side%':>7} {'gain':>10} {'net of ctrl':>12} {'verdict':>11}")
    for entry in sorted(report["rate_points"], key=lambda e: -e["bits"]):
        for channel, block in entry["channels"].items():
            base = block["baseline"]["net_bits_per_symbol"]
            for arm, result in block["arms"].items():
                marker = " *" if arm == block["control_arm"] else "  "
                shown = (block["net_of_control_percent"] if arm == block["control_arm"]
                         else result["gain_percent_vs_baseline"])
                print(f"{entry['bits']:>5} {channel:>10} {arm:>16}{marker} {base:>10.5f} "
                      f"{result['net_bits_per_symbol']:>10.5f} "
                      f"{100 * result['side_share_of_net']:>6.2f}% "
                      f"{result['gain_percent_vs_baseline']:>+9.4f}% {shown:>+11.4f}% "
                      f"{m23.verdict(shown):>11}")
    print("\n  * = the arm the control was run against; its 'net of ctrl' column is net "
          "of the permuted-signal control")
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
