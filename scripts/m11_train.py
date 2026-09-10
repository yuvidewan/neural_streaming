"""M11 - train channel-autoregressive entropy models, then check actual bytes and time.

Per rate point, on the SAME cached TRAIN/VAL symbols as the offline gate:

  M11-G{g}        warm-started from M10K, channel groups of size g
                  (g = 1 is full channel autoregression: 64 sequential steps)
  M11-nocontext   identical network and training, context planes held at zero

All are fitted on TRAIN, selected by VAL-A, reported on VAL-B. The no-context
arm is the learned counterpart of the gate's "recalibrated M10L": continued
training alone can recover some of M10K's miscalibration without any
autoregression, and without it that gain would be credited to causal context.

Each model's probabilities reach the coder two ways - one table per position,
and through a 512-entry codebook fitted to that model's TRAIN predictions - and
every VAL-B P-frame is actually arithmetic-coded, decoded back through the
group-sequential decoder, checked bit-exact, and timed stage by stage.

OPERATING POINT (pre-declared before any group size was measured)
------------------------------------------------------------------
Group size trades context for sequential steps. The practical point is the
LARGEST group size - fewest decode steps - whose VAL-A context gain keeps at
least half of the G = 1 gain at every rate point. G = 1 is always reported as
the compression ceiling of this model family.

TEST is never read.

Run (after scripts/m11_offline_gate.py has cached the data):
  ./.venv/Scripts/python.exe scripts/m11_train.py
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

from nvc.compression.range_coder import decode_symbols, encode_symbols
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m11_autoregressive_entropy")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
DEFAULT_M10L_DIR = Path("outputs/m10l_shared_codebook/codebooks")
RATE_POINTS = (5, 4, 3)
GROUP_SIZES = (1, 2, 4, 8, 16)
RETAINED_FRACTION = 0.5          # pre-declared operating-point rule


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percent(baseline: float, value: float) -> float:
    return (baseline - value) / baseline * 100.0


def fit_model_codebook(ml, model, train_set, zero, *, bits, size, rows, device, seed):
    """A shared codebook fitted to THIS model's predictions on TRAIN symbols."""
    alphabet = 2 ** bits
    total = train_set[0].shape[0] * train_set[0][0].numel()
    stride = max(1, total // rows)
    samples = []
    with torch.no_grad():
        for start in range(0, train_set[0].shape[0], 8):
            target = train_set[0][start:start + 8].to(device)
            log_probabilities = model.log_probabilities(train_set[1][start:start + 8].to(device),
                                                        model.planes(target, zero.to(device)))
            flat = log_probabilities.exp().permute(0, 1, 3, 4, 2).reshape(-1, alphabet)
            samples.append(flat[::stride].double().cpu().numpy())
    samples = np.concatenate(samples)[:rows]
    return ml.fit_codebook(samples, size, bits=bits, seed=seed,
                           provenance={"source": "M11 predictions on TRAIN"})


def coded_gate(ma, mc, model, report_set, zero, *, bits, device, codebook, frames):
    """Actually code VAL-B frames; decode back; time every stage."""
    enc, dec = {}, {}
    ideal_bits, emitted, exact = 0.0, 0, True
    with torch.no_grad(), mc.deterministic_kernels():
        warm = report_set[0][0].numpy()
        warm_ref = report_set[1][0][None].to(device)
        payload, _ = ma.encode_frame(model, warm_ref, warm, zero, bits=bits, codebook=codebook)
        ma.decode_frame(model, payload, warm_ref, zero, bits=bits, shape=warm.shape,
                        codebook=codebook)
        for i in frames:
            symbols = report_set[0][i].numpy()
            reference = report_set[1][i][None].to(device)
            payload, ideal = ma.encode_frame(model, reference, symbols, zero, bits=bits,
                                             codebook=codebook, timings=enc)
            back = ma.decode_frame(model, payload, reference, zero, bits=bits,
                                   shape=symbols.shape, codebook=codebook, timings=dec)
            exact &= bool(np.array_equal(back, symbols.reshape(-1)))
            ideal_bits += ideal
            emitted += len(payload)
    count = len(frames)
    to_ms = lambda d: {k: v / count * 1000.0 for k, v in d.items()}
    return {"ideal_bits": ideal_bits, "bytes": emitted, "round_trip_exact": exact,
            "encode_ms": to_ms(enc), "decode_ms": to_ms(dec),
            "encode_total_ms": sum(to_ms(enc).values()),
            "decode_total_ms": sum(to_ms(dec).values())}


def m10l_coded(ml, mc, m10k, codebook, report_set, *, device, frames):
    enc, dec = {"tables": 0.0, "coder": 0.0}, {"tables": 0.0, "coder": 0.0}
    ideal_bits, emitted, exact = 0.0, 0, True
    with torch.no_grad(), mc.deterministic_kernels():
        warm_ref = report_set[1][0][None].to(device)
        table = ml.frame_table_index(m10k, codebook, warm_ref)
        encode_symbols(report_set[0][0].numpy().reshape(-1), codebook.cumulative, table)
        for i in frames:
            flat = report_set[0][i].numpy().reshape(-1)
            reference = report_set[1][i][None].to(device)
            for side in (enc, dec):
                started = time.perf_counter()
                table = ml.frame_table_index(m10k, codebook, reference)
                side["tables"] += time.perf_counter() - started
            started = time.perf_counter()
            payload = encode_symbols(flat, codebook.cumulative, table)
            enc["coder"] += time.perf_counter() - started
            started = time.perf_counter()
            back = decode_symbols(payload, flat.size, codebook.cumulative, table)
            dec["coder"] += time.perf_counter() - started
            exact &= bool(np.array_equal(back, flat))
            ideal_bits += codebook.expected_bits(flat, table)
            emitted += len(payload)
    count = len(frames)
    to_ms = lambda d: {k: v / count * 1000.0 for k, v in d.items()}
    return {"ideal_bits": ideal_bits, "bytes": emitted, "round_trip_exact": exact,
            "encode_ms": to_ms(enc), "decode_ms": to_ms(dec),
            "encode_total_ms": sum(to_ms(enc).values()),
            "decode_total_ms": sum(to_ms(dec).values())}


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M11: train channel-autoregressive entropy models; coded-byte gate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m10k-dir", type=Path, default=DEFAULT_M10K_DIR)
    parser.add_argument("--m10l-dir", type=Path, default=DEFAULT_M10L_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--group-sizes", type=int, nargs="+", default=list(GROUP_SIZES))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--codebook-fit-rows", type=int, default=100_000)
    parser.add_argument("--coded-frames", type=int, default=None,
                        help="limit VAL-B frames in the coded-byte gate (default: all)")
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--train-frames", type=int, default=600)
    parser.add_argument("--val-frames-per-sequence", type=int, default=40)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    if 1 not in args.group_sizes:
        print("[ERROR] group size 1 is the reference every other size is judged against",
              file=sys.stderr)
        return 1

    mc = _load_script("m10h_motion_compensation")
    mk = _load_script("m10k_learned_entropy")
    ml = _load_script("m10l_shared_codebook")
    md = _load_script("m11_data")
    ma = _load_script("m11_ar_entropy")
    cx = _load_script("m11_causal_context")
    ev = _load_script("m10l_evaluate")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    print("=" * 116, flush=True)
    print("M11 - CHANNEL-AUTOREGRESSIVE ENTROPY MODELS: TRAINING, CODED BYTES, LATENCY")
    print("=" * 116)
    print(f"  group sizes {args.group_sizes}; fitted on TRAIN, selected on VAL-A, "
          f"reported on VAL-B; TEST unread")
    print(f"  operating point (pre-declared): largest G keeping >= {RETAINED_FRACTION:.0%} of "
          f"the G=1 VAL-A context gain at every rate point")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"phase": "M11 training, coded bytes and latency",
                              "context_definition": ma.CONTEXT_DEFINITION,
                              "group_sizes": args.group_sizes,
                              "retained_fraction_rule": RETAINED_FRACTION, "rate_points": []}

    for bits in args.rate_points:
        print(f"\n  ---- {bits}-bit ----", flush=True)
        data = md.load_or_collect(
            model, checkpoint=args.checkpoint, manifest=args.manifest, bits=bits,
            device=device, calibration_frames=args.calibration_frames,
            train_frames=args.train_frames,
            val_frames_per_sequence=args.val_frames_per_sequence,
            cache_dir=args.cache_dir or md.DEFAULT_CACHE_DIR,
            log=lambda m: print(m, flush=True))
        calibration = data["calibration"]
        select_mask, report_mask = md.split_validation(data)
        as_long = lambda a: torch.from_numpy(np.asarray(a)).long()
        as_float = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
        train_set = (as_long(data["train_symbols"]), as_float(data["train_references"]))
        select_set = (as_long(data["val_symbols"][select_mask]),
                      as_float(data["val_references"][select_mask]))
        report_set = (as_long(data["val_symbols"][report_mask]),
                      as_float(data["val_references"][report_mask]))
        channels = train_set[0].shape[1]
        zero = torch.from_numpy(cx.zero_symbols(calibration["residual_params"], channels))
        frames = list(range(report_set[0].shape[0] if args.coded_frames is None
                            else min(args.coded_frames, report_set[0].shape[0])))

        m10k, m10k_checkpoint = mk.load_entropy_model(
            args.m10k_dir / f"learned_entropy_{bits}bit.pt", device=device)
        m10k_identity = ev._m10k_model_identity(m10k_checkpoint["model_state_dict"])
        signature = ev.calibration_signature(calibration, bits=bits,
                                             calibration_frames=args.calibration_frames,
                                             quant_mode="per_channel")
        m10l_codebook = ml.SharedCodebook.from_dict(json.loads(
            (args.m10l_dir / f"codebook_{bits}bit_K512_code_length.json").read_text(
                encoding="utf-8")))

        # --- baselines on VAL-B ------------------------------------------------------
        with torch.no_grad(), mc.deterministic_kernels():
            m10k_bits = mk.cross_entropy_bits(m10k, list(report_set[0].numpy()),
                                              list(report_set[1].numpy()), device=device)
            # M10L's VAL-B cross-entropy over EVERY VAL-B frame, independent of how
            # many frames the (slower) coded-byte gate actually codes - otherwise a
            # --coded-frames limit would compare different frame sets.
            m10l_total = 0.0
            for i in range(report_set[0].shape[0]):
                table = ml.frame_table_index(m10k, m10l_codebook,
                                             report_set[1][i][None].to(device))
                m10l_total += m10l_codebook.expected_bits(report_set[0][i].numpy().reshape(-1),
                                                          table)
        m10l_bits = m10l_total / report_set[0].numel()
        m10l = m10l_coded(ml, mc, m10k, m10l_codebook, report_set, device=device, frames=frames)
        coded_symbols = len(frames) * report_set[0][0].numel()

        # --- the arms ------------------------------------------------------------------
        arms: dict[str, dict[str, Any]] = {}
        specs = [("nocontext", False, 1)] + [(f"G{g}", True, g) for g in args.group_sizes]
        for label, use_context, group in specs:
            print(f"    training M11-{label} ...", flush=True)
            arm = ma.from_m10k(m10k, use_context=use_context, group_size=group).to(device)
            history = ma.train(arm, train_set, select_set, zero, epochs=args.epochs,
                               batch_size=args.batch_size, learning_rate=args.learning_rate,
                               device=device, seed=args.seed,
                               log=lambda m: print("      " + m, flush=True))
            selected = min(history, key=lambda h: (h["val_loss"], h["epoch"]))
            with torch.no_grad():
                val_b = ma.nll_bits(arm, *report_set, zero, device=device)
            entry = {"use_context": use_context, "group_size": group,
                     "sequential_steps": channels // group if use_context else 1,
                     "selected_epoch": selected["epoch"], "val_a_bits": selected["val_loss"],
                     "val_b_bits": val_b, "history": history,
                     "parameters": sum(p.numel() for p in arm.parameters())}
            path = args.output_dir / f"m11_{label}_entropy_{bits}bit.pt"
            if use_context:
                codebook = fit_model_codebook(ml, arm, train_set, zero, bits=bits,
                                              size=args.codebook_size,
                                              rows=args.codebook_fit_rows, device=device,
                                              seed=args.seed)
                (args.output_dir / f"m11_{label}_codebook_{bits}bit_K{args.codebook_size}.json"
                 ).write_text(json.dumps(codebook.to_dict()), encoding="utf-8")
                entry["coded_per_position"] = coded_gate(
                    ma, mc, arm, report_set, zero, bits=bits, device=device, codebook=None,
                    frames=frames)
                entry["coded_codebook"] = coded_gate(
                    ma, mc, arm, report_set, zero, bits=bits, device=device, codebook=codebook,
                    frames=frames)
                entry["identity"] = ma.model_identity(
                    arm, m10k_identity=m10k_identity, calibration_signature=signature,
                    bits=bits).hex()
                entry["identity_codebook"] = ma.model_identity(
                    arm, m10k_identity=m10k_identity, calibration_signature=signature,
                    bits=bits, codebook=codebook).hex()
            torch.save({"model_state_dict": arm.state_dict(), "model_config": arm.config_dict(),
                        "history": history, "selected_epoch": selected["epoch"], "bits": bits,
                        "context_definition_id": ma.context_definition_id(group),
                        "m10k_identity": m10k_identity.hex(),
                        "calibration_signature": signature,
                        "split": {"fit": "train", "select": "val-a", "report": "val-b"}}, path)
            entry["checkpoint_bytes"] = path.stat().st_size
            arms[label] = entry
            line = (f"      -> epoch {selected['epoch']}, VAL-B {val_b:.5f} bits/symbol "
                    f"({_percent(m10l_bits, val_b):+.3f}% vs M10L)")
            if use_context:
                pp, cb = entry["coded_per_position"], entry["coded_codebook"]
                line += (f" | coded bytes {_percent(m10l['bytes'], pp['bytes']):+.3f}% "
                         f"(codebook {_percent(m10l['bytes'], cb['bytes']):+.3f}%) | decode "
                         f"{pp['decode_total_ms']:.1f} ms (codebook {cb['decode_total_ms']:.1f}) "
                         f"| exact {pp['round_trip_exact'] and cb['round_trip_exact']}")
            print(line, flush=True)
            if use_context and not (entry["coded_per_position"]["round_trip_exact"]
                                    and entry["coded_codebook"]["round_trip_exact"]):
                print("[ERROR] a round trip was not bit-exact - STOPPING", file=sys.stderr)
                return 1

        # --- decomposition --------------------------------------------------------------
        nocontext_a = arms["nocontext"]["val_a_bits"]
        for entry in arms.values():
            entry["context_gain_val_a_percent"] = _percent(nocontext_a, entry["val_a_bits"])
            entry["context_gain_val_b_percent"] = _percent(arms["nocontext"]["val_b_bits"],
                                                           entry["val_b_bits"])
            entry["vs_m10l_val_b_percent"] = _percent(m10l_bits, entry["val_b_bits"])
        report["rate_points"].append({
            "bits": bits, "val_b_symbols_coded": coded_symbols, "coded_frames": len(frames),
            "m10k_val_b_bits": m10k_bits, "m10l_val_b_bits": m10l_bits, "m10l_coded": m10l,
            "arms": arms, "m10k_identity": m10k_identity.hex(),
            "calibration_signature": signature})

        print(f"\n    {bits}-bit summary (VAL-B; M10L {m10l_bits:.5f} bits/symbol, "
              f"{m10l['bytes']:,} B, decode {m10l['decode_total_ms']:.2f} ms/P):")
        print(f"      {'arm':<11} {'steps':>5} {'bits/sym':>9} {'vs M10L':>8} {'ctx gain':>8} "
              f"{'bytes vs L':>10} {'cb bytes':>9} {'enc ms':>7} {'dec ms':>7} "
              f"{'cb dec ms':>9}")
        for label, entry in arms.items():
            if not entry["use_context"]:
                print(f"      {'nocontext':<11} {1:>5} {entry['val_b_bits']:>9.5f} "
                      f"{entry['vs_m10l_val_b_percent']:>+7.3f}%")
                continue
            pp, cb = entry["coded_per_position"], entry["coded_codebook"]
            print(f"      {label:<11} {entry['sequential_steps']:>5} {entry['val_b_bits']:>9.5f} "
                  f"{entry['vs_m10l_val_b_percent']:>+7.3f}% "
                  f"{entry['context_gain_val_b_percent']:>+7.3f}% "
                  f"{_percent(m10l['bytes'], pp['bytes']):>+9.3f}% "
                  f"{_percent(m10l['bytes'], cb['bytes']):>+8.3f}% "
                  f"{pp['encode_total_ms']:>7.2f} {pp['decode_total_ms']:>7.2f} "
                  f"{cb['decode_total_ms']:>9.2f}")

    # --- the pre-declared operating point, on VAL-A across every rate point ----------
    chosen = 1
    for group in sorted(g for g in args.group_sizes if g > 1):
        keeps = all(rp["arms"][f"G{group}"]["context_gain_val_a_percent"]
                    >= RETAINED_FRACTION * rp["arms"]["G1"]["context_gain_val_a_percent"]
                    for rp in report["rate_points"])
        if keeps:
            chosen = group
    report["operating_point_group_size"] = chosen
    print(f"\n  OPERATING POINT (VAL-A rule): G = {chosen} "
          f"({64 // chosen} sequential decode steps)")
    path = args.output_dir / "m11_training.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
