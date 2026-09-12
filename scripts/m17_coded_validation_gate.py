"""M17 Phase F - coded validation gate. Triggered because Phase E's
total-stream upper bound cleared 0.5% at every tested bit depth (2.03% /
6.75% / 14.86% at 5/4/3-bit - see m17_residual_diagnostic.json).

WHAT "CODED VALIDATION" CAN AND CANNOT MEAN FOR AN ORACLE CANDIDATE
--------------------------------------------------------------------
The oracle reference is, by construction, unavailable to any real decoder:
it is built from the RAW, uncoded previous frame (`model.decode(model.encode(
raw_previous_frame))`), which a real decoder never has - it only ever has
whatever the coded stream itself carries. There is therefore no such thing
as a deployable "oracle .nvct stream": producing one would require
transmitting the previous frame twice (once as the reference, once as
itself), which defeats the entire purpose of temporal compression. This
script does NOT attempt to construct one.

What CAN be verified, and is the entire point of this script: that the
byte counts reported in Phase B/D/E are REAL, CORRECTLY ENTROPY-CODED
NUMBERS - not a computation artifact - by proving each oracle-variant
payload actually DECODES back to the exact symbols it was built from,
through the SAME unmodified `m13_recalibration.decode_frame_recalibrated`,
given the SAME reference a hypothetical oracle-aware decoder would need.
This corroborates "the bytes are genuine" without ever claiming "this is
implementable" - those are different questions, and only the first one is
in scope here.

No `.nvct` v2 container is created (this operates on raw frame payloads via
`encode_frame_recalibrated`/`decode_frame_recalibrated` directly, exactly as
Phase B/D/E did) - so there is no container-level identity to check; the
underlying model11/assign_codebook/coding_codebook objects are the SAME,
already-verified-against-the-record ones from Phase 0, never new.

Run:
  ./.venv/Scripts/python.exe scripts/m17_coded_validation_gate.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m17_residual_oracle_audit")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M11_DIR = Path("outputs/m11_autoregressive_entropy")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
RATE_POINTS = (5, 4, 3)
M11_G16_GROUP_SIZE = 16


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M17 Phase F: verify oracle-variant byte counts are genuine, decodable numbers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=DEFAULT_M11_DIR)
    parser.add_argument("--m10k-dir", type=Path, default=DEFAULT_M10K_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--val-sequences", type=int, default=1)
    parser.add_argument("--val-frames-per-sequence", type=int, default=30)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
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

    mc = _load_script("m10h_motion_compensation")
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    md = _load_script("m11_data")
    cx = _load_script("m11_causal_context")
    mk = _load_script("m10k_learned_entropy")
    m13 = _load_script("m13_recalibration")
    ev = _load_script("m10l_evaluate")
    ev_m11 = _load_script("m11_evaluate")
    gate_script = _load_script("m12_spatial_offline_gate")
    m17 = _load_script("m17_residual_diagnostic")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    val_sequences = discover_sequences(args.manifest, split="val",
                                       max_frames_per_sequence=args.val_frames_per_sequence)
    val_b = val_sequences[1::2][:args.val_sequences]
    train_sequences_full = discover_sequences(args.manifest, split="train")

    print("=" * 118, flush=True)
    print("M17 PHASE F - CODED VALIDATION GATE (oracle-variant round-trip proof)")
    print("=" * 118)
    print(f"  triggered by: Phase E total-stream upper bound >= 0.5% at every rate point "
         f"(2.03% / 6.75% / 14.86% at 5/4/3-bit)")
    print(f"  VAL sample: {[s.sequence_id for s in val_b]}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"phase": "M17 Phase F coded validation gate", "rate_points": []}
    overall_pass = True

    for bits in args.rate_points:
        calibration = mc.calibrate_grids(
            model, train_sequences_full, bits=bits, mode="per_channel", gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, reference_mode="mc",
            max_frames=args.calibration_frames)
        signature = ev.calibration_signature(calibration, bits=bits,
                                             calibration_frames=args.calibration_frames,
                                             quant_mode="per_channel")
        intra_params, intra_entropy_model = calibration["intra_params"], calibration["intra_entropy_model"]
        residual_params = calibration["residual_params"]

        data = md.load_or_collect(model, checkpoint=args.checkpoint, manifest=args.manifest,
                                  bits=bits, device=device, cache_dir=cache_dir, log=lambda m: None)
        train_symbols = data["train_symbols"].astype(np.int64)
        val_symbols = data["val_symbols"].astype(np.int64)
        select_mask, _ = md.split_validation(data)
        channels = train_symbols.shape[1]
        zero = torch.from_numpy(cx.zero_symbols(residual_params, channels))
        m10k, m10k_checkpoint = mk.load_entropy_model(
            args.m10k_dir / f"learned_entropy_{bits}bit.pt", device=device)
        m10k_identity = ev._m10k_model_identity(m10k_checkpoint["model_state_dict"])
        model11, checkpoint11 = ma.load_model(args.m11_dir / f"m11_G16_entropy_{bits}bit.pt",
                                              device=device)
        ev_m11.check_provenance(checkpoint11, signature=signature, bits=bits,
                                group_size=M11_G16_GROUP_SIZE,
                                context_definition_id=ma.context_definition_id(M11_G16_GROUP_SIZE),
                                m10k_identity=m10k_identity)
        assign_codebook = ml.SharedCodebook.from_dict(json.loads(
            (args.m11_dir / f"m11_G16_codebook_{bits}bit_K512.json").read_text(encoding="utf-8")))
        train_k = gate_script.m11_g16_prototype_indices(model11, assign_codebook, train_symbols,
                                                        data["train_references"], zero, device=device)
        val_k = gate_script.m11_g16_prototype_indices(model11, assign_codebook, val_symbols,
                                                      data["val_references"], zero, device=device)
        flat = lambda a: a.reshape(-1)
        frequencies, strength, _, _ = m13.fit_recalibrated_frequencies(
            assign_codebook, flat(train_symbols), flat(train_k),
            flat(val_symbols[select_mask]), flat(val_k[select_mask]), alphabet=2 ** bits)
        coding_codebook = m13.build_recalibrated_codebook(
            assign_codebook, frequencies, provenance={"bits": bits, "strength": strength})
        residual_identity = ma.model_identity(model11, m10k_identity=m10k_identity,
                                              calibration_signature=signature, bits=bits,
                                              codebook=coding_codebook)

        rows = []
        for sequence in val_b:
            rows.extend(m17.diagnose_residual_oracle(
                mc, ma, m13, mk, model, model11, assign_codebook, coding_codebook, sequence,
                bits=bits, intra_params=intra_params, intra_entropy_model=intra_entropy_model,
                residual_params=residual_params, zero=zero, gop_size=args.gop,
                block_size=args.block_size, search_range=args.search_range, device=device,
                keep_payloads_and_references=True))

        checked = 0
        all_match = True
        byte_mismatch_examples = []
        latent_shape = tuple(rows[0]["C_full_oracle_reference"].shape[1:]) if rows else None
        for r in rows:
            for name in ("A_real", "B_oracle_ref_real_motion", "C_full_oracle"):
                payload = r[f"{name}_payload"]
                reference = r[f"{name}_reference"].to(device)
                expected_symbols = r[f"{name}_symbols"]
                decoded = m13.decode_frame_recalibrated(
                    model11, assign_codebook, coding_codebook, payload, reference, zero,
                    bits=bits, shape=latent_shape)
                match = np.array_equal(decoded, expected_symbols)
                all_match &= match
                checked += 1
                if not match:
                    byte_mismatch_examples.append({"index": r["index"], "variant": name})
                # The reported byte count IS len(payload) by construction (Phase B/D/E
                # computed it the same way) - re-confirm here rather than trust it twice.
                assert len(payload) == r[f"{name}_bytes"]

        overall_pass &= all_match
        total_a = sum(r["A_real_bytes"] for r in rows)
        total_c = sum(r["C_full_oracle_bytes"] for r in rows)
        gain = (total_a - total_c) / total_a * 100
        print(f"  {bits}-bit: {checked} payloads round-trip-verified, all_match={all_match}  "
             f"(sample: {len(rows)} P-frames, A={total_a:,} bytes, C={total_c:,} bytes, "
             f"gain={gain:+.4f}% on this smaller sample)", flush=True)
        report["rate_points"].append({
            "bits": bits, "residual_identity": residual_identity.hex(),
            "payloads_checked": checked, "all_round_trips_match": all_match,
            "mismatch_examples": byte_mismatch_examples,
            "sample_total_bytes": {"A_real": total_a, "C_full_oracle": total_c},
            "sample_channel_gain_percent": gain,
        })

    report["overall_verdict"] = (
        "CORROBORATED - every oracle-variant payload decodes back to its exact symbols; "
        "reported byte savings are genuine entropy-coder output, not a computation artifact. "
        "This does NOT mean the oracle is implementable - no real decoder can ever possess "
        "the oracle reference without transmitting the previous frame twice, which this "
        "milestone's freezes correctly forbid attempting to design around."
        if overall_pass else
        "FAILED - a payload did not round-trip; STOP and investigate before trusting Phase B/D/E's bytes.")
    print(f"\n  VERDICT: {report['overall_verdict']}")

    path = args.output_dir / "m17_coded_validation_gate.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
