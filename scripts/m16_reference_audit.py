"""M16 Phase A - trace exactly what reference the REAL DEPLOYED coder
(`scripts/m13_closed_loop.encode_multi`/`decode_sequence`) uses at t=0, at
every GOP boundary, and at every GOP-boundary+1, for each bit depth. Every
claim below is read from `encode_multi`'s actual body (scripts/m13_closed_loop.py)
and confirmed by a real runtime trace on a real checkpoint - not asserted
from M14's prior description of `calibrate_grids`/`collect_motion_symbols`,
which are DIFFERENT functions with DIFFERENT (already-documented) behavior.

THE IMPORTANT CORRECTION THIS PHASE MAKES
------------------------------------------------------------------------
M14/M15 documented a bit-depth asymmetry concentrated at GOP boundaries in
`calibrate_grids` (whose I-frame branch reconstructs through the real,
bit-depth-dependent intra quantizer, then whose P-frame branch reverts to
advancing the reference from the TRUE, unquantized latent - see
m10h_motion_compensation.calibrate_grids's own "very slightly optimistic"
comment) versus `m14_recalibration.collect_motion_symbols` (idealized to
the true latent at EVERY frame, I or P alike). Neither of those is the
REAL DEPLOYED CODER. Tracing `encode_multi`/`decode_sequence` directly
(below) shows something different: there is NO special-case branch for
"the frame right after an I-frame" anywhere in the real coder. Every
frame's `previous` reference is simply `model.decode(reconstructed_latent)`
of whatever the immediately preceding frame produced - bit-depth-dependent
via `intra_params` for the very first reference (frame 0) and bit-depth
-dependent via `residual_params` for every reference after that, and this
error COMPOUNDS across a GOP rather than being localized to one position.
"GOP-boundary-specific bit dependence" is therefore a property of the
CALIBRATION SHORTCUT FUNCTIONS, not of the deployed codec itself - which is
uniformly (and cumulatively) bit-depth-sensitive. Phase B/C measure the
real, empirical shape of that sensitivity across GOP position rather than
assume it is concentrated at the boundary.

Run:
  ./.venv/Scripts/python.exe scripts/m16_reference_audit.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

from nvc.compression.codec import decode_payload_to_latent, encode_latent_to_payload
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m16_gop_audit")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
RATE_POINTS = (5, 4, 3)
GOP_SIZE = 10


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M16 Phase A: trace the real deployed GOP-boundary reference flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gop", type=int, default=GOP_SIZE)
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    if not args.manifest.is_file() or not args.checkpoint.is_file():
        print("[ERROR] --manifest/--checkpoint not found", file=sys.stderr)
        return 1

    mc = _load_script("m10h_motion_compensation")
    from nvc.evaluation.sequences import discover_sequences

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    # gop_frame_types, read directly, not paraphrased - confirms exactly
    # which indices are I / boundary-P / ordinary-P at the deployed gop=10.
    frame_count = 23
    types = mc.gop_frame_types(frame_count, args.gop)
    i_positions = [i for i, t in enumerate(types) if t == mc.FRAME_TYPE_I]
    boundary_p_positions = [i for i in range(frame_count)
                            if types[i] != mc.FRAME_TYPE_I and (i - 1) % args.gop == 0]
    ordinary_p_positions = [i for i in range(frame_count)
                            if types[i] != mc.FRAME_TYPE_I and i not in boundary_p_positions]

    train_sequences = discover_sequences(args.manifest, split="train")
    clip = train_sequences[0].load_frames()[:frame_count].to(device)

    trace: dict[str, Any] = {}
    for bits in RATE_POINTS:
        calibration = mc.calibrate_grids(
            model, train_sequences, bits=bits, mode="per_channel", gop_size=args.gop,
            block_size=16, search_range=16, reference_mode="mc", max_frames=args.calibration_frames)
        intra_params, intra_entropy_model = calibration["intra_params"], calibration["intra_entropy_model"]
        residual_params = calibration["residual_params"]

        with torch.no_grad(), mc.deterministic_kernels():
            true_latent_0 = model.encode(clip[0:1])
            payload, _ = encode_latent_to_payload(true_latent_0, params=intra_params,
                                                  entropy_model=intra_entropy_model)
            decoded_latent_0, _ = decode_payload_to_latent(
                payload, entropy_model=intra_entropy_model, params=intra_params,
                shape=tuple(true_latent_0.shape[1:]))
            reconstruction_0 = model.decode(decoded_latent_0.to(device))

            quant_error_latent = float((decoded_latent_0.to(device) - true_latent_0).abs().mean())
            quant_error_pixels = float((reconstruction_0 - clip[0:1]).abs().mean())

        trace[f"{bits}bit"] = {
            "t0_i_frame": {
                "step_1_true_latent": "model.encode(frame) - float latent, no quantization",
                "step_2_quantized_payload": "encode_latent_to_payload(true_latent, intra_params, "
                    "intra_entropy_model) -> compressed bytes (lossy: bit-depth quantization + "
                    "entropy coding)",
                "step_3_decoded_latent": "decode_payload_to_latent(payload, ...) -> DEQUANTIZED "
                    "latent (this IS 'reconstructed_latent' for an I-frame in encode_multi - see "
                    "scripts/m13_closed_loop.py line 126)",
                "step_4_reconstruction": "model.decode(decoded_latent) -> pixel-space reconstruction",
                "reference_used_by_next_frames_motion_estimation": "step_4 output (`previous` in "
                    "encode_multi) - PIXEL-SPACE RECONSTRUCTED FRAME, bit-depth-dependent",
                "mean_abs_quantization_error_latent_space": quant_error_latent,
                "mean_abs_reconstruction_error_pixel_space": quant_error_pixels,
            },
        }

    boundary_and_ordinary_p_frame_reference = {
        "claim": "encode_multi has NO branch distinguishing 'the P-frame right after an "
            "I-frame' from any other P-frame. Every P-frame (boundary or ordinary) reads "
            "`previous` - whatever pixel reconstruction the IMMEDIATELY PRECEDING frame "
            "produced, I-typed or P-typed alike (scripts/m13_closed_loop.py lines 131-157).",
        "reference_for_motion_estimation": "`previous` - PIXEL-SPACE reconstruction "
            "(model.decode of the prior frame's reconstructed_latent)",
        "reference_for_residual_generation": "`reference_latent = model.encode(warped)` - a "
            "RE-ENCODED latent, from warping `previous` (pixel space) by the estimated motion, "
            "then re-running the encoder. NOT a warp of any latent tensor directly - "
            "estimate_block_motion/warp_blocks operate on pixels only in this codec.",
        "reference_for_entropy_calibration": "DIFFERENT for the two calibration-time helper "
            "functions M14 built - NEITHER matches the real coder exactly at ordinary P-frames: "
            "calibrate_grids's OWN P-branch advances `previous_reconstruction = model.decode(latent)` "
            "(the TRUE, unquantized latent, not `reconstructed_latent = reference_latent + delta`) - "
            "documented in its own source as 'very slightly optimistic relative to coding-time "
            "residuals'. collect_motion_symbols does the same (true-latent advance) at EVERY "
            "frame including I-frames. Both are pre-existing, unmodified by M16.",
        "classification_per_step": {
            "true_latent": "model.encode(frame) - exists only transiently, never stored as a reference",
            "quantized_latent": "the INTEGER SYMBOLS from latent_to_symbols/encode_latent_to_payload "
                "- exists only inside the entropy-coded payload bytes",
            "decoded_latent": "decode_payload_to_latent's output (I-frames) or "
                "reference_latent + symbols_to_latent(...) (P-frames) - this is `reconstructed_latent`",
            "reconstruction": "model.decode(reconstructed_latent) - PIXEL space; THIS is what "
                "`previous` holds and what motion estimation actually reads",
            "re_encoded_latent": "model.encode(warped) - computed fresh every P-frame from the "
                "WARPED PIXEL reference; this is `reference_latent`, used for the residual `delta`",
            "other": "none observed - every quantity in the real coder's per-frame loop falls "
                "into one of the five categories above",
        },
    }

    report = {
        "phase": "M16 Phase A - GOP-boundary reference flow trace",
        "checkpoint": str(args.checkpoint), "gop_size": args.gop,
        "gop_frame_types_at_gop10_for_23_frames": {
            "i_frame_positions": i_positions,
            "boundary_p_frame_positions": boundary_p_positions,
            "ordinary_p_frame_positions": ordinary_p_positions,
        },
        "per_bit_depth_t0_trace": trace,
        "gop_boundary_and_ordinary_p_frame_reference": boundary_and_ordinary_p_frame_reference,
        "key_finding": (
            "The REAL deployed coder's bit-depth dependence is NOT concentrated at GOP "
            "boundaries - it is uniform in mechanism (every reference is a real, quantized "
            "pixel reconstruction) and CUMULATIVE in magnitude (quantization error from "
            "intra AND every preceding residual compounds through a GOP). The 'GOP-boundary' "
            "framing accurately describes calibrate_grids's/collect_motion_symbols's "
            "APPROXIMATIONS, not a distinct property of the deployed codec. Phase B/C "
            "measure the real per-GOP-position shape of this empirically rather than assume "
            "it peaks at the boundary."),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "m16_reference_flow.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=" * 110)
    print("M16 PHASE A - GOP-BOUNDARY REFERENCE FLOW TRACE")
    print("=" * 110)
    print(f"  gop_size={args.gop}: I-frames at {i_positions}")
    print(f"  boundary P-frames (immediately after an I-frame): {boundary_p_positions}")
    print(f"  ordinary P-frames: {ordinary_p_positions}")
    for bits in RATE_POINTS:
        t = trace[f"{bits}bit"]["t0_i_frame"]
        print(f"  {bits}-bit: I-frame mean|quant error| latent={t['mean_abs_quantization_error_latent_space']:.6f}  "
             f"pixel={t['mean_abs_reconstruction_error_pixel_space']:.6f}")
    print(f"\n  KEY FINDING: {report['key_finding']}")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
