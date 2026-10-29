"""M24 Phase A2 - does the RESIDUAL quantizer have the same ceiling?

Phase A measured the INTRA quantizer and found that the percentile grid caps
reconstruction near 32.6 dB whatever the bit depth, because it clips the tails.
Intra is ~12% of the video stream. The residual quantizer is the SAME scheme -
`calibrate_quantization_params` at percentiles (0.1, 99.9), per channel, fixed
depth - applied to the P-frame residuals, which are **79.6-84.2% of the stream**.

If it caps too, the ceiling is the binding constraint on the whole codec rather
than on one twelfth of it.

WHY THIS IS WORTH SUSPECTING, AND WHAT IT WOULD REINTERPRET
-----------------------------------------------------------
Five milestones probed the P-residual path and came back negative:

  M17  +2.0/+6.7/+14.9% oracle gap - real, meaningful, "unimplementable"
  M18  intra recalibration does not close it; "aggregate MSE improvement does
       not transfer to byte savings"
  M19  residual-symbol-change is 88-91% of the excess bits
  M20  codebook hysteresis rejected at +0.0007%
  M21  "the reference is a local optimum at 5/4-bit"

A clipping ceiling in the residual quantizer is a plausible COMMON CAUSE that
none of them measured. M18's finding in particular - better aggregate MSE that
does not become bytes - is what clipping looks like from the outside. And the one
milestone on that path that did ship a win, M22, changed the GRID (re-centering
it), the only one of the five to touch the grid at all.

This phase does not prove that reinterpretation. It tests its precondition: does
the residual grid saturate with depth the way the intra grid does?

THE WALK, AND WHY IT IS VERIFIED RATHER THAN TRUSTED
----------------------------------------------------
Distortion has to be scored in pixels, which needs the REFERENCE latent for each
residual, and `m22_residual.collect_train_residuals` does not return references.
Its docstring records that two details of this walk are load-bearing and that one
of them was gotten wrong once before `verify_deployed_grid` caught it: the
I-frame reference is the intra ROUND TRIP, not `decode(latent)`, and the
reference then advances from the TRUE latent.

So this module walks the loop itself to keep references, and then checks its
residual stack against `m22_residual.collect_train_residuals` element by element.
If they differ at all, the run stops. A reimplemented walk that silently
disagreed would shift every grid fitted below it.

ARMS
----
The same three as Phase A, on residuals instead of latents: `grid_histogram`
(the deployed residual coder), `grid_laplace` (density isolated),
`unit_step_laplace` (grid rule isolated).

Quality is decoded-pixel PSNR of `reference + dequantized residual`, scored by
`benchmark_parity`'s scorer against the original frame, so these numbers sit on
the project's usual axis. TRAIN fits, VAL-B reports, TEST never opened.
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

from nvc.compression.codec import decode_payload_to_latent, encode_latent_to_payload
from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def walk_with_references(mc, model, sequences, *, intra_params, intra_entropy_model,
                         gop_size: int, block_size: int, search_range: int,
                         max_frames: int, device, keep_frames: bool) -> dict[str, Any]:
    """`m22_residual.collect_train_residuals`' walk, also keeping the references.

    Deliberately line-for-line the same arithmetic, because its two load-bearing
    details are exactly where a reimplementation goes wrong: the I-frame
    reference is the intra ROUND TRIP of the latent, and the reference advances
    from the TRUE latent afterwards. `main` checks the residual stack this
    produces against m22's own and refuses to continue on any difference.
    """
    model.eval()
    intra_latents = []
    seen = 0
    for sequence in sequences:
        frames = sequence.load_frames()
        for index in range(frames.shape[0]):
            if seen >= max_frames:
                break
            intra_latents.append(model.encode(frames[index:index + 1].to(device)).cpu())
            seen += 1
        if seen >= max_frames:
            break
    latent_shape = tuple(torch.cat(intra_latents, dim=0).shape[1:])

    residuals, references, originals = [], [], []
    seen = 0
    with mc.deterministic_kernels():
        for sequence in sequences:
            frames = sequence.load_frames()
            types = mc.gop_frame_types(frames.shape[0], gop_size)
            previous = None
            for index in range(frames.shape[0]):
                if seen >= max_frames:
                    break
                frame = frames[index:index + 1].to(device)
                latent = model.encode(frame)
                if types[index] == mc.FRAME_TYPE_I:
                    payload, _ = encode_latent_to_payload(
                        latent, params=intra_params, entropy_model=intra_entropy_model)
                    decoded, _ = decode_payload_to_latent(
                        payload, entropy_model=intra_entropy_model, params=intra_params,
                        shape=latent_shape)
                    previous = model.decode(decoded.to(device))
                elif previous is not None:
                    motion = mc.estimate_block_motion(previous, frame,
                                                      block_size=block_size,
                                                      search_range=search_range)
                    warped = mc.warp_blocks(previous, motion, block_size=block_size)
                    reference = model.encode(warped)
                    residuals.append((latent - reference).cpu())
                    references.append(reference.cpu())
                    if keep_frames:
                        originals.append(frames[index:index + 1].cpu())
                    previous = model.decode(latent)
                seen += 1
            if seen >= max_frames:
                break
    return {
        "residual_stack": torch.cat(residuals, dim=0),
        "reference_stack": torch.cat(references, dim=0),
        "frames": torch.cat(originals, dim=0) if keep_frames else None,
        "latent_shape": latent_shape,
    }


@torch.no_grad()
def measure_residual_point(m24, model, bp, arm: str, *, train_residuals: torch.Tensor,
                           val_residuals: torch.Tensor, val_references: torch.Tensor,
                           val_frames: torch.Tensor, setting: float, device,
                           batch: int = 8) -> dict[str, Any]:
    """One arm at one operating point, on residuals.

    The reconstructed latent is `reference + dequantized residual`, so the
    reference cancels out of the latent-domain error and what is measured is the
    residual quantizer's own damage - but it is scored in PIXELS, through the
    decoder, against the original frame, so it shares the project's axis.
    """
    channels = train_residuals.shape[1]
    shape = tuple(train_residuals.shape[1:])

    if arm.startswith("grid"):
        bits = int(setting)
        params = m24.deployed_grid(train_residuals, bits=bits)
        train_symbols = m24.grid_symbols(train_residuals, params)
        val_symbols = m24.grid_symbols(val_residuals, params)
        dequantized = m24.grid_reconstruct(val_symbols, params, shape)
        alphabet = 2 ** bits
        if arm == "grid_histogram":
            probabilities = m24.fit_histogram(train_symbols, alphabet=alphabet,
                                              channels=channels)
            total_bits = m24.histogram_bits(val_symbols, probabilities, channels=channels)
        else:
            location, scale = m24.fit_laplace(train_symbols, channels=channels)
            total_bits = m24.discretised_laplace_bits(val_symbols, location, scale,
                                                      channels=channels)
        extra = {"bits": bits, "alphabet": alphabet,
                 "step_median": float(params.scale.flatten().median()),
                 "at_alphabet_edge_fraction":
                     float(np.mean((val_symbols == 0) | (val_symbols == alphabet - 1)))}
    else:
        step = float(setting)
        train_symbols = m24.unit_step_symbols(train_residuals, step=step)
        val_symbols = m24.unit_step_symbols(val_residuals, step=step)
        dequantized = m24.unit_step_reconstruct(val_symbols, step=step, shape=shape)
        location, scale = m24.fit_laplace(train_symbols, channels=channels)
        total_bits = m24.discretised_laplace_bits(val_symbols, location, scale,
                                                  channels=channels)
        span = int(np.abs(val_symbols).max())
        extra = {"step": step, "symbol_span": span,
                 "distinct_symbols": int(np.unique(val_symbols).size)}

    latent_mse = float(((dequantized - val_residuals) ** 2).mean())
    decoded = []
    for start in range(0, val_residuals.shape[0], batch):
        chunk = (val_references[start:start + batch]
                 + dequantized[start:start + batch]).to(device)
        decoded.append(model.decode(chunk).detach().cpu())
    decoded = torch.cat(decoded, dim=0)
    scores = bp.score_frames(bp.to_uint8_frames(decoded), val_frames, device=device)

    pixels = int(val_frames.shape[0] * val_frames.shape[2] * val_frames.shape[3])
    return {
        "arm": arm, "setting": setting,
        "total_ideal_bits": total_bits,
        "bpp": total_bits / pixels,
        "bits_per_symbol": total_bits / val_symbols.size,
        "latent_residual_mse": latent_mse,
        "psnr": float(np.mean(scores["psnr"])),
        "msssim": float(np.mean(scores["msssim"])),
        "symbols": int(val_symbols.size), "pixels": pixels, **extra,
    }


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    m23 = _load_script("m23_hyperprior")
    m24 = _load_script("m24_quantizer")
    parser = argparse.ArgumentParser(
        description="M24 Phase A2: does the residual quantizer cap quality too?",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=m23.DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m24_quantizer"))
    parser.add_argument("--output-name", type=str, default="m24_residual_gate.json")
    parser.add_argument("--arms", nargs="+", choices=list(m24.ARMS), default=list(m24.ARMS))
    parser.add_argument("--grid-bits", type=int, nargs="+",
                        default=list(m24.DEFAULT_GRID_BITS))
    parser.add_argument("--step-multipliers", type=float, nargs="+",
                        default=list(m24.DEFAULT_STEP_MULTIPLIERS))
    parser.add_argument("--calibration-bits", type=int, default=4,
                        help="the depth whose deployed grid sets the reference step")
    parser.add_argument("--train-frames", type=int, default=400)
    parser.add_argument("--val-frames-per-sequence", type=int, default=20)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--skip-walk-check", action="store_true",
                        help="smoke runs only; the real run must verify the walk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    m24 = _load_script("m24_quantizer")
    m22 = _load_script("m22_residual")
    mc = _load_script("m10h_motion_compensation")
    m21 = _load_script("m21_refinement")
    bp = _load_script("benchmark_parity")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / args.output_name

    train_sequences = discover_sequences(args.manifest, split="train")
    val_sequences = m21.val_b_sequences(args.manifest, count=4,
                                        max_frames=args.val_frames_per_sequence)

    print("=" * 108)
    print("M24 PHASE A2 - DOES THE RESIDUAL QUANTIZER CAP QUALITY TOO?")
    print(f"  the residual path is 79.6-84.2% of the stream; intra was ~12%")
    print(f"  VAL-B: {[s.sequence_id for s in val_sequences]}")
    print("=" * 108, flush=True)

    # The intra quantizer is inside the walk, so it has to be the deployed one.
    started = time.perf_counter()
    calibration = mc.calibrate_grids(
        model, train_sequences, bits=args.calibration_bits, mode="per_channel",
        gop_size=args.gop, block_size=args.block_size, search_range=args.search_range,
        reference_mode="mc", max_frames=args.train_frames)
    print(f"  calibrated the deployed grids in {time.perf_counter() - started:.0f}s",
          flush=True)

    walk = dict(intra_params=calibration["intra_params"],
                intra_entropy_model=calibration["intra_entropy_model"],
                gop_size=args.gop, block_size=args.block_size,
                search_range=args.search_range, device=device)
    started = time.perf_counter()
    train = walk_with_references(mc, model, train_sequences, max_frames=args.train_frames,
                                 keep_frames=False, **walk)
    val = walk_with_references(mc, model, val_sequences, max_frames=10 ** 9,
                               keep_frames=True, **walk)
    print(f"  TRAIN residuals {tuple(train['residual_stack'].shape)}, "
          f"VAL-B {tuple(val['residual_stack'].shape)} in "
          f"{time.perf_counter() - started:.0f}s", flush=True)

    # --- the identity control: this walk must reproduce m22's exactly -----------
    walk_check: dict[str, Any] = {"skipped": bool(args.skip_walk_check)}
    if not args.skip_walk_check:
        started = time.perf_counter()
        theirs = m22.collect_train_residuals(
            mc, model, train_sequences, bits=args.calibration_bits,
            intra_params=calibration["intra_params"],
            intra_entropy_model=calibration["intra_entropy_model"],
            gop_size=args.gop, block_size=args.block_size,
            search_range=args.search_range, max_frames=args.train_frames, device=device,
            cache=None)
        same_shape = theirs["residual_stack"].shape == train["residual_stack"].shape
        identical = same_shape and torch.equal(theirs["residual_stack"],
                                               train["residual_stack"])
        walk_check = {"skipped": False, "m22_shape": list(theirs["residual_stack"].shape),
                      "ours_shape": list(train["residual_stack"].shape),
                      "identical": bool(identical),
                      "max_abs_difference": (float((theirs["residual_stack"]
                                                    - train["residual_stack"]).abs().max())
                                             if same_shape else None),
                      "seconds": time.perf_counter() - started}
        print(f"  walk check vs m22_residual.collect_train_residuals: "
              f"{'IDENTICAL' if identical else 'DIFFERENT'}", flush=True)
        if not identical:
            print("[ERROR] this walk does not reproduce m22's residual stack; every grid "
                  "fitted below it would be fitted on the wrong data.", file=sys.stderr)
            report_path.write_text(json.dumps({"walk_check": walk_check}, indent=2),
                                   encoding="utf-8")
            return 1
        # and the grid rebuilt from it must be the deployed one
        m22.verify_deployed_grid(train["residual_stack"],
                                 calibration["residual_params"], bits=args.calibration_bits)
        print("  verify_deployed_grid: the rebuilt grid IS the deployed one", flush=True)

    reference = float(m24.deployed_grid(
        train["residual_stack"], bits=args.calibration_bits).scale.flatten().median())
    print(f"  reference step (median deployed {args.calibration_bits}-bit residual step): "
          f"{reference:.5f}", flush=True)

    report: dict[str, Any] = {
        "phase": "M24 Phase A2 - the residual quantizer at frozen weights",
        "question": ("Phase A found the INTRA percentile grid caps quality near 32.6 dB "
                     "whatever the depth. The residual path uses the same scheme and is "
                     "79.6-84.2% of the stream. Does it cap too?"),
        "would_reinterpret": ("M17-M21 probed this path and came back negative five times; "
                              "a clipping ceiling is a plausible common cause none of them "
                              "measured, and M18's 'aggregate MSE does not transfer to "
                              "bytes' is what clipping looks like from outside"),
        "walk_check": walk_check,
        "split_discipline": "TRAIN fits grids and densities, VAL-B reports, TEST never opened",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {k: str(v) if isinstance(v, Path) else v
                          for k, v in vars(args).items()},
        "reference_step": reference,
        "train_p_frames": int(train["residual_stack"].shape[0]),
        "val_b_p_frames": int(val["residual_stack"].shape[0]),
        "val_b_sequences": [s.sequence_id for s in val_sequences],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": str(device), "platform": platform.platform()},
        "points": [],
    }

    def persist() -> None:
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    for arm in args.arms:
        settings = (args.grid_bits if arm.startswith("grid")
                    else [m * reference for m in args.step_multipliers])
        print(f"\n  {arm}", flush=True)
        for setting in settings:
            point = measure_residual_point(
                m24, model, bp, arm, train_residuals=train["residual_stack"],
                val_residuals=val["residual_stack"],
                val_references=val["reference_stack"], val_frames=val["frames"],
                setting=setting, device=device, batch=args.batch)
            report["points"].append(point)
            persist()
            label = (f"{int(setting)}-bit" if arm.startswith("grid")
                     else f"step {setting:.5f}")
            detail = (f"edge {100 * point['at_alphabet_edge_fraction']:.2f}%"
                      if arm.startswith("grid") else f"span +-{point['symbol_span']}")
            print(f"    {label:<18} {point['bpp']:.5f} bpp  {point['psnr']:.3f} dB  "
                  f"latent MSE {point['latent_residual_mse']:.6f}  {detail}", flush=True)

    curves = {arm: sorted((p["bpp"], p["psnr"]) for p in report["points"]
                          if p["arm"] == arm) for arm in args.arms}
    comparisons = {}
    for anchor in args.arms:
        for test in args.arms:
            if anchor != test:
                comparisons[f"{test}_vs_{anchor}"] = {
                    "bd_rate_percent": bp.bd_rate(curves[anchor], curves[test]),
                    "quality_overlap": bp.overlap(curves[anchor], curves[test])}
    report["curves"] = {k: [list(p) for p in v] for k, v in curves.items()}
    report["comparisons"] = comparisons
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    persist()

    print("\n" + "=" * 108)
    print("DOES THE RESIDUAL GRID SATURATE? (the ceiling test)")
    print("=" * 108)
    grid = sorted((p["bits"], p["psnr"], p["bpp"], p["latent_residual_mse"])
                  for p in report["points"] if p["arm"] == "grid_histogram")
    previous = None
    for bits, psnr, bpp, mse in grid:
        gain = "-" if previous is None else f"{psnr - previous:+.2f}"
        print(f"  {bits}-bit  {psnr:.3f} dB  ({gain:>6} over the previous depth)  "
              f"{bpp:.5f} bpp  latent MSE {mse:.6f}")
        previous = psnr
    print()
    for key, value in comparisons.items():
        bd = value["bd_rate_percent"]
        print(f"  {key:44s} {'n/a' if bd is None else f'{bd:+.2f}%'}")
    if {"grid_histogram", "grid_laplace", "unit_step_laplace"} <= set(args.arms):
        density = comparisons["grid_laplace_vs_grid_histogram"]["bd_rate_percent"]
        rule = comparisons["unit_step_laplace_vs_grid_laplace"]["bd_rate_percent"]
        print(f"\n  density model: {density:+.2f}%     grid rule: {rule:+.2f}%")
        print("  Phase A's intra figures, for comparison: +0.37% and -8.30%")
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
