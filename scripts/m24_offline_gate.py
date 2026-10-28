"""M24 Phase A - how much of the CompressAI gap is the quantizer, at frozen weights?

The anchor measured `bmshj2018-factorized` at about HALF our all-intra bits, on a
design that differs from ours in the quantizer, the density model and
rate-aware training, all at once. Those need different work: the first two are
coding changes, the third needs Vimeo and a GPU. This gate separates them on the
latents the current transform already produces, so the project knows which it is
buying before paying for either.

See `scripts/m24_quantizer.py` for the three arms and why they are the right
three. In short: `grid_histogram` is our codec, `grid_laplace` changes only the
density, `unit_step_laplace` changes only the grid rule.

WHAT A PASS WOULD MEAN
----------------------
A large `grid_laplace` -> `unit_step_laplace` BD-rate is a coding win available
WITHOUT retraining, and it would make the quantizer the next milestone. A small
one says the gap lives in rate-aware training instead, which is Phase B and a
different kind of work. Either outcome is decision-relevant, which is why this
is worth measuring before committing GPU time.

TRAIN fits every grid and every density. VAL-B reports. TEST is never opened -
the anchor's own numbers are on TEST, and this gate is not entitled to a second
read of it.

Run:
  ./.venv/Scripts/python.exe scripts/m24_offline_gate.py
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


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    m23 = _load_script("m23_hyperprior")
    m24 = _load_script("m24_quantizer")
    parser = argparse.ArgumentParser(
        description="M24 Phase A: is the CompressAI gap the quantizer or the training?",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=m23.DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m24_quantizer"))
    parser.add_argument("--output-name", type=str, default="m24_offline_gate.json")
    parser.add_argument("--arms", nargs="+", choices=list(m24.ARMS), default=list(m24.ARMS))
    parser.add_argument("--grid-bits", type=int, nargs="+",
                        default=list(m24.DEFAULT_GRID_BITS))
    parser.add_argument("--step-multipliers", type=float, nargs="+",
                        default=list(m24.DEFAULT_STEP_MULTIPLIERS))
    parser.add_argument("--train-frames", type=int, default=400)
    parser.add_argument("--val-frames-per-sequence", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


@torch.no_grad()
def collect_latents(model, sequences, *, max_frames: int, device,
                    batch: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Latents and the frames they came from, so distortion can be scored in pixels."""
    model.eval()
    latents, frames_out = [], []
    seen = 0
    for sequence in sequences:
        frames = sequence.load_frames()
        for start in range(0, frames.shape[0], batch):
            if seen >= max_frames:
                break
            chunk = frames[start:start + min(batch, max_frames - seen)]
            latents.append(model.encode(chunk.to(device)).detach().cpu())
            frames_out.append(chunk)
            seen += chunk.shape[0]
        if seen >= max_frames:
            break
    return torch.cat(latents, dim=0), torch.cat(frames_out, dim=0)


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    m24 = _load_script("m24_quantizer")
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
    # VAL-B is the odd-indexed half, the same split M13-M23 reported on.
    val_sequences = m21.val_b_sequences(args.manifest, count=4,
                                        max_frames=args.val_frames_per_sequence)

    print("=" * 108)
    print("M24 PHASE A - IS THE COMPRESSAI GAP THE QUANTIZER OR THE TRAINING?")
    print(f"  transform: {args.checkpoint}")
    print(f"  arms: {args.arms}")
    print(f"  VAL-B: {[s.sequence_id for s in val_sequences]}")
    print("=" * 108, flush=True)

    started = time.perf_counter()
    train_latents, _ = collect_latents(model, train_sequences, max_frames=args.train_frames,
                                       device=device, batch=args.batch)
    val_latents, val_frames = collect_latents(model, val_sequences, max_frames=10 ** 9,
                                              device=device, batch=args.batch)
    print(f"  TRAIN latents {tuple(train_latents.shape)}, VAL-B {tuple(val_latents.shape)} "
          f"in {time.perf_counter() - started:.0f}s", flush=True)

    reference = m24.reference_step(train_latents, bits=4)
    print(f"  reference step (median deployed 4-bit step): {reference:.5f}", flush=True)

    report: dict[str, Any] = {
        "phase": "M24 Phase A - the quantizer at frozen weights",
        "question": ("the CompressAI anchor found bmshj2018-factorized at ~half our "
                     "all-intra bits, differing in the quantizer, the density and "
                     "rate-aware training at once. How much is reachable WITHOUT "
                     "retraining?"),
        "arms": {
            "grid_histogram": "our intra codec: percentile grid, fixed depth, "
                              "per-channel histogram",
            "grid_laplace": "the same grid with a discretised Laplace - isolates the "
                            "DENSITY",
            "unit_step_laplace": "rounding at a swept step - isolates the GRID RULE",
        },
        "cannot_answer": ("how much rate-aware training would add on top; that is Phase B "
                          "and needs Vimeo and a GPU"),
        "split_discipline": "TRAIN fits grids and densities, VAL-B reports, TEST never opened",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {k: str(v) if isinstance(v, Path) else v
                          for k, v in vars(args).items()},
        "reference_step": reference,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": str(device), "platform": platform.platform()},
        "train_frames": int(train_latents.shape[0]),
        "val_b_frames": int(val_latents.shape[0]),
        "val_b_sequences": [s.sequence_id for s in val_sequences],
        "points": [],
    }

    def persist() -> None:
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    for arm in args.arms:
        settings = (args.grid_bits if arm.startswith("grid")
                    else [m * reference for m in args.step_multipliers])
        print(f"\n  {arm}", flush=True)
        for setting in settings:
            point = m24.measure_point(model, bp, arm, train_latents=train_latents,
                                      val_latents=val_latents, val_frames=val_frames,
                                      setting=setting, device=device, batch=args.batch)
            report["points"].append(point)
            persist()
            label = (f"{int(setting)}-bit" if arm.startswith("grid")
                     else f"step {setting:.5f}")
            detail = (f"edge {100 * point['at_alphabet_edge_fraction']:.2f}%"
                      if arm.startswith("grid")
                      else f"span +-{point['symbol_span']}, "
                           f"{point['effective_bits']:.2f} eff bits")
            print(f"    {label:<18} {point['bpp']:.5f} bpp  {point['psnr']:.3f} dB  "
                  f"{point['bits_per_symbol']:.4f} b/sym  {detail}", flush=True)

    # ---- BD-rate between the arms, on the curves just measured -------------------
    curves = {}
    for arm in args.arms:
        points = [p for p in report["points"] if p["arm"] == arm]
        curves[arm] = sorted((p["bpp"], p["psnr"]) for p in points)
    comparisons = {}
    for anchor in args.arms:
        for test in args.arms:
            if anchor == test:
                continue
            comparisons[f"{test}_vs_{anchor}"] = {
                "bd_rate_percent": bp.bd_rate(curves[anchor], curves[test]),
                "quality_overlap": bp.overlap(curves[anchor], curves[test]),
            }
    report["curves"] = {k: [list(p) for p in v] for k, v in curves.items()}
    report["comparisons"] = comparisons
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    persist()

    print("\n" + "=" * 108)
    print("BD-RATE BETWEEN THE ARMS (negative = fewer bits at equal quality)")
    print("=" * 108)
    for key, value in comparisons.items():
        bd = value["bd_rate_percent"]
        print(f"  {key:44s} {'n/a' if bd is None else f'{bd:+.2f}%'}")
    if ("unit_step_laplace" in args.arms and "grid_laplace" in args.arms
            and "grid_histogram" in args.arms):
        density = comparisons["grid_laplace_vs_grid_histogram"]["bd_rate_percent"]
        grid = comparisons["unit_step_laplace_vs_grid_laplace"]["bd_rate_percent"]
        whole = comparisons["unit_step_laplace_vs_grid_histogram"]["bd_rate_percent"]
        print("\nDECOMPOSITION, at frozen weights:")
        print(f"  density model (histogram -> Laplace):     "
              f"{'n/a' if density is None else f'{density:+.2f}%'}")
        print(f"  grid rule (percentile depth -> unit step): "
              f"{'n/a' if grid is None else f'{grid:+.2f}%'}")
        print(f"  both together:                            "
              f"{'n/a' if whole is None else f'{whole:+.2f}%'}")
        print("\n  The anchor's +90.9% is this plus whatever rate-aware training adds.")
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
