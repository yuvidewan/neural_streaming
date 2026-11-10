"""P2 gate 1: how much would better motion help the learned P-frame coder?

`docs/PLAN_learned_motion.md`. The gate 3 P-coder only ever sees `x_warp`, the previous
reconstruction moved by integer 16x16 block shifts. This asks what the P-frames would cost
with a better `x_warp`, before anything is built:

    block16       the deployed motion (integer 16x16, +-16 px), motion paid for
    block8        integer 8x8 blocks, same search, motion paid for
    block8_free   8x8 blocks, motion FREE           (upper bounds: how much of the residual
    block4_free   4x4 blocks, motion FREE            is wrong motion, if motion cost nothing)
    dense_oracle  dense Farneback flow from the previous RECONSTRUCTION to x_t, bilinear
                  warp, motion FREE
    dense_clean   the same flow estimated between the ORIGINAL frames t-1 and t (better flow,
                  still warping the reconstruction), motion FREE
    zero          no motion (the floor)

Each arm runs its own closed loop: GOP 10 on the 4 VAL-B sequences, every P-frame coded
from that arm's previous reconstruction by the gate 3 P-coder for that lambda. I-frames are
the intra model's unit-step reconstructions (as in gates 1 and 2) and are identical across
arms; only P-frames are scored.

Rates are ideal code lengths. The P-coder's own conditional Laplace is within 0.01% of
real bytes (gate 3). Block motion is scored against an empirical dy/dx table fitted on that
arm's own vectors over the whole run. That is slightly optimistic, and equally so for both
block sizes.

Gate (fixed in docs/PLAN_learned_motion.md before any result): dense_oracle cuts P-frame
BD-rate (motion + residual bits vs PSNR, 3 lambdas) by at least 25% vs block16. Dense,
smooth flow is what a learned motion coder would transmit; if even free flow cannot save
that much, a coded flow field (which must pay for itself) cannot.

The free BLOCK arms are diagnostics, not the gate: a free 4x4 vector chosen by searching
x_t itself carries ~10 bits of information about x_t per 16 pixels, so its "saving" is
partly the target smuggled through the motion, not better prediction.

Usage: p2_motion_oracle.py [--models-dir outputs/p1_pframe/gate2_local/stage2]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_parity as bp  # noqa: E402
import p1_pframe as p1  # noqa: E402

from nvc.utils.config import load_default_config  # noqa: E402
from nvc.utils.device import get_device  # noqa: E402

ARMS = ("block16", "block8", "block8_free", "block4_free", "dense_oracle", "dense_clean", "zero")
BLOCK = {"block16": 16, "block8": 8, "block8_free": 8, "block4_free": 4}
PAID = ("block16", "block8")
GATE_SAVING = 25.0
OUTPUT_DIR = Path("outputs/p2_motion")
SEARCH_RANGE = 16


def empirical_bits(symbols: list[np.ndarray]) -> float:
    """Ideal total bits of all `symbols` under their own empirical distribution."""
    if not symbols:
        return 0.0
    values = np.concatenate(symbols)
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(-(counts * np.log2(p)).sum())


@torch.no_grad()
def run_arm(arm: str, mc, intra, pcoder, sequences, device, gop: int) -> dict:
    p_bits, p_mse, p_psnr, i_psnr, motion_dy, motion_dx = [], [], [], [], [], []
    with mc.deterministic_kernels():
        for sequence in sequences:
            frames = sequence.load_frames()
            previous = None
            for t in range(frames.shape[0]):
                frame = frames[t:t + 1].to(device)
                if t % gop == 0:
                    previous = intra.decode(torch.round(intra.encode(frame))).clamp(0, 1)
                    i_psnr.append(p1.psnr(previous, frame))
                    continue
                if arm in BLOCK:
                    block = BLOCK[arm]
                    motion = p1.batched_block_motion(previous, frame, block_size=block,
                                                     search_range=SEARCH_RANGE)[0]
                    warped = mc.warp_blocks(previous, motion, block_size=block)
                    if arm in PAID:
                        motion_dy.append(motion[0].flatten().cpu().numpy())
                        motion_dx.append(motion[1].flatten().cpu().numpy())
                elif arm == "dense_oracle":
                    flow = mc.estimate_dense_flow_oracle(previous, frame)
                    warped = mc.warp_dense(previous, flow).clamp(0, 1)
                elif arm == "dense_clean":
                    flow = mc.estimate_dense_flow_oracle(frames[t - 1:t].to(device), frame)
                    warped = mc.warp_dense(previous, flow).clamp(0, 1)
                else:
                    warped = previous
                result = pcoder(frame, warped)
                previous = result["reconstruction"]
                p_bits.append(float(result["bits"]))
                mse = float(torch.mean((previous - frame) ** 2))
                p_mse.append(mse)
                p_psnr.append(10 * math.log10(1 / mse))
    pixels = frames.shape[2] * frames.shape[3]
    residual = float(np.sum(p_bits))
    motion = empirical_bits(motion_dy) + empirical_bits(motion_dx)
    count = len(p_bits)
    return {"p_frames": count,
            "residual_bits_per_frame": residual / count,
            "motion_bits_per_frame": motion / count,
            "bits_per_frame": (residual + motion) / count,
            "bpp": (residual + motion) / count / pixels,
            "p_psnr": float(np.mean(p_psnr)),
            "p_pooled_psnr": 10 * math.log10(1 / float(np.mean(p_mse))),
            "i_psnr": float(np.mean(i_psnr))}


def gate(bd_rates: dict) -> dict:
    oracle = bd_rates.get("dense_oracle")
    return {"arm": "dense_oracle", "threshold_percent": -GATE_SAVING, "bd_rate": oracle,
            "passed": oracle is not None and oracle <= -GATE_SAVING}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-dir", type=Path,
                        default=p1.OUTPUT_DIR / "gate2_local" / "stage2")
    parser.add_argument("--lambdas", nargs="+", default=list(p1.LAMBDAS))
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "gate1_oracle.json")
    args = parser.parse_args(argv)
    device = get_device()
    mc = p1._load_script("m10h_motion_compensation")
    m21 = p1._load_script("m21_refinement")
    g3 = p1._load_script("p1_gate3")
    manifest = load_default_config().processed_data_dir / "manifest.json"
    sequences = m21.val_b_sequences(manifest, count=args.sequences)
    report = {"models_dir": str(args.models_dir), "gop": args.gop,
              "val_b": [s.sequence_id for s in sequences], "points": {}}
    for lam in args.lambdas:
        intra = p1._intra_model(lam, device)
        pcoder = g3.load_pcoder(args.models_dir, lam, device)
        report["points"][lam] = {}
        for arm in args.arms:
            point = run_arm(arm, mc, intra, pcoder, sequences, device, args.gop)
            report["points"][lam][arm] = point
            print(f"  [{lam}] {arm:13s} {point['bits_per_frame']:>9.0f} bits/P "
                  f"(motion {point['motion_bits_per_frame']:>6.0f})  "
                  f"P PSNR {point['p_psnr']:.3f}", flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def curve(arm):
        return sorted((report["points"][lam][arm]["bpp"], report["points"][lam][arm]["p_psnr"])
                      for lam in args.lambdas)

    report["bd_rate_vs_block16"] = {arm: bp.bd_rate(curve("block16"), curve(arm))
                                    for arm in args.arms if arm != "block16"}
    report["gate"] = gate(report["bd_rate_vs_block16"])
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nP-frame BD-rate vs block16 (negative = fewer bits):")
    for arm, value in report["bd_rate_vs_block16"].items():
        print(f"  {arm:13s} {'n/a' if value is None else f'{value:+.1f}%'}")
    print(f"GATE 1 ({'PASSED' if report['gate']['passed'] else 'NOT PASSED'}): dense_oracle "
          f"<= -{GATE_SAVING:.0f}%\nReport: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
