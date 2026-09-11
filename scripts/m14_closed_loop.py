"""M14 Phase D - a generalized `run_sequences`, reusing
`m13_closed_loop.encode_multi`/`decode_sequence`/`aggregate` UNMODIFIED.

`m13_closed_loop.run_sequences` hardcodes the module-level `ARMS = ("m11_op",
"m13_recal")` tuple throughout its body (7 uses) instead of deriving arm
names from the `arms` dict it's given - fine for M13, which only ever
compared exactly those two residual arms, but wrong for M14: the axis of
variation here is `intra_entropy_model`/`motion_entropy_model` (already
plain keyword parameters `encode_multi`/`decode_sequence` accept
unmodified), with a SINGLE, FROZEN residual arm (M13's `m13_recal`) passed
through every call. `run_sequences_for_arms` below is the same function with
that one hardcoding generalized to `sorted(arms)`; everything else -
including the actual encode/decode logic - is `m13_closed_loop`'s, imported
and called, not copied.
"""

from __future__ import annotations

import importlib.util
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.evaluation.perceptual_metrics import msssim


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_sequences_for_arms(mc, ma, m13, model, sequences, arms, stream_dir, *, intra_params,
                           intra_entropy_model, residual_params, motion_entropy_model, bits,
                           gop_size, block_size, search_range) -> dict[str, Any]:
    """Identical to `m13_closed_loop.run_sequences`, with arm names taken
    from `arms` (works for any number of arms, including one) instead of
    the hardcoded `("m11_op", "m13_recal")` tuple. `encode_multi`/
    `decode_sequence`/`aggregate` are `m13_closed_loop`'s own, unmodified."""
    cl = _load_script("m13_closed_loop")
    arm_names = sorted(arms)

    per_sequence_rows: list[dict[str, Any]] = []
    arm_rows = {arm: [] for arm in arm_names}
    decode_accumulator = {arm: {} for arm in arm_names}
    invariants = {"symbols": True, "reconstruction": True, "motion": True, "metrics": True}

    for sequence in sequences:
        frames = sequence.load_frames()
        paths = {arm: stream_dir / f"{arm}_{bits}bit_{sequence.sequence_id}.nvct"
                 for arm in arm_names}
        result = cl.encode_multi(
            mc, ma, m13, model, frames, arms, paths, intra_params=intra_params,
            intra_entropy_model=intra_entropy_model, residual_params=residual_params,
            motion_entropy_model=motion_entropy_model, bits=bits, gop_size=gop_size,
            block_size=block_size, search_range=search_range)
        recon = result["reconstructions"]
        mse = torch.mean((recon - frames) ** 2).item()
        psnr = 10.0 * math.log10(1.0 / mse)
        quality = float(msssim(recon.clamp(0, 1), frames).mean())
        invariants["motion"] &= len({result["arms"][a]["motion_bytes"] for a in arm_names}) == 1
        record = {"sequence": sequence.sequence_id, "bits": bits}
        for arm in arm_names:
            stats = result["arms"][arm]
            decoded, decoded_symbols, timing = cl.decode_sequence(
                mc, ma, m13, model, paths[arm], arm, arms[arm],
                intra_entropy_model=intra_entropy_model, motion_entropy_model=motion_entropy_model,
                bits=bits)
            for key, value in timing.items():
                decode_accumulator[arm][key] = decode_accumulator[arm].get(key, 0.0) + value
            invariants["symbols"] &= all(
                np.array_equal(a.reshape(-1), b.reshape(-1))
                for a, b in zip(result["symbols"], decoded_symbols))
            invariants["reconstruction"] &= torch.equal(decoded.cpu(), recon)
            arm_rows[arm].append({"sequence": sequence.sequence_id, **stats,
                                  "total_pixels": sequence.total_pixels,
                                  "raw_rgb_bytes": sequence.raw_rgb_bytes(),
                                  "mean_psnr_db": psnr, "mean_msssim": quality})
            record[f"{arm}_residual_bytes"] = stats["residual_bytes"]
            record[f"{arm}_total_bytes"] = stats["container_bytes"]
        per_sequence_rows.append(record)

    metrics = {(round(statistics.fmean(r["mean_psnr_db"] for r in arm_rows[a]), 9),
               round(statistics.fmean(r["mean_msssim"] for r in arm_rows[a]), 9))
              for a in arm_names}
    invariants["metrics"] = len(metrics) == 1

    results = {arm: cl.aggregate(arm, bits, arm_rows[arm]) for arm in arm_names}
    decode_times = {arm: decode_accumulator[arm] for arm in arm_names}
    return {"results": results, "decode_times": decode_times, "invariants": invariants,
           "per_sequence": per_sequence_rows}
