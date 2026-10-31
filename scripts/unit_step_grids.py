"""The video path's calibration under the quantizer a checkpoint was TRAINED for.

Phase B's models (`colab_run5_phase_b`) were fine-tuned for fixed-step rounding,
and their checkpoints say so: `extra = {"quantizer": "unit_step", "unit_step": 1.0}`.
`m10h_motion_compensation.calibrate_grids` only knows the percentile grid, and
every P-frame refit script (M10K, M10L, M11, M13, M14, `benchmark_parity` through
`m21.prepare_rate_point`) gets its intra and residual params from it. This module
is the one place that decides which quantizer a checkpoint gets, so the refit
scripts cannot disagree with each other or with the model.

WHY IT WRAPS `calibrate_grids` RATHER THAN EDITING IT
----------------------------------------------------
`m10h_motion_compensation.py` is hashed into `m22_residual.code_identity()`, which
every shipped M22 refit checkpoint records. Editing it would make those records
stale. And duplicating its ~80-line closed-loop walk would let the two drift. So
`calibrate_grids` runs unmodified, with its one fitting primitive,
`calibrate_quantization_params`, swapped for the unit step on that module
instance for the duration of the call. The walk, the deterministic-kernel guard,
the entropy tables and the motion table are all its own.

ONE ALPHABET WIDTH FOR BOTH PARAMS BLOCKS
-----------------------------------------
The `.nvct` header has a single `quantization_bits` for the intra and residual
params (`nvc.video.container`). At step 1.0 the lambda 3e-4 model's intra latent
needs 8 bits and its residuals 7, so the stream is 8 bits wide and the residual
alphabet is padded: its `zero_point` still puts the lowest TRAIN symbol at 0, and
the unused top of the table holds only floor frequencies. The `bits` a caller
passes must equal the wider of the two fitted widths - anything else is refused,
so a rate point is a checkpoint and its `bits` is a fact about that checkpoint,
not a knob.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import torch

from nvc.compression.calibration import unit_step_quantization_params

PERCENTILE = "percentile"
UNIT_STEP = "unit_step"


def checkpoint_quantizer(checkpoint: str | Path) -> dict[str, Any]:
    """`{"quantizer": "unit_step", "step": s}` for a checkpoint trained for the
    unit step, `{"quantizer": "percentile"}` for every other one (run4 and the
    deployed transform carry no such field)."""
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    extra = payload.get("extra") or {}
    if extra.get("quantizer") == UNIT_STEP:
        step = float(extra.get("unit_step", 1.0))
        return {"quantizer": UNIT_STEP, "step": step}
    return {"quantizer": PERCENTILE}


def unit_step_width(latents: torch.Tensor, *, step: float) -> int:
    """The alphabet width the unit step needs for `latents` on its own."""
    return unit_step_quantization_params(latents, step=step).bits


@contextlib.contextmanager
def _unit_step_primitive(mc, *, step: float, fitted: list[int]) -> Iterator[None]:
    original = mc.calibrate_quantization_params

    def fit(latents: torch.Tensor, *, bits: int, mode: str = "per_channel", **_: Any):
        if mode != "per_channel":
            raise ValueError(f"the unit step is per-channel only, got mode={mode!r}")
        params = unit_step_quantization_params(latents, step=step, max_bits=bits)
        fitted.append(params.bits)
        return replace(params, bits=bits)

    mc.calibrate_quantization_params = fit
    try:
        yield
    finally:
        mc.calibrate_quantization_params = original


def calibrate_grids(mc, model, sequences, *, checkpoint: str | Path,
                    **kwargs: Any) -> dict[str, Any]:
    """`mc.calibrate_grids(model, sequences, **kwargs)` under the quantizer
    `checkpoint` was trained for. Percentile checkpoints get exactly the call
    they always got."""
    quantizer = checkpoint_quantizer(checkpoint)
    if quantizer["quantizer"] == PERCENTILE:
        return mc.calibrate_grids(model, sequences, **kwargs)

    bits = kwargs["bits"]
    fitted: list[int] = []
    try:
        with _unit_step_primitive(mc, step=quantizer["step"], fitted=fitted):
            calibration = mc.calibrate_grids(model, sequences, **kwargs)
    except ValueError as error:
        if "symbols at step" in str(error):
            raise ValueError(
                f"{checkpoint} is a unit-step checkpoint and needs a wider alphabet "
                f"than {bits} bits at step {quantizer['step']:g}: {error}") from error
        raise
    if max(fitted) != bits:
        raise ValueError(
            f"{checkpoint} is a unit-step checkpoint (step {quantizer['step']:g}); its "
            f"intra/residual alphabets need {fitted[0]}/{fitted[1]} bits, so the stream is "
            f"{max(fitted)} bits wide. Pass that as the rate point, not {bits}.")
    calibration["provenance"] = {
        **calibration["provenance"], "quantizer": UNIT_STEP, "unit_step": quantizer["step"],
        "intra_bits_needed": fitted[0], "residual_bits_needed": fitted[1]}
    return calibration


def cache_identity(checkpoint: str | Path) -> dict[str, str]:
    """What a calibration cache key must add for a unit-step checkpoint: this
    module's own source, since it decides the params. Empty for percentile
    checkpoints, so their existing caches keep their keys."""
    if checkpoint_quantizer(checkpoint)["quantizer"] == PERCENTILE:
        return {}
    source = Path(__file__).read_bytes()
    return {"unit_step_grids_source": hashlib.sha256(source).hexdigest()}
