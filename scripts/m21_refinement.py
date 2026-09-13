"""M21 - causal reference refinement: the candidate family, and a closed loop
that applies it identically at encoder and decoder.

`src/nvc/` is never touched; no M10-M20 script is modified; `.nvct` v2 is
unchanged and no field is added to it.

WHERE THE REFINEMENT GOES, AND WHY THERE
----------------------------------------
The deployed P-frame path (traced in full by `m21_baseline.py`, not assumed from
earlier reports) is:

    previous  = model.decode(reconstructed_latent)        [1,3,H,W] in (0,1)
    motion    = estimate_block_motion(previous, frame)     ENCODER ONLY
    motion    = decode_motion_payload(encode_motion_payload(motion))
    warped    = warp_blocks(previous, motion)             [1,3,H,W]
    reference = model.encode(warped)                      [1,64,16,16]
    delta     = latent - reference                         ENCODER ONLY
    symbols   = latent_to_symbols(delta, residual_params)
    ... M11-G16 -> K=512 assignment -> M13 tables -> range coder ...
    reconstructed_latent = reference + symbols_to_latent(symbols)

Exactly two points in that chain hold a tensor both sides possess *before* the
current frame's symbols exist:

  PIXEL   `previous`, the decoded previous reconstruction. It is upstream of
          BOTH motion estimation and warping, so refining it also re-aims the
          motion search - which is legitimate: the decoder never estimates
          motion, it reads the coded vectors, and it needs the refined
          `previous` only to warp.
  LATENT  `reference`, the encoded warped reference. Downstream of motion, so
          motion vectors and motion bytes are untouched.

A refinement at either point sees `previous` (and the already-decoded motion
field) and nothing else. It cannot see the current source frame, the current
target latent, the current residual, future frames, or the pre-coding motion
vectors - `refine()` is handed one tensor and the frozen model, and
`tests/test_m21_reference_refinement.py` pins that by signature.

WHAT MOVES AND WHAT DOES NOT
----------------------------
Unlike M20 (which re-routed a fixed symbol and left the reconstruction
bit-identical), refining the reference changes `delta`, hence the symbols,
hence `reconstructed_latent = reference + dequant(symbols)`. So M21 moves along
a rate/distortion curve and BOTH ends must be reported: a candidate that buys
bytes by degrading PSNR/MS-SSIM is not a compression win. The gate is actual
coded bytes, read together with the frozen distortion metrics.

PRE-REGISTRATION
----------------
`CANDIDATES` below is frozen in this module - declared before any VAL-B coded
result was inspected - and every entry carries a closed-form deterministic
definition. `m21_sweep.py` may only run what is listed here.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from nvc.compression.codec import (
    decode_payload_to_latent, encode_latent_to_payload, latent_to_symbols, symbols_to_latent,
)
from nvc.evaluation.perceptual_metrics import msssim

DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M11_DIR = Path("outputs/m11_autoregressive_entropy")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
DEFAULT_OUTPUT_DIR = Path("outputs/m21_reference_refinement")
RATE_POINTS = (5, 4, 3)
M11_G16_GROUP_SIZE = 16
DEPLOYED_MOTION_IDENTITY = "d7e7b237b6451885"   # M14's recalibrated table, all bit depths

WEAK_BELOW_PERCENT = 0.5
MEANINGFUL_ABOVE_PERCENT = 1.0

# --- Phase 5/7 protocol, declared before any VAL-B coded result was read -------
#
# The closed-loop sweep is the expensive measurement (one full encode + one full
# decode per candidate per bit depth over all 246 VAL-B P-frames), so it runs in
# two stages with a FIXED admission threshold rather than a "keep the best"
# rule - which would be selection on the held-out set:
#
#   Stage 1  every pre-registered candidate at SCREEN_BITS.
#   Stage 2  the identity control plus every candidate whose Stage-1
#            total-stream effect is at least STAGE2_ADMISSION_PERCENT, re-run at
#            the remaining bit depths with full decode verification.
#
# The threshold is deliberately loose (a candidate only has to be "not clearly
# harmful") so that a candidate which is bit-depth dependent cannot be screened
# out by a single rate point.
#
# FLOOR (added after Phase 2's OPEN-LOOP diagnostic, before any closed-loop
# Stage-1 result existed): if the admission threshold happens to admit nothing,
# Stage 2 would compare the identity control against itself and report nothing.
# So the best STAGE2_MINIMUM_CANDIDATES by Stage-1 ranking are carried forward
# regardless, clearly labelled as a robustness check rather than a selection -
# this only ever WIDENS what gets measured at the other rate points, and it
# cannot turn a harmful candidate into a passing one, because the gate is still
# the absolute 0.5% total-stream threshold.
SCREEN_BITS = 4
STAGE2_ADMISSION_PERCENT = -0.25
STAGE2_MINIMUM_CANDIDATES = 3
CONFIRM_BITS = (5, 3)


def verdict(gain_percent: float) -> str:
    """The project's frozen total-stream thresholds, reused verbatim."""
    return ("meaningful" if gain_percent >= MEANINGFUL_ABOVE_PERCENT else
            "marginal" if gain_percent >= WEAK_BELOW_PERCENT else "weak")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the deterministic kernels the candidates are built from --------------------


def _reflect_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Replicate padding, matching `warp_blocks`' own edge convention so the
    refinement does not invent a different border behaviour."""
    return F.pad(x, (pad,) * 4, mode="replicate")


def median3(x: torch.Tensor) -> torch.Tensor:
    """Per-channel 3x3 median. Deterministic: `torch.median` on an odd window
    returns the exact middle order statistic, with no tie ambiguity."""
    patches = F.unfold(_reflect_pad(x, 1), kernel_size=3)          # [B, C*9, L]
    batch, channels, height, width = x.shape
    patches = patches.view(batch, channels, 9, height * width)
    return patches.median(dim=2).values.view(batch, channels, height, width)


def _box_kernel(size: int, device, dtype) -> torch.Tensor:
    return torch.full((1, 1, size, size), 1.0 / (size * size), device=device, dtype=dtype)


def _binomial_kernel(size: int, device, dtype) -> torch.Tensor:
    """Separable binomial (Pascal) low-pass - a fixed integer kernel, so it is
    bit-reproducible and carries no fitted parameter."""
    row = torch.tensor([math.comb(size - 1, k) for k in range(size)], device=device, dtype=dtype)
    kernel = torch.outer(row, row)
    return (kernel / kernel.sum()).view(1, 1, size, size)


def _depthwise(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = x.shape[1]
    pad = kernel.shape[-1] // 2
    return F.conv2d(_reflect_pad(x, pad), kernel.expand(channels, 1, -1, -1), groups=channels)


def blend_box(x: torch.Tensor, *, size: int, alpha: float) -> torch.Tensor:
    """x + alpha * (box_smooth(x) - x). alpha = 0 is the identity, alpha = 1 is
    the plain smoother."""
    return x + alpha * (_depthwise(x, _box_kernel(size, x.device, x.dtype)) - x)


def blend_binomial(x: torch.Tensor, *, size: int, alpha: float) -> torch.Tensor:
    return x + alpha * (_depthwise(x, _binomial_kernel(size, x.device, x.dtype)) - x)


def unsharp(x: torch.Tensor, *, size: int, beta: float) -> torch.Tensor:
    """x + beta * (x - smooth(x)) - the sign-flipped counterpart of the blends,
    included so the sweep brackets the identity from both sides instead of only
    testing low-pass."""
    return x + beta * (x - _depthwise(x, _binomial_kernel(size, x.device, x.dtype)))


def blend_median(x: torch.Tensor, *, alpha: float) -> torch.Tensor:
    return x + alpha * (median3(x) - x)


def ae_reproject(x: torch.Tensor, model) -> torch.Tensor:
    """One autoencoder round trip, model.decode(model.encode(x)).

    The most theoretically motivated candidate here, and the only one that uses
    the model's own notion of a valid reconstruction. M17's oracle reference IS
    an autoencoder round trip of the raw previous frame; the deployed reference
    is `model.decode(reference_latent + dequant(residual))`, which has drifted
    off that manifold. Re-projecting moves it back toward the map's fixed point
    using nothing but the decoded frame.
    """
    return model.decode(model.encode(x))


def blend_ae_reproject(x: torch.Tensor, model, *, alpha: float) -> torch.Tensor:
    return x + alpha * (ae_reproject(x, model) - x)


def clamp01(x: torch.Tensor) -> torch.Tensor:
    """The decoder's own Sigmoid guarantees (0,1); the blends and especially
    `unsharp` can leave it, and `estimate_block_motion` documents its inputs as
    [0,1]. Clamping keeps every candidate inside the contract the frozen motion
    estimator was written against."""
    return x.clamp(0.0, 1.0)


# --- the pre-registered candidate family ---------------------------------------


class Refinement:
    """One pre-registered candidate. `domain` names the insertion point; `apply`
    receives ONLY the tensor at that point plus the frozen autoencoder.

    A plain class rather than a dataclass on purpose: these modules are loaded
    through `importlib.util.spec_from_file_location` without being registered in
    `sys.modules`, and `@dataclass` resolves its annotations through
    `sys.modules[cls.__module__]`, which does not exist under that loader.
    """

    __slots__ = ("name", "domain", "definition", "apply", "family", "params")

    def __init__(self, *, name: str, domain: str, definition: str,
                 apply: Callable[[torch.Tensor, Any], torch.Tensor],
                 family: str = "A", params: dict[str, Any] | None = None) -> None:
        self.name = name
        self.domain = domain              # "pixel" (previous), "latent" (reference), "none"
        self.definition = definition
        self.apply = apply
        self.family = family
        self.params = dict(params or {})

    def __repr__(self) -> str:
        return f"Refinement({self.name!r}, domain={self.domain!r})"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "domain": self.domain, "definition": self.definition,
                "family": self.family, "params": self.params}

    def __call__(self, tensor: torch.Tensor, model) -> torch.Tensor:
        out = self.apply(tensor, model)
        if out.shape != tensor.shape or out.dtype != tensor.dtype:
            raise ValueError(
                f"refinement {self.name!r} changed shape/dtype: {tuple(tensor.shape)}/"
                f"{tensor.dtype} -> {tuple(out.shape)}/{out.dtype}")
        return out


def _pixel(name, definition, fn, *, family="A", **params) -> Refinement:
    return Refinement(name=name, domain="pixel", definition=definition,
                      apply=fn, family=family, params=params)


def _latent(name, definition, fn, *, family="B", **params) -> Refinement:
    return Refinement(name=name, domain="latent", definition=definition,
                      apply=fn, family=family, params=params)


IDENTITY = Refinement(
    name="identity", domain="none", definition="R' = R  (the deployed path, unchanged)",
    apply=lambda x, model: x, family="control")

# FROZEN BEFORE ANY VAL-B CODED RESULT WAS INSPECTED.
CANDIDATES: tuple[Refinement, ...] = (
    IDENTITY,
    # --- Family A: pixel domain, applied to `previous` -------------------------
    _pixel("px_median3", "R' = clamp01(median_3x3(R))",
           lambda x, m: clamp01(median3(x))),
    _pixel("px_median3_a50", "R' = clamp01(R + 0.50 * (median_3x3(R) - R))",
           lambda x, m: clamp01(blend_median(x, alpha=0.50)), alpha=0.50),
    _pixel("px_box3_a25", "R' = clamp01(R + 0.25 * (box_3x3(R) - R))",
           lambda x, m: clamp01(blend_box(x, size=3, alpha=0.25)), alpha=0.25, size=3),
    _pixel("px_box3_a50", "R' = clamp01(R + 0.50 * (box_3x3(R) - R))",
           lambda x, m: clamp01(blend_box(x, size=3, alpha=0.50)), alpha=0.50, size=3),
    _pixel("px_box3_a100", "R' = clamp01(box_3x3(R))",
           lambda x, m: clamp01(blend_box(x, size=3, alpha=1.00)), alpha=1.00, size=3),
    _pixel("px_binom5_a25", "R' = clamp01(R + 0.25 * (binomial_5x5(R) - R))",
           lambda x, m: clamp01(blend_binomial(x, size=5, alpha=0.25)), alpha=0.25, size=5),
    _pixel("px_binom5_a50", "R' = clamp01(R + 0.50 * (binomial_5x5(R) - R))",
           lambda x, m: clamp01(blend_binomial(x, size=5, alpha=0.50)), alpha=0.50, size=5),
    _pixel("px_unsharp3_b25", "R' = clamp01(R + 0.25 * (R - binomial_3x3(R)))",
           lambda x, m: clamp01(unsharp(x, size=3, beta=0.25)), beta=0.25, size=3),
    _pixel("px_unsharp3_b50", "R' = clamp01(R + 0.50 * (R - binomial_3x3(R)))",
           lambda x, m: clamp01(unsharp(x, size=3, beta=0.50)), beta=0.50, size=3),
    _pixel("px_ae_reproject", "R' = model.decode(model.encode(R))",
           lambda x, m: ae_reproject(x, m)),
    _pixel("px_ae_reproject_a50", "R' = R + 0.50 * (model.decode(model.encode(R)) - R)",
           lambda x, m: blend_ae_reproject(x, m, alpha=0.50), alpha=0.50),
    _pixel("px_ae_reproject_a25", "R' = R + 0.25 * (model.decode(model.encode(R)) - R)",
           lambda x, m: blend_ae_reproject(x, m, alpha=0.25), alpha=0.25),
    # --- Family B: latent domain, applied to `reference` -----------------------
    _latent("lat_box3_a25", "Z' = Z + 0.25 * (box_3x3(Z) - Z), per latent channel",
            lambda x, m: blend_box(x, size=3, alpha=0.25), alpha=0.25, size=3),
    _latent("lat_box3_a50", "Z' = Z + 0.50 * (box_3x3(Z) - Z), per latent channel",
            lambda x, m: blend_box(x, size=3, alpha=0.50), alpha=0.50, size=3),
    _latent("lat_unsharp3_b25", "Z' = Z + 0.25 * (Z - binomial_3x3(Z))",
            lambda x, m: unsharp(x, size=3, beta=0.25), beta=0.25, size=3),
    _latent("lat_ae_reproject", "Z' = model.encode(model.decode(Z))",
            lambda x, m: m.encode(m.decode(x))),
    _latent("lat_ae_reproject_a50", "Z' = Z + 0.50 * (model.encode(model.decode(Z)) - Z)",
            lambda x, m: x + 0.50 * (m.encode(m.decode(x)) - x), alpha=0.50),
)

CANDIDATES_BY_NAME = {c.name: c for c in CANDIDATES}


def candidate(name: str) -> Refinement:
    if name not in CANDIDATES_BY_NAME:
        raise ValueError(f"unknown candidate {name!r}; pre-registered: "
                         f"{sorted(CANDIDATES_BY_NAME)}")
    return CANDIDATES_BY_NAME[name]


# --- the refined closed loop ----------------------------------------------------


@torch.no_grad()
def encode_sequence_refined(mc, m13, model, frames, spec, path, refinement, *, intra_params,
                            intra_entropy_model, residual_params, motion_entropy_model, bits,
                            gop_size, block_size, search_range) -> dict[str, Any]:
    """One closed-loop encode with `refinement` applied at its declared point.

    Structurally identical to `m13_closed_loop.encode_multi` restricted to the
    deployed `m13_recal` arm - same GOP handling, same motion round trip
    (`decode_motion_payload(encode_motion_payload(...))`, so the encoder warps
    with exactly the vectors the decoder will read), same `.nvct` v2 writer,
    same header fields. A parallel implementation rather than a modification,
    because every M10-M20 script is frozen; `test_m21_matches_m13_closed_loop_
    at_identity` pins that the identity candidate reproduces it byte-for-byte.
    """
    model.eval()
    device = next(model.parameters()).device
    frame_count = frames.shape[0]
    types = mc.gop_frame_types(frame_count, gop_size)
    latent_shape = tuple(model.encode(frames[0:1].to(device)).shape[1:])

    header = mc.TemporalStreamHeader(
        gop_size=gop_size, quantization_bits=intra_params.bits,
        quantization_mode=intra_params.mode, image_width=frames.shape[3],
        image_height=frames.shape[2], image_channels=frames.shape[1],
        latent_channels=latent_shape[0], latent_height=latent_shape[1],
        latent_width=latent_shape[2], frame_count=frame_count,
        num_intra_quantization_params=intra_params.scale.numel(),
        num_residual_quantization_params=residual_params.scale.numel(),
        block_size=block_size, search_range=search_range,
        motion_bits=motion_entropy_model.bits, reference_mode="mc",
        intra_entropy_model_id=intra_entropy_model.model_id(),
        residual_entropy_model_id=spec["identity"],
        motion_entropy_model_id=motion_entropy_model.model_id())
    writer = mc.TemporalStreamWriter(path, header, intra_params, residual_params)

    reconstructions, symbol_log, records, references = [], [], [], []
    previous = None
    ideal_bits = 0.0
    timings: dict[str, float] = {}
    completed = False
    try:
        with mc.deterministic_kernels():
            for index in range(frame_count):
                frame = frames[index:index + 1].to(device)
                latent = model.encode(frame)
                if types[index] == mc.FRAME_TYPE_I:
                    payload, _ = encode_latent_to_payload(
                        latent, params=intra_params, entropy_model=intra_entropy_model)
                    decoded, _ = decode_payload_to_latent(
                        payload, entropy_model=intra_entropy_model, params=intra_params,
                        shape=latent_shape)
                    reconstructed_latent = decoded.to(device)
                    writer.append_frame(types[index], b"", payload)
                    records.append({"frame_type": "I", "motion_bytes": 0,
                                    "residual_bytes": len(payload),
                                    "gop_position": index % gop_size})
                else:
                    # --- THE ONLY DIVERGENCE FROM THE DEPLOYED LOOP ------------
                    search_reference = (refinement(previous, model)
                                        if refinement.domain == "pixel" else previous)
                    motion = mc.estimate_block_motion(
                        search_reference, frame, block_size=block_size,
                        search_range=search_range)
                    motion_payload = mc.encode_motion_payload(
                        motion, search_range=search_range, entropy_model=motion_entropy_model)
                    decoded_motion = mc.decode_motion_payload(
                        motion_payload, (frames.shape[2] // block_size,
                                         frames.shape[3] // block_size),
                        search_range=search_range, entropy_model=motion_entropy_model)
                    warped = mc.warp_blocks(search_reference, decoded_motion,
                                            block_size=block_size)
                    reference_latent = model.encode(warped)
                    if refinement.domain == "latent":
                        reference_latent = refinement(reference_latent, model)
                    # -----------------------------------------------------------
                    symbols = latent_to_symbols(latent - reference_latent, residual_params)
                    symbol_log.append(symbols.reshape(latent_shape))
                    references.append(reference_latent.detach().cpu())
                    payload, ideal = m13.encode_frame_recalibrated(
                        spec["model"], spec["assign_codebook"], spec["coding_codebook"],
                        reference_latent, symbols.reshape(latent_shape), spec["zero"],
                        bits=bits, timings=timings)
                    ideal_bits += ideal
                    writer.append_frame(types[index], motion_payload, payload)
                    records.append({"frame_type": "P", "motion_bytes": len(motion_payload),
                                    "residual_bytes": len(payload),
                                    "gop_position": index % gop_size})
                    reconstructed_latent = reference_latent + symbols_to_latent(
                        symbols, latent_shape, residual_params).to(device)
                reconstruction = model.decode(reconstructed_latent)
                previous = reconstruction
                reconstructions.append(reconstruction.detach().cpu())
        completed = True
    finally:
        # On the happy path `close()` also asserts the frame count matches the
        # header. On an error path that assertion would REPLACE the real
        # exception with a confusing "only 0 frames written", so it is
        # suppressed there and the original traceback survives.
        if completed:
            writer.close()
        else:
            with contextlib.suppress(Exception):
                writer.close()

    container = path.stat().st_size
    motion_total = sum(r["motion_bytes"] for r in records)
    residual_total = sum(r["residual_bytes"] for r in records)
    return {
        "frame_count": frame_count, "symbols": symbol_log, "records": records,
        "reference_latents": references,
        "reconstructions": torch.cat(reconstructions, dim=0),
        "p_frames": sum(1 for r in records if r["frame_type"] == "P"),
        "i_frame_residual_bytes": sum(r["residual_bytes"] for r in records
                                      if r["frame_type"] == "I"),
        "p_frame_residual_bytes": sum(r["residual_bytes"] for r in records
                                      if r["frame_type"] == "P"),
        "motion_bytes": motion_total, "residual_bytes": residual_total,
        "container_bytes": container,
        "container_overhead_bytes": container - motion_total - residual_total,
        "p_frame_ideal_bits": ideal_bits, "encode_seconds": dict(timings),
    }


@torch.no_grad()
def decode_sequence_refined(mc, m13, model, path, spec, refinement, *, intra_entropy_model,
                            motion_entropy_model, bits) -> dict[str, Any]:
    """The decoder side, and the decoder-compatibility proof.

    It applies the SAME refinement at the SAME point, from state it already has:
    `previous` (which it reconstructed) and the motion field it just decoded. It
    receives no side information, and the `.nvct` v2 header is read, not
    extended - all three entropy identities are still checked exactly as
    `m13_closed_loop.decode_sequence` checks them.
    """
    model.eval()
    device = next(model.parameters()).device
    reader = mc.TemporalStreamReader(path)
    header = reader.header
    for label, expected, actual in (
            ("residual", header.residual_entropy_model_id, spec["identity"]),
            ("intra", header.intra_entropy_model_id, intra_entropy_model.model_id()),
            ("motion", header.motion_entropy_model_id, motion_entropy_model.model_id())):
        if expected != actual:
            raise mc.TemporalFormatError(
                f"{label} entropy model mismatch: stream declares {expected.hex()}, "
                f"supplied is {actual.hex()}")

    reconstructions, symbol_log, references = [], [], []
    previous = None
    with mc.deterministic_kernels():
        for frame_type, motion_payload, residual_payload in reader:
            if frame_type == mc.FRAME_TYPE_I:
                latent, _ = decode_payload_to_latent(
                    residual_payload, entropy_model=intra_entropy_model,
                    params=reader.intra_params, shape=header.latent_shape)
                reconstructed_latent = latent.to(device)
            else:
                search_reference = (refinement(previous, model)
                                    if refinement.domain == "pixel" else previous)
                motion = mc.decode_motion_payload(
                    motion_payload, header.block_grid, search_range=header.search_range,
                    entropy_model=motion_entropy_model)
                warped = mc.warp_blocks(search_reference, motion, block_size=header.block_size)
                reference_latent = model.encode(warped)
                if refinement.domain == "latent":
                    reference_latent = refinement(reference_latent, model)
                symbols = m13.decode_frame_recalibrated(
                    spec["model"], spec["assign_codebook"], spec["coding_codebook"],
                    residual_payload, reference_latent, spec["zero"], bits=bits,
                    shape=header.latent_shape)
                symbol_log.append(symbols.reshape(header.latent_shape))
                references.append(reference_latent.detach().cpu())
                reconstructed_latent = reference_latent + symbols_to_latent(
                    symbols, header.latent_shape, reader.residual_params).to(device)
            reconstruction = model.decode(reconstructed_latent)
            reconstructions.append(reconstruction)
            previous = reconstruction
    return {"reconstructions": torch.cat(reconstructions, dim=0), "symbols": symbol_log,
            "reference_latents": references}


# --- per-sequence driver + aggregation -----------------------------------------


def sequence_metrics(reconstructions: torch.Tensor, frames: torch.Tensor) -> tuple[float, float]:
    mse = torch.mean((reconstructions - frames) ** 2).item()
    return 10.0 * math.log10(1.0 / mse), float(msssim(reconstructions.clamp(0, 1), frames).mean())


def run_candidate(mc, m13, model, sequences, spec, refinement, stream_dir, *, intra_params,
                  intra_entropy_model, residual_params, motion_entropy_model, bits, gop_size,
                  block_size, search_range, verify_decode: bool = True) -> dict[str, Any]:
    """Encode every sequence under one candidate, decode each back, and
    aggregate. The decode is not optional bookkeeping - it is the Phase 4 proof
    that the refinement is reconstructible, so it defaults to on."""
    stream_dir.mkdir(parents=True, exist_ok=True)
    rows, per_sequence = [], []
    checks = {"symbols_exact": True, "reconstruction_exact": True, "references_exact": True,
              "sequences_verified": 0}
    gop_bytes: dict[int, dict[str, int]] = {}

    for sequence in sequences:
        frames = sequence.load_frames()
        path = stream_dir / f"{refinement.name}_{bits}bit_{sequence.sequence_id}.nvct"
        result = encode_sequence_refined(
            mc, m13, model, frames, spec, path, refinement, intra_params=intra_params,
            intra_entropy_model=intra_entropy_model, residual_params=residual_params,
            motion_entropy_model=motion_entropy_model, bits=bits, gop_size=gop_size,
            block_size=block_size, search_range=search_range)
        psnr, quality = sequence_metrics(result["reconstructions"], frames)

        if verify_decode:
            decoded = decode_sequence_refined(
                mc, m13, model, path, spec, refinement,
                intra_entropy_model=intra_entropy_model,
                motion_entropy_model=motion_entropy_model, bits=bits)
            checks["symbols_exact"] &= all(
                np.array_equal(np.asarray(a).reshape(-1), np.asarray(b).reshape(-1))
                for a, b in zip(result["symbols"], decoded["symbols"]))
            checks["reconstruction_exact"] &= torch.equal(
                decoded["reconstructions"].cpu(), result["reconstructions"])
            checks["references_exact"] &= all(
                torch.equal(a, b) for a, b in
                zip(result["reference_latents"], decoded["reference_latents"]))
            checks["sequences_verified"] += 1

        for record in result["records"]:
            bucket = gop_bytes.setdefault(
                record["gop_position"],
                {"residual_bytes": 0, "motion_bytes": 0, "frames": 0})
            bucket["residual_bytes"] += record["residual_bytes"]
            bucket["motion_bytes"] += record["motion_bytes"]
            bucket["frames"] += 1

        rows.append({"sequence": sequence.sequence_id, **{
            k: v for k, v in result.items()
            if k not in ("symbols", "reconstructions", "records", "reference_latents")},
            "total_pixels": sequence.total_pixels, "raw_rgb_bytes": sequence.raw_rgb_bytes(),
            "mean_psnr_db": psnr, "mean_msssim": quality})
        per_sequence.append({"sequence": sequence.sequence_id,
                             "residual_bytes": result["residual_bytes"],
                             "p_frame_residual_bytes": result["p_frame_residual_bytes"],
                             "motion_bytes": result["motion_bytes"],
                             "container_bytes": result["container_bytes"],
                             "mean_psnr_db": psnr, "mean_msssim": quality})

    return {"aggregate": aggregate(refinement.name, bits, rows), "per_sequence": per_sequence,
            "gop_position": {str(k): v for k, v in sorted(gop_bytes.items())},
            "decode_checks": checks}


def aggregate(name: str, bits: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Byte/metric aggregation with the SAME field names and the same
    `byte_accounting_closes` invariant `m13_closed_loop.aggregate` uses, so M21
    totals are directly comparable to M13/M14's recorded ones."""
    total = sum(r["container_bytes"] for r in rows)
    pixels = sum(r["total_pixels"] for r in rows)
    encode: dict[str, float] = {}
    for row in rows:
        for key, value in row["encode_seconds"].items():
            encode[key] = encode.get(key, 0.0) + value
    return {
        "arm": name, "bits": bits,
        "total_motion_bytes": sum(r["motion_bytes"] for r in rows),
        "total_residual_bytes": sum(r["residual_bytes"] for r in rows),
        "total_i_frame_residual_bytes": sum(r["i_frame_residual_bytes"] for r in rows),
        "total_p_frame_residual_bytes": sum(r["p_frame_residual_bytes"] for r in rows),
        "total_container_overhead_bytes": sum(r["container_overhead_bytes"] for r in rows),
        "total_container_bytes": total, "total_pixels": pixels,
        "stream_bpp": total * 8 / pixels,
        "compression_ratio": sum(r["raw_rgb_bytes"] for r in rows) / total,
        "mean_psnr_db": statistics.fmean(r["mean_psnr_db"] for r in rows),
        "mean_msssim": statistics.fmean(r["mean_msssim"] for r in rows),
        "p_frame_ideal_bits": sum(r["p_frame_ideal_bits"] for r in rows),
        "p_frames": sum(r["p_frames"] for r in rows),
        "encode_seconds": encode,
        "byte_accounting_closes": sum(
            r["motion_bytes"] + r["residual_bytes"] + r["container_overhead_bytes"]
            for r in rows) == total,
    }


# --- the frozen rig -------------------------------------------------------------


def prepare_rate_point(model, *, bits: int, manifest: Path, checkpoint: Path, m11_dir: Path,
                       m10k_dir: Path, device, cache_dir, train_full, motion_train,
                       calibration_frames: int, gop_size: int, block_size: int,
                       search_range: int, motion_cache: Path | None = None) -> dict[str, Any]:
    """Rebuild the DEPLOYED arm for one bit depth: M13-recalibrated residual
    tables plus M14's recalibrated motion table.

    The residual half is assembled exactly as M19/M20 assemble it. The motion
    half is M14's own `collect_motion_symbols` + `fit_empirical` over the same
    broad-TRAIN allocation, cached to disk because it costs a full-search motion
    pass; the cache is only ever trusted after its identity is checked against
    M14's recorded `d7e7b237b6451885`.
    """
    mc = _load_script("m10h_motion_compensation")
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    md = _load_script("m11_data")
    cx = _load_script("m11_causal_context")
    mk = _load_script("m10k_learned_entropy")
    m13 = _load_script("m13_recalibration")
    m14 = _load_script("m14_recalibration")
    ev = _load_script("m10l_evaluate")
    ev_m11 = _load_script("m11_evaluate")
    gate_script = _load_script("m12_spatial_offline_gate")

    calibration = mc.calibrate_grids(
        model, train_full, bits=bits, mode="per_channel", gop_size=gop_size,
        block_size=block_size, search_range=search_range, reference_mode="mc",
        max_frames=calibration_frames)
    signature = ev.calibration_signature(calibration, bits=bits,
                                         calibration_frames=calibration_frames,
                                         quant_mode="per_channel")
    data = md.load_or_collect(model, checkpoint=checkpoint, manifest=manifest, bits=bits,
                              device=device, cache_dir=cache_dir, log=lambda m: None)
    train_symbols = data["train_symbols"].astype(np.int64)
    val_symbols = data["val_symbols"].astype(np.int64)
    select_mask, _ = md.split_validation(data)
    zero = torch.from_numpy(cx.zero_symbols(calibration["residual_params"],
                                            train_symbols.shape[1]))

    _, m10k_checkpoint = mk.load_entropy_model(m10k_dir / f"learned_entropy_{bits}bit.pt",
                                               device=device)
    m10k_identity = ev._m10k_model_identity(m10k_checkpoint["model_state_dict"])
    model11, checkpoint11 = ma.load_model(m11_dir / f"m11_G16_entropy_{bits}bit.pt", device=device)
    ev_m11.check_provenance(checkpoint11, signature=signature, bits=bits,
                            group_size=M11_G16_GROUP_SIZE,
                            context_definition_id=ma.context_definition_id(M11_G16_GROUP_SIZE),
                            m10k_identity=m10k_identity)
    assign_codebook = ml.SharedCodebook.from_dict(json.loads(
        (m11_dir / f"m11_G16_codebook_{bits}bit_K512.json").read_text(encoding="utf-8")))
    train_k = gate_script.m11_g16_prototype_indices(model11, assign_codebook, train_symbols,
                                                    data["train_references"], zero, device=device)
    val_k = gate_script.m11_g16_prototype_indices(model11, assign_codebook, val_symbols,
                                                  data["val_references"], zero, device=device)
    frequencies, strength, _, _ = m13.fit_recalibrated_frequencies(
        assign_codebook, train_symbols.reshape(-1), train_k.reshape(-1),
        val_symbols[select_mask].reshape(-1), val_k[select_mask].reshape(-1), alphabet=2 ** bits)
    coding_codebook = m13.build_recalibrated_codebook(
        assign_codebook, frequencies, provenance={"bits": bits, "strength": strength})
    residual_identity = ma.model_identity(model11, m10k_identity=m10k_identity,
                                          calibration_signature=signature, bits=bits,
                                          codebook=coding_codebook)

    motion_entropy_model = deployed_motion_model(
        mc, m14, model, motion_train, device=device, gop_size=gop_size, block_size=block_size,
        search_range=search_range, cache=motion_cache)

    return {
        "bits": bits, "calibration": calibration, "signature": signature,
        "intra_params": calibration["intra_params"],
        "intra_entropy_model": calibration["intra_entropy_model"],
        "residual_params": calibration["residual_params"], "zero": zero,
        "motion_entropy_model": motion_entropy_model,
        "spec": {"model": model11, "zero": zero, "assign_codebook": assign_codebook,
                 "coding_codebook": coding_codebook, "identity": residual_identity},
        "model11": model11, "assign_codebook": assign_codebook,
        "coding_codebook": coding_codebook,
        "residual_identity": residual_identity.hex(),
        "assign_codebook_id": assign_codebook.codebook_id().hex(),
        "coding_codebook_id": coding_codebook.codebook_id().hex(),
        "intra_identity": calibration["intra_entropy_model"].model_id().hex(),
        "motion_identity": motion_entropy_model.model_id().hex(),
        "m10k_identity": m10k_identity.hex(), "m13_strength": float(strength),
    }


def deployed_motion_model(mc, m14, model, motion_train, *, device, gop_size, block_size,
                          search_range, cache: Path | None):
    """M14's deployed recalibrated motion table, cached on disk.

    The cache is validated by IDENTITY, not by existence: a cached table whose
    `model_id()` is not M14's recorded `d7e7b237b6451885` is rejected and
    refitted, so a stale cache is a loud failure rather than a silent
    rate difference.
    """
    from nvc.compression.entropy_model import EmpiricalEntropyModel

    motion_bits = mc.motion_alphabet_bits(search_range)
    if cache is not None and cache.is_file():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        table = EmpiricalEntropyModel(np.array(stored["frequencies"], dtype=np.int64),
                                      bits=motion_bits)
        if table.model_id().hex() == DEPLOYED_MOTION_IDENTITY:
            return table
    symbols = m14.collect_motion_symbols(
        mc, model, motion_train, block_size=block_size, search_range=search_range,
        gop_size=gop_size, max_frames=10 ** 9, reference_mode="mc", device=device)
    table = m14.fit_empirical(symbols, bits=motion_bits, num_tables=2)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"frequencies": table.frequencies.tolist(),
                                     "identity": table.model_id().hex(),
                                     "motion_train_p_frames": int(symbols.shape[0])}),
                         encoding="utf-8")
    return table


def val_b_sequences(manifest: Path, *, count: int, max_frames: int | None = None):
    """VAL-B is the odd-indexed half of VAL - the same held-out split M13-M20
    used, selected the same way, never TRAIN and never TEST."""
    from nvc.evaluation.sequences import discover_sequences
    sequences = discover_sequences(manifest, split="val", max_frames_per_sequence=max_frames)
    return sequences[1::2][:count]


def broad_train_sequences(manifest: Path, *, frames_per_sequence: int = 8):
    """M14's own broad-TRAIN allocation for motion-table fitting, reproduced
    with the same call and the same default."""
    from nvc.evaluation.sequences import discover_sequences
    return discover_sequences(manifest, split="train",
                              max_frames_per_sequence=frames_per_sequence)
