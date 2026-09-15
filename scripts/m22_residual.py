"""M22 - the residual-quantizer freeze-lift: grid variants, the downstream
refit they force, and the coupling audit that explains why a refit is forced.

`src/nvc/` is never touched; no M10-M21 script is modified; `.nvct` v2 is
unchanged.

THE COUPLING THAT DEFINES THIS MILESTONE
----------------------------------------
`m10l_evaluate.calibration_signature` hashes the RESIDUAL quantizer's scale and
zero_point (and nothing else about the calibration) into a 16-hex digest, and
`m11_evaluate.check_provenance` refuses to load the M11-G16 checkpoint unless
that digest matches the one it was trained under. M10K's deployment lesson - a
learned entropy model scored against a different grid silently costs bits rather
than failing - was deliberately turned into a hard error here.

The consequence for M22 is structural, not incidental: **the residual quantizer
has no interface at which it can be changed in isolation.** Changing it
invalidates, in order, the M10K entropy model, the M11-G16 model trained from
it, the K=512 assignment codebook fitted to G16's predictions, and the M13
coding frequencies fitted to that codebook's assignments. `m22_baseline.py`
demonstrates each link by probing it rather than asserting it.

M22 therefore measures every grid variant through TWO arms, kept explicitly
apart so nothing is attributed to the wrong cause:

  STALE   the new grid with the deployed M10K/G16/codebook/M13 left exactly as
          they are (the provenance guard bypassed by an explicit, recorded
          override). This isolates the grid change itself and prices the
          coupling - it is what "just change the quantizer" actually costs.
  REFIT   the new grid with the whole downstream stack rebuilt on TRAIN by the
          same, unmodified fitting code that produced the deployed one. This is
          the honest deployable arm. It is a compound change, and is labelled as
          one everywhere; the CAUSE is still a single declared knob (the grid),
          and every downstream step is mechanical rather than a second design
          choice.

WHAT IS FROZEN
--------------
GOP, motion estimator, motion coding, motion entropy table, `.nvct` v2, the
range coder, the G16 ARCHITECTURE (group size 16, context definition), K=512 as
the codebook size, the M13 mechanism, lambda, the autoencoder weights, and the
intra quantizer. Only the residual grid moves - and, in the REFIT arm, the
fitted parameters of the components that the grid invalidates.

PRE-REGISTRATION
----------------
`GRID_VARIANTS` is frozen in this module before any VAL-B coded byte was read
(Phase 2's diagnostics informed which knobs are worth turning; Phase 2 reads
distributions, never coded results - that is the milestone's own ordering).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.quantization import QuantizationParams

DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M11_DIR = Path("outputs/m11_autoregressive_entropy")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
DEFAULT_OUTPUT_DIR = Path("outputs/m22_residual_freeze_lift")
RATE_POINTS = (5, 4, 3)
M11_G16_GROUP_SIZE = 16
DEPLOYED_MOTION_IDENTITY = "d7e7b237b6451885"

# The deployed percentiles, from `calibrate_grids`' own defaults.
DEPLOYED_LOWER_PERCENTILE = 0.1
DEPLOYED_UPPER_PERCENTILE = 99.9

WEAK_BELOW_PERCENT = 0.5
MEANINGFUL_ABOVE_PERCENT = 1.0

# Unlike M20 (which re-routed a fixed symbol) and M21 (whose winner improved
# quality as well), a residual grid change moves the codec along its
# rate/distortion curve by construction: a coarser grid makes symbols more
# stable and cheaper while reconstructing worse. The milestone allows a bitrate
# win that does not improve PSNR, "provided ... no unacceptable distortion
# regression occurs" - so the acceptable regression is declared here, before any
# coded result, rather than argued afterwards. BD-rate over the three rate
# points is reported alongside, because that is the metric designed for exactly
# this trade.
MAX_PSNR_REGRESSION_DB = 0.10
MAX_MSSSIM_REGRESSION = 0.0010

# Phase 5/17 protocol, declared before any VAL-B coded result was read.
SCREEN_BITS = 3          # the rate point with the largest M17 oracle gap (+19.6%)
CONFIRM_BITS = (5, 4)
STAGE2_ADMISSION_PERCENT = -0.25
STAGE2_MINIMUM_CANDIDATES = 2


def verdict(gain_percent: float) -> str:
    """The project's frozen total-stream thresholds, reused verbatim."""
    return ("meaningful" if gain_percent >= MEANINGFUL_ABOVE_PERCENT else
            "marginal" if gain_percent >= WEAK_BELOW_PERCENT else "weak")


def distortion_regression(delta_psnr_db: float, delta_msssim: float) -> str | None:
    """The pre-declared distortion guard. Returns a reason string when the
    regression is unacceptable, or None when the candidate may be judged on
    rate alone. Never used to PREFER a candidate - only to disqualify one."""
    reasons = []
    if delta_psnr_db < -MAX_PSNR_REGRESSION_DB:
        reasons.append(f"PSNR {delta_psnr_db:+.4f} dB exceeds the declared "
                       f"-{MAX_PSNR_REGRESSION_DB:.2f} dB allowance")
    if delta_msssim < -MAX_MSSSIM_REGRESSION:
        reasons.append(f"MS-SSIM {delta_msssim:+.6f} exceeds the declared "
                       f"-{MAX_MSSSIM_REGRESSION:.4f} allowance")
    return "; ".join(reasons) if reasons else None


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- grid construction ----------------------------------------------------------


def grid_signature(params: QuantizationParams) -> str:
    """A stable fingerprint of a residual grid, in the SAME form
    `m10l_evaluate.calibration_signature` uses - so a grid can be identified
    without having to build a whole calibration dict around it."""
    digest = hashlib.sha256()
    digest.update(np.asarray(params.scale.cpu().numpy(), dtype=np.float64).tobytes())
    digest.update(np.asarray(params.zero_point.cpu().numpy(), dtype=np.float64).tobytes())
    digest.update(json.dumps({"bits": params.bits, "mode": params.mode},
                             sort_keys=True).encode())
    return digest.hexdigest()[:16]


def _params_from_range(x_min: torch.Tensor, x_max: torch.Tensor, *, bits: int,
                       mode: str = "per_channel") -> QuantizationParams:
    """The SAME affine derivation `calibrate_quantization_params` performs, given
    an explicit range - reproduced here (and pinned by a test against the real
    function) because every variant below differs only in how [x_min, x_max] is
    chosen, never in how it becomes a grid."""
    levels = 2 ** bits - 1
    degenerate = x_max == x_min
    if degenerate.any():
        x_min = torch.where(degenerate, x_min - 1e-3, x_min)
        x_max = torch.where(degenerate, x_max + 1e-3, x_max)
    scale = (x_max - x_min) / levels
    zero_point = -torch.round(x_min / scale)
    return QuantizationParams(scale=scale, zero_point=zero_point, bits=bits, mode=mode)


def _per_channel(residuals: torch.Tensor) -> torch.Tensor:
    """[N, C, H, W] -> [C, N*H*W] float32, the layout every statistic below uses."""
    channels = residuals.shape[1]
    return residuals.permute(1, 0, 2, 3).reshape(channels, -1).to(torch.float32)


def percentile_grid(residuals: torch.Tensor, *, bits: int, lower: float,
                    upper: float) -> QuantizationParams:
    """The deployed rule, with the percentiles exposed."""
    return calibrate_quantization_params(residuals, bits=bits, mode="per_channel",
                                         lower_percentile=lower, upper_percentile=upper)


def symmetric_percentile_grid(residuals: torch.Tensor, *, bits: int, lower: float,
                              upper: float) -> QuantizationParams:
    """Force the grid symmetric about zero: half-width = max(|p_lo|, |p_hi|).

    Motivated by the residual being a motion-compensated difference, whose
    distribution is close to zero-symmetric; an asymmetric percentile range
    spends levels on a tail that is only heavy on one side of some channels.
    """
    values = _per_channel(residuals)
    low = torch.quantile(values, lower / 100.0, dim=1)
    high = torch.quantile(values, upper / 100.0, dim=1)
    half = torch.maximum(low.abs(), high.abs()).reshape(1, -1, 1, 1)
    return _params_from_range(-half, half, bits=bits)


def mad_grid(residuals: torch.Tensor, *, bits: int, k: float) -> QuantizationParams:
    """Robust scale: half-width = k * MAD, symmetric about zero.

    The median absolute deviation is insensitive to the long tails a percentile
    rule has to chase, so this trades clipping frequency for step size under a
    statistic the tails cannot move.
    """
    values = _per_channel(residuals)
    median = values.median(dim=1).values
    mad = (values - median[:, None]).abs().median(dim=1).values
    half = (k * mad).clamp(min=1e-6).reshape(1, -1, 1, 1)
    return _params_from_range(-half, half, bits=bits)


def mse_optimal_uniform_grid(residuals: torch.Tensor, *, bits: int,
                             candidates: int = 48) -> QuantizationParams:
    """Per channel, pick the symmetric half-width minimising ACTUAL quantization
    MSE on TRAIN, by scanning a fixed geometric ladder of half-widths.

    This is the only variant that optimises anything rather than applying a
    closed-form rule, and what it optimises is the quantizer's own error on the
    calibration set - a TRAIN-only, deterministic scan over a declared ladder,
    not a search over held-out data. It exists so the family contains a
    principled "best possible uniform grid" reference point: if even this does
    not help, no re-placement of a uniform grid will.
    """
    values = _per_channel(residuals)
    channels, levels = values.shape[0], 2 ** bits - 1
    reference = values.abs().max(dim=1).values.clamp(min=1e-6)
    ladder = torch.tensor([0.02 * (1.25 ** i) for i in range(candidates)],
                          dtype=torch.float32)
    best_half = torch.empty(channels, dtype=torch.float32)
    for channel in range(channels):
        column = values[channel]
        halves = (ladder * reference[channel]).clamp(min=1e-6)
        errors = []
        for half in halves:
            scale = (2.0 * half) / levels
            zero_point = -torch.round(-half / scale)
            quantized = torch.round(column / scale) + zero_point
            quantized = quantized.clamp(0, levels)
            errors.append(float(((quantized - zero_point) * scale - column).pow(2).mean()))
        best_half[channel] = halves[int(np.argmin(errors))]
    half = best_half.reshape(1, -1, 1, 1)
    return _params_from_range(-half, half, bits=bits)


class GridVariant:
    """One pre-registered residual-grid rule. `build` receives the TRAIN residual
    stack and the bit depth, and returns a QuantizationParams. It never sees
    VAL-B, VAL, or TEST data."""

    __slots__ = ("name", "definition", "build", "family", "params")

    def __init__(self, *, name: str, definition: str,
                 build: Callable[[torch.Tensor, int], QuantizationParams],
                 family: str = "A", params: dict[str, Any] | None = None) -> None:
        self.name = name
        self.definition = definition
        self.build = build
        self.family = family
        self.params = dict(params or {})

    def __repr__(self) -> str:
        return f"GridVariant({self.name!r})"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "definition": self.definition, "family": self.family,
                "params": self.params}


DEPLOYED = GridVariant(
    name="deployed",
    definition=("scale/zero_point from calibrate_quantization_params at percentiles "
                "(0.1, 99.9), per channel - the frozen production grid"),
    build=lambda residuals, bits: percentile_grid(
        residuals, bits=bits, lower=DEPLOYED_LOWER_PERCENTILE,
        upper=DEPLOYED_UPPER_PERCENTILE),
    family="control")

# FROZEN BEFORE ANY VAL-B CODED BYTE WAS READ.
GRID_VARIANTS: tuple[GridVariant, ...] = (
    DEPLOYED,
    GridVariant(name="broad_p001",
                definition="percentiles (0.01, 99.99): wider range, coarser step, less clipping",
                build=lambda r, b: percentile_grid(r, bits=b, lower=0.01, upper=99.99),
                params={"lower": 0.01, "upper": 99.99}),
    GridVariant(name="tight_p1",
                definition="percentiles (1.0, 99.0): narrower range, finer step, more clipping",
                build=lambda r, b: percentile_grid(r, bits=b, lower=1.0, upper=99.0),
                params={"lower": 1.0, "upper": 99.0}),
    GridVariant(name="tight_p05",
                definition="percentiles (0.5, 99.5): a milder tightening than tight_p1",
                build=lambda r, b: percentile_grid(r, bits=b, lower=0.5, upper=99.5),
                params={"lower": 0.5, "upper": 99.5}),
    GridVariant(name="symmetric_p01",
                definition="symmetric about zero, half-width = max(|p0.1|, |p99.9|)",
                build=lambda r, b: symmetric_percentile_grid(
                    r, bits=b, lower=DEPLOYED_LOWER_PERCENTILE,
                    upper=DEPLOYED_UPPER_PERCENTILE)),
    GridVariant(name="mad_k8",
                definition="symmetric, half-width = 8 * per-channel MAD (robust scale)",
                build=lambda r, b: mad_grid(r, bits=b, k=8.0), params={"k": 8.0}),
    GridVariant(name="mad_k12",
                definition="symmetric, half-width = 12 * per-channel MAD",
                build=lambda r, b: mad_grid(r, bits=b, k=12.0), params={"k": 12.0}),
    GridVariant(name="mse_optimal",
                definition=("symmetric, per-channel half-width minimising TRAIN quantization "
                            "MSE over a fixed 48-point geometric ladder"),
                build=lambda r, b: mse_optimal_uniform_grid(r, bits=b),
                params={"candidates": 48}),
)

GRID_VARIANTS_BY_NAME = {variant.name: variant for variant in GRID_VARIANTS}


def grid_variant(name: str) -> GridVariant:
    if name not in GRID_VARIANTS_BY_NAME:
        raise ValueError(f"unknown grid variant {name!r}; pre-registered: "
                         f"{sorted(GRID_VARIANTS_BY_NAME)}")
    return GRID_VARIANTS_BY_NAME[name]


# --- TRAIN residual collection (shared by the diagnostics and every variant) -----


@torch.no_grad()
def collect_train_residuals(mc, model, sequences, *, bits, intra_params, intra_entropy_model,
                            gop_size, block_size, search_range, max_frames, device,
                            cache: Path | None = None) -> dict[str, Any]:
    """The residual stack `calibrate_grids` fits its grid on, reproduced by its
    EXACT walk so every variant is fitted on identical data.

    Two details are load-bearing, and one of them was gotten wrong once before
    `verify_deployed_grid` caught it:

      * the I-frame reference is `model.decode(INTRA ROUND TRIP of the latent)`,
        not `model.decode(latent)` - the intra quantizer is inside this loop,
        and skipping it perturbs every residual downstream, shifting the fitted
        grid and silently invalidating the identity control;
      * the reference then advances from the TRUE latent, exactly as
        `calibrate_grids` documents, because the residual grid being fitted here
        does not exist yet and cannot be used to close the loop.
    """
    from nvc.compression.codec import decode_payload_to_latent, encode_latent_to_payload

    key = (bits, gop_size, block_size, search_range, max_frames, "intra_roundtrip_v2")
    if cache is not None and cache.is_file():
        stored = torch.load(cache, weights_only=False)
        if stored.get("key") == key:
            return stored

    model.eval()
    intra_latents: list[torch.Tensor] = []
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
    intra_stack = torch.cat(intra_latents, dim=0)
    latent_shape = tuple(intra_stack.shape[1:])

    residuals: list[torch.Tensor] = []
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
                    motion = mc.estimate_block_motion(previous, frame, block_size=block_size,
                                                      search_range=search_range)
                    warped = mc.warp_blocks(previous, motion, block_size=block_size)
                    residuals.append((latent - model.encode(warped)).cpu())
                    previous = model.decode(latent)
                seen += 1
            if seen >= max_frames:
                break
    out = {"key": key, "residual_stack": torch.cat(residuals, dim=0), "intra_stack": intra_stack}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, cache)
    return out


def verify_deployed_grid(residual_stack: torch.Tensor, deployed_params, *, bits: int) -> None:
    """The identity control, enforced rather than hoped for.

    If the `deployed` variant rebuilt from this stack is not the SAME grid
    `calibrate_grids` produced, every delta in the sweep is measured from the
    wrong origin - so this RAISES rather than quietly reporting a control that
    is not the control. It exists because exactly that happened once: a missing
    intra round trip in the collection walk shifted the 3-bit mean step from
    1.7323 to 1.7266 and moved the control's VAL-B total from 723,381 bytes to
    897,872.
    """
    rebuilt = DEPLOYED.build(residual_stack, bits)
    if grid_signature(rebuilt) != grid_signature(deployed_params):
        raise ValueError(
            f"the TRAIN residual stack does not reproduce the deployed {bits}-bit grid: "
            f"rebuilt {grid_signature(rebuilt)} vs deployed {grid_signature(deployed_params)} "
            f"(mean step {float(rebuilt.scale.mean()):.6f} vs "
            f"{float(deployed_params.scale.mean()):.6f}) - the collection walk has drifted "
            f"from calibrate_grids")


# --- grid statistics (Phase 2 A-F, and reported for every variant) --------------


def grid_statistics(residuals: torch.Tensor, params: QuantizationParams, *,
                    bits: int) -> dict[str, Any]:
    """Distribution, step, clipping and symbol entropy for one (residuals, grid)
    pair. Everything here is a property of TRAIN data plus the grid - no held-out
    data and no coded bytes are involved."""
    values = _per_channel(residuals)
    levels = 2 ** bits - 1
    scale = params.scale.reshape(-1).to(torch.float32)
    zero_point = params.zero_point.reshape(-1).to(torch.float32)

    quantized = torch.round(values / scale[:, None]) + zero_point[:, None]
    clipped = quantized.clamp(0, levels)
    clipping = (quantized != clipped).to(torch.float32)
    dequantized = (clipped - zero_point[:, None]) * scale[:, None]
    error = dequantized - values

    # Distance from each value to ITS OWN reconstruction level, in units of the
    # step: 0 means exactly on a level (maximally stable under a perturbation),
    # 0.5 means exactly on a decision boundary (maximally fragile).
    offset = (values / scale[:, None] + zero_point[:, None])
    level_distance = (offset - torch.round(offset)).abs()

    flat = clipped.reshape(-1).to(torch.int64).clamp(0, levels)
    counts = torch.bincount(flat, minlength=2 ** bits).to(torch.float64)
    probability = counts / counts.sum()
    nonzero = probability[probability > 0]
    entropy = float(-(nonzero * torch.log2(nonzero)).sum())

    return {
        "grid_signature": grid_signature(params),
        "step_mean": float(scale.mean()), "step_min": float(scale.min()),
        "step_max": float(scale.max()),
        "step_per_channel": scale.tolist(),
        "zero_point_mean": float(zero_point.mean()),
        "zero_point_in_range_fraction": float(
            ((zero_point >= 0) & (zero_point <= levels)).to(torch.float32).mean()),
        "residual_std_per_channel": values.std(dim=1).tolist(),
        "residual_std_mean": float(values.std(dim=1).mean()),
        "residual_abs_mean": float(values.abs().mean()),
        "residual_rms": float(values.pow(2).mean().sqrt()),
        "residual_p999_abs_mean": float(
            torch.quantile(values.abs(), 0.999, dim=1).mean()),
        "dynamic_range_mean": float(
            (values.max(dim=1).values - values.min(dim=1).values).mean()),
        "step_over_std_mean": float((scale / values.std(dim=1).clamp(min=1e-9)).mean()),
        "clipping_fraction": float(clipping.mean()),
        "clipping_fraction_per_channel": clipping.mean(dim=1).tolist(),
        "quantization_mse": float(error.pow(2).mean()),
        "quantization_snr_db": float(
            10.0 * torch.log10(values.pow(2).mean() / error.pow(2).mean().clamp(min=1e-30))),
        "symbol_entropy_bits": entropy,
        "symbol_entropy_efficiency": entropy / bits,
        "distinct_symbols_used": int((counts > 0).sum()),
        "alphabet": 2 ** bits,
        "level_distance_mean": float(level_distance.mean()),
        "fraction_within_0_05_of_level": float((level_distance < 0.05).to(
            torch.float32).mean()),
        "fraction_within_0_10_of_level": float((level_distance < 0.10).to(
            torch.float32).mean()),
        "fraction_within_0_25_of_level": float((level_distance < 0.25).to(
            torch.float32).mean()),
        "zero_exactly_representable": bool(
            torch.allclose(torch.round(zero_point), zero_point)),
    }


# --- the downstream stack a new grid forces to be refitted -----------------------


def residual_entropy_model_for_grid(residuals: torch.Tensor, residual_params, *, bits: int):
    """The per-channel EMPIRICAL residual model `calibrate_grids` fits alongside
    the grid, refitted for a new grid by the same call it uses.

    It is not part of the deployed G16/codebook coding path - it is what
    `collect_training_symbols` uses to CLOSE THE LOOP while gathering symbols -
    but it is fitted to the grid, so a new grid invalidates it too. Refitting it
    here (TRAIN only) is one more mechanical consequence of the grid change, and
    is recorded as such rather than left stale.
    """
    from nvc.compression.codec import latent_to_symbols
    from nvc.compression.entropy_model import EmpiricalEntropyModel

    channels = residuals.shape[1]
    symbols = np.stack([
        latent_to_symbols(residuals[i:i + 1], residual_params).reshape(channels, -1)
        for i in range(residuals.shape[0])])
    return EmpiricalEntropyModel.from_symbols(symbols, bits=bits, num_tables=channels)


@torch.no_grad()
def collect_symbols_for_grid(mc, ce, model, sequences, calibration, *, gop_size, block_size,
                             search_range, device, max_frames) -> tuple[list, list]:
    """`m10j_conditional_entropy.collect_training_symbols` with an EXPLICIT
    residual grid, reusing that function unmodified by handing it a calibration
    dict whose residual_params (and residual_entropy_model) are the variant's.

    `m11_data.load_or_collect`'s cache key does not include the grid, so calling
    it for a non-deployed grid would silently return symbols collected under the
    deployed one. That trap is the reason this wrapper exists.
    """
    for field in ("intra_params", "intra_entropy_model", "residual_params",
                  "residual_entropy_model", "motion_entropy_model"):
        if field not in calibration:
            raise KeyError(f"collect_symbols_for_grid needs calibration[{field!r}]")
    return ce.collect_training_symbols(
        mc, model, sequences, calibration, gop_size=gop_size, block_size=block_size,
        search_range=search_range, device=device, max_frames=max_frames)


def refit_downstream(mc, mk, ma, ml, mt, m13, cx, ce, ev, model, *, bits, residual_params,
                     intra_params, intra_entropy_model, motion_entropy_model, residual_stack,
                     train_sequences, val_sequences, device, gop_size, block_size,
                     search_range, train_frames, val_frames_per_sequence, seed, m10k_epochs,
                     g16_epochs, codebook_size, codebook_rows, log=print) -> dict[str, Any]:
    """Rebuild M10K -> M11-G16 -> K=512 codebook -> M13 frequencies for a grid.

    Every step calls the ORIGINAL, unmodified fitting function that produced the
    deployed component - `mk.train_entropy_model`, `ma.from_m10k`/`ma.train`,
    `m11_train.fit_model_codebook`, `m13.fit_recalibrated_frequencies` - with the
    same hyperparameters, seed and data discipline. Nothing here is a new design
    choice; the only input that differs is the grid.
    """
    started = time.perf_counter()
    channels = int(intra_params.scale.numel())
    zero = torch.from_numpy(cx.zero_symbols(residual_params, channels))

    log(f"      collecting TRAIN symbols under grid {grid_signature(residual_params)} ...")
    calibration = {
        "intra_params": intra_params, "intra_entropy_model": intra_entropy_model,
        "residual_params": residual_params,
        "residual_entropy_model": residual_entropy_model_for_grid(
            residual_stack, residual_params, bits=bits),
        "motion_entropy_model": motion_entropy_model}
    train_symbols, train_references = collect_symbols_for_grid(
        mc, ce, model, train_sequences, calibration, gop_size=gop_size, block_size=block_size,
        search_range=search_range, device=device, max_frames=train_frames)
    val_symbols, val_references, val_sequence_index = [], [], []
    for index, sequence in enumerate(val_sequences):
        symbols, references = collect_symbols_for_grid(
            mc, ce, model, [sequence], calibration, gop_size=gop_size, block_size=block_size,
            search_range=search_range, device=device, max_frames=sequence.frame_count)
        val_symbols += symbols
        val_references += references
        val_sequence_index += [index] * len(symbols)

    train_symbols = np.stack(train_symbols).astype(np.int64)
    train_references = np.stack(train_references).astype(np.float32)
    val_symbols = np.stack(val_symbols).astype(np.int64)
    val_references = np.stack(val_references).astype(np.float32)
    select_mask = np.asarray(val_sequence_index) % 2 == 0

    as_long = lambda a: torch.from_numpy(np.asarray(a)).long()
    as_float = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
    train_set = (as_long(train_symbols), as_float(train_references))
    select_set = (as_long(val_symbols[select_mask]), as_float(val_references[select_mask]))

    log("      training M10K ...")
    m10k = mk.build_model({"latent_channels": channels, "alphabet": 2 ** bits,
                           "hidden": 32}).to(device)
    mk.train_entropy_model(
        m10k, (list(train_set[0].numpy()), list(train_set[1].numpy())),
        (list(select_set[0].numpy()), list(select_set[1].numpy())),
        epochs=m10k_epochs, batch_size=8, learning_rate=1e-3, device=device, seed=seed,
        log=lambda m: log("        " + m))
    m10k_identity = ev._m10k_model_identity(m10k.state_dict())

    log("      training M11-G16 ...")
    g16 = ma.from_m10k(m10k, use_context=True, group_size=M11_G16_GROUP_SIZE).to(device)
    history = ma.train(g16, train_set, select_set, zero, epochs=g16_epochs, batch_size=8,
                       learning_rate=1e-3, device=device, seed=seed,
                       log=lambda m: log("        " + m))
    selected = min(history, key=lambda h: (h["val_loss"], h["epoch"]))

    log("      fitting the K=512 codebook ...")
    assign_codebook = mt.fit_model_codebook(ml, g16, train_set, zero, bits=bits,
                                            size=codebook_size, rows=codebook_rows,
                                            device=device, seed=seed)
    train_k = m13.m11_g16_prototype_indices(g16, assign_codebook, train_symbols,
                                            train_references, zero, device=device)
    val_k = m13.m11_g16_prototype_indices(g16, assign_codebook, val_symbols, val_references,
                                          zero, device=device)
    log("      fitting M13 coding frequencies ...")
    frequencies, strength, _, _ = m13.fit_recalibrated_frequencies(
        assign_codebook, train_symbols.reshape(-1), train_k.reshape(-1),
        val_symbols[select_mask].reshape(-1), val_k[select_mask].reshape(-1),
        alphabet=2 ** bits)
    coding_codebook = m13.build_recalibrated_codebook(
        assign_codebook, frequencies, provenance={"bits": bits, "strength": strength,
                                                  "m22_grid": grid_signature(residual_params)})
    return {
        "model11": g16, "assign_codebook": assign_codebook, "coding_codebook": coding_codebook,
        "zero": zero, "m10k_identity": m10k_identity.hex(),
        "m13_strength": float(strength), "g16_selected_epoch": selected["epoch"],
        "g16_val_a_bits": selected["val_loss"], "g16_history": history,
        "residual_entropy_model_id": calibration["residual_entropy_model"].model_id().hex(),
        "train_p_frames": int(train_symbols.shape[0]),
        "val_p_frames": int(val_symbols.shape[0]),
        "refit_seconds": time.perf_counter() - started,
    }


def save_refit_checkpoint(path: Path, refit: dict[str, Any], *, variant_name: str, bits: int,
                          residual_params, signature: str, residual_identity: str, seed: int,
                          m10k_epochs: int, g16_epochs: int, codebook_size: int,
                          train_sequences, val_sequences) -> dict[str, Any]:
    """Phase 11: write the refitted stack with everything needed to prove a
    checkpoint was never evaluated under a different quantizer.

    The digest is taken AFTER the file is written and stored beside it, so the
    recorded SHA256 is of the artifact a later run actually loads.
    """
    ma = _load_script("m11_ar_entropy")
    model11 = refit["model11"]
    payload = {
        "m22_variant": variant_name, "bits": bits,
        "model_state_dict": model11.state_dict(),
        "model_config": model11.config_dict(),
        "assign_codebook": refit["assign_codebook"].to_dict(),
        "coding_codebook": refit["coding_codebook"].to_dict(),
        "residual_scale": residual_params.scale.cpu(),
        "residual_zero_point": residual_params.zero_point.cpu(),
        "quantizer_identity": {"grid_signature": grid_signature(residual_params),
                               "calibration_signature": signature,
                               "bits": bits, "mode": residual_params.mode},
        "architecture_identity": {
            "group_size": M11_G16_GROUP_SIZE,
            "context_definition_id": ma.context_definition_id(M11_G16_GROUP_SIZE),
            "codebook_size": codebook_size},
        "entropy_identities": {
            "residual": residual_identity,
            "m10k": refit["m10k_identity"],
            "assign_codebook": refit["assign_codebook"].codebook_id().hex(),
            "coding_codebook": refit["coding_codebook"].codebook_id().hex(),
            "residual_entropy_model": refit.get("residual_entropy_model_id")},
        "training": {
            "seed": seed, "m10k_epochs": m10k_epochs, "g16_epochs": g16_epochs,
            "objective": ("pure rate: -log2 P(symbol | reference, causal context), in "
                          "bits/symbol - the SAME objective m10k/m11 were trained under. The "
                          "autoencoder is frozen, so lambda never enters this milestone's "
                          "training and no sensitivity term is used"),
            "lambda": 3e-4, "lambda_role": "inherited from the frozen M10F checkpoint, not swept",
            "gamma": None, "gamma_role": "not applicable - no candidate C was trained",
            "selected_epoch": refit["g16_selected_epoch"],
            "best_val_a_bits": refit["g16_val_a_bits"],
            "final_val_a_bits": refit["g16_history"][-1]["val_loss"],
            "selection_rule": "min (val_loss, epoch) on the VAL-A selection split; VAL-B unread"},
        "dataset_identity": {
            "train_sequences": [s.sequence_id for s in train_sequences],
            "train_frames": sum(s.frame_count for s in train_sequences),
            "val_sequences": [s.sequence_id for s in val_sequences],
            "val_frames": sum(s.frame_count for s in val_sequences),
            "train_p_frames_collected": refit["train_p_frames"],
            "val_p_frames_collected": refit["val_p_frames"],
            "split_discipline": "TRAIN fits, VAL-A selects, VAL-B reports, TEST unread"},
        "code_identity": code_identity(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {"path": str(path).replace("\\", "/"), "sha256": digest,
              "bytes": path.stat().st_size,
              **{k: payload[k] for k in ("quantizer_identity", "architecture_identity",
                                         "entropy_identities", "training", "dataset_identity",
                                         "code_identity")}}
    path.with_suffix(".provenance.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def code_identity() -> dict[str, str]:
    """SHA256 of every script whose behaviour could change a refitted stack, so
    a checkpoint cannot be silently reinterpreted by edited code."""
    here = Path(__file__).parent
    names = ("m22_residual.py", "m22_sweep.py", "m10h_motion_compensation.py",
             "m10j_conditional_entropy.py", "m10k_learned_entropy.py",
             "m10l_shared_codebook.py", "m11_ar_entropy.py", "m11_train.py",
             "m13_recalibration.py")
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest()[:16]
            for name in names if (here / name).is_file()}


def residual_identity_for(ma, model11, *, m10k_identity_hex: str, signature: str, bits: int,
                          coding_codebook) -> str:
    """`m11_ar_entropy.model_identity`, unmodified - the same 8-byte digest the
    `.nvct` v2 residual-entropy-model field carries, so a refitted stack is a
    DIFFERENT stream identity rather than a silent rate change."""
    return ma.model_identity(model11, m10k_identity=bytes.fromhex(m10k_identity_hex),
                             calibration_signature=signature, bits=bits,
                             codebook=coding_codebook).hex()


def calibration_signature_for(ev, residual_params, *, bits: int, calibration_frames: int,
                              quant_mode: str = "per_channel") -> str:
    """`m10l_evaluate.calibration_signature` needs a whole calibration dict; this
    hands it the one field it actually hashes."""
    return ev.calibration_signature({"residual_params": residual_params}, bits=bits,
                                    calibration_frames=calibration_frames,
                                    quant_mode=quant_mode)


def val_b_sequences(manifest: Path, *, count: int, max_frames: int | None = None):
    """VAL-B is the odd-indexed half of VAL - the same held-out split M13-M21
    used, selected the same way, never TRAIN and never TEST."""
    from nvc.evaluation.sequences import discover_sequences
    sequences = discover_sequences(manifest, split="val", max_frames_per_sequence=max_frames)
    return sequences[1::2][:count]


def broad_train_sequences(manifest: Path, *, frames_per_sequence: int = 8):
    """M14's own broad-TRAIN allocation for motion-table fitting."""
    from nvc.evaluation.sequences import discover_sequences
    return discover_sequences(manifest, split="train",
                              max_frames_per_sequence=frames_per_sequence)
