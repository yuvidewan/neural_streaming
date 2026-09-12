"""M20 - codebook-assignment hysteresis: the rule, and an encoder/decoder pair
that implement it OUTSIDE production (`src/nvc/` is never touched).

WHAT M19 FOUND, AND WHAT THIS TESTS
-----------------------------------
M19's 2x2 decomposition split the reference-error excess bits two ways:

  * ~88-91%  the reference error changes the residual SYMBOL itself;
  * ~9-12%   the symbol is unchanged but the codebook ASSIGNMENT flips,
             so an identical target gets coded by a different table.

Only the second component is addressable without changing the residual
transform, and M19 separately showed the assignment is BRITTLE: it flips for
35-55% of positions even in the SMALLEST error decile. That is the hypothesis
this module exists to test - not to assume: a stability margin might remove
pointless churn, or might pin symbols onto worse tables, or might do nothing.

THE RULE
--------
`SharedCodebook.assign_tensor` picks

    k*(i) = argmin_k  cost(i, k),   cost(i, k) = sum_s p_i(s) * -log2 q_k(s)

(cross-entropy in BITS - the expected code length; ties to the lowest index).
Hysteresis adds one previous-assignment candidate and a strict margin:

    advantage(i) = cost(i, prev(i)) - cost(i, k*(i))        >= 0 always
    a(i) = prev(i)   if prev(i) is defined and advantage(i) <  margin
           k*(i)     otherwise

The comparison is STRICT `<`, which makes `margin = 0` reduce to the deployed
rule EXACTLY (advantage is never negative, so no position is ever held) -
`margin = 0` is therefore an identity control by construction, not by
measurement. `tests/test_m20_codebook_hysteresis.py` pins that anyway.

`margin` is in BITS of expected code length, the same unit as `cost`, so the
sweep is directly interpretable and Phase C's empirical-scale check compares
it against the observed distribution of `advantage`.

THREE "PREVIOUS" DEFINITIONS, ALL CAUSAL
----------------------------------------
`prev(i)` is not obvious a priori, so three pre-declared definitions are
swept rather than one being assumed:

  temporal       the FINAL assignment at the same flat position in the
                 previous P-frame of the same GOP. Undefined at GOP
                 position 1 (the frame after an I-frame), where the rule
                 degenerates to the deployed argmin.
  channel_group  the FINAL assignment at the same (h, w) in channel c - G,
                 i.e. the previous G16 decoding group. Undefined in group 0.
  raster         the FINAL assignment at flat index i - 1 (C-major, the
                 coder's own order). Undefined at i = 0.

DECODER COMPATIBILITY - WHY THIS IS CAUSAL (Phase B)
-----------------------------------------------------
The load-bearing fact is that `cost` depends ONLY on `rows`, and `rows` is
`G16(reference, context_planes(symbols))`, whose planes for channel c use
only channels before G*floor(c/G). So:

  * the assignment never looks at the symbol it is about to code;
  * choosing a different table does NOT change `rows` - the coding is
    lossless, so the decoder recovers the identical symbols and therefore
    the identical context planes and the identical `rows`. Hysteresis
    cannot feed back into its own inputs.

Consequently every quantity in the rule - cost(i, .), k*(i), and each of the
three prev(i) definitions - is computable at the decoder BEFORE the symbol at
i is decoded, using only what it has already decoded. No side information is
transmitted and `.nvct` v2 is untouched. `decode_frame_hysteresis` below is
the executable proof: it reconstructs the assignment from the payload alone.

A second consequence worth stating because it bounds the whole milestone:
hysteresis changes ONLY which table codes an unchanged symbol. Symbols,
reconstruction, motion, G16 context and PSNR/MS-SSIM are mathematically
untouched - the only thing that can move is residual payload bytes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.range_coder import ResumableDecoder, encode_symbols

# Pre-declared (Phase C) - fixed here, before any held-out result is read.
#
# The brief's own grid is 0 .. 0.10. A 10-P-frame pilot measured the quantity
# the margin is actually compared against - advantage = cost(i, prev(i)) -
# cost(i, k*(i)) - and found (temporal, 5-bit) p25 = 0.005, p50 = 0.031,
# p75 = 0.107, p90 = 0.251 bits. So the brief's grid spans roughly p10..p75 but
# stops short of p90, and Phase C asks for a sweep reaching the 90th percentile.
# 0.25 and 0.50 are therefore appended - BEFORE any full VAL-B result was read -
# so the response curve is measured past p90 and a finite optimum at a large
# margin cannot be missed by construction. Nothing here is tuned on TEST.
STATES = ("temporal", "channel_group", "raster")
MARGIN_SWEEP = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50)

# NOT M20 candidates. Two reference points that bound how much the routing-only
# component could be worth to ANY re-routing rule, reported alongside the sweep
# so a null hysteresis result can be read as "not exploitable this way" rather
# than "not exploitable at all":
#   coding_metric  argmin of cross-entropy against the RECALIBRATED (M13)
#                  coding tables instead of the original prototypes. Causal and
#                  fully decoder-available - it exists because the deployed rule
#                  scores the ORIGINAL prototypes while the coder spends bits on
#                  M13's recalibrated ones, a mismatch M13 introduced by design.
#   oracle_table   the table that minimises the ACTUAL code length of the true
#                  symbol. Non-causal (it reads the symbol it is about to code),
#                  so it is an upper bound, never a candidate.
DIAGNOSTIC_RULES = ("coding_metric", "oracle_table")


DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M11_DIR = Path("outputs/m11_autoregressive_entropy")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
DEFAULT_OUTPUT_DIR = Path("outputs/m20_codebook_hysteresis")
RATE_POINTS = (5, 4, 3)
M11_G16_GROUP_SIZE = 16

WEAK_BELOW_PERCENT = 0.5
MEANINGFUL_ABOVE_PERCENT = 1.0


def verdict(gain_percent: float) -> str:
    """The project's frozen total-stream thresholds, reused verbatim."""
    return ("meaningful" if gain_percent >= MEANINGFUL_ABOVE_PERCENT else
            "marginal" if gain_percent >= WEAK_BELOW_PERCENT else "weak")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the frozen rig, assembled exactly as M17/M18/M19 assemble it ----------------


def prepare_rate_point(model, *, bits: int, manifest: Path, checkpoint: Path, m11_dir: Path,
                       m10k_dir: Path, device, cache_dir, train_full, calibration_frames: int,
                       gop_size: int, block_size: int, search_range: int) -> dict[str, Any]:
    """Rebuild the frozen M11-G16 + M13 residual arm for one bit depth.

    Line-for-line the same construction M19's own Phase 0 uses (same
    calibration, same M10K/G16 provenance check, same K=512 assignment
    codebook, same TRAIN-fit/VAL-A-selected M13 recalibrated coding tables) -
    reproduced here rather than imported because M19's version is welded into
    its `main()`. The identities it returns are what Phase 0 checks against
    M13/M14/M15/M17/M18/M19's recorded values, so any drift fails loudly.
    """
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
    channels = train_symbols.shape[1]
    zero = torch.from_numpy(cx.zero_symbols(calibration["residual_params"], channels))

    m10k, m10k_checkpoint = mk.load_entropy_model(
        m10k_dir / f"learned_entropy_{bits}bit.pt", device=device)
    del m10k
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

    def flat(array):
        return array.reshape(-1)

    frequencies, strength, _, _ = m13.fit_recalibrated_frequencies(
        assign_codebook, flat(train_symbols), flat(train_k), flat(val_symbols[select_mask]),
        flat(val_k[select_mask]), alphabet=2 ** bits)
    coding_codebook = m13.build_recalibrated_codebook(
        assign_codebook, frequencies, provenance={"bits": bits, "strength": strength})
    residual_identity = ma.model_identity(model11, m10k_identity=m10k_identity,
                                          calibration_signature=signature, bits=bits,
                                          codebook=coding_codebook)
    return {
        "bits": bits, "calibration": calibration, "signature": signature,
        "intra_params": calibration["intra_params"],
        "intra_entropy_model": calibration["intra_entropy_model"],
        "residual_params": calibration["residual_params"], "zero": zero,
        "model11": model11, "assign_codebook": assign_codebook,
        "coding_codebook": coding_codebook, "m10k_identity": m10k_identity.hex(),
        "residual_identity": residual_identity.hex(),
        "assign_codebook_id": assign_codebook.codebook_id().hex(),
        "coding_codebook_id": coding_codebook.codebook_id().hex(),
        "m13_strength": float(strength),
    }


def val_b_sequences(manifest: Path, *, count: int, max_frames: int | None = None):
    """VAL-B is the odd-indexed half of VAL - the same held-out split M13-M19
    used, selected the same way, never TRAIN and never TEST."""
    from nvc.evaluation.sequences import discover_sequences
    sequences = discover_sequences(manifest, split="val", max_frames_per_sequence=max_frames)
    return sequences[1::2][:count]


# --- the assignment cost, reused rather than re-derived -------------------------


def assignment_costs(rows: torch.Tensor, codebook) -> torch.Tensor:
    """[N, A] predicted distributions -> [N, K] assignment cost, in BITS.

    This is literally the tensor `SharedCodebook.assign_tensor` reduces over -
    same operand (`codebook._torch_prototypes(rows, "log2")`), same dtype, same
    device, same GEMM - so `argmin_lowest_index` of it is bit-identical to the
    deployed assignment rather than a re-derivation that could drift. Reaching
    for the private prototype cache is deliberate: recomputing `-log2(p)` here
    would be a second, independently-rounded copy of the same constant.
    """
    if codebook.metric == "l1":
        prototypes = codebook._torch_prototypes(rows, "probabilities")
        return (rows[:, None, :] - prototypes[None, :, :]).abs().sum(dim=2)
    return rows @ codebook._torch_prototypes(rows, "log2").T


def baseline_assignment(costs: torch.Tensor) -> np.ndarray:
    """The deployed argmin (ties to the lowest index), from a cost matrix."""
    ml = _load_script("m10l_shared_codebook")
    return ml._torch_argmin_lowest_index(costs).to("cpu").numpy().astype(np.int64)


def coding_metric_assignment(rows: torch.Tensor, coding_codebook) -> np.ndarray:
    """DIAGNOSTIC, not a candidate. The same cross-entropy argmin, but scored
    against the tables the coder actually spends bits on (M13's recalibrated
    frequencies) rather than the original prototypes. Causal - it reads only
    `rows` - and therefore decoder-available, which is exactly what makes it a
    meaningful reference point for what re-routing could be worth."""
    return baseline_assignment(assignment_costs(rows, coding_codebook))


def oracle_table_assignment(coding_codebook, symbols_flat: np.ndarray) -> np.ndarray:
    """DIAGNOSTIC upper bound, NOT decoder-compatible. Routes every position to
    the table that minimises the actual code length of the symbol being coded.

    The best table depends only on the symbol VALUE, so it is a single [A]
    lookup; `argmax` returns the first maximum, matching the lowest-index tie
    rule used everywhere else."""
    best_for_symbol = np.argmax(coding_codebook.probabilities, axis=0).astype(np.int64)
    return best_for_symbol[np.asarray(symbols_flat, dtype=np.int64).reshape(-1)]


# --- the hysteresis rule --------------------------------------------------------


def _advantage(costs: np.ndarray, previous: np.ndarray, best_cost: np.ndarray) -> np.ndarray:
    """cost(i, prev(i)) - cost(i, k*(i)) for a vector of candidate previous
    assignments. Non-negative by construction (k* is the argmin)."""
    return costs[np.arange(costs.shape[0]), previous] - best_cost


def apply_hysteresis(costs: np.ndarray, best: np.ndarray, *, margin: float, state: str,
                     previous_frame: np.ndarray | None, shape: tuple[int, int, int],
                     group_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    """The rule. Returns (assignment, trace).

    `costs` is [N, K] in coder order (C-major); `best` is the deployed argmin.
    `previous_frame` is the previous P-frame's FINAL assignment (temporal
    state only); None means "undefined", which degenerates to `best`.

    The trace records exactly how many positions were HELD (kept on the
    previous assignment instead of the argmin) and their advantage, which is
    what Phase E decomposes.
    """
    if state not in STATES:
        raise ValueError(f"unknown hysteresis state {state!r}; expected one of {STATES}")
    costs = np.asarray(costs, dtype=np.float64)
    best = np.asarray(best, dtype=np.int64)
    channels, height, width = shape
    plane = height * width
    total = channels * plane
    if costs.shape[0] != total or best.shape[0] != total:
        raise ValueError(f"costs/best have {costs.shape[0]}/{best.shape[0]} rows, expected {total}")
    best_cost = costs[np.arange(total), best]

    # margin = 0 is the identity control: `advantage` is never negative, so a
    # STRICT `<` can never hold a position. Short-circuiting here keeps that
    # exact rather than relying on floating-point luck at the boundary.
    if margin <= 0.0:
        return best.copy(), {"held": 0, "eligible": 0, "held_advantage_sum": 0.0}

    assignment = best.copy()
    held_total = 0
    eligible_total = 0
    advantage_sum = 0.0

    if state == "temporal":
        if previous_frame is not None:
            previous = np.asarray(previous_frame, dtype=np.int64)
            if previous.shape[0] != total:
                raise ValueError("previous_frame assignment has the wrong length")
            advantage = _advantage(costs, previous, best_cost)
            hold = advantage < margin
            assignment = np.where(hold, previous, best)
            held_total = int(np.count_nonzero(hold & (previous != best)))
            eligible_total = total
            advantage_sum = float(advantage[hold].sum())
    elif state == "channel_group":
        # Group g's previous is group g-1's FINAL assignment at the same (h, w),
        # so the groups must be resolved in decoding order - which is also the
        # order the decoder resolves them in.
        for start in range(group_size, channels, group_size):
            current = slice(start * plane, (start + group_size) * plane)
            earlier = slice((start - group_size) * plane, start * plane)
            previous = assignment[earlier]
            block_costs = costs[current]
            block_best = best[current]
            advantage = _advantage(block_costs, previous, best_cost[current])
            hold = advantage < margin
            assignment[current] = np.where(hold, previous, block_best)
            held_total += int(np.count_nonzero(hold & (previous != block_best)))
            eligible_total += block_costs.shape[0]
            advantage_sum += float(advantage[hold].sum())
    else:  # raster
        # A genuine sequential recurrence over the coder's own order. The
        # decoder can run it too: every cost in a decoding group is known
        # before ANY symbol of that group is decoded (costs depend on `rows`,
        # not on the symbols being coded), so the scan never needs a symbol it
        # has not already decoded.
        current = int(best[0])
        for index in range(1, total):
            candidate = int(best[index])
            if current != candidate:
                advantage = costs[index, current] - best_cost[index]
                if advantage < margin:
                    advantage_sum += float(advantage)
                    held_total += 1
                else:
                    current = candidate
            assignment[index] = current
            eligible_total += 1

    return assignment, {"held": held_total, "eligible": eligible_total,
                        "held_advantage_sum": advantage_sum}


# --- frame coding with the rule (assignment is the ONLY thing that changes) ------


@torch.no_grad()
def frame_rows(model, reference: torch.Tensor, symbols: np.ndarray,
               zero: torch.Tensor) -> torch.Tensor:
    """The G16 predicted-probability rows for a whole frame, exactly as
    `m13.encode_frame_recalibrated` computes them."""
    ma = _load_script("m11_ar_entropy")
    device = reference.device
    target = torch.from_numpy(np.asarray(symbols, dtype=np.int64))[None].to(device)
    return ma._rows(model.log_probabilities(reference, model.planes(target, zero.to(device))))


def code_with_assignment(coding_codebook, symbols_flat: np.ndarray,
                         table_index: np.ndarray) -> tuple[bytes, float]:
    """Arithmetic-code `symbols_flat` under `table_index`. Identical to the
    tail of `m13.encode_frame_recalibrated`, with the assignment supplied."""
    flat = np.asarray(symbols_flat, dtype=np.int64).reshape(-1)
    payload = encode_symbols(flat, coding_codebook.cumulative, table_index)
    totals = coding_codebook.cumulative[:, -1]
    lower = coding_codebook.cumulative[table_index, flat]
    upper = coding_codebook.cumulative[table_index, flat + 1]
    ideal = float(-np.log2((upper - lower) / totals[table_index]).sum())
    return payload, ideal


@torch.no_grad()
def encode_frame_hysteresis(model, assign_codebook, coding_codebook, reference: torch.Tensor,
                            symbols: np.ndarray, zero: torch.Tensor, *, bits: int,
                            margin: float, state: str,
                            previous_frame: np.ndarray | None) -> dict[str, Any]:
    """`m13.encode_frame_recalibrated` with the hysteresis rule in front of the
    coder. Returns the payload, the ideal bits, and both assignments."""
    del bits
    rows = frame_rows(model, reference, symbols, zero)
    costs = assignment_costs(rows, assign_codebook)
    best = baseline_assignment(costs)
    shape = tuple(np.asarray(symbols).shape)
    assignment, trace = apply_hysteresis(
        costs.double().cpu().numpy(), best, margin=margin, state=state,
        previous_frame=previous_frame, shape=shape, group_size=model.group_size)
    payload, ideal = code_with_assignment(coding_codebook, np.asarray(symbols).reshape(-1),
                                          assignment)
    return {"payload": payload, "ideal_bits": ideal, "assignment": assignment,
            "baseline_assignment": best, "trace": trace}


@torch.no_grad()
def decode_frame_hysteresis(model, assign_codebook, coding_codebook, payload: bytes,
                            reference: torch.Tensor, zero: torch.Tensor, *, bits: int,
                            shape: tuple[int, int, int], margin: float, state: str,
                            previous_frame: np.ndarray | None) -> dict[str, Any]:
    """The decoder side, and the executable decoder-compatibility proof.

    Structurally `m13.decode_frame_recalibrated` with the same hysteresis rule
    spliced in. It receives NO side information: the assignment is rebuilt from
    `rows` (recomputed from the reference and the symbols decoded so far) plus
    the previous frame's assignment, which the decoder already reconstructed
    when it decoded that frame.

    `channel_group`/`raster` are resolved incrementally, group by group, in the
    decoder's own order - the same order and the same result the encoder's
    whole-frame call produces, which is the property the tests pin.
    """
    del bits
    ma = _load_script("m11_ar_entropy")
    channels, height, width = shape
    plane = height * width
    group = model.group_size
    device = reference.device
    zero_d = zero.to(device)
    decoded = zero_d.view(channels, 1, 1).expand(channels, height, width).clone()
    symbols = np.empty(channels * plane, dtype=np.int64)
    assignment = np.empty(channels * plane, dtype=np.int64)
    baseline = np.empty(channels * plane, dtype=np.int64)
    previous = None if previous_frame is None else np.asarray(previous_frame, dtype=np.int64)

    decoder = ResumableDecoder(payload)
    for start in range(0, channels, group):
        stop = start + group
        span = slice(start * plane, stop * plane)
        log_probabilities = model.log_probabilities(reference, model.planes(decoded[None], zero_d))
        rows = ma._rows(log_probabilities[:, start:stop])
        cost_tensor = assignment_costs(rows, assign_codebook)
        best = baseline_assignment(cost_tensor)
        costs = cost_tensor.double().cpu().numpy()
        baseline[span] = best

        if margin <= 0.0:
            group_assignment = best.copy()
        elif state == "temporal":
            if previous is None:
                group_assignment = best.copy()
            else:
                group_assignment, _ = apply_hysteresis(
                    costs, best, margin=margin, state="temporal",
                    previous_frame=previous[span], shape=(group, height, width),
                    group_size=group)
        elif state == "channel_group":
            if start == 0:
                group_assignment = best.copy()
            else:
                earlier = assignment[(start - group) * plane:start * plane]
                best_cost = costs[np.arange(costs.shape[0]), best]
                advantage = _advantage(costs, earlier, best_cost)
                group_assignment = np.where(advantage < margin, earlier, best)
        elif state == "raster":
            # Continue the scan across the group boundary - the last position of
            # the previous group is already final.
            group_assignment = np.empty(group * plane, dtype=np.int64)
            best_cost = costs[np.arange(costs.shape[0]), best]
            if start:
                current = int(assignment[start * plane - 1])
                first = 0
            else:
                current = int(best[0])
                group_assignment[0] = current
                first = 1
            for index in range(first, group * plane):
                candidate = int(best[index])
                if current != candidate and costs[index, current] - best_cost[index] >= margin:
                    current = candidate
                group_assignment[index] = current
        else:
            raise ValueError(f"unknown hysteresis state {state!r}; expected one of {STATES}")

        assignment[span] = group_assignment
        group_symbols = decoder.decode_group(coding_codebook.cumulative, group_assignment)
        symbols[span] = group_symbols
        decoded[start:stop] = torch.from_numpy(
            group_symbols.reshape(group, height, width)).to(device)
    decoder.close()
    return {"symbols": symbols, "assignment": assignment, "baseline_assignment": baseline}
