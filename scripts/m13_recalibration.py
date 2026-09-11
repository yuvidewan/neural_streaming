"""M13 - recalibrate M11-G16's DEPLOYED 512-entry codebook frequency tables
against TRAIN's true symbol histogram, and prove it as an actual coded-byte
gain rather than an offline entropy estimate.

WHAT "RECALIBRATION" MEANS HERE (Phase A audit)
-------------------------------------------------
M11-G16's codebook (`m10l_shared_codebook.SharedCodebook`, fitted by
`m11_train.fit_model_codebook`) holds 512 prototype distributions. Those
prototypes are Lloyd's-algorithm CENTROIDS of the NETWORK's own predicted
probability rows on TRAIN - the mean predicted distribution among the rows
assigned to each cluster - quantized once via
`m10k_learned_entropy.probabilities_to_frequencies`. They were never fitted
against what symbols ACTUALLY occurred at those positions; they approximate
the network's own prediction, not the empirical outcome.

M12's offline gate (scripts/m12_spatial_offline_gate.py) found that simply
counting which symbol actually occurred at every TRAIN position assigned to
prototype k, and refitting P(s|k) from THOSE counts (hierarchically smoothed
back toward the deployed prototype when a (k, s) pair is rarely observed -
`m11_causal_context.fit_parent`/`smoothed_parent`, exactly M11's own
methodology, reused unmodified here) reduces held-out entropy by 1.6-5.1%
- a "recalibration" gain M12 explicitly separated from spatial context and
flagged as the more attractive lever for a future milestone. That estimate
was FLOAT bits/symbol under `cx.parent_bits`, never quantized to an actual
integer frequency table or run through the real arithmetic coder. This
module turns it into exactly that: a real, deployable, `.nvct`-v2-compatible
coding table.

THE ONE-WAY SPLIT: ASSIGNMENT vs CODING
-----------------------------------------
`SharedCodebook.assign_tensor` (which prototype k a position routes to) and
`SharedCodebook.cumulative`/`.frequencies` (what the arithmetic coder uses
once k is known) both live on the same object today, because M10L/M11 never
needed to vary one without the other. M13 does: the experiment is defined as
"exact same 512 prototypes, exact same symbol-to-prototype assignments,
DIFFERENT frequency tables", so recalibration must NEVER be allowed to
change which k a position is assigned to - only what P(s|k) the coder uses
once k is fixed.

Concretely: `encode_frame_recalibrated`/`decode_frame_recalibrated` below
take TWO codebook objects - `assign_codebook` (the ORIGINAL, deployed
M11-G16 codebook, unchanged, used ONLY for `.assign_tensor`) and
`coding_codebook` (a separate `SharedCodebook` holding the recalibrated
frequencies, used ONLY for `.cumulative`). `coding_codebook.assign_tensor`
is never called by this module - if it were, its (different) `.frequencies`
would produce a (different) argmin-tie-break-sensitive assignment, silently
turning "recalibration only" into "recalibration + a redefinition of which
prototype every position uses", which is exactly what the milestone forbids.
`tests/test_m13_recalibration.py` pins this by construction (mocking
`assign_tensor` to fail loudly if called on the coding codebook).

PROVENANCE FOR FREE
--------------------
`m11_ar_entropy.model_identity(model, ..., codebook=codebook)` already
hashes `codebook.frequencies` into its 8-byte digest - so simply passing the
recalibrated `coding_codebook` into that SAME, UNMODIFIED function
automatically produces a distinct identity from the deployed arm's, with no
new identity scheme and no `.nvct` format change (Phase B requirement
11/12). `SharedCodebook.codebook_id()` likewise already hashes frequencies.
Both are exercised unmodified here.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.range_coder import ResumableDecoder, encode_symbols


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- fitting: TRAIN counts, VAL-A-selected smoothing strength, toward the deployed prior --


def fit_recalibrated_frequencies(original_codebook, train_symbols_flat: np.ndarray,
                                 train_k_flat: np.ndarray, select_symbols_flat: np.ndarray,
                                 select_k_flat: np.ndarray, *, alphabet: int):
    """TRAIN-only, deterministic. Returns (frequencies, strength, val_a_bits, val_a_baseline_bits).

    `train_*` fits raw counts; `select_*` (VAL-A) picks the smoothing strength
    from M11's own fixed grid (`cx.SMOOTHING_GRID`) by minimizing held-out
    bits/symbol - never touching VAL-B or TEST. The prior each count is
    smoothed toward is the DEPLOYED codebook's own probabilities
    (`original_codebook.probabilities`), so an under-observed (k, s) pair
    falls back to what is already shipped rather than to something arbitrary.
    """
    cx = _load_script("m11_causal_context")
    mk = _load_script("m10k_learned_entropy")

    groups = original_codebook.size
    prior = original_codebook.probabilities
    counts = cx.fit_parent(train_symbols_flat, train_k_flat, groups, alphabet)

    baseline_bits = cx.parent_bits(prior, select_symbols_flat, select_k_flat) / select_symbols_flat.size
    strength, val_a_bits = cx.select_strength(lambda s: cx.parent_bits(
        cx.smoothed_parent(counts, prior, s), select_symbols_flat, select_k_flat)
        / select_symbols_flat.size)

    probabilities = cx.smoothed_parent(counts, prior, strength)
    frequencies = mk.probabilities_to_frequencies(probabilities)
    return frequencies, strength, val_a_bits, baseline_bits


def build_recalibrated_codebook(original_codebook, frequencies: np.ndarray, *,
                                provenance: dict[str, Any] | None = None):
    """A SEPARATE `SharedCodebook` holding the recalibrated frequencies.

    Same K, same bit depth, same assignment metric field (irrelevant here -
    see the module docstring: this object's `.assign_tensor` must never be
    called). Exists only to supply `.cumulative`/`.frequencies` to the coder
    and a distinct `.codebook_id()`/feed into `ma.model_identity` for
    provenance.
    """
    ml = _load_script("m10l_shared_codebook")
    return ml.SharedCodebook(frequencies, bits=original_codebook.bits,
                             metric=original_codebook.metric,
                             provenance=dict(provenance or {}, kind="m13_recalibrated"))


# --- prototype indices (reused from M12, not reimplemented) -------------------------------


def m11_g16_prototype_indices(model, codebook, symbols: np.ndarray, references: np.ndarray,
                              zero: torch.Tensor, *, device, batch_size: int = 8) -> np.ndarray:
    """Delegates to M12's implementation (scripts/m12_spatial_offline_gate.py) -
    identical need (M11-G16's own codebook index at every TRUE-symbol
    position), so it is reused rather than duplicated."""
    gate = _load_script("m12_spatial_offline_gate")
    return gate.m11_g16_prototype_indices(model, codebook, symbols, references, zero,
                                          device=device, batch_size=batch_size)


# --- per-frame coding: assignment from the ORIGINAL codebook, coding from the recalibrated one --


@torch.no_grad()
def encode_frame_recalibrated(model, assign_codebook, coding_codebook, reference: torch.Tensor,
                              symbols: np.ndarray, zero: torch.Tensor, *, bits: int,
                              timings: dict | None = None) -> tuple[bytes, float]:
    """Identical to `m11_ar_entropy.encode_frame`, except the arithmetic
    coder reads `coding_codebook.cumulative` while symbol-to-prototype
    ASSIGNMENT still comes from `assign_codebook` (the deployed, unchanged
    one) - see the module docstring."""
    ma = _load_script("m11_ar_entropy")
    mk = _load_script("m10k_learned_entropy")
    device = reference.device
    started = time.perf_counter()
    target = torch.from_numpy(np.asarray(symbols, dtype=np.int64))[None].to(device)
    rows = ma._rows(model.log_probabilities(reference, model.planes(target, zero.to(device))))
    if device.type == "cuda":
        torch.cuda.synchronize()
    network = time.perf_counter() - started

    started = time.perf_counter()
    _, table_index = ma._tables_for(rows, assign_codebook, mk)
    tables = time.perf_counter() - started

    flat = np.asarray(symbols, dtype=np.int64).reshape(-1)
    started = time.perf_counter()
    payload = encode_symbols(flat, coding_codebook.cumulative, table_index)
    coder = time.perf_counter() - started
    if timings is not None:
        for key, value in (("network", network), ("tables", tables), ("coder", coder)):
            timings[key] = timings.get(key, 0.0) + value

    totals = coding_codebook.cumulative[:, -1]
    lower = coding_codebook.cumulative[table_index, flat]
    upper = coding_codebook.cumulative[table_index, flat + 1]
    ideal = float(-np.log2((upper - lower) / totals[table_index]).sum())
    return payload, ideal


@torch.no_grad()
def decode_frame_recalibrated(model, assign_codebook, coding_codebook, payload: bytes,
                              reference: torch.Tensor, zero: torch.Tensor, *, bits: int,
                              shape: tuple[int, int, int], timings: dict | None = None) -> np.ndarray:
    """Inverse of `encode_frame_recalibrated`. Decodes through M12's
    `ResumableDecoder` (proving Phase F's "resumable decoder works with both
    old and new tables" by construction, and getting G16's coder speedup for
    free) rather than the legacy prefix-redecoding path."""
    ma = _load_script("m11_ar_entropy")
    mk = _load_script("m10k_learned_entropy")
    channels, height, width = shape
    plane = height * width
    group = model.group_size
    device = reference.device
    zero_d = zero.to(device)
    decoded = zero_d.view(channels, 1, 1).expand(channels, height, width).clone()
    stage = {"network": 0.0, "tables": 0.0, "coder": 0.0}
    symbols = np.empty(channels * plane, dtype=np.int64)

    decoder = ResumableDecoder(payload)
    for start in range(0, channels, group):
        stop = start + group
        began = time.perf_counter()
        log_probabilities = model.log_probabilities(reference, model.planes(decoded[None], zero_d))
        rows = ma._rows(log_probabilities[:, start:stop])
        if device.type == "cuda":
            torch.cuda.synchronize()
        stage["network"] += time.perf_counter() - began

        began = time.perf_counter()
        _, group_table_index = ma._tables_for(rows, assign_codebook, mk)
        stage["tables"] += time.perf_counter() - began

        began = time.perf_counter()
        group_symbols = decoder.decode_group(coding_codebook.cumulative, group_table_index)
        stage["coder"] += time.perf_counter() - began

        symbols[start * plane:stop * plane] = group_symbols
        decoded[start:stop] = torch.from_numpy(
            group_symbols.reshape(group, height, width)).to(device)
    decoder.close()

    if timings is not None:
        for key, value in stage.items():
            timings[key] = timings.get(key, 0.0) + value
    return symbols
