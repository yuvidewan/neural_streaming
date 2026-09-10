"""M11 - causal residual context: scan order, dependency graph, contexts, estimators.

THE QUESTION
------------
M10K/M10L model P(R_i | z_ref, channel). M11 asks whether residual symbols that
the decoder has ALREADY decoded carry information about R_i that z_ref does not
- enough to justify making the probability model depend on the decoding order.

SCAN ORDER (audited, not assumed)
---------------------------------
`nvc.compression.codec.latent_to_symbols` flattens a [1, C, H, W] quantized
latent with `reshape(-1)`, and the arithmetic coder consumes that array
strictly in order. The coding order is therefore C-MAJOR RASTER:

    flat index i = c * H * W + y * W + x        (C=64, H=W=16 here)

so every symbol of channel 0 is decoded before any of channel 1, and within a
channel rows go top to bottom, columns left to right.

CAUSAL DEPENDENCY GRAPH
-----------------------
When symbol (c, y, x) is decoded, the decoder already holds exactly:

    (c', y', x')  for every c' < c                  - ALL of every earlier channel
    (c,  y', x')  for every y' < y                  - every earlier row
    (c,  y,  x')  for every x' < x                  - earlier columns of this row

Two consequences the context design depends on:

  * within channel c, left (y, x-1), up (y-1, x), up-left (y-1, x-1) AND
    up-right (y-1, x+1) are all available - the whole row above is done;
  * EVERY position of every earlier channel is available, including positions
    that are spatially "ahead" (right of / below) the current one. Channel
    context is therefore not limited to a causal half-plane.

Anything else - the current symbol, later positions of this channel, any
position of a later channel - is future and invalid. `check_causality` proves
this for each context function by perturbing every symbol at flat index >= i
and requiring the context of symbol i to be unchanged; a deliberately leaky
context is included in the tests and must be caught.

This matters for more than correctness. A context that only reads EARLIER
CHANNELS lets every position of channel c be predicted in parallel once channel
c-1 is decoded: 64 sequential steps. A context that reads the same channel's
left/up neighbours forces a sequential dependency between neighbouring
positions: up to 16,384 steps. The gate measures the two families separately
so that trade-off is visible before anything is built.

MAGNITUDE
---------
Contexts are built from residual MAGNITUDE, |s - z_c|, where z_c is the symbol
a zero residual maps to in channel c under the deployed calibration - computed
from the quantizer, so it is exact and known to the decoder. Magnitudes are
clipped at 3 ({0, 1, 2, 3+}) plus an explicit NOT-AVAILABLE level at frame and
channel boundaries, so boundary symbols are never silently treated as zeros.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

MAGNITUDE_CLIP = 3
MAGNITUDE_LEVELS = MAGNITUDE_CLIP + 1          # {0, 1, 2, 3+}
NOT_AVAILABLE = MAGNITUDE_LEVELS               # its own level, never confused with 0
NEIGHBOUR_LEVELS = MAGNITUDE_LEVELS + 1        # magnitude levels + NA
NEIGHBOURHOOD_BUCKETS = (0, 2, 5)              # sum of 4 clipped mags -> {0,1-2,3-5,6+}
CHANNEL_ACTIVITY_BUCKETS = (0.0, 0.15, 0.5)    # mean prev-channel mag -> 4 levels (+NA)
SMOOTHING_GRID = (1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0,
                  262144.0, 1048576.0)
CONTEXT_VERSION = 1


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- symbol geometry ------------------------------------------------------------


def coding_index(channel: int, row: int, column: int, height: int, width: int) -> int:
    """Flat position of (c, y, x) in the coder's C-major raster order."""
    return (channel * height + row) * width + column


def zero_symbols(residual_params, channels: int) -> np.ndarray:
    """Per-channel symbol that a ZERO residual quantizes to.

    Derived by quantizing an all-zero latent with the deployed parameters, so it
    is exactly what the coder would emit and is known to the decoder from the
    stream's own quantization block.
    """
    from nvc.compression.codec import latent_to_symbols

    zeros = torch.zeros(1, channels, 1, 1)
    return latent_to_symbols(zeros, residual_params).reshape(channels).astype(np.int64)


def magnitudes(symbols: np.ndarray, zero: np.ndarray) -> np.ndarray:
    """[C, H, W] symbols -> clipped residual magnitudes in {0, .., MAGNITUDE_CLIP}."""
    return np.minimum(np.abs(symbols - zero[:, None, None]), MAGNITUDE_CLIP)


def _shift(values: np.ndarray, *, dc: int = 0, dy: int = 0, dx: int = 0,
           fill: int = NOT_AVAILABLE) -> np.ndarray:
    """values[c - dc, y - dy, x - dx], with `fill` where that index is outside.

    Positive offsets look backwards (earlier channel / row above / column to the
    left); dx = -1 with dy = 1 is the up-right neighbour.
    """
    out = np.full_like(values, fill)
    channels, height, width = values.shape
    c0, y0, x0 = max(dc, 0), max(dy, 0), max(dx, 0)
    c1, y1, x1 = channels + min(dc, 0), height + min(dy, 0), width + min(dx, 0)
    out[c0:c1, y0:y1, x0:x1] = values[c0 - dc:c1 - dc, y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


# --- the candidate contexts -----------------------------------------------------
#
# Every function maps (symbols [C, H, W], zero [C], alphabet) -> int context
# array [C, H, W] with values in [0, cardinality). Each is verified causal by
# `check_causality` in the gate and in the tests.


def ctx_left(symbols, zero, alphabet):
    return _shift(magnitudes(symbols, zero), dx=1)


def ctx_up(symbols, zero, alphabet):
    return _shift(magnitudes(symbols, zero), dy=1)


def ctx_left_up(symbols, zero, alphabet):
    m = magnitudes(symbols, zero)
    return _shift(m, dx=1) * NEIGHBOUR_LEVELS + _shift(m, dy=1)


def ctx_left_up_upleft(symbols, zero, alphabet):
    m = magnitudes(symbols, zero)
    return ((_shift(m, dx=1) * NEIGHBOUR_LEVELS + _shift(m, dy=1)) * NEIGHBOUR_LEVELS
            + _shift(m, dy=1, dx=1))


def _neighbourhood_sum(m):
    total = np.zeros_like(m)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):   # left, up, up-left, up-right
        shifted = _shift(m, dy=dy, dx=dx, fill=0)       # an absent neighbour adds 0
        total += shifted
    return total


def ctx_neighbourhood(symbols, zero, alphabet):
    """Bucketed sum of clipped magnitudes over left, up, up-left, up-right."""
    return np.digitize(_neighbourhood_sum(magnitudes(symbols, zero)),
                       NEIGHBOURHOOD_BUCKETS, right=True)


def ctx_left_raw(symbols, zero, alphabet):
    """The left neighbour's full symbol value - keeps sign, which magnitude drops."""
    return _shift(symbols, dx=1, fill=alphabet)


def ctx_prev_channel(symbols, zero, alphabet):
    """Magnitude at the same position in channel c-1."""
    return _shift(magnitudes(symbols, zero), dc=1)


def _prev_channel_activity(m):
    """Mean clipped magnitude over channels 0..c-1 at the same position."""
    cumulative = np.cumsum(m, axis=0, dtype=np.float64)
    earlier = np.zeros_like(cumulative)
    earlier[1:] = cumulative[:-1]
    counts = np.arange(m.shape[0], dtype=np.float64)[:, None, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, earlier / np.maximum(counts, 1.0), np.nan)


def ctx_channel_activity(symbols, zero, alphabet):
    """Bucketed running activity of all earlier channels at this position; NA for c=0.

    The latent channel index is an arbitrary learned ordering, so "channel c-1"
    need not be related to channel c. The mean over EVERY earlier channel is the
    robust version of the same idea: "has this position been busy so far?".
    """
    activity = _prev_channel_activity(magnitudes(symbols, zero))
    buckets = np.digitize(np.nan_to_num(activity, nan=0.0), CHANNEL_ACTIVITY_BUCKETS,
                          right=True)
    return np.where(np.isnan(activity), len(CHANNEL_ACTIVITY_BUCKETS) + 1, buckets)


def ctx_neighbourhood_x_channel(symbols, zero, alphabet):
    return (ctx_neighbourhood(symbols, zero, alphabet) * (len(CHANNEL_ACTIVITY_BUCKETS) + 2)
            + ctx_channel_activity(symbols, zero, alphabet))


def ctx_left_up_x_channel(symbols, zero, alphabet):
    return (ctx_left_up(symbols, zero, alphabet) * (len(CHANNEL_ACTIVITY_BUCKETS) + 2)
            + ctx_channel_activity(symbols, zero, alphabet))


CHANNEL_LEVELS = len(CHANNEL_ACTIVITY_BUCKETS) + 2      # 4 buckets + NA
NEIGHBOURHOOD_LEVELS = len(NEIGHBOURHOOD_BUCKETS) + 1

# name -> (function, cardinality(alphabet), family, description)
CONTEXTS: dict[str, tuple[Callable, Callable[[int], int], str, str]] = {
    "left": (ctx_left, lambda a: NEIGHBOUR_LEVELS, "spatial",
             "magnitude of (c, y, x-1)"),
    "up": (ctx_up, lambda a: NEIGHBOUR_LEVELS, "spatial",
           "magnitude of (c, y-1, x)"),
    "left_up": (ctx_left_up, lambda a: NEIGHBOUR_LEVELS ** 2, "spatial",
                "left x up"),
    "left_up_upleft": (ctx_left_up_upleft, lambda a: NEIGHBOUR_LEVELS ** 3, "spatial",
                       "left x up x up-left"),
    "neighbourhood": (ctx_neighbourhood, lambda a: NEIGHBOURHOOD_LEVELS, "spatial",
                      "bucketed sum over left, up, up-left, up-right"),
    "left_raw": (ctx_left_raw, lambda a: a + 1, "spatial",
                 "full symbol of (c, y, x-1), sign included"),
    "prev_channel": (ctx_prev_channel, lambda a: NEIGHBOUR_LEVELS, "channel",
                     "magnitude of (c-1, y, x)"),
    "channel_activity": (ctx_channel_activity, lambda a: CHANNEL_LEVELS, "channel",
                         "mean magnitude over channels < c at (y, x)"),
    "neighbourhood_x_channel": (ctx_neighbourhood_x_channel,
                                lambda a: NEIGHBOURHOOD_LEVELS * CHANNEL_LEVELS, "both",
                                "neighbourhood x channel_activity"),
    "left_up_x_channel": (ctx_left_up_x_channel,
                          lambda a: NEIGHBOUR_LEVELS ** 2 * CHANNEL_LEVELS, "both",
                          "left x up x channel_activity"),
}


def context_definition_id(name: str) -> str:
    """Identity of a context DEFINITION, for binding a deployed model to it.

    Hashes the name, the version and the constants the definition depends on,
    so a model trained under one context cannot silently be run under another.
    """
    function, _, family, description = CONTEXTS[name]
    payload = {"name": name, "version": CONTEXT_VERSION, "family": family,
               "description": description, "clip": MAGNITUDE_CLIP,
               "neighbourhood_buckets": NEIGHBOURHOOD_BUCKETS,
               "channel_buckets": CHANNEL_ACTIVITY_BUCKETS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def compute_context(name: str, symbols: np.ndarray, zero: np.ndarray, alphabet: int):
    function, cardinality, _, _ = CONTEXTS[name]
    values = function(np.asarray(symbols, dtype=np.int64), zero, alphabet).astype(np.int64)
    card = cardinality(alphabet)
    if values.min() < 0 or values.max() >= card:
        raise ValueError(f"context {name} produced values outside [0, {card})")
    return values, card


# --- causality verification -----------------------------------------------------


def check_causality(function: Callable, symbols: np.ndarray, zero: np.ndarray,
                    alphabet: int, *, probes: int = 64, seed: int = 0) -> dict[str, Any]:
    """Prove that context[i] depends only on symbols at flat indices < i.

    For each probe i, every symbol at flat index >= i (the current one and all
    future ones in coding order) is replaced with random values; the context
    at i must not change. A single change is future-context leakage. Probes
    include the first symbol, channel starts and row starts - the boundaries
    where an off-by-one is most likely.
    """
    symbols = np.asarray(symbols, dtype=np.int64)
    channels, height, width = symbols.shape
    total = symbols.size
    rng = np.random.default_rng(seed)
    base = function(symbols, zero, alphabet).reshape(-1)
    boundary = {0, total - 1, width, height * width, (channels // 2) * height * width,
                (channels // 2) * height * width + width - 1}
    positions = sorted(boundary | set(rng.integers(0, total, size=probes).tolist()))
    leaks = []
    for position in positions:
        perturbed = symbols.reshape(-1).copy()
        perturbed[position:] = rng.integers(0, alphabet, size=total - position)
        again = function(perturbed.reshape(symbols.shape), zero, alphabet).reshape(-1)
        if again[position] != base[position]:
            leaks.append(int(position))
    return {"causal": not leaks, "probes": len(positions), "leaking_positions": leaks}


# --- random control ---------------------------------------------------------------


def permuted_context(context: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The same context values, shuffled across EVERY position of the array given.

    The gate passes a whole split (all frames at once). That keeps the split's
    exact context histogram - cardinality and sparsity identical - while
    destroying any alignment with the symbols, so a held-out "gain" is
    estimator bias and nothing else.

    It must NOT be applied frame by frame. Measured on 5-bit validation: a
    within-frame shuffle keeps each frame's context histogram, which is a
    frame-level "how busy is this frame" statistic assembled from positions all
    over the frame - future ones included - and it showed up as a spurious
    +0.147% for the skewed neighbourhood context (+0.061% for left). A global
    shuffle and M10J's uniform-random labels agreed at -0.08% to -0.09%, the
    honest price of adding a context that carries no information.
    """
    flat = context.reshape(-1).copy()
    rng.shuffle(flat)
    return flat.reshape(context.shape)


# --- estimators ---------------------------------------------------------------------


def _bits(probabilities: np.ndarray) -> float:
    return float(-np.log2(np.maximum(probabilities, 1e-300)).sum())


def fit_parent(train_symbols, train_groups, groups: int, alphabet: int):
    """Counts n(g, s) over TRAIN for a grouping g (channel, or M10L prototype)."""
    counts = np.bincount(train_groups * alphabet + train_symbols,
                         minlength=groups * alphabet).reshape(groups, alphabet)
    return counts.astype(np.float64)


def smoothed_parent(counts, prior, strength: float):
    """P(s | g) = (n(g, s) + strength * prior(g, s)) / (n(g) + strength)."""
    return (counts + strength * prior) / (counts.sum(axis=1, keepdims=True) + strength)


def fit_child(train_symbols, train_groups, train_context, card: int, groups: int,
              alphabet: int):
    joint = train_groups * card + train_context
    counts = np.bincount(joint * alphabet + train_symbols,
                         minlength=groups * card * alphabet)
    return counts.reshape(groups * card, alphabet).astype(np.float64)


def child_bits(child_counts, parent_probabilities, symbols, groups_index, context, card,
               strength: float) -> float:
    """Held-out bits under P(s | g, ctx) smoothed hierarchically toward P(s | g).

    A context seen rarely in TRAIN falls back smoothly to its parent instead of
    being trusted - the standard remedy for the plug-in bias that made M10J add
    a random control in the first place.
    """
    joint = groups_index * card + context
    numerator = child_counts[joint, symbols] + strength * parent_probabilities[groups_index,
                                                                                symbols]
    # Row totals once, then gathered: indexing whole rows per symbol would
    # materialise an [N, alphabet] array (hundreds of MB on a validation split).
    denominator = child_counts.sum(axis=1)[joint] + strength
    return _bits(numerator / denominator)


def parent_bits(parent_probabilities, symbols, groups_index) -> float:
    return _bits(parent_probabilities[groups_index, symbols])


def select_strength(evaluate: Callable[[float], float], grid=SMOOTHING_GRID):
    """Pick the smoothing strength on the SELECTION split; returns (strength, bits)."""
    scores = [(evaluate(strength), strength) for strength in grid]
    bits, strength = min(scores)
    return strength, bits
