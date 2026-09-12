"""M15 Phase A/B - calibration-policy functions, kept entirely separate from
the deployed `m10h_motion_compensation.calibrate_grids`. Nothing here is
imported by production code; these are candidate SEQUENCE-SELECTION
strategies only, tested offline before any change to what is deployed.

Every policy takes the FULL, untruncated list of TRAIN `BenchmarkSequence`
objects (`nvc.evaluation.sequences.discover_sequences(manifest,
split="train")`, 72 sequences in M15's DAVIS split) and returns a NEW list
of (possibly frame_paths-truncated) `BenchmarkSequence` objects whose
frame_paths are unmodified copies of a PREFIX of the original - never a
different frame selection within a sequence, never a mutation of the input.
Downstream, `m14_recalibration.collect_intra_symbols` /
`collect_motion_symbols` (both reused unmodified) walk whatever list they
are given exactly as `calibrate_grids` walks its own `sequences` argument -
so the entire policy question reduces to "which list of BenchmarkSequences
gets fed in", never to a change in the collectors, the quantizer, the
motion estimator, or any frozen component.

POLICY A - CURRENT (`sequential_allocation`)
    Reproduces `calibrate_grids`'s own internal walk explicitly: take
    frames from the front of each sequence, in the given (manifest) order,
    until `total_budget` is reached, truncating the boundary sequence
    mid-way. This is not a new policy - it is the deployed one, made
    explicit so its coverage statistics are directly comparable to the
    others' (Phase D) and so it can be fed through the SAME collector
    functions as every other candidate (Phase C).

POLICY B - UNIFORM SEQUENCE COVERAGE (`uniform_allocation`)
    Distributes `total_budget` as evenly as possible across every sequence
    in the given order: `base = total_budget // n`, and the first
    `total_budget % n` sequences (in that order) get one extra frame.
    Deterministic, exact total, every sequence represented whenever
    `base >= 1`.

POLICY C - M14 BROAD RECIPE (`broad_flat_allocation`)
    A flat per-sequence cap (M14 shipped `frames_per_sequence=8`, 72
    sequences = 576 total) - exactly `discover_sequences(...,
    max_frames_per_sequence=K)`'s own truncation rule, reproduced here so
    it shares this module's `BenchmarkSequence`-in/out interface with the
    other three policies.

POLICY D - SHUFFLED GLOBAL COVERAGE (`shuffled_uniform_allocation`)
    Policy B's own allocation rule, applied after a deterministic seeded
    shuffle of sequence order. Isolates whether a result depends on
    manifest ordering (which sequences happen to be first / get the
    remainder) rather than on coverage itself - if this closely matches
    Policy B, order does not matter and coverage alone explains the
    difference from Policy A.
"""

from __future__ import annotations

import dataclasses
import random
import statistics
from typing import Any, Sequence as TypingSequence

from nvc.evaluation.sequences import BenchmarkSequence

DEFAULT_SEED = 42


def sequential_allocation(
    sequences: TypingSequence[BenchmarkSequence], total_budget: int
) -> list[BenchmarkSequence]:
    """Policy A - CURRENT. First `total_budget` frames, manifest order."""
    allocated: list[BenchmarkSequence] = []
    remaining = total_budget
    for sequence in sequences:
        if remaining <= 0:
            break
        take = min(remaining, sequence.frame_count)
        if take > 0:
            allocated.append(dataclasses.replace(sequence, frame_paths=sequence.frame_paths[:take]))
        remaining -= take
    if remaining > 0:
        raise ValueError(
            f"total_budget={total_budget} exceeds the {total_budget - remaining} frames "
            f"available across all {len(sequences)} sequences")
    return allocated


def uniform_allocation(
    sequences: TypingSequence[BenchmarkSequence], total_budget: int, *,
    order: TypingSequence[int] | None = None,
) -> list[BenchmarkSequence]:
    """Policy B - UNIFORM SEQUENCE COVERAGE. `total_budget` spread evenly
    across every sequence in `order` (default: the given order); the first
    `total_budget % n` sequences get one extra frame."""
    ordered = list(sequences) if order is None else [sequences[i] for i in order]
    n = len(ordered)
    if n == 0:
        raise ValueError("no sequences to allocate across")
    base, remainder = divmod(total_budget, n)
    allocated: list[BenchmarkSequence] = []
    for index, sequence in enumerate(ordered):
        take = base + (1 if index < remainder else 0)
        if take > sequence.frame_count:
            raise ValueError(
                f"sequence '{sequence.sequence_id}' has only {sequence.frame_count} frames, "
                f"policy needs {take} (reduce total_budget or its per-sequence share)")
        if take > 0:
            allocated.append(dataclasses.replace(sequence, frame_paths=sequence.frame_paths[:take]))
    return allocated


def broad_flat_allocation(
    sequences: TypingSequence[BenchmarkSequence], frames_per_sequence: int
) -> list[BenchmarkSequence]:
    """Policy C - M14 BROAD RECIPE. A flat per-sequence cap, every sequence
    (M14 shipped `frames_per_sequence=8` -> 576 total across 72 TRAIN
    sequences), matching `discover_sequences(max_frames_per_sequence=...)`."""
    allocated: list[BenchmarkSequence] = []
    for sequence in sequences:
        take = min(frames_per_sequence, sequence.frame_count)
        if take > 0:
            allocated.append(dataclasses.replace(sequence, frame_paths=sequence.frame_paths[:take]))
    return allocated


def shuffled_uniform_allocation(
    sequences: TypingSequence[BenchmarkSequence], total_budget: int, *, seed: int = DEFAULT_SEED,
) -> list[BenchmarkSequence]:
    """Policy D - SHUFFLED GLOBAL COVERAGE. Policy B's allocation rule,
    applied after a deterministic seeded shuffle of sequence order."""
    order = list(range(len(sequences)))
    random.Random(seed).shuffle(order)
    return uniform_allocation(sequences, total_budget, order=order)


def coverage_statistics(
    allocated: TypingSequence[BenchmarkSequence], all_train_sequences: TypingSequence[BenchmarkSequence],
) -> dict[str, Any]:
    """Phase D coverage numbers for one policy's allocation, against the
    full TRAIN population it was drawn from."""
    total_train_sequences = len(all_train_sequences)
    total_train_frames = sum(s.frame_count for s in all_train_sequences)
    counts = [s.frame_count for s in allocated]
    represented = len(counts)
    selected = sum(counts)
    return {
        "sequences_represented": represented,
        "total_train_sequences": total_train_sequences,
        "percent_sequences_covered": represented / total_train_sequences * 100.0,
        "total_frames_selected": selected,
        "total_train_frames": total_train_frames,
        "percent_train_frame_volume_covered": selected / total_train_frames * 100.0,
        "min_frames_per_represented_sequence": min(counts) if counts else 0,
        "max_frames_per_represented_sequence": max(counts) if counts else 0,
        "mean_frames_per_represented_sequence": statistics.fmean(counts) if counts else 0.0,
        "stdev_frames_per_represented_sequence": statistics.pstdev(counts) if len(counts) > 1 else 0.0,
        "represented_sequence_ids": [s.sequence_id for s in allocated],
    }


# Single source of truth for every candidate policy's construction, shared
# by the audit, offline-gate, coded-validation and DAVIS-benchmark scripts
# so none of them can drift out of sync with each other about what a named
# policy actually means.
POLICY_NAMES = ("A_sequential_400", "B_uniform_400", "C_broad_576", "D_shuffled_uniform_400")


def build_policy(
    name: str, train_sequences: TypingSequence[BenchmarkSequence], *, seed: int = DEFAULT_SEED,
    total_budget: int | None = None, frames_per_sequence: int | None = None,
) -> list[BenchmarkSequence]:
    """Dispatch to the named policy using ITS OWN deployed/milestone budget
    (400 total for A/B/D, 8 frames/sequence = 576 total for C) unless
    `total_budget`/`frames_per_sequence` overrides it - the override exists
    so tests can exercise the same dispatch against small synthetic
    populations; production call sites never pass it."""
    if name == "A_sequential_400":
        return sequential_allocation(train_sequences, total_budget if total_budget is not None else 400)
    if name == "B_uniform_400":
        return uniform_allocation(train_sequences, total_budget if total_budget is not None else 400)
    if name == "C_broad_576":
        return broad_flat_allocation(
            train_sequences, frames_per_sequence if frames_per_sequence is not None else 8)
    if name == "D_shuffled_uniform_400":
        return shuffled_uniform_allocation(
            train_sequences, total_budget if total_budget is not None else 400, seed=seed)
    raise ValueError(f"unknown policy '{name}', expected one of {POLICY_NAMES}")
