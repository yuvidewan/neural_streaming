"""M10L - shared entropy-table codebook for the M10K learned entropy model.

WHY THIS EXISTS
---------------
M10K predicts an independent probability distribution at every one of the
64 x 16 x 16 = 16,384 latent positions and hands the arithmetic coder 16,384
frequency tables. That works, and it wins ~1% of residual bytes over M10J -
but profiling showed the *network* costs 0.28 ms/P-frame while building those
16,384 integer frequency tables costs 4.5-19.3 ms. The model is cheap; its
probability representation is not.

M10L asks one question: can a small SHARED codebook of K tables approximate
M10K's per-position distributions closely enough to keep the rate gain, while
making per-frame table construction disappear?

THE IDEA
--------
The K prototype tables are fitted ONCE, offline, on TRAIN data, and quantized
ONCE into the coder's integer format. At encode time the per-frame work is:

    z_ref -> network -> probabilities -> nearest prototype -> table_index

`table_index` then selects among K precomputed tables instead of 16,384
freshly-built ones. No per-frame float->int conversion, no per-frame cumsum,
no per-frame allocation of a 4.3 MB cumulative array.

Nothing about the bitstream format, the coder, the symbols, the motion or the
reconstruction changes. This is purely a cheaper way to say the same thing.

ASSIGNMENT METRIC
-----------------
A predicted distribution p is assigned to the prototype q that minimises the
EXPECTED CODE LENGTH of a symbol drawn from p:

    cost(p, q) = sum_s p(s) * -log2 q(s)                      [cross-entropy]

This is the quantity that actually costs bits, and it is what the milestone
brief calls "additional ideal code length". Two useful consequences:

  * Minimising cross-entropy H(p, q) is IDENTICAL to minimising KL(p || q),
    because KL(p||q) = H(p,q) - H(p) and H(p) does not depend on q. So the
    "kl" and "code_length" metrics give the same assignment, and a test pins
    that equivalence rather than leaving it as a claim.
  * It is a matrix product P @ (-log2 Q)^T, so the whole assignment is one
    GEMM - which is why the cheap representation is also the fast one. An L1
    metric is provided for comparison; it is not a matmul and is measurably
    slower.

CLUSTERING
----------
Lloyd's algorithm under the same cross-entropy assignment. The centroid that
minimises sum_i KL(p_i || q) over a cluster is the arithmetic mean of the
p_i, so the update step is just a mean - no line search, no gradient, no
learned codebook. Initialisation is deterministic k-means++ with a fixed seed.

DETERMINISM
-----------
Encoder and decoder must derive the SAME table_index from the SAME z_ref with
no side information. Two places could go wrong and both are pinned:

  * the argmin tie-break - resolved to the LOWEST prototype index by an
    integer reduction, not by `argmin`, whose CUDA tie behaviour is
    unspecified;
  * the float->int quantization of prototypes - done once at fit time and
    stored, so encode and decode read identical integers rather than
    re-deriving them.

PROVENANCE
----------
M10K established that a learned entropy model is silently coupled to the
quantization calibration it was fitted under (a mismatched grid cost 14.8%
rather than raising). A codebook adds a second such coupling. `codebook_id()`
therefore hashes the M10K weights, the calibration signature, the bit depth,
K, the metric AND the prototype frequencies together, and that 8-byte digest
is what goes in the .nvct v2 residual-entropy-model field - so a stale
codebook is a stream identity mismatch, not a silent rate loss.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.entropy_model import TOTAL_FREQUENCY, EmpiricalEntropyModel

CANDIDATE_K = (16, 32, 64, 128, 256, 512)
DEFAULT_METRIC = "code_length"
METRICS = ("code_length", "kl", "l1")
DEFAULT_SEED = 42
DEFAULT_MAX_ITERATIONS = 25
CODEBOOK_VERSION = 1
CODEBOOK_NAME = "m10l_shared_prototype_v1"

# Guards log2 of an exactly-zero prototype. Prototypes come from
# `probabilities_to_frequencies`, which floors every entry at MIN_FREQUENCY, so
# this floor is never reached in the deployed path - it only keeps the float
# helpers total on hand-constructed inputs.
_PROBABILITY_EPSILON = 1e-300


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- prototype assignment ------------------------------------------------------


def prototype_costs(probabilities: np.ndarray, prototypes: np.ndarray, *,
                    metric: str = DEFAULT_METRIC) -> np.ndarray:
    """[N, A] distributions x [K, A] prototypes -> [N, K] assignment cost.

    `code_length` and `kl` produce the SAME argmin - they differ by H(p), a
    per-row constant. `kl` returns the divergence itself so it can be reported
    in bits; `code_length` returns the expected code length, which is the
    number that turns into actual bytes.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prototypes = np.asarray(prototypes, dtype=np.float64)
    if probabilities.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("probabilities and prototypes must both be 2-D [rows, alphabet]")
    if probabilities.shape[1] != prototypes.shape[1]:
        raise ValueError(
            f"alphabet mismatch: {probabilities.shape[1]} vs {prototypes.shape[1]}")

    if metric == "l1":
        return np.abs(probabilities[:, None, :] - prototypes[None, :, :]).sum(axis=2)

    costs = probabilities @ (-np.log2(np.maximum(prototypes, _PROBABILITY_EPSILON))).T
    if metric == "kl":
        safe = np.maximum(probabilities, _PROBABILITY_EPSILON)
        entropy = -(probabilities * np.log2(safe)).sum(axis=1, keepdims=True)
        return costs - entropy
    return costs


def argmin_lowest_index(costs: np.ndarray) -> np.ndarray:
    """argmin along axis 1, ties resolved to the LOWEST index, deterministically.

    `np.argmin` already does this, but the deployed path runs on CUDA where
    torch's tie behaviour is unspecified - and encoder and decoder disagreeing
    on a single tie would corrupt the stream. Reducing over INDICES rather than
    over values makes the answer independent of reduction order on every
    backend, so both paths use the same rule.
    """
    costs = np.asarray(costs)
    minimum = costs.min(axis=1, keepdims=True)
    index = np.where(costs == minimum, np.arange(costs.shape[1])[None, :], costs.shape[1])
    return index.min(axis=1).astype(np.int64)


def _torch_argmin_lowest_index(costs: torch.Tensor) -> torch.Tensor:
    """Device-side twin of `argmin_lowest_index`, and bit-identical to it.

    `argmin` already returns the lowest index on CPU, and its CUDA tie behaviour
    only matters on rows that actually tie - which are rare. So the fast path is
    a plain `argmin`, and the explicit lowest-index reduction runs only on tied
    rows. Doing that reduction over the whole matrix would allocate an [N, K]
    int64 tensor (67 MB at K=512, on every frame), which is exactly the kind of
    cost this milestone exists to remove.
    """
    minimum = costs.min(dim=1, keepdim=True).values
    index = costs.argmin(dim=1)
    is_minimum = costs == minimum
    tied = torch.nonzero(is_minimum.sum(dim=1) > 1, as_tuple=False).flatten()
    if tied.numel():
        rows = is_minimum[tied]
        columns = torch.arange(costs.shape[1], device=costs.device).expand_as(rows)
        index[tied] = torch.where(rows, columns, costs.shape[1]).min(dim=1).values
    return index


# --- the codebook --------------------------------------------------------------


class SharedCodebook:
    """K prototype distributions, pre-quantized into coder frequency tables.

    Holds both representations on purpose. The integer `frequencies` are what
    the coder consumes; the float probabilities used for assignment are derived
    FROM those integers, not from the pre-rounding prototypes - so assignment
    scores the table the coder will actually use.
    """

    def __init__(self, frequencies: np.ndarray, *, bits: int,
                 metric: str = DEFAULT_METRIC,
                 provenance: dict[str, Any] | None = None) -> None:
        frequencies = np.asarray(frequencies, dtype=np.int64)
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
        # EmpiricalEntropyModel enforces the two coder invariants for us - every
        # row sums to exactly TOTAL_FREQUENCY, no entry below MIN_FREQUENCY - so
        # an invalid codebook cannot be constructed at all, rather than failing
        # later inside the coder with a less obvious message.
        self.entropy_model = EmpiricalEntropyModel(frequencies, bits=bits)
        self.bits = bits
        self.metric = metric
        self.frequencies = self.entropy_model.frequencies
        self.cumulative = self.entropy_model.cumulative
        self.provenance = dict(provenance or {})
        self.probabilities = self.frequencies / float(TOTAL_FREQUENCY)
        self._log2_costs = -np.log2(self.probabilities)
        self._torch_cache: dict[Any, torch.Tensor] = {}

    @property
    def size(self) -> int:
        return int(self.frequencies.shape[0])

    @property
    def alphabet(self) -> int:
        return int(self.frequencies.shape[1])

    def table_memory_bytes(self) -> int:
        """Bytes the coder-facing tables occupy: frequencies plus cumulative."""
        return int(self.frequencies.nbytes + self.cumulative.nbytes)

    def assign(self, probabilities: np.ndarray) -> np.ndarray:
        """[N, A] predicted distributions -> [N] prototype indices (numpy path)."""
        costs = prototype_costs(probabilities, self.probabilities, metric=self.metric)
        return argmin_lowest_index(costs)

    def assign_tensor(self, probabilities: torch.Tensor) -> np.ndarray:
        """Same assignment, computed where the network already is.

        This is the deployment path: one GEMM against a [A, K] constant, an
        integer reduction for the tie-break, and a single [N] transfer back -
        replacing M10K's per-frame [N, A] float->int table construction.
        """
        if self.metric == "l1":
            prototypes = self._torch_prototypes(probabilities, "probabilities")
            costs = (probabilities[:, None, :] - prototypes[None, :, :]).abs().sum(dim=2)
        else:
            costs = probabilities @ self._torch_prototypes(probabilities, "log2").T
        return _torch_argmin_lowest_index(costs).to("cpu").numpy().astype(np.int64)

    def _torch_prototypes(self, like: torch.Tensor, kind: str) -> torch.Tensor:
        key = (kind, str(like.device), like.dtype)
        if key not in self._torch_cache:
            source = self._log2_costs if kind == "log2" else self.probabilities
            self._torch_cache[key] = torch.as_tensor(
                source, dtype=like.dtype, device=like.device)
        return self._torch_cache[key]

    def expected_bits(self, symbols: np.ndarray, table_index: np.ndarray) -> float:
        """Ideal code length of `symbols` under the assigned prototypes, in bits."""
        return float(-np.sum(np.log2(self.probabilities[table_index, symbols])))

    def codebook_id(self, *, model_identity: bytes = b"",
                    calibration_signature: str = "") -> bytes:
        """8-byte identity for the .nvct v2 residual-entropy-model field.

        Binds the codebook to the M10K weights AND the quantization calibration
        it was fitted under, so a stale combination is rejected by the container
        check that already exists instead of silently costing bits.
        """
        digest = hashlib.sha256()
        digest.update(json.dumps({
            "name": CODEBOOK_NAME, "version": CODEBOOK_VERSION, "bits": self.bits,
            "size": self.size, "metric": self.metric,
            "model_identity": bytes(model_identity).hex(),
            "calibration_signature": calibration_signature,
        }, sort_keys=True).encode("utf-8"))
        digest.update(self.frequencies.tobytes())
        return digest.digest()[:8]

    def to_dict(self) -> dict[str, Any]:
        return {"name": CODEBOOK_NAME, "version": CODEBOOK_VERSION, "bits": self.bits,
                "metric": self.metric, "size": self.size,
                "frequencies": self.frequencies.tolist(), "provenance": self.provenance}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SharedCodebook":
        if data.get("version") != CODEBOOK_VERSION:
            raise ValueError(
                f"unsupported codebook version {data.get('version')!r}; "
                f"this build understands version {CODEBOOK_VERSION}")
        return cls(np.array(data["frequencies"], dtype=np.int64), bits=int(data["bits"]),
                   metric=data.get("metric", DEFAULT_METRIC),
                   provenance=data.get("provenance"))


# --- the zero-loss reference ---------------------------------------------------


def identity_codebook(probabilities: np.ndarray, *, bits: int,
                      metric: str = DEFAULT_METRIC) -> tuple[SharedCodebook, np.ndarray]:
    """K = N: every position gets its own table, mapped to itself.

    This is M10K expressed through the M10L plumbing, and it exists to prove
    the codebook path is not quietly doing something different. With this
    codebook the emitted bytes MUST be identical to M10K's, because the tables
    come from the same `probabilities_to_frequencies` and `table_index` is the
    same `arange`. If that ever fails, no K-value result from this path can be
    trusted.
    """
    mk = _load_script("m10k_learned_entropy")
    frequencies = mk.probabilities_to_frequencies(np.asarray(probabilities, dtype=np.float64))
    codebook = SharedCodebook(frequencies, bits=bits, metric=metric,
                              provenance={"kind": "identity_reference"})
    return codebook, np.arange(frequencies.shape[0], dtype=np.int64)


# --- fitting -------------------------------------------------------------------


DEFAULT_CHUNK_ROWS = 50_000


def _accumulate_cluster_sums(chunk: np.ndarray, assignment: np.ndarray, size: int,
                             totals: np.ndarray, counts: np.ndarray) -> None:
    """Add one chunk into the running per-cluster sums.

    One bincount per symbol; `np.add.at` is correct here too but an order of
    magnitude slower at these row counts.
    """
    counts += np.bincount(assignment, minlength=size)
    for column in range(chunk.shape[1]):
        totals[:, column] += np.bincount(
            assignment, weights=chunk[:, column], minlength=size)


def _assign_chunked(samples: np.ndarray, prototypes: np.ndarray, *, metric: str,
                    chunk_rows: int = DEFAULT_CHUNK_ROWS):
    """Assign every row to its nearest prototype without materialising [N, K].

    At the gate's sample counts a full cost matrix would be gigabytes for the
    larger K, so the fit walks the rows in chunks. The result is identical to
    the unchunked computation - each row's cost depends only on that row.
    """
    count, size = samples.shape[0], prototypes.shape[0]
    if metric == "l1":
        # L1 has no matmul form: it broadcasts to [rows, K, alphabet]. Shrink the
        # chunk so that intermediate stays bounded regardless of K.
        chunk_rows = max(1, min(chunk_rows, 4_000_000 // max(1, size * samples.shape[1])))
    assignment = np.empty(count, dtype=np.int64)
    own_cost = np.empty(count, dtype=np.float64)
    totals = np.zeros((size, samples.shape[1]), dtype=np.float64)
    counts = np.zeros(size, dtype=np.int64)
    for start in range(0, count, chunk_rows):
        chunk = samples[start:start + chunk_rows]
        costs = prototype_costs(chunk, prototypes, metric=metric)
        picked = argmin_lowest_index(costs)
        assignment[start:start + chunk.shape[0]] = picked
        own_cost[start:start + chunk.shape[0]] = costs[np.arange(chunk.shape[0]), picked]
        _accumulate_cluster_sums(chunk, picked, size, totals, counts)
    return assignment, own_cost, totals, counts


def _seed_prototypes(samples: np.ndarray, size: int, *, metric: str,
                     seed: int) -> np.ndarray:
    """Deterministic k-means++ seeding under the assignment metric."""
    seed_metric = "l1" if metric == "l1" else "kl"
    rng = np.random.RandomState(seed)
    count = samples.shape[0]
    chosen = [int(rng.randint(count))]
    closest = prototype_costs(samples, samples[chosen[0]:chosen[0] + 1],
                              metric=seed_metric)[:, 0]
    while len(chosen) < size:
        weights = np.maximum(closest, 0.0)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            # Every remaining sample is already represented exactly: this is the
            # "K larger than the number of distinct distributions" case. Fill
            # the rest deterministically instead of sampling from a degenerate
            # distribution - duplicated prototypes are harmless (they simply go
            # unused) and must not raise.
            taken = set(chosen)
            for index in range(count):
                if len(chosen) >= size:
                    break
                if index not in taken:
                    chosen.append(index)
            while len(chosen) < size:
                chosen.append(chosen[0])
            break
        pick = int(np.searchsorted(np.cumsum(weights / total), rng.random_sample()))
        chosen.append(min(pick, count - 1))
        closest = np.minimum(closest, prototype_costs(
            samples, samples[chosen[-1]:chosen[-1] + 1], metric=seed_metric)[:, 0])
    return samples[np.array(chosen[:size], dtype=np.int64)].copy()


def fit_codebook(samples: np.ndarray, size: int, *, bits: int,
                 metric: str = DEFAULT_METRIC, seed: int = DEFAULT_SEED,
                 max_iterations: int = DEFAULT_MAX_ITERATIONS,
                 chunk_rows: int = DEFAULT_CHUNK_ROWS,
                 provenance: dict[str, Any] | None = None) -> SharedCodebook:
    """Lloyd's algorithm over predicted distributions, then quantize once.

    The centroid must match the metric, or the iteration is not descent:

      * under KL / expected code length, the minimiser of sum_i KL(p_i || q)
        over a cluster is the arithmetic MEAN of its members;
      * under L1 it is the component-wise MEDIAN (renormalised, since a median
        of probability vectors need not sum to one).

    Using the mean for both would quietly handicap the L1 arm and make the
    metric comparison meaningless, so each gets its own update step.

    Empty clusters are re-seeded to the worst-represented samples,
    deterministically by cost rank - which is also what makes
    K > (number of distinct distributions) safe.
    """
    mk = _load_script("m10k_learned_entropy")
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(f"expected [rows, alphabet] samples, got {samples.shape}")
    if size < 1:
        raise ValueError(f"codebook size must be >= 1, got {size}")
    if samples.shape[0] < 1:
        raise ValueError("cannot fit a codebook on zero samples")
    if samples.shape[1] != 2 ** bits:
        raise ValueError(
            f"alphabet {samples.shape[1]} does not match {bits}-bit symbols")
    samples = np.maximum(samples, 0.0)
    samples = samples / samples.sum(axis=1, keepdims=True)

    iterations, occupied_count = 0, min(size, samples.shape[0])
    if size >= samples.shape[0]:
        # Nothing to cluster: keep every sample, pad by repeating the first.
        prototypes = samples.copy()
        if size > samples.shape[0]:
            prototypes = np.concatenate(
                [prototypes, np.repeat(samples[:1], size - samples.shape[0], axis=0)], axis=0)
    else:
        prototypes = _seed_prototypes(samples, size, metric=metric, seed=seed)
        assignment = None
        for iterations in range(1, max_iterations + 1):
            updated, own_cost, totals, counts = _assign_chunked(
                samples, prototypes, metric=metric, chunk_rows=chunk_rows)
            if assignment is not None and np.array_equal(updated, assignment):
                break
            assignment = updated
            occupied = counts > 0
            if metric == "l1":
                for cluster in np.nonzero(occupied)[0]:
                    median = np.median(samples[assignment == cluster], axis=0)
                    total = float(median.sum())
                    prototypes[cluster] = (median / total if total > 0.0
                                           else np.full(samples.shape[1],
                                                        1.0 / samples.shape[1]))
            else:
                prototypes[occupied] = totals[occupied] / counts[occupied][:, None]
            empty = np.nonzero(~occupied)[0]
            if empty.size:
                # Deterministic re-seed: the samples currently paying the most
                # under their own prototype, worst first, ties by row index.
                prototypes[empty] = samples[
                    np.argsort(-own_cost, kind="stable")[:empty.size]]
        occupied_count = int(np.bincount(assignment, minlength=size).astype(bool).sum())

    prototypes = np.maximum(prototypes, 0.0)
    prototypes = prototypes / prototypes.sum(axis=1, keepdims=True)
    record = {"kind": "fitted", "size": size, "metric": metric, "seed": seed,
              "iterations": int(iterations), "occupied_clusters": occupied_count,
              "training_rows": int(samples.shape[0]), "split": "train"}
    record.update(provenance or {})
    return SharedCodebook(mk.probabilities_to_frequencies(prototypes), bits=bits,
                          metric=metric, provenance=record)


# --- per-frame deployment path -------------------------------------------------


@torch.no_grad()
def frame_probabilities(model, reference: torch.Tensor) -> torch.Tensor:
    """z_ref -> [C*H*W, A] predicted distributions, in the coder's symbol order.

    C-major to match `latent_to_symbols`, exactly as M10K's `frame_entropy_model`
    orders its tables, so row i is the distribution for flat symbol i.
    """
    probabilities = model.log_probabilities(reference).exp()[0]
    channels, alphabet, height, width = probabilities.shape
    return probabilities.permute(0, 2, 3, 1).reshape(channels * height * width, alphabet)


@torch.no_grad()
def frame_table_index(model, codebook: SharedCodebook, reference: torch.Tensor) -> np.ndarray:
    """The whole M10L per-frame entropy step: z_ref -> table_index."""
    return codebook.assign_tensor(frame_probabilities(model, reference))


def unique_tables_used(table_index: np.ndarray) -> int:
    return int(np.unique(np.asarray(table_index)).size)


# --- selecting K and the metric, on VALIDATION only ----------------------------


def select_codebook_size(candidates):
    """Pick one candidate from a rate point's VALIDATION frontier.

    The rule: among candidates meeting BOTH pre-registered criteria, take the
    one with the lowest held-out cost; break ties toward the smaller codebook.

    Deliberately not "the smallest K that passes". The runtime criterion is a
    threshold that has already been met, so spending the remaining headroom on
    rate is what the codec is for - and the measured frontier shows the smallest
    passing K is not even the fastest, so that rule would have given up rate for
    nothing. Returns None when nothing passes, which is a real answer.
    """
    passing = [row for row in candidates if row.get("passes_gate")]
    if not passing:
        return None
    return min(passing, key=lambda row: (round(row["held_out_bits_per_symbol"], 9),
                                         row["codebook_size"]))


def select_metric(candidates):
    """Pick the assignment metric on VALIDATION, at the rate point where the
    alternatives were actually measured against each other.

    Falls back to the default when only one metric was evaluated, and when no
    metric produced a passing candidate - there is then nothing to choose
    between, and the caller reports that rather than pretending otherwise.
    """
    metrics = sorted({row["metric"] for row in candidates})
    if len(metrics) < 2:
        return metrics[0] if metrics else DEFAULT_METRIC
    scored = []
    for metric in metrics:
        best = select_codebook_size([r for r in candidates if r["metric"] == metric])
        if best is not None:
            scored.append((round(best["held_out_bits_per_symbol"], 9), metric))
    return min(scored)[1] if scored else DEFAULT_METRIC


def sample_training_distributions(model, references, *, device, max_rows: int,
                                  batch_size: int = 8, seed: int = DEFAULT_SEED):
    """Predicted distributions from TRAIN reference latents, for codebook fitting.

    Rows are subsampled with a fixed stride per frame rather than randomly: the
    full set is ~8.8M rows per rate point, and a deterministic stride keeps the
    fit reproducible and the memory bounded without needing an RNG in the path.
    """
    if max_rows < 1:
        raise ValueError(f"max_rows must be >= 1, got {max_rows}")
    frames = len(references)
    if frames == 0:
        raise ValueError("no reference frames supplied")
    per_frame = max(1, max_rows // frames)
    collected = []
    stacked = torch.from_numpy(np.stack(references)).float()
    with torch.no_grad():
        for start in range(0, frames, batch_size):
            batch = stacked[start:start + batch_size].to(device)
            probabilities = model.log_probabilities(batch).exp()
            batch_size_actual, channels, alphabet, height, width = probabilities.shape
            flat = probabilities.permute(0, 1, 3, 4, 2).reshape(
                batch_size_actual, channels * height * width, alphabet)
            rows = flat.shape[1]
            stride = max(1, rows // per_frame)
            collected.append(flat[:, ::stride, :].reshape(-1, alphabet).double().cpu().numpy())
    samples = np.concatenate(collected, axis=0)
    return samples[:max_rows] if samples.shape[0] > max_rows else samples
