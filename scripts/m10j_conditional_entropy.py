"""M10J: a reference-conditioned entropy model for the M10H residual symbols.

WHAT CHANGES, AND WHAT DOES NOT
---------------------------------
    M10H:  residual symbol            ->  per-CHANNEL frequency table
    M10J:  residual symbol + context  ->  per-(CHANNEL, CONTEXT) frequency table

Nothing else moves. Same motion estimator, same motion quantization, same warp,
same z_ref, same residual definition, same residual quantization, same
arithmetic coder, same `.nvct` v2 container, same GOP. The residual SYMBOLS are
bit-identical between the two arms; only which probability table codes each one
differs, so reconstruction is identical by construction and any byte difference
is attributable to the probability model alone.

HOW THAT IS GUARANTEED RATHER THAN ASSERTED
---------------------------------------------
`encode_multi` runs the closed loop ONCE per sequence and produces every arm's
payload from the SAME symbol array in the same pass. There is no second motion
search, no second quantization and no second reconstruction that could drift.
The arms cannot differ in anything but their tables, because only the tables are
computed twice.

THE CONTEXT
-------------
Deterministic, causal, and computable by the decoder from `z_ref` alone -
`z_ref = Encoder(Warp(x_hat_{t-1}, decoded motion))`, which the decoder already
holds before it touches the residual payload. Two schemes survived the offline
analysis (see `m10j_entropy_analysis.py`):

    magnitude4        which per-channel |z_ref| quantile band the position is in
    local_activity4   how much z_ref varies in a 3x3 neighbourhood, bucketed

Both use 4 contexts, so the deployed model has 64 x 4 = 256 tables. Thresholds
are per-channel quantiles fitted on TRAINING references only and shipped with
the calibration, exactly like the quantization grids and frequency tables
already are.

WHY NO CONTAINER CHANGE IS NEEDED
-----------------------------------
`.nvct` v2 stores an 8-byte residual entropy model id, and the decoder verifies
it. A 256-table conditional model hashes differently from a 64-table marginal
one, so a marginal decoder handed a conditional stream fails loudly rather than
emitting garbage - and an M10H stream remains decodable with its own declared
model. The context definition travels with the calibration, which is already
out-of-band for every other table in this codec.

Example usage (PowerShell, from the project root, with .venv activated):

    .venv\\Scripts\\python.exe scripts\\m10j_conditional_entropy.py --stage smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.codec import (
    channel_table_index,
    decode_payload_to_latent,
    encode_latent_to_payload,
    latent_to_symbols,
    symbols_to_latent,
)
from nvc.compression.entropy_model import EmpiricalEntropyModel, _counts_to_frequencies
from nvc.compression.range_coder import decode_symbols, encode_symbols
from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device

DEFAULT_OUTPUT_DIR = Path("outputs/m10j_conditional_entropy")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
FROZEN_LAMBDA = 3.0e-4
RATE_POINTS = (5, 4, 3)
DEFAULT_GOP = 10
DEFAULT_BLOCK_SIZE = 16
DEFAULT_SEARCH_RANGE = 16
# Contexts kept for deployment, chosen from the offline analysis: the simplest
# useful reference-value context, and the strongest one. Both cardinality 4.
DEPLOYED_CONTEXTS = ("magnitude4", "local_activity4")
MIN_SAMPLES_PER_CONTEXT = 1000


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- context model ------------------------------------------------------------


class ReferenceContextModel:
    """Maps a reference latent to a per-element context index.

    Deterministic and decoder-computable: it needs only `z_ref` and the
    per-channel thresholds fitted on training data.
    """

    def __init__(self, scheme: str, thresholds: np.ndarray | None, cardinality: int) -> None:
        if scheme not in ("marginal",) + DEPLOYED_CONTEXTS:
            raise ValueError(f"Unknown context scheme {scheme!r}")
        self.scheme = scheme
        self.thresholds = None if thresholds is None else np.asarray(thresholds, dtype=np.float64)
        self.cardinality = cardinality

    def contexts(self, reference: np.ndarray) -> np.ndarray:
        """[C, H, W] reference latent -> [C, H, W] context indices."""
        if self.scheme == "marginal":
            return np.zeros(reference.shape, dtype=np.int64)
        signal = self._signal(reference)
        out = np.empty(reference.shape, dtype=np.int64)
        for channel in range(reference.shape[0]):
            out[channel] = np.searchsorted(self.thresholds[channel], signal[channel])
        return np.clip(out, 0, self.cardinality - 1)

    def _signal(self, reference: np.ndarray) -> np.ndarray:
        if self.scheme == "magnitude4":
            return np.abs(reference)
        tensor = torch.from_numpy(np.ascontiguousarray(reference)).unsqueeze(0).float()
        mean = torch.nn.functional.avg_pool2d(tensor, 3, stride=1, padding=1)
        activity = torch.nn.functional.avg_pool2d((tensor - mean).abs(), 3, stride=1, padding=1)
        return activity[0].numpy()

    def table_index(self, channels: int, height: int, width: int,
                    contexts: np.ndarray) -> np.ndarray:
        """table = channel * cardinality + context, matching the C-major symbol order."""
        channel_index = channel_table_index(channels, height, width)
        return channel_index * self.cardinality + contexts.reshape(-1)

    def to_dict(self) -> dict[str, Any]:
        return {"scheme": self.scheme, "cardinality": self.cardinality,
                "thresholds": None if self.thresholds is None else self.thresholds.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceContextModel":
        thresholds = data.get("thresholds")
        return cls(data["scheme"],
                   None if thresholds is None else np.asarray(thresholds, dtype=np.float64),
                   int(data["cardinality"]))


def fit_context_model(scheme: str, references: list[np.ndarray], *,
                      cardinality: int = 4) -> ReferenceContextModel:
    """Fit per-channel quantile thresholds on TRAINING references only."""
    if scheme == "marginal":
        return ReferenceContextModel("marginal", None, 1)
    probe = ReferenceContextModel(scheme, None, cardinality)
    channels = references[0].shape[0]
    signals = np.concatenate([probe._signal(r).reshape(channels, -1) for r in references], axis=1)
    quantiles = np.linspace(0, 100, cardinality + 1)[1:-1]
    thresholds = np.stack([np.percentile(signals[c], quantiles) for c in range(channels)])
    return ReferenceContextModel(scheme, thresholds, cardinality)


def build_conditional_entropy_model(symbol_frames: list[np.ndarray],
                                    reference_frames: list[np.ndarray],
                                    context_model: ReferenceContextModel, *,
                                    bits: int) -> dict[str, Any]:
    """Count (channel, context) -> symbol on TRAIN data and build coder tables.

    Contexts with too few training samples fall back deterministically to that
    channel's MARGINAL distribution rather than being left as a tiny, unstable
    histogram - a rare context fitted on a handful of samples would cost more
    bits than the marginal table it replaced.
    """
    channels = symbol_frames[0].shape[0]
    alphabet = 2 ** bits
    cardinality = context_model.cardinality
    counts = np.zeros((channels, cardinality, alphabet), dtype=np.int64)

    for symbols, reference in zip(symbol_frames, reference_frames):
        contexts = context_model.contexts(reference)
        flat_symbols = symbols.reshape(channels, -1)
        flat_contexts = contexts.reshape(channels, -1)
        for channel in range(channels):
            index = flat_contexts[channel] * alphabet + flat_symbols[channel]
            counts[channel] += np.bincount(
                index, minlength=cardinality * alphabet).reshape(cardinality, alphabet)

    # Recorded BEFORE the fallback substitution below: replacing a sparse
    # context's counts with the marginal ones would otherwise inflate the
    # reported training-symbol total above the number of symbols actually seen.
    observed_symbols = int(counts.sum())
    marginal = counts.sum(axis=1)
    occupancy = counts.sum(axis=2)
    fallbacks = 0
    for channel in range(channels):
        for context in range(cardinality):
            if occupancy[channel, context] < MIN_SAMPLES_PER_CONTEXT:
                counts[channel, context] = marginal[channel]
                fallbacks += 1

    frequencies = _counts_to_frequencies(counts.reshape(channels * cardinality, alphabet))
    model = EmpiricalEntropyModel(frequencies, bits=bits)
    return {
        "entropy_model": model, "context_model": context_model,
        "provenance": {
            "scheme": context_model.scheme, "cardinality": cardinality,
            "tables": channels * cardinality,
            "training_symbols": observed_symbols,
            "training_frames": len(symbol_frames),
            "fallback_tables": fallbacks,
            "min_samples_per_context": MIN_SAMPLES_PER_CONTEXT,
            "smoothing": "project Laplace smoothing via _counts_to_frequencies",
            "entropy_model_id": model.model_id().hex(),
            "split": "train",
        },
    }


def _ideal_code_bits(symbols: np.ndarray, contexts: np.ndarray, *, shape,
                     entropy_model: EmpiricalEntropyModel,
                     context_model: ReferenceContextModel) -> float:
    """Shannon code length of these symbols under these tables, in bits.

    Computed from the coder's own integer frequency tables (not a re-estimated
    histogram), so it is exactly the cost the arithmetic coder is approximating.
    """
    channels, height, width = shape
    table = context_model.table_index(channels, height, width, contexts)
    frequencies = entropy_model.frequencies.astype(np.float64)
    probabilities = frequencies / frequencies.sum(axis=1, keepdims=True)
    return float(-np.log2(probabilities[table, symbols.reshape(-1)]).sum())


def encode_residual_symbols(symbols: np.ndarray, contexts: np.ndarray, *, shape,
                            entropy_model: EmpiricalEntropyModel,
                            context_model: ReferenceContextModel) -> bytes:
    channels, height, width = shape
    table = context_model.table_index(channels, height, width, contexts)
    return encode_symbols(symbols.reshape(-1), entropy_model.cumulative, table)


def decode_residual_symbols(payload: bytes, contexts: np.ndarray, *, shape,
                            entropy_model: EmpiricalEntropyModel,
                            context_model: ReferenceContextModel) -> np.ndarray:
    channels, height, width = shape
    table = context_model.table_index(channels, height, width, contexts)
    return decode_symbols(payload, channels * height * width, entropy_model.cumulative, table)


# --- multi-arm coding: one closed loop, many entropy models -------------------


@torch.no_grad()
def encode_multi(mc, model, frames, arms: dict[str, dict[str, Any]], paths: dict[str, Path], *,
                 intra_params, intra_entropy_model, residual_params, motion_entropy_model,
                 gop_size=DEFAULT_GOP, block_size=DEFAULT_BLOCK_SIZE,
                 search_range=DEFAULT_SEARCH_RANGE) -> dict[str, Any]:
    """Code one sequence under EVERY arm in a single closed-loop pass.

    Motion, warp, z_ref, the residual and its quantized symbols are computed
    once and shared. Each arm only re-codes those same symbols with its own
    tables, so identical symbols / motion / reconstruction across arms is a
    property of the code path rather than something to be checked afterwards.
    """
    model.eval()
    device = next(model.parameters()).device
    frame_count = frames.shape[0]
    types = mc.gop_frame_types(frame_count, gop_size)
    latent_shape = tuple(model.encode(frames[0:1].to(device)).shape[1:])
    blocks = (frames.shape[2] // block_size, frames.shape[3] // block_size)

    writers, records = {}, {arm: [] for arm in arms}
    for arm, spec in arms.items():
        header = mc.TemporalStreamHeader(
            gop_size=gop_size, quantization_bits=intra_params.bits,
            quantization_mode=intra_params.mode,
            image_width=frames.shape[3], image_height=frames.shape[2],
            image_channels=frames.shape[1], latent_channels=latent_shape[0],
            latent_height=latent_shape[1], latent_width=latent_shape[2],
            frame_count=frame_count,
            num_intra_quantization_params=intra_params.scale.numel(),
            num_residual_quantization_params=residual_params.scale.numel(),
            block_size=block_size, search_range=search_range,
            motion_bits=motion_entropy_model.bits, reference_mode="mc",
            intra_entropy_model_id=intra_entropy_model.model_id(),
            residual_entropy_model_id=spec["entropy_model"].model_id(),
            motion_entropy_model_id=motion_entropy_model.model_id())
        writers[arm] = mc.TemporalStreamWriter(paths[arm], header, intra_params, residual_params)

    reconstructions, previous = [], None
    symbol_log = []
    # Ideal code length of the P-frame residual symbols under each arm's own
    # tables. This is what the arithmetic coder is trying to achieve, measured on
    # the symbols it actually coded - so comparing it against the emitted bytes
    # separates "the model is better" from "the coder captured it".
    ideal_bits = {arm: 0.0 for arm in arms}
    try:
        with mc.deterministic_kernels():
            for index in range(frame_count):
                frame = frames[index:index + 1].to(device)
                latent = model.encode(frame)
                frame_type = types[index]

                if frame_type == mc.FRAME_TYPE_I:
                    payload, _ = encode_latent_to_payload(
                        latent, params=intra_params, entropy_model=intra_entropy_model)
                    decoded, _ = decode_payload_to_latent(
                        payload, entropy_model=intra_entropy_model, params=intra_params,
                        shape=latent_shape)
                    reconstructed_latent = decoded.to(device)
                    for arm in arms:
                        writers[arm].append_frame(frame_type, b"", payload)
                        records[arm].append({"index": index, "frame_type": "I",
                                             "motion_bytes": 0,
                                             "residual_bytes": len(payload)})
                else:
                    motion = mc.estimate_block_motion(
                        previous, frame, block_size=block_size, search_range=search_range)
                    motion_payload = mc.encode_motion_payload(
                        motion, search_range=search_range, entropy_model=motion_entropy_model)
                    decoded_motion = mc.decode_motion_payload(
                        motion_payload, blocks, search_range=search_range,
                        entropy_model=motion_entropy_model)
                    warped = mc.warp_blocks(previous, decoded_motion, block_size=block_size)
                    reference_latent = model.encode(warped)
                    delta = latent - reference_latent

                    # ONE quantization, shared by every arm.
                    symbols = latent_to_symbols(delta, residual_params)
                    reference_numpy = reference_latent[0].cpu().numpy()
                    symbol_log.append(symbols.reshape(latent_shape))

                    for arm, spec in arms.items():
                        contexts = spec["context_model"].contexts(reference_numpy)
                        payload = encode_residual_symbols(
                            symbols, contexts, shape=latent_shape,
                            entropy_model=spec["entropy_model"],
                            context_model=spec["context_model"])
                        ideal_bits[arm] += _ideal_code_bits(
                            symbols, contexts, shape=latent_shape,
                            entropy_model=spec["entropy_model"],
                            context_model=spec["context_model"])
                        writers[arm].append_frame(frame_type, motion_payload, payload)
                        records[arm].append({"index": index, "frame_type": "P",
                                             "motion_bytes": len(motion_payload),
                                             "residual_bytes": len(payload)})

                    decoded_delta = symbols_to_latent(symbols, latent_shape, residual_params)
                    reconstructed_latent = reference_latent + decoded_delta.to(device)

                reconstruction = model.decode(reconstructed_latent)
                previous = reconstruction
                reconstructions.append(reconstruction.detach().cpu())
    finally:
        for writer in writers.values():
            writer.close()

    result = {"frame_count": frame_count,
              "reconstructions": torch.cat(reconstructions, dim=0),
              "symbols": symbol_log, "arms": {}}
    for arm in arms:
        rows = records[arm]
        container = paths[arm].stat().st_size
        motion_total = sum(r["motion_bytes"] for r in rows)
        residual_total = sum(r["residual_bytes"] for r in rows)
        result["arms"][arm] = {
            "path": str(paths[arm]), "frames": rows,
            "i_frames": sum(1 for r in rows if r["frame_type"] == "I"),
            "p_frames": sum(1 for r in rows if r["frame_type"] == "P"),
            "i_frame_residual_bytes": sum(r["residual_bytes"] for r in rows
                                          if r["frame_type"] == "I"),
            "p_frame_residual_bytes": sum(r["residual_bytes"] for r in rows
                                          if r["frame_type"] == "P"),
            "motion_bytes": motion_total, "residual_bytes": residual_total,
            "container_bytes": container,
            "container_overhead_bytes": container - motion_total - residual_total,
            "p_frame_ideal_bits": ideal_bits[arm],
        }
    return result


@torch.no_grad()
def decode_sequence(mc, model, path, *, intra_entropy_model, residual_entropy_model,
                    context_model, motion_entropy_model, return_symbols: bool = False):
    """Decode a conditional stream using only the bitstream and the calibration.

    The decoder rebuilds `z_ref` first, derives the SAME contexts from it, and
    only then decodes the residual payload - which is what makes the context
    legal: it is available before the symbols it conditions.
    """
    model.eval()
    device = next(model.parameters()).device
    reader = mc.TemporalStreamReader(path)
    header = reader.header
    for label, expected, supplied in (
        ("intra", header.intra_entropy_model_id, intra_entropy_model),
        ("residual", header.residual_entropy_model_id, residual_entropy_model),
        ("motion", header.motion_entropy_model_id, motion_entropy_model),
    ):
        if supplied.model_id() != expected:
            raise mc.TemporalFormatError(
                f"{label} entropy model mismatch: stream declares {expected.hex()}, "
                f"supplied model is {supplied.model_id().hex()}")

    reconstructions, symbol_log, previous = [], [], None
    with mc.deterministic_kernels():
        for index, (frame_type, motion_payload, residual_payload) in enumerate(reader):
            if frame_type == mc.FRAME_TYPE_I:
                latent, _ = decode_payload_to_latent(
                    residual_payload, entropy_model=intra_entropy_model,
                    params=reader.intra_params, shape=header.latent_shape)
                reconstructed_latent = latent.to(device)
            else:
                if previous is None:
                    raise mc.TemporalFormatError(
                        f"Frame {index} is a P-frame but no reference is available")
                motion = mc.decode_motion_payload(
                    motion_payload, header.block_grid, search_range=header.search_range,
                    entropy_model=motion_entropy_model)
                warped = mc.warp_blocks(previous, motion, block_size=header.block_size)
                reference_latent = model.encode(warped)
                contexts = context_model.contexts(reference_latent[0].cpu().numpy())
                symbols = decode_residual_symbols(
                    residual_payload, contexts, shape=header.latent_shape,
                    entropy_model=residual_entropy_model, context_model=context_model)
                symbol_log.append(symbols.reshape(header.latent_shape))
                delta = symbols_to_latent(symbols, header.latent_shape, reader.residual_params)
                reconstructed_latent = reference_latent + delta.to(device)

            reconstruction = model.decode(reconstructed_latent)
            reconstructions.append(reconstruction)
            previous = reconstruction

    frames = torch.cat(reconstructions, dim=0)
    return (frames, symbol_log) if return_symbols else frames


@torch.no_grad()
def collect_training_symbols(mc, model, sequences, calibration, *, gop_size, block_size,
                             search_range, device, max_frames):
    """M10H residual symbols + their reference latents, from TRAIN sequences."""
    analysis = _load_script("m10j_entropy_analysis")
    return analysis.collect_symbols(
        mc, model, sequences, calibration, gop_size=gop_size, block_size=block_size,
        search_range=search_range, device=device, max_frames=max_frames)


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10J: reference-conditioned entropy coding of M10H residual symbols.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage", choices=["smoke"], default="smoke")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--gop", type=int, default=DEFAULT_GOP)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--search-range", type=int, default=DEFAULT_SEARCH_RANGE)
    parser.add_argument("--quant-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--table-frames", type=int, default=200)
    parser.add_argument("--calibration-frames", type=int, default=200)
    parser.add_argument("--max-sequences", type=int, default=2)
    parser.add_argument("--max-frames-per-sequence", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    for label, path in (("--manifest", args.manifest), ("--checkpoint", args.checkpoint)):
        if not path.is_file():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            return 1

    mc = _load_script("m10h_motion_compensation")
    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    train_sequences = discover_sequences(args.manifest, split="train")
    test_sequences = discover_sequences(
        args.manifest, split="test", max_sequences=args.max_sequences,
        max_frames_per_sequence=args.max_frames_per_sequence)

    print("=" * 104)
    print("M10J - CONDITIONAL ENTROPY CODING (smoke)")
    print("=" * 104)
    calibration = mc.calibrate_grids(
        model, train_sequences, bits=args.bits, mode=args.quant_mode, gop_size=args.gop,
        block_size=args.block_size, search_range=args.search_range, reference_mode="mc",
        max_frames=args.calibration_frames)
    symbols, references = collect_training_symbols(
        mc, model, train_sequences, calibration, gop_size=args.gop,
        block_size=args.block_size, search_range=args.search_range, device=device,
        max_frames=args.table_frames)
    print(f"  table-fitting symbols: {len(symbols)} P-frames from the TRAIN split")

    arms = {"marginal": build_conditional_entropy_model(
        symbols, references, fit_context_model("marginal", references), bits=args.bits)}
    for scheme in DEPLOYED_CONTEXTS:
        arms[scheme] = build_conditional_entropy_model(
            symbols, references, fit_context_model(scheme, references), bits=args.bits)
    for arm, spec in arms.items():
        p = spec["provenance"]
        print(f"    {arm:<18} tables={p['tables']:>4}  fallbacks={p['fallback_tables']}  "
              f"id={p['entropy_model_id'][:12]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stream_dir = args.output_dir / "smoke_streams"
    stream_dir.mkdir(parents=True, exist_ok=True)
    print()
    print(f"{'sequence':<20} " + " ".join(f"{a + ' resid':>16}" for a in arms) + f" {'checks':>28}")
    for sequence in test_sequences:
        frames = sequence.load_frames()
        paths = {arm: stream_dir / f"{arm}_{sequence.sequence_id}.nvct" for arm in arms}
        result = encode_multi(
            mc, model, frames, arms, paths,
            intra_params=calibration["intra_params"],
            intra_entropy_model=calibration["intra_entropy_model"],
            residual_params=calibration["residual_params"],
            motion_entropy_model=calibration["motion_entropy_model"],
            gop_size=args.gop, block_size=args.block_size, search_range=args.search_range)

        checks = []
        motions = {a: result["arms"][a]["motion_bytes"] for a in arms}
        checks.append("motion=" + ("same" if len(set(motions.values())) == 1 else "DIFFERENT"))
        for arm, spec in arms.items():
            decoded, decoded_symbols = decode_sequence(
                mc, model, paths[arm],
                intra_entropy_model=calibration["intra_entropy_model"],
                residual_entropy_model=spec["entropy_model"],
                context_model=spec["context_model"],
                motion_entropy_model=calibration["motion_entropy_model"],
                return_symbols=True)
            exact = all(np.array_equal(a.reshape(-1), b.reshape(-1))
                        for a, b in zip(result["symbols"], decoded_symbols))
            same_pixels = torch.equal(decoded.cpu(), result["reconstructions"])
            if not (exact and same_pixels):
                checks.append(f"{arm}=FAIL")
        checks.append("symbols+recon=exact" if len(checks) == 1 else "")
        print(f"{sequence.sequence_id:<20} " +
              " ".join(f"{result['arms'][a]['residual_bytes']:>16,}" for a in arms) +
              f" {' '.join(c for c in checks if c):>28}")

    print(f"\nStreams: {stream_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
