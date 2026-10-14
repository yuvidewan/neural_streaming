"""Stage 1 diagnostic: is a 192-channel latent at stride 16 the right shape?

The first Stage 1 run's gate (`outputs/stage1_vimeo/colab_run1/`) came in at
+560.6% intra BD-rate against the deployed codec's +176.2%, and the byte ratio
at every operating point - 3.15x, 3.19x, 3.25x - is the channel count: both
models are stride 16, the baseline carries 64 channels and Stage 1 carries 192,
so Stage 1 codes exactly 3x the symbols per frame.

Phase A optimizes MSE alone, so nothing asked it to make those channels cheap.
Phase B (`D + lambda*R`) is the step that would, and it has never run. Before
spending another ~20 GPU-hours on that, this script answers the prior question:
**is there anything for rate pressure to find?**

Three measurements, all on the already-trained checkpoint, none needing a retrain:

1. **Per-channel energy.** The variance each latent channel carries, over real
   DAVIS TEST frames. A flat profile means information is spread across all 192
   channels and the shape is doing real work. A skewed one means most channels
   are nearly idle and the capacity is notional.

2. **Per-channel coded cost.** What the real quantizer and the real entropy
   model actually charge per channel, in bits, via the same `calibrate_grids`
   path the gate uses. This matters because `EmpiricalEntropyModel` holds one
   static frequency table PER CHANNEL, so a degenerate channel should cost
   almost nothing - the stack can already exploit sparsity if the transform
   produces any. Measuring both (1) and (2) separates "the model has no
   sparsity" from "the model has sparsity the coder is not cashing in".

3. **A channel-pruning sweep.** Keep the top-k channels by energy, zero the
   rest, decode, and score. This traces the distortion cost of a smaller latent
   FOR THIS TRAINED MODEL without retraining - an upper bound on what a
   narrower latent would lose, since a model trained at k channels would do at
   least as well as this one truncated to k.

Read together: if PSNR survives a large prune, 192 channels is the wrong shape
and the fix is architectural, not another training run. If it collapses
immediately, the channels are earning their keep and Phase B's rate pressure is
the right next move.

Diagnostic only - writes a report and modifies no production source.

Usage:
  ./.venv/Scripts/python.exe scripts/stage1_latent_shape.py \
      --checkpoint outputs/stage1_vimeo/colab_run1/best.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.calibration import (
    allocate_bits_per_channel,
    calibrate_quantization_params,
)
from nvc.compression.codec import encode_latent_to_payload, latent_to_symbols
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.evaluation.basic_metrics import psnr
from nvc.evaluation.perceptual_metrics import msssim
from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

ROOT = Path(__file__).resolve().parents[1]

# The two checkpoints the comparison is between: Stage 1's salvaged Phase A
# transform and the deployed 64-channel model the gate measures it against.
STAGE1_CHECKPOINT = Path("outputs/stage1_vimeo/colab_run1/best.pt")
BASELINE_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")

# Matches scripts/benchmark_intra_gate.py, so the coded costs here are the same
# quantizer and entropy tables the gate's bytes came from.
CALIBRATION_GOP = 10


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=STAGE1_CHECKPOINT)
    parser.add_argument("--baseline-checkpoint", type=Path, default=BASELINE_CHECKPOINT,
                        help="The 64-channel model the gate compares against. "
                             "Measured alongside, so the profiles are read on the "
                             "same frames with the same code.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/stage1_latent_shape"))
    parser.add_argument("--output-name", type=str, default="latent_shape.json")
    parser.add_argument("--bits", type=int, default=4,
                        help="Which gate operating point to price channels at. 4 is "
                             "the middle of the gate's 5/4/3.")
    parser.add_argument("--calibration-frames", type=int, default=400,
                        help="Matches the gate's default; changing it changes the "
                             "quantizer the costs are measured through.")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--prune-keep", type=int, nargs="+", default=None,
                        help="Channel counts to keep in the pruning sweep. The "
                             "default adapts to the model's own channel count.")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--energy-frames", type=int, default=120,
                        help="Frames sampled for the energy and cost profiles.")
    parser.add_argument("--rate-points", type=int, nargs="+", default=[5, 4, 3],
                        help="Bit depths for the rate/quality curve; matches the gate.")
    parser.add_argument("--prune-frames", type=int, default=40,
                        help="Frames scored per pruning point. Each point decodes "
                             "them all again, so this dominates runtime.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


@torch.no_grad()
def channel_energy(model, frames_by_sequence, device, *, limit: int) -> np.ndarray:
    """Per-channel variance of the latent, pooled over `limit` frames.

    Variance rather than mean absolute value: the quantizer is per-channel and
    its scale tracks spread, so variance is the quantity that turns into bits.
    """
    model.eval()
    sums = None
    squares = None
    count = 0
    for frames in frames_by_sequence:
        for index in range(frames.shape[0]):
            if count >= limit:
                break
            latent = model.encode(frames[index:index + 1].to(device))
            flat = latent[0].reshape(latent.shape[1], -1).double()
            if sums is None:
                sums = torch.zeros(flat.shape[0], dtype=torch.float64, device=flat.device)
                squares = torch.zeros_like(sums)
            sums += flat.sum(dim=1)
            squares += (flat * flat).sum(dim=1)
            count += 1
        if count >= limit:
            break
    elements = count * (latent.shape[2] * latent.shape[3])
    mean = sums / elements
    return (squares / elements - mean * mean).cpu().numpy(), count


@torch.no_grad()
def channel_coded_cost(model, frames_by_sequence, rig, device, *, limit: int):
    """Bits per element each channel actually costs under the real tables.

    Quantizes with the gate's own `intra_params` and prices the resulting symbols
    with the gate's own `intra_entropy_model`. `EmpiricalEntropyModel` keeps one
    frequency table per channel, so this is the number that decides whether an
    idle channel is free or is being paid for anyway.
    """
    model.eval()
    intra_params = rig["intra_params"]
    entropy_model = rig["intra_entropy_model"]
    frequencies = entropy_model.frequencies.astype(np.float64)
    # Code length of each symbol under its channel's table, in bits.
    totals = frequencies.sum(axis=1, keepdims=True)
    code_bits = -np.log2(frequencies / totals)

    channels = None
    bit_totals = None
    element_counts = 0
    histogram = None
    count = 0
    for frames in frames_by_sequence:
        for index in range(frames.shape[0]):
            if count >= limit:
                break
            latent = model.encode(frames[index:index + 1].to(device))
            # The codec's own quantize+code path, so these are the very symbols
            # the gate's bytes were produced from - not a reimplementation.
            _, symbols = encode_latent_to_payload(
                latent, params=intra_params, entropy_model=entropy_model)
            array = np.asarray(symbols).reshape(latent.shape[1], -1).astype(np.int64)
            if channels is None:
                channels = array.shape[0]
                bit_totals = np.zeros(channels, dtype=np.float64)
                histogram = np.zeros((channels, entropy_model.num_symbols), dtype=np.int64)
            for channel in range(channels):
                row = array[channel]
                # One table per channel when the model has that many; a model with
                # a single shared table prices every channel through row 0.
                table = channel if code_bits.shape[0] > channel else 0
                bit_totals[channel] += code_bits[table][row].sum()
                histogram[channel] += np.bincount(
                    row, minlength=entropy_model.num_symbols)
            element_counts += array.shape[1]
            count += 1
        if count >= limit:
            break

    bits_per_element = bit_totals / element_counts
    # How concentrated each channel's symbol distribution is: a channel pinned to
    # one symbol has entropy 0 and should be nearly free.
    probabilities = histogram / np.maximum(histogram.sum(axis=1, keepdims=True), 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(probabilities > 0, np.log2(probabilities), 0.0)
    empirical_entropy = -(probabilities * logs).sum(axis=1)
    return bits_per_element, empirical_entropy, count


@torch.no_grad()
def prune_sweep(model, frames_by_sequence, energy, device, keeps, *,
                limit: int, chunk: int = 4) -> list[dict[str, Any]]:
    """Keep the top-k channels by energy, zero the rest, decode, score.

    No quantizer and no entropy coder here on purpose: this isolates what the
    TRANSFORM loses when the latent is narrowed, with nothing else moving. It is
    an upper bound on a narrower model's distortion, because a model actually
    trained at k channels can only do better than this one truncated to k.

    Read it with care. Zeroing channels hands the decoder input it never saw in
    training, so part of any drop is the decoder being off-distribution rather
    than information genuinely lost - which is why the baseline's own curve is
    measured alongside as a control. If the baseline falls just as hard while
    retaining far more of its energy, the sweep is reporting decoder brittleness
    and not channel importance, and should not be used to argue a channel count.

    Frames are decoded in small batches: a whole DAVIS sequence of 1080p frames
    through the 192-channel decoder does not fit in 8GB.
    """
    model.eval()
    order = np.argsort(energy)[::-1].copy()  # strongest channel first
    results = []
    for keep in keeps:
        mask = torch.zeros(len(energy), device=device)
        mask[torch.from_numpy(order[:keep]).to(device)] = 1.0
        mask = mask.reshape(1, -1, 1, 1)

        psnr_values, msssim_values = [], []
        scored = 0
        for frames in frames_by_sequence:
            for start in range(0, frames.shape[0], chunk):
                if scored >= limit:
                    break
                batch = frames[start:start + chunk].to(device)
                latent = model.encode(batch) * mask
                reconstruction = model.decode(latent).clamp(0.0, 1.0)
                psnr_values.append(float(psnr(reconstruction, batch)))
                msssim_values.append(float(msssim(reconstruction, batch)))
                scored += batch.shape[0]
            if scored >= limit:
                break
        results.append({
            "channels_kept": int(keep),
            "fraction_kept": keep / len(energy),
            "energy_fraction_kept": float(energy[order[:keep]].sum() / energy.sum()),
            "psnr": float(np.mean(psnr_values)),
            "msssim": float(np.mean(msssim_values)),
        })
    return results


def _textbook_allocation(variances: np.ndarray, *, average_bits: int,
                         min_bits: int = 1) -> tuple[int, ...]:
    """Reverse water-filling, clamped at `average_bits` and NOT rebalanced.

    `nvc.compression.calibration.allocate_bits_per_channel` implements the same
    formula but then rebalances the realized average back to `average_bits`
    while also capping every channel at `average_bits`. Those two constraints
    together are only satisfiable by giving every channel exactly
    `average_bits`, so that function is identically a no-op at its default
    `max_bits` - measured, not assumed; the `library_rebalanced` arm records it.

    Dropping the rebalance is what makes this a real test: quiet channels give
    up levels, nothing takes them back, and the average falls. That is a
    rate REDUCTION bought with some distortion, so this arm is only meaningful
    against a uniform arm at a comparable average depth - which is why the
    caller also prices uniform coding at the next bit depth down.
    """
    geometric_mean = np.exp(np.mean(np.log(np.maximum(variances, 1e-30))))
    raw = average_bits + 0.5 * np.log2(np.maximum(variances, 1e-30) / geometric_mean)
    return tuple(int(b) for b in np.clip(np.rint(raw), min_bits, average_bits))


@torch.no_grad()
def _price_arm(model, frames_by_sequence, stack, device, *, bits, per_channel, limit):
    """Fit the quantizer and entropy model on TRAIN latents, then code TEST frames.

    Returns real coded bytes from the real arithmetic coder, and the PSNR of the
    decoded reconstruction - because every arm here sits at a different operating
    point and bytes alone cannot rank them.
    """
    from nvc.compression.codec import decode_payload_to_latent

    channels = stack.shape[1]
    params = calibrate_quantization_params(
        stack, bits=bits, mode="per_channel", bits_per_channel=per_channel)
    symbols = np.stack([
        latent_to_symbols(stack[i:i + 1], params).reshape(channels, -1)
        for i in range(stack.shape[0])])
    entropy_model = EmpiricalEntropyModel.from_symbols(
        symbols, bits=bits, num_tables=channels)

    total_bytes, elements, scored = 0, 0, 0
    psnr_values = []
    latent_shape = None
    for frames in frames_by_sequence:
        for index in range(frames.shape[0]):
            if scored >= limit:
                break
            frame = frames[index:index + 1].to(device)
            latent = model.encode(frame)
            latent_shape = latent_shape or tuple(latent.shape[1:])
            payload, _ = encode_latent_to_payload(
                latent, params=params, entropy_model=entropy_model)
            decoded, _ = decode_payload_to_latent(
                payload, entropy_model=entropy_model, params=params, shape=latent_shape)
            reconstruction = model.decode(decoded.to(device)).clamp(0.0, 1.0)
            total_bytes += len(payload)
            elements += latent.numel()
            psnr_values.append(float(psnr(reconstruction, frame)))
            scored += 1
        if scored >= limit:
            break
    return {
        "bytes": int(total_bytes),
        "bits_per_element": 8.0 * total_bytes / elements,
        "psnr": float(np.mean(psnr_values)),
        "frames": scored,
        "average_allocated_bits": float(np.mean(per_channel)) if per_channel else float(bits),
    }


@torch.no_grad()
def collect_train_latents(model, train_sequences, device, *, limit: int) -> torch.Tensor:
    """The same TRAIN-only latent collection `calibrate_grids` does for its intra
    grid. Gathered once per arm and shared, because it is the slow part and every
    quantizer below must be fitted on exactly the same latents to be comparable.
    """
    model.eval()
    latents, seen = [], 0
    for sequence in train_sequences:
        frames = sequence.load_frames()
        for index in range(frames.shape[0]):
            if seen >= limit:
                break
            latents.append(model.encode(frames[index:index + 1].to(device)).cpu())
            seen += 1
        if seen >= limit:
            break
    return torch.cat(latents, dim=0)


@torch.no_grad()
def rate_quality_at_depth(model, frames_by_sequence, stack, device, *, depths, limit):
    """Bytes and PSNR at each bit depth, plus the no-quantizer reference.

    The float row is the transform's own ceiling - what it reconstructs with no
    quantizer in the way at all. Comparing two models' float rows separates "is
    this a better transform" from "does its latent cost more", which the gate's
    single BD-rate number necessarily conflates.

    The per-depth loss against that ceiling also tests, cheaply, whether the
    decoder is brittle to quantization error it never trained on. Stage 1 ran
    without QAT (`build_model` is called with no `quantization_noise`) while the
    deployed baseline trained with `--qat-enabled`, so that asymmetry is a real
    candidate explanation for part of the gate gap and worth measuring rather
    than assuming either way.
    """
    from nvc.compression.codec import decode_payload_to_latent

    channels = stack.shape[1]
    float_psnr, float_msssim = [], []
    for frames in frames_by_sequence:
        for index in range(min(frames.shape[0], limit)):
            frame = frames[index:index + 1].to(device)
            reconstruction = model.decode(model.encode(frame)).clamp(0.0, 1.0)
            float_psnr.append(float(psnr(reconstruction, frame)))
            float_msssim.append(float(msssim(reconstruction, frame)))
    reference = {"psnr": float(np.mean(float_psnr)),
                 "msssim": float(np.mean(float_msssim)),
                 "frames": len(float_psnr)}

    rows = []
    for bits in depths:
        params = calibrate_quantization_params(stack, bits=bits, mode="per_channel")
        symbols = np.stack([
            latent_to_symbols(stack[i:i + 1], params).reshape(channels, -1)
            for i in range(stack.shape[0])])
        entropy_model = EmpiricalEntropyModel.from_symbols(
            symbols, bits=bits, num_tables=channels)
        total_bytes, elements, values = 0, 0, []
        for frames in frames_by_sequence:
            for index in range(min(frames.shape[0], limit)):
                frame = frames[index:index + 1].to(device)
                latent = model.encode(frame)
                payload, _ = encode_latent_to_payload(
                    latent, params=params, entropy_model=entropy_model)
                decoded, _ = decode_payload_to_latent(
                    payload, entropy_model=entropy_model, params=params,
                    shape=tuple(latent.shape[1:]))
                values.append(float(psnr(
                    model.decode(decoded.to(device)).clamp(0.0, 1.0), frame)))
                total_bytes += len(payload)
                elements += latent.numel()
        mean = float(np.mean(values))
        rows.append({
            "bits": bits, "bytes": int(total_bytes), "psnr": mean,
            "bits_per_element": 8.0 * total_bytes / elements,
            "quantization_loss_db": reference["psnr"] - mean,
        })
    return {"float_reference": reference, "depths": rows}


@torch.no_grad()
def waterfilling_comparison(model, frames_by_sequence, train_sequences, device, *,
                            bits: int, calibration_frames: int, limit: int,
                            stack=None):
    """Can per-channel bit allocation make a wide latent cheaper?

    The energy/cost profiles show the quietest quartile of channels paying ~80%
    of full bit depth while carrying ~1/15th the energy, because the quantizer
    runs `per_channel` and fits each channel's scale to its own spread - which
    re-spreads a near-idle channel's tiny values across the whole alphabet.
    Per-channel bit ALLOCATION is the mechanism that should fix that, and it is
    already implemented in `nvc.compression.calibration`, unused.

    Four arms, all fitted on the same TRAIN latents and coded on the same TEST
    frames with the real coder:

      uniform             - every channel at `bits`, i.e. what ships today
      library_rebalanced  - `allocate_bits_per_channel` exactly as provided
      waterfilled         - the same formula without the rebalance, so quiet
                            channels genuinely lose levels
      uniform_one_lower   - every channel at `bits - 1`

    The last is the control that makes the third interpretable: water-filling
    lowers the average depth, so it has to beat simply turning the dial down.
    """
    model.eval()
    if stack is None:
        stack = collect_train_latents(model, train_sequences, device,
                                     limit=calibration_frames)
    channels = stack.shape[1]

    variances = stack.transpose(0, 1).reshape(channels, -1).var(dim=1).numpy()
    library = allocate_bits_per_channel(stack, average_bits=bits)
    textbook = _textbook_allocation(variances, average_bits=bits)

    arms = {
        "uniform": _price_arm(model, frames_by_sequence, stack, device, bits=bits,
                              per_channel=None, limit=limit),
        "library_rebalanced": _price_arm(model, frames_by_sequence, stack, device,
                                         bits=bits, per_channel=library, limit=limit),
        "waterfilled": _price_arm(model, frames_by_sequence, stack, device, bits=bits,
                                  per_channel=textbook, limit=limit),
    }
    if bits - 1 >= 1:
        arms["uniform_one_lower"] = _price_arm(
            model, frames_by_sequence, stack, device, bits=bits - 1,
            per_channel=None, limit=limit)

    baseline = arms["uniform"]
    for name, arm in arms.items():
        arm["bytes_change_percent"] = 100.0 * (arm["bytes"] - baseline["bytes"]) / baseline["bytes"]
        arm["psnr_change_db"] = arm["psnr"] - baseline["psnr"]

    return {
        "arms": arms,
        "library_allocation_histogram": {
            str(b): int((np.asarray(library) == b).sum()) for b in sorted(set(library))},
        "waterfilled_allocation_histogram": {
            str(b): int((np.asarray(textbook) == b).sum()) for b in sorted(set(textbook))},
        "library_is_a_no_op": bool(len(set(library)) == 1 and library[0] == bits),
    }


def cost_versus_energy(energy: np.ndarray, bits_per_element: np.ndarray,
                       empirical_entropy: np.ndarray, *, bits: int) -> dict[str, Any]:
    """Does a near-idle channel actually cost less than a busy one?

    This is the question the 3x rate penalty turns on. `EmpiricalEntropyModel`
    holds one static frequency table per channel, so in principle a channel
    pinned to a single symbol is nearly free and extra channels are cheap unless
    they carry real content. But the intra quantizer runs in `per_channel` mode:
    each channel's scale is fitted to that channel's OWN spread. A channel
    carrying almost nothing therefore gets a very small scale, which re-spreads
    its tiny values across the full alphabet - and a near-uniform symbol
    distribution costs nearly the full `bits` per element no matter how little
    information it represents.

    If that is what is happening, the correlation between energy and cost will be
    weak and the quietest channels will still cost most of `bits`. Then channel
    count sets rate almost directly, Phase B's rate pressure has nothing to cash
    in, and the fix is in the quantizer, not in training.
    """
    order = np.argsort(energy)
    quietest = order[:max(1, energy.size // 4)]
    busiest = order[-max(1, energy.size // 4):]
    # Spearman: monotone association without assuming a functional form, and the
    # energies span several orders of magnitude.
    ranks_energy = np.argsort(np.argsort(energy)).astype(np.float64)
    ranks_cost = np.argsort(np.argsort(bits_per_element)).astype(np.float64)
    spearman = float(np.corrcoef(ranks_energy, ranks_cost)[0, 1])
    return {
        "spearman_energy_vs_bits": spearman,
        "quietest_quartile": {
            "mean_energy": float(energy[quietest].mean()),
            "mean_bits_per_element": float(bits_per_element[quietest].mean()),
            "mean_empirical_entropy": float(empirical_entropy[quietest].mean()),
            "fraction_of_full_bit_depth": float(bits_per_element[quietest].mean() / bits),
        },
        "busiest_quartile": {
            "mean_energy": float(energy[busiest].mean()),
            "mean_bits_per_element": float(bits_per_element[busiest].mean()),
            "mean_empirical_entropy": float(empirical_entropy[busiest].mean()),
            "fraction_of_full_bit_depth": float(bits_per_element[busiest].mean() / bits),
        },
        "energy_ratio_busiest_over_quietest": float(
            energy[busiest].mean() / max(energy[quietest].mean(), 1e-30)),
        "cost_ratio_busiest_over_quietest": float(
            bits_per_element[busiest].mean() / max(bits_per_element[quietest].mean(), 1e-30)),
    }


def profile_summary(energy: np.ndarray) -> dict[str, Any]:
    """The few numbers that decide the question, rather than 192 variances."""
    total = energy.sum()
    descending = np.sort(energy)[::-1]
    cumulative = np.cumsum(descending) / total
    return {
        "channels": int(energy.size),
        "channels_for_50pct_energy": int(np.searchsorted(cumulative, 0.50) + 1),
        "channels_for_90pct_energy": int(np.searchsorted(cumulative, 0.90) + 1),
        "channels_for_99pct_energy": int(np.searchsorted(cumulative, 0.99) + 1),
        # Guarded: if more than half the channels are dead the median is
        # exactly 0, which is a meaningful profile rather than an error.
        "max_over_median_energy": (float(descending[0] / np.median(energy))
                                   if np.median(energy) > 0 else float("inf")),
        # A channel at <1% of the strongest one is doing essentially nothing.
        "channels_below_1pct_of_max": int((energy < 0.01 * descending[0]).sum()),
    }


def analyse(label, checkpoint_path, args, sequences, frame_cache, mc, train_sequences,
            device) -> dict[str, Any]:
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device=device)
    model.eval()
    architecture = checkpoint.get("architecture", "BaselineAutoencoder")
    frames_by_sequence = [frame_cache[s.sequence_id] for s in sequences]

    energy, energy_frames = channel_energy(
        model, frames_by_sequence, device, limit=args.energy_frames)

    rig = mc.calibrate_grids(
        model, train_sequences, bits=args.bits, mode="per_channel",
        gop_size=CALIBRATION_GOP, block_size=args.block_size,
        search_range=args.search_range, reference_mode="mc",
        max_frames=args.calibration_frames)
    bits_per_element, empirical_entropy, cost_frames = channel_coded_cost(
        model, frames_by_sequence, rig, device, limit=args.energy_frames)

    train_stack = collect_train_latents(
        model, train_sequences, device, limit=args.calibration_frames)
    depth_curve = rate_quality_at_depth(
        model, frames_by_sequence, train_stack, device,
        depths=args.rate_points, limit=args.prune_frames)
    print(f"  no quantizer at all: PSNR {depth_curve['float_reference']['psnr']:.3f} dB "
          f"MS-SSIM {depth_curve['float_reference']['msssim']:.4f} "
          f"over {depth_curve['float_reference']['frames']} frames "
          f"- this is the transform's own ceiling")
    for row in depth_curve["depths"]:
        print(f"    {row['bits']}-bit  {row['bytes']:>10,} B  "
              f"{row['bits_per_element']:.4f} bits/element  PSNR {row['psnr']:6.3f} dB  "
              f"(-{row['quantization_loss_db']:.3f} dB vs the ceiling)")

    waterfilling = waterfilling_comparison(
        model, frames_by_sequence, train_sequences, device, bits=args.bits,
        calibration_frames=args.calibration_frames, limit=args.prune_frames,
        stack=train_stack)
    print(f"  per-channel bit allocation ({args.bits}-bit nominal, "
          f"{waterfilling['arms']['uniform']['frames']} frames):")
    for name, arm in waterfilling["arms"].items():
        print(f"    {name:<19} avg {arm['average_allocated_bits']:.2f} bits  "
              f"{arm['bytes']:>10,} B  {arm['bytes_change_percent']:+7.2f}%  "
              f"PSNR {arm['psnr']:6.2f} dB  {arm['psnr_change_db']:+.2f} dB")
    print(f"    library allocation {waterfilling['library_allocation_histogram']}"
          + ("  <-- no-op by construction" if waterfilling["library_is_a_no_op"] else ""))
    print(f"    water-filled alloc {waterfilling['waterfilled_allocation_histogram']}")

    channels = int(energy.size)
    keeps = sorted({min(k, channels) for k in args.prune_keep}) if args.prune_keep else sorted(
        {k for k in (channels, channels // 2, channels // 3, channels // 4,
                     channels // 6, channels // 8, 16, 8) if 0 < k <= channels})
    pruning = prune_sweep(model, frames_by_sequence, energy, device, keeps,
                          limit=args.prune_frames)

    print(f"\n{label}: {architecture}, {model.num_parameters():,} parameters, "
          f"{channels} channels")
    summary = profile_summary(energy)
    print(f"  energy: {summary['channels_for_50pct_energy']}/{channels} channels carry "
          f"50%, {summary['channels_for_90pct_energy']} carry 90%, "
          f"{summary['channels_for_99pct_energy']} carry 99%")
    print(f"  {summary['channels_below_1pct_of_max']} channels below 1% of the "
          f"strongest channel's energy")
    print(f"  coded cost at {args.bits}-bit: mean {bits_per_element.mean():.4f} "
          f"bits/element, min {bits_per_element.min():.4f}, "
          f"max {bits_per_element.max():.4f}")
    print(f"  cheapest decile mean {np.sort(bits_per_element)[:max(1, channels // 10)].mean():.4f} "
          f"bits/element - if this is not near zero, idle channels are still being paid for")
    relation = cost_versus_energy(energy, bits_per_element, empirical_entropy,
                                  bits=args.bits)
    quiet, busy = relation["quietest_quartile"], relation["busiest_quartile"]
    print(f"  quietest quartile: {quiet['mean_energy']:.3e} energy -> "
          f"{quiet['mean_bits_per_element']:.3f} bits/element "
          f"({quiet['fraction_of_full_bit_depth']:.1%} of the {args.bits}-bit depth)")
    print(f"  busiest  quartile: {busy['mean_energy']:.3e} energy -> "
          f"{busy['mean_bits_per_element']:.3f} bits/element "
          f"({busy['fraction_of_full_bit_depth']:.1%} of the {args.bits}-bit depth)")
    print(f"  energy ratio busiest/quietest {relation['energy_ratio_busiest_over_quietest']:.1f}x, "
          f"but cost ratio only {relation['cost_ratio_busiest_over_quietest']:.2f}x "
          f"(Spearman {relation['spearman_energy_vs_bits']:+.3f})")
    print("  pruning (no quantizer; transform only):")
    for row in pruning:
        print(f"    keep {row['channels_kept']:>3}/{channels} "
              f"({row['energy_fraction_kept']:6.2%} of energy)  "
              f"PSNR {row['psnr']:6.2f} dB  MS-SSIM {row['msssim']:.4f}")

    return {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "architecture": architecture,
        "parameters": model.num_parameters(),
        "latent_channels": channels,
        "energy_frames": energy_frames,
        "cost_frames": cost_frames,
        "bits": args.bits,
        "energy_summary": summary,
        "rate_quality_at_depth": depth_curve,
        "cost_versus_energy": relation,
        "waterfilling": waterfilling,
        "per_channel_energy": energy.tolist(),
        "per_channel_bits_per_element": bits_per_element.tolist(),
        "per_channel_empirical_entropy": empirical_entropy.tolist(),
        "mean_bits_per_element": float(bits_per_element.mean()),
        "cheapest_decile_bits_per_element": float(
            np.sort(bits_per_element)[:max(1, channels // 10)].mean()),
        "pruning": pruning,
        "quantizer_provenance": str(rig["provenance"]),
    }


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    mc = _load_script("m10h_motion_compensation")
    sequences = discover_sequences(args.manifest, split="test",
                                   max_sequences=args.max_sequences,
                                   max_frames_per_sequence=args.max_frames)
    frame_cache = {s.sequence_id: s.load_frames() for s in sequences}
    train_sequences = discover_sequences(args.manifest, split="train")

    print("=" * 92)
    print("STAGE 1 LATENT SHAPE - is 192 channels at stride 16 the right shape?")
    print(f"{len(sequences)} DAVIS TEST sequences, "
          f"{sum(frame_cache[s.sequence_id].shape[0] for s in sequences)} frames; "
          f"profiles on the first {args.energy_frames}")
    print("=" * 92)

    arms = [("stage1", args.checkpoint)]
    if args.baseline_checkpoint is not None and args.baseline_checkpoint.is_file():
        arms.append(("baseline", args.baseline_checkpoint))
    else:
        print(f"[warn] baseline checkpoint {args.baseline_checkpoint} not found; "
              "reporting Stage 1 alone")

    results = [analyse(label, path, args, sequences, frame_cache, mc, train_sequences,
                       device)
               for label, path in arms]

    report = {
        "stage": "PARITY_ROADMAP Stage 1 - latent shape diagnostic",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "bits": args.bits, "calibration_frames": args.calibration_frames,
            "calibration_gop": CALIBRATION_GOP, "energy_frames": args.energy_frames,
            "block_size": args.block_size, "search_range": args.search_range,
            "sequences": [s.sequence_id for s in sequences], "seed": args.seed,
        },
        "arms": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / args.output_name
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
