"""M10I: a learned residual codec CONDITIONED on the warped reference.

THE HYPOTHESIS
---------------
M10H codes the motion-compensated latent residual `r = z_t - z_ref` against its
MARGINAL statistics: one per-channel percentile grid and one frequency table,
fitted once over the training set and applied identically to every residual
regardless of what the reference looked like. But a residual sitting on top of a
flat, well-predicted region of `z_ref` and one sitting on a badly-predicted edge
are very different signals, and the coder cannot tell them apart.

M10I tests whether conditioning on `z_ref` makes the residual easier to code.

THE MECHANISM (this is the part that must not be hand-waved)
--------------------------------------------------------------
Conditioning enters through BOTH transforms, not through a renamed marginal
model:

    analysis    w     = r     + f_a([r,     z_ref])      encoder side
    synthesis   r_hat = w_hat + f_s([w_hat, z_ref])      decoder side

`f_a` and `f_s` are small 3-layer convolutional networks that each take the
CONCATENATION of the residual (or its reconstruction) with the reference latent,
so every output element can depend on the reference at that position. `w` - not
`r` - is what gets quantized and entropy-coded, so the network is free to
reshape the residual into something cheaper to code, using information the
decoder also has.

WHY THE SKIP CONNECTIONS AND ZERO INIT MATTER
-----------------------------------------------
The last convolution of each network is zero-initialised, so at step 0:

    f_a = f_s = 0    =>    w = r    and    r_hat = w_hat

which is EXACTLY M10H's marginal path, bit for bit. The experiment therefore
starts from the baseline rather than from a random codec, the comparison is
strictly an ablation of the conditioning mechanism, and any measured difference
is attributable to what training changed. It also means a failure to improve is
informative rather than an artefact of a bad initialisation.

WHAT IS TRAINED, AND WHAT IS NOT
----------------------------------
Frozen: the M10F lambda=3e-4 Encoder and Decoder (the frozen intra operating
point) and the block-matching motion estimator. Trainable: only `f_a` and `f_s`.
That isolates the conditioning mechanism - the thing under test - from every
other moving part, and keeps the motion payload byte-identical to M10H's by
construction.

TRAINING DATA IS PRECOMPUTED, AND WHY THAT IS SOUND
------------------------------------------------------
Each training sample is a causal `(z_t, z_ref)` pair produced by running the
M10H motion-compensated codec closed-loop over the DAVIS train split. Because
the model is initialised to the identity, those pairs are exactly on-policy at
step 0 and drift only as training moves away from the baseline. This is an
open-loop approximation and is documented as such rather than presented as
closed-loop training.

CAUSALITY
----------
`z_ref = Encoder(Warp(x_hat_{t-1}, decoded motion))` - built from the previous
RECONSTRUCTED frame and the DECODED motion, both of which the decoder holds. No
future frame, no original previous frame, no untransmitted side information.

Example usage (PowerShell, from the project root, with .venv activated):

    .venv\\Scripts\\python.exe scripts\\m10i_conditional_residual.py --stage smoke
    .venv\\Scripts\\python.exe scripts\\m10i_conditional_residual.py --stage cache
    .venv\\Scripts\\python.exe scripts\\m10i_conditional_residual.py --stage train
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.codec import (
    decode_payload_to_latent,
    encode_latent_to_payload,
    latent_to_symbols,
)
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.data.image_io import read_image_as_tensor
from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.training.quantization_noise import QuantizationNoise
from nvc.training.rate_estimator import RateEstimator
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m10i_temporal_residual")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
FROZEN_LAMBDA = 3.0e-4
DEFAULT_GOP = 10
DEFAULT_BLOCK_SIZE = 16
DEFAULT_SEARCH_RANGE = 16
# Data-driven: an 8/6/5-bit probe of the M10H MC path moved rate 2x while moving
# PSNR only 0.08 dB - a vertical curve with nothing for BD-rate to integrate
# over. 5/4/3 spans 29.710 -> 28.289 dB (1.42 dB) across a 2.1x rate range.
RATE_POINTS = (5, 4, 3)
TRAIN_BITS = 4  # the middle rate point; one model is evaluated at all three


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the conditional residual codec ------------------------------------------


class ConditionalResidualCodec(nn.Module):
    """Analysis and synthesis transforms conditioned on the warped reference.

        w     = r     + f_a([r,     z_ref])
        r_hat = w_hat + f_s([w_hat, z_ref])

    Both halves see `z_ref`, so conditioning affects what gets coded AND how it
    is reconstructed. The final convolution of each branch is zero-initialised,
    making the whole module the identity at step 0 - i.e. exactly M10H.
    """

    def __init__(self, latent_channels: int = 64, hidden_channels: int = 96) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_channels = hidden_channels
        self.analysis = self._branch(latent_channels, hidden_channels)
        self.synthesis = self._branch(latent_channels, hidden_channels)
        for branch in (self.analysis, self.synthesis):
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)

    @staticmethod
    def _branch(latent_channels: int, hidden_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(2 * latent_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, latent_channels, 3, padding=1),
        )

    def encode_residual(self, residual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return residual + self.analysis(torch.cat([residual, reference], dim=1))

    def decode_residual(self, coded: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return coded + self.synthesis(torch.cat([coded, reference], dim=1))

    def config_dict(self) -> dict[str, int]:
        return {"latent_channels": self.latent_channels,
                "hidden_channels": self.hidden_channels}


def build_codec(config: dict[str, int] | None = None) -> ConditionalResidualCodec:
    return ConditionalResidualCodec(**(config or {}))


def load_codec(path: str | Path, *, device=None) -> tuple[ConditionalResidualCodec, dict]:
    checkpoint = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
    codec = build_codec(checkpoint["codec_config"])
    codec.load_state_dict(checkpoint["codec_state_dict"])
    if device is not None:
        codec = codec.to(device)
    codec.eval()
    return codec, checkpoint


# --- the training-pair cache --------------------------------------------------


@torch.no_grad()
def build_pair_cache(mc, model, sequences, calibration, *, gop_size: int, block_size: int,
                     search_range: int, device, max_frames: int | None = None) -> dict[str, Any]:
    """Run the M10H MC codec closed-loop and record every P-frame's (z_t, z_ref).

    Frame pixels are NOT cached - only the frame path - so the cache stays small
    and the training loop reads the original image from disk. Latents are stored
    as float16, which is well inside the precision the downstream quantizer
    uses.
    """
    model.eval()
    latents_t, latents_ref, frame_paths = [], [], []
    seen = 0

    with mc.deterministic_kernels():
        for sequence in sequences:
            frames = sequence.load_frames()
            types = mc.gop_frame_types(frames.shape[0], gop_size)
            latent_shape = None
            previous_reconstruction = None
            for index in range(frames.shape[0]):
                if max_frames is not None and seen >= max_frames:
                    break
                frame = frames[index:index + 1].to(device)
                latent = model.encode(frame)
                if latent_shape is None:
                    latent_shape = tuple(latent.shape[1:])
                if types[index] == mc.FRAME_TYPE_I:
                    payload, _ = encode_latent_to_payload(
                        latent, params=calibration["intra_params"],
                        entropy_model=calibration["intra_entropy_model"])
                    decoded, _ = decode_payload_to_latent(
                        payload, entropy_model=calibration["intra_entropy_model"],
                        params=calibration["intra_params"], shape=latent_shape)
                    reconstructed = decoded.to(device)
                else:
                    motion = mc.estimate_block_motion(
                        previous_reconstruction, frame,
                        block_size=block_size, search_range=search_range)
                    payload = mc.encode_motion_payload(
                        motion, search_range=search_range,
                        entropy_model=calibration["motion_entropy_model"])
                    decoded_motion = mc.decode_motion_payload(
                        payload, (frames.shape[2] // block_size, frames.shape[3] // block_size),
                        search_range=search_range,
                        entropy_model=calibration["motion_entropy_model"])
                    warped = mc.warp_blocks(previous_reconstruction, decoded_motion,
                                            block_size=block_size)
                    reference_latent = model.encode(warped)

                    latents_t.append(latent[0].detach().to("cpu", torch.float16))
                    latents_ref.append(reference_latent[0].detach().to("cpu", torch.float16))
                    frame_paths.append(str(sequence.frame_paths[index]))

                    delta = latent - reference_latent
                    residual_payload, _ = encode_latent_to_payload(
                        delta, params=calibration["residual_params"],
                        entropy_model=calibration["residual_entropy_model"])
                    decoded_delta, _ = decode_payload_to_latent(
                        residual_payload, entropy_model=calibration["residual_entropy_model"],
                        params=calibration["residual_params"], shape=latent_shape)
                    reconstructed = reference_latent + decoded_delta.to(device)
                previous_reconstruction = model.decode(reconstructed)
                seen += 1
            if max_frames is not None and seen >= max_frames:
                break

    if not latents_t:
        raise ValueError("No P-frame pairs were produced (is gop_size larger than every sequence?)")
    return {
        "z_t": torch.stack(latents_t),
        "z_ref": torch.stack(latents_ref),
        "frame_paths": frame_paths,
        "count": len(latents_t),
    }


class PairDataset(torch.utils.data.Dataset):
    """(z_t, z_ref, x_t) triples; the frame is read from disk on demand."""

    def __init__(self, cache: dict[str, Any]) -> None:
        self.z_t = cache["z_t"]
        self.z_ref = cache["z_ref"]
        self.frame_paths = cache["frame_paths"]

    def __len__(self) -> int:
        return self.z_t.shape[0]

    def __getitem__(self, index: int):
        return (self.z_t[index].float(), self.z_ref[index].float(),
                read_image_as_tensor(Path(self.frame_paths[index])))


# --- training -----------------------------------------------------------------


def train_codec(codec, model, train_cache, val_cache, *, lambda_rate: float, bits: int,
                epochs: int, batch_size: int, learning_rate: float, rate_lr: float,
                device, seed: int = 42, log=print) -> dict[str, Any]:
    """L = D + lambda*R, exactly the established objective.

    D is pixel MSE after the FROZEN decoder, so the conditional codec is
    optimised against real reconstruction quality rather than a latent-space
    proxy for it. R is the project's differentiable Laplace rate proxy on `w` -
    the tensor actually entropy-coded - with QAT noise standing in for
    quantization, and scale tracking enabled exactly as every rate-aware
    milestone since M10A.
    """
    seed_everything(seed)
    codec = codec.to(device).train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    train_loader = torch.utils.data.DataLoader(
        PairDataset(train_cache), batch_size=batch_size, shuffle=True, num_workers=0,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        PairDataset(val_cache), batch_size=batch_size, shuffle=False, num_workers=0)

    # The residual grid the deployed coder will use, as the starting scale for
    # both the QAT relaxation and the rate proxy.
    residual_sample = (train_cache["z_t"][:256].float() - train_cache["z_ref"][:256].float())
    params = calibrate_quantization_params(residual_sample, bits=bits, mode="per_channel")
    noise = QuantizationNoise(params.scale.clone().to(device), bits=bits, mode="per_channel")
    estimator = RateEstimator(params.scale.clone(), bits=bits, mode="per_channel",
                              track_scale=True, scale_momentum=0.99).to(device)

    optimizer = torch.optim.Adam([
        {"params": codec.parameters(), "lr": learning_rate},
        {"params": estimator.parameters(), "lr": rate_lr},
    ])

    image_pixels = 256 * 256
    history: list[dict[str, Any]] = []
    # The M10G convention deploys the BEST-VALIDATION checkpoint, so its weights
    # have to be kept as training continues past it - reporting a selected epoch
    # while saving the final weights would defeat the whole convention.
    best_state: dict[str, Any] | None = None
    best_objective = float("inf")
    for epoch in range(1, epochs + 1):
        codec.train()
        started = time.time()
        totals = {"loss": 0.0, "distortion": 0.0, "rate": 0.0, "n": 0}
        grad_norm = 0.0
        for z_t, z_ref, frame in train_loader:
            z_t, z_ref, frame = z_t.to(device), z_ref.to(device), frame.to(device)
            residual = z_t - z_ref
            coded = codec.encode_residual(residual, z_ref)
            noisy = noise.apply(coded)
            reconstructed_residual = codec.decode_residual(noisy, z_ref)
            reconstruction = model.decode(z_ref + reconstructed_residual)

            distortion = F.mse_loss(reconstruction, frame)
            rate = estimator(coded, image_pixels)
            loss = distortion + lambda_rate * rate

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(codec.parameters(), 1e9))
            optimizer.step()
            estimator.update_bin_width(coded.detach())

            batch = z_t.shape[0]
            totals["loss"] += float(loss.detach()) * batch
            totals["distortion"] += float(distortion.detach()) * batch
            totals["rate"] += float(rate.detach()) * batch
            totals["n"] += batch

        codec.eval()
        val_totals = {"loss": 0.0, "distortion": 0.0, "rate": 0.0, "n": 0, "mse": 0.0}
        with torch.no_grad():
            for z_t, z_ref, frame in val_loader:
                z_t, z_ref, frame = z_t.to(device), z_ref.to(device), frame.to(device)
                residual = z_t - z_ref
                coded = codec.encode_residual(residual, z_ref)
                noisy = noise.apply(coded)
                reconstruction = model.decode(z_ref + codec.decode_residual(noisy, z_ref))
                distortion = F.mse_loss(reconstruction, frame)
                rate = estimator(coded, image_pixels)
                batch = z_t.shape[0]
                val_totals["loss"] += float(distortion + lambda_rate * rate) * batch
                val_totals["distortion"] += float(distortion) * batch
                val_totals["rate"] += float(rate) * batch
                val_totals["n"] += batch

        record = {
            "epoch": epoch,
            "train_loss": totals["loss"] / totals["n"],
            "train_distortion": totals["distortion"] / totals["n"],
            "train_rate_bpp": totals["rate"] / totals["n"],
            "val_loss": val_totals["loss"] / val_totals["n"],
            "val_distortion": val_totals["distortion"] / val_totals["n"],
            "val_rate_bpp": val_totals["rate"] / val_totals["n"],
            "val_psnr": 10.0 * math.log10(1.0 / (val_totals["distortion"] / val_totals["n"])),
            "grad_norm": grad_norm,
            "rate_enabled": True,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        if record["val_loss"] < best_objective:
            best_objective = record["val_loss"]
            best_state = {k: v.detach().clone() for k, v in codec.state_dict().items()}
        log(f"[EPOCH {epoch:>3}] train {record['train_loss']:.6e}  "
            f"val {record['val_loss']:.6e}  val D {record['val_distortion']:.6e}  "
            f"val R {record['val_rate_bpp']:.4f} bpp  PSNR {record['val_psnr']:.2f} dB  "
            f"|g| {grad_norm:.3e}  {record['elapsed_seconds']:.0f}s")
    return {"history": history, "estimator": estimator,
            "best_state_dict": best_state,
            "final_state_dict": {k: v.detach().clone()
                                 for k, v in codec.state_dict().items()}}


# --- deployed coding ----------------------------------------------------------


@torch.no_grad()
def calibrate_conditional_grid(codec, cache, *, bits: int, mode: str = "per_channel",
                               device, max_samples: int = 400) -> dict[str, Any]:
    """Fit the quantization grid and frequency tables to `w`, the tensor the
    conditional codec actually emits - not to the raw residual."""
    codec.eval()
    coded = []
    for start in range(0, min(max_samples, cache["z_t"].shape[0]), 32):
        z_t = cache["z_t"][start:start + 32].float().to(device)
        z_ref = cache["z_ref"][start:start + 32].float().to(device)
        coded.append(codec.encode_residual(z_t - z_ref, z_ref).cpu())
    stack = torch.cat(coded, dim=0)
    params = calibrate_quantization_params(stack, bits=bits, mode=mode)
    symbols = np.stack([
        latent_to_symbols(stack[i:i + 1], params).reshape(stack.shape[1], -1)
        for i in range(stack.shape[0])])
    entropy_model = EmpiricalEntropyModel.from_symbols(
        symbols, bits=bits, num_tables=stack.shape[1])
    return {"params": params, "entropy_model": entropy_model, "samples": int(stack.shape[0])}


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M10I: learned residual codec conditioned on the warped reference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage", choices=["smoke", "cache", "train", "all"], default="smoke")
    parser.add_argument("--gop", type=int, default=DEFAULT_GOP)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--search-range", type=int, default=DEFAULT_SEARCH_RANGE)
    parser.add_argument("--train-bits", type=int, default=TRAIN_BITS)
    parser.add_argument("--quant-mode", choices=["global", "per_channel"], default="per_channel")
    parser.add_argument("--lambda-rate", type=float, default=FROZEN_LAMBDA)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rate-lr", type=float, default=defaults.rate_lr)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--max-train-frames", type=int, default=None)
    parser.add_argument("--max-val-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
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
    convention = _load_script("m10g_evaluation_convention")
    device = get_device() if args.device == "auto" else torch.device(args.device)
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    smoke = args.stage == "smoke"
    max_train = args.max_train_frames if not smoke else 40
    max_val = args.max_val_frames if not smoke else 20

    train_sequences = discover_sequences(args.manifest, split="train")
    val_sequences = discover_sequences(args.manifest, split="val")

    print("=" * 104)
    print("M10I - LEARNED CONDITIONAL RESIDUAL CODEC")
    print("=" * 104)
    print(f"  conditioning : analysis AND synthesis, both on z_ref (concat), "
          f"zero-init => identity at step 0 (== M10H)")
    print(f"  frozen       : Encoder, Decoder (lambda={FROZEN_LAMBDA:.1e}), motion estimator")
    print(f"  objective    : D + {args.lambda_rate:.1e} * R  (QAT {args.train_bits}-bit, "
          f"scale tracking on)")
    print(f"  rate points  : {list(RATE_POINTS)} bit (data-driven; 8/6/5 are vertically aligned)")

    print("\n  building the M10H calibration the cache is generated under ...", flush=True)
    calibration = mc.calibrate_grids(
        model, train_sequences, bits=args.train_bits, mode=args.quant_mode,
        gop_size=args.gop, block_size=args.block_size, search_range=args.search_range,
        reference_mode="mc", max_frames=args.calibration_frames)

    cache_path = args.output_dir / f"pairs_{args.train_bits}bit.pt"
    if args.stage in ("cache", "train", "all", "smoke"):
        if cache_path.is_file() and args.stage in ("train", "all"):
            print(f"  reusing pair cache: {cache_path}")
            caches = torch.load(cache_path, map_location="cpu", weights_only=False)
        else:
            print("  building causal (z_t, z_ref) pair cache ...", flush=True)
            started = time.time()
            train_cache = build_pair_cache(
                mc, model, train_sequences, calibration, gop_size=args.gop,
                block_size=args.block_size, search_range=args.search_range,
                device=device, max_frames=max_train)
            val_cache = build_pair_cache(
                mc, model, val_sequences, calibration, gop_size=args.gop,
                block_size=args.block_size, search_range=args.search_range,
                device=device, max_frames=max_val)
            caches = {"train": train_cache, "val": val_cache}
            if not smoke:
                torch.save(caches, cache_path)
            print(f"  cache: {train_cache['count']} train pairs, {val_cache['count']} val pairs "
                  f"({time.time() - started:.0f}s)")

    if args.stage == "cache":
        return 0

    codec = build_codec({"latent_channels": model.encoder.latent_channels,
                         "hidden_channels": args.hidden_channels})
    trainable = sum(p.numel() for p in codec.parameters())
    frozen = sum(p.numel() for p in model.parameters())
    print(f"\n  trainable (conditional codec): {trainable:,}")
    print(f"  frozen (encoder + decoder)   : {frozen:,}")

    epochs = 2 if smoke else args.epochs
    result = train_codec(
        codec, model, caches["train"], caches["val"],
        lambda_rate=args.lambda_rate, bits=args.train_bits, epochs=epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        rate_lr=args.rate_lr, device=device, seed=args.seed)
    history = result["history"]

    finite = all(math.isfinite(r["train_loss"]) and math.isfinite(r["val_loss"])
                 for r in history)
    gradients = all(r["grad_norm"] > 0 for r in history)
    print(f"\n  finite losses: {finite}   nonzero gradients through the conditional path: "
          f"{gradients}")
    if not (finite and gradients):
        print("[ERROR] training produced non-finite values or dead gradients", file=sys.stderr)
        return 1

    # Checkpoint selection uses the M10G convention and VALIDATION ONLY.
    selection = convention.select_checkpoint(history, objective_key="val_loss")
    print(f"  selected checkpoint: epoch {selection['selected_epoch']} "
          f"(val objective {selection['selected_objective']:.6e}); final epoch "
          f"{selection['final_epoch']}, best-vs-final gap "
          f"{selection['best_vs_final_percent']:.2f}%")

    # PRIMARY = the selected best-validation weights; the final weights are kept
    # alongside as the secondary convergence diagnostic, never as the deployed model.
    codec.load_state_dict(result["best_state_dict"])
    torch.save({
        "codec_state_dict": result["best_state_dict"],
        "final_codec_state_dict": result["final_state_dict"],
        "deployed_epoch": selection["selected_epoch"],
        "codec_config": codec.config_dict(),
        "history": history, "selection": selection,
        "lambda_rate": args.lambda_rate, "train_bits": args.train_bits,
        "frozen_checkpoint": str(args.checkpoint), "seed": args.seed,
    }, args.output_dir / ("smoke_codec.pt" if smoke else "conditional_codec.pt"))

    summary = {
        "phase": "M10I smoke" if smoke else "M10I training",
        "conditioning": "analysis and synthesis, both on z_ref; zero-init identity at step 0",
        "frozen_checkpoint": str(args.checkpoint), "frozen_lambda": FROZEN_LAMBDA,
        "objective": f"D + {args.lambda_rate} * R",
        "train_bits": args.train_bits, "rate_points": list(RATE_POINTS),
        "trainable_parameters": trainable, "frozen_parameters": frozen,
        "epochs": epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "rate_lr": args.rate_lr,
        "seed": args.seed, "device": str(device), "torch_version": torch.__version__,
        "train_pairs": caches["train"]["count"], "val_pairs": caches["val"]["count"],
        "finite_losses": finite, "nonzero_gradients": gradients,
        "checkpoint_selection": selection,
        "history": history,
        "training_note": (
            "Training pairs are precomputed from the M10H closed loop. Because the codec is "
            "initialised to the identity, they are exactly on-policy at step 0 and drift only "
            "as training moves away from the baseline. This is an open-loop approximation."
        ),
    }
    path = args.output_dir / ("smoke_training.json" if smoke else "training_summary.json")
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nSummary: {path}")
    return 0


# --- deployed conditional coding ---------------------------------------------
#
# These mirror M10H's encode/decode exactly, with ONE substitution: the tensor
# handed to the quantizer/entropy coder is `w = f_a(r, z_ref)` rather than `r`,
# and reconstruction runs `r_hat = f_s(w_hat, z_ref)`. Motion estimation, motion
# coding, the container and the I-frame path are M10H's, imported rather than
# reimplemented - which is what keeps the motion payload identical between the
# two arms by construction rather than by inspection.
#
# The stream declares reference mode "mc" because the motion path IS M10H's. The
# two arms are told apart by their residual entropy model id, which the decoder
# verifies against the header - so an M10H decoder handed an M10I stream fails
# loudly instead of silently reconstructing something else.


@torch.no_grad()
def encode_sequence_conditional(mc, model, codec, frames, path, *, intra_params,
                                intra_entropy_model, residual_params, residual_entropy_model,
                                motion_entropy_model, gop_size=DEFAULT_GOP,
                                block_size=DEFAULT_BLOCK_SIZE,
                                search_range=DEFAULT_SEARCH_RANGE):
    model.eval()
    codec.eval()
    device = next(model.parameters()).device
    frame_count = frames.shape[0]
    types = mc.gop_frame_types(frame_count, gop_size)
    latent_shape = tuple(model.encode(frames[0:1].to(device)).shape[1:])
    blocks = (frames.shape[2] // block_size, frames.shape[3] // block_size)

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
        residual_entropy_model_id=residual_entropy_model.model_id(),
        motion_entropy_model_id=motion_entropy_model.model_id())

    records, encoder_latents, encoder_reconstructions = [], [], []
    previous_reconstruction = None

    with mc.deterministic_kernels():
        with mc.TemporalStreamWriter(path, header, intra_params, residual_params) as writer:
            for index in range(frame_count):
                frame = frames[index:index + 1].to(device)
                latent = model.encode(frame)
                frame_type = types[index]
                motion_payload = b""

                if frame_type == mc.FRAME_TYPE_I:
                    residual_payload, _ = encode_latent_to_payload(
                        latent, params=intra_params, entropy_model=intra_entropy_model)
                    decoded, _ = decode_payload_to_latent(
                        residual_payload, entropy_model=intra_entropy_model,
                        params=intra_params, shape=latent_shape)
                    reconstructed_latent = decoded.to(device)
                else:
                    if previous_reconstruction is None:
                        raise mc.CausalityViolationError(
                            f"Frame {index} is a P-frame but no reconstructed reference exists.")
                    motion = mc.estimate_block_motion(
                        previous_reconstruction, frame,
                        block_size=block_size, search_range=search_range)
                    motion_payload = mc.encode_motion_payload(
                        motion, search_range=search_range, entropy_model=motion_entropy_model)
                    decoded_motion = mc.decode_motion_payload(
                        motion_payload, blocks, search_range=search_range,
                        entropy_model=motion_entropy_model)
                    warped = mc.warp_blocks(previous_reconstruction, decoded_motion,
                                            block_size=block_size)
                    reference_latent = model.encode(warped)

                    coded = codec.encode_residual(latent - reference_latent, reference_latent)
                    residual_payload, _ = encode_latent_to_payload(
                        coded, params=residual_params, entropy_model=residual_entropy_model)
                    decoded_coded, _ = decode_payload_to_latent(
                        residual_payload, entropy_model=residual_entropy_model,
                        params=residual_params, shape=latent_shape)
                    reconstructed_residual = codec.decode_residual(
                        decoded_coded.to(device), reference_latent)
                    reconstructed_latent = reference_latent + reconstructed_residual

                writer.append_frame(frame_type, motion_payload, residual_payload)
                reconstruction = model.decode(reconstructed_latent)
                previous_reconstruction = reconstruction
                encoder_latents.append(reconstructed_latent.detach().cpu())
                encoder_reconstructions.append(reconstruction.detach().cpu())
                records.append({
                    "index": index, "frame_type": mc.FRAME_TYPE_NAMES[frame_type],
                    "motion_bytes": len(motion_payload),
                    "residual_bytes": len(residual_payload),
                })

    container_bytes = Path(path).stat().st_size
    motion_total = sum(r["motion_bytes"] for r in records)
    residual_total = sum(r["residual_bytes"] for r in records)
    return {
        "path": str(path), "frame_count": frame_count, "gop_size": gop_size, "mode": "mc",
        "conditional": True, "rate_accounted": True,
        "frames": records,
        "i_frames": sum(1 for t in types if t == mc.FRAME_TYPE_I),
        "p_frames": sum(1 for t in types if t == mc.FRAME_TYPE_P),
        "i_frame_residual_bytes": sum(r["residual_bytes"] for r, t in zip(records, types)
                                      if t == mc.FRAME_TYPE_I),
        "p_frame_residual_bytes": sum(r["residual_bytes"] for r, t in zip(records, types)
                                      if t == mc.FRAME_TYPE_P),
        "motion_bytes": motion_total, "residual_bytes": residual_total,
        "payload_bytes": motion_total + residual_total,
        "container_bytes": container_bytes,
        "container_overhead_bytes": container_bytes - motion_total - residual_total,
        "latent_shape": latent_shape,
        "encoder_latents": torch.cat(encoder_latents, dim=0),
        "encoder_reconstructions": torch.cat(encoder_reconstructions, dim=0),
    }


@torch.no_grad()
def decode_sequence_conditional(mc, model, codec, path, *, intra_entropy_model,
                                residual_entropy_model, motion_entropy_model,
                                return_latents: bool = False):
    model.eval()
    codec.eval()
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

    reconstructions, decoded_latents = [], []
    previous_reconstruction = None
    with mc.deterministic_kernels():
        for index, (frame_type, motion_payload, residual_payload) in enumerate(reader):
            if frame_type == mc.FRAME_TYPE_I:
                latent, _ = decode_payload_to_latent(
                    residual_payload, entropy_model=intra_entropy_model,
                    params=reader.intra_params, shape=header.latent_shape)
                reconstructed_latent = latent.to(device)
            else:
                if previous_reconstruction is None:
                    raise mc.TemporalFormatError(
                        f"Frame {index} is a P-frame but no reference is available")
                motion = mc.decode_motion_payload(
                    motion_payload, header.block_grid, search_range=header.search_range,
                    entropy_model=motion_entropy_model)
                warped = mc.warp_blocks(previous_reconstruction, motion,
                                        block_size=header.block_size)
                reference_latent = model.encode(warped)
                coded, _ = decode_payload_to_latent(
                    residual_payload, entropy_model=residual_entropy_model,
                    params=reader.residual_params, shape=header.latent_shape)
                reconstructed_latent = reference_latent + codec.decode_residual(
                    coded.to(device), reference_latent)

            reconstruction = model.decode(reconstructed_latent)
            reconstructions.append(reconstruction)
            decoded_latents.append(reconstructed_latent.detach().cpu())
            previous_reconstruction = reconstruction

    frames = torch.cat(reconstructions, dim=0)
    if return_latents:
        return frames, torch.cat(decoded_latents, dim=0)
    return frames


if __name__ == "__main__":
    sys.exit(main())
