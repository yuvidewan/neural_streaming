"""M19 - characterize the SHAPE of the reference error (M17's real-vs-oracle
discrepancy), not just its magnitude (M18 already showed magnitude/MSE
improvements do not reliably translate into byte savings).

One comprehensive collection pass per bit depth over VAL-B (the same sample
M17 used) computes everything Phases A-I need in a single GPU sweep -
reusing M17's exact real/oracle chain construction (`diagnose_sequence`
-style, verified against `encode_multi` the same way M16/M17/M18 all did)
so results are directly comparable to M17's own recorded numbers, never a
re-derivation that could silently drift.

DEFINITIONS (Phase A, kept separate throughout - never conflated):
  R_real   = model.decode(reconstructed_latent) - the REAL, deployed
             previous-frame reconstruction (bit-depth-dependent).
  R_oracle = model.decode(model.encode(raw_previous_frame)) - M16/M17's own
             idealization, recomputed fresh from ground truth every step.
  E_pixel  = R_real - R_oracle                                (pixel space)
  Z_real   = model.encode(R_real);  Z_oracle = model.encode(R_oracle)
  E_latent = Z_real - Z_oracle                                (latent space)

Z_real/Z_oracle are the DIRECT encodings of the two candidate reference
frames - decoupled from motion/warping, which Phase G treats as a separate
covariate. The motion-compensated `reference_latent_{real,oracle}` used for
actual residual coding (`= encode(warp(R_{real,oracle}, mv_{real,oracle}))`)
is ALSO computed, for the codebook-routing/G16 analyses that need it.

Run:
  ./.venv/Scripts/python.exe scripts/m19_reference_error_diagnostic.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from nvc.compression.codec import (
    decode_payload_to_latent, encode_latent_to_payload, latent_to_symbols, symbols_to_latent,
)
from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m19_reference_error_audit")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M11_DIR = Path("outputs/m11_autoregressive_entropy")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
RATE_POINTS = (5, 4, 3)
M11_G16_GROUP_SIZE = 16


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _err_stats(x: torch.Tensor) -> dict[str, float]:
    return {"mean": float(x.mean()), "mae": float(x.abs().mean()),
           "rms": float((x ** 2).mean().sqrt()), "variance": float(x.var())}


def _autocorr_lag1(x: torch.Tensor, dim: int) -> float:
    """Pearson correlation between x and x shifted by one pixel along `dim`
    (2=rows/vertical, 3=cols/horizontal for a [1,C,H,W] tensor)."""
    if x.shape[dim] < 2:
        return 0.0
    a = x.narrow(dim, 0, x.shape[dim] - 1).flatten()
    b = x.narrow(dim, 1, x.shape[dim] - 1).flatten()
    a, b = a - a.mean(), b - b.mean()
    denom = (a.norm() * b.norm())
    return float((a @ b) / denom) if denom > 0 else 0.0


def _low_high_freq_energy(x: torch.Tensor) -> tuple[float, float]:
    """FFT-based low/high frequency energy split of a [1,C,H,W] error map
    (summed over channels first) - low = inner quarter of the spectrum
    (both axes), high = everything else."""
    gray = x.mean(dim=1, keepdim=False)[0]  # [H, W]
    spectrum = torch.fft.fftshift(torch.fft.fft2(gray))
    power = (spectrum.abs() ** 2)
    h, w = power.shape
    cy, cx = h // 2, w // 2
    ry, rx = max(1, h // 8), max(1, w // 8)
    mask_low = torch.zeros_like(power, dtype=torch.bool)
    mask_low[max(0, cy - ry):cy + ry, max(0, cx - rx):cx + rx] = True
    low = float(power[mask_low].sum())
    high = float(power[~mask_low].sum())
    return low, high


def _edge_flat_texture_masks(reference_frame: torch.Tensor) -> dict[str, torch.Tensor]:
    """Deterministic, non-learned Sobel-gradient-magnitude tercile split of
    `reference_frame` [1,3,H,W] into flat/texture/edge boolean masks [H,W] -
    a fixed, reproducible partition rule, never TRAIN-fit."""
    gray = reference_frame.mean(dim=1, keepdim=True)  # [1,1,H,W]
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=reference_frame.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    magnitude = (gx ** 2 + gy ** 2).sqrt()[0, 0]  # [H, W]
    flat_thresh, edge_thresh = torch.quantile(
        magnitude.flatten(), torch.tensor([1.0 / 3, 2.0 / 3], device=magnitude.device))
    return {"flat": magnitude <= flat_thresh, "texture": (magnitude > flat_thresh) & (magnitude <= edge_thresh),
           "edge": magnitude > edge_thresh}


def _block_boundary_mask(height: int, width: int, block_size: int, device) -> torch.Tensor:
    rows = torch.arange(height, device=device)
    cols = torch.arange(width, device=device)
    boundary_rows = (rows % block_size == 0) | (rows % block_size == block_size - 1)
    boundary_cols = (cols % block_size == 0) | (cols % block_size == block_size - 1)
    return boundary_rows.view(-1, 1) | boundary_cols.view(1, -1)


@torch.no_grad()
def diagnose_reference_error(mc, ma, m13, mk, model, model11, assign_codebook, coding_codebook,
                             sequence, *, bits, intra_params, intra_entropy_model,
                             residual_params, zero, gop_size, block_size, search_range,
                             device, seed) -> list[dict[str, Any]]:
    frames = sequence.load_frames()
    frame_count = frames.shape[0]
    types = mc.gop_frame_types(frame_count, gop_size)

    real_previous = None
    oracle_previous_raw = None
    latent_shape: tuple[int, ...] | None = None
    rows: list[dict[str, Any]] = []

    with mc.deterministic_kernels():
        for index in range(frame_count):
            frame = frames[index:index + 1].to(device)
            latent = model.encode(frame)
            latent_shape = tuple(latent.shape[1:])
            if types[index] == mc.FRAME_TYPE_I:
                payload, _ = encode_latent_to_payload(latent, params=intra_params,
                                                      entropy_model=intra_entropy_model)
                decoded, _ = decode_payload_to_latent(
                    payload, entropy_model=intra_entropy_model, params=intra_params,
                    shape=latent_shape)
                real_previous = model.decode(decoded.to(device))
                oracle_previous_raw = frame
                continue

            is_boundary = (index - 1) % gop_size == 0
            gop_position = index % gop_size

            r_real, r_oracle = real_previous, model.decode(model.encode(oracle_previous_raw))
            e_pixel = r_real - r_oracle
            z_real, z_oracle = model.encode(r_real), model.encode(r_oracle)
            e_latent = z_real - z_oracle

            # --- Phase B: spatial structure -------------------------------------------------
            autocorr_h = _autocorr_lag1(e_pixel, dim=3)
            autocorr_v = _autocorr_lag1(e_pixel, dim=2)
            low_freq, high_freq = _low_high_freq_energy(e_pixel)
            masks = _edge_flat_texture_masks(r_oracle)
            gray_error = e_pixel.mean(dim=1)[0].abs()  # [H, W]
            spatial_region_error = {name: float(gray_error[mask].mean()) if mask.any() else 0.0
                                    for name, mask in masks.items()}
            boundary_mask = _block_boundary_mask(gray_error.shape[0], gray_error.shape[1],
                                                 block_size, device)
            boundary_error = float(gray_error[boundary_mask].mean())
            interior_error = float(gray_error[~boundary_mask].mean()) if (~boundary_mask).any() else 0.0

            # --- REAL closed-loop continuation (identical to encode_multi) -------------------
            mv_real = mc.estimate_block_motion(r_real, frame, block_size=block_size,
                                               search_range=search_range)
            warped_real = mc.warp_blocks(r_real, mv_real, block_size=block_size)
            reference_latent_real = model.encode(warped_real)
            delta_real = latent - reference_latent_real
            symbols_real = latent_to_symbols(delta_real, residual_params).reshape(latent_shape)

            # --- ORACLE continuation (diagnostic only) ---------------------------------------
            mv_oracle = mc.estimate_block_motion(r_oracle, frame, block_size=block_size,
                                                 search_range=search_range)
            warped_oracle = mc.warp_blocks(r_oracle, mv_oracle, block_size=block_size)
            reference_latent_oracle = model.encode(warped_oracle)
            delta_oracle = latent - reference_latent_oracle
            symbols_oracle = latent_to_symbols(delta_oracle, residual_params).reshape(latent_shape)

            # --- SHUFFLED control: same E_pixel magnitude distribution, spatial structure destroyed --
            generator = torch.Generator(device="cpu").manual_seed(seed * 100000 + index)
            flat_e = e_pixel[0].reshape(e_pixel.shape[1], -1)
            perm = torch.randperm(flat_e.shape[1], generator=generator)
            e_pixel_shuffled = flat_e[:, perm].reshape(e_pixel.shape)
            r_shuffled = (r_oracle + e_pixel_shuffled).clamp(0.0, 1.0)
            mv_shuffled = mc.estimate_block_motion(r_shuffled, frame, block_size=block_size,
                                                   search_range=search_range)
            warped_shuffled = mc.warp_blocks(r_shuffled, mv_shuffled, block_size=block_size)
            reference_latent_shuffled = model.encode(warped_shuffled)
            delta_shuffled = latent - reference_latent_shuffled
            symbols_shuffled = latent_to_symbols(delta_shuffled, residual_params).reshape(latent_shape)

            # --- G16 rows / codebook assignment / actual bytes, all three variants -----------
            def _code(reference, symbols):
                payload, ideal = m13.encode_frame_recalibrated(
                    model11, assign_codebook, coding_codebook, reference, symbols, zero, bits=bits)
                target = torch.from_numpy(np.asarray(symbols, dtype=np.int64))[None].to(device)
                model_rows = ma._rows(model11.log_probabilities(
                    reference, model11.planes(target, zero.to(device))))
                table_index = assign_codebook.assign_tensor(model_rows)
                probabilities = coding_codebook.probabilities[
                    table_index, np.asarray(symbols, dtype=np.int64).reshape(-1)]
                return payload, ideal, table_index, probabilities

            payload_real, ideal_real, table_real, prob_real = _code(reference_latent_real, symbols_real)
            payload_oracle, ideal_oracle, table_oracle, prob_oracle = _code(
                reference_latent_oracle, symbols_oracle)
            payload_shuffled, ideal_shuffled, _, _ = _code(reference_latent_shuffled, symbols_shuffled)

            channels = latent_shape[0]
            e_latent_flat = e_latent[0].reshape(channels, -1)  # [C, HW]
            channel_mae = e_latent_flat.abs().mean(dim=1).cpu().numpy()
            channel_rms = (e_latent_flat ** 2).mean(dim=1).sqrt().cpu().numpy()
            symbols_real_flat = symbols_real.reshape(channels, -1)
            symbols_oracle_flat = symbols_oracle.reshape(channels, -1)
            table_real_c = table_real.reshape(channels, -1)
            table_oracle_c = table_oracle.reshape(channels, -1)
            channel_churn = (table_real_c != table_oracle_c).mean(axis=1)
            channel_residual_std = (delta_real[0].reshape(channels, -1)).std(dim=1).cpu().numpy()

            assignment_changed = (table_real != table_oracle)
            symbol_changed = (symbols_real.reshape(-1) != symbols_oracle.reshape(-1))
            # Per-(channel, position) magnitude, channel-major flat - matches
            # table_real/symbols_real's own flat order exactly (ma._rows'
            # "C-major like the coder" convention), so index i here IS the
            # error at the exact symbol table_real[i]/symbols_real.reshape(-1)[i]
            # routes - never a channel-averaged stand-in.
            error_magnitude_per_symbol = e_latent_flat.abs().reshape(-1).cpu().numpy()

            code_len_real = -np.log2(np.maximum(prob_real, 1e-300))
            code_len_oracle = -np.log2(np.maximum(prob_oracle, 1e-300))
            delta_code_len = code_len_real - code_len_oracle

            reconstructed_latent = reference_latent_real + symbols_to_latent(
                symbols_real.reshape(-1), latent_shape, residual_params).to(device)
            reconstruction = model.decode(reconstructed_latent)

            rows.append({
                "sequence_id": sequence.sequence_id, "index": index, "is_boundary": is_boundary,
                "gop_position": gop_position,
                "pixel_error": _err_stats(e_pixel), "latent_error": _err_stats(e_latent),
                "autocorr_h": autocorr_h, "autocorr_v": autocorr_v,
                "low_freq_energy": low_freq, "high_freq_energy": high_freq,
                "spatial_region_error": spatial_region_error,
                "boundary_error": boundary_error, "interior_error": interior_error,
                "channel_mae": channel_mae, "channel_rms": channel_rms,
                "channel_churn": channel_churn, "channel_residual_std": channel_residual_std,
                "assignment_changed_fraction": float(assignment_changed.mean()),
                "symbol_changed_fraction": float(symbol_changed.mean()),
                "error_magnitude_per_symbol": error_magnitude_per_symbol,
                "assignment_changed_per_symbol": assignment_changed.astype(np.int8),
                "symbol_changed_per_symbol": symbol_changed.astype(np.int8),
                "delta_code_len_per_symbol": delta_code_len,
                "code_len_real_per_symbol": code_len_real, "code_len_oracle_per_symbol": code_len_oracle,
                "real_bytes": len(payload_real), "oracle_bytes": len(payload_oracle),
                "shuffled_bytes": len(payload_shuffled),
                "real_ideal_bits": ideal_real, "oracle_ideal_bits": ideal_oracle,
                "shuffled_ideal_bits": ideal_shuffled,
                "sad_reference": float((frame - warped_real).abs().mean()),
                "motion_magnitude": float(mv_real.abs().float().mean()),
                "residual_abs_mean": float(delta_real.abs().mean()),
            })

            real_previous = reconstruction
            oracle_previous_raw = frame

    return rows


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M19: characterize the shape of the M17 reference error.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=DEFAULT_M11_DIR)
    parser.add_argument("--m10k-dir", type=Path, default=DEFAULT_M10K_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--val-sequences", type=int, default=4)
    parser.add_argument("--val-frames-per-sequence", type=int, default=None)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=None)
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
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    md = _load_script("m11_data")
    cx = _load_script("m11_causal_context")
    mk = _load_script("m10k_learned_entropy")
    m13 = _load_script("m13_recalibration")
    ev = _load_script("m10l_evaluate")
    ev_m11 = _load_script("m11_evaluate")
    gate_script = _load_script("m12_spatial_offline_gate")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    val_sequences = discover_sequences(args.manifest, split="val",
                                       max_frames_per_sequence=args.val_frames_per_sequence)
    val_b = val_sequences[1::2][:args.val_sequences]
    train_full = discover_sequences(args.manifest, split="train")

    print("=" * 122, flush=True)
    print("M19 - REFERENCE ERROR SHAPE DIAGNOSTIC")
    print("=" * 122)
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: dict[int, list[dict[str, Any]]] = {}
    identities: dict[int, dict[str, str]] = {}

    for bits in args.rate_points:
        print(f"\n  ---- {bits}-bit ----", flush=True)
        started = time.perf_counter()
        calibration = mc.calibrate_grids(
            model, train_full, bits=bits, mode="per_channel", gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, reference_mode="mc",
            max_frames=args.calibration_frames)
        signature = ev.calibration_signature(calibration, bits=bits,
                                             calibration_frames=args.calibration_frames,
                                             quant_mode="per_channel")
        intra_params, intra_entropy_model = calibration["intra_params"], calibration["intra_entropy_model"]
        residual_params = calibration["residual_params"]

        data = md.load_or_collect(model, checkpoint=args.checkpoint, manifest=args.manifest,
                                  bits=bits, device=device, cache_dir=cache_dir, log=lambda m: None)
        train_symbols = data["train_symbols"].astype(np.int64)
        val_symbols = data["val_symbols"].astype(np.int64)
        select_mask, _ = md.split_validation(data)
        channels = train_symbols.shape[1]
        zero = torch.from_numpy(cx.zero_symbols(residual_params, channels))
        m10k, m10k_checkpoint = mk.load_entropy_model(
            args.m10k_dir / f"learned_entropy_{bits}bit.pt", device=device)
        m10k_identity = ev._m10k_model_identity(m10k_checkpoint["model_state_dict"])
        model11, checkpoint11 = ma.load_model(args.m11_dir / f"m11_G16_entropy_{bits}bit.pt",
                                              device=device)
        ev_m11.check_provenance(checkpoint11, signature=signature, bits=bits,
                                group_size=M11_G16_GROUP_SIZE,
                                context_definition_id=ma.context_definition_id(M11_G16_GROUP_SIZE),
                                m10k_identity=m10k_identity)
        assign_codebook = ml.SharedCodebook.from_dict(json.loads(
            (args.m11_dir / f"m11_G16_codebook_{bits}bit_K512.json").read_text(encoding="utf-8")))
        train_k = gate_script.m11_g16_prototype_indices(model11, assign_codebook, train_symbols,
                                                        data["train_references"], zero, device=device)
        val_k = gate_script.m11_g16_prototype_indices(model11, assign_codebook, val_symbols,
                                                      data["val_references"], zero, device=device)
        flat = lambda a: a.reshape(-1)
        frequencies, strength, _, _ = m13.fit_recalibrated_frequencies(
            assign_codebook, flat(train_symbols), flat(train_k),
            flat(val_symbols[select_mask]), flat(val_k[select_mask]), alphabet=2 ** bits)
        coding_codebook = m13.build_recalibrated_codebook(
            assign_codebook, frequencies, provenance={"bits": bits, "strength": strength})
        residual_identity = ma.model_identity(model11, m10k_identity=m10k_identity,
                                              calibration_signature=signature, bits=bits,
                                              codebook=coding_codebook)
        identities[bits] = {"residual_identity": residual_identity.hex(),
                            "assign_codebook_id": assign_codebook.codebook_id().hex(),
                            "coding_codebook_id": coding_codebook.codebook_id().hex()}

        rows: list[dict[str, Any]] = []
        for sequence in val_b:
            rows.extend(diagnose_reference_error(
                mc, ma, m13, mk, model, model11, assign_codebook, coding_codebook, sequence,
                bits=bits, intra_params=intra_params, intra_entropy_model=intra_entropy_model,
                residual_params=residual_params, zero=zero, gop_size=args.gop,
                block_size=args.block_size, search_range=args.search_range, device=device,
                seed=args.seed))
        print(f"    diagnosed {len(rows)} VAL-B P-frames ({time.perf_counter() - started:.1f}s), "
             f"residual identity {residual_identity.hex()}", flush=True)

        total_real = sum(r["real_bytes"] for r in rows)
        total_oracle = sum(r["oracle_bytes"] for r in rows)
        total_shuffled = sum(r["shuffled_bytes"] for r in rows)
        churn = np.mean([r["assignment_changed_fraction"] for r in rows])
        print(f"    bytes: real={total_real:,} oracle={total_oracle:,} shuffled={total_shuffled:,}  "
             f"real->oracle gain={((total_real - total_oracle) / total_real * 100):+.4f}%  "
             f"real->shuffled gain={((total_real - total_shuffled) / total_real * 100):+.4f}%  "
             f"mean assignment churn={churn * 100:.2f}%", flush=True)

        all_rows[bits] = rows

    # Persist raw per-frame rows (numpy arrays -> lists) for the analysis pass.
    for bits, rows in all_rows.items():
        serializable = []
        for r in rows:
            row_copy = dict(r)
            for key, value in row_copy.items():
                if isinstance(value, np.ndarray):
                    row_copy[key] = value.tolist()
            serializable.append(row_copy)
        path = args.output_dir / f"m19_raw_{bits}bit.json"
        path.write_text(json.dumps(serializable, default=str), encoding="utf-8")
        print(f"  wrote {path}")

    (args.output_dir / "m19_identities.json").write_text(
        json.dumps(identities, indent=2), encoding="utf-8")
    print(f"\nRaw data + identities written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
