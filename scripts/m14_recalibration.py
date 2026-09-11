"""M14 Phase B/D - collect bigger/full-TRAIN symbol samples for the intra and
motion entropy tables UNDER THE EXISTING FROZEN PARAMETERS, and refit
empirical frequency tables from them.

WHY A SEPARATE COLLECTOR (not just calling `calibrate_grids` again with a
bigger `max_frames`)
--------------------------------------------------------------------------
`m10h_motion_compensation.calibrate_grids` does two things at once: fits the
QUANTIZATION GRID (`intra_params`/`residual_params`, from percentiles of
whatever latents it collects) AND fits the entropy tables from the same
pass. Calling it again with a bigger frame budget would refit the
quantization grid too - shifting percentile bounds slightly - which would
change SYMBOLS, violating M14's frozen-quantizer/fixed-symbol invariant
(the same invariant M13 preserved for the residual codebook). The functions
below reuse `calibrate_grids`'s SAME primitives
(`estimate_block_motion`/`warp_blocks`/`motion_to_symbols`/`gop_frame_types`/
`deterministic_kernels`/`latent_to_symbols`, all imported unmodified) but
quantize intra latents under a SUPPLIED, FROZEN `intra_params` instead of
fitting a new one, and only ever return TABLES/SYMBOLS - never a
quantization grid.

Motion needs no such care: `motion_to_symbols` depends only on the frozen
motion estimator/block_size/search_range, never on any entropy table or
quantization grid - collecting more motion samples cannot change a single
motion symbol.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.codec import latent_to_symbols
from nvc.compression.entropy_model import EmpiricalEntropyModel


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- symbol collection under FROZEN params ---------------------------------------------


@torch.no_grad()
def collect_intra_symbols(model, sequences, *, intra_params, max_frames: int,
                          device) -> np.ndarray:
    """Every frame's latent (I- and P-typed alike - matching
    `calibrate_grids`'s own intra loop, since the intra table must cover the
    FULL latent distribution, not just I-frame-typed positions), quantized
    under the SUPPLIED, FROZEN `intra_params`. Returns [N, C, H*W] int64,
    channel-major per frame (matching the coder's own symbol order).
    """
    model.eval()
    symbols = []
    seen = 0
    for sequence in sequences:
        frames = sequence.load_frames()
        for index in range(frames.shape[0]):
            if seen >= max_frames:
                break
            latent = model.encode(frames[index:index + 1].to(device))
            symbols.append(latent_to_symbols(latent, intra_params).reshape(latent.shape[1], -1))
            seen += 1
        if seen >= max_frames:
            break
    return np.stack(symbols)


@torch.no_grad()
def collect_motion_symbols(mc, model, sequences, *, block_size: int, search_range: int,
                           gop_size: int, max_frames: int, reference_mode: str,
                           device) -> np.ndarray:
    """The SAME reference-chain walk `calibrate_grids`'s second loop uses
    (advancing from the TRUE latent, exactly as documented there), but
    discarding residuals - only (dy, dx) motion symbols are collected.
    Returns [N, 2, blocks] int64 (dy row, dx row - matching
    `motion_to_symbols`'s own layout).
    """
    model.eval()
    motion_symbol_frames: list[np.ndarray] = []
    seen = 0
    with mc.deterministic_kernels():
        for sequence in sequences:
            frames = sequence.load_frames()
            types = mc.gop_frame_types(frames.shape[0], gop_size)
            previous_reconstruction = None
            for index in range(frames.shape[0]):
                if seen >= max_frames:
                    break
                frame = frames[index:index + 1].to(device)
                latent = model.encode(frame)
                if types[index] == mc.FRAME_TYPE_I:
                    previous_reconstruction = model.decode(latent)
                elif previous_reconstruction is not None:
                    if reference_mode == "prev":
                        pass  # no motion coded in "prev" mode
                    else:
                        motion = mc.estimate_block_motion(
                            previous_reconstruction, frame,
                            block_size=block_size, search_range=search_range)
                        motion_symbol_frames.append(
                            mc.motion_to_symbols(motion, search_range=search_range).reshape(2, -1))
                    previous_reconstruction = model.decode(latent)
                seen += 1
            if seen >= max_frames:
                break
    if not motion_symbol_frames:
        raise ValueError("no motion frames collected")
    return np.stack(motion_symbol_frames)


# --- fitting and comparison -------------------------------------------------------------


def fit_empirical(symbols: np.ndarray, *, bits: int, num_tables: int) -> EmpiricalEntropyModel:
    """Exactly the SAME function/convention `calibrate_grids` already uses
    (`EmpiricalEntropyModel.from_symbols`, Laplace smoothing) - kept
    identical per Phase B's "same smoothing/normalization conventions"
    instruction; no new smoothing scheme is introduced for these tables."""
    return EmpiricalEntropyModel.from_symbols(symbols, bits=bits, num_tables=num_tables)


def held_out_bits_per_symbol(model: EmpiricalEntropyModel, symbols: np.ndarray,
                             table_index: np.ndarray) -> float:
    """-log2 P(true symbol) under `model`, averaged over `symbols` - the
    same estimator M11/M12/M13 used throughout (`parent_bits`-equivalent)."""
    probabilities = model.probabilities()
    flat_symbols = np.asarray(symbols).reshape(-1)
    flat_table = np.asarray(table_index).reshape(-1)
    total = float(-np.sum(np.log2(np.maximum(probabilities[flat_table, flat_symbols], 1e-300))))
    return total / flat_symbols.size


def intra_table_index(num_tables: int, plane: int, frames: int) -> np.ndarray:
    """channel c owns table c, tiled over every frame - matches
    `EmpiricalEntropyModel.from_symbols`'s own table-per-channel convention."""
    return np.tile(np.repeat(np.arange(num_tables, dtype=np.int64), plane), frames)


def motion_table_index(blocks: int, frames: int) -> np.ndarray:
    """table 0 = every dy symbol, table 1 = every dx symbol - matches
    `m10h_motion_compensation.motion_table_index`."""
    return np.tile(np.repeat(np.arange(2, dtype=np.int64), blocks), frames)
