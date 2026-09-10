"""M11 - collect (or reload) the closed-loop residual symbols every M11 stage uses.

Collecting M10H residual symbols means running the full motion-compensated
codec closed-loop - block matching, warping, two network passes per frame -
over ~1,400 frames per rate point. On this machine that is hours of wall clock,
and the offline gate, the learned-model training and the coded-byte check all
need the SAME symbols. So they are collected once and cached.

A cache is only safe if a stale one cannot be mistaken for a fresh one. The key
therefore hashes everything the symbols depend on: the checkpoint's bytes, the
M10H module's source (so pre- and post-determinism-fix data can never mix), the
bit depth, every codec setting, the frame budgets and the exact sequence lists.
Because M11 phase 0 made the codec bit-reproducible across processes, a cache
hit is byte-identical to a fresh collection, not merely close to one.

The cache lives in the system temp directory by default, not in the repository
- it is large, derived, and would otherwise churn the OneDrive sync this project
lives under.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

DATA_VERSION = 1
DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "nvc_m11_cache"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_key(*, checkpoint: Path, bits: int, quant_mode: str, gop: int, block_size: int,
              search_range: int, calibration_frames: int, train_frames: int,
              val_frames_per_sequence: int, train_ids, val_ids) -> str:
    payload = {
        "version": DATA_VERSION,
        "checkpoint": _file_digest(Path(checkpoint)),
        "m10h_source": _file_digest(Path(__file__).parent / "m10h_motion_compensation.py"),
        "bits": bits, "quant_mode": quant_mode, "gop": gop, "block_size": block_size,
        "search_range": search_range, "calibration_frames": calibration_frames,
        "train_frames": train_frames, "val_frames_per_sequence": val_frames_per_sequence,
        "train_ids": list(train_ids), "val_ids": list(val_ids),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_or_collect(model, *, checkpoint: Path, manifest: Path, bits: int, device,
                    quant_mode: str = "per_channel", gop: int = 10, block_size: int = 16,
                    search_range: int = 16, calibration_frames: int = 400,
                    train_frames: int = 600, val_frames_per_sequence: int = 40,
                    cache_dir: Path | None = DEFAULT_CACHE_DIR, log=print) -> dict[str, Any]:
    """TRAIN symbols (first `train_frames` frames, as M10K/M10L used) and VAL
    symbols (`val_frames_per_sequence` from EVERY validation sequence), plus the
    calibration they were produced under. TEST is never read here."""
    from nvc.evaluation.sequences import discover_sequences

    mc = _load_script("m10h_motion_compensation")
    ce = _load_script("m10j_conditional_entropy")
    train_sequences = discover_sequences(manifest, split="train")
    val_sequences = discover_sequences(manifest, split="val",
                                       max_frames_per_sequence=val_frames_per_sequence)
    key = cache_key(checkpoint=checkpoint, bits=bits, quant_mode=quant_mode, gop=gop,
                    block_size=block_size, search_range=search_range,
                    calibration_frames=calibration_frames, train_frames=train_frames,
                    val_frames_per_sequence=val_frames_per_sequence,
                    train_ids=[s.sequence_id for s in train_sequences],
                    val_ids=[s.sequence_id for s in val_sequences])
    path = None if cache_dir is None else Path(cache_dir) / f"m11_data_{bits}bit_{key[:20]}.pt"
    if path is not None and path.is_file():
        data = torch.load(path, weights_only=False)
        if data.get("key") == key:
            log(f"    {bits}-bit data: cache hit {path.name}")
            data["from_cache"] = True
            return data

    started = time.perf_counter()
    log(f"    {bits}-bit data: collecting closed-loop symbols (no cache) ...")
    calibration = mc.calibrate_grids(
        model, train_sequences, bits=bits, mode=quant_mode, gop_size=gop,
        block_size=block_size, search_range=search_range, reference_mode="mc",
        max_frames=calibration_frames)
    train_symbols, train_references = ce.collect_training_symbols(
        mc, model, train_sequences, calibration, gop_size=gop, block_size=block_size,
        search_range=search_range, device=device, max_frames=train_frames)
    val_symbols, val_references, val_sequence_index = [], [], []
    for index, sequence in enumerate(val_sequences):
        symbols, references = ce.collect_training_symbols(
            mc, model, [sequence], calibration, gop_size=gop, block_size=block_size,
            search_range=search_range, device=device, max_frames=sequence.frame_count)
        val_symbols += symbols
        val_references += references
        val_sequence_index += [index] * len(symbols)

    data = {
        "key": key, "bits": bits, "calibration": calibration,
        "train_symbols": np.stack(train_symbols).astype(np.uint8),
        "train_references": np.stack(train_references).astype(np.float32),
        "val_symbols": np.stack(val_symbols).astype(np.uint8),
        "val_references": np.stack(val_references).astype(np.float32),
        "val_sequence_index": np.asarray(val_sequence_index, dtype=np.int64),
        "val_sequence_ids": [s.sequence_id for s in val_sequences],
        "provenance": {"train_frames": train_frames, "calibration_frames": calibration_frames,
                       "val_frames_per_sequence": val_frames_per_sequence, "split": "train/val",
                       "collect_seconds": time.perf_counter() - started},
    }
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)
        log(f"    {bits}-bit data: cached to {path}")
    data["from_cache"] = False
    return data


def split_validation(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """VAL-A / VAL-B masks over validation P-frames, split BY SEQUENCE.

    VAL-A tunes (smoothing strength, context choice); VAL-B reports. Splitting by
    sequence rather than by frame keeps temporally adjacent frames - which are
    near-duplicates - from landing on both sides of the split.
    """
    sequence = data["val_sequence_index"]
    selection = sequence % 2 == 0
    return selection, ~selection
