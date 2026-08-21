"""Rate-distortion benchmark orchestration, aggregation, and result schema.

Runs a set of codec configurations over a set of sequences, aggregates the
per-sequence measurements into dataset-level figures, and writes a
self-describing, machine-readable record of exactly what produced them.

AGGREGATION METHODOLOGY
------------------------
Sequences have different lengths (DAVIS test sequences range from 52 to 104
frames), so averaging per-sequence averages would silently over-weight
short sequences. Every dataset-level figure here is therefore
frame/pixel-weighted:

- `aggregate_bpp`      = sum(bytes) * 8 / sum(pixels)      over all frames
- `mean_psnr`          = frame-weighted mean of per-frame PSNR
- `mean_msssim`        = frame-weighted mean of per-frame MS-SSIM
- `pooled_psnr`        = PSNR of the MSE pooled over every pixel
- `compression_ratio`  = sum(raw uint8 RGB bytes) / sum(compressed bytes)

`mean_psnr` and `pooled_psnr` are both reported because they answer
different questions and neither is universally "correct": the mean of
per-frame PSNR is what codec literature usually quotes, while pooling MSE
first is the error-weighted figure and is dominated by the worst frames.
They are labeled distinctly rather than one being silently chosen.

Nothing in this module invents a measurement. Every number written to
results.json comes from an actual encode/decode executed in this run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from nvc.compression.quantization import QuantizationParams, count_clipped
from nvc.evaluation.basic_metrics import psnr_from_mse
from nvc.evaluation.codecs import Codec, CodecResult
from nvc.evaluation.sequences import BenchmarkSequence

# Calibration derived from a *different* checkpoint clips far more latent
# values than a matched one. Measured on this project: a matched pair clips
# ~0.09% of values, while the DAVIS calibration applied to the Vimeo
# checkpoint clips ~17.9%. The percentile calibration targets ~0.2% by
# construction, so this threshold sits an order of magnitude above the
# design target and an order of magnitude below the known-bad case.
CALIBRATION_CLIP_WARNING_PERCENT = 2.0

METHODOLOGY_NOTE = (
    "NVC currently operates as an intra-only, frame-independent neural codec, "
    "while H.264 and H.265 exploit temporal redundancy through inter-frame "
    "prediction. Unless a configuration is marked intra_only, the classical "
    "codecs therefore benefit from a structural advantage that is unrelated to "
    "transform quality. Compression ratios are against raw uint8 RGB storage "
    "(frames x width x height x 3), which is not equivalent to uncompressed "
    "YUV video. H.264/H.265 additionally subsample chroma when pix_fmt is "
    "yuv420p, whereas NVC codes full-resolution RGB."
)


class CalibrationMismatchError(RuntimeError):
    """The calibration does not fit the checkpoint being benchmarked.

    Encoding would still *run* - the .nvc entropy-model id check only
    compares the encoder and decoder side, and both read the same
    calibration file - so this failure mode is silent by default and would
    produce badly degraded, methodologically invalid numbers. Hence a hard
    stop rather than a warning.
    """


@dataclass
class BenchmarkRun:
    """Accumulates results and metadata for one benchmark invocation."""

    output_dir: Path
    metadata: dict
    results: list[CodecResult]

    def add(self, result: CodecResult) -> None:
        self.results.append(result)


def check_calibration_fit(
    model: torch.nn.Module,
    sequences: list[BenchmarkSequence],
    params: QuantizationParams,
    *,
    device: torch.device,
    max_frames: int = 16,
) -> dict:
    """Measure how much a calibration clips on the model actually being used.

    Empirical rather than metadata-based on purpose: the calibration file
    records the checkpoint *path* it was built from, and checkpoints get
    renamed (this project's `best.pt` became `davis_baseline_best.pt`), so
    a filename comparison produces false alarms and false reassurance in
    equal measure. Measuring the actual clipping rate tests the property
    that matters directly.
    """
    frames = []
    for sequence in sequences:
        for path in sequence.frame_paths:
            frames.append(path)
            if len(frames) >= max_frames:
                break
        if len(frames) >= max_frames:
            break

    from nvc.data.image_io import read_image_as_tensor

    batch = torch.stack([read_image_as_tensor(p) for p in frames]).to(device)
    with torch.no_grad():
        latents = model.encode(batch)

    clipping = count_clipped(latents, params.to(latents.device))
    return {
        "frames_probed": len(frames),
        "clipped_percent": clipping["clipped_percent"],
        "latent_min": float(latents.min()),
        "latent_max": float(latents.max()),
        "threshold_percent": CALIBRATION_CLIP_WARNING_PERCENT,
        "fits": clipping["clipped_percent"] <= CALIBRATION_CLIP_WARNING_PERCENT,
    }


def require_calibration_fit(fit: dict, *, checkpoint: str, calibration: str) -> None:
    """Raise unless the calibration actually fits the checkpoint."""
    if fit["fits"]:
        return
    raise CalibrationMismatchError(
        f"The calibration does not fit this checkpoint: "
        f"{fit['clipped_percent']:.2f}% of latent values fall outside its quantization "
        f"range (threshold {fit['threshold_percent']:.1f}%; a matched pair clips well "
        f"under 1%).\n"
        f"  checkpoint:  {checkpoint}\n"
        f"  calibration: {calibration}\n\n"
        "This happens when a calibration built for one checkpoint is used with "
        "another - the quantization ranges and entropy tables are model-specific. "
        "Encoding would still run and would silently produce badly degraded, "
        "invalid results, so it is refused here.\n\n"
        "Fix: generate a calibration for THIS checkpoint (calibration reads the "
        "training split only, never the evaluation frames):\n"
        "  python scripts\\calibrate_quantizer.py --checkpoint <this checkpoint> "
        "--bits <bits> --output <new calibration path>\n\n"
        "Or pass --allow-calibration-mismatch to proceed anyway (results will NOT "
        "be methodologically valid and are labeled as such in the output)."
    )


def file_sha256(path: str | Path) -> str | None:
    """SHA-256 of a file, for traceability. None when unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metadata(
    *,
    dataset: str,
    sequences: list[BenchmarkSequence],
    codecs: list[Codec],
    checkpoint: Path | None,
    calibration: Path | None,
    device: torch.device,
    seed: int,
    ffmpeg_version: str | None,
    extra: dict | None = None,
) -> dict:
    """Everything needed to trace a result back to what produced it."""
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "methodology_note": METHODOLOGY_NOTE,
        "dataset": dataset,
        "sequence_ids": [s.sequence_id for s in sequences],
        "frame_counts": {s.sequence_id: s.frame_count for s in sequences},
        "total_frames": sum(s.frame_count for s in sequences),
        "resolution": {
            "width": sequences[0].width if sequences else None,
            "height": sequences[0].height if sequences else None,
        },
        "codecs": [c.describe() for c in codecs],
        "seed": seed,
        "device": str(device),
        "environment": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "platform": platform.platform(),
            "ffmpeg_version": ffmpeg_version,
        },
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
        "calibration": str(calibration) if calibration else None,
        "calibration_sha256": file_sha256(calibration) if calibration else None,
    }
    try:
        from importlib.metadata import version

        metadata["project_version"] = version("nvc")
    except Exception:  # pragma: no cover - package may not be installed
        metadata["project_version"] = None

    if extra:
        metadata.update(extra)
    return metadata


def aggregate_results(results: list[CodecResult]) -> list[dict]:
    """Collapse per-sequence results into one row per codec configuration.

    Frame/pixel-weighted throughout - see this module's docstring for why
    averaging sequence averages would be wrong here.
    """
    groups: dict[tuple[str, str], list[CodecResult]] = {}
    for result in results:
        groups.setdefault((result.codec, result.configuration), []).append(result)

    aggregates: list[dict] = []
    for (codec, configuration), group in groups.items():
        total_bytes = sum(r.total_bytes for r in group)
        total_pixels = sum(r.total_pixels for r in group)
        total_frames = sum(r.frame_count for r in group)
        raw_bytes = total_pixels * 3

        # Frame-weighted: each sequence contributes in proportion to length.
        weighted_psnr = sum(r.mean_psnr * r.frame_count for r in group) / total_frames
        scored = [r for r in group if r.mean_msssim is not None]
        # Weight by the frames that actually produced an MS-SSIM score, not
        # the sequence's nominal frame_count - a result can have fewer
        # (e.g. some frames dropped below MS-SSIM's minimum spatial size),
        # and weighting by the nominal count would over-count its influence
        # on this figure relative to how many frames it actually scored.
        weighted_msssim = (
            sum(r.mean_msssim * (r.msssim_frame_count or r.frame_count) for r in scored)
            / sum((r.msssim_frame_count or r.frame_count) for r in scored)
            if scored else None
        )
        pooled_mse = sum(r.pooled_mse * r.total_pixels * 3 for r in group) / raw_bytes

        aggregates.append({
            "codec": codec,
            "codec_configuration": configuration,
            "sequences": len(group),
            "total_frames": total_frames,
            "total_bytes": total_bytes,
            "aggregate_bpp": total_bytes * 8 / total_pixels,
            "compression_ratio": raw_bytes / total_bytes,
            "mean_psnr": weighted_psnr,
            "mean_msssim": weighted_msssim,
            "pooled_psnr": float(psnr_from_mse(pooled_mse)),
            "pooled_mse": pooled_mse,
            "total_encode_seconds": sum(r.encode_seconds for r in group),
            "total_decode_seconds": sum(r.decode_seconds for r in group),
            "encode_seconds_per_frame": sum(r.encode_seconds for r in group) / total_frames,
            "decode_seconds_per_frame": sum(r.decode_seconds for r in group) / total_frames,
            **{
                key: value for key, value in group[0].details.items()
                if key in (
                    "encoder", "crf", "preset", "pix_fmt", "intra_only",
                    "temporal_prediction", "checkpoint", "quantization_bits",
                    "quantization_mode", "entropy_model_id",
                )
            },
        })

    aggregates.sort(key=lambda row: (row["codec"], row["aggregate_bpp"]))
    return aggregates


def create_run_directory(base_dir: str | Path, run_name: str | None = None) -> Path:
    """Timestamped run directory, so no earlier benchmark is overwritten."""
    base_dir = Path(base_dir)
    name = run_name or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = base_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    return run_dir


def _json_safe(value):
    """Recursively replace non-finite floats (inf/-inf/nan) with None.

    `json.dumps` happily emits `Infinity`/`NaN` for these by default - valid
    Python, but not valid JSON (RFC 8259), so any strict-mode parser (a
    browser results viewer, `jq`, most non-Python JSON libraries) fails to
    read the file. A pixel-perfect frame or sequence can genuinely produce
    an infinite PSNR (see `psnr_from_mse`), so this only affects how such a
    value is written to results.json, never the value used elsewhere.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def write_results(run_dir: Path, run: BenchmarkRun) -> dict[str, Path]:
    """Write results.json, results.csv, and metadata.json."""
    run_dir = Path(run_dir)
    rows = [result.to_row() for result in run.results]
    aggregates = aggregate_results(run.results) if run.results else []

    document = {
        "metadata": run.metadata,
        "per_sequence": rows,
        "aggregate": aggregates,
        "aggregation_methodology": {
            "aggregate_bpp": "sum(total_bytes) * 8 / sum(frames * width * height)",
            "mean_psnr": "frame-weighted mean of per-frame PSNR",
            "mean_msssim": "frame-weighted mean of per-frame MS-SSIM",
            "pooled_psnr": "PSNR derived from MSE pooled over every pixel of every frame",
            "compression_ratio": "sum(frames * width * height * 3) / sum(total_bytes)",
            "note": (
                "Sequence lengths differ, so sequence averages are never averaged "
                "directly; every dataset-level figure is frame- or pixel-weighted."
            ),
        },
    }

    json_path = run_dir / "results.json"
    json_path.write_text(json.dumps(_json_safe(document), indent=2), encoding="utf-8")

    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(run.metadata, indent=2), encoding="utf-8")

    csv_path = run_dir / "results.csv"
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)

    aggregate_csv_path = run_dir / "aggregate.csv"
    if aggregates:
        fieldnames = []
        for row in aggregates:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with aggregate_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(aggregates)

    return {
        "results_json": json_path,
        "results_csv": csv_path,
        "aggregate_csv": aggregate_csv_path,
        "metadata_json": metadata_path,
    }
