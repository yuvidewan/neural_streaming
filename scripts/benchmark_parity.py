"""PARITY_ROADMAP Stage 0 - the single-pass H.264/H.265 scoreboard.

Why this exists: the "H.264 needs 4.8x fewer bits" figure paired H.264 rows from
the August benchmark (`outputs/benchmarks/vimeo_vs_h264_h265_davis/`, whose NVC
arm was intra-only and pre-motion-compensation) with M22 numbers measured by a
different script. The two sides also used DIFFERENT PSNR definitions:

  old H.264 harness   mean of per-frame PSNR over all 719 frames
  M21/M22 NVC code    per-sequence pooled-MSE PSNR, averaged over 9 sequences

Mean-of-dB is never lower than pooled-MSE dB (Jensen), so that comparison was not
like-for-like. This script measures every codec in ONE pass:

  * the same frames - loaded once per sequence, fed to NVC as tensors and to
    FFmpeg as raw rgb24 bytes over a pipe (bit-identical to the PNGs on disk);
  * the same decoded-pixel convention - every reconstruction is scored as the
    uint8 frame a real decoder would deliver (NVC's float output is rounded;
    FFmpeg's output already is uint8);
  * the same metric function and the same aggregation, for every arm;
  * real bytes: the whole `.nvct` container for NVC, the whole `.mp4` for FFmpeg.

PRIMARY CONVENTION (fixed before any result was seen)
-----------------------------------------------------
Rate:     pooled BPP = total file bytes * 8 / total pixels, all 719 frames.
Quality:  per-frame PSNR (and MS-SSIM) averaged within each sequence, then
          averaged over the 9 sequences - the per-sequence reporting used by the
          learned-video-codec literature this project is benchmarking against.
Headline: piecewise-linear BD-rate (the project's settled `_bd_rate_linear`) of
          NVC against libx264 at its default configuration, on PSNR.

The three other conventions (all-frame mean, per-sequence pooled, global pooled)
are reported alongside as a sensitivity check, never substituted after the fact.

ARMS
----
  nvc_deployed     the frozen production codec (M10H + M11-G16 + K=512 + M13/M14),
                   rebuilt exactly as M22 Phase 19 rebuilt its baseline.
  nvc_m22          the same with M22's locked `symmetric_p01` residual grid and
                   its refitted stack (SHA256-verified checkpoints).
  h264 / h265      libx264 / libx265, preset medium, yuv420p, default GOP and
                   B-frames - the codecs as they are actually used.
  h264_lowdelay /  the same encoders forced into NVC's structure: an I-frame
  h265_lowdelay    every 10 frames, no B-frames, no scene-cut keyframes.

Both NVC arms must reproduce the byte totals M22 recorded on DAVIS TEST before
their numbers are used; a mismatch is reported and marks the run invalid.

Run:
  ./.venv/Scripts/python.exe scripts/benchmark_parity.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.evaluation.ffmpeg import (
    find_ffmpeg,
    ffmpeg_version,
    require_encoders,
)
from nvc.evaluation.perceptual_metrics import msssim
from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

PRIMARY_CONVENTION = "sequence_mean"
CONVENTIONS = ("sequence_mean", "frame_mean", "sequence_pooled", "global_pooled")
DEFAULT_CRFS = (20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44)
M22_CANDIDATE = "symmetric_p01"
FRAMERATE = 30


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------------
# Scoring - one function for every arm
# --------------------------------------------------------------------------------------------


def to_uint8_frames(frames: torch.Tensor) -> torch.Tensor:
    """Round a [N, 3, H, W] float [0, 1] tensor to the uint8 values a real
    decoder would hand to a display, returned as float [0, 1] again."""
    return torch.round(frames.clamp(0.0, 1.0) * 255.0) / 255.0


@torch.no_grad()
def score_frames(reconstructions: torch.Tensor, references: torch.Tensor, *,
                 device: torch.device, batch: int = 32) -> dict[str, list[float]]:
    """Per-frame MSE, PSNR and MS-SSIM for one sequence.

    MS-SSIM is computed one frame at a time (batch of 1 inside `msssim`'s
    size_average) so every value is a genuine per-frame score regardless of how
    the frames are chunked for speed.
    """
    if reconstructions.shape != references.shape:
        raise ValueError(f"shape mismatch: {tuple(reconstructions.shape)} vs "
                         f"{tuple(references.shape)}")
    mse_values, psnr_values, msssim_values = [], [], []
    for start in range(0, references.shape[0], batch):
        rec = reconstructions[start:start + batch].to(device, torch.float32)
        ref = references[start:start + batch].to(device, torch.float32)
        per_frame_mse = torch.mean((rec - ref) ** 2, dim=(1, 2, 3))
        for index in range(ref.shape[0]):
            value = float(per_frame_mse[index])
            mse_values.append(value)
            psnr_values.append(math.inf if value == 0.0 else 10.0 * math.log10(1.0 / value))
            msssim_values.append(float(msssim(rec[index:index + 1].clamp(0.0, 1.0),
                                              ref[index:index + 1])))
    return {"mse": mse_values, "psnr": psnr_values, "msssim": msssim_values}


def aggregate_quality(per_sequence: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """PSNR and MS-SSIM under every convention, from per-frame scores.

    `per_sequence` rows carry `mse`, `psnr` and `msssim` lists. Returns
    {convention: {"psnr": ..., "msssim": ...}}.
    """
    def finite_mean(values):
        finite = [v for v in values if math.isfinite(v)]
        return statistics.fmean(finite) if finite else math.inf

    def pooled_psnr(mse_values):
        mean = statistics.fmean(mse_values)
        return math.inf if mean == 0.0 else 10.0 * math.log10(1.0 / mean)

    all_mse = [v for row in per_sequence for v in row["mse"]]
    all_psnr = [v for row in per_sequence for v in row["psnr"]]
    all_msssim = [v for row in per_sequence for v in row["msssim"]]
    return {
        "sequence_mean": {
            "psnr": statistics.fmean(finite_mean(r["psnr"]) for r in per_sequence),
            "msssim": statistics.fmean(statistics.fmean(r["msssim"]) for r in per_sequence),
        },
        "frame_mean": {"psnr": finite_mean(all_psnr), "msssim": statistics.fmean(all_msssim)},
        "sequence_pooled": {
            "psnr": statistics.fmean(pooled_psnr(r["mse"]) for r in per_sequence),
            "msssim": statistics.fmean(statistics.fmean(r["msssim"]) for r in per_sequence),
        },
        "global_pooled": {"psnr": pooled_psnr(all_mse), "msssim": statistics.fmean(all_msssim)},
    }


def summarize_point(label: str, arm: str, setting: str, rows: list[dict[str, Any]],
                    extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """One operating point: pooled rate plus quality under every convention.
    Per-frame lists are reduced to per-sequence summaries before storage."""
    total_bytes = sum(r["bytes"] for r in rows)
    total_pixels = sum(r["pixels"] for r in rows)
    return {
        "label": label, "arm": arm, "setting": setting,
        "total_bytes": total_bytes, "total_pixels": total_pixels,
        "frames": sum(len(r["psnr"]) for r in rows),
        "bpp": total_bytes * 8 / total_pixels,
        "quality": aggregate_quality(rows),
        "encode_seconds": sum(r.get("encode_seconds", 0.0) for r in rows),
        "decode_seconds": sum(r.get("decode_seconds", 0.0) for r in rows),
        "per_sequence": [{
            "sequence": r["sequence"], "bytes": r["bytes"],
            "bpp": r["bytes"] * 8 / r["pixels"],
            "mean_psnr": statistics.fmean(r["psnr"]),
            "mean_msssim": statistics.fmean(r["msssim"]),
        } for r in rows],
        **(extra or {}),
    }


# --------------------------------------------------------------------------------------------
# Rate-distortion comparisons
# --------------------------------------------------------------------------------------------


def curve(points: list[dict[str, Any]], convention: str, metric: str) -> list[tuple[float, float]]:
    return sorted((p["bpp"], p["quality"][convention][metric]) for p in points)


def bd_rate(base: list[tuple[float, float]], test: list[tuple[float, float]]) -> float | None:
    """The project's settled piecewise-linear BD-rate (`m10b_evaluate`).
    Positive = `test` needs MORE bits than `base` for the same quality."""
    return _load_script("m10b_evaluate")._bd_rate_linear(base, test)


def rate_ratio_at_quality(base: list[tuple[float, float]], bpp: float,
                          quality: float) -> float | None:
    """How many times more bits a point at (`bpp`, `quality`) spends than the
    `base` curve needs for the same quality, interpolating the base curve
    piecewise-linearly in log-rate. None when `quality` is outside its range."""
    ordered = sorted(base, key=lambda p: p[1])
    qualities = np.array([p[1] for p in ordered])
    if not qualities.min() <= quality <= qualities.max():
        return None
    log_rate = float(np.interp(quality, qualities, np.log10([p[0] for p in ordered])))
    return bpp / 10 ** log_rate


def overlap(base: list[tuple[float, float]], test: list[tuple[float, float]]):
    low = max(min(q for _, q in base), min(q for _, q in test))
    high = min(max(q for _, q in base), max(q for _, q in test))
    return (low, high) if high > low else None


# --------------------------------------------------------------------------------------------
# Classical codecs over a raw pipe
# --------------------------------------------------------------------------------------------


class ClassicalArm:
    """libx264/libx265 at one structural configuration. Plain class, not a
    dataclass: scripts are loaded without sys.modules registration."""

    __slots__ = ("name", "encoder", "lowdelay", "gop")

    def __init__(self, name: str, encoder: str, *, lowdelay: bool, gop: int) -> None:
        self.name, self.encoder, self.lowdelay, self.gop = name, encoder, lowdelay, gop

    def encoder_arguments(self, crf: int) -> list[str]:
        arguments = ["-c:v", self.encoder, "-crf", str(crf), "-preset", "medium",
                     "-pix_fmt", "yuv420p"]
        if self.encoder == "libx264":
            if self.lowdelay:
                arguments += ["-g", str(self.gop), "-keyint_min", str(self.gop),
                              "-sc_threshold", "0", "-bf", "0"]
        elif self.encoder == "libx265":
            params = ["log-level=error"]
            if self.lowdelay:
                params += [f"keyint={self.gop}", f"min-keyint={self.gop}", "scenecut=0",
                           "bframes=0"]
            arguments += ["-x265-params", ":".join(params)]
        else:
            raise ValueError(f"unsupported encoder {self.encoder!r}")
        return arguments

    def describe(self) -> dict[str, Any]:
        return {"encoder": self.encoder, "preset": "medium", "pix_fmt": "yuv420p",
                "lowdelay": self.lowdelay,
                "structure": (f"I every {self.gop}, P only" if self.lowdelay
                              else "encoder default GOP and B-frames")}


def classical_arms(gop: int) -> list[ClassicalArm]:
    return [ClassicalArm("h264", "libx264", lowdelay=False, gop=gop),
            ClassicalArm("h265", "libx265", lowdelay=False, gop=gop),
            ClassicalArm("h264_lowdelay", "libx264", lowdelay=True, gop=gop),
            ClassicalArm("h265_lowdelay", "libx265", lowdelay=True, gop=gop)]


def ffmpeg_round_trip(frames_uint8: np.ndarray, arm: ClassicalArm, crf: int,
                      workdir: Path) -> dict[str, Any]:
    """Encode [N, H, W, 3] uint8 RGB through FFmpeg into an .mp4, measure the
    file, decode it back to uint8 RGB. The frame count is checked by exact byte
    length of the decoded raw stream, so a dropped or duplicated frame fails."""
    count, height, width, _ = frames_uint8.shape
    ffmpeg = find_ffmpeg()
    video = workdir / f"{arm.name}_crf{crf}.mp4"
    encode = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
              "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
              "-framerate", str(FRAMERATE), "-i", "pipe:0",
              *arm.encoder_arguments(crf), str(video)]
    started = time.perf_counter()
    completed = subprocess.run(encode, input=np.ascontiguousarray(frames_uint8).tobytes(),
                               capture_output=True, check=False)
    encode_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"{arm.name} crf{crf} encode failed: "
                           f"{completed.stderr.decode(errors='replace')[-800:]}")
    size = video.stat().st_size

    decode = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(video),
              "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    started = time.perf_counter()
    completed = subprocess.run(decode, capture_output=True, check=False)
    decode_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"{arm.name} crf{crf} decode failed: "
                           f"{completed.stderr.decode(errors='replace')[-800:]}")
    expected = count * height * width * 3
    if len(completed.stdout) != expected:
        raise RuntimeError(
            f"{arm.name} crf{crf} decoded {len(completed.stdout) // (height * width * 3)} "
            f"frames, source has {count}; per-frame metrics would be misaligned")
    video.unlink()
    decoded = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(count, height, width, 3)
    return {"bytes": size, "decoded": decoded, "encode_seconds": encode_seconds,
            "decode_seconds": decode_seconds}


def frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """[N, 3, H, W] float [0, 1] loaded from 8-bit PNGs -> [N, H, W, 3] uint8,
    verifying the conversion is exact (the PNG values survive untouched)."""
    scaled = frames * 255.0
    rounded = torch.round(scaled)
    if float(torch.max(torch.abs(scaled - rounded))) > 1e-3:
        raise ValueError("frames are not exact 8-bit values; refusing to re-quantize the source")
    return rounded.to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy()


def uint8_to_frames(array: np.ndarray) -> torch.Tensor:
    # .contiguous(): a permuted (channels-last) tensor gave wrong CUDA MS-SSIM;
    # msssim() now guards against that too, this keeps the frames standard NCHW.
    return (torch.from_numpy(array.copy()).permute(0, 3, 1, 2).contiguous()
            .to(torch.float32) / 255.0)


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage 0: single-pass NVC vs H.264/H.265 on DAVIS TEST.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt"))
    parser.add_argument("--m11-dir", type=Path, default=Path("outputs/m11_autoregressive_entropy"))
    parser.add_argument("--m10k-dir", type=Path, default=Path("outputs/m10k_learned_entropy"))
    parser.add_argument("--m22-dir", type=Path, default=Path("outputs/m22_residual_freeze_lift"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks/parity_s0"))
    parser.add_argument("--output-name", type=str, default="parity.json")
    parser.add_argument("--rate-points", type=int, nargs="+", default=[5, 4, 3])
    parser.add_argument("--crf", type=int, nargs="+", default=list(DEFAULT_CRFS))
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--train-frames-per-sequence", type=int, default=8)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--skip-nvc", action="store_true",
                        help="classical arms only (for a quick harness check)")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="smoke runs only; a truncated run cannot reproduce M22's totals")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="smoke runs only; a truncated run cannot reproduce M22's totals")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def recorded_nvc_totals(m22_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """What M22 Phase 19 recorded on DAVIS TEST, per arm and bit depth."""
    davis = json.loads((m22_dir / "m22_davis.json").read_text(encoding="utf-8"))
    recorded: dict[str, dict[int, dict[str, Any]]] = {"nvc_deployed": {}, "nvc_m22": {}}
    for point in davis["rate_points"]:
        for arm, key in (("nvc_deployed", "baseline"), ("nvc_m22", "candidate")):
            recorded[arm][point["bits"]] = {
                "total_container_bytes": point[key]["total_container_bytes"],
                "sequence_pooled_psnr": point[key]["mean_psnr_db"],
                "sequence_mean_msssim": point[key]["mean_msssim"],
            }
    return recorded


def run_nvc(args, device, sequences, frame_cache, report, persist) -> None:
    mc = _load_script("m10h_motion_compensation")
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    cx = _load_script("m11_causal_context")
    m13 = _load_script("m13_recalibration")
    md = _load_script("m11_data")
    m21 = _load_script("m21_refinement")
    m22 = _load_script("m22_residual")
    mech = _load_script("m22_mechanism")

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    recorded = recorded_nvc_totals(args.m22_dir)
    train_full = discover_sequences(args.manifest, split="train")
    motion_train = m21.broad_train_sequences(
        args.manifest, frames_per_sequence=args.train_frames_per_sequence)
    stream_dir = Path(tempfile.mkdtemp(prefix="parity_nvct_"))

    for bits in args.rate_points:
        labels = {arm: f"{arm}/{bits}bit" for arm in ("nvc_deployed", "nvc_m22")}
        if all(label in report["points"] for label in labels.values()):
            print(f"  [resume] NVC {bits}-bit already measured", flush=True)
            continue

        rig = m21.prepare_rate_point(
            model, bits=bits, manifest=args.manifest, checkpoint=args.checkpoint,
            m11_dir=args.m11_dir, m10k_dir=args.m10k_dir, device=device,
            cache_dir=args.cache_dir or md.DEFAULT_CACHE_DIR, train_full=train_full,
            motion_train=motion_train, calibration_frames=args.calibration_frames,
            gop_size=args.gop, block_size=args.block_size, search_range=args.search_range,
            motion_cache=args.m22_dir / "m22_deployed_motion_table.json")
        checkpoint = args.m22_dir / "checkpoints" / f"m22_{M22_CANDIDATE}_{bits}bit.pt"
        loaded = mech.load_refit_checkpoint(checkpoint, device=device, m22=m22, ma=ma, ml=ml,
                                            cx=cx)
        arms = {"nvc_deployed": (rig["spec"], rig["residual_params"], None),
                "nvc_m22": (loaded["spec"], loaded["residual_params"],
                            loaded["record"]["sha256"])}

        for arm, (spec, residual_params, sha) in arms.items():
            label = labels[arm]
            if label in report["points"]:
                continue
            rows, float_rows, exact = [], [], True
            for sequence in sequences:
                frames = frame_cache[sequence.sequence_id]
                path = stream_dir / f"{arm}_{bits}bit_{sequence.sequence_id}.nvct"
                started = time.perf_counter()
                result = m21.encode_sequence_refined(
                    mc, m13, model, frames, spec, path, m21.IDENTITY,
                    intra_params=rig["intra_params"],
                    intra_entropy_model=rig["intra_entropy_model"],
                    residual_params=residual_params,
                    motion_entropy_model=rig["motion_entropy_model"], bits=bits,
                    gop_size=args.gop, block_size=args.block_size,
                    search_range=args.search_range)
                encode_seconds = time.perf_counter() - started
                size = path.stat().st_size
                started = time.perf_counter()
                decoded = m21.decode_sequence_refined(
                    mc, m13, model, path, spec, m21.IDENTITY,
                    intra_entropy_model=rig["intra_entropy_model"],
                    motion_entropy_model=rig["motion_entropy_model"], bits=bits)
                decode_seconds = time.perf_counter() - started
                path.unlink()
                exact &= torch.equal(decoded["reconstructions"].cpu(), result["reconstructions"])

                reconstruction = decoded["reconstructions"].cpu()
                scores = score_frames(to_uint8_frames(reconstruction), frames, device=device)
                rows.append({"sequence": sequence.sequence_id, "bytes": size,
                             "pixels": sequence.total_pixels, "encode_seconds": encode_seconds,
                             "decode_seconds": decode_seconds, **scores})
                # The un-rounded float scores exist only to prove this run reproduces
                # what M22 recorded; they are never used for the comparison.
                float_rows.append({"sequence": sequence.sequence_id, "bytes": size,
                                   "pixels": sequence.total_pixels,
                                   **score_frames(reconstruction, frames, device=device)})

            point = summarize_point(label, arm, f"{bits}bit", rows, extra={
                "bits": bits, "decode_reconstruction_exact": exact,
                "checkpoint_sha256": sha,
            })
            float_quality = aggregate_quality(float_rows)
            expected = recorded[arm].get(bits)
            check = {"recorded": expected,
                     "measured_bytes": point["total_bytes"],
                     "measured_float_sequence_pooled_psnr":
                         float_quality["sequence_pooled"]["psnr"],
                     "measured_float_sequence_mean_msssim":
                         float_quality["sequence_mean"]["msssim"]}
            if expected is None:
                check["reproduces"] = None
            else:
                check["bytes_match"] = point["total_bytes"] == expected["total_container_bytes"]
                check["psnr_abs_diff_db"] = abs(float_quality["sequence_pooled"]["psnr"]
                                                - expected["sequence_pooled_psnr"])
                check["msssim_abs_diff"] = abs(float_quality["sequence_mean"]["msssim"]
                                               - expected["sequence_mean_msssim"])
                check["reproduces"] = (check["bytes_match"] and check["psnr_abs_diff_db"] < 1e-3
                                       and check["msssim_abs_diff"] < 1e-4)
            point["reproduction_check"] = check
            report["points"][label] = point
            persist()
            q = point["quality"][PRIMARY_CONVENTION]
            print(f"  {label:24s} {point['total_bytes']:>10,} B  bpp {point['bpp']:.4f}  "
                  f"PSNR {q['psnr']:.3f}  MS-SSIM {q['msssim']:.4f}  "
                  f"decode-exact={exact}  reproduces-M22={check['reproduces']}", flush=True)


def run_classical(args, device, sequences, frame_cache, report, persist) -> None:
    workdir = Path(tempfile.mkdtemp(prefix="parity_ffmpeg_"))
    for arm in classical_arms(args.gop):
        for crf in args.crf:
            label = f"{arm.name}/crf{crf}"
            if label in report["points"]:
                continue
            rows = []
            for sequence in sequences:
                frames = frame_cache[sequence.sequence_id]
                result = ffmpeg_round_trip(frames_to_uint8(frames), arm, crf, workdir)
                scores = score_frames(uint8_to_frames(result["decoded"]), frames, device=device)
                rows.append({"sequence": sequence.sequence_id, "bytes": result["bytes"],
                             "pixels": sequence.total_pixels,
                             "encode_seconds": result["encode_seconds"],
                             "decode_seconds": result["decode_seconds"], **scores})
            point = summarize_point(label, arm.name, f"crf{crf}", rows,
                                    extra={"crf": crf, **arm.describe()})
            report["points"][label] = point
            persist()
            q = point["quality"][PRIMARY_CONVENTION]
            print(f"  {label:24s} {point['total_bytes']:>10,} B  bpp {point['bpp']:.4f}  "
                  f"PSNR {q['psnr']:.3f}  MS-SSIM {q['msssim']:.4f}", flush=True)


def compare(report: dict[str, Any]) -> dict[str, Any]:
    """BD-rate of each NVC arm against each classical arm, under every
    convention and both metrics, plus the bit ratio at each NVC operating point."""
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for point in report["points"].values():
        by_arm.setdefault(point["arm"], []).append(point)
    results: dict[str, Any] = {}
    for nvc in ("nvc_deployed", "nvc_m22"):
        if len(by_arm.get(nvc, [])) < 2:
            continue
        for classical in ("h264", "h265", "h264_lowdelay", "h265_lowdelay"):
            if len(by_arm.get(classical, [])) < 2:
                continue
            entry: dict[str, Any] = {}
            for convention in CONVENTIONS:
                for metric in ("psnr", "msssim"):
                    base = curve(by_arm[classical], convention, metric)
                    test = curve(by_arm[nvc], convention, metric)
                    entry[f"{convention}/{metric}"] = {
                        "bd_rate_percent": bd_rate(base, test),
                        "quality_overlap": overlap(base, test),
                        "rate_ratio_at_nvc_points": [
                            {"bits": p["bits"], "nvc_bpp": p["bpp"],
                             "quality": p["quality"][convention][metric],
                             "ratio": rate_ratio_at_quality(
                                 base, p["bpp"], p["quality"][convention][metric])}
                            for p in sorted(by_arm[nvc], key=lambda p: p["bits"])],
                    }
            results[f"{nvc}_vs_{classical}"] = entry
    return results


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    require_encoders(["libx264", "libx265"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / args.output_name
    configuration = {"rate_points": args.rate_points, "crf": args.crf, "gop": args.gop,
                     "block_size": args.block_size, "search_range": args.search_range,
                     "calibration_frames": args.calibration_frames,
                     "checkpoint": str(args.checkpoint), "m22_candidate": M22_CANDIDATE,
                     "primary_convention": PRIMARY_CONVENTION,
                     "max_sequences": args.max_sequences, "max_frames": args.max_frames}
    report: dict[str, Any] = {
        "stage": "PARITY_ROADMAP Stage 0 - single-pass scoreboard",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": str(device), "ffmpeg": ffmpeg_version(),
                        "platform": platform.platform()},
        "points": {},
    }
    if report_path.is_file() and not args.no_resume:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("configuration") == configuration:
            report["points"] = previous.get("points", {})
            print(f"[resume] {len(report['points'])} operating points already measured",
                  flush=True)
        else:
            print("[resume] configuration changed; starting fresh", flush=True)

    def persist() -> None:
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # DAVIS TEST - the same split and order M14/M21/M22 evaluated on.
    sequences = discover_sequences(args.manifest, split="test",
                                   max_sequences=args.max_sequences,
                                   max_frames_per_sequence=args.max_frames)
    frame_cache = {s.sequence_id: s.load_frames() for s in sequences}
    report["dataset"] = {"split": "test", "sequences": [s.sequence_id for s in sequences],
                         "frames": sum(s.frame_count for s in sequences),
                         "resolution": [sequences[0].width, sequences[0].height]}
    print("=" * 110)
    print(f"STAGE 0 SCOREBOARD - {len(sequences)} sequences, {report['dataset']['frames']} "
          f"frames, primary convention: {PRIMARY_CONVENTION}")
    print("=" * 110, flush=True)

    run_classical(args, device, sequences, frame_cache, report, persist)
    if not args.skip_nvc:
        run_nvc(args, device, sequences, frame_cache, report, persist)

    report["comparisons"] = compare(report)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["valid"] = all(p.get("reproduction_check", {}).get("reproduces", True) is not False
                          and p.get("decode_reconstruction_exact", True)
                          for p in report["points"].values())
    persist()

    print("\nHEADLINE (primary convention, PSNR):")
    for key, entry in report["comparisons"].items():
        primary = entry[f"{PRIMARY_CONVENTION}/psnr"]
        bd = primary["bd_rate_percent"]
        ratios = ", ".join(f"{r['bits']}b {r['ratio']:.2f}x" if r["ratio"] else f"{r['bits']}b n/a"
                           for r in primary["rate_ratio_at_nvc_points"])
        print(f"  {key:32s} BD-rate {'n/a' if bd is None else f'{bd:+.1f}%'}   "
              f"bits vs classical at NVC's quality: {ratios}")
    print(f"\nrun valid: {report['valid']}\nReport: {report_path}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
