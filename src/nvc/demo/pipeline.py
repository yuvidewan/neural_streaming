"""One clip through one codec bundle, written to disk for the page to read.

    <clip_dir>/<bundle_id>/
        result.json          everything the page plots (see `process_clip`)
        stream.nvct          the coded stream itself
        nvc/000.png          the decoder's frames
        pred/001.png         the motion-compensated prediction (P-frames only)
        h264.mp4, h264/      H.264 at the same bitrate, and its decoded frames

Every quality figure is computed on the 8-bit frames a viewer actually sees
(the PNGs above), with the project's own PSNR and MS-SSIM, the way the Stage 0
scoreboard scores codecs. Bitrate counts whole files: the `.nvct` stream and the
`.mp4`.

`process_clip` also decodes the stream it just wrote with the real decoder and
records whether the frames match the encoder's bit for bit. That is the codec's
central invariant, and the page shows it rather than assuming it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from nvc.demo import reference
from nvc.demo.bundles import BundleInfo
from nvc.demo.clips import ClipError, frame_name, read_frames, write_frames
from nvc.demo.trace import trace_encode
from nvc.evaluation.basic_metrics import psnr
from nvc.evaluation.ffmpeg import FFmpegError
from nvc.evaluation.perceptual_metrics import MetricInputError, msssim
from nvc.video.codec import VideoCodec
from nvc.video.container import FRAME_TYPE_I

RESULT_VERSION = 1
Progress = Callable[[str, int, int], None]


def to_uint8(frames: torch.Tensor) -> np.ndarray:
    """[N, 3, H, W] float in [0, 1] -> [N, H, W, 3] uint8, rounded as a display would."""
    return (frames.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()


def to_float(frames_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).contiguous().float() / 255.0


def score(decoded_uint8: np.ndarray, original_uint8: np.ndarray) -> dict[str, Any]:
    """Per-frame PSNR and MS-SSIM of 8-bit frames against 8-bit originals, and their means.

    A frame identical to its original has infinite PSNR; it is reported as None
    (JSON has no infinity) and left out of the mean. MS-SSIM is None when the
    frames are too small for its five scales.
    """
    decoded, original = to_float(decoded_uint8), to_float(original_uint8)
    psnr_values: list[float | None] = []
    msssim_values: list[float] | None = []
    for index in range(decoded.shape[0]):
        value = float(psnr(decoded[index:index + 1], original[index:index + 1]))
        psnr_values.append(value if np.isfinite(value) else None)
        if msssim_values is not None:
            try:
                msssim_values.append(float(msssim(decoded[index:index + 1],
                                                  original[index:index + 1])))
            except MetricInputError:
                msssim_values = None
    finite = [v for v in psnr_values if v is not None]
    return {
        "psnr": float(np.mean(finite)) if finite else None,
        "msssim": float(np.mean(msssim_values)) if msssim_values else None,
        "psnr_per_frame": psnr_values,
    }


def _h264_reference(clip_dir: Path, out_dir: Path, original: np.ndarray, *, target_bytes: int,
                    fps: float) -> dict[str, Any]:
    if not reference.h264_available():
        return {"available": False, "reason": "FFmpeg with libx264 was not found on PATH."}
    try:
        match = reference.match_bitrate(clip_dir / "original" / "%03d.png", out_dir / "h264.mp4",
                                        target_bytes=target_bytes, fps=fps)
        count = reference.decode_to_frames(out_dir / "h264.mp4", out_dir / "h264")
        if count != original.shape[0]:
            raise FFmpegError(f"H.264 decoded {count} frames, the clip has {original.shape[0]}")
        decoded = read_frames(out_dir / "h264", count)
    except (FFmpegError, ClipError, OSError) as exc:
        return {"available": False, "reason": f"H.264 encode failed: {exc}"}
    pixels = original.shape[0] * original.shape[1] * original.shape[2]
    result = {"available": True, "encoder": reference.ENCODER, "preset": reference.PRESET,
              "crf": match.crf, "total_bytes": match.total_bytes,
              "bpp": match.total_bytes * 8 / pixels, "within_target": match.within_target}
    result.update(score(decoded, original))
    return result


def process_clip(clip_dir: str | Path, bundle: BundleInfo, codec: VideoCodec, *,
                 progress: Progress | None = None, with_h264: bool = True) -> dict[str, Any]:
    """Run the clip in `clip_dir` through `codec` and write `<clip_dir>/<bundle.id>/`.

    Returns the contents of `result.json`. `progress(stage, done, total)` is
    called as the work advances; stages are "encode", "decode", "h264", "score".
    """
    clip_dir = Path(clip_dir)
    info = json.loads((clip_dir / "clip.json").read_text(encoding="utf-8"))
    count, fps = int(info["frames"]), float(info["fps"])
    report = progress or (lambda stage, done, total: None)
    out_dir = clip_dir / bundle.id
    out_dir.mkdir(parents=True, exist_ok=True)
    # A half-written result must never be served: drop the marker file first.
    (out_dir / "result.json").unlink(missing_ok=True)

    original = read_frames(clip_dir / "original", count)
    frames = to_float(original)

    started = time.perf_counter()
    trace = trace_encode(codec, frames, progress=lambda done, total: report("encode", done, total))
    encode_seconds = time.perf_counter() - started
    encoded = trace.encoded
    (out_dir / "stream.nvct").write_bytes(encoded.data)

    report("decode", 0, count)
    started = time.perf_counter()
    decoded = codec.decode(encoded.data)
    decode_seconds = time.perf_counter() - started
    decoder_exact = bool(torch.equal(decoded, trace.reconstructions))
    report("decode", count, count)

    decoded_uint8 = to_uint8(decoded)
    write_frames(out_dir / "nvc", decoded_uint8)
    prediction_dir = out_dir / "pred"
    prediction_dir.mkdir(exist_ok=True)
    for stale in prediction_dir.glob("*.png"):
        stale.unlink()
    for frame in trace.frames:
        if frame.prediction is not None:
            image = to_uint8(frame.prediction[None])[0]
            cv2.imwrite(str(prediction_dir / frame_name(frame.index)),
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    report("h264", 0, 1)
    h264 = (_h264_reference(clip_dir, out_dir, original, target_bytes=encoded.total_bytes, fps=fps)
            if with_h264 else {"available": False, "reason": "H.264 comparison was not requested."})
    report("h264", 1, 1)

    report("score", 0, 1)
    pixels = count * encoded.width * encoded.height
    payload_bytes = sum(encoded.motion_bytes) + sum(encoded.residual_bytes)
    nvc = {
        "total_bytes": encoded.total_bytes,
        "bpp": encoded.bits_per_pixel,
        "container_bytes": encoded.total_bytes - payload_bytes,
        "frame_types": ["I" if t == FRAME_TYPE_I else "P" for t in encoded.frame_types],
        "motion_bytes": list(encoded.motion_bytes),
        "residual_bytes": list(encoded.residual_bytes),
        "motion": [None if f.motion is None else f.motion.tolist() for f in trace.frames],
        "activity": [None if f.activity is None else np.round(f.activity, 3).tolist()
                     for f in trace.frames],
        "encode_ms_per_frame": encode_seconds * 1000 / count,
        "decode_ms_per_frame": decode_seconds * 1000 / count,
        "decoder_exact": decoder_exact,
    }
    nvc.update(score(decoded_uint8, original))
    result = {
        "version": RESULT_VERSION,
        "clip": info["id"],
        "bundle": bundle.to_json(),
        "frames": count, "width": encoded.width, "height": encoded.height, "fps": fps,
        "raw_bytes": pixels * 3,
        "device": str(codec.device),
        "nvc": nvc,
        "h264": h264,
    }
    (out_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    report("score", 1, 1)
    return result
