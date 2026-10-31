"""Demo clips: a video file or a list of frame images, turned into the fixed-size
frame sequence the codec was trained and measured on.

A clip is a directory:

    <clips_dir>/<clip_id>/
        clip.json            name, source, frame count, size, fps
        original/000.png     the frames, lossless
        <bundle_id>/...      one result per codec bundle (see `pipeline.process_clip`)

Frames are centre-cropped to a square and resized to 256x256. The codec is fully
convolutional and would accept other sizes, but nothing in this project has been
measured at another resolution, so the demo does not pretend otherwise.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

CLIP_SIZE = 256
DEFAULT_MAX_FRAMES = 60
DEFAULT_FPS = 25.0
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
_CLIP_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class ClipError(ValueError):
    """A source that cannot be turned into a clip, or a clip that is malformed."""


def is_clip_id(value: str) -> bool:
    """True for the ids this module generates - and so for nothing containing a
    path separator, a dot or a drive letter."""
    return bool(_CLIP_ID.match(value))


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:40] or "clip"


def fit_frame(frame_bgr: np.ndarray, size: int = CLIP_SIZE) -> np.ndarray:
    """Centre-crop a BGR frame to a square and resize it to `size` x `size`."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ClipError(f"expected a 3-channel frame, got shape {frame_bgr.shape}")
    height, width = frame_bgr.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    square = frame_bgr[top:top + side, left:left + side]
    if side == size:
        return np.ascontiguousarray(square)
    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(square, (size, size), interpolation=interpolation)


def frames_from_video(path: str | Path, *, max_frames: int = DEFAULT_MAX_FRAMES,
                      size: int = CLIP_SIZE) -> tuple[np.ndarray, float]:
    """Read the first `max_frames` frames of a video file.

    Returns ([N, size, size, 3] uint8 RGB, frames per second).
    """
    path = Path(path)
    if not path.is_file():
        raise ClipError(f"video not found: {path}")
    if max_frames < 1:
        raise ClipError("max_frames must be at least 1")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ClipError(f"could not open '{path.name}' as a video")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = []
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames.append(cv2.cvtColor(fit_frame(frame, size), cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ClipError(f"'{path.name}' contains no decodable frames")
    if not 1.0 <= fps <= 120.0:
        fps = DEFAULT_FPS
    return np.stack(frames), fps


def frames_from_images(paths: Sequence[str | Path], *, max_frames: int = DEFAULT_MAX_FRAMES,
                       size: int = CLIP_SIZE) -> np.ndarray:
    """Read frame images, in the order given. Returns [N, size, size, 3] uint8 RGB."""
    frames = []
    for path in list(paths)[:max_frames]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ClipError(f"could not read image: {path}")
        frames.append(cv2.cvtColor(fit_frame(image, size), cv2.COLOR_BGR2RGB))
    if not frames:
        raise ClipError("no frame images given")
    return np.stack(frames)


def frame_name(index: int) -> str:
    return f"{index:03d}.png"


def write_frames(directory: Path, frames_rgb: np.ndarray) -> None:
    """Write [N, H, W, 3] uint8 RGB frames as 000.png, 001.png, ... (lossless)."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames_rgb):
        target = directory / frame_name(index)
        if not cv2.imwrite(str(target), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
            raise ClipError(f"could not write {target}")


def read_frames(directory: Path, count: int) -> np.ndarray:
    """Read back `count` frames written by `write_frames`."""
    frames = []
    for index in range(count):
        image = cv2.imread(str(directory / frame_name(index)), cv2.IMREAD_COLOR)
        if image is None:
            raise ClipError(f"missing frame {frame_name(index)} in {directory}")
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def create_clip(clips_dir: str | Path, frames_rgb: np.ndarray, *, name: str, source: str,
                fps: float = DEFAULT_FPS, clip_id: str | None = None) -> dict[str, Any]:
    """Write a new clip and return its `clip.json` contents.

    `clip_id` fixes the directory name (used for the precomputed clips, so a
    rebuild overwrites them); otherwise one is generated from `name`.
    """
    if frames_rgb.ndim != 4 or frames_rgb.shape[3] != 3 or frames_rgb.dtype != np.uint8:
        raise ClipError("frames must be uint8 with shape [N, H, W, 3]")
    if clip_id is None:
        clip_id = f"{slugify(name)}-{uuid.uuid4().hex[:8]}"
    if not is_clip_id(clip_id):
        raise ClipError(f"invalid clip id {clip_id!r}")
    clip_dir = Path(clips_dir) / clip_id
    write_frames(clip_dir / "original", frames_rgb)
    info = {
        "id": clip_id,
        "name": name,
        "source": source,
        "frames": int(frames_rgb.shape[0]),
        "height": int(frames_rgb.shape[1]),
        "width": int(frames_rgb.shape[2]),
        "fps": float(fps),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (clip_dir / "clip.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def load_clip(clips_dir: str | Path, clip_id: str) -> dict[str, Any]:
    if not is_clip_id(clip_id):
        raise ClipError(f"invalid clip id {clip_id!r}")
    path = Path(clips_dir) / clip_id / "clip.json"
    if not path.is_file():
        raise ClipError(f"no such clip: {clip_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_clips(clips_dir: str | Path) -> list[dict[str, Any]]:
    """Every clip, each with the ids of the bundles it already has results for.
    Precomputed clips first, by name; then uploads, newest first."""
    clips_dir = Path(clips_dir)
    clips = []
    if clips_dir.is_dir():
        for clip_json in clips_dir.glob("*/clip.json"):
            try:
                info = json.loads(clip_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            info["results"] = sorted(
                p.parent.name for p in clip_json.parent.glob("*/result.json"))
            clips.append(info)
    precomputed = sorted((c for c in clips if c.get("source") == "precomputed"),
                         key=lambda c: c.get("name", ""))
    uploads = sorted((c for c in clips if c.get("source") != "precomputed"),
                     key=lambda c: c.get("created_utc", ""), reverse=True)
    return precomputed + uploads
