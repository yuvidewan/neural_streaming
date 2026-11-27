"""The fair-evaluation protocol of docs/PLAN_fair_eval.md, as a library.

Every codec is judged against the ORIGINAL 8-bit 4:2:0 YUV:

- x264 and x265 get that YUV directly and write raw elementary streams (`.264`, `.265`), which
  are decoded back to 4:2:0 (`encode_anchor`).
- A codec that works in RGB (ours, the pilot) converts YUV -> RGB with `yuv_to_rgb` and its
  decoded RGB back with `rgb_to_yuv420`, so both conversions count against it.
- Quality is PSNR per plane (8-bit, peak 255), averaged over frames, combined 6:1:1
  (`psnr_yuv`); rate is the whole stream file in bits per luma pixel.
- BD-rate is per video, log-rate interpolated against PSNR_YUV with monotone piecewise-cubic
  (PCHIP) curves (`bd_rate_pchip`), with the old piecewise-linear method as a check.

Nothing here depends on torch, so the anchors run on any machine with FFmpeg.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Both conversions of an RGB codec (docs/PLAN_fair_eval.md, "How each codec gets the video").
YUV_TO_RGB_FILTER = ("scale=in_color_matrix=bt709:in_range=tv:out_range=pc:"
                     "flags=accurate_rnd+full_chroma_int")
RGB_TO_YUV_FILTER = ("scale=out_color_matrix=bt709:in_range=pc:out_range=tv:"
                     "flags=accurate_rnd+full_chroma_int")
# A frame identical to its reference has infinite PSNR. It is capped here so averages stay
# finite; only lossless proofs ever reach it, and they compare bytes, not PSNR.
PSNR_CAP = 100.0
STREAMING_BAND_BPP = (0.016, 0.129)   # 1 and 8 Mbps at 1920x1080, 30 frames/s
CRF_LADDER = tuple(range(8, 47, 2))   # 8, 10, ..., 46
ROBUST_WIN = -3.0


def ffmpeg_binary() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("ffmpeg is not on PATH")
    return path


def ffmpeg_version() -> str:
    out = subprocess.run([ffmpeg_binary(), "-hide_banner", "-version"], capture_output=True,
                         text=True, check=True).stdout
    return out.splitlines()[0]


# --------------------------------------------------------------------------------------------
# YUV video
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class YuvVideo:
    """An 8-bit planar 4:2:0 file: `frames` frames of `width` x `height`, at `fps`."""

    name: str
    path: Path
    width: int
    height: int
    fps: float
    frames: int

    def __post_init__(self) -> None:
        if self.width % 2 or self.height % 2:
            raise ValueError(f"{self.name}: 4:2:0 needs even sides, got {self.width}x{self.height}")

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * 3 // 2

    @property
    def luma_pixels(self) -> int:
        """Bits-per-pixel denominator: luma pixels over every frame."""
        return self.width * self.height * self.frames

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "path": str(self.path), "width": self.width,
                "height": self.height, "fps": self.fps, "frames": self.frames}

    @classmethod
    def from_json(cls, item: dict[str, Any], base: Path | None = None) -> "YuvVideo":
        path = Path(item["path"])
        if base is not None and not path.is_absolute():
            path = base / path
        return cls(item["name"], path, int(item["width"]), int(item["height"]),
                   float(item["fps"]), int(item["frames"]))


def load_manifest(path: Path) -> list[YuvVideo]:
    """A fair-eval manifest: {"videos": [{name, path, width, height, fps, frames}, ...]}.
    Relative paths are taken from the manifest's folder."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [YuvVideo.from_json(item, Path(path).parent) for item in data["videos"]]


def split_planes(raw: bytes | np.ndarray, width: int, height: int):
    """Raw 4:2:0 bytes of n frames -> (Y [n,h,w], U [n,h/2,w/2], V [n,h/2,w/2]) uint8."""
    data = np.frombuffer(raw, dtype=np.uint8) if isinstance(raw, (bytes, bytearray)) else raw
    frame = width * height * 3 // 2
    if data.size % frame:
        raise ValueError(f"{data.size} bytes is not a whole number of {width}x{height} frames")
    frames = data.reshape(-1, frame)
    luma, chroma = width * height, (width // 2) * (height // 2)
    y = frames[:, :luma].reshape(-1, height, width)
    u = frames[:, luma:luma + chroma].reshape(-1, height // 2, width // 2)
    v = frames[:, luma + chroma:].reshape(-1, height // 2, width // 2)
    return y, u, v


def read_yuv(video: YuvVideo, frames: int | None = None) -> np.ndarray:
    """The first `frames` (default: all of `video.frames`) frames as flat uint8 bytes."""
    count = video.frames if frames is None else frames
    need = count * video.frame_bytes
    data = np.fromfile(video.path, dtype=np.uint8, count=need)
    if data.size != need:
        raise ValueError(f"{video.path}: need {need} bytes for {count} frames, got {data.size}")
    return data


def _pipe(arguments: list[str], payload: bytes | None) -> bytes:
    head = [ffmpeg_binary(), "-hide_banner", "-v", "error"]
    if payload is None:
        head.append("-nostdin")
    result = subprocess.run([*head, *arguments], input=payload, capture_output=True,
                            check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): "
                           f"{result.stderr.decode(errors='replace')[-2000:]}")
    return result.stdout


def yuv_to_rgb(raw: np.ndarray | bytes, width: int, height: int) -> np.ndarray:
    """4:2:0 BT.709 limited range -> RGB full range, uint8 [n, h, w, 3]."""
    payload = raw.tobytes() if isinstance(raw, np.ndarray) else raw
    out = _pipe(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}", "-i", "-",
                 "-vf", YUV_TO_RGB_FILTER, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], payload)
    return np.frombuffer(out, dtype=np.uint8).reshape(-1, height, width, 3)


def rgb_to_yuv420(rgb: np.ndarray) -> np.ndarray:
    """RGB full range uint8 [n, h, w, 3] -> 4:2:0 BT.709 limited range, flat uint8."""
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[3] != 3:
        raise ValueError(f"need uint8 [n,h,w,3], got {rgb.dtype} {rgb.shape}")
    height, width = rgb.shape[1:3]
    out = _pipe(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-i", "-",
                 "-vf", RGB_TO_YUV_FILTER, "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"],
                np.ascontiguousarray(rgb).tobytes())
    return np.frombuffer(out, dtype=np.uint8)


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------


def plane_psnr(reference: np.ndarray, distorted: np.ndarray) -> np.ndarray:
    """Per-frame PSNR of one plane stack [n, h, w] (8-bit, peak 255), capped at PSNR_CAP."""
    if reference.shape != distorted.shape:
        raise ValueError(f"shape mismatch {reference.shape} vs {distorted.shape}")
    diff = reference.astype(np.float64) - distorted.astype(np.float64)
    mse = np.mean(diff * diff, axis=(1, 2))
    with np.errstate(divide="ignore"):
        psnr = 10.0 * np.log10(255.0 ** 2 / mse)
    return np.minimum(psnr, PSNR_CAP)


def score_yuv(reference: np.ndarray | bytes, distorted: np.ndarray | bytes, width: int,
              height: int) -> dict[str, Any]:
    """PSNR per plane, each averaged over frames, and PSNR_YUV = (6Y + U + V) / 8."""
    ref, dist = split_planes(reference, width, height), split_planes(distorted, width, height)
    if ref[0].shape != dist[0].shape:
        raise ValueError(f"frame count differs: {ref[0].shape[0]} vs {dist[0].shape[0]}")
    per_frame = [plane_psnr(r, d) for r, d in zip(ref, dist)]
    y, u, v = (float(np.mean(p)) for p in per_frame)
    return {"psnr_y": y, "psnr_u": u, "psnr_v": v, "psnr_yuv": psnr_yuv(y, u, v),
            "frames": int(ref[0].shape[0]),
            "per_frame_y": [float(x) for x in per_frame[0]]}


def psnr_yuv(y: float, u: float, v: float) -> float:
    return (6.0 * y + u + v) / 8.0


# --------------------------------------------------------------------------------------------
# Anchors
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorArm:
    """One classical configuration. `extra` goes after the rate control."""

    name: str
    encoder: str            # libx264 | libx265
    preset: str = "medium"
    tune: str | None = "psnr"
    extra: tuple[str, ...] = ()
    headline: bool = False

    @property
    def suffix(self) -> str:
        return ".264" if self.encoder == "libx264" else ".265"

    @property
    def muxer(self) -> str:
        return "h264" if self.encoder == "libx264" else "hevc"

    def arguments(self, crf: int | None = None, *, lossless: bool = False) -> list[str]:
        args = ["-c:v", self.encoder, "-preset", self.preset]
        if self.tune is not None and not lossless:
            args += ["-tune", self.tune]
        x265_params = [p for p in self.extra if p.startswith("x265:")]
        plain = [p for p in self.extra if not p.startswith("x265:")]
        if lossless:
            if self.encoder == "libx264":
                args += ["-qp", "0"]
            else:
                x265_params.append("x265:lossless=1")
        else:
            args += ["-crf", str(crf)]
        args += plain
        if self.encoder == "libx265":
            params = ["log-level=error"] + [p[5:] for p in x265_params]
            args += ["-x265-params", ":".join(params)]
        return args


ARMS: dict[str, AnchorArm] = {a.name: a for a in (
    # headline (docs/PLAN_fair_eval.md, "Anchors")
    AnchorArm("x264_default", "libx264", headline=True),
    AnchorArm("x264_lowdelay", "libx264", extra=("-bf", "0", "-g", "30"), headline=True),
    AnchorArm("x265_default", "libx265", headline=True),
    AnchorArm("x265_lowdelay", "libx265", extra=("x265:bframes=0", "x265:keyint=30"),
              headline=True),
    # reported, not headline
    AnchorArm("x264_default_visual", "libx264", tune=None),
    AnchorArm("x264_lowdelay_visual", "libx264", tune=None, extra=("-bf", "0", "-g", "30")),
    AnchorArm("x264_veryslow_default", "libx264", preset="veryslow"),
    AnchorArm("x264_veryslow_lowdelay", "libx264", preset="veryslow",
              extra=("-bf", "0", "-g", "30")),
)}
HEADLINE_ARMS = tuple(name for name, arm in ARMS.items() if arm.headline)


def encode_anchor(video: YuvVideo, arm: AnchorArm, crf: int | None, stream_path: Path, *,
                  lossless: bool = False, threads: int | None = None) -> dict[str, Any]:
    """Encode `video` (original YUV, real frame rate) to a raw elementary stream, decode it
    back to 4:2:0 and score it. The stream file is kept at `stream_path`."""
    stream_path = Path(stream_path)
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    source = read_yuv(video)
    thread_args = [] if threads is None else ["-threads", str(threads)]
    _pipe(["-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{video.width}x{video.height}",
           "-r", f"{video.fps:g}", "-i", "-", "-frames:v", str(video.frames),
           *arm.arguments(crf, lossless=lossless), *thread_args, "-pix_fmt", "yuv420p",
           "-f", arm.muxer, str(stream_path)], source.tobytes())
    decoded = decode_stream(stream_path)
    if decoded.size != source.size:
        raise RuntimeError(f"{arm.name} crf {crf}: decoded {decoded.size} bytes, "
                           f"source {source.size}")
    size = stream_path.stat().st_size
    point = rate_point(size, video)
    point.update(score_yuv(source, decoded, video.width, video.height))
    point.update({"arm": arm.name, "crf": crf, "lossless": lossless,
                  "decoded_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
                  "identical_to_source": bool(np.array_equal(decoded, source))})
    return point


def decode_stream(stream_path: Path) -> np.ndarray:
    out = _pipe(["-i", str(stream_path), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"], None)
    return np.frombuffer(out, dtype=np.uint8)


def rate_point(size_bytes: int, video: YuvVideo) -> dict[str, float]:
    bits = size_bytes * 8
    seconds_real = video.frames / video.fps
    return {"bytes": int(size_bytes), "bpp": bits / video.luma_pixels,
            "mbps_real_fps": bits / seconds_real / 1e6,
            "mbps_30fps": bits / (video.frames / 30.0) / 1e6}


# --------------------------------------------------------------------------------------------
# BD-rate
# --------------------------------------------------------------------------------------------


Curve = Sequence[tuple[float, float]]   # (bpp, quality)


def _prepared(curve: Curve):
    """Sort by quality; drop repeated qualities (keep the cheaper point). Returns
    (quality, log10 rate, monotone) where monotone says rate rises with quality."""
    points = sorted((float(q), float(r)) for r, q in curve)
    qualities, rates = [], []
    for q, r in points:
        if qualities and q <= qualities[-1]:
            rates[-1] = min(rates[-1], r)
            continue
        qualities.append(q)
        rates.append(r)
    log_rates = np.log10(rates)
    monotone = bool(np.all(np.diff(log_rates) > 0))
    return np.array(qualities), log_rates, monotone


def overlap(a: Curve, b: Curve) -> tuple[float, float] | None:
    low = max(min(q for _, q in a), min(q for _, q in b))
    high = min(max(q for _, q in a), max(q for _, q in b))
    return (low, high) if high > low else None


def _mean_log_rate(qualities, log_rates, low: float, high: float) -> float:
    from scipy.interpolate import PchipInterpolator
    if len(qualities) < 2:
        raise ValueError("a curve needs at least two distinct quality values")
    anti = PchipInterpolator(qualities, log_rates).antiderivative()
    return float((anti(high) - anti(low)) / (high - low))


def bd_rate_pchip(anchor: Curve, test: Curve, band: tuple[float, float] | None = None
                  ) -> dict[str, Any]:
    """BD-rate of `test` against `anchor` in percent (positive = test needs MORE bits).

    Log-rate is interpolated against quality with PCHIP on each curve and the difference is
    averaged over the quality range where both exist, intersected with `band` if given.
    Returns {"bd_rate": float | None, "low", "high", "monotone_anchor", "monotone_test",
    "reason"}."""
    qa, ra, ma = _prepared(anchor)
    qt, rt, mt = _prepared(test)
    out: dict[str, Any] = {"bd_rate": None, "low": None, "high": None,
                           "monotone_anchor": ma, "monotone_test": mt, "reason": None}
    if len(qa) < 2 or len(qt) < 2:
        out["reason"] = "fewer than two points"
        return out
    low, high = max(qa[0], qt[0]), min(qa[-1], qt[-1])
    if band is not None:
        low, high = max(low, band[0]), min(high, band[1])
    if not high > low:
        out["reason"] = "no overlap"
        return out
    diff = _mean_log_rate(qt, rt, low, high) - _mean_log_rate(qa, ra, low, high)
    out.update({"bd_rate": (10.0 ** diff - 1.0) * 100.0, "low": float(low), "high": float(high)})
    return out


def bd_rate_linear(anchor: Curve, test: Curve) -> float | None:
    """Piecewise-linear log-rate BD-rate over the overlap (the check method). Positive = test
    needs more bits."""
    qa, ra, _ = _prepared(anchor)
    qt, rt, _ = _prepared(test)
    if len(qa) < 2 or len(qt) < 2:
        return None
    low, high = max(qa[0], qt[0]), min(qa[-1], qt[-1])
    if not high > low:
        return None
    grid = np.linspace(low, high, 2001)
    diff = np.trapezoid(np.interp(grid, qt, rt) - np.interp(grid, qa, ra), grid) / (high - low)
    return float((10.0 ** diff - 1.0) * 100.0)


def quality_at_rate(curve: Curve, bpp: float) -> float:
    """Our curve's quality at `bpp`, PCHIP in log-rate, clipped to the tested rate range."""
    from scipy.interpolate import PchipInterpolator
    points = sorted((float(r), float(q)) for r, q in curve)
    rates = np.log10([r for r, _ in points])
    qualities = np.array([q for _, q in points])
    x = float(np.clip(np.log10(bpp), rates[0], rates[-1]))
    if len(points) < 2 or np.any(np.diff(rates) <= 0):
        return float(np.interp(x, rates, qualities))
    return float(PchipInterpolator(rates, qualities)(x))


def streaming_band(test: Curve, bpp_range: tuple[float, float] = STREAMING_BAND_BPP
                   ) -> tuple[float, float] | None:
    """The quality interval between `test`'s quality at the two band rates, clipped to its
    tested range. None (not scored) if narrower than 1 dB."""
    low, high = (quality_at_rate(test, b) for b in bpp_range)
    return (low, high) if high - low >= 1.0 else None


def coverage(anchor: Curve, test: Curve) -> dict[str, bool]:
    """Does the anchor's quality range reach past the test codec's highest and lowest points?"""
    qa = [q for _, q in anchor]
    qt = [q for _, q in test]
    return {"top": max(qa) >= max(qt), "bottom": min(qa) <= min(qt)}


def extra_crfs(points: list[dict[str, Any]], test: Curve, metric: str = "psnr_yuv"
               ) -> list[int]:
    """The coverage rule: CRFs to add, in steps of 2, at whichever end falls short. `points`
    are one arm's anchor points on one video (each with "crf" and `metric`). Returns the next
    CRF to try at each short end (call again after coding them)."""
    if not points:
        return list(CRF_LADDER)
    curve = [(p["bpp"], p[metric]) for p in points]
    covered = coverage(curve, test)
    crfs = [p["crf"] for p in points]
    out = []
    if not covered["top"] and min(crfs) - 2 >= 0:
        out.append(min(crfs) - 2)
    if not covered["bottom"] and max(crfs) + 2 <= 51:
        out.append(max(crfs) + 2)
    return out


def compare(anchor: Curve, test: Curve, *, band: tuple[float, float] | None = None
            ) -> dict[str, Any]:
    """Full-range PCHIP BD-rate with the linear check, plus the band BD-rate if `band`."""
    full = bd_rate_pchip(anchor, test)
    linear = bd_rate_linear(anchor, test)
    out = {"bd_rate": full["bd_rate"], "linear": linear,
           "method_gap_flag": (full["bd_rate"] is not None and linear is not None
                               and abs(full["bd_rate"] - linear) > 3.0),
           "overlap": [full["low"], full["high"]], "coverage": coverage(anchor, test),
           "monotone": full["monotone_anchor"] and full["monotone_test"]}
    if band is not None:
        out["band_bd_rate"] = bd_rate_pchip(anchor, test, band)["bd_rate"]
    return out


def summarize(per_video: dict[str, float | None]) -> dict[str, Any]:
    """Mean and median of per-video BD-rates, and wins (< 0) and robust wins (<= -3%)."""
    values = [v for v in per_video.values() if v is not None and math.isfinite(v)]
    return {"videos": len(per_video), "scored": len(values),
            "mean": float(np.mean(values)) if values else None,
            "median": float(np.median(values)) if values else None,
            "wins": sum(v < 0 for v in values),
            "robust_wins": sum(v <= ROBUST_WIN for v in values)}


def curve_of(points: Iterable[dict[str, Any]], metric: str = "psnr_yuv") -> list[tuple[float, float]]:
    return sorted((p["bpp"], p[metric]) for p in points)


@dataclass
class AnchorStore:
    """Resumable anchor results: {video: {arm: [point, ...]}} in one JSON file."""

    path: Path
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def open(cls, path: Path) -> "AnchorStore":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        data.setdefault("points", {})
        return cls(path, data)

    def points(self, video: str, arm: str) -> list[dict[str, Any]]:
        return self.data["points"].get(video, {}).get(arm, [])

    def has(self, video: str, arm: str, crf: int) -> bool:
        return any(p["crf"] == crf for p in self.points(video, arm))

    def add(self, video: str, arm: str, point: dict[str, Any]) -> None:
        entries = self.data["points"].setdefault(video, {}).setdefault(arm, [])
        entries[:] = [p for p in entries if p["crf"] != point["crf"]] + [point]
        entries.sort(key=lambda p: p["crf"])
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        tmp.replace(self.path)
