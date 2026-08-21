"""Codec backends for the rate-distortion benchmark.

Every codec - the neural NVC codec, H.264, H.265 - is measured through the
same interface and the same project-side metric implementations, so the
numbers are directly comparable:

    encode a sequence -> measure real file bytes on disk
                      -> decode it back -> score against the ORIGINAL frames

Two rules keep the comparison honest:

1. **Metrics are always computed here**, from decoded pixels, using
   `nvc.evaluation.basic_metrics` and `nvc.evaluation.perceptual_metrics`.
   FFmpeg's own `-psnr` output is never parsed as authoritative - a codec
   scoring itself with its own metric implementation, in its own internal
   color space, is not comparable to NVC being scored by ours.
2. **Sizes are always real bytes on disk** (`Path.stat().st_size`), never
   an estimate: the whole `.nvc` file for NVC, the whole container for
   FFmpeg. Payload-only figures are never compared against total-file ones.

METHODOLOGY LIMITATION - READ THIS BEFORE COMPARING THE NUMBERS
----------------------------------------------------------------
NVC currently operates as an intra-only, frame-independent neural codec,
while H.264 and H.265 exploit temporal redundancy through inter-frame
prediction. In the default benchmark mode the classical codecs therefore
have a large structural advantage that has nothing to do with how good
either transform is. `intra_only=True` builds an all-intra configuration
that removes exactly that advantage, for the closer like-for-like
comparison; it is offered as an explicit option, not as the default,
because normal video encoding is how these codecs are actually used.

A second, smaller asymmetry: H.264/H.265 default to `yuv420p`, which
subsamples chroma, while NVC codes full-resolution RGB. `pix_fmt` is
configurable for that reason and is recorded in every result.
"""

from __future__ import annotations

import logging
import math
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch

from nvc.compression.codec import decode_frame, encode_frame
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.compression.quantization import QuantizationParams
from nvc.data.image_io import read_image_as_tensor
from nvc.evaluation.basic_metrics import mse as mse_metric
from nvc.evaluation.basic_metrics import psnr_from_mse
from nvc.evaluation.ffmpeg import (
    FFmpegCommandError,
    find_ffmpeg,
    probe_frame_count,
)
from nvc.evaluation.ffmpeg import run_command as run_ffmpeg_command
from nvc.evaluation.perceptual_metrics import MetricInputError, msssim
from nvc.evaluation.sequences import (
    BenchmarkSequence,
    materialize_sequence_for_ffmpeg,
)

# Image sequences carry no inherent frame rate. This value only sets
# container timing; with CRF (quality-targeted, not bitrate-targeted) rate
# control it does not meaningfully change the encoded bytes, and every
# metric here is per-pixel rather than per-second.
DEFAULT_FRAMERATE = 30
DEFAULT_PRESET = "medium"
DEFAULT_PIX_FMT = "yuv420p"
DEFAULT_CRF_VALUES = (18, 23, 28, 33)
DEFAULT_NVC_BIT_DEPTHS = (8, 6, 4)

_LOGGER = logging.getLogger(__name__)


@dataclass
class FrameMetrics:
    """Per-frame quality scores, kept so aggregation can be pixel-weighted."""

    psnr_values: list[float] = field(default_factory=list)
    msssim_values: list[float] = field(default_factory=list)
    msssim_dropped: int = 0
    squared_error_sum: float = 0.0
    element_count: int = 0

    def add(
        self,
        reconstruction: torch.Tensor,
        reference: torch.Tensor,
        *,
        frame_label: str | None = None,
    ) -> None:
        """Score one [1, 3, H, W] reconstruction against its reference.

        `frame_label` (e.g. "seq/000012") is only used to identify the frame
        in the warning logged when MS-SSIM cannot be scored for it.
        """
        frame_mse = mse_metric(reconstruction, reference)
        self.psnr_values.append(float(psnr_from_mse(frame_mse)))
        self.squared_error_sum += float(torch.sum((reconstruction - reference) ** 2))
        self.element_count += reference.numel()
        try:
            self.msssim_values.append(float(msssim(reconstruction, reference)))
        except MetricInputError as exc:
            # Frames below MS-SSIM's 161px floor still get PSNR; the result
            # reports MS-SSIM as unavailable rather than a fabricated value.
            # Logged (not silent) so a caller can tell a clean result apart
            # from one where some frames were dropped from the average.
            self.msssim_dropped += 1
            _LOGGER.warning(
                "MS-SSIM skipped for frame %s: %s",
                frame_label or "<unknown>", exc,
            )

    @property
    def mean_psnr(self) -> float:
        """Frame-weighted mean PSNR in dB.

        A bit-exact frame yields PSNR = +inf (the correct mathematical
        limit - see `psnr_from_mse`). Averaging dB values directly means a
        single +inf frame would make a plain sum()/len() report +inf for
        the WHOLE sequence, even when every other frame has real,
        substantial error - silently hiding that error. So +inf is only
        reported here when every frame was bit-exact; otherwise the mean is
        taken over the finite frames only, which is the only numerically
        meaningful way to combine a mix of finite and infinite dB values.
        """
        finite = [value for value in self.psnr_values if math.isfinite(value)]
        if not finite:
            return float("inf")
        return sum(finite) / len(finite)

    @property
    def mean_msssim(self) -> float | None:
        if not self.msssim_values:
            return None
        return sum(self.msssim_values) / len(self.msssim_values)

    @property
    def msssim_frame_count(self) -> int:
        """Number of frames that actually contributed to `mean_msssim`."""
        return len(self.msssim_values)

    @property
    def pooled_mse(self) -> float:
        return self.squared_error_sum / self.element_count


@dataclass(frozen=True)
class CodecResult:
    """One codec configuration measured on one sequence."""

    codec: str
    configuration: str
    sequence_id: str
    dataset: str
    frame_count: int
    width: int
    height: int
    total_bytes: int
    mean_psnr: float
    mean_msssim: float | None
    pooled_mse: float
    encode_seconds: float
    decode_seconds: float
    details: dict = field(default_factory=dict)
    # Frames that actually contributed to mean_msssim (None: unknown, treat
    # as frame_count). Differs from frame_count whenever a frame was
    # dropped, e.g. below MS-SSIM's minimum spatial size.
    msssim_frame_count: int | None = None

    @property
    def total_pixels(self) -> int:
        return self.frame_count * self.width * self.height

    @property
    def bpp(self) -> float:
        """Bits per pixel: total compressed bytes x 8 / total pixels."""
        return self.total_bytes * 8 / self.total_pixels

    @property
    def compression_ratio(self) -> float:
        """Versus raw uint8 RGB storage (frames x W x H x 3), not YUV video."""
        return (self.total_pixels * 3) / self.total_bytes

    @property
    def pooled_psnr(self) -> float:
        return float(psnr_from_mse(self.pooled_mse))

    def to_row(self) -> dict:
        return {
            "dataset": self.dataset,
            "sequence_id": self.sequence_id,
            "codec": self.codec,
            "codec_configuration": self.configuration,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "total_bytes": self.total_bytes,
            "bpp": self.bpp,
            "compression_ratio": self.compression_ratio,
            "mean_psnr": self.mean_psnr,
            "mean_msssim": self.mean_msssim,
            "msssim_frame_count": self.msssim_frame_count,
            "pooled_psnr": self.pooled_psnr,
            "pooled_mse": self.pooled_mse,
            "encode_time_seconds": self.encode_seconds,
            "decode_time_seconds": self.decode_seconds,
            **self.details,
        }


class Codec(ABC):
    """A codec that can round-trip one sequence and report real measurements."""

    name: str
    configuration: str

    @abstractmethod
    def run(self, sequence: BenchmarkSequence, workdir: Path) -> CodecResult:
        """Encode, measure, decode, and score one sequence."""

    def describe(self) -> dict:
        return {"codec": self.name, "codec_configuration": self.configuration}


# --------------------------------------------------------------------------
# Classical codecs (H.264 / H.265) via FFmpeg
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FFmpegCodecConfig:
    """Everything that distinguishes one classical-codec operating point.

    H.265 differs from H.264 only in the values here, which is what keeps
    `FFmpegVideoCodec` a single implementation rather than two.
    """

    name: str
    encoder: str            # e.g. "libx264"
    container: str = "mp4"
    crf: int = 23
    preset: str = DEFAULT_PRESET
    pix_fmt: str = DEFAULT_PIX_FMT
    framerate: int = DEFAULT_FRAMERATE
    intra_only: bool = False

    @property
    def configuration(self) -> str:
        """A label unique to this operating point, used to group results.

        Only crf (+ an -intra suffix) appears for the common case, to keep
        the label short and existing run directories/labels unchanged. If
        preset or pix_fmt is overridden from the default, it's appended too
        - otherwise two configs that differ only in preset or pix_fmt would
        share a label and aggregate_results() would silently average them
        together as if they were the same operating point.
        """
        parts = [f"crf{self.crf}"]
        if self.preset != DEFAULT_PRESET:
            parts.append(f"preset-{self.preset}")
        if self.pix_fmt != DEFAULT_PIX_FMT:
            parts.append(f"pixfmt-{self.pix_fmt}")
        suffix = "-intra" if self.intra_only else ""
        return "-".join(parts) + suffix

    def intra_arguments(self) -> list[str]:
        """Encoder-specific flags that force every frame to be a keyframe."""
        if not self.intra_only:
            return []
        if self.encoder == "libx264":
            return ["-g", "1"]
        if self.encoder == "libx265":
            return ["-x265-params", "keyint=1:min-keyint=1"]
        return ["-g", "1"]


H264_CODEC = FFmpegCodecConfig(name="h264", encoder="libx264")
H265_CODEC = FFmpegCodecConfig(name="h265", encoder="libx265")


class FFmpegVideoCodec(Codec):
    """H.264/H.265 encoding of a real image sequence through FFmpeg."""

    def __init__(self, config: FFmpegCodecConfig) -> None:
        self.config = config
        self.name = config.name
        self.configuration = config.configuration

    def run(self, sequence: BenchmarkSequence, workdir: Path) -> CodecResult:
        workdir = Path(workdir)
        input_dir = workdir / "input_frames"
        decoded_dir = workdir / "decoded_frames"
        video_path = workdir / f"encoded.{self.config.container}"

        for directory in (decoded_dir,):
            if directory.exists():
                shutil.rmtree(directory)
        decoded_dir.mkdir(parents=True, exist_ok=True)

        _, pattern = materialize_sequence_for_ffmpeg(sequence, input_dir)
        ffmpeg = find_ffmpeg()

        try:
            encode_command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(self.config.framerate),
                "-start_number", "1",
                "-i", str(input_dir / pattern),
                "-c:v", self.config.encoder,
                "-crf", str(self.config.crf),
                "-preset", self.config.preset,
                "-pix_fmt", self.config.pix_fmt,
                *self.config.intra_arguments(),
                str(video_path),
            ]
            start = time.perf_counter()
            run_ffmpeg_command(encode_command, timeout=3600)
            encode_seconds = time.perf_counter() - start

            total_bytes = video_path.stat().st_size

            # Verify the encode itself preserved the frame count via ffprobe
            # (which actually decodes every frame to count it) before
            # trusting the decode step below - a dropped/duplicated frame
            # here would otherwise misalign every per-frame metric silently.
            encoded_frame_count = probe_frame_count(video_path)
            if encoded_frame_count != sequence.frame_count:
                raise FFmpegCommandError(
                    f"{self.name} encoded {encoded_frame_count} frames but the source "
                    f"sequence '{sequence.sequence_id}' has {sequence.frame_count}. "
                    "Per-frame metrics would be misaligned; refusing to score."
                )

            decode_command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(video_path),
                "-start_number", "1",
                str(decoded_dir / pattern),
            ]
            start = time.perf_counter()
            run_ffmpeg_command(decode_command, timeout=3600)
            decode_seconds = time.perf_counter() - start

            decoded_paths = sorted(decoded_dir.glob("frame_*.png"))
            if len(decoded_paths) != sequence.frame_count:
                raise FFmpegCommandError(
                    f"{self.name} decoded {len(decoded_paths)} frames but the source "
                    f"sequence '{sequence.sequence_id}' has {sequence.frame_count}. "
                    "Per-frame metrics would be misaligned; refusing to score."
                )

            metrics = FrameMetrics()
            for reference_path, decoded_path in zip(sequence.frame_paths, decoded_paths):
                reference = read_image_as_tensor(reference_path).unsqueeze(0)
                decoded = read_image_as_tensor(decoded_path).unsqueeze(0)
                metrics.add(
                    decoded, reference,
                    frame_label=f"{sequence.sequence_id}/{decoded_path.name}",
                )
        except Exception:
            # Nothing here is meant to survive a failed run - clean up
            # whatever partial artifacts were written so they don't linger
            # on disk until the caller's end-of-run temp cleanup (which is
            # skipped entirely under --keep-temp).
            if video_path.exists():
                video_path.unlink(missing_ok=True)
            if decoded_dir.exists():
                shutil.rmtree(decoded_dir, ignore_errors=True)
            raise

        return CodecResult(
            codec=self.name,
            configuration=self.configuration,
            sequence_id=sequence.sequence_id,
            dataset=sequence.dataset,
            frame_count=sequence.frame_count,
            width=sequence.width,
            height=sequence.height,
            total_bytes=total_bytes,
            mean_psnr=metrics.mean_psnr,
            mean_msssim=metrics.mean_msssim,
            pooled_mse=metrics.pooled_mse,
            encode_seconds=encode_seconds,
            decode_seconds=decode_seconds,
            details={
                "encoder": self.config.encoder,
                "crf": self.config.crf,
                "preset": self.config.preset,
                "pix_fmt": self.config.pix_fmt,
                "framerate": self.config.framerate,
                "intra_only": self.config.intra_only,
                "container": self.config.container,
                "temporal_prediction": not self.config.intra_only,
            },
            msssim_frame_count=metrics.msssim_frame_count,
        )


# --------------------------------------------------------------------------
# The neural codec
# --------------------------------------------------------------------------


class NVCCodec(Codec):
    """The existing neural codec, measured frame-by-frame.

    Uses the Python API (`encode_frame` / `decode_frame`) directly rather
    than shelling out to scripts/encode.py, and re-implements none of the
    quantization, entropy coding, or container logic.

    Real `.nvc` files are written to the working directory and their
    on-disk size is what gets measured - header included - so the figure is
    the same kind of quantity as the H.264/H.265 container size.
    """

    name = "nvc"

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        params: QuantizationParams,
        entropy_model: EmpiricalEntropyModel,
        checkpoint_name: str,
        device: torch.device,
        keep_files: bool = False,
    ) -> None:
        self.model = model
        self.params = params
        self.entropy_model = entropy_model
        self.checkpoint_name = checkpoint_name
        self.device = device
        self.keep_files = keep_files
        self.configuration = f"{params.bits}bit-{params.mode}"

    def run(self, sequence: BenchmarkSequence, workdir: Path) -> CodecResult:
        workdir = Path(workdir)
        nvc_dir = workdir / "nvc_files"
        nvc_dir.mkdir(parents=True, exist_ok=True)

        metrics = FrameMetrics()
        total_bytes = 0
        encode_seconds = 0.0
        decode_seconds = 0.0

        for index, reference_path in enumerate(sequence.frame_paths, start=1):
            reference = read_image_as_tensor(reference_path).unsqueeze(0).to(self.device)

            start = time.perf_counter()
            encoded = encode_frame(
                self.model, reference,
                params=self.params, entropy_model=self.entropy_model,
            )
            encode_seconds += time.perf_counter() - start

            # Write the real container and measure the file, not len(bytes).
            nvc_path = nvc_dir / f"frame_{index:06d}.nvc"
            nvc_path.write_bytes(encoded.data)
            total_bytes += nvc_path.stat().st_size

            try:
                start = time.perf_counter()
                reconstruction, _ = decode_frame(
                    self.model, nvc_path.read_bytes(), entropy_model=self.entropy_model,
                )
                decode_seconds += time.perf_counter() - start

                metrics.add(
                    reconstruction.to(reference.device), reference,
                    frame_label=f"{sequence.sequence_id}/frame_{index:06d}",
                )
            finally:
                # Clean up even on a decode failure - otherwise a bad frame
                # mid-sequence leaves its .nvc file orphaned on disk rather
                # than relying entirely on the caller's end-of-run cleanup
                # (which is skipped entirely under --keep-temp).
                if not self.keep_files:
                    nvc_path.unlink(missing_ok=True)

        return CodecResult(
            codec=self.name,
            configuration=self.configuration,
            sequence_id=sequence.sequence_id,
            dataset=sequence.dataset,
            frame_count=sequence.frame_count,
            width=sequence.width,
            height=sequence.height,
            total_bytes=total_bytes,
            mean_psnr=metrics.mean_psnr,
            mean_msssim=metrics.mean_msssim,
            pooled_mse=metrics.pooled_mse,
            encode_seconds=encode_seconds,
            decode_seconds=decode_seconds,
            details={
                "checkpoint": self.checkpoint_name,
                "quantization_bits": self.params.bits,
                "quantization_mode": self.params.mode,
                "entropy_model_id": self.entropy_model.model_id().hex(),
                "temporal_prediction": False,
                "intra_only": True,
            },
            msssim_frame_count=metrics.msssim_frame_count,
        )
