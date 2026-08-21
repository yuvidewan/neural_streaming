"""Rate-distortion benchmark: NVC vs. H.264 vs. H.265 on real sequences.

Encodes each sequence with every selected codec configuration, measures the
real compressed file size, decodes it back, and scores the decoded frames
against the originals with the SAME project-side PSNR/MS-SSIM
implementations for every codec.

METHODOLOGY LIMITATION (also recorded in every results.json):
NVC currently operates as an intra-only, frame-independent neural codec,
while H.264 and H.265 exploit temporal redundancy through inter-frame
prediction. Pass --intra-only for the closer like-for-like comparison.

Nothing here is hardcoded or estimated - every number comes from an encode
and decode executed during the run.

Example usage (PowerShell, from the project root, with .venv activated):

    # Smoke test: 2 sequences, 10 frames each - validates the whole path fast
    python scripts\\benchmark_rd.py `
        --checkpoint outputs\\checkpoints\\davis_baseline_best.pt `
        --calibration outputs\\calibration\\latent_quantization.json `
        --codecs nvc h264 h265 `
        --max-sequences 2 --max-frames-per-sequence 10

    # Full DAVIS test split
    python scripts\\benchmark_rd.py `
        --checkpoint outputs\\checkpoints\\davis_baseline_best.pt `
        --calibration outputs\\calibration\\latent_quantization.json `
        --codecs nvc h264 h265

Run `python scripts\\benchmark_rd.py --help` for the full option list.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

import torch

from nvc.compression import EmpiricalEntropyModel, QuantizationParams, load_calibration
from nvc.data.validation import DatasetValidationError
from nvc.evaluation.codecs import (
    DEFAULT_CRF_VALUES,
    DEFAULT_NVC_BIT_DEPTHS,
    DEFAULT_PIX_FMT,
    DEFAULT_PRESET,
    FFmpegCodecConfig,
    FFmpegVideoCodec,
    NVCCodec,
)
from nvc.evaluation.ffmpeg import (
    FFmpegError,
    ffmpeg_version,
    is_ffmpeg_available,
    require_encoders,
)
from nvc.evaluation.rd_benchmark import (
    BenchmarkRun,
    CalibrationMismatchError,
    build_metadata,
    check_calibration_fit,
    create_run_directory,
    require_calibration_fit,
    write_results,
)
from nvc.evaluation.sequences import discover_sequences, validate_sequence_frames
from nvc.training import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

_CODEC_CHOICES = ("nvc", "h264", "h265")
_ENCODER_FOR = {"h264": "libx264", "h265": "libx265"}


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the NVC neural codec against H.264 and H.265.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default="davis", help="Dataset label recorded in results.")
    parser.add_argument(
        "--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json",
        help="Frame manifest defining the sequences to benchmark.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--codecs", nargs="+", default=list(_CODEC_CHOICES), choices=_CODEC_CHOICES,
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Model checkpoint for NVC. Required when 'nvc' is in --codecs.",
    )
    parser.add_argument(
        "--calibration", type=Path, default=None,
        help="Calibration JSON matching --checkpoint. Required when 'nvc' is selected. "
             "Additional per-bit-depth files are looked up automatically.",
    )
    parser.add_argument(
        "--nvc-bits", type=int, nargs="+", default=list(DEFAULT_NVC_BIT_DEPTHS),
        help="NVC quantization bit depths to benchmark.",
    )
    parser.add_argument(
        "--crf", type=int, nargs="+", default=list(DEFAULT_CRF_VALUES),
        help="CRF operating points for H.264/H.265 (lower = higher quality).",
    )
    parser.add_argument("--preset", default=DEFAULT_PRESET, help="x264/x265 preset.")
    parser.add_argument(
        "--pix-fmt", default=DEFAULT_PIX_FMT,
        help="FFmpeg pixel format. yuv420p subsamples chroma; NVC codes full RGB.",
    )
    parser.add_argument(
        "--intra-only", action="store_true",
        help="Force all-intra H.264/H.265 (every frame a keyframe) for a closer "
             "like-for-like comparison against frame-independent NVC.",
    )
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=defaults.benchmarks_dir)
    parser.add_argument("--run-name", default=None, help="Defaults to a UTC timestamp.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Keep intermediate encode/decode artifacts under the run's tmp/ directory.",
    )
    parser.add_argument(
        "--allow-calibration-mismatch", action="store_true",
        help="Proceed even when the calibration does not fit the checkpoint. Results "
             "will NOT be methodologically valid and are flagged as such.",
    )
    return parser


def _calibration_path_for_bits(base: Path, bits: int, default_bits: int) -> Path:
    """Locate the calibration file for a given bit depth.

    The project's convention from Milestone 6: the primary depth lives in
    the base filename and other depths get a `_<n>bit` suffix.
    """
    if bits == default_bits:
        return base
    candidate = base.with_name(f"{base.stem}_{bits}bit{base.suffix}")
    return candidate


def _build_nvc_codecs(args, sequences, device) -> tuple[list, dict]:
    if args.checkpoint is None or args.calibration is None:
        raise SystemExit(
            "[ERROR] --checkpoint and --calibration are required when 'nvc' is selected."
        )
    if not args.checkpoint.is_file():
        raise SystemExit(f"[ERROR] Checkpoint not found: {args.checkpoint}")

    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device=device)

    base_calibration = load_calibration(args.calibration)
    default_bits = int(base_calibration["quantization"]["bits"])

    codecs = []
    calibration_report = {}
    for bits in args.nvc_bits:
        path = _calibration_path_for_bits(args.calibration, bits, default_bits)
        if not path.is_file():
            raise SystemExit(
                f"[ERROR] No calibration for {bits}-bit at {path}.\n"
                f"        Generate it with:\n"
                f"        python scripts\\calibrate_quantizer.py "
                f"--checkpoint {args.checkpoint} --bits {bits} --output {path}"
            )
        document = load_calibration(path)
        params = QuantizationParams.from_dict(document["quantization"])
        entropy_model = EmpiricalEntropyModel.from_dict(document["entropy_model"])

        fit = check_calibration_fit(model, sequences, params, device=device)
        fit["calibration"] = str(path)
        fit["bits"] = bits
        calibration_report[f"{bits}bit"] = fit

        print(
            f"  [calibration] {bits}-bit: {fit['clipped_percent']:.2f}% of latents clip "
            f"({'OK' if fit['fits'] else 'MISMATCH'}) - {path.name}"
        )
        if not fit["fits"]:
            if not args.allow_calibration_mismatch:
                require_calibration_fit(
                    fit, checkpoint=str(args.checkpoint), calibration=str(path)
                )
            print(
                f"  [WARNING] Proceeding with a mismatched calibration for {bits}-bit "
                "because --allow-calibration-mismatch was passed. These NVC results are "
                "NOT methodologically valid."
            )

        codecs.append(NVCCodec(
            model, params=params, entropy_model=entropy_model,
            checkpoint_name=args.checkpoint.name, device=device,
        ))

    return codecs, {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "calibration_fit": calibration_report,
        "calibration_mismatch_allowed": bool(args.allow_calibration_mismatch),
    }


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    if not args.manifest.is_file():
        print(f"[ERROR] Manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    try:
        sequences = discover_sequences(
            args.manifest, split=args.split, dataset=args.dataset,
            max_sequences=args.max_sequences,
            max_frames_per_sequence=args.max_frames_per_sequence,
        )
    except DatasetValidationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    total_frames = sum(s.frame_count for s in sequences)
    print(f"Benchmarking {len(sequences)} sequence(s), {total_frames} frames, "
          f"split='{args.split}'.")

    classical = [c for c in args.codecs if c in _ENCODER_FOR]
    version_line = None
    if classical:
        if not is_ffmpeg_available():
            print(
                "[ERROR] ffmpeg/ffprobe were not found on PATH, which "
                f"{'/'.join(classical)} require.\n"
                "        Install FFmpeg and open a NEW terminal, or re-run with "
                "--codecs nvc to benchmark the neural codec alone.",
                file=sys.stderr,
            )
            return 1
        try:
            require_encoders([_ENCODER_FOR[c] for c in classical])
            version_line = ffmpeg_version()
        except FFmpegError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        print(f"Using {version_line}")

    codecs = []
    nvc_metadata: dict = {}
    if "nvc" in args.codecs:
        try:
            nvc_codecs, nvc_metadata = _build_nvc_codecs(args, sequences, device)
        except CalibrationMismatchError as exc:
            print(f"\n[ERROR] {exc}\n", file=sys.stderr)
            return 1
        codecs.extend(nvc_codecs)

    for name in classical:
        for crf in args.crf:
            codecs.append(FFmpegVideoCodec(FFmpegCodecConfig(
                name=name, encoder=_ENCODER_FOR[name], crf=crf,
                preset=args.preset, pix_fmt=args.pix_fmt, intra_only=args.intra_only,
            )))

    if not codecs:
        print("[ERROR] No codec configurations selected.", file=sys.stderr)
        return 1

    run_dir = create_run_directory(args.output_dir, args.run_name)
    temp_root = run_dir / "tmp"
    metadata = build_metadata(
        dataset=args.dataset, sequences=sequences, codecs=codecs,
        checkpoint=args.checkpoint if "nvc" in args.codecs else None,
        calibration=args.calibration if "nvc" in args.codecs else None,
        device=device, seed=args.seed, ffmpeg_version=version_line,
        extra={
            "split": args.split,
            "crf_values": args.crf if classical else [],
            "nvc_bit_depths": args.nvc_bits if "nvc" in args.codecs else [],
            "preset": args.preset if classical else None,
            "pix_fmt": args.pix_fmt if classical else None,
            "intra_only": args.intra_only,
            "max_sequences": args.max_sequences,
            "max_frames_per_sequence": args.max_frames_per_sequence,
            **nvc_metadata,
        },
    )
    run = BenchmarkRun(output_dir=run_dir, metadata=metadata, results=[])

    print(f"\nRunning {len(codecs)} codec configuration(s) x {len(sequences)} sequence(s)...")
    failures = []
    for sequence in sequences:
        try:
            validate_sequence_frames(sequence)
        except DatasetValidationError as exc:
            print(f"  [SKIP] {sequence.sequence_id}: {exc}")
            failures.append({"sequence_id": sequence.sequence_id, "message": str(exc)})
            continue

        for codec in codecs:
            label = f"{codec.name}/{codec.configuration}"
            workdir = temp_root / sequence.sequence_id / f"{codec.name}_{codec.configuration}"
            workdir.mkdir(parents=True, exist_ok=True)
            try:
                result = codec.run(sequence, workdir)
                run.add(result)
                print(
                    f"  {sequence.sequence_id:<16} {label:<22} "
                    f"{result.bpp:>7.4f} bpp  {result.mean_psnr:>6.2f} dB  "
                    f"MS-SSIM {result.mean_msssim if result.mean_msssim is None else f'{result.mean_msssim:.4f}'}"
                )
            except Exception as exc:  # noqa: BLE001 - one bad config must not abort the run
                print(f"  [FAIL] {sequence.sequence_id} {label}: {exc}")
                failures.append({
                    "sequence_id": sequence.sequence_id, "codec": label,
                    "message": str(exc), "traceback": traceback.format_exc(limit=3),
                })

    run.metadata["failures"] = failures
    paths = write_results(run_dir, run)

    if not args.keep_temp and temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)

    print("\n" + "=" * 78)
    print(f"Benchmark complete - {len(run.results)} measurement(s), {len(failures)} failure(s)")
    print("=" * 78)
    print(run.metadata["methodology_note"])
    print("=" * 78)
    for key, path in paths.items():
        print(f"{key:>14}: {path}")
    print(f"\nPlot with:\n  python scripts\\plot_rate_distortion.py --run-dir {run_dir}")
    return 0 if run.results else 1


if __name__ == "__main__":
    sys.exit(main())
