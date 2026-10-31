"""PARITY_ROADMAP Stage 1 gate - intra-only BD-rate for ANY transform.

WHY THIS EXISTS
---------------
`benchmark_parity.py --gop 1` measured the gate's denominator (+176.2% PSNR for
the deployed codec against x264 all-intra), but it gets there through
`m21.prepare_rate_point`, which rebuilds the whole deployed stack: the M11-G16
context model, the M10K learned entropy model, the K=512 codebooks, M14's motion
table, and a provenance check on each. Every one of those is fitted to a
**64-channel** latent. The Stage 1 transform's latent has 192 channels, and
`ChannelContextEntropyModel`'s `nn.Embedding(latent_channels, hidden)` alone
makes the trained G16 checkpoint unloadable against it. Taken at face value that
means "retrain the entropy stack before you can measure anything".

It does not, because **an I-frame never touches any of it.** Look at
`m21.encode_sequence_refined`: the P-branch uses the context model, the
codebooks and motion; the I-branch uses `intra_params` and
`intra_entropy_model`, and nothing else. At GOP 1 there are no P-frames - which
is exactly why `nvc_deployed` and `nvc_m22` came out byte-identical in the
`--gop 1` run, M22's residual re-centering being inert without residuals.

So this script needs one thing: `calibrate_grids`, which just runs the
autoencoder over TRAIN frames and fits the grids. It is architecture-agnostic.
No trained entropy model, no codebook, no provenance gate, no retraining.

WHAT IT MEASURES, AND AGAINST WHAT
----------------------------------
All-intra `.nvct` streams over DAVIS TEST, scored by importing
`benchmark_parity`'s own `score_frames` / `aggregate_quality` / `bd_rate` rather
than reimplementing them - the credibility of the Stage 0 and intra-only
scoreboards rests on one scorer, and a second copy of it would end that.

The x264 all-intra curve is read from the committed
`outputs/benchmarks/parity_intra/parity_intra.json` rather than re-encoded, so
the reference side is bit-for-bit the same curve the +176.2% came from and
FFmpeg does not need to run at all.

THE HEADER'S THREE UNUSED FIELDS
--------------------------------
`.nvct` v2 records `residual_entropy_model_id`, `motion_entropy_model_id` and
`motion_bits` whether or not the stream has P-frames. The deployed path fills
the residual one from the M11-G16+M13 identity, which does not exist here. This
fills it from `calibrate_grids`'s own static residual model instead.

That changes 8 bytes of a **fixed 56-byte** header and nothing else: the ids are
truncated/padded to 8 bytes by `TemporalStreamHeader.pack`, and the quantization
blocks that follow are sized by the params' `numel`, which comes from the same
`calibrate_grids` call either way. So total stream bytes are unchanged, and
`--verify` checks exactly that against the committed run.

THE UNIT STEP (`--quantizer unit_step`)
---------------------------------------
M24 Phase B trains each transform for ONE fixed rounding step (bmshj2018's rule)
and gets its rate from lambda, so its operating points are not bit depths of one
model but one point per trained model. With `--quantizer unit_step`, every
`--checkpoint` is coded at every `--unit-steps` value, and all of them form ONE
curve - pass the whole lambda sweep at `--unit-steps 1.0` to gate it the way
PHASE_B.md asks ("gate the sweep as a curve, never a single run").

The quantizer is the same affine grid with `scale = step` and an integer zero
point (`calibration.unit_step_quantization_params`), so the `.nvct` format, the
range coder and the per-channel histogram tables are all unchanged and the
stream stays self-contained. Only the alphabet is fitted on TRAIN, and no
percentile clipping happens; any TEST value beyond the TRAIN span is clamped and
counted in the point's `clipped_values`. There is no motion table in this rig, so
the header's motion id is zero and its motion bits are the search range's own.

Run:
  # reproduce the committed denominator with the lean path (the verification)
  ./.venv/Scripts/python.exe scripts/benchmark_intra_gate.py --verify

  # measure a Stage 1 checkpoint
  ./.venv/Scripts/python.exe scripts/benchmark_intra_gate.py \
      --checkpoint outputs/stage1_vimeo/checkpoints/best.pt \
      --output-dir outputs/benchmarks/intra_gate_stage1

  # gate a Phase B lambda sweep as one curve
  ./.venv/Scripts/python.exe scripts/benchmark_intra_gate.py --quantizer unit_step \
      --checkpoint <dir>_unitstep_lambda1e-4/checkpoints/best.pt \
                   <dir>_unitstep_lambda3e-4/checkpoints/best.pt \
                   <dir>_unitstep_lambda1e-3/checkpoints/best.pt \
      --output-dir outputs/benchmarks/intra_gate_phase_b
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.calibration import unit_step_quantization_params
from nvc.compression.codec import (
    decode_payload_to_latent,
    encode_latent_to_payload,
    latent_to_symbols,
)
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.compression.quantization import count_clipped
from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything
from nvc.video.container import (
    FRAME_TYPE_I,
    TemporalStreamHeader,
    TemporalStreamReader,
    TemporalStreamWriter,
)
from nvc.video.motion import deterministic_kernels, motion_alphabet_bits

# The committed intra-only run, whose x264 all-intra curve is the gate's
# reference and whose NVC totals are what --verify reproduces.
DENOMINATOR = Path("outputs/benchmarks/parity_intra/parity_intra.json")
REFERENCE_ARM = "h264_intra"
# The DEPLOYED autoencoder - M10F's lambda=3e-4 seed-42 model, the same default
# benchmark_parity.py uses. Not `outputs/checkpoints/vimeo_epoch17_best.pt`,
# which is where the lineage STARTS (Vimeo training, before the DAVIS
# fine-tune) and which calibrates to a different, visibly worse intra grid -
# getting this wrong is what made the first version of this script code
# 38% more bytes at 2.5 dB lower PSNR than the run it was meant to reproduce.
# `tests/test_benchmark_intra_gate.py` pins the two defaults together.
DEPLOYED_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
# calibrate_grids fits the residual grid and motion table on P-frames, so it has
# to run at a GOP that has some. The intra half it produces - the only half this
# script codes with - is fitted on I-frame latents and does not depend on the
# GOP. Same reasoning as benchmark_parity's calibration_gop.
CALIBRATION_GOP = 10


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Intra-only BD-rate for any analysis/synthesis transform.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, nargs="+", default=DEPLOYED_CHECKPOINT,
                        help="Any checkpoint; the architecture comes from the checkpoint itself. "
                             "The default is the deployed M10F model, so --verify compares like "
                             "with like. Several only with --quantizer unit_step, one point "
                             "each.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks/intra_gate"))
    parser.add_argument("--output-name", type=str, default="intra_gate.json")
    parser.add_argument("--quantizer", choices=("percentile", "unit_step"), default="percentile",
                        help="percentile: the deployed grid, one point per --rate-points depth. "
                             "unit_step: M24 Phase B's fixed-step rounding, one point per "
                             "checkpoint and --unit-steps value.")
    parser.add_argument("--rate-points", type=int, nargs="+", default=[5, 4, 3],
                        help="Quantizer bit depths, one operating point each (percentile).")
    parser.add_argument("--unit-steps", type=float, nargs="+", default=[1.0],
                        help="Rounding steps (unit_step). 1.0 is the step Phase B trains at.")
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--denominator", type=Path, default=DENOMINATOR,
                        help="The committed intra-only run supplying the x264 all-intra curve.")
    parser.add_argument("--verify", action="store_true",
                        help="Also check this run reproduces the denominator's own NVC totals, "
                             "PSNR and MS-SSIM. Only meaningful with the baseline checkpoint the "
                             "denominator was measured on; exits non-zero on a mismatch.")
    parser.add_argument("--max-sequences", type=int, default=None, help="Smoke runs only.")
    parser.add_argument("--max-frames", type=int, default=None, help="Smoke runs only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def intra_rig(mc, model, train_sequences, *, bits: int, calibration_frames: int,
              block_size: int, search_range: int) -> dict[str, Any]:
    """Everything an all-intra stream needs, and nothing else.

    `calibrate_grids` runs the autoencoder over TRAIN frames and fits the intra
    grid, the residual grid and the motion table. Only the intra grid is used for
    coding here; the other two supply header fields (see the module docstring).
    Crucially it loads no trained entropy model and checks no provenance, so it
    works for a transform whose latent has a different channel count from the
    one the deployed stack was fitted to.
    """
    calibration = mc.calibrate_grids(
        model, train_sequences, bits=bits, mode="per_channel", gop_size=CALIBRATION_GOP,
        block_size=block_size, search_range=search_range, reference_mode="mc",
        max_frames=calibration_frames)
    return {
        "bits": bits,
        "intra_params": calibration["intra_params"],
        "intra_entropy_model": calibration["intra_entropy_model"],
        "residual_params": calibration["residual_params"],
        "residual_entropy_model": calibration["residual_entropy_model"],
        "motion_entropy_model": calibration["motion_entropy_model"],
        "provenance": calibration["provenance"],
    }


@torch.no_grad()
def unit_step_rig(model, train_sequences, *, step: float,
                  calibration_frames: int) -> dict[str, Any]:
    """The unit-step counterpart of `intra_rig`: the alphabet and the per-channel
    tables, both fitted on the first `calibration_frames` TRAIN frames.

    There are no P-frames to fit a residual grid or a motion table on, so the
    residual slot repeats the intra params (the header gives both one bit depth)
    and the motion slot is empty.
    """
    model.eval()
    device = next(model.parameters()).device
    latents, seen = [], 0
    for sequence in train_sequences:
        frames = sequence.load_frames()
        for index in range(min(frames.shape[0], calibration_frames - seen)):
            latents.append(model.encode(frames[index:index + 1].to(device)).cpu())
        seen = len(latents)
        if seen >= calibration_frames:
            break
    stack = torch.cat(latents, dim=0)
    params = unit_step_quantization_params(stack, step=step)
    symbols = np.stack([latent_to_symbols(stack[i:i + 1], params).reshape(stack.shape[1], -1)
                        for i in range(stack.shape[0])])
    entropy_model = EmpiricalEntropyModel.from_symbols(
        symbols, bits=params.bits, num_tables=stack.shape[1])
    return {
        "bits": params.bits,
        "intra_params": params,
        "intra_entropy_model": entropy_model,
        "residual_params": params,
        "residual_entropy_model": entropy_model,
        "motion_entropy_model": None,
        "provenance": {"quantizer": "unit_step", "step": step, "split": "train",
                       "calibration_frames": int(stack.shape[0])},
    }


def encode_intra_sequence(model, frames: torch.Tensor, rig: dict[str, Any], path: Path,
                          *, block_size: int, search_range: int) -> dict[str, Any]:
    """Write every frame of `frames` as an I-frame to a `.nvct` v2 stream.

    Deliberately built on the promoted package (`nvc.video.container`,
    `nvc.compression.codec`) rather than on a research script: with no P-frames
    there is no motion, no residual and no context model, so none of the
    research-side machinery is needed and the loop is the eight lines below.
    `--verify` is what proves those eight lines agree with the research path.
    """
    model.eval()
    device = next(model.parameters()).device
    intra_params = rig["intra_params"]
    intra_entropy_model = rig["intra_entropy_model"]
    latent_shape = tuple(model.encode(frames[0:1].to(device)).shape[1:])
    motion_model = rig["motion_entropy_model"]

    header = TemporalStreamHeader(
        gop_size=1, quantization_bits=intra_params.bits,
        quantization_mode=intra_params.mode, image_width=frames.shape[3],
        image_height=frames.shape[2], image_channels=frames.shape[1],
        latent_channels=latent_shape[0], latent_height=latent_shape[1],
        latent_width=latent_shape[2], frame_count=frames.shape[0],
        num_intra_quantization_params=intra_params.scale.numel(),
        num_residual_quantization_params=rig["residual_params"].scale.numel(),
        block_size=block_size, search_range=search_range,
        motion_bits=(motion_model.bits if motion_model is not None
                     else motion_alphabet_bits(search_range)),
        reference_mode="mc",
        intra_entropy_model_id=intra_entropy_model.model_id(),
        residual_entropy_model_id=rig["residual_entropy_model"].model_id(),
        motion_entropy_model_id=(motion_model.model_id() if motion_model is not None
                                 else bytes(8)))

    writer = TemporalStreamWriter(path, header, intra_params, rig["residual_params"])
    reconstructions = []
    clipped = 0
    completed = False
    try:
        with deterministic_kernels():
            for index in range(frames.shape[0]):
                latent = model.encode(frames[index:index + 1].to(device))
                clipped += count_clipped(latent, intra_params.to(latent.device))["clipped_total"]
                payload, _ = encode_latent_to_payload(
                    latent, params=intra_params, entropy_model=intra_entropy_model)
                decoded, _ = decode_payload_to_latent(
                    payload, entropy_model=intra_entropy_model, params=intra_params,
                    shape=latent_shape)
                writer.append_frame(FRAME_TYPE_I, b"", payload)
                reconstructions.append(model.decode(decoded.to(device)).detach().cpu())
        completed = True
    finally:
        if completed:
            writer.close()
        else:
            try:
                writer.close()
            except Exception:
                pass
    return {"reconstructions": torch.cat(reconstructions, dim=0), "clipped_values": clipped}


def decode_intra_sequence(model, path: Path, rig: dict[str, Any]) -> torch.Tensor:
    """Decode a stream written by `encode_intra_sequence`, from the file alone."""
    model.eval()
    device = next(model.parameters()).device
    reader = TemporalStreamReader(path)
    header = reader.header
    latent_shape = header.latent_shape
    reconstructions = []
    with deterministic_kernels():
        # The reader yields (frame_type, motion_payload, residual_payload) tuples.
        for frame_type, motion_payload, residual_payload in reader:
            if frame_type != FRAME_TYPE_I:
                raise ValueError("an intra-only stream must contain only I-frames")
            if motion_payload:
                raise ValueError("an I-frame must carry no motion payload")
            decoded, _ = decode_payload_to_latent(
                residual_payload, entropy_model=rig["intra_entropy_model"],
                params=reader.intra_params, shape=latent_shape)
            reconstructions.append(model.decode(decoded.to(device)).detach().cpu())
    return torch.cat(reconstructions, dim=0)


def reference_curve(denominator: Path, convention: str, metric: str,
                    parity) -> list[tuple[float, float]]:
    """The x264 all-intra rate-quality curve from the committed run."""
    report = json.loads(denominator.read_text(encoding="utf-8"))
    points = [p for p in report["points"].values() if p["arm"] == REFERENCE_ARM]
    if len(points) < 2:
        raise SystemExit(
            f"{denominator} has fewer than two {REFERENCE_ARM} points; it cannot supply a "
            "reference curve. Re-run scripts/benchmark_parity.py --gop 1 first.")
    return parity.curve(points, convention, metric)


def recorded_intra_totals(denominator: Path) -> dict[int, dict[str, Any]]:
    """What the committed intra-only run measured for the deployed codec, keyed
    by bit depth - the target `--verify` reproduces."""
    report = json.loads(denominator.read_text(encoding="utf-8"))
    recorded = {}
    for point in report["points"].values():
        if point["arm"] == "nvc_deployed":
            recorded[point["bits"]] = {
                "total_bytes": point["total_bytes"],
                "quality": point["quality"],
            }
    return recorded


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    parity = _load_script("benchmark_parity")
    mc = _load_script("m10h_motion_compensation")

    checkpoints = args.checkpoint if isinstance(args.checkpoint, list) else [args.checkpoint]
    if args.quantizer == "percentile" and len(checkpoints) > 1:
        raise SystemExit("several --checkpoint values need --quantizer unit_step: a percentile "
                         "curve is the bit depths of ONE transform")
    if args.verify and args.quantizer != "percentile":
        raise SystemExit("--verify reproduces the committed percentile run; there is nothing "
                         "for it to check under --quantizer unit_step")
    loaded = []
    for path in checkpoints:
        unit_model, unit_checkpoint = load_model_from_checkpoint(path, device=device)
        loaded.append((path, unit_model.eval(), unit_checkpoint))
    _, model, checkpoint = loaded[0]
    architecture = checkpoint.get("architecture", "BaselineAutoencoder")

    sequences = discover_sequences(args.manifest, split="test",
                                   max_sequences=args.max_sequences,
                                   max_frames_per_sequence=args.max_frames)
    frame_cache = {s.sequence_id: s.load_frames() for s in sequences}
    train_sequences = discover_sequences(args.manifest, split="train")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "stage": "PARITY_ROADMAP Stage 1 gate - intra-only BD-rate",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "checkpoint": (str(checkpoints[0]) if len(checkpoints) == 1
                           else [str(c) for c in checkpoints]),
            "architecture": architecture,
            "model_config": checkpoint.get("model_config"),
            "parameters": model.num_parameters(),
            "quantizer": args.quantizer,
            "rate_points": args.rate_points if args.quantizer == "percentile" else None,
            "unit_steps": args.unit_steps if args.quantizer == "unit_step" else None,
            "gop": 1,
            "calibration_gop": CALIBRATION_GOP,
            "calibration_frames": args.calibration_frames,
            "block_size": args.block_size, "search_range": args.search_range,
            "denominator": str(args.denominator), "reference_arm": REFERENCE_ARM,
            "primary_convention": parity.PRIMARY_CONVENTION,
            "max_sequences": args.max_sequences, "max_frames": args.max_frames,
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": str(device), "platform": platform.platform()},
        "dataset": {"split": "test", "sequences": [s.sequence_id for s in sequences],
                    "frames": sum(s.frame_count for s in sequences)},
        "points": {},
    }

    print("=" * 100)
    print(f"INTRA-ONLY GATE - {architecture}, {model.num_parameters():,} parameters, "
          f"{args.quantizer} quantizer")
    print(f"{len(sequences)} sequences, {report['dataset']['frames']} frames, "
          f"primary convention: {parity.PRIMARY_CONVENTION}")
    print("=" * 100, flush=True)

    def operating_points():
        """(model, rig, label, setting, extra) for every point on the curve."""
        if args.quantizer == "percentile":
            for bits in args.rate_points:
                rig = intra_rig(mc, model, train_sequences, bits=bits,
                                calibration_frames=args.calibration_frames,
                                block_size=args.block_size, search_range=args.search_range)
                yield model, rig, f"intra/{bits}bit", f"{bits}bit", {"bits": bits}
            return
        for path, unit_model, unit_checkpoint in loaded:
            trained = unit_checkpoint.get("extra") or {}
            for step in args.unit_steps:
                rig = unit_step_rig(unit_model, train_sequences, step=step,
                                    calibration_frames=args.calibration_frames)
                tag = f"{path.parent.parent.name}/{path.stem}/step{step:g}"
                yield unit_model, rig, f"intra/unit_step/{tag}", f"step{step:g}", {
                    "bits": rig["bits"], "quantizer": "unit_step", "step": step,
                    "checkpoint": str(path),
                    # What the model was TRAINED for, so a point coded at a step it
                    # never saw (or a percentile-trained model) reads as such.
                    "trained_quantizer": trained.get("quantizer", "percentile"),
                    "trained_unit_step": trained.get("unit_step"),
                    "rate_lambda": trained.get("rate_lambda"),
                }

    stream_dir = Path(tempfile.mkdtemp(prefix="intra_gate_"))
    for point_model, rig, label, setting, extra in operating_points():
        if label in report["points"]:
            raise SystemExit(f"two operating points share the label {label!r}")
        rows, exact, clipped = [], True, 0
        for sequence in sequences:
            frames = frame_cache[sequence.sequence_id]
            path = stream_dir / f"{setting}_{sequence.sequence_id}.nvct"
            started = time.perf_counter()
            encoded = encode_intra_sequence(point_model, frames, rig, path,
                                            block_size=args.block_size,
                                            search_range=args.search_range)
            encode_seconds = time.perf_counter() - started
            size = path.stat().st_size
            started = time.perf_counter()
            decoded = decode_intra_sequence(point_model, path, rig)
            decode_seconds = time.perf_counter() - started
            path.unlink()
            # The decoder must reach the encoder's reconstruction from the file
            # alone; anything less means the stream is not self-contained.
            exact &= torch.equal(decoded, encoded["reconstructions"])
            clipped += encoded["clipped_values"]
            scores = parity.score_frames(
                parity.to_uint8_frames(decoded), frames, device=device)
            rows.append({"sequence": sequence.sequence_id, "bytes": size,
                         "pixels": sequence.total_pixels,
                         "encode_seconds": encode_seconds,
                         "decode_seconds": decode_seconds, **scores})

        point = parity.summarize_point(
            label, "nvc_intra", setting, rows,
            extra={**extra, "decode_reconstruction_exact": exact, "clipped_values": clipped,
                   "calibration_provenance": rig["provenance"]})
        report["points"][point["label"]] = point
        quality = point["quality"][parity.PRIMARY_CONVENTION]
        name = f"{rig['bits']}-bit" if args.quantizer == "percentile" else label
        print(f"  {name}  {point['total_bytes']:>10,} B  bpp {point['bpp']:.4f}  "
              f"PSNR {quality['psnr']:.3f}  MS-SSIM {quality['msssim']:.4f}  "
              f"clipped={clipped}  decode-exact={exact}", flush=True)

    nvc_points = list(report["points"].values())
    report["comparisons"] = {}
    for convention in parity.CONVENTIONS:
        for metric in ("psnr", "msssim"):
            base = reference_curve(args.denominator, convention, metric, parity)
            test = parity.curve(nvc_points, convention, metric)
            report["comparisons"][f"{convention}/{metric}"] = {
                "bd_rate_percent": parity.bd_rate(base, test),
                "quality_overlap": parity.overlap(base, test),
            }

    report["valid"] = all(p.get("decode_reconstruction_exact", True)
                          for p in report["points"].values())

    if args.verify:
        recorded = recorded_intra_totals(args.denominator)
        checks = []
        for point in nvc_points:
            expected = recorded.get(point["bits"])
            if expected is None:
                checks.append({"bits": point["bits"], "reproduces": None,
                               "reason": "no recorded total at this bit depth"})
                continue
            got = point["quality"][parity.PRIMARY_CONVENTION]
            want = expected["quality"][parity.PRIMARY_CONVENTION]
            checks.append({
                "bits": point["bits"],
                "recorded_bytes": expected["total_bytes"],
                "measured_bytes": point["total_bytes"],
                "bytes_match": point["total_bytes"] == expected["total_bytes"],
                "psnr_abs_diff_db": abs(got["psnr"] - want["psnr"]),
                "msssim_abs_diff": abs(got["msssim"] - want["msssim"]),
                "reproduces": (point["total_bytes"] == expected["total_bytes"]
                               and abs(got["psnr"] - want["psnr"]) < 1e-6
                               and abs(got["msssim"] - want["msssim"]) < 1e-6),
            })
        report["verification"] = checks
        report["valid"] = report["valid"] and all(
            c["reproduces"] is not False for c in checks)
        print("\nVERIFICATION against the committed intra-only run:")
        for check in checks:
            if check["reproduces"] is None:
                print(f"  {check['bits']}-bit  skipped: {check['reason']}")
            else:
                print(f"  {check['bits']}-bit  bytes {check['measured_bytes']:,} vs recorded "
                      f"{check['recorded_bytes']:,}  "
                      f"dPSNR {check['psnr_abs_diff_db']:.2e}  "
                      f"dMS-SSIM {check['msssim_abs_diff']:.2e}  "
                      f"-> {check['reproduces']}")

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report_path = args.output_dir / args.output_name
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    key = f"{parity.PRIMARY_CONVENTION}/psnr"
    headline = report["comparisons"][key]["bd_rate_percent"]
    print(f"\nBD-rate vs {REFERENCE_ARM} (primary convention, PSNR): "
          f"{'n/a' if headline is None else f'{headline:+.1f}%'}")
    print(f"run valid: {report['valid']}\nReport: {report_path}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
