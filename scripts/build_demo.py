"""Precompute the demo: a codec to run, and a few clips already encoded with it.

    python scripts/build_demo.py

Two things happen, both under outputs/demo/ (git-ignored):

1. If no trained codec bundle is present in outputs/codec_bundles/ - they are
   git-ignored, so a fresh clone has none - a STAND-IN bundle is built per bit
   depth from a local autoencoder checkpoint and training frames. A stand-in is
   a working codec with real pictures but an untrained entropy model; the page
   labels it as one, and its bitrate is not a project result. Once the trained
   bundles are copied in, they take over and the stand-ins are ignored.
2. A few sequences from the test frames are turned into demo clips and encoded
   with every chosen bundle, alongside H.264 at the same bitrate, so the page
   has something to show the moment it opens.

Then run scripts/run_demo.py.

Example usage (PowerShell, from the project root, with .venv activated):

    python scripts\\build_demo.py --sequences bmx-bumps drone --max-frames 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from nvc.demo import bundles as demo_bundles
from nvc.demo import clips as demo_clips
from nvc.demo.pipeline import process_clip, to_float
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.video.codec import AUTOENCODER_STRIDE, VideoCodec

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEQUENCES = ("bmx-bumps", "drone", "cows", "car-turn")
# Tried in order for the stand-in's autoencoder: newest training lineage first.
CHECKPOINT_CANDIDATES = (
    "outputs/qat_combined/checkpoints_qat_noise/best.pt",
    "outputs/checkpoints/vimeo_qat_noise_best.pt",
    "outputs/checkpoints/vimeo_epoch17_best.pt",
    "outputs/checkpoints/davis_baseline_best.pt",
)
CALIBRATION_SEQUENCES = 12
CALIBRATION_FRAMES_EACH = 10


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the demo's codec bundles (if needed) and precomputed clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--frames-dir", type=Path, default=None,
                        help="Folder of <sequence>_<index>.png test frames "
                             "(default: <repo>/data/frames/test).")
    parser.add_argument("--calibration-dir", type=Path, default=None,
                        help="Folder of training frames for a stand-in bundle "
                             "(default: <repo>/data/frames/train).")
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES),
                        help="Sequence names to turn into clips.")
    parser.add_argument("--max-frames", type=int, default=demo_clips.DEFAULT_MAX_FRAMES)
    parser.add_argument("--size", type=int, default=demo_clips.CLIP_SIZE,
                        help="Clip side in pixels; the codec is only measured at 256.")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5],
                        help="Bit depths of the stand-in bundles to build.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Autoencoder checkpoint for a stand-in bundle "
                             "(default: the newest known local checkpoint).")
    parser.add_argument("--force-standin", action="store_true",
                        help="Build stand-in bundles even if trained ones are present.")
    parser.add_argument("--all-bundles", action="store_true",
                        help="Encode the clips with every bundle, not only the default family.")
    parser.add_argument("--no-h264", action="store_true", help="Skip the H.264 comparison.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def group_by_sequence(directory: Path) -> dict[str, list[Path]]:
    """<sequence>_<index>.png files, grouped by sequence and ordered by index."""
    groups: dict[str, list[Path]] = {}
    for path in sorted(directory.glob("*.png")):
        name, separator, index = path.stem.rpartition("_")
        if separator and index.isdigit():
            groups.setdefault(name, []).append(path)
    return groups


def find_checkpoint(repo_root: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.is_file() else None
    for candidate in CHECKPOINT_CANDIDATES:
        if (repo_root / candidate).is_file():
            return repo_root / candidate
    return None


def build_standins(args, repo_root: Path, device: str) -> int:
    """Write one stand-in bundle per requested bit depth. Returns an exit code."""
    checkpoint = find_checkpoint(repo_root, args.checkpoint)
    if checkpoint is None:
        wanted = args.checkpoint or "any of: " + ", ".join(CHECKPOINT_CANDIDATES)
        print(f"[ERROR] no autoencoder checkpoint found ({wanted}). A stand-in bundle needs "
              "one; pass --checkpoint, or copy trained bundles into outputs/codec_bundles/.",
              file=sys.stderr)
        return 1
    calibration_dir = args.calibration_dir or repo_root / "data" / "frames" / "train"
    groups = {name: paths for name, paths in group_by_sequence(calibration_dir).items()
              if len(paths) >= 2} if calibration_dir.is_dir() else {}
    if not groups:
        print(f"[ERROR] no calibration frames in {calibration_dir} (expected "
              "<sequence>_<index>.png, two or more per sequence).", file=sys.stderr)
        return 1
    names = sorted(groups)
    stride = max(1, len(names) // CALIBRATION_SEQUENCES)
    chosen = names[::stride][:CALIBRATION_SEQUENCES]
    sequences = [to_float(demo_clips.frames_from_images(
        groups[name], max_frames=CALIBRATION_FRAMES_EACH, size=args.size)).to(device)
        for name in chosen]

    model, _ = load_model_from_checkpoint(checkpoint, device=device)
    if type(model).__name__ != "BaselineAutoencoder":
        print(f"[ERROR] {checkpoint} holds a {type(model).__name__}; codec bundles carry the "
              "baseline autoencoder only.", file=sys.stderr)
        return 1
    out_dir = repo_root / demo_bundles.STANDIN_DIR
    for bits in args.bits:
        bundle = demo_bundles.build_standin_bundle(
            model, sequences, bits=bits,
            provenance={"autoencoder_checkpoint": str(checkpoint),
                        "calibration_sequences": chosen})
        bundle.save(out_dir / f"standin_{bits}bit.pt")
        print(f"  built stand-in bundle: standin_{bits}bit.pt "
              f"({len(chosen)} calibration sequences, {checkpoint.name})")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    if args.size < AUTOENCODER_STRIDE or args.size % AUTOENCODER_STRIDE:
        print(f"[ERROR] --size must be a positive multiple of {AUTOENCODER_STRIDE}.", file=sys.stderr)
        return 1
    if args.max_frames < 2:
        print("[ERROR] --max-frames must be at least 2.", file=sys.stderr)
        return 1
    frames_dir = args.frames_dir or repo_root / "data" / "frames" / "test"
    if not frames_dir.is_dir():
        print(f"[ERROR] frames folder not found: {frames_dir}", file=sys.stderr)
        return 1
    available = group_by_sequence(frames_dir)
    missing = [name for name in args.sequences if name not in available]
    if missing:
        print(f"[ERROR] no frames for sequence(s) {', '.join(missing)} in {frames_dir}. "
              f"Found: {', '.join(sorted(available)) or 'none'}.", file=sys.stderr)
        return 1
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    found = demo_bundles.discover_bundles(repo_root)
    have_trained = any(b.kind == demo_bundles.TRAINED_KIND for b in found)
    if args.force_standin or not have_trained:
        print("No trained codec bundle found; building stand-ins." if not have_trained
              else "Building stand-in bundles (--force-standin).")
        code = build_standins(args, repo_root, device)
        if code:
            return code
        found = demo_bundles.discover_bundles(repo_root)
    if not found:
        print("[ERROR] no usable codec bundle.", file=sys.stderr)
        return 1

    chosen = demo_bundles.default_bundle(found)
    family = (chosen.kind, chosen.label.split(",")[0])
    selected = found if args.all_bundles else [
        b for b in found if (b.kind, b.label.split(",")[0]) == family]
    print(f"Codec bundles: {', '.join(b.label for b in selected)} (device: {device})")

    clips_dir = repo_root / "outputs" / "demo" / "clips"
    codecs = {b.id: VideoCodec.from_file(b.path, device=device) for b in selected}
    for name in args.sequences:
        frames = demo_clips.frames_from_images(available[name], max_frames=args.max_frames,
                                               size=args.size)
        clip = demo_clips.create_clip(clips_dir, frames, name=name, source="precomputed",
                                      clip_id=demo_clips.slugify(name))
        for bundle in selected:
            result = process_clip(clips_dir / clip["id"], bundle, codecs[bundle.id],
                                  with_h264=not args.no_h264)
            nvc, h264 = result["nvc"], result["h264"]
            line = (f"  {name:<14} {bundle.label:<18} {nvc['total_bytes']:>8} B  "
                    f"{nvc['bpp']:.3f} bpp  {nvc['psnr']:.2f} dB  "
                    f"exact={nvc['decoder_exact']}")
            if h264.get("available"):
                line += f"  | H.264 crf{h264['crf']} {h264['psnr']:.2f} dB"
            print(line)
            if not nvc["decoder_exact"]:
                print("[ERROR] the decoder did not reproduce the encoder's frames.", file=sys.stderr)
                return 1
    print(f"Demo data written to {clips_dir.parent}. Next: python scripts/run_demo.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
