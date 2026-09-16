"""Freeze the research codec state into `nvc.video` codec bundles.

Every research script rebuilds its codec from TRAIN data on each run
(`m21_refinement.prepare_rate_point`: grid calibration, M13 table fitting, the M14
motion table). This script runs that rebuild ONCE per operating point and writes
the result as a `CodecBundle`, the self-contained file `nvc.video.VideoCodec`
loads. Two arms per bit depth:

  deployed   the frozen production stack - M10H motion, M11-G16, K=512 codebook,
             M13 recalibrated tables, M14 motion table
  m22        the same with M22's locked `symmetric_p01` residual grid and its
             refitted stack, loaded from the SHA256-verified M22 checkpoint

Every identity the bundle records is cross-checked against the identity the
research rig itself computed, before the file is written. Byte-level equivalence
on DAVIS TEST is then proven separately by `scripts/verify_promoted_codec.py`.

Output: `outputs/codec_bundles/nvc_<arm>_<bits>bit.pt` plus `bundles.json`
(identities, file digests and provenance).

Run:
  ./.venv/Scripts/python.exe scripts/export_codec_bundle.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything
from nvc.video.bundle import CodecBundle

ARMS = ("deployed", "m22")
M22_CANDIDATE = "symmetric_p01"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt"))
    parser.add_argument("--m11-dir", type=Path, default=Path("outputs/m11_autoregressive_entropy"))
    parser.add_argument("--m10k-dir", type=Path, default=Path("outputs/m10k_learned_entropy"))
    parser.add_argument("--m22-dir", type=Path, default=Path("outputs/m22_residual_freeze_lift"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/codec_bundles"))
    parser.add_argument("--rate-points", type=int, nargs="+", default=[5, 4, 3])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--train-frames-per-sequence", type=int, default=8)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)
    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    cx = _load_script("m11_causal_context")
    md = _load_script("m11_data")
    m21 = _load_script("m21_refinement")
    m22 = _load_script("m22_residual")
    mech = _load_script("m22_mechanism")

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    checkpoint_sha = _sha256(args.checkpoint)

    train_full = discover_sequences(args.manifest, split="train")
    motion_train = m21.broad_train_sequences(
        args.manifest, frames_per_sequence=args.train_frames_per_sequence)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "bundles.json"
    manifest: dict[str, Any] = (json.loads(manifest_path.read_text(encoding="utf-8"))
                                if manifest_path.is_file() else {"bundles": {}})
    manifest["bundles"] = manifest.get("bundles", {})

    for bits in args.rate_points:
        print(f"\n---- {bits}-bit: rebuilding the research rig ----", flush=True)
        rig = m21.prepare_rate_point(
            model, bits=bits, manifest=args.manifest, checkpoint=args.checkpoint,
            m11_dir=args.m11_dir, m10k_dir=args.m10k_dir, device=device,
            cache_dir=args.cache_dir or md.DEFAULT_CACHE_DIR, train_full=train_full,
            motion_train=motion_train, calibration_frames=args.calibration_frames,
            gop_size=args.gop, block_size=args.block_size, search_range=args.search_range,
            motion_cache=args.m22_dir / "m22_deployed_motion_table.json")

        for arm in args.arms:
            if arm == "deployed":
                residual_params = rig["residual_params"]
                context_model = rig["model11"]
                assign, coding = rig["assign_codebook"], rig["coding_codebook"]
                m10k_identity, signature = rig["m10k_identity"], rig["signature"]
                residual_identity = rig["residual_identity"]
                source = {"arm": "deployed", "description":
                          "M10H + M11-G16 + K=512 + M13 recalibrated tables + M14 motion table"}
            else:
                path = args.m22_dir / "checkpoints" / f"m22_{M22_CANDIDATE}_{bits}bit.pt"
                loaded = mech.load_refit_checkpoint(path, device=device, m22=m22, ma=ma, ml=ml, cx=cx)
                payload = torch.load(path, map_location="cpu", weights_only=False)
                spec = loaded["spec"]
                residual_params = loaded["residual_params"]
                context_model = spec["model"]
                assign, coding = spec["assign_codebook"], spec["coding_codebook"]
                m10k_identity = payload["entropy_identities"]["m10k"]
                signature = payload["quantizer_identity"]["calibration_signature"]
                residual_identity = spec["identity"].hex()
                source = {"arm": "m22", "description":
                          f"deployed stack with M22's locked {M22_CANDIDATE} residual grid "
                          "and its refitted M10K/G16/K=512/M13 stack",
                          "m22_checkpoint": str(path).replace("\\", "/"),
                          "m22_checkpoint_sha256": loaded["record"]["sha256"]}

            name = f"nvc_{arm}_{bits}bit"
            bundle = CodecBundle.build(
                name=name, bits=bits, gop_size=args.gop, block_size=args.block_size,
                search_range=args.search_range, calibration_frames=args.calibration_frames,
                autoencoder=model, intra_params=rig["intra_params"],
                intra_entropy_model=rig["intra_entropy_model"], residual_params=residual_params,
                context_model=context_model, assign_codebook_frequencies=assign.frequencies,
                assign_codebook_metric=assign.metric,
                coding_codebook_frequencies=coding.frequencies,
                motion_entropy_model=rig["motion_entropy_model"], m10k_identity=m10k_identity,
                calibration_signature=signature, residual_entropy_model_id=residual_identity,
                provenance={**source,
                            "autoencoder_checkpoint": str(args.checkpoint).replace("\\", "/"),
                            "autoencoder_checkpoint_sha256": checkpoint_sha,
                            "calibration_split": "train", "exported_at_commit": _git_head(),
                            "exported_utc": datetime.now(timezone.utc).isoformat()})

            # The bundle recomputed its identities in build(); they must also equal
            # the identities the research rig computed for the same state.
            expected = {"intra": rig["intra_identity"], "motion": rig["motion_identity"],
                        "residual": residual_identity}
            actual = {"intra": bundle.intra_entropy_model_id, "motion": bundle.motion_entropy_model_id,
                      "residual": bundle.residual_entropy_model_id}
            if expected != actual:
                print(f"[ERROR] {name}: identities differ from the research rig: "
                      f"{expected} vs {actual}", file=sys.stderr)
                return 1

            path = args.output_dir / f"{name}.pt"
            file_sha = bundle.save(path)
            CodecBundle.load(path)                              # re-verifies from disk
            manifest["bundles"][name] = {
                "file": path.name, "sha256": file_sha, "bytes": path.stat().st_size,
                "bits": bits, "arm": arm,
                "identities": {"intra": bundle.intra_entropy_model_id,
                               "residual": bundle.residual_entropy_model_id,
                               "motion": bundle.motion_entropy_model_id,
                               "calibration_signature": bundle.calibration_signature,
                               "autoencoder_sha256": bundle.autoencoder_sha256},
                "provenance": bundle.provenance,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"  {name}: {path.stat().st_size:,} bytes  residual {residual_identity}  "
                  f"intra {bundle.intra_entropy_model_id}  motion {bundle.motion_entropy_model_id}",
                  flush=True)

    print(f"\nManifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
