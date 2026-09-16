"""Prove `nvc.video` is the research codec, byte for byte, on DAVIS TEST.

For every exported bundle and every DAVIS TEST sequence:

  1. encode with the PACKAGE (`nvc.video.VideoCodec`);
  2. encode with the RESEARCH path (`scripts/m21_refinement.encode_sequence_refined`
     at its identity candidate), using research objects rebuilt from the same
     bundle - so a difference can only come from code, not from state;
  3. require the two streams to be byte-identical and their reconstructions equal;
  4. decode the package stream with the package and require bit-exact frames.

Then, per bundle, require the total bytes to equal what M22 Phase 19 recorded on
DAVIS TEST for that arm and bit depth - which ties the bundle's frozen STATE back
to the research rig that produced every published number.

Run:
  ./.venv/Scripts/python.exe scripts/verify_promoted_codec.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.video import CodecBundle, VideoCodec

RECORDED_KEYS = {"deployed": "baseline", "m22": "candidate"}


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def research_spec(bundle: CodecBundle, *, device, ma, ml, cx) -> dict[str, Any]:
    model11 = ma.ChannelContextEntropyModel(**bundle.context_model_config)
    model11.load_state_dict(bundle.context_model_state)
    model11 = model11.to(device).eval()
    residual_params = bundle.residual_params()
    channels = int(residual_params.scale.numel())
    return {"model": model11,
            "zero": torch.from_numpy(cx.zero_symbols(residual_params, channels)),
            "assign_codebook": ml.SharedCodebook(bundle.assign_codebook_frequencies.numpy(),
                                                 bits=bundle.bits,
                                                 metric=bundle.assign_codebook_metric),
            "coding_codebook": ml.SharedCodebook(bundle.coding_codebook_frequencies.numpy(),
                                                 bits=bundle.bits,
                                                 metric=bundle.assign_codebook_metric),
            "identity": bytes.fromhex(bundle.residual_entropy_model_id)}


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--bundle-dir", type=Path, default=Path("outputs/codec_bundles"))
    parser.add_argument("--m22-davis", type=Path,
                        default=Path("outputs/m22_residual_freeze_lift/m22_davis.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/codec_bundles/verification.json"))
    parser.add_argument("--bundles", nargs="*", default=None, help="bundle names; default all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    mc = _load_script("m10h_motion_compensation")
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    cx = _load_script("m11_causal_context")
    m13 = _load_script("m13_recalibration")
    m21 = _load_script("m21_refinement")

    listing = json.loads((args.bundle_dir / "bundles.json").read_text(encoding="utf-8"))["bundles"]
    names = args.bundles or sorted(listing)
    recorded = {point["bits"]: point for point in
                json.loads(args.m22_davis.read_text(encoding="utf-8"))["rate_points"]}
    sequences = discover_sequences(args.manifest, split="test")
    frame_cache = {s.sequence_id: s.load_frames() for s in sequences}
    scratch = Path(tempfile.mkdtemp(prefix="verify_nvct_"))

    report: dict[str, Any] = {"sequences": [s.sequence_id for s in sequences],
                              "frames": sum(s.frame_count for s in sequences), "bundles": {}}
    all_ok = True
    for name in names:
        entry = listing[name]
        bundle = CodecBundle.load(args.bundle_dir / entry["file"])
        codec = VideoCodec(bundle, device=device)
        spec = research_spec(bundle, device=device, ma=ma, ml=ml, cx=cx)
        rows, started = [], time.perf_counter()
        for sequence in sequences:
            frames = frame_cache[sequence.sequence_id]
            encoded, reconstructions = codec.encode(frames, return_reconstructions=True)
            path = scratch / f"{name}_{sequence.sequence_id}.nvct"
            research = m21.encode_sequence_refined(
                mc, m13, codec.autoencoder, frames, spec, path, m21.IDENTITY,
                intra_params=bundle.intra_params(), intra_entropy_model=codec.intra_entropy_model,
                residual_params=bundle.residual_params(),
                motion_entropy_model=codec.motion_entropy_model, bits=bundle.bits,
                gop_size=bundle.gop_size, block_size=bundle.block_size,
                search_range=bundle.search_range)
            research_bytes = path.read_bytes()
            path.unlink()
            decoded = codec.decode(encoded.data)
            row = {"sequence": sequence.sequence_id, "bytes": encoded.total_bytes,
                   "bytes_identical_to_research": encoded.data == research_bytes,
                   "reconstruction_identical_to_research":
                       torch.equal(reconstructions, research["reconstructions"]),
                   "decode_bit_exact": torch.equal(decoded, reconstructions)}
            rows.append(row)
            print(f"  {name:20s} {sequence.sequence_id:16s} {row['bytes']:>9,} B  "
                  f"same-bytes={row['bytes_identical_to_research']}  "
                  f"same-recon={row['reconstruction_identical_to_research']}  "
                  f"decode-exact={row['decode_bit_exact']}", flush=True)

        total = sum(r["bytes"] for r in rows)
        expected = recorded[bundle.bits][RECORDED_KEYS[entry["arm"]]]["total_container_bytes"]
        ok = (all(r["bytes_identical_to_research"] and r["reconstruction_identical_to_research"]
                  and r["decode_bit_exact"] for r in rows) and total == expected)
        all_ok &= ok
        report["bundles"][name] = {"total_bytes": total, "m22_recorded_total_bytes": expected,
                                   "matches_recorded_total": total == expected, "verified": ok,
                                   "bundle_sha256": entry["sha256"],
                                   "seconds": time.perf_counter() - started, "sequences": rows}
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  => {name}: {total:,} bytes (M22 recorded {expected:,})  VERIFIED={ok}", flush=True)

    report["all_verified"] = all_ok
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nall bundles verified: {all_ok}\nReport: {args.output}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
