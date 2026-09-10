"""M11 phase 0 - cross-process reproducibility of the temporal codec.

WHY THIS EXISTS
---------------
M10L found that `model.decode` is not bit-reproducible on GPU without
`deterministic_kernels()`, and that `calibrate_grids` called it outside that
guard. The quantization grid therefore differed slightly between PROCESSES, and
the difference propagated: grid -> residual symbols -> reconstruction -> motion
estimation -> motion bytes (M10L's run: 181,160 motion bytes against M10K's
published 181,207). Within one run every arm shared one calibration, so no
comparison was wrong - but no two runs could be compared byte for byte.

The guard was added to `calibrate_grids` in commit 61dd8434, on a CPU-only
machine where the drift cannot occur, so that commit could only verify that the
guard is ACTIVE. This script verifies the property that actually matters: two
independent processes, given identical inputs, produce identical bytes.

WHAT IS FINGERPRINTED
---------------------
Each child process computes, from scratch:

  calibration      intra / residual quantization grids and all three entropy
                   model identities
  closed loop      every P-frame's reference latent z_ref and residual symbols
  stream           the complete M10H .nvct v2 file and its byte counts
  reconstruction   the decoded frames
  entropy models   (when the M10K/M10L artefacts are present) M10K frequencies
                   and M10L table indices on the first P-frame

and writes SHA-256 digests plus the raw grids. The driver runs two children
and compares every field. M11 extends the same fingerprint with its own
probabilities, so the permanent regression covers the new model too.

Run (real checkpoint, DAVIS validation clip):
  ./.venv/Scripts/python.exe scripts/m11_reproducibility.py --source real

Measure the pre-fix behaviour against a copy of the old module:
  ./.venv/Scripts/python.exe scripts/m11_reproducibility.py --source real \\
      --m10h-module path/to/m10h_prefix.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M10K_DIR = Path("outputs/m10k_learned_entropy")
DEFAULT_M10L_DIR = Path("outputs/m10l_shared_codebook/codebooks")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value) -> str:
    """SHA-256 of an array/tensor/bytes, over its exact bytes and shape."""
    digest = hashlib.sha256()
    if isinstance(value, (bytes, bytearray)):
        digest.update(bytes(value))
    else:
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) \
            else np.asarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _digest_list(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(_digest(value).encode())
    return digest.hexdigest()


# --- synthetic source: runs anywhere, no data or checkpoint needed ------------


class SyntheticSequence:
    """Stands in for a BenchmarkSequence: `load_frames()` is all the codec uses."""

    def __init__(self, frames: torch.Tensor, sequence_id: str) -> None:
        self._frames = frames
        self.sequence_id = sequence_id
        self.frame_count = frames.shape[0]

    def load_frames(self) -> torch.Tensor:
        return self._frames.clone()


def synthetic_frames(count: int, size: int, *, seed: int, step: int = 3) -> torch.Tensor:
    """A textured frame translated a few pixels per step - real motion for the
    block matcher to find, so the motion path is exercised, not bypassed."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 3, size, size, generator=generator)
    base = torch.nn.functional.avg_pool2d(base, 3, stride=1, padding=1)
    return torch.cat([torch.roll(base, shifts=(i * step, i * (step - 1)), dims=(2, 3))
                      for i in range(count)], dim=0).clamp(0.0, 1.0)


def synthetic_setup(device, *, seed: int, size: int, frames: int):
    from nvc.models.autoencoder import BaselineAutoencoder

    torch.manual_seed(seed)
    model = BaselineAutoencoder().to(device).eval()  # the real architecture, random weights
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train = [SyntheticSequence(synthetic_frames(frames, size, seed=seed + i), f"train{i}")
             for i in range(2)]
    clip = SyntheticSequence(synthetic_frames(frames, size, seed=seed + 100), "clip")
    return model, train, clip


# --- the fingerprint ------------------------------------------------------------


@torch.no_grad()
def decode_self_consistency(mc, model, latent, *, repeats: int) -> dict[str, float]:
    """Max abs difference between repeated decodes of ONE latent, with and
    without the guard. This is the root cause, measured in isolation."""
    unguarded = [model.decode(latent) for _ in range(repeats)]
    with mc.deterministic_kernels():
        guarded = [model.decode(latent) for _ in range(repeats)]
    spread = lambda outs: max(float((outs[0] - o).abs().max()) for o in outs[1:])
    return {"unguarded_max_abs_diff": spread(unguarded),
            "guarded_max_abs_diff": spread(guarded),
            "guarded_vs_unguarded_max_abs_diff": float((guarded[0] - unguarded[0]).abs().max())}


@torch.no_grad()
def fingerprint(mc, model, train_sequences, clip, *, bits: int, calibration_frames: int,
                gop_size: int, block_size: int, search_range: int, device,
                stream_path: Path, entropy_extras=None) -> dict[str, Any]:
    ce = _load_module(SCRIPT_DIR / "m10j_conditional_entropy.py", "m10j_conditional_entropy")
    record: dict[str, Any] = {"torch": torch.__version__, "device": str(device),
                              "cuda": torch.cuda.is_available()}

    calibration = mc.calibrate_grids(
        model, train_sequences, bits=bits, mode="per_channel", gop_size=gop_size,
        block_size=block_size, search_range=search_range, reference_mode="mc",
        max_frames=calibration_frames)
    record["calibration"] = {
        "intra_scale": _digest(calibration["intra_params"].scale),
        "intra_zero_point": _digest(calibration["intra_params"].zero_point),
        "residual_scale": _digest(calibration["residual_params"].scale),
        "residual_zero_point": _digest(calibration["residual_params"].zero_point),
        "intra_entropy_model_id": calibration["intra_entropy_model"].model_id().hex(),
        "residual_entropy_model_id": calibration["residual_entropy_model"].model_id().hex(),
        "motion_entropy_model_id": calibration["motion_entropy_model"].model_id().hex(),
    }
    record["raw"] = {
        "residual_scale": calibration["residual_params"].scale.flatten().cpu().tolist(),
        "residual_zero_point": calibration["residual_params"].zero_point.flatten().cpu().tolist(),
    }

    symbols, references = ce.collect_training_symbols(
        mc, model, [clip], calibration, gop_size=gop_size, block_size=block_size,
        search_range=search_range, device=device, max_frames=clip.frame_count)
    record["closed_loop"] = {"p_frames": len(symbols),
                             "z_ref": _digest_list(references),
                             "residual_symbols": _digest_list(symbols)}

    encoded = mc.encode_sequence(
        model, clip.load_frames(), stream_path,
        intra_params=calibration["intra_params"],
        intra_entropy_model=calibration["intra_entropy_model"],
        residual_params=calibration["residual_params"],
        residual_entropy_model=calibration["residual_entropy_model"],
        motion_entropy_model=calibration["motion_entropy_model"],
        mode="mc", gop_size=gop_size, block_size=block_size, search_range=search_range)
    record["stream"] = {
        "sha256": _digest(stream_path.read_bytes()),
        "container_bytes": int(encoded["container_bytes"]),
        "residual_bytes": int(encoded["residual_bytes"]),
        "motion_bytes": int(encoded["motion_bytes"]),
    }
    record["reconstruction"] = _digest(encoded["encoder_reconstructions"])

    if entropy_extras is not None and references:
        record["entropy"] = entropy_extras(references[0], symbols[0], bits=bits,
                                           calibration=calibration)
    return record


def m11_extras(device, *, seed: int, group_size: int = 4):
    """M11 probabilities, frequency tables and payload on one P-frame.

    Uses a seeded model with LIVE context weights (not the zero-initialised warm
    start, which would make the context path trivially reproducible), so the
    cross-process comparison exercises the planes, the network and the
    group-sequential decoder.
    """
    mk = _load_module(SCRIPT_DIR / "m10k_learned_entropy.py", "m10k_learned_entropy")
    ma = _load_module(SCRIPT_DIR / "m11_ar_entropy.py", "m11_ar_entropy")
    cx = _load_module(SCRIPT_DIR / "m11_causal_context.py", "m11_causal_context")
    mc_guard = _load_module(SCRIPT_DIR / "m10h_motion_compensation.py", "m10h_guard_m11")

    def extras(reference, symbols, *, bits, calibration):
        channels = symbols.shape[0]
        torch.manual_seed(seed)
        base = mk.build_model({"latent_channels": channels, "alphabet": 2 ** bits,
                               "hidden": 16})
        model = ma.from_m10k(base, group_size=group_size).to(device).eval()
        generator = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            model.features[0].weight[:, 1:] = torch.randn(
                model.features[0].weight[:, 1:].shape, generator=generator).to(device) * 0.5
        zero = torch.from_numpy(cx.zero_symbols(calibration["residual_params"], channels))
        tensor = torch.from_numpy(np.asarray(reference, dtype=np.float32))[None].to(device)
        with torch.no_grad(), mc_guard.deterministic_kernels():
            target = torch.from_numpy(symbols)[None].to(device)
            rows = ma._rows(model.log_probabilities(tensor, model.planes(target, zero.to(device))))
            cumulative, _ = ma._tables_for(rows, None, mk)
            payload, _ = ma.encode_frame(model, tensor, symbols, zero, bits=bits)
            decoded = ma.decode_frame(model, payload, tensor, zero, bits=bits,
                                      shape=symbols.shape)
        return {"m11_probabilities": _digest(rows), "m11_frequencies": _digest(cumulative),
                "m11_payload": _digest(payload),
                "m11_round_trip": bool(np.array_equal(decoded, symbols.reshape(-1)))}

    return extras


def m10k_m10l_extras(device, m10k_dir: Path, m10l_dir: Path):
    """M10K frequencies and M10L table indices on one P-frame, when available."""
    mk = _load_module(SCRIPT_DIR / "m10k_learned_entropy.py", "m10k_learned_entropy")
    ml = _load_module(SCRIPT_DIR / "m10l_shared_codebook.py", "m10l_shared_codebook")

    def extras(reference, symbols, *, bits, calibration):
        model_path = m10k_dir / f"learned_entropy_{bits}bit.pt"
        codebook_path = m10l_dir / f"codebook_{bits}bit_K512_code_length.json"
        if not model_path.is_file():
            return {"available": False}
        learned, _ = mk.load_entropy_model(model_path, device=device)
        tensor = torch.from_numpy(reference).float()[None].to(device)
        mc_guard = _load_module(SCRIPT_DIR / "m10h_motion_compensation.py", "m10h_guard")
        with mc_guard.deterministic_kernels():
            entropy_model, _ = mk.frame_entropy_model(learned, tensor, bits=bits)
            probabilities = ml.frame_probabilities(learned, tensor)
            out = {"available": True,
                   "m10k_probabilities": _digest(probabilities),
                   "m10k_frequencies": _digest(entropy_model.frequencies)}
            if codebook_path.is_file():
                codebook = ml.SharedCodebook.from_dict(
                    json.loads(codebook_path.read_text(encoding="utf-8")))
                out["m10l_table_index"] = _digest(codebook.assign_tensor(probabilities))
        return out

    return extras


# --- comparing two fingerprints -------------------------------------------------


def _flatten(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


# Fields that describe the run rather than its output, and the raw grids (which
# are compared numerically instead, so the size of any drift is reported).
_NOT_COMPARED = ("torch", "device", "cuda", "decode.", "raw.")


def compare(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    a, b = _flatten(first), _flatten(second)
    keys = sorted(k for k in set(a) | set(b) if not k.startswith(_NOT_COMPARED))
    mismatched = [k for k in keys if a.get(k) != b.get(k)]
    scale_a = np.asarray(first["raw"]["residual_scale"])
    scale_b = np.asarray(second["raw"]["residual_scale"])
    return {
        "fields_compared": len(keys),
        "mismatched_fields": mismatched,
        "identical": not mismatched,
        "residual_scale_max_abs_diff": float(np.abs(scale_a - scale_b).max()),
        "residual_scale_max_rel_diff": float(
            (np.abs(scale_a - scale_b) / np.maximum(np.abs(scale_a), 1e-12)).max()),
        "stream_byte_diff": {
            field: second["stream"][field] - first["stream"][field]
            for field in ("container_bytes", "residual_bytes", "motion_bytes")},
    }


# --- process plumbing ------------------------------------------------------------


def run_child(args) -> int:
    sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
    from nvc.utils.seed import seed_everything

    seed_everything(args.seed)
    device = torch.device(args.device) if args.device != "auto" else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    mc = _load_module(Path(args.m10h_module), "m10h_under_test")

    if args.source == "synthetic":
        model, train, clip = synthetic_setup(device, seed=args.seed, size=args.size,
                                             frames=args.frames)
        extras = m11_extras(device, seed=args.seed)
    else:
        from nvc.evaluation.sequences import discover_sequences
        from nvc.training.checkpoint import load_model_from_checkpoint
        from nvc.utils.config import load_default_config

        manifest = args.manifest or (load_default_config().processed_data_dir
                                     / "manifest.json")
        model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
        model.eval()
        train = discover_sequences(manifest, split="train")
        clip = discover_sequences(manifest, split="val", max_sequences=1,
                                  max_frames_per_sequence=args.frames)[0]
        extras = m10k_m10l_extras(device, args.m10k_dir, args.m10l_dir)

    with tempfile.TemporaryDirectory() as scratch:
        record = fingerprint(
            mc, model, train, clip, bits=args.bits,
            calibration_frames=args.calibration_frames, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, device=device,
            stream_path=Path(scratch) / "clip.nvct", entropy_extras=extras)

    probe_frame = clip.load_frames()[0:1].to(device)
    record["decode"] = decode_self_consistency(
        mc, model, model.encode(probe_frame), repeats=args.decode_repeats)
    Path(args.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return 0


def run_pair(args, *, extra: list[str] | None = None) -> dict[str, Any]:
    """Two INDEPENDENT processes, identical arguments; returns both and the diff."""
    records = []
    with tempfile.TemporaryDirectory() as scratch:
        for index in range(2):
            out = Path(scratch) / f"child{index}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--child",
                       "--out", str(out), *(extra or [])]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(f"child {index} failed:\n{completed.stderr[-4000:]}")
            records.append(json.loads(out.read_text(encoding="utf-8")))
    return {"first": records[0], "second": records[1],
            "comparison": compare(records[0], records[1])}


def _forward_args(args) -> list[str]:
    forwarded = ["--source", args.source, "--m10h-module", str(args.m10h_module),
                 "--bits", str(args.bits), "--calibration-frames", str(args.calibration_frames),
                 "--frames", str(args.frames), "--size", str(args.size),
                 "--gop", str(args.gop), "--block-size", str(args.block_size),
                 "--search-range", str(args.search_range), "--seed", str(args.seed),
                 "--device", args.device, "--decode-repeats", str(args.decode_repeats),
                 "--checkpoint", str(args.checkpoint), "--m10k-dir", str(args.m10k_dir),
                 "--m10l-dir", str(args.m10l_dir)]
    if args.manifest:
        forwarded += ["--manifest", str(args.manifest)]
    return forwarded


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M11 phase 0: cross-process reproducibility of the temporal codec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--m10h-module", type=Path,
                        default=SCRIPT_DIR / "m10h_motion_compensation.py",
                        help="the M10H module under test (point at an old copy to "
                             "measure pre-fix behaviour)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--m10k-dir", type=Path, default=DEFAULT_M10K_DIR)
    parser.add_argument("--m10l-dir", type=Path, default=DEFAULT_M10L_DIR)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--calibration-frames", type=int, default=60)
    parser.add_argument("--frames", type=int, default=20,
                        help="frames per synthetic sequence / in the real validation clip")
    parser.add_argument("--size", type=int, default=128, help="synthetic frame size")
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--decode-repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.child:
        return run_child(args)

    result = run_pair(args, extra=_forward_args(args))
    comparison = result["comparison"]
    decode = result["first"]["decode"]
    print("=" * 96)
    print("M11 PHASE 0 - CROSS-PROCESS REPRODUCIBILITY")
    print("=" * 96)
    print(f"  source {args.source}   device {result['first']['device']}   "
          f"module {Path(args.m10h_module).name}   {args.bits}-bit")
    print(f"\n  decode() self-consistency (one latent, {args.decode_repeats} repeats, "
          f"within one process):")
    print(f"    unguarded max |diff|             {decode['unguarded_max_abs_diff']:.3e}")
    print(f"    guarded   max |diff|             {decode['guarded_max_abs_diff']:.3e}")
    print(f"    guarded vs unguarded max |diff|  "
          f"{decode['guarded_vs_unguarded_max_abs_diff']:.3e}")
    print(f"\n  two independent processes:")
    print(f"    fields compared                  {comparison['fields_compared']}")
    print(f"    residual grid max |diff|         {comparison['residual_scale_max_abs_diff']:.3e}"
          f"  (relative {comparison['residual_scale_max_rel_diff']:.3e})")
    print(f"    stream byte diff                 {comparison['stream_byte_diff']}")
    print(f"    mismatched fields                "
          f"{comparison['mismatched_fields'] or 'none'}")
    print(f"\n  IDENTICAL ACROSS PROCESSES: {comparison['identical']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nReport: {args.out}")
    return 0 if comparison["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
