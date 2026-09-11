"""M12 Phase A4 - does resumability remove M11's repeated prefix-decoding cost?

M11's `decode_frame` (scripts/m11_ar_entropy.py) decodes channel group g by
calling the existing STATELESS `decode_symbols(payload, count, ...)` with
`count` = every symbol from position 0 through the end of group g - so group
g re-decodes every earlier group's symbols from byte 0 of the payload. At
G=16 (the deployed M11-G16 operating point) that is 4 redundant re-decodes;
at G=1 (full channel autoregression) it is 64, which is why G=1 measured
18-27x M10L's decode latency in M11 and was rejected as impractical - and
why per-spatial-position (G -> 1/256th of a channel) context was never
attempted at all.

This script decodes the SAME payloads two ways - legacy prefix redecoding
(`m11_ar_entropy.decode_frame`, untouched) and the new
`ResumableDecoder`-based path defined here - and:

  1. proves the two produce bit-identical symbols (the zero-loss requirement:
     resumability must not change what gets decoded), then
  2. times both, split into network / tables / coder stages, at G=16 (today's
     deployed point) and at G=1 (full channel autoregression - impractical
     under the legacy decoder, and the group size Phase B's finest spatial
     contexts would need).

Run:
  ./.venv/Scripts/python.exe scripts/m12_resumable_decode.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.range_coder import ResumableDecoder
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m12_resumable_decoder")
DEFAULT_CHECKPOINT = Path("outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt")
DEFAULT_M11_DIR = Path("outputs/m11_autoregressive_entropy")
RATE_POINTS = (5, 4, 3)
BENCHMARK_FRAMES = 30


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the resumable decode path -------------------------------------------------


@torch.no_grad()
def decode_frame_resumable(ma, model, payload: bytes, reference: torch.Tensor,
                           zero: torch.Tensor, *, bits: int, shape: tuple[int, int, int],
                           codebook=None, timings: dict | None = None) -> np.ndarray:
    """Same contract as `ma.decode_frame`, but each group is decoded by ONE
    `ResumableDecoder.decode_group` call carrying only that group's own
    symbols and tables - no growing prefix, no re-decoding earlier groups.

    Everything except the coder step (network forward pass, table/codebook
    construction) is identical to `ma.decode_frame`'s loop body - only HOW
    the coder is driven changes, so any latency difference between this
    function and `ma.decode_frame` is attributable to resumability alone.
    """
    mk = _load_script("m10k_learned_entropy")
    channels, height, width = shape
    plane = height * width
    group = model.group_size
    device = reference.device
    zero_d = zero.to(device)
    decoded = zero_d.view(channels, 1, 1).expand(channels, height, width).clone()
    stage = {"network": 0.0, "tables": 0.0, "coder": 0.0}
    symbols = np.empty(channels * plane, dtype=np.int64)

    decoder = ResumableDecoder(payload)
    for start in range(0, channels, group):
        stop = start + group
        began = time.perf_counter()
        log_probabilities = model.log_probabilities(reference, model.planes(decoded[None], zero_d))
        rows = ma._rows(log_probabilities[:, start:stop])
        if device.type == "cuda":
            torch.cuda.synchronize()
        stage["network"] += time.perf_counter() - began

        began = time.perf_counter()
        block, assigned = ma._tables_for(rows, codebook, mk)
        if codebook is None:
            group_cumulative = block
            group_table_index = np.arange(group * plane, dtype=np.int64)
        else:
            group_cumulative = codebook.cumulative
            group_table_index = assigned
        stage["tables"] += time.perf_counter() - began

        began = time.perf_counter()
        group_symbols = decoder.decode_group(group_cumulative, group_table_index)
        stage["coder"] += time.perf_counter() - began

        symbols[start * plane:stop * plane] = group_symbols
        decoded[start:stop] = torch.from_numpy(
            group_symbols.reshape(group, height, width)).to(device)
    decoder.close()

    if timings is not None:
        for key, value in stage.items():
            timings[key] = timings.get(key, 0.0) + value
    return symbols


# --- correctness: legacy vs resumable, on real M11-G16 streams -----------------


def verify_symbol_equality(ma, model, codebook, references, symbols_true, zero, *,
                           bits: int, shape: tuple[int, int, int], device) -> dict[str, Any]:
    mismatches = 0
    for reference, true_symbols in zip(references, symbols_true):
        tensor = torch.from_numpy(np.asarray(reference, dtype=np.float32))[None].to(device)
        payload, _ = ma.encode_frame(model, tensor, true_symbols, zero,
                                     bits=bits, codebook=codebook)
        legacy = ma.decode_frame(model, payload, tensor, zero, bits=bits, shape=shape,
                                 codebook=codebook)
        resumable = decode_frame_resumable(ma, model, payload, tensor, zero, bits=bits,
                                           shape=shape, codebook=codebook)
        if not (np.array_equal(legacy, true_symbols.reshape(-1))
                and np.array_equal(resumable, true_symbols.reshape(-1))
                and np.array_equal(legacy, resumable)):
            mismatches += 1
    return {"frames_checked": len(references), "mismatches": mismatches,
           "all_identical": mismatches == 0}


# --- latency benchmark: legacy prefix redecode vs resumable, at two group sizes -


def benchmark_group_size(ma, base_model_config, model_state, codebook, references,
                         symbols_true, zero, *, bits: int, shape: tuple[int, int, int],
                         group_size: int, device, frames: int) -> dict[str, Any]:
    config = dict(base_model_config, group_size=group_size)
    model = ma.build_model(config).to(device)
    model.load_state_dict(model_state)
    model.eval()

    payloads = []
    for reference, true_symbols in zip(references[:frames], symbols_true[:frames]):
        tensor = torch.from_numpy(np.asarray(reference, dtype=np.float32))[None].to(device)
        payload, _ = ma.encode_frame(model, tensor, true_symbols, zero,
                                     bits=bits, codebook=codebook)
        payloads.append(payload)

    legacy_timings: dict[str, float] = {}
    resumable_timings: dict[str, float] = {}
    legacy_symbols_all, resumable_symbols_all = [], []
    for reference, payload in zip(references[:frames], payloads):
        tensor = torch.from_numpy(np.asarray(reference, dtype=np.float32))[None].to(device)
        legacy_symbols_all.append(ma.decode_frame(model, payload, tensor, zero, bits=bits,
                                                   shape=shape, codebook=codebook,
                                                   timings=legacy_timings))
    for reference, payload in zip(references[:frames], payloads):
        tensor = torch.from_numpy(np.asarray(reference, dtype=np.float32))[None].to(device)
        resumable_symbols_all.append(decode_frame_resumable(
            ma, model, payload, tensor, zero, bits=bits, shape=shape, codebook=codebook,
            timings=resumable_timings))

    identical = all(np.array_equal(a, b) for a, b in zip(legacy_symbols_all, resumable_symbols_all))

    def _ms(timings):
        per_frame = {k: 1000.0 * v / frames for k, v in timings.items()}
        per_frame["total"] = sum(per_frame.values())
        return per_frame

    legacy_ms, resumable_ms = _ms(legacy_timings), _ms(resumable_timings)
    return {
        "group_size": group_size, "frames": frames, "symbols_identical": identical,
        "legacy_decode_ms": legacy_ms, "resumable_decode_ms": resumable_ms,
        "coder_speedup": (legacy_ms["coder"] / resumable_ms["coder"]
                          if resumable_ms["coder"] > 0 else float("inf")),
        "total_speedup": legacy_ms["total"] / resumable_ms["total"],
    }


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M12 Phase A4: resumable-decoder benchmark vs legacy prefix redecoding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=DEFAULT_M11_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    parser.add_argument("--benchmark-frames", type=int, default=BENCHMARK_FRAMES)
    parser.add_argument("--correctness-frames", type=int, default=8)
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[16, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    md = _load_script("m11_data")
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    cx = _load_script("m11_causal_context")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    print("=" * 100)
    print("M12 PHASE A4 - RESUMABLE DECODER vs LEGACY PREFIX REDECODING")
    print("=" * 100)

    report: dict[str, Any] = {"phase": "M12 Phase A4", "rate_points": []}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for bits in args.rate_points:
        print(f"\n---- {bits}-bit ----", flush=True)
        # `load_or_collect` needs a motion-compensation `model` only on a cache
        # MISS; the M11 data cache already exists for this checkpoint/bit depth,
        # so this hits cache and the model argument is not actually exercised.
        from nvc.training.checkpoint import load_model_from_checkpoint
        base_model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
        base_model.eval()
        data = md.load_or_collect(
            base_model, checkpoint=args.checkpoint, manifest=args.manifest, bits=bits,
            device=device, cache_dir=cache_dir, log=lambda m: print(m, flush=True))
        channels, height, width = data["val_symbols"].shape[1:]
        shape = (channels, height, width)
        zero = torch.from_numpy(cx.zero_symbols(data["calibration"]["residual_params"], channels))

        model, checkpoint = ma.load_model(args.m11_dir / f"m11_G16_entropy_{bits}bit.pt",
                                          device=device)
        codebook = ml.SharedCodebook.from_dict(json.loads(
            (args.m11_dir / f"m11_G16_codebook_{bits}bit_K512.json").read_text(encoding="utf-8")))

        val_symbols = data["val_symbols"][:max(args.benchmark_frames, args.correctness_frames)]
        val_references = data["val_references"][:max(args.benchmark_frames, args.correctness_frames)]

        correctness = verify_symbol_equality(
            ma, model, codebook, val_references[:args.correctness_frames],
            val_symbols[:args.correctness_frames], zero, bits=bits, shape=shape, device=device)
        print(f"  correctness (G=16 deployed model): {correctness}", flush=True)
        if not correctness["all_identical"]:
            print("[ERROR] legacy/resumable symbol mismatch - STOPPING", file=sys.stderr)
            return 1

        rate_point: dict[str, Any] = {"bits": bits, "correctness_g16": correctness,
                                      "group_size_benchmarks": []}
        for group_size in args.group_sizes:
            result = benchmark_group_size(
                ma, checkpoint["model_config"], checkpoint["model_state_dict"], codebook,
                val_references, val_symbols, zero, bits=bits, shape=shape,
                group_size=group_size, device=device, frames=args.benchmark_frames)
            print(f"  G={group_size:>3}  legacy {result['legacy_decode_ms']['total']:7.3f} ms  "
                 f"resumable {result['resumable_decode_ms']['total']:7.3f} ms  "
                 f"coder {result['legacy_decode_ms']['coder']:6.3f}->"
                 f"{result['resumable_decode_ms']['coder']:6.3f} ms "
                 f"({result['coder_speedup']:.1f}x)  total speedup {result['total_speedup']:.2f}x  "
                 f"identical={result['symbols_identical']}", flush=True)
            rate_point["group_size_benchmarks"].append(result)
        report["rate_points"].append(rate_point)

    path = args.output_dir / "m12_decoder_benchmark.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
