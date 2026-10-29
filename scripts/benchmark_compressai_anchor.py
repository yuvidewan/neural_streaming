"""A learned intra anchor: CompressAI's pretrained bmshj2018 models on DAVIS TEST.

Why this exists: every BD-rate in the paper is measured against libx264/libx265.
A reviewer also needs a learned reference point, and the two published models
closest to this codec's intra path are Balle et al.'s factorized prior and scale
hyperprior (`bmshj2018-factorized`, `bmshj2018-hyperprior`, MSE-trained,
qualities 1-8). They run directly on the 256x256 frames.

Measured exactly as `benchmark_parity.py` measures every other arm:

  * the same 719 DAVIS TEST frames, loaded by the same `discover_sequences`;
  * real bytes: the length of every string `compress()` emits (CompressAI's own
    rANS coder), with frames decoded by `decompress()` from those strings only;
  * reconstructions rounded to uint8, scored by `benchmark_parity.score_frames`;
  * quality aggregated by `benchmark_parity.aggregate_quality`, rate pooled;
  * BD-rate by the same piecewise-linear `_bd_rate_linear`.

Nothing is tuned. Every quality level is coded and reported.

CompressAI has no Windows wheel. It was built from source (MSVC) into a
directory passed with --compressai-path; that is the only extra dependency.

Run:
  ./.venv/Scripts/python.exe scripts/benchmark_compressai_anchor.py \
      --compressai-path <dir with compressai and its deps>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

QUALITIES = (1, 2, 3, 4, 5, 6, 7, 8)
MODELS = ("bmshj2018-factorized", "bmshj2018-hyperprior")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compressai-path", type=Path, nargs="+", default=[])
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.json"))
    parser.add_argument("--intra-report", type=Path,
                        default=Path("outputs/benchmarks/parity_intra/parity_intra.json"))
    parser.add_argument("--qat-gate", type=Path,
                        default=Path("outputs/stage1_vimeo/colab_run4_64ch_qat/gate_qat_epoch23"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/benchmarks/compressai_anchor/compressai_anchor.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    for path in reversed(args.compressai_path):
        sys.path.insert(0, str(path))
    from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior
    builders = {"bmshj2018-factorized": bmshj2018_factorized,
                "bmshj2018-hyperprior": bmshj2018_hyperprior}

    from nvc.evaluation.sequences import discover_sequences

    bp = _load_script("benchmark_parity")
    device = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    sequences = discover_sequences(args.manifest, split="test")
    frame_cache = {s.sequence_id: s.load_frames() for s in sequences}
    print(f"{len(sequences)} sequences, {sum(s.frame_count for s in sequences)} frames", flush=True)

    points = []
    for model_name in MODELS:
        for quality in QUALITIES:
            net = builders[model_name](quality=quality, metric="mse", pretrained=True)
            net = net.eval().to(device)
            rows = []
            started = time.perf_counter()
            for sequence in sequences:
                frames = frame_cache[sequence.sequence_id]
                decoded, total_bytes = [], 0
                with torch.no_grad():
                    for index in range(frames.shape[0]):
                        x = frames[index:index + 1].to(device)
                        out = net.compress(x)
                        total_bytes += sum(len(s[0]) for s in out["strings"])
                        x_hat = net.decompress(out["strings"], out["shape"])["x_hat"]
                        decoded.append(bp.to_uint8_frames(x_hat).cpu())
                scores = bp.score_frames(torch.cat(decoded), frames, device=device)
                rows.append({"sequence": sequence.sequence_id, "bytes": total_bytes,
                             "pixels": sequence.total_pixels, **scores})
            label = f"{model_name}/q{quality}"
            point = bp.summarize_point(label, model_name, f"q{quality}", rows,
                                       extra={"quality_level": quality,
                                              "seconds": time.perf_counter() - started})
            points.append(point)
            q = point["quality"][bp.PRIMARY_CONVENTION]
            print(f"  {label:<26} bpp {point['bpp']:.4f}  PSNR {q['psnr']:.3f}  "
                  f"MS-SSIM {q['msssim']:.4f}", flush=True)

    intra = json.loads(args.intra_report.read_text(encoding="utf-8"))
    gate = json.loads(args.qat_gate.read_text(encoding="utf-8"))
    reference = {
        "h264_intra": [p for p in intra["points"].values() if p["arm"] == "h264_intra"],
        "nvc_593k_intra": [p for p in intra["points"].values() if p["arm"] == "nvc_deployed"],
        "nvc_64ch_qat_intra": [p for p in gate["points"].values()
                               if str(p["arm"]).startswith("nvc")],
    }
    by_arm = {name: [p for p in points if p["arm"] == name] for name in MODELS}
    by_arm.update(reference)

    comparisons = {}
    pairs = [(anchor, test) for anchor in ("h264_intra", *MODELS)
             for test in (*MODELS, "nvc_593k_intra", "nvc_64ch_qat_intra")
             if anchor != test]
    for anchor, test in pairs:
        entry = {}
        for convention in bp.CONVENTIONS:
            for metric in ("psnr", "msssim"):
                entry[f"{convention}/{metric}"] = bp.bd_rate(
                    bp.curve(by_arm[anchor], convention, metric),
                    bp.curve(by_arm[test], convention, metric))
        comparisons[f"{test}_vs_{anchor}"] = entry

    print("\nBD-rate, primary convention (positive = test needs more bits):")
    for key, entry in comparisons.items():
        psnr = entry[f"{bp.PRIMARY_CONVENTION}/psnr"]
        msssim = entry[f"{bp.PRIMARY_CONVENTION}/msssim"]
        fmt = lambda v: "n/a" if v is None else f"{v:+.1f}%"
        print(f"  {key:<52} PSNR {fmt(psnr):>9}  MS-SSIM {fmt(msssim):>9}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "stage": "learned intra anchor (CompressAI bmshj2018, pretrained, MSE)",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torch_version": torch.__version__,
        "primary_convention": bp.PRIMARY_CONVENTION,
        "rate_definition": "sum of compress() string lengths * 8 / total pixels, pooled",
        "points": points,
        "reference_points": {k: [{"label": p["label"], "bpp": p["bpp"],
                                  "quality": p["quality"]} for p in v]
                             for k, v in reference.items()},
        "comparisons": comparisons,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
