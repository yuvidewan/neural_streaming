"""Large-scale check that the native motion search equals the PyTorch reference on REAL data.

    python docs/c_rewrite_prototypes/verify_native_motion.py [--pairs-per-sequence 8]

What it feeds the search is what the codec feeds it: the reference is a frame that has been
through a REAL trained autoencoder (encode -> decode), the current frame is the next original
frame, both resized to 256x256 like the benchmark. Every DAVIS sequence under
data/external/DAVIS is used. For each pair the native result must equal
`estimate_block_motion_torch` exactly; both native code paths (scalar and AVX2) are checked.

This is strong evidence, not proof - the proof-grade gate is still
`scripts/verify_promoted_codec.py` (byte-identical streams on all of DAVIS TEST, with the
real codec bundles) run once with NVC_MOTION_BACKEND=torch and once with =native.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import nvc.video.motion as motion  # noqa: E402
from nvc.training.checkpoint import load_model_from_checkpoint  # noqa: E402
from nvc.video import _native  # noqa: E402

SIZE, BLOCK, RANGE = 256, 16, 16


def load(path: Path) -> torch.Tensor:
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)[None].contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-per-sequence", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/checkpoints/vimeo_qat_noise_best.pt")
    args = parser.parse_args()

    lib = _native.load()
    if lib is None:
        raise SystemExit(f"native library unavailable: {_native.last_error()}")
    modes = [("scalar", 1)] + ([("avx2", 2)] if lib.nvc_bs_has_avx2() else [])
    model, _ = load_model_from_checkpoint(args.checkpoint, device="cpu")

    root = ROOT / "data/external/DAVIS/JPEGImages/480p"
    sequences = sorted(p for p in root.iterdir() if p.is_dir())
    pairs = mismatches = 0
    t_reference, t_native = [], []
    start = time.perf_counter()
    for sequence in sequences:
        frames = sorted(sequence.glob("*.jpg"))
        if len(frames) < 2:
            continue
        step = max(1, (len(frames) - 1) // args.pairs_per_sequence)
        for index in range(0, len(frames) - 1, step)[: args.pairs_per_sequence]:
            with torch.no_grad():
                reference = model.decode(model.encode(load(frames[index])))      # what the codec holds
            current = load(frames[index + 1])
            t = time.perf_counter()
            expected = motion.estimate_block_motion_torch(reference, current, block_size=BLOCK, search_range=RANGE)
            t_reference.append(time.perf_counter() - t)
            for name, mode in modes:
                t = time.perf_counter()
                got = motion._run_native(lib, reference, current, block_size=BLOCK, search_range=RANGE,
                                         threads=torch.get_num_threads(), early_exit=True, mode=mode)
                if name == "avx2" or len(modes) == 1:
                    t_native.append(time.perf_counter() - t)
                if not torch.equal(expected, got):
                    mismatches += 1
                    print(f"MISMATCH {sequence.name}[{index}] {name}: "
                          f"{int((expected != got).any(dim=0).sum())} of {expected.shape[1] * expected.shape[2]} blocks")
            pairs += 1
        print(f"  {sequence.name:24s} done   pairs so far {pairs}   mismatches {mismatches}", flush=True)

    print(f"\n{pairs} real frame pairs x {len(modes)} native paths ({', '.join(n for n, _ in modes)}): "
          f"{mismatches} mismatches   [{time.perf_counter() - start:.0f}s]")
    print(f"median per pair: torch reference {statistics.median(t_reference) * 1000:.0f} ms, "
          f"native ({torch.get_num_threads()} threads) {statistics.median(t_native) * 1000:.1f} ms  "
          f"-> {statistics.median(t_reference) / statistics.median(t_native):.0f}x")
    print("PASS" if mismatches == 0 else "FAIL")


if __name__ == "__main__":
    main()
