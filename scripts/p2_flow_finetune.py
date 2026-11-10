"""P2 gate 1 fairness check: does dense flow help once the P-coder has trained on it?

`outputs/p2_motion/README.md`: free dense flow saved only 4.1% of P-frame bits and made the
residual BIGGER. But the P-coder had only ever seen block-shifted references. This check
gives both warps the same chance:

    finetune_block   the gate 3 P-coder, fine-tuned N steps on BLOCK16-warped references
    finetune_dense   the same coder, fine-tuned the same N steps, same data, same seed,
                     on DENSE-FLOW-warped references (Farneback, reconstruction -> x_t,
                     bilinear; exactly the oracle's dense_oracle warp)

Both start from the gate 2 stage 2 models, so "more training" cancels out. The oracle then
scores block16 with finetune_block against dense_oracle with finetune_dense, each coder on
the warp it was tuned for. The bar is unchanged: dense flow (free) must save >= 25%.

Data: DAVIS TRAIN pairs (x_t, warp of the intra unit-step reconstruction of x_{t-1}), the
gate 1 pilot's cache. Its block warps are reused; the dense warps are computed here in the
same pair order, checked against the cached targets.

Usage:
    p2_flow_finetune.py cache --lambda 3e-4
    p2_flow_finetune.py train --lambda 3e-4 --kind dense [--steps 3000]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import p1_pframe as p1  # noqa: E402

OUTPUT_DIR = Path("outputs/p2_motion")
INIT_DIR = p1.OUTPUT_DIR / "gate2_local" / "stage2"


def dense_cache_path(lam: str) -> Path:
    return p1.CACHE_DIR / f"train_dense_{lam}.pt"


@torch.no_grad()
def command_cache(args) -> None:
    from nvc.evaluation.sequences import discover_sequences
    from nvc.utils.config import load_default_config
    from nvc.utils.device import get_device
    device = get_device()
    mc = p1._load_script("m10h_motion_compensation")
    intra = p1._intra_model(args.lam, device)
    cached = torch.load(p1.CACHE_DIR / f"train_{args.lam}.pt")["targets"]
    manifest = load_default_config().processed_data_dir / "manifest.json"
    warps, index = [], 0
    with mc.deterministic_kernels():
        for sequence in discover_sequences(manifest, split="train"):
            frames = sequence.load_frames().to(device)
            reconstructions = torch.cat([
                intra.decode(torch.round(intra.encode(frames[i:i + 16]))).clamp(0, 1)
                for i in range(0, len(frames), 16)])
            for t in range(1, len(frames)):
                frame = frames[t:t + 1]
                target = (frame * 255).round().to(torch.uint8).cpu()
                if not torch.equal(target, cached[index:index + 1]):
                    raise RuntimeError(f"pair {index} does not match the gate 1 cache")
                reference = reconstructions[t - 1:t]
                flow = mc.estimate_dense_flow_oracle(reference, frame)
                warped = mc.warp_dense(reference, flow).clamp(0, 1)
                warps.append((warped * 255).round().to(torch.uint8).cpu())
                index += 1
            print(f"    {sequence.sequence_id}: {len(frames) - 1} pairs", flush=True)
    if index != len(cached):
        raise RuntimeError(f"{index} pairs built, the gate 1 cache has {len(cached)}")
    torch.save({"warps": torch.cat(warps)}, dense_cache_path(args.lam))
    print(f"[{args.lam}] {index} dense-warped pairs -> {dense_cache_path(args.lam)}")


def command_train(args) -> None:
    from nvc.utils.device import get_device
    device = get_device()
    torch.manual_seed(args.seed)
    targets = torch.load(p1.CACHE_DIR / f"train_{args.lam}.pt")["targets"]
    warps = (torch.load(p1.CACHE_DIR / f"train_{args.lam}.pt")["warps"] if args.kind == "block"
             else torch.load(dense_cache_path(args.lam))["warps"])
    saved = torch.load(INIT_DIR / f"lambda{args.lam}" / "model.pt", map_location="cpu",
                       weights_only=False)
    model = p1.ConditionalPFrameCoder(**saved["config"])
    model.load_state_dict(saved["model_state_dict"])
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lam, pixels = float(args.lam), targets.shape[-2] * targets.shape[-1]
    out = OUTPUT_DIR / f"finetune_{args.kind}" / f"lambda{args.lam}"
    out.mkdir(parents=True, exist_ok=True)
    history, running, started = [], {"mse": 0.0, "bpp": 0.0}, time.time()
    for step in range(1, args.steps + 1):
        index = torch.randint(0, len(targets), (args.batch,))
        flip = bool(torch.rand(()) < 0.5)
        frame = targets[index].to(device).float() / 255
        warped = warps[index].to(device).float() / 255
        if flip:
            frame, warped = frame.flip(-1), warped.flip(-1)
        result = model(frame, warped)
        mse = F.mse_loss(result["reconstruction"], frame)
        bpp = result["bits"].mean() / pixels
        loss = mse + lam * bpp
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running["mse"] += mse.item()
        running["bpp"] += bpp.item()
        if step % args.log_every == 0:
            entry = {k: v / args.log_every for k, v in running.items()}
            entry.update(step=step, psnr=10 * math.log10(1 / entry["mse"]),
                         minutes=(time.time() - started) / 60)
            history.append(entry)
            print(f"  [{args.kind} {args.lam}] step {step:5d}  {entry['psnr']:.2f} dB  "
                  f"{entry['bpp']:.4f} bpp  {entry['minutes']:.1f} min", flush=True)
            running = dict.fromkeys(running, 0.0)
    torch.save({"model_state_dict": model.state_dict(), "config": model.config_dict(),
                "lambda": args.lam, "init": str(INIT_DIR), "kind": args.kind,
                "steps": args.steps, "lr": args.lr, "batch": args.batch, "seed": args.seed},
               out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--lambda", dest="lam", required=True, choices=p1.LAMBDAS)
    train = sub.add_parser("train")
    train.add_argument("--lambda", dest="lam", required=True, choices=p1.LAMBDAS)
    train.add_argument("--kind", required=True, choices=("block", "dense"))
    train.add_argument("--steps", type=int, default=3000)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--lr", type=float, default=5e-5)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--log-every", type=int, default=250)
    args = parser.parse_args(argv)
    {"cache": command_cache, "train": command_train}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
