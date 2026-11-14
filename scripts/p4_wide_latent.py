"""Plan A gate 1: does a wider latent lift the P-frame coder's quality ceiling?

`docs/PLAN_wider_latent.md`. The stage 3 P-coder has a 64-channel latent at 1/16 resolution
(0.25 values per pixel). On detailed videos it cannot pass ~28.5 dB even without rounding.
Two ways to add capacity, both warm-started so step 0 reconstructs exactly like stage 3:

    A1  +128 channels at 1/16           (192 in total)
    A2  +32-channel detail latent at 1/8

The new latent comes from the encoder's stride-8 features. Its prior sees the reference
context and the decoded 1/16 latent. Its synthesis branch starts at zero.

Usage:
    p4_wide_latent.py train --lambda 3e-4 --kind a1|a2 [--steps 5000]
    p4_wide_latent.py evaluate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import p1_pframe as p1  # noqa: E402

OUTPUT_DIR = Path("outputs/p4_wide_latent")
INIT_DIR = p1.OUTPUT_DIR / "gate2_local" / "stage3"
PLAIN_DIR = Path("outputs/p3_hyperprior/finetune_plain")
EXTRA = {"a1": 128, "a2": 32}
GATE_SAVING = 10.0
GATE_CEILING_DB = 1.5
TARGET_INIT_STD = 0.7


def quantize(y, mean, scale, training: bool):
    """The base coder's rounding rule: noise for the rate, straight-through for the decoder."""
    centred = y - mean
    if training:
        rate_input = centred + torch.empty_like(centred).uniform_(-0.5, 0.5)
        decoded = mean + centred + (torch.round(centred) - centred).detach()
    else:
        rate_input = torch.round(centred)
        decoded = mean + rate_input
    return rate_input, decoded, p1.laplace_bits(rate_input, scale).flatten(1).sum(1)


class WideLatentPFrameCoder(nn.Module):
    """The stage 3 P-coder plus an extra latent (kind 'a1' or 'a2'), or nothing ('plain')."""

    def __init__(self, base: p1.ConditionalPFrameCoder, kind: str = "plain") -> None:
        super().__init__()
        self.base, self.kind = base, kind
        if kind == "plain":
            return
        k, n, m = EXTRA[kind], 192, base.latent_channels
        half = base.prior[-1].in_channels                     # prior features, stride 8
        if kind == "a1":
            self.extra_enc = nn.Conv2d(n, k, 5, stride=2, padding=2)          # -> 1/16
            self.extra_ctx = nn.Sequential(p1._down(half, half), nn.LeakyReLU(0.1))
            self.extra_syn = nn.ConvTranspose2d(k, n, 5, stride=2, padding=2,
                                                output_padding=1)              # -> 1/8
        else:
            self.extra_enc = nn.Conv2d(n, k, 3, padding=1)                     # stays 1/8
            self.extra_ctx = nn.Identity()
            self.extra_syn = nn.Conv2d(k, n, 3, padding=1)                     # 1/8
        self.extra_prior = nn.Sequential(
            nn.Conv2d(half + m, 96, 3, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(96, 2 * k, 3, padding=1))
        for layer in (self.extra_syn, self.extra_prior[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def config_dict(self) -> dict:
        return {"base": self.base.config_dict(), "kind": self.kind}

    def new_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("base.")]

    @torch.no_grad()
    def calibrate(self, frame, warped) -> float:
        """Scale the new encoder branch so its latent starts with std TARGET_INIT_STD."""
        h = self.base.encoder.net[:-1](torch.cat([frame, self.base.context(warped)], 1))
        std = float(self.extra_enc(h).std())
        factor = TARGET_INIT_STD / max(std, 1e-6)
        self.extra_enc.weight.mul_(factor)
        self.extra_enc.bias.mul_(factor)
        return factor

    def forward(self, frame: torch.Tensor, warped: torch.Tensor,
                rounding: bool = True) -> dict[str, torch.Tensor]:
        base, training = self.base, self.training
        context = base.context(warped)
        h = base.encoder.net[:-1](torch.cat([frame, context], dim=1))
        y = base.encoder.net[-1](h)
        features = base.prior[:-1](context)
        mean, log_scale = base.prior[-1](features).chunk(2, dim=1)
        scale = log_scale.exp().clamp_min(p1.SCALE_FLOOR)
        rate_input, decoded, bits = quantize(y, mean, scale, training)
        if not rounding:
            decoded = y
        syn = base.synthesis.net
        if self.kind == "plain":
            synthesized = syn(decoded)
        else:
            x = self.extra_enc(h)
            coarse = decoded if self.kind == "a1" else F.interpolate(decoded, scale_factor=2,
                                                                     mode="nearest")
            prior_in = torch.cat([self.extra_ctx(features), coarse], dim=1)
            x_mean, x_log_scale = self.extra_prior(prior_in).chunk(2, dim=1)
            x_scale = x_log_scale.exp().clamp_min(p1.SCALE_FLOOR)
            _, x_decoded, x_bits = quantize(x, x_mean, x_scale, training)
            if not rounding:
                x_decoded = x
            bits = bits + x_bits
            if self.kind == "a1":
                s = syn[0](decoded) + self.extra_syn(x_decoded)
                synthesized = syn[1:](s)
            else:
                s = syn[2](syn[1](syn[0](decoded)))
                synthesized = syn[3:](s + self.extra_syn(x_decoded))
        reconstruction = synthesized + base.refine(torch.cat([synthesized, warped, context], 1))
        if not training:
            reconstruction = reconstruction.clamp(0, 1)
        return {"reconstruction": reconstruction, "bits": bits, "symbols": rate_input}


def load_base(directory: Path, lam: str) -> p1.ConditionalPFrameCoder:
    saved = torch.load(directory / f"lambda{lam}" / "model.pt", map_location="cpu",
                       weights_only=False)
    model = p1.ConditionalPFrameCoder(**saved["config"])
    model.load_state_dict(saved["model_state_dict"])
    return model


def load_arm(kind: str, lam: str, device) -> WideLatentPFrameCoder:
    if kind == "plain":
        return WideLatentPFrameCoder(load_base(PLAIN_DIR, lam)).to(device).eval()
    saved = torch.load(OUTPUT_DIR / f"finetune_{kind}" / f"lambda{lam}" / "model.pt",
                       map_location="cpu", weights_only=False)
    model = WideLatentPFrameCoder(p1.ConditionalPFrameCoder(**saved["config"]["base"]), kind)
    model.load_state_dict(saved["model_state_dict"])
    return model.to(device).eval()


def command_train(args) -> None:
    """The p3_hyperprior plain recipe exactly: same data order, steps, seed and base lr."""
    from nvc.utils.device import get_device
    device = get_device()
    torch.manual_seed(args.seed)
    cache = torch.load(p1.CACHE_DIR / f"train_{args.lam}.pt")
    targets, warps = cache["targets"], cache["warps"]
    model = WideLatentPFrameCoder(load_base(INIT_DIR, args.lam), args.kind).to(device)
    new = {id(p) for p in model.new_parameters()}
    optimizer = torch.optim.Adam([
        {"params": [p for p in model.parameters() if id(p) not in new], "lr": args.lr},
        {"params": model.new_parameters(), "lr": args.new_lr}])
    lam, pixels = float(args.lam), targets.shape[-2] * targets.shape[-1]
    out = OUTPUT_DIR / f"finetune_{args.kind}" / f"lambda{args.lam}"
    out.mkdir(parents=True, exist_ok=True)
    order = torch.Generator().manual_seed(args.seed)
    history, running, started, factor = [], {"mse": 0.0, "bpp": 0.0}, time.time(), None
    for step in range(1, args.steps + 1):
        index = torch.randint(0, len(targets), (args.batch,), generator=order)
        flip = bool(torch.rand((), generator=order) < 0.5)
        frame = targets[index].to(device).float() / 255
        warped = warps[index].to(device).float() / 255
        if flip:
            frame, warped = frame.flip(-1), warped.flip(-1)
        if factor is None:
            factor = model.calibrate(frame, warped)
        model.train()
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
                "steps": args.steps, "lr": args.lr, "new_lr": args.new_lr,
                "batch": args.batch, "seed": args.seed, "init_scale": factor}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


@torch.no_grad()
def ceiling(model, intra, mc, sequences, device, frames_per_sequence: int) -> dict[str, float]:
    """Mean PSNR per sequence with NO rounding, one P-frame at a time from the intra
    reconstruction of the previous frame, block16-warped."""
    out = {}
    with mc.deterministic_kernels():
        for sequence in sequences:
            frames = sequence.load_frames()[:frames_per_sequence + 1].to(device)
            refs = intra.decode(torch.round(intra.encode(frames[:-1]))).clamp(0, 1)
            values = []
            for t in range(1, frames.shape[0]):
                frame, ref = frames[t:t + 1], refs[t - 1:t]
                motion = p1.batched_block_motion(ref, frame, block_size=16, search_range=16)[0]
                warped = mc.warp_blocks(ref, motion, block_size=16)
                values.append(p1.psnr(model(frame, warped, rounding=False)["reconstruction"],
                                      frame))
            out[sequence.sequence_id] = sum(values) / len(values)
    return out


def command_evaluate(args) -> None:
    import benchmark_parity as bp
    from nvc.utils.config import load_default_config
    from nvc.utils.device import get_device
    oracle = p1._load_script("p2_motion_oracle")
    device = get_device()
    mc = p1._load_script("m10h_motion_compensation")
    m21 = p1._load_script("m21_refinement")
    manifest = load_default_config().processed_data_dir / "manifest.json"
    sequences = m21.val_b_sequences(manifest, count=4)
    kinds = ["plain"] + args.kinds
    report = {"val_b": [s.sequence_id for s in sequences], "gop": args.gop, "points": {},
              "ceiling": {}}
    for lam in p1.LAMBDAS:
        intra = p1._intra_model(lam, device)
        report["points"][lam], report["ceiling"][lam] = {}, {}
        for kind in kinds:
            model = load_arm(kind, lam, device)
            point = oracle.run_arm("block16", mc, intra, model, sequences, device, args.gop)
            report["points"][lam][kind] = point
            report["ceiling"][lam][kind] = ceiling(model, intra, mc, sequences, device,
                                                   args.ceiling_frames)
            c = report["ceiling"][lam][kind]
            print(f"  [{lam}] {kind:5s} {point['bits_per_frame']:>9.0f} bits/P  "
                  f"P PSNR {point['p_psnr']:.3f}  ceiling "
                  + " ".join(f"{s}:{v:.2f}" for s, v in c.items()), flush=True)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / "gate1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def curve(kind):
        return sorted((report["points"][lam][kind]["bpp"], report["points"][lam][kind]["p_psnr"])
                      for lam in p1.LAMBDAS)

    # The judged video: lowest plain ceiling, averaged over the three lambdas.
    names = report["val_b"]
    mean_ceiling = {kind: {s: sum(report["ceiling"][lam][kind][s] for lam in p1.LAMBDAS) / 3
                           for s in names} for kind in kinds}
    judged = min(names, key=lambda s: mean_ceiling["plain"][s])
    report["judged_video"], report["gate"] = judged, {}
    for kind in args.kinds:
        bd = bp.bd_rate(curve("plain"), curve(kind))
        lift = mean_ceiling[kind][judged] - mean_ceiling["plain"][judged]
        report["gate"][kind] = {
            "bd_rate_vs_plain": bd, "ceiling_lift_db": lift,
            "passed": bd is not None and bd <= -GATE_SAVING and lift >= GATE_CEILING_DB,
            "near_miss": bd is not None and -GATE_SAVING < bd <= 0 and lift >= GATE_CEILING_DB}
    (OUTPUT_DIR / "gate1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nJudged video (lowest plain ceiling): {judged}")
    for kind, g in report["gate"].items():
        verdict = "PASSED" if g["passed"] else ("NEAR-MISS" if g["near_miss"] else "NOT PASSED")
        print(f"  {kind}: BD-rate vs plain {g['bd_rate_vs_plain']:+.1f}%, ceiling lift "
              f"{g['ceiling_lift_db']:+.2f} dB -> {verdict}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--lambda", dest="lam", required=True, choices=p1.LAMBDAS)
    train.add_argument("--kind", required=True, choices=tuple(EXTRA))
    train.add_argument("--steps", type=int, default=5000)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--lr", type=float, default=5e-5)
    train.add_argument("--new-lr", type=float, default=5e-4)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--log-every", type=int, default=250)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--kinds", nargs="+", default=list(EXTRA), choices=tuple(EXTRA))
    evaluate.add_argument("--gop", type=int, default=10)
    evaluate.add_argument("--ceiling-frames", type=int, default=20)
    args = parser.parse_args(argv)
    {"train": command_train, "evaluate": command_evaluate}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
