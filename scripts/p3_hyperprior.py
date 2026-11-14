"""Hyperprior gate 1: does side information help the learned P-frame coder?

`docs/PLAN_pframe_hyperprior.md`. The stage 3 P-coder predicts each latent's Laplace
(mean, scale) from the warped reference alone. Here the encoder also sends `z`, a small
summary of the latent (32 ch at stride 64), and the decoder adds `h_s(z_hat)` to the
reference-based prediction:

    mean, log_scale = prior(context) + h_s(z_hat)        z = h_a(y, prior(context))

`h_s`'s last conv starts at zero, so at step 0 the model IS the stage 3 coder plus the cost
of `z`. `z` has a factorized zero-mean Laplace per channel (learned scale). That is
range-codable with the gate 3 scale tables.

Gate 1 (fixed in the plan before any result): two copies of the stage 3 coder get the same
fine-tune (steps, data, seed, base lr; DAVIS TRAIN block16 pairs, the gate 1 pilot cache).
One copy has the hyperprior. Both are scored with the motion oracle's closed loop
(VAL-B, GOP 10, block16 motion paid, ideal bits including z). The hyperprior passes if it
cuts P-frame BD-rate by at least 8% vs the plain copy.

Usage:
    p3_hyperprior.py train --lambda 3e-4 --kind plain|hyper [--steps 5000]
    p3_hyperprior.py evaluate
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

OUTPUT_DIR = Path("outputs/p3_hyperprior")
INIT_DIR = p1.OUTPUT_DIR / "gate2_local" / "stage3"
GATE_SAVING = 8.0
HYPER_CHANNELS = 32


class HyperpriorPFrameCoder(nn.Module):
    """The conditional P-coder with a transmitted hyper-latent `z`."""

    def __init__(self, base: p1.ConditionalPFrameCoder, hyper_channels: int = HYPER_CHANNELS,
                 hidden: int = 96) -> None:
        super().__init__()
        self.base = base
        m, k = base.latent_channels, hyper_channels
        self.hyper_channels = hyper_channels
        self.h_a = nn.Sequential(
            nn.Conv2d(3 * m, hidden, 3, padding=1), nn.LeakyReLU(0.1),
            p1._down(hidden, hidden), nn.LeakyReLU(0.1), p1._down(hidden, k))
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(k, hidden, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(hidden, 2 * m, 3, padding=1))
        nn.init.zeros_(self.h_s[-1].weight)
        nn.init.zeros_(self.h_s[-1].bias)
        self.z_log_scale = nn.Parameter(torch.zeros(1, k, 1, 1))

    def config_dict(self) -> dict:
        return {"base": self.base.config_dict(), "hyper_channels": self.hyper_channels}

    def new_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("base.")]

    def forward(self, frame: torch.Tensor, warped: torch.Tensor) -> dict[str, torch.Tensor]:
        base = self.base
        context = base.context(warped)
        y = base.encoder(torch.cat([frame, context], dim=1))
        prior = base.prior(context)
        z = self.h_a(torch.cat([y, prior], dim=1))
        z_scale = self.z_log_scale.exp().clamp_min(p1.SCALE_FLOOR)
        if self.training:
            z_rate = z + torch.empty_like(z).uniform_(-0.5, 0.5)
            z_hat = z + (torch.round(z) - z).detach()
        else:
            z_rate = z_hat = torch.round(z)
        z_bits = p1.laplace_bits(z_rate, z_scale.expand_as(z_rate))
        side = self.h_s(z_hat)
        if side.shape[-2:] != prior.shape[-2:]:
            side = F.interpolate(side, size=prior.shape[-2:], mode="nearest")
        mean, log_scale = (prior + side).chunk(2, dim=1)
        scale = log_scale.exp().clamp_min(p1.SCALE_FLOOR)
        centred = y - mean
        if self.training:
            rate_input = centred + torch.empty_like(centred).uniform_(-0.5, 0.5)
            decoded = mean + centred + (torch.round(centred) - centred).detach()
        else:
            rate_input = torch.round(centred)
            decoded = mean + rate_input
        y_bits = p1.laplace_bits(rate_input, scale).flatten(1).sum(1)
        z_bits = z_bits.flatten(1).sum(1)
        synthesized = base.synthesis(decoded)
        reconstruction = synthesized + base.refine(torch.cat([synthesized, warped, context], 1))
        if not self.training:
            reconstruction = reconstruction.clamp(0, 1)
        return {"reconstruction": reconstruction, "bits": y_bits + z_bits, "y_bits": y_bits,
                "z_bits": z_bits, "symbols": rate_input, "z_symbols": z_rate}


def load_base(models_dir: Path, lam: str) -> p1.ConditionalPFrameCoder:
    saved = torch.load(models_dir / f"lambda{lam}" / "model.pt", map_location="cpu",
                       weights_only=False)
    model = p1.ConditionalPFrameCoder(**saved["config"])
    model.load_state_dict(saved["model_state_dict"])
    return model


def load_model(kind: str, lam: str, device) -> nn.Module:
    saved = torch.load(OUTPUT_DIR / f"finetune_{kind}" / f"lambda{lam}" / "model.pt",
                       map_location="cpu", weights_only=False)
    if kind == "hyper":
        model = HyperpriorPFrameCoder(p1.ConditionalPFrameCoder(**saved["config"]["base"]),
                                      saved["config"]["hyper_channels"])
    else:
        model = p1.ConditionalPFrameCoder(**saved["config"])
    model.load_state_dict(saved["model_state_dict"])
    return model.to(device).eval()


def command_train(args) -> None:
    from nvc.utils.device import get_device
    device = get_device()
    torch.manual_seed(args.seed)
    cache = torch.load(p1.CACHE_DIR / f"train_{args.lam}.pt")
    targets, warps = cache["targets"], cache["warps"]
    base = load_base(INIT_DIR, args.lam)
    model = HyperpriorPFrameCoder(base) if args.kind == "hyper" else base
    model = model.to(device).train()
    if args.kind == "hyper":
        new = {id(p) for p in model.new_parameters()}
        groups = [{"params": [p for p in model.parameters() if id(p) not in new], "lr": args.lr},
                  {"params": model.new_parameters(), "lr": args.new_lr}]
    else:
        groups = [{"params": list(model.parameters()), "lr": args.lr}]
    optimizer = torch.optim.Adam(groups)
    lam, pixels = float(args.lam), targets.shape[-2] * targets.shape[-1]
    out = OUTPUT_DIR / f"finetune_{args.kind}" / f"lambda{args.lam}"
    out.mkdir(parents=True, exist_ok=True)
    history, started = [], time.time()
    running = {"mse": 0.0, "bpp": 0.0, "z_bpp": 0.0}
    # Same data order for both kinds: draw indices from a generator seeded independently of
    # the model's own random draws (the hyper model consumes extra noise for z).
    order = torch.Generator().manual_seed(args.seed)
    for step in range(1, args.steps + 1):
        index = torch.randint(0, len(targets), (args.batch,), generator=order)
        flip = bool(torch.rand((), generator=order) < 0.5)
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
        if "z_bits" in result:
            running["z_bpp"] += (result["z_bits"].mean() / pixels).item()
        if step % args.log_every == 0:
            entry = {k: v / args.log_every for k, v in running.items()}
            entry.update(step=step, psnr=10 * math.log10(1 / entry["mse"]),
                         minutes=(time.time() - started) / 60)
            history.append(entry)
            print(f"  [{args.kind} {args.lam}] step {step:5d}  {entry['psnr']:.2f} dB  "
                  f"{entry['bpp']:.4f} bpp (z {entry['z_bpp']:.4f})  "
                  f"{entry['minutes']:.1f} min", flush=True)
            running = dict.fromkeys(running, 0.0)
    torch.save({"model_state_dict": model.state_dict(), "config": model.config_dict(),
                "lambda": args.lam, "init": str(INIT_DIR), "kind": args.kind,
                "steps": args.steps, "lr": args.lr, "new_lr": args.new_lr,
                "batch": args.batch, "seed": args.seed}, out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


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
    report = {"val_b": [s.sequence_id for s in sequences], "gop": args.gop, "points": {}}
    for lam in p1.LAMBDAS:
        intra = p1._intra_model(lam, device)
        report["points"][lam] = {}
        for kind in ("plain", "hyper"):
            model = load_model(kind, lam, device)
            z_bits = []
            if kind == "hyper":
                forward = model.forward

                def counting(frame, warped, _forward=forward):
                    result = _forward(frame, warped)
                    z_bits.append(float(result["z_bits"].sum()))
                    return result
                model.forward = counting
            point = oracle.run_arm("block16", mc, intra, model, sequences, device, args.gop)
            point["z_bits_per_frame"] = sum(z_bits) / len(z_bits) if z_bits else 0.0
            report["points"][lam][kind] = point
            print(f"  [{lam}] {kind:5s} {point['bits_per_frame']:>9.0f} bits/P "
                  f"(z {point['z_bits_per_frame']:>5.0f}, motion "
                  f"{point['motion_bits_per_frame']:>5.0f})  P PSNR {point['p_psnr']:.3f}",
                  flush=True)

    def curve(kind):
        return sorted((report["points"][lam][kind]["bpp"], report["points"][lam][kind]["p_psnr"])
                      for lam in p1.LAMBDAS)

    bd = bp.bd_rate(curve("plain"), curve("hyper"))
    report["bd_rate_hyper_vs_plain"] = bd
    report["gate"] = {"threshold_percent": -GATE_SAVING, "bd_rate": bd,
                      "passed": bd is not None and bd <= -GATE_SAVING}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "gate1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nP-frame BD-rate, hyperprior vs plain: {bd:+.1f}%\n"
          f"GATE 1 ({'PASSED' if report['gate']['passed'] else 'NOT PASSED'}): "
          f"<= -{GATE_SAVING:.0f}%")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--lambda", dest="lam", required=True, choices=p1.LAMBDAS)
    train.add_argument("--kind", required=True, choices=("plain", "hyper"))
    train.add_argument("--steps", type=int, default=5000)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--lr", type=float, default=5e-5)
    train.add_argument("--new-lr", type=float, default=1e-3,
                       help="learning rate for the hyperprior's new layers only")
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--log-every", type=int, default=250)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--gop", type=int, default=10)
    args = parser.parse_args(argv)
    {"train": command_train, "evaluate": command_evaluate}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
