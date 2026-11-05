"""P1: a learned conditional P-frame coder, local pilot (gate 1 of docs/PLAN_learned_pframe.md).

THE QUESTION
-------------
`outputs/benchmarks/bit_split_phase_b/` measured the deployed P-frame path at 0.59-0.68x an
I-frame's bytes: `dz = E(x_t) - E(warp(x_hat_{t-1}))` through an encoder trained only on
single images. Can a coder TRAINED for P-frames, which sees the warped reference as a
condition, do much better? Gate 1 asks it on the proxy, before any Colab time:

    learned P-frame bits <= 0.45x the intra model's bits at matched PSNR, on VAL-B.

WHAT IS COMPARED (one P-step, reference = the intra codec's reconstruction of t-1)
-------------------------------------------------------------------------------------
    intra       z = E(x_t), round(z), x_hat = D(round z)          the Phase B model
    hand-built  dz = round(E(x_t) - E(x_warp)), x_hat = D(E(x_warp) + dz)
    learned     y = E_p(x_t, c), y_hat = mu + round(y - mu), x_hat = x_warp + R(S(y_hat), c)
                with c = F(x_warp) and the prior (mu, b) = P(c)

Motion is the deployed integer 16x16 block search (`m10h_motion_compensation`) on the
intra reconstruction, unchanged and identical for hand-built and learned.

Rates are ideal code lengths, not coded bytes. Intra and hand-built use a static
per-channel histogram fitted on TRAIN and scored on VAL-B (what M10J-style coding gets);
learned uses its own conditional Laplace. The deployed codec adds context modelling on
top of the static code (M11: +3-8%), so the hand-built row is the control: its proxy
ratio should land near the measured 0.59-0.68 real ratio, or this rig is off.

Nothing selects on VAL-B: each model is trained for a fixed budget and scored once.

Usage:
    p1_pframe.py cache --lambda 3e-4
    p1_pframe.py train --lambda 3e-4 [--steps N]
    p1_pframe.py gate
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvc.models.residual_transform import AnalysisTransform, ResidualBlock, SynthesisTransform

LAMBDAS = ("1e-3", "3e-4", "1e-4")
CHECKPOINT = ("outputs/stage1_vimeo/colab_run5_phase_b/phase_b_unitstep_lambda{}"
              "/checkpoints/best.pt")
OUTPUT_DIR = Path("outputs/p1_pframe")
CACHE_DIR = Path(tempfile.gettempdir()) / "nvc_p1_cache"
GATE_RATIO = 0.45
SYMBOL_OFFSET = 512          # histogram symbols live in [-512, 512)
LIKELIHOOD_FLOOR = 1e-9
SCALE_FLOOR = 0.11


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------------


def _down(cin: int, cout: int) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 5, stride=2, padding=2)


class ConditionalPFrameCoder(nn.Module):
    """Codes x_t given the warped reference x_warp, which the decoder also has.

    Built to start from the trained intra codec (`from_intra`): the encoder's extra
    context inputs and the refinement's last conv start at zero, so at step 0 the coder
    IS the intra codec with a conditional prior, and training can only teach it to use
    the reference. Trained from scratch instead, the first pilot (2026-11-05) sat at the
    zero-rate optimum - x_hat ~= x_warp, 26 dB, 0.05 bpp - for 16k steps and never left.
    """

    def __init__(self, latent_channels: int = 64, base_channels: int = 192,
                 context_channels: int = 32, residual_blocks: int = 1) -> None:
        super().__init__()
        n, c, m = base_channels, context_channels, latent_channels
        self.latent_channels = latent_channels
        self.context = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1), nn.LeakyReLU(0.1), ResidualBlock(c), ResidualBlock(c))
        self.encoder = AnalysisTransform(3 + c, m, n, residual_blocks)
        self.prior = nn.Sequential(
            _down(c, n // 2), nn.LeakyReLU(0.1), _down(n // 2, n // 2), nn.LeakyReLU(0.1),
            ResidualBlock(n // 2), _down(n // 2, n // 2), nn.LeakyReLU(0.1),
            ResidualBlock(n // 2), _down(n // 2, 2 * m))
        self.synthesis = SynthesisTransform(3, m, n, residual_blocks)
        self.refine = nn.Sequential(
            nn.Conv2d(6 + c, 2 * c, 3, padding=1), nn.LeakyReLU(0.1),
            ResidualBlock(2 * c), ResidualBlock(2 * c), nn.Conv2d(2 * c, 3, 3, padding=1))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    @classmethod
    def from_intra(cls, intra: nn.Module, context_channels: int = 32) -> "ConditionalPFrameCoder":
        """Copy the intra transform in; the context's share of the first conv is zero."""
        model = cls(intra.latent_channels, intra.base_channels, context_channels,
                    intra.residual_blocks)
        state = intra.encoder.state_dict()
        first = "net.0.weight"
        widened = torch.zeros_like(model.encoder.state_dict()[first])
        widened[:, :3] = state[first]
        state[first] = widened
        model.encoder.load_state_dict(state)
        model.synthesis.load_state_dict(intra.decoder.state_dict())
        return model

    def config_dict(self) -> dict[str, int]:
        return {"latent_channels": self.latent_channels,
                "base_channels": self.synthesis.net[0].out_channels,
                "context_channels": self.context[0].out_channels,
                "residual_blocks": sum(isinstance(m, ResidualBlock) for m in self.synthesis.net)
                // 2}

    def forward(self, frame: torch.Tensor, warped: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.context(warped)
        y = self.encoder(torch.cat([frame, context], dim=1))
        mean, log_scale = self.prior(context).chunk(2, dim=1)
        scale = log_scale.exp().clamp_min(SCALE_FLOOR)
        centred = y - mean
        if self.training:
            # Additive noise for the rate, straight-through rounding for the decoder.
            rate_input = centred + torch.empty_like(centred).uniform_(-0.5, 0.5)
            decoded = mean + centred + (torch.round(centred) - centred).detach()
        else:
            rate_input = torch.round(centred)
            decoded = mean + rate_input
        bits = laplace_bits(rate_input, scale)
        base = self.synthesis(decoded)
        reconstruction = base + self.refine(torch.cat([base, warped, context], dim=1))
        if not self.training:
            reconstruction = reconstruction.clamp(0, 1)
        return {"reconstruction": reconstruction, "bits": bits.flatten(1).sum(1),
                "symbols": rate_input}


def laplace_bits(centred: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """-log2 of a zero-mean Laplace(scale)'s mass on [v - 1/2, v + 1/2], computed on |v|
    so neither tail cancels catastrophically."""
    a = centred.abs()
    lower, upper = a - 0.5, a + 0.5
    tail = 0.5 * (torch.exp(-lower.clamp_min(0) / scale) - torch.exp(-upper / scale))
    middle = 1 - 0.5 * (torch.exp(-upper / scale) + torch.exp(lower.clamp_max(0) / scale))
    probability = torch.where(lower >= 0, tail, middle)
    return -torch.log2(probability.clamp_min(LIKELIHOOD_FLOOR))


# --------------------------------------------------------------------------------------------
# Pair cache: (x_t, warp of the intra reconstruction of x_{t-1}) per lambda
# --------------------------------------------------------------------------------------------


def _intra_model(lam: str, device):
    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(Path(CHECKPOINT.format(lam)), device=device)
    return model.eval()


def _histogram_add(histogram: torch.Tensor, symbols: torch.Tensor) -> None:
    """histogram [C, 1024] += counts of integer `symbols` [N, C, H, W] per channel."""
    channels = symbols.shape[1]
    s = (symbols.long() + SYMBOL_OFFSET).clamp(0, 2 * SYMBOL_OFFSET - 1)
    s = s.permute(1, 0, 2, 3).reshape(channels, -1)
    histogram += torch.zeros_like(histogram).scatter_add_(1, s, torch.ones_like(s,
                                                                                dtype=histogram.dtype))


@torch.no_grad()
def build_pairs(model, sequences, mc, device, *, keep_float: bool, histograms=None):
    """Every (t-1, t) pair inside each sequence. Returns targets, warps, and for
    `keep_float` the baselines' per-frame PSNR and symbols."""
    targets, warps, records = [], [], {"intra_psnr": [], "hand_psnr": [], "warp_psnr": [],
                                       "intra_symbols": [], "hand_symbols": []}
    with mc.deterministic_kernels():
        for sequence in sequences:
            frames = sequence.load_frames().to(device)
            latents = torch.cat([model.encode(frames[i:i + 16]) for i in range(0, len(frames), 16)])
            reconstructions = torch.cat([model.decode(torch.round(latents[i:i + 16]))
                                         for i in range(0, len(frames), 16)])
            for t in range(1, len(frames)):
                reference, frame = reconstructions[t - 1:t], frames[t:t + 1]
                motion = mc.estimate_block_motion(reference, frame)
                warped = mc.warp_blocks(reference, motion)
                warped_latent = model.encode(warped)
                hand_symbols = torch.round(latents[t:t + 1] - warped_latent)
                intra_symbols = torch.round(latents[t:t + 1])
                if histograms is not None:
                    _histogram_add(histograms["intra"], intra_symbols)
                    _histogram_add(histograms["hand"], hand_symbols)
                targets.append((frame * 255).round().to(torch.uint8).cpu())
                if keep_float:
                    warps.append(warped.half().cpu())
                    hand = model.decode(warped_latent + hand_symbols)
                    records["intra_psnr"].append(psnr(reconstructions[t:t + 1], frame))
                    records["hand_psnr"].append(psnr(hand, frame))
                    records["warp_psnr"].append(psnr(warped, frame))
                    records["intra_symbols"].append(intra_symbols.short().cpu())
                    records["hand_symbols"].append(hand_symbols.short().cpu())
                else:
                    warps.append((warped * 255).round().to(torch.uint8).cpu())
            print(f"    {sequence.sequence_id}: {len(frames) - 1} pairs", flush=True)
    return torch.cat(targets), torch.cat(warps), records


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.float().clamp(0, 1), b.float()).item()
    return 10 * math.log10(1 / max(mse, 1e-12))


def command_cache(args) -> None:
    from nvc.evaluation.sequences import discover_sequences
    from nvc.utils.config import load_default_config
    from nvc.utils.device import get_device
    mc, m21 = _load_script("m10h_motion_compensation"), _load_script("m21_refinement")
    device = get_device()
    manifest = load_default_config().processed_data_dir / "manifest.json"
    model = _intra_model(args.lam, device)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    channels = model.latent_channels
    histograms = {k: torch.zeros(channels, 2 * SYMBOL_OFFSET, dtype=torch.float64,
                                 device=device) for k in ("intra", "hand")}
    print(f"[{args.lam}] TRAIN pairs", flush=True)
    train = discover_sequences(manifest, split="train")
    targets, warps, _ = build_pairs(model, train, mc, device, keep_float=False,
                                    histograms=histograms)
    torch.save({"targets": targets, "warps": warps}, CACHE_DIR / f"train_{args.lam}.pt")

    print(f"[{args.lam}] VAL-B pairs", flush=True)
    val_b = m21.val_b_sequences(manifest, count=4)
    targets, warps, records = build_pairs(model, val_b, mc, device, keep_float=True)
    records["intra_symbols"] = torch.cat(records["intra_symbols"])
    records["hand_symbols"] = torch.cat(records["hand_symbols"])
    torch.save({"targets": targets, "warps": warps, "records": records,
                "histograms": {k: v.cpu() for k, v in histograms.items()},
                "val_b": [s.sequence_id for s in val_b]}, CACHE_DIR / f"val_b_{args.lam}.pt")
    print(f"[{args.lam}] {len(targets)} VAL-B pairs; warp {np.mean(records['warp_psnr']):.2f} dB, "
          f"intra {np.mean(records['intra_psnr']):.2f}, hand-built {np.mean(records['hand_psnr']):.2f}")


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------


def command_train(args) -> None:
    from nvc.utils.device import get_device
    device = get_device()
    torch.manual_seed(args.seed)
    data = torch.load(CACHE_DIR / f"train_{args.lam}.pt")
    targets, warps = data["targets"], data["warps"]
    lam = float(args.lam)
    model = ConditionalPFrameCoder.from_intra(_intra_model(args.lam, "cpu")).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pixels = targets.shape[-2] * targets.shape[-1]
    out = OUTPUT_DIR / f"lambda{args.lam}"
    out.mkdir(parents=True, exist_ok=True)
    history, started = [], time.time()
    running = {"loss": 0.0, "mse": 0.0, "bpp": 0.0}
    for step in range(1, args.steps + 1):
        if step == int(args.steps * 0.9):
            for group in optimizer.param_groups:
                group["lr"] = args.lr / 10
        index = torch.randint(0, len(targets), (args.batch,))
        frame = targets[index].to(device).float() / 255
        warped = warps[index].to(device).float() / 255
        if torch.rand(()) < 0.5:
            frame, warped = frame.flip(-1), warped.flip(-1)
        result = model(frame, warped)
        mse = F.mse_loss(result["reconstruction"], frame)
        bpp = result["bits"].mean() / pixels
        loss = mse + lam * bpp
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for key, value in (("loss", loss), ("mse", mse), ("bpp", bpp)):
            running[key] += value.item()
        if step % args.log_every == 0:
            entry = {k: v / args.log_every for k, v in running.items()}
            entry.update(step=step, psnr=10 * math.log10(1 / entry["mse"]),
                         minutes=(time.time() - started) / 60)
            history.append(entry)
            print(f"  step {step:6d}  {entry['psnr']:.2f} dB  {entry['bpp']:.4f} bpp  "
                  f"loss {entry['loss']:.6f}  {entry['minutes']:.1f} min", flush=True)
            running = dict.fromkeys(running, 0.0)
    torch.save({"model_state_dict": model.state_dict(), "config": model.config_dict(),
                "lambda": args.lam, "warm_start": CHECKPOINT.format(args.lam),
                "steps": args.steps, "batch": args.batch, "lr": args.lr, "seed": args.seed},
               out / "model.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------------------------


def static_bits(symbols: torch.Tensor, histogram: torch.Tensor) -> torch.Tensor:
    """Per-frame code length of `symbols` [N, C, H, W] under per-channel TRAIN
    histograms, with one pseudo-count spread over every bin (unseen symbols stay finite)."""
    smoothed = histogram + 1.0 / histogram.shape[1]
    log_p = torch.log2(smoothed / smoothed.sum(1, keepdim=True))
    s = (symbols.long() + SYMBOL_OFFSET).clamp(0, 2 * SYMBOL_OFFSET - 1)
    channels = symbols.shape[1]
    per = torch.stack([-log_p[c][s[:, c]] for c in range(channels)], dim=1)
    return per.flatten(1).sum(1)


@torch.no_grad()
def evaluate_point(lam: str, device) -> dict[str, dict[str, float]]:
    data = torch.load(CACHE_DIR / f"val_b_{lam}.pt")
    records, histograms = data["records"], data["histograms"]
    frames = len(data["targets"])
    point = {
        "intra": {"bits": static_bits(records["intra_symbols"], histograms["intra"]).mean().item(),
                  "psnr": float(np.mean(records["intra_psnr"]))},
        "hand_built": {"bits": static_bits(records["hand_symbols"], histograms["hand"]).mean().item(),
                       "psnr": float(np.mean(records["hand_psnr"]))},
        "warp_only_psnr": float(np.mean(records["warp_psnr"])),
        "frames": frames, "val_b": data["val_b"],
    }
    path = OUTPUT_DIR / f"lambda{lam}" / "model.pt"
    if path.is_file():
        saved = torch.load(path)
        model = ConditionalPFrameCoder(**saved["config"]).to(device)
        model.load_state_dict(saved["model_state_dict"])
        model.eval()
        bits, scores = [], []
        for i in range(frames):
            frame = data["targets"][i:i + 1].to(device).float() / 255
            result = model(frame, data["warps"][i:i + 1].to(device).float())
            bits.append(result["bits"].item())
            scores.append(psnr(result["reconstruction"], frame))
        point["learned"] = {"bits": float(np.mean(bits)), "psnr": float(np.mean(scores))}
    return point


def command_gate(args) -> None:
    from nvc.utils.device import get_device
    bp = _load_script("benchmark_parity")
    device = get_device()
    points = {lam: evaluate_point(lam, device) for lam in LAMBDAS
              if (CACHE_DIR / f"val_b_{lam}.pt").is_file()}
    curves = {arm: [(p[arm]["bits"], p[arm]["psnr"]) for p in points.values() if arm in p]
              for arm in ("intra", "hand_built", "learned")}
    report = {"points": points, "gate_ratio": GATE_RATIO, "comparisons": {}}
    print(f"{'lambda':>7} {'arm':>11} {'bits/frame':>11} {'PSNR':>7} {'x intra at same PSNR':>22}")
    for lam, p in points.items():
        for arm in ("intra", "hand_built", "learned"):
            if arm not in p:
                continue
            ratio = (bp.rate_ratio_at_quality(curves["intra"], p[arm]["bits"], p[arm]["psnr"])
                     if len(curves["intra"]) > 1 else None)
            p[arm]["ratio_vs_intra"] = ratio
            print(f"{lam:>7} {arm:>11} {p[arm]['bits']:11.0f} {p[arm]['psnr']:7.2f} "
                  f"{('%.3f' % ratio) if ratio else 'outside range':>22}")
    for arm in ("hand_built", "learned"):
        if len(curves[arm]) > 1 and len(curves["intra"]) > 1:
            report["comparisons"][f"{arm}_vs_intra_bd_rate"] = bp.bd_rate(curves["intra"],
                                                                          curves[arm])
    if len(curves["learned"]) > 1:
        report["comparisons"]["learned_vs_hand_built_bd_rate"] = bp.bd_rate(
            curves["hand_built"], curves["learned"])
    ratios = [p["learned"].get("ratio_vs_intra") for p in points.values() if "learned" in p]
    measured = [r for r in ratios if r is not None]
    report["gate_passed"] = bool(measured) and max(measured) <= GATE_RATIO
    for key, value in report["comparisons"].items():
        print(f"  {key}: {('%+.1f%%' % value) if value is not None else 'n/a'}")
    print(f"GATE 1 ({'PASSED' if report['gate_passed'] else 'NOT PASSED'}): learned P-frame "
          f"<= {GATE_RATIO}x intra at matched PSNR at every measured point: {measured}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "gate1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("cache")
    cache.add_argument("--lambda", dest="lam", choices=LAMBDAS, required=True)
    train = sub.add_parser("train")
    train.add_argument("--lambda", dest="lam", choices=LAMBDAS, required=True)
    train.add_argument("--steps", type=int, default=20000)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--log-every", type=int, default=500)
    sub.add_parser("gate")
    args = parser.parse_args(argv)
    {"cache": command_cache, "train": command_train, "gate": command_gate}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
