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
    p1_pframe.py gate [--models-dir DIR --output-name gate2.json]
    p1_pframe.py train-vimeo --output-dir D --scratch-dir S --chunks 1 2 3 [--unroll 3]
                             (gate 2, on Colab: see colab_train_pframe.ipynb)
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
# Gate 2: training on Vimeo-90K septuplets (Colab), one step or unrolled
# --------------------------------------------------------------------------------------------

BLOCK_SIZE = 16
SEARCH_RANGE = 16            # m10h_motion_compensation's defaults; the gate uses those


@torch.no_grad()
def batched_block_motion(reference: torch.Tensor, current: torch.Tensor, *,
                         block_size: int = BLOCK_SIZE,
                         search_range: int = SEARCH_RANGE) -> torch.Tensor:
    """`m10h_motion_compensation.estimate_block_motion` over a batch: the same full SAD
    search and the same choice per block - lowest cost, then smallest |dy| + |dx|, then
    the first candidate in (dy, dx) order. [N, 2, by, bx].

    All candidates' costs are computed one dy row at a time and the choice is made once,
    instead of 1,089 sequential updates; that is ~an order of magnitude faster on a GPU."""
    n, _, height, width = reference.shape
    span = 2 * search_range + 1
    padded = F.pad(reference, (search_range,) * 4, mode="replicate")
    rows = []
    for top in range(span):
        band = padded[:, :, top:top + height]
        shifted = torch.stack([band[..., left:left + width] for left in range(span)])
        absdiff = (current.unsqueeze(0) - shifted).abs().sum(dim=2)          # [span, N, H, W]
        cost = F.avg_pool2d(absdiff.flatten(0, 1).unsqueeze(1), block_size)[:, 0]
        rows.append(cost.view(span, n, *cost.shape[1:]) * (block_size * block_size))
    cost = torch.cat(rows)                                                      # [span^2, N, by, bx]
    offsets = torch.arange(-search_range, search_range + 1, device=reference.device)
    dy = offsets.repeat_interleave(span)
    dx = offsets.repeat(span)
    key = (dy.abs() + dx.abs()).float().view(-1, 1, 1, 1).expand_as(cost)
    best = cost == cost.min(0, keepdim=True).values
    key = torch.where(best, key, torch.full_like(key, float("inf")))
    best &= key == key.min(0, keepdim=True).values
    index = best.to(torch.uint8).argmax(0)                                      # first True
    return torch.stack([dy[index], dx[index]], dim=1)


@torch.no_grad()
def batched_warp(reference: torch.Tensor, motion: torch.Tensor, *,
                 block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """`m10h_motion_compensation.warp_blocks` over a batch: integer block copies with
    replicate edges."""
    n, channels, height, width = reference.shape
    dy = motion[:, 0].repeat_interleave(block_size, 1).repeat_interleave(block_size, 2)
    dx = motion[:, 1].repeat_interleave(block_size, 1).repeat_interleave(block_size, 2)
    rows = (torch.arange(height, device=reference.device).view(1, height, 1) + dy).clamp(0, height - 1)
    cols = (torch.arange(width, device=reference.device).view(1, 1, width) + dx).clamp(0, width - 1)
    flat = (rows * width + cols).view(n, 1, -1).expand(n, channels, -1)
    return reference.reshape(n, channels, -1).gather(2, flat).reshape(n, channels, height, width)


class VimeoClips(torch.utils.data.Dataset):
    """`length` consecutive frames from a random start inside each septuplet, one random
    crop and flip shared by the whole clip. uint8 [length, 3, crop, crop]."""

    def __init__(self, clip_dirs: list[Path], length: int, crop: int) -> None:
        if not 2 <= length <= 7:
            raise ValueError(f"a clip needs 2..7 frames, got {length}")
        self.clip_dirs, self.length, self.crop = clip_dirs, length, crop

    def __len__(self) -> int:
        return len(self.clip_dirs)

    def __getitem__(self, index: int) -> torch.Tensor:
        import random

        import cv2
        directory = self.clip_dirs[index]
        start = random.randint(1, 8 - self.length)
        frames = np.stack([cv2.imread(str(directory / f"im{start + k}.png"))[:, :, ::-1]
                           for k in range(self.length)])
        height, width = frames.shape[1:3]
        y, x = random.randint(0, height - self.crop), random.randint(0, width - self.crop)
        frames = frames[:, y:y + self.crop, x:x + self.crop]
        if random.random() < 0.5:
            frames = frames[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)


def complete_clip_dirs(group_root: Path) -> list[Path]:
    """Every <group>/<clip> folder under `group_root` holding all seven frames."""
    return [clip for group in sorted(p for p in group_root.iterdir() if p.is_dir())
            for clip in sorted(p for p in group.iterdir() if p.is_dir())
            if all((clip / f"im{i}.png").is_file() for i in range(1, 8))]


def unrolled_step(model: ConditionalPFrameCoder, intra: nn.Module, clip: torch.Tensor,
                  lam: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean MSE and bits per P-frame of `clip` [N, T, 3, H, W] in [0, 1].

    Frame 0 is the I-frame (the frozen intra codec). Each P-frame references the
    previous reconstruction, so with T > 2 the coder sees, and pays for, its own
    accumulated error - stage 2 of the plan. The reference is detached: the chain is
    trained on its own outputs, not differentiated through them.

    With `lam` given, each P-frame's share of (mse + lam * bits/pixel) is backpropagated
    as soon as it is computed. Because the references are detached the gradient is the
    same as one backward over the whole chain, but only one P-frame's activations are
    alive at a time: holding all of them overflowed an 8 GB GPU at T = 4 and ran ~15x
    slower. The returned values are then detached, for logging."""
    with torch.no_grad():
        reference = intra.decode(torch.round(intra.encode(clip[:, 0]))).clamp(0, 1)
    steps = clip.shape[1] - 1
    pixels = clip.shape[-1] * clip.shape[-2]
    mse = bits = 0.0
    for t in range(1, clip.shape[1]):
        frame = clip[:, t]
        warped = batched_warp(reference, batched_block_motion(reference, frame))
        result = model(frame, warped)
        frame_mse = F.mse_loss(result["reconstruction"], frame)
        frame_bits = result["bits"].mean()
        if lam is not None:
            ((frame_mse + lam * frame_bits / pixels) / steps).backward()
            frame_mse, frame_bits = frame_mse.detach(), frame_bits.detach()
        mse = mse + frame_mse
        bits = bits + frame_bits
        reference = result["reconstruction"].detach().clamp(0, 1)
    return mse / steps, bits / steps


def _save_json(path: Path, value) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def train_lambda_on_chunk(lam: str, chunk: int, clip_dirs: list[Path], args, device) -> None:
    """Train one lambda's P-coder for `args.steps_per_chunk` steps on one chunk, resuming
    from its own latest.pt (which records the chunk and step reached)."""
    from nvc.training.checkpoint import load_model_from_checkpoint
    out = Path(args.output_dir) / f"lambda{lam}"
    out.mkdir(parents=True, exist_ok=True)
    intra_path = Path(args.intra_checkpoint.format(lam))
    intra, _ = load_model_from_checkpoint(intra_path, device=device)
    intra.eval().requires_grad_(False)

    latest = out / "latest.pt"
    start_step, history = 0, []
    if latest.is_file():
        saved = torch.load(latest, map_location="cpu")
        model = ConditionalPFrameCoder(**saved["config"])
        model.load_state_dict(saved["model_state_dict"])
        source = f"resumed {latest}"
    elif args.init_from:
        saved = torch.load(Path(args.init_from) / f"lambda{lam}" / "model.pt", map_location="cpu")
        model = ConditionalPFrameCoder(**saved["config"])
        model.load_state_dict(saved["model_state_dict"])
        saved = None
        source = f"init from {args.init_from}"
    else:
        model = ConditionalPFrameCoder.from_intra(intra.cpu())
        intra.to(device)
        saved = None
        source = f"warm start from intra {intra_path}"
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        history = saved.get("history", [])
        if saved.get("chunk") == chunk:
            start_step = saved["chunk_step"]
    decayed = chunk in args.chunks[len(args.chunks) - args.lr_decay_chunks:] if args.lr_decay_chunks else False
    lr = args.lr * (0.1 if decayed else 1.0)
    for group in optimizer.param_groups:
        group["lr"] = lr
    print(f"[lambda {lam} chunk {chunk}] {source}; {len(clip_dirs)} clips, unroll {args.unroll}, "
          f"lr {lr:g}, steps {start_step}->{args.steps_per_chunk}", flush=True)

    loader = torch.utils.data.DataLoader(
        VimeoClips(clip_dirs, args.unroll + 1, args.crop), batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0)

    def checkpoint(step: int) -> None:
        state = {"model_state_dict": model.state_dict(), "config": model.config_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "chunk": chunk,
                 "chunk_step": step, "history": history, "lambda": lam,
                 "intra_checkpoint": str(intra_path), "unroll": args.unroll}
        torch.save(state, out / "latest.tmp")
        (out / "latest.tmp").replace(latest)

    lam_value, pixels = float(lam), args.crop * args.crop
    step, started = start_step, time.time()
    running = {"loss": 0.0, "mse": 0.0, "bpp": 0.0}
    seen = 0
    while step < args.steps_per_chunk:
        for clip in loader:
            clip = clip.to(device, non_blocking=True).float() / 255
            optimizer.zero_grad(set_to_none=True)
            mse, bits = unrolled_step(model, intra, clip, lam_value)   # backpropagates
            bpp = bits / pixels
            loss = mse + lam_value * bpp
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            seen += 1
            for key, value in (("loss", loss), ("mse", mse), ("bpp", bpp)):
                running[key] += value.item()
            if step % args.log_every == 0:
                entry = {k: v / seen for k, v in running.items()}
                entry.update(chunk=chunk, step=step, psnr=10 * math.log10(1 / entry["mse"]),
                             lr=lr, minutes=(time.time() - started) / 60)
                history.append(entry)
                print(f"  [{lam} c{chunk}] step {step:6d}  {entry['psnr']:.2f} dB  "
                      f"{entry['bpp']:.4f} bpp  {entry['minutes']:.1f} min", flush=True)
                running, seen = dict.fromkeys(running, 0.0), 0
            if step % args.save_every == 0:
                checkpoint(step)
            if step >= args.steps_per_chunk:
                break
    checkpoint(step)
    torch.save({"model_state_dict": model.state_dict(), "config": model.config_dict(),
                "lambda": lam, "warm_start": str(intra_path), "unroll": args.unroll,
                "chunks_done_through": chunk, "data": "vimeo90k"}, out / "model.pt")
    _save_json(out / "history.json", history)


def command_train_vimeo(args) -> None:
    import shutil

    from nvc.utils.device import get_device
    chunks = _load_script("train_vimeo_qat_combined")
    device = get_device()
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.json"
    progress = (json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file()
                else {"done": []})
    for chunk in args.chunks:
        todo = [lam for lam in args.lambdas if [lam, chunk] not in progress["done"]]
        if not todo:
            print(f"[chunk {chunk}] already trained for {args.lambdas}, skipping", flush=True)
            continue
        scratch = Path(args.scratch_dir) / f"chunk_{chunk}"
        if args.local_vimeo:
            group_root = Path(args.local_vimeo)
        else:
            group_root = chunks.download_and_extract_chunk(
                chunk, scratch, dataset_owner=args.kaggle_owner, dataset_prefix=args.kaggle_prefix)
        clip_dirs = complete_clip_dirs(group_root)
        if args.max_clips:
            clip_dirs = clip_dirs[:args.max_clips]
        for lam in todo:
            train_lambda_on_chunk(lam, chunk, clip_dirs, args, device)
            progress["done"].append([lam, chunk])
            _save_json(progress_path, progress)
        if not args.local_vimeo:
            shutil.rmtree(scratch, ignore_errors=True)
    print("train-vimeo complete:", progress["done"], flush=True)


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
def evaluate_point(lam: str, device, models_dir: Path = OUTPUT_DIR) -> dict[str, dict[str, float]]:
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
    path = models_dir / f"lambda{lam}" / "model.pt"
    if path.is_file():
        saved = torch.load(path, map_location="cpu")
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
    models_dir = Path(args.models_dir)
    print(f"learned models from {models_dir}")
    points = {lam: evaluate_point(lam, device, models_dir) for lam in LAMBDAS
              if (CACHE_DIR / f"val_b_{lam}.pt").is_file()}
    curves = {arm: [(p[arm]["bits"], p[arm]["psnr"]) for p in points.values() if arm in p]
              for arm in ("intra", "hand_built", "learned")}
    report = {"points": points, "gate_ratio": GATE_RATIO, "models_dir": str(models_dir),
              "comparisons": {}}
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
    print(f"GATE ({'PASSED' if report['gate_passed'] else 'NOT PASSED'}): learned P-frame "
          f"<= {GATE_RATIO}x intra at matched PSNR at every measured point: {measured}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / args.output_name).write_text(json.dumps(report, indent=2), encoding="utf-8")


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
    gate = sub.add_parser("gate")
    gate.add_argument("--models-dir", default=str(OUTPUT_DIR),
                      help="folder holding lambda<L>/model.pt (a Colab run's output folder)")
    gate.add_argument("--output-name", default="gate1.json")
    vimeo = sub.add_parser("train-vimeo", help="gate 2: train on Vimeo-90K chunks (Colab)")
    vimeo.add_argument("--output-dir", required=True)
    vimeo.add_argument("--scratch-dir", required=True)
    vimeo.add_argument("--lambdas", nargs="+", choices=LAMBDAS, default=list(LAMBDAS))
    vimeo.add_argument("--chunks", nargs="+", type=int, required=True)
    vimeo.add_argument("--intra-checkpoint", default=CHECKPOINT,
                       help="path pattern with {} for the lambda tag")
    vimeo.add_argument("--init-from", default=None,
                       help="folder with lambda<L>/model.pt to start from (stage 2 starts from stage 1)")
    vimeo.add_argument("--unroll", type=int, default=1, help="P-frames per clip: 1 = stage 1")
    vimeo.add_argument("--steps-per-chunk", type=int, default=15000)
    vimeo.add_argument("--batch", type=int, default=8)
    vimeo.add_argument("--crop", type=int, default=256)
    vimeo.add_argument("--lr", type=float, default=1e-4)
    vimeo.add_argument("--lr-decay-chunks", type=int, default=1,
                       help="the last N chunks of --chunks train at lr/10")
    vimeo.add_argument("--seed", type=int, default=42)
    vimeo.add_argument("--num-workers", type=int, default=2)
    vimeo.add_argument("--log-every", type=int, default=250)
    vimeo.add_argument("--save-every", type=int, default=1000)
    vimeo.add_argument("--kaggle-owner", default="wangsally")
    vimeo.add_argument("--kaggle-prefix", default="vimeo-90k")
    vimeo.add_argument("--local-vimeo", default=None,
                       help="use this folder of <group>/<clip>/im*.png instead of downloading")
    vimeo.add_argument("--max-clips", type=int, default=0, help="smoke tests only")
    args = parser.parse_args(argv)
    {"cache": command_cache, "train": command_train, "gate": command_gate,
     "train-vimeo": command_train_vimeo}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
