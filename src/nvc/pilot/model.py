"""A small conditional video codec for the budget pilot (docs/PLAN_budget_pilot.md).

One network codes both kinds of frame. A P-frame's temporal context is the previous frame's
decoded features at 1/8 resolution (no motion vectors); an I-frame gets an all-zero context,
so the I-frame decoder is the same network and inside the same budget.

    analysis    RGB pixel-unshuffled by 8 (192 ch at 1/8) + context -> depthwise-separable
                blocks at 1/8 -> y, `latent_channels` at 1/16
    hyperprior  y -> z at 1/64 (zero-mean Laplace per channel) -> hyper features at 1/16
    prior       hyper features + context (at 1/16) -> mean and scale of y, in two checkerboard
                steps (the second also sees the decoded anchor half)
    synthesis   y_hat at 1/16 -> 1/8, fused with the context -> blocks -> features F_t (the next
                frame's context) -> RGB by pixel shuffle

Rates: one model for every rate. Level l of `levels` has a learned per-channel step vector
q_l. The prior predicts (mean, b) of y; the coded symbols are round(y/q - mean/q), a Laplace
of scale b/q in unit bins, and y_hat = q * (mean/q + symbols). Training samples l uniformly
and minimises MSE + lambda_l * bpp, lambda_l log-spaced from lambda_max (l = 0, lowest
quality) to lambda_min (highest quality) - the repository's MSE + lambda * bpp convention.

All costs are counted with `count_macs` (PyTorch's FLOP counter, MAC = FLOP / 2).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

SCALE_FLOOR = 0.11          # as the current codec (p1_pframe.SCALE_FLOOR)
LIKELIHOOD_FLOOR = 1e-9
PAD = 64                    # 1/64 hyperprior: sides are padded to a multiple of 64
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "pilot_budget.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    data = json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def laplace_bits(centred: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """-log2 of a zero-mean Laplace(scale)'s mass on [v - 1/2, v + 1/2] (p1_pframe.laplace_bits,
    the mass the range coder's tables are built from)."""
    a = centred.abs()
    lower, upper = a - 0.5, a + 0.5
    tail = 0.5 * (torch.exp(-lower.clamp_min(0) / scale) - torch.exp(-upper / scale))
    middle = 1 - 0.5 * (torch.exp(-upper / scale) + torch.exp(lower.clamp_max(0) / scale))
    probability = torch.where(lower >= 0, tail, middle)
    return -torch.log2(probability.clamp_min(LIKELIHOOD_FLOOR))


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


class DSBlock(nn.Module):
    """Residual depthwise-separable block: depthwise 3x3, 1x1 expand, LeakyReLU, 1x1 project."""

    def __init__(self, channels: int, expand: int = 2) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels * expand, 1)
        self.pw2 = nn.Conv2d(channels * expand, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw2(F.leaky_relu(self.pw1(self.dw(x)), 0.1))


def checkerboard(height: int, width: int, device) -> torch.Tensor:
    """1 at anchor positions ((i + j) even), coded first; [1, 1, h, w]."""
    i = torch.arange(height, device=device).view(-1, 1)
    j = torch.arange(width, device=device).view(1, -1)
    return ((i + j) % 2 == 0).float().view(1, 1, height, width)


class PilotCodec(nn.Module):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = dict(config or load_config())
        self.config = cfg
        c, cy, cz = cfg["feature_channels"], cfg["latent_channels"], cfg["hyper_channels"]
        ch, cp, ce = cfg["hyper_features"], cfg["prior_channels"], cfg["encoder_channels"]
        e = cfg["expand"]
        blocks = lambda n, w: nn.Sequential(*[DSBlock(w, e) for _ in range(n)])  # noqa: E731

        # encoder only
        self.enc_in = nn.Conv2d(192 + c, ce, 1)
        self.enc_blocks = blocks(cfg["encoder_blocks"], ce)
        self.enc_down = nn.Conv2d(4 * ce, cy, 1)
        self.enc_out = DSBlock(cy, e)
        self.hyper_enc = nn.Sequential(DSBlock(cy, e), nn.PixelUnshuffle(4),
                                       nn.Conv2d(16 * cy, cz, 1))
        # decoder (the encoder runs these too)
        self.hyper_dec = nn.Sequential(nn.Conv2d(cz, 16 * ch, 1), nn.PixelShuffle(4),
                                       DSBlock(ch, e))
        self.temporal = blocks(cfg["context_blocks"], c)
        self.ctx_down = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(4 * c, ch, 1))
        self.prior_a = nn.Sequential(nn.Conv2d(2 * ch, cp, 1), DSBlock(cp, e), DSBlock(cp, e))
        self.head_a = nn.Conv2d(cp, 2 * cy, 1)
        self.anchor_dw = nn.Conv2d(cy, cy, 3, padding=1, groups=cy)
        self.prior_b = nn.Sequential(nn.Conv2d(cp + cy, cp, 1), DSBlock(cp, e))
        self.head_b = nn.Conv2d(cp, 2 * cy, 1)
        self.syn_up = nn.Sequential(nn.Conv2d(cy, 4 * c, 1), nn.PixelShuffle(2))
        self.syn_fuse = nn.Conv2d(2 * c, c, 1)
        self.syn_blocks = blocks(cfg["synthesis_blocks"], c)
        self.syn_out = nn.Sequential(nn.Conv2d(c, 192, 1), nn.PixelShuffle(8))
        # rates
        levels = cfg["levels"]
        lam = torch.exp(torch.linspace(math.log(cfg["lambda_max"]), math.log(cfg["lambda_min"]),
                                       levels))
        self.register_buffer("lambdas", lam, persistent=False)
        # initial steps ~ sqrt(lambda): coarser at low quality
        init = 0.5 * torch.log(lam / lam.median())
        self.log_q = nn.Parameter(init.view(-1, 1).repeat(1, cy))
        self.log_gain = nn.Parameter(torch.zeros(levels, cy))
        self.log_z_scale = nn.Parameter(torch.zeros(cz))
        self.channels = c

    # ------------------------------------------------------------------ shared pieces

    def empty_context(self, frame: torch.Tensor) -> torch.Tensor:
        n, _, h, w = frame.shape
        return frame.new_zeros(n, self.channels, h // 8, w // 8)

    def side(self, z_hat: torch.Tensor, f_prev: torch.Tensor):
        """Decoder: hyper features, the 1/8 context and its 1/16 version."""
        ctx = self.temporal(f_prev)
        return self.hyper_dec(z_hat), ctx, self.ctx_down(ctx)

    def step_a(self, hyper: torch.Tensor, ctx16: torch.Tensor):
        feats = self.prior_a(torch.cat([hyper, ctx16], 1))
        mean, log_b = self.head_a(feats).chunk(2, 1)
        return feats, mean, log_b

    def step_b(self, feats_a: torch.Tensor, y_anchor: torch.Tensor):
        mean, log_b = self.head_b(self.prior_b(torch.cat([feats_a, self.anchor_dw(y_anchor)],
                                                         1))).chunk(2, 1)
        return mean, log_b

    def synthesize(self, y_hat: torch.Tensor, ctx: torch.Tensor, level: torch.Tensor):
        gain = self.log_gain[level].exp()[..., None, None]
        f = self.syn_blocks(self.syn_fuse(torch.cat([self.syn_up(y_hat * gain), ctx], 1)))
        return self.syn_out(f), f

    def step_sizes(self, level: torch.Tensor) -> torch.Tensor:
        return self.log_q[level].exp()[..., None, None]

    def analysis(self, frame: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        h = self.enc_blocks(self.enc_in(torch.cat([F.pixel_unshuffle(frame, 8), ctx], 1)))
        return self.enc_out(self.enc_down(F.pixel_unshuffle(h, 2)))

    def z_scale(self) -> torch.Tensor:
        return self.log_z_scale.exp().clamp_min(SCALE_FLOOR)[None, :, None, None]

    # ------------------------------------------------------------------ training

    def forward(self, frame: torch.Tensor, f_prev: torch.Tensor | None, level: torch.Tensor):
        """One frame, training mode (noise for the rate, straight-through rounding for the
        reconstruction). Returns reconstruction, features F_t, and bits per sample [N]."""
        if f_prev is None:
            f_prev = self.empty_context(frame)
        ctx = self.temporal(f_prev)
        y = self.analysis(frame, ctx)
        z = self.hyper_enc(y)
        if self.training:
            z_rate = z + torch.empty_like(z).uniform_(-0.5, 0.5)
            z_hat = ste_round(z)
        else:
            z_rate = z_hat = torch.round(z)
        z_bits = laplace_bits(z_rate.float(), self.z_scale()).sum((1, 2, 3))
        hyper = self.hyper_dec(z_hat)
        feats, mean_a, log_b_a = self.step_a(hyper, self.ctx_down(ctx))
        q = self.step_sizes(level)
        anchor = checkerboard(y.shape[2], y.shape[3], y.device)
        ys = y / q

        def quantize(mean):
            m = mean / q
            return m + (ste_round(ys - m) if self.training else torch.round(ys - m)), m

        ys_hat_a, _ = quantize(mean_a)
        mean_b, log_b_b = self.step_b(feats, ys_hat_a * q * anchor)
        mean = torch.where(anchor.bool(), mean_a, mean_b)
        log_b = torch.where(anchor.bool(), log_b_a, log_b_b)
        ys_hat, m = quantize(mean)
        b = (log_b.float().exp() / q.float()).clamp_min(SCALE_FLOOR)
        centred = (ys - m).float()
        if self.training:
            centred = centred + torch.empty_like(centred).uniform_(-0.5, 0.5)
        else:
            centred = torch.round(centred)
        y_bits = laplace_bits(centred, b).sum((1, 2, 3))
        x_hat, f = self.synthesize(ys_hat * q, ctx, level)
        return x_hat, f, y_bits + z_bits

    # ------------------------------------------------------------------ coding

    @torch.no_grad()
    def encode(self, frame: torch.Tensor, f_prev: torch.Tensor | None, level: int):
        """Exact encode of one frame (batch 1): symbols, the scales the decoder will derive,
        the reconstruction and next-frame features. Frame sides must divide by 64."""
        if f_prev is None:
            f_prev = self.empty_context(frame)
        lv = torch.tensor([level], device=frame.device)
        ctx = self.temporal(f_prev)
        y = self.analysis(frame, ctx)
        z_sym = torch.round(self.hyper_enc(y))
        hyper, _, ctx16 = self.hyper_dec(z_sym), ctx, self.ctx_down(ctx)
        feats, mean_a, log_b_a = self.step_a(hyper, ctx16)
        q = self.step_sizes(lv)
        anchor = checkerboard(y.shape[2], y.shape[3], y.device).bool()
        ys = y / q
        sym_a = torch.round(ys - mean_a / q)
        ys_hat_a = mean_a / q + sym_a
        mean_b, log_b_b = self.step_b(feats, ys_hat_a * q * anchor)
        sym_b = torch.round(ys - mean_b / q)
        ys_hat = torch.where(anchor, ys_hat_a, mean_b / q + sym_b)
        x_hat, f = self.synthesize(ys_hat * q, ctx, lv)
        return {"z": z_sym, "z_scale": self.z_scale().expand_as(z_sym),
                "anchor": anchor, "sym_a": sym_a, "scale_a": self._scale(log_b_a, q),
                "sym_b": sym_b, "scale_b": self._scale(log_b_b, q),
                "x_hat": x_hat.clamp(0, 1), "features": f}

    @staticmethod
    def _scale(log_b: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return (log_b.exp() / q).clamp_min(SCALE_FLOOR)

    @torch.no_grad()
    def decode(self, read: Callable, f_prev: torch.Tensor | None, level: int, shape):
        """Exact decode of one frame. `read(kind, scale, mask)` returns the symbols of `kind`
        ('z', 'a' anchors, 'b' the rest) given their scales; `shape` = (1, 3, H, W).
        The networks run here are the decoder's whole per-frame cost."""
        n, _, h, w = shape
        device = self.log_q.device
        if f_prev is None:
            f_prev = torch.zeros(n, self.channels, h // 8, w // 8, device=device)
        lv = torch.tensor([level], device=device)
        cz = self.log_z_scale.numel()
        z_scale = self.z_scale().expand(n, cz, h // 64, w // 64)
        z_sym = read("z", z_scale, None)
        hyper, ctx, ctx16 = self.side(z_sym, f_prev)
        feats, mean_a, log_b_a = self.step_a(hyper, ctx16)
        q = self.step_sizes(lv)
        anchor = checkerboard(h // 16, w // 16, device).bool()
        sym_a = read("a", self._scale(log_b_a, q), anchor)
        ys_hat_a = mean_a / q + sym_a
        mean_b, log_b_b = self.step_b(feats, ys_hat_a * q * anchor)
        sym_b = read("b", self._scale(log_b_b, q), ~anchor)
        ys_hat = torch.where(anchor, ys_hat_a, mean_b / q + sym_b)
        x_hat, f = self.synthesize(ys_hat * q, ctx, lv)
        return x_hat.clamp(0, 1), f

    @staticmethod
    def ideal_bits(coded: dict[str, torch.Tensor]) -> float:
        """Ideal code length of an `encode` result: z, anchors and the rest, each under the
        exact Laplace the decoder derives (no table snapping)."""
        anchor = coded["anchor"]
        bits = laplace_bits(coded["z"], coded["z_scale"]).sum()
        bits += (laplace_bits(coded["sym_a"], coded["scale_a"]) * anchor).sum()
        bits += (laplace_bits(coded["sym_b"], coded["scale_b"]) * ~anchor).sum()
        return float(bits)


def pad_to(frame: torch.Tensor, multiple: int = PAD) -> torch.Tensor:
    h, w = frame.shape[-2:]
    ph, pw = -h % multiple, -w % multiple
    return F.pad(frame, (0, pw, 0, ph), mode="replicate") if ph or pw else frame


def count_macs(fn: Callable[[], Any]) -> int:
    """Multiply-adds of everything `fn` runs: PyTorch's FLOP counter (convolutions and matrix
    products) / 2."""
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as counter:
        fn()
    return counter.get_total_flops() // 2


def decoder_macs_per_pixel(model: PilotCodec, height: int = 1088, width: int = 1920,
                           intra: bool = False) -> float:
    """Decoder MAC per pixel for one frame (I-frame: empty context; P-frame: features)."""
    device = model.log_q.device
    f_prev = None if intra else torch.zeros(1, model.channels, height // 8, width // 8,
                                            device=device)

    def read(kind, scale, mask):
        return torch.zeros_like(scale)

    return count_macs(lambda: model.decode(read, f_prev, 0, (1, 3, height, width))) / (
        height * width)


def encoder_macs_per_pixel(model: PilotCodec, height: int = 1088, width: int = 1920,
                           intra: bool = False) -> float:
    device = model.log_q.device
    frame = torch.rand(1, 3, height, width, device=device)
    f_prev = None if intra else torch.zeros(1, model.channels, height // 8, width // 8,
                                            device=device)
    return count_macs(lambda: model.encode(frame, f_prev, 0)) / (height * width)
