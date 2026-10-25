"""M23 - the hyperprior: does a TRANSMITTED description of the current content
buy more bits than it costs?

THE GAP THIS MILESTONE ADDRESSES
--------------------------------
Nothing in this codec, at training time or at coding time, can make a bit's cost
depend on what is actually at that position in this frame:

  training rate signal   `src/nvc/training/rate_estimator.py` - ONE Laplace per
                         latent channel. Content-independent by construction.
  deployed intra coder   `empirical_per_channel_static_v1` - one integer
                         histogram per channel. Content-independent.
  deployed P-residual    M10K + M11-G16 + K=512 + M13 - content-ADAPTIVE, but
                         only on information the decoder already has (`z_ref`
                         and causally-decoded neighbours). It cannot describe
                         the frame being coded.

That is the mechanism the latent-shape diagnostic measured as broken: cost is
decoupled from content (Spearman +0.215 between a channel's energy and its coded
cost; a channel at 1/20 of peak energy still pays ~68% of full bit depth), which
is why 192 channels cost 3.2x the rate for +0.43 dB and why reverse
water-filling recovers 0.00% on the trained 64-channel model. Both are the same
fact: the only adaptivity available is per-CHANNEL, and the problem is
per-ELEMENT.

A hyperprior is the one mechanism that closes it: a small side channel, coded
into the bitstream, from which the decoder recovers a per-element prediction
before decoding the latent. It is worth gating for two reasons, and the second
matters more than the first:

  1 it is content-adaptive where everything we have is not;
  2 it is a DIFFERENTIABLE, content-adaptive rate model, so it can go in the
    training loss. Nothing the project currently owns can: the static table is
    not differentiable, and M10K/M11 are fitted post-hoc. The hyperprior is
    therefore the prerequisite for genuine end-to-end R-D training.

WHAT IS MEASURED, AND WHAT IS NOT
---------------------------------
Three conditioning arms, ONE architecture family, so the comparison is
apples-to-apples and the only thing that moves is what the predictor is allowed
to see:

  zref_only         conditioned on `z_ref` alone. This is M10K's family, and it
                    is the baseline the hyperprior has to beat: the information
                    the decoder gets FREE.
  hyper_only        conditioned on the decoded hyper-latent alone. Answers
                    whether transmitted side info can stand on its own.
  hyper_plus_zref   both. The deployable question.

Every arm emits a categorical distribution over the 2**bits alphabet, exactly as
M10K does, so bits/symbol is directly comparable to every number M10K/M10L/M11
reported rather than being a new estimator.

**The side channel is CHARGED.** The hyper-latent is rounded to integers at step
1.0 (Balle's formulation - the network learns its own scale, and the training
noise width matches the step, so there is no grid-calibration circularity), coded
with a per-channel integer histogram fitted on TRAIN, and its real bits are added
before any arm is compared. Reported per y-symbol, so the denominator is the same
for all arms. A hyperprior that is not charged for its side info is not a
measurement of anything.

**The control.** `permuted_hyper_signal` permutes the signal h_a sees across the
WHOLE split, so the hyper-latent describes a different frame while the target and
`z_ref` stay with their own. A "gain" that survives that is estimator bias or a
better marginal, not content adaptation. Whole-split, never per-frame - a
per-frame shuffle leaks frame-level activity and manufactured a spurious ~0.1%
gain in M12 (see `m11_causal_context.permuted_context`).

WHAT THIS GATE CANNOT SAY
-------------------------
The in-family baseline is `zref_only`, i.e. M10K's conditioning. The DEPLOYED
P-residual stack adds M11-G16's causal context and M13 on top of M10K, worth a
further +5.3-6.0% of coded bytes on this transform. So a gain measured here is a
gain over z_ref conditioning, NOT over the full deployed stack, and whether it
survives alongside M11's causal context is a separate question. The gate reports
the deployed stack's own VAL-B figure beside its arms as context and flags the
joint test for Phase B rather than quietly assuming additivity.

It also measures the coding gain at FROZEN transform weights, which the current
weights were not trained for - they were fitted under a per-channel Laplace rate
term. That makes this a LOWER bound on what joint training would give, which is
the useful direction for a go/no-go.

HARD SCOPE
----------
Frozen: the transform weights, lambda, GOP, motion estimator, motion coding, the
intra and residual quantizer grids, the `.nvct` format, the range coder, the
TRAIN/VAL/TEST split. Nothing under `src/nvc/` is modified. No earlier
milestone's script is rewritten; they are imported read-only through
`_load_script`. TEST is never opened by this module.

Thresholds are the project's pre-registered ones: <0.5% net = weak, 0.5-1.0% =
marginal, >1.0% = meaningful.
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.utils.seed import seed_everything

DEFAULT_OUTPUT_DIR = Path("outputs/m23_hyperprior")
DEFAULT_CHECKPOINT = Path("outputs/stage1_vimeo/colab_run4_64ch_qat/best.pt")
DEFAULT_M10K_DIR = Path("outputs/stage1_refit/m10k_64ch_qat")
DEFAULT_M11_DIR = Path("outputs/stage1_refit/m11_64ch_qat")
RATE_POINTS = (5, 4, 3)
DEFAULT_GOP = 10
DEFAULT_BLOCK_SIZE = 16
DEFAULT_SEARCH_RANGE = 16

# One architecture family, three things the predictor is allowed to see.
ARMS = ("zref_only", "hyper_only", "hyper_plus_zref")
HYPER_ARMS = ("hyper_only", "hyper_plus_zref")

# The project's pre-registered gate bars (scripts/m11_offline_gate.py).
WEAK_BELOW_PERCENT = 0.5
MEANINGFUL_ABOVE_PERCENT = 1.0

# The hyper-latent is rounded at step 1.0 and coded on a bounded alphabet. 8 bits
# centred on zero covers [-128, 127]; the clamped fraction is always reported so
# a badly-scaled hyper-latent is visible rather than silently distorted.
DEFAULT_HYPER_BITS = 8
DEFAULT_HYPER_CHANNELS = 8
DEFAULT_HYPER_HIDDEN = 32
DEFAULT_HIDDEN = 32


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verdict(gain_percent: float) -> str:
    if gain_percent < WEAK_BELOW_PERCENT:
        return "weak"
    if gain_percent < MEANINGFUL_ABOVE_PERCENT:
        return "marginal"
    return "meaningful"


# --------------------------------------------------------------------------------------------
# The hyper-latent: analysis, coding, and its honest bit cost
# --------------------------------------------------------------------------------------------


class HyperAnalysis(nn.Module):
    """h_a: the signal being coded -> a small, spatially coarse hyper-latent.

    Two stride-2 convolutions, so the hyper-latent is 1/16th the positions of the
    signal, times `hyper_channels` instead of `latent_channels`. At the
    benchmark's 16x16 latent and 64 channels that is 128 values against 16,384
    symbols - 0.78% of the positions, which is what makes a side channel
    affordable at all.

    Unlike the per-channel-shared stack in `ConditionalEntropyModel` below, this
    sees every channel at once: the whole point of a hyperprior is to describe
    structure that a per-channel view cannot.
    """

    def __init__(self, latent_channels: int = 64, hyper_channels: int = DEFAULT_HYPER_CHANNELS,
                 hidden: int = DEFAULT_HYPER_HIDDEN) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.hyper_channels = hyper_channels
        self.hidden = hidden
        self.body = nn.Sequential(
            nn.Conv2d(latent_channels, hidden, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hyper_channels, 3, padding=1),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, hyper_channels, ceil(H/4), ceil(W/4)]."""
        if signal.shape[1] != self.latent_channels:
            raise ValueError(
                f"expected {self.latent_channels} channels, got {signal.shape[1]}")
        return self.body(signal)


class HyperRatePrior(nn.Module):
    """A differentiable per-hyper-channel Laplace density, for the TRAINING rate
    term on the hyper-latent only.

    Laplace and per-channel for the same reasons `src/nvc/training/
    rate_estimator.py` gives: this project's latents are sharply peaked with long
    tails, and one distribution per channel mirrors the deployed table layout.
    It is a training proxy and nothing else - every REPORTED side-info cost comes
    from `hyper_latent_bits` below, which reads the real integer histogram the
    real coder would use. The gate reports both so the gap between proxy and
    coder is visible instead of assumed.
    """

    def __init__(self, hyper_channels: int = DEFAULT_HYPER_CHANNELS) -> None:
        super().__init__()
        self.location = nn.Parameter(torch.zeros(hyper_channels))
        self.log_scale = nn.Parameter(torch.zeros(hyper_channels))

    def bits(self, z: torch.Tensor) -> torch.Tensor:
        """Expected bits to code `z` at unit step, summed over all elements."""
        scale = F.softplus(self.log_scale).clamp_min(1e-6)[None, :, None, None]
        location = self.location[None, :, None, None]
        # Probability MASS of the unit-width bin centred on z, under Laplace.
        upper = z + 0.5
        lower = z - 0.5
        cdf_upper = _laplace_cdf(upper, location, scale)
        cdf_lower = _laplace_cdf(lower, location, scale)
        mass = (cdf_upper - cdf_lower).clamp_min(1e-9)
        return -torch.log2(mass).sum()


def _laplace_cdf(x: torch.Tensor, location: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    centred = x - location
    return torch.where(
        centred < 0,
        0.5 * torch.exp(centred / scale),
        1.0 - 0.5 * torch.exp(-centred / scale),
    )


def hyper_symbol_offset(bits: int) -> int:
    """The hyper-latent is signed; the coder's alphabet is [0, 2**bits)."""
    return 1 << (bits - 1)


def quantize_hyper_latent(z: torch.Tensor, *, bits: int) -> tuple[np.ndarray, torch.Tensor,
                                                                  float]:
    """Round to integers at step 1.0 and map onto the coder's alphabet.

    Returns (symbols, dequantized, clamped_fraction). The clamped fraction is
    returned rather than warned about because it is a number the gate must
    report: a hyper-latent that needs clamping is one whose learned scale does
    not fit the alphabet, and that would quietly change what is being measured.
    """
    offset = hyper_symbol_offset(bits)
    rounded = torch.round(z)
    clamped = rounded.clamp(-offset, offset - 1)
    fraction = float((clamped != rounded).float().mean())
    symbols = (clamped + offset).to(torch.int64).cpu().numpy()
    return symbols, clamped, fraction


def fit_hyper_entropy_model(symbols: np.ndarray, *, bits: int,
                            hyper_channels: int) -> EmpiricalEntropyModel:
    """One integer histogram per hyper-channel, by the SAME function and the same
    Laplace smoothing `calibrate_grids` already uses for every other table in
    this codec (`EmpiricalEntropyModel.from_symbols`). No new smoothing scheme.

    `symbols` is [N, hyper_channels, h, w]. `from_symbols` already takes axis 1
    as the table axis and flattens the rest itself, so it is handed the array
    unchanged rather than pre-transposed.
    """
    return EmpiricalEntropyModel.from_symbols(
        np.asarray(symbols), bits=bits, num_tables=hyper_channels)


def hyper_latent_bits(entropy_model: EmpiricalEntropyModel, symbols: np.ndarray, *,
                      hyper_channels: int) -> float:
    """Total REAL bits to code one or more hyper-latents under `entropy_model`.

    -log2 of the table probability of each symbol, summed. This is the cost the
    range coder would actually pay (to within its documented 0.026% overhead),
    not a density estimate.
    """
    flat = np.asarray(symbols).reshape(-1, hyper_channels, int(np.prod(symbols.shape[2:])))
    frequencies = np.asarray(entropy_model.frequencies, dtype=np.float64)
    totals = frequencies.sum(axis=1, keepdims=True)
    probabilities = frequencies / totals
    total = 0.0
    for channel in range(hyper_channels):
        picked = probabilities[channel][flat[:, channel, :]]
        total += float(-np.log2(picked).sum())
    return total


# --------------------------------------------------------------------------------------------
# The predictor: one family, three things it may see
# --------------------------------------------------------------------------------------------


class ConditionalEntropyModel(nn.Module):
    """P(symbol | what this arm is allowed to see, channel), per position.

    The `z_ref` path is deliberately M10K's: the same per-channel-shared 3x3
    stack, the same learned per-channel embedding, the same 1x1 head onto the
    alphabet, so the `zref_only` arm really is M10K's family and a difference
    between arms is the conditioning, not the architecture.

    The hyper path adds one term: the upsampled hyper-latent features, broadcast
    across channels and summed into the same hidden representation. Additive and
    broadcast, so it carries per-POSITION information that no per-channel table
    can - which is the whole hypothesis under test.
    """

    def __init__(self, *, arm: str, latent_channels: int = 64, alphabet: int = 16,
                 hidden: int = DEFAULT_HIDDEN,
                 hyper_channels: int = DEFAULT_HYPER_CHANNELS,
                 hyper_hidden: int = DEFAULT_HYPER_HIDDEN) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.arm = arm
        self.latent_channels = latent_channels
        self.alphabet = alphabet
        self.hidden = hidden
        self.hyper_channels = hyper_channels
        self.hyper_hidden = hyper_hidden
        self.uses_reference = arm in ("zref_only", "hyper_plus_zref")
        self.uses_hyper = arm in HYPER_ARMS

        if self.uses_reference:
            self.features = nn.Sequential(
                nn.Conv2d(1, hidden, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
            )
        if self.uses_hyper:
            self.hyper_synthesis = nn.Sequential(
                nn.Conv2d(hyper_channels, hyper_hidden, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(hyper_hidden, hidden, 3, padding=1),
            )
        self.channel_embedding = nn.Embedding(latent_channels, hidden)
        self.head = nn.Conv2d(hidden, alphabet, 1)
        nn.init.zeros_(self.channel_embedding.weight)

    def forward(self, *, reference: torch.Tensor | None = None,
                hyper: torch.Tensor | None = None,
                shape: tuple[int, int, int, int] | None = None) -> torch.Tensor:
        """-> [B, C, alphabet, H, W] logits.

        `reference` is [B, C, H, W]; `hyper` is the DEQUANTIZED hyper-latent
        [B, hyper_channels, h, w]. `shape` is required only when this arm has no
        reference to take the output geometry from.
        """
        if self.uses_reference:
            if reference is None:
                raise ValueError(f"arm {self.arm!r} needs a reference")
            batch, channels, height, width = reference.shape
        else:
            if shape is None:
                raise ValueError(f"arm {self.arm!r} needs an explicit output shape")
            batch, channels, height, width = shape
        if channels != self.latent_channels:
            raise ValueError(
                f"expected {self.latent_channels} latent channels, got {channels}")

        if self.uses_reference:
            stacked = reference.reshape(batch * channels, 1, height, width)
            features = self.features(stacked)
        else:
            weight = self.head.weight
            features = torch.zeros((batch * channels, self.hidden, height, width),
                                   device=weight.device, dtype=weight.dtype)

        if self.uses_hyper:
            if hyper is None:
                raise ValueError(f"arm {self.arm!r} needs a hyper-latent")
            # Upsample to the symbol grid, then broadcast the SAME map to every
            # channel: the side channel describes positions, not channels.
            upsampled = F.interpolate(hyper, size=(height, width), mode="nearest")
            hyper_features = self.hyper_synthesis(upsampled)
            features = features + hyper_features.repeat_interleave(channels, dim=0)

        embedding = self.channel_embedding.weight.repeat(batch, 1)
        features = features + embedding[:, :, None, None]
        logits = self.head(features)
        return logits.reshape(batch, channels, self.alphabet, height, width)

    def log_probabilities(self, **kwargs) -> torch.Tensor:
        return F.log_softmax(self.forward(**kwargs), dim=2)

    def config_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "latent_channels": self.latent_channels,
                "alphabet": self.alphabet, "hidden": self.hidden,
                "hyper_channels": self.hyper_channels, "hyper_hidden": self.hyper_hidden}


class HyperpriorModel(nn.Module):
    """One arm, end to end: h_a (if any), its rate prior, and the predictor."""

    def __init__(self, *, arm: str, latent_channels: int = 64, alphabet: int = 16,
                 hidden: int = DEFAULT_HIDDEN,
                 hyper_channels: int = DEFAULT_HYPER_CHANNELS,
                 hyper_hidden: int = DEFAULT_HYPER_HIDDEN) -> None:
        super().__init__()
        self.arm = arm
        self.predictor = ConditionalEntropyModel(
            arm=arm, latent_channels=latent_channels, alphabet=alphabet, hidden=hidden,
            hyper_channels=hyper_channels, hyper_hidden=hyper_hidden)
        self.uses_hyper = self.predictor.uses_hyper
        if self.uses_hyper:
            self.analysis = HyperAnalysis(latent_channels=latent_channels,
                                          hyper_channels=hyper_channels, hidden=hyper_hidden)
            self.rate_prior = HyperRatePrior(hyper_channels=hyper_channels)

    def config_dict(self) -> dict[str, Any]:
        return self.predictor.config_dict()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(arm: str, config: dict[str, Any] | None = None) -> HyperpriorModel:
    settings = dict(config or {})
    settings.pop("arm", None)
    return HyperpriorModel(arm=arm, **settings)


def load_model(path, *, device=None) -> tuple[HyperpriorModel, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
    config = dict(checkpoint["model_config"])
    model = build_model(config["arm"], config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device or "cpu").eval(), checkpoint


# --------------------------------------------------------------------------------------------
# The control
# --------------------------------------------------------------------------------------------


def permuted_hyper_signal(count: int, *, seed: int) -> np.ndarray:
    """A WHOLE-SPLIT permutation of the frames h_a sees.

    The target symbols and `z_ref` stay with their own frame; only the signal the
    hyper-latent is computed from moves. So the side channel describes a
    different frame's content, at the same cost, and any surviving "gain" is
    estimator bias or a better marginal rather than content adaptation.

    Whole-split, not per-frame: a per-frame shuffle leaks frame-level activity
    and manufactured a spurious ~0.1% gain in M12. See
    `m11_causal_context.permuted_context`.

    A fixed point would leak the true signal for that frame, so the permutation
    is derangement-corrected by rotating any element that maps to itself.
    """
    generator = np.random.default_rng(seed)
    order = generator.permutation(count)
    if count > 1:
        for index in range(count):
            if order[index] == index:
                partner = (index + 1) % count
                order[index], order[partner] = order[partner], order[index]
    return order


# --------------------------------------------------------------------------------------------
# Training and evaluation - bits/symbol, with the side channel charged
# --------------------------------------------------------------------------------------------


def _as_tensor(array, dtype) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(dtype)
    return torch.from_numpy(np.asarray(array)).to(dtype)


def _forward_arm(model: HyperpriorModel, *, signal: torch.Tensor | None,
                 reference: torch.Tensor | None, shape: tuple[int, int, int, int],
                 hyper_bits: int, noisy: bool) -> tuple[torch.Tensor, torch.Tensor | None,
                                                        torch.Tensor | None]:
    """-> (log probabilities, hyper-latent before rounding, dequantized or noised).

    During training the hyper-latent is perturbed by U(-0.5, 0.5) instead of
    rounded, which is Balle's standard differentiable stand-in and matches the
    unit step the real coder uses - the same reasoning
    `src/nvc/training/quantization_noise.py` documents for the latent.
    """
    hyper_raw = hyper_used = None
    if model.uses_hyper:
        if signal is None:
            raise ValueError(f"arm {model.arm!r} needs a signal for h_a")
        hyper_raw = model.analysis(signal)
        if noisy:
            noise = torch.empty_like(hyper_raw).uniform_(-0.5, 0.5)
            hyper_used = hyper_raw + noise
        else:
            _, clamped, _ = quantize_hyper_latent(hyper_raw.detach(), bits=hyper_bits)
            hyper_used = clamped.to(hyper_raw.device)
    log_probabilities = model.predictor.log_probabilities(
        reference=reference, hyper=hyper_used, shape=shape)
    return log_probabilities, hyper_raw, hyper_used


def train_arm(model: HyperpriorModel, train_batch, select_batch, *, epochs: int,
              batch_size: int, learning_rate: float, device, hyper_bits: int,
              seed: int = 42, log=print) -> list[dict[str, Any]]:
    """Minimise (y bits/symbol + side bits/symbol). Pure rate, nothing else.

    The side term is in the loss deliberately: without it h_a is free to emit an
    arbitrarily expensive hyper-latent and the "gain" would be an artefact of
    unpriced side information. The training side term uses the differentiable
    Laplace proxy; every reported number uses the real histogram.

    Best-validation weights are kept and restored, which is the M10G convention -
    reporting a selected epoch while measuring the final weights is the exact
    defect M10I hit.
    """
    seed_everything(seed)
    model = model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_symbols, train_references, train_signals = train_batch
    select_symbols, select_references, select_signals = select_batch

    def epoch_bits(symbols, references, signals, *, training: bool) -> float:
        total_bits, total_symbols = 0.0, 0
        count = symbols.shape[0]
        order = torch.randperm(count) if training else torch.arange(count)
        for start in range(0, count, batch_size):
            index = order[start:start + batch_size]
            target = symbols[index].to(device)
            reference = (references[index].to(device)
                         if model.predictor.uses_reference else None)
            signal = signals[index].to(device) if model.uses_hyper else None
            shape = (target.shape[0], target.shape[1], *_spatial(target, reference, signal))
            target_map = target.reshape(shape)

            log_probabilities, hyper_raw, _ = _forward_arm(
                model, signal=signal, reference=reference, shape=shape,
                hyper_bits=hyper_bits, noisy=training)
            picked = log_probabilities.gather(2, target_map.unsqueeze(2)).squeeze(2)
            y_bits = -picked.sum() / math.log(2.0)
            side_bits = (model.rate_prior.bits(hyper_raw) if model.uses_hyper
                         else torch.zeros((), device=device))
            symbols_here = target_map.numel()
            loss = (y_bits + side_bits) / symbols_here

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_bits += float((y_bits + side_bits).detach())
            total_symbols += symbols_here
        return total_bits / total_symbols

    history: list[dict[str, Any]] = []
    best_state, best_bits = None, float("inf")
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        train_bits = epoch_bits(train_symbols, train_references, train_signals, training=True)
        model.eval()
        with torch.no_grad():
            select_bits = epoch_bits(select_symbols, select_references, select_signals,
                                     training=False)
        history.append({"epoch": epoch, "train_bits": train_bits,
                        "val_a_bits": select_bits, "val_loss": select_bits,
                        "elapsed_seconds": time.time() - started})
        if select_bits < best_bits:
            best_bits = select_bits
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"        [epoch {epoch:>3}] train {train_bits:.5f}  VAL-A {select_bits:.5f} "
            f"bits/symbol  {history[-1]['elapsed_seconds']:.0f}s")
    model.load_state_dict(best_state)
    return history


def _spatial(target: torch.Tensor, reference: torch.Tensor | None,
             signal: torch.Tensor | None) -> tuple[int, int]:
    """The [H, W] the symbol map should take.

    Symbols arrive flattened per channel ([N, C, H*W], the coder's own order), so
    the geometry comes from whichever conditioning tensor is present; with
    neither, a square latent is the only consistent reading and a non-square one
    is refused rather than guessed.
    """
    for tensor in (reference, signal):
        if tensor is not None:
            return int(tensor.shape[-2]), int(tensor.shape[-1])
    positions = int(target.shape[-1])
    side = int(round(math.sqrt(positions)))
    if side * side != positions:
        raise ValueError(
            f"cannot infer the latent geometry for {positions} positions with no "
            "conditioning tensor to read it from; pass a reference or a signal")
    return side, side


@torch.no_grad()
def evaluate_arm(model: HyperpriorModel, symbols, references, signals, *, device,
                 hyper_bits: int, hyper_entropy_model: EmpiricalEntropyModel | None = None,
                 batch_size: int = 16) -> dict[str, Any]:
    """Held-out bits/symbol with the side channel charged at its REAL cost.

    Returns the three numbers the gate compares - `y_bits_per_symbol`,
    `side_bits_per_symbol`, `net_bits_per_symbol` - plus the proxy's own estimate
    of the side cost and the clamped fraction, so a hyper-latent that does not
    fit its alphabet cannot pass unnoticed.

    `hyper_entropy_model` must be the table fitted on TRAIN (see
    `collect_hyper_symbols`); passing None fits nothing and charges nothing,
    which is only valid for an arm that transmits nothing.
    """
    model = model.to(device).eval()
    total_y_bits, total_symbols = 0.0, 0
    total_side_bits, total_proxy_bits, total_hyper_values = 0.0, 0.0, 0
    clamped_total = 0.0
    count = symbols.shape[0]
    for start in range(0, count, batch_size):
        index = torch.arange(start, min(start + batch_size, count))
        target = symbols[index].to(device)
        reference = references[index].to(device) if model.predictor.uses_reference else None
        signal = signals[index].to(device) if model.uses_hyper else None
        shape = (target.shape[0], target.shape[1], *_spatial(target, reference, signal))
        target_map = target.reshape(shape)

        log_probabilities, hyper_raw, _ = _forward_arm(
            model, signal=signal, reference=reference, shape=shape,
            hyper_bits=hyper_bits, noisy=False)
        picked = log_probabilities.gather(2, target_map.unsqueeze(2)).squeeze(2)
        total_y_bits += float(-picked.sum()) / math.log(2.0)
        total_symbols += target_map.numel()

        if model.uses_hyper:
            hyper_symbols, _, clamped_fraction = quantize_hyper_latent(
                hyper_raw, bits=hyper_bits)
            if hyper_entropy_model is None:
                raise ValueError(
                    f"arm {model.arm!r} transmits a hyper-latent; its TRAIN-fitted "
                    "entropy model is required so the side channel can be charged")
            total_side_bits += hyper_latent_bits(
                hyper_entropy_model, hyper_symbols,
                hyper_channels=model.predictor.hyper_channels)
            total_proxy_bits += float(model.rate_prior.bits(hyper_raw))
            clamped_total += clamped_fraction * hyper_raw.numel()
            total_hyper_values += hyper_raw.numel()

    y_per_symbol = total_y_bits / total_symbols
    side_per_symbol = total_side_bits / total_symbols
    return {
        "arm": model.arm,
        "y_bits_per_symbol": y_per_symbol,
        "side_bits_per_symbol": side_per_symbol,
        "net_bits_per_symbol": y_per_symbol + side_per_symbol,
        "side_bits_total": total_side_bits,
        "side_proxy_bits_total": total_proxy_bits,
        "side_share_of_net": (side_per_symbol / (y_per_symbol + side_per_symbol)
                              if y_per_symbol + side_per_symbol > 0 else 0.0),
        "hyper_values_transmitted": total_hyper_values,
        "hyper_clamped_fraction": (clamped_total / total_hyper_values
                                   if total_hyper_values else 0.0),
        "symbols_scored": total_symbols,
    }


@torch.no_grad()
def collect_hyper_symbols(model: HyperpriorModel, signals, *, device, hyper_bits: int,
                          batch_size: int = 16) -> tuple[np.ndarray, float]:
    """The TRAIN hyper-latent symbols its coding table is fitted from.

    Fitted on TRAIN and only TRAIN, like every other table in this codec: a table
    fitted on the split it is scored against would make the side channel look
    free.
    """
    model = model.to(device).eval()
    if not model.uses_hyper:
        raise ValueError(f"arm {model.arm!r} transmits nothing")
    collected, clamped_total, values = [], 0.0, 0
    for start in range(0, signals.shape[0], batch_size):
        signal = signals[start:start + batch_size].to(device)
        hyper_raw = model.analysis(signal)
        symbols, _, clamped_fraction = quantize_hyper_latent(hyper_raw, bits=hyper_bits)
        collected.append(symbols)
        clamped_total += clamped_fraction * hyper_raw.numel()
        values += hyper_raw.numel()
    stacked = np.concatenate(collected, axis=0)
    return stacked, (clamped_total / values if values else 0.0)


# --------------------------------------------------------------------------------------------
# Signals: what h_a sees, for each of the two channels of the stream
# --------------------------------------------------------------------------------------------


def dequantized_residual(symbols: np.ndarray, residual_params, *, shape) -> torch.Tensor:
    """The residual as the DECODER reconstructs it, which is what h_a is given.

    Deliberately the dequantized residual rather than the continuous one: the
    continuous residual would give h_a strictly more information than this
    measurement can promise a real encoder reuses, so this is the conservative
    choice and biases the gate DOWN rather than up.
    """
    from nvc.compression.codec import symbols_to_latent
    frames = []
    for index in range(symbols.shape[0]):
        frames.append(symbols_to_latent(symbols[index].reshape(-1), shape, residual_params))
    return torch.cat(frames, dim=0)


def intra_signal_and_symbols(model, sequences, *, intra_params, max_frames: int,
                             device) -> tuple[np.ndarray, torch.Tensor]:
    """I-frame-path latents: the symbols to be coded, and the dequantized latent
    h_a sees.

    Every frame's latent, I- and P-typed alike, matching `calibrate_grids`' own
    intra loop and `m14_recalibration.collect_intra_symbols` - the intra table
    covers the FULL latent distribution, not just I-typed positions, so a signal
    collected any other way would not describe the table it is compared against.
    """
    from nvc.compression.codec import latent_to_symbols, symbols_to_latent
    model.eval()
    symbol_frames, signal_frames = [], []
    seen = 0
    with torch.no_grad():
        for sequence in sequences:
            frames = sequence.load_frames()
            for index in range(frames.shape[0]):
                if seen >= max_frames:
                    break
                latent = model.encode(frames[index:index + 1].to(device))
                channels, height, width = latent.shape[1], latent.shape[2], latent.shape[3]
                flat = latent_to_symbols(latent, intra_params)
                symbol_frames.append(flat.reshape(channels, -1))
                signal_frames.append(
                    symbols_to_latent(flat, (channels, height, width), intra_params))
                seen += 1
            if seen >= max_frames:
                break
    return np.stack(symbol_frames), torch.cat(signal_frames, dim=0)
