"""M11 - channel-autoregressive residual entropy model.

WHAT IT PREDICTS
----------------
    P(R[c, y, x] | z_ref, channel c, residual symbols of earlier channels)

The offline gate found that already-decoded residual symbols carry information
z_ref cannot: the residual is `latent - E(warp(prev))`, so where motion
compensation FAILS the residual is large in every channel at once, and z_ref -
which describes the reference, not the error - has no way to know where that
is. Once some channels are decoded, their activity at a position is a good
estimate of exactly that.

WHY CHANNEL CONTEXT ONLY
------------------------
The gate also found spatial context (left / up) informative, and the best
discrete context combined both. This model deliberately uses only EARLIER
CHANNELS, because of a constraint measured in the audit: the existing coder's
`rc_decode` is stateless and batch-only. Channel context decodes one channel
(or channel group) per step - every position of a channel predicted in
parallel - bit-exact through the unmodified coder with zero rate overhead. A
context reading the same channel's neighbours needs a table per SYMBOL, i.e.
~16,384 prefix decodes: O(N^2), seconds per frame. The spatial upside is
reported as the measured cost of that constraint.

Every position of every earlier channel is available (C-major coding order),
including positions spatially "ahead" of the current one, so the 3x3
convolutions may read the previous channel's full neighbourhood.

CHANNEL GROUPS - THE LATENCY KNOB
---------------------------------
Measured: one decode step costs ~0.7 ms whether it processes 1 channel or all
64 - it is kernel-launch bound - so decode time is set by the NUMBER of
sequential steps, not by the work per step. `group_size` G trades context for
steps: channels are decoded G at a time, and channel c may use only channels
before its own group, start(c) = G * floor(c / G). G = 1 is full channel
autoregression (64 steps); G = 64 is one step with no context at all - exactly
the no-context ablation.

ARCHITECTURE
------------
M10K's network with three extra input planes per channel sample:

    [z_ref[c], prev = |R[c - G]|, activity = mean over channels < start(c),
     available = (start(c) > 0)]
      -> conv 4->32 (3x3) -> ReLU -> conv 32->32 (3x3) -> ReLU
      + learned channel embedding -> conv 32->alphabet (1x1)

Magnitudes are clipped at 3 and scaled to [0, 1], measured from each channel's
own zero symbol. The model is warm-started from M10K with the new planes'
weights at ZERO, so at step 0 it is exactly M10K and any gain is attributable
to training from there. The no-context ablation trains the same network with
the planes held at zero, separating what context contributes from what
continued training alone would.

ENCODER / DECODER AGREEMENT
---------------------------
The encoder knows every symbol, so it runs ONE forward pass over all channel
samples. The decoder runs one pass per group, building the planes from the
channels decoded so far (later channels held at their zero symbol) and keeping
that group's samples. A sample's input depends only on channels before its
group, so it is identical on both sides; every pass has the same batch shape,
so the same kernels run and each sample's arithmetic is independent of the
others. Tests assert both properties and a bit-exact round trip.

Probabilities reach the coder either as one table per position (as M10K) or
through a shared codebook (as M10L); the codebook path swaps per-frame table
construction for one prototype assignment per group.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nvc.compression.entropy_model import TOTAL_FREQUENCY, EmpiricalEntropyModel
from nvc.compression.range_coder import decode_symbols, encode_symbols
from nvc.utils.seed import seed_everything

CONTEXT_PLANES = 3
MAGNITUDE_CLIP = 3
MODEL_VERSION = 1
MODEL_NAME = "m11_channel_autoregressive_v1"
CONTEXT_DEFINITION = {
    "family": "channel", "version": 2,
    "planes": ["prev_group_magnitude", "running_channel_activity", "available"],
    "magnitude_clip": MAGNITUDE_CLIP, "scan_order": "C-major raster",
}


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context_definition_id(group_size: int = 1) -> str:
    """Identity of the context DEFINITION, group size included - a model trained
    for one grouping must not be run under another."""
    payload = dict(CONTEXT_DEFINITION, group_size=int(group_size))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# --- causal context planes ----------------------------------------------------------


def group_starts(channels: int, group_size: int) -> torch.Tensor:
    """First channel of each channel's decoding group: G * floor(c / G)."""
    if group_size < 1 or channels % group_size:
        raise ValueError(f"group_size {group_size} must divide {channels} channels")
    return (torch.arange(channels) // group_size) * group_size


def context_planes(symbols: torch.Tensor, zero: torch.Tensor,
                   group_size: int = 1) -> torch.Tensor:
    """[B, C, H, W] symbols -> [B, C, 3, H, W] causal planes for every channel.

    Planes for channel c use ONLY channels before start(c) = G * floor(c / G):
      prev      clipped |R[c - G]| / 3       (0 in the first group)
      activity  mean of that over every channel before start(c)  (0 in group 0)
      available 1 outside the first group - so "nothing decoded yet" is never
                confused with "everything decoded so far was quiet".
    With G = 1, prev is |R[c-1]| and activity covers every c' < c.
    """
    magnitude = (symbols - zero.view(1, -1, 1, 1)).abs().clamp(max=MAGNITUDE_CLIP).float()
    magnitude = magnitude / MAGNITUDE_CLIP
    channels = magnitude.shape[1]
    starts = group_starts(channels, group_size).to(magnitude.device)
    prev = torch.zeros_like(magnitude)
    prev[:, group_size:] = magnitude[:, :-group_size]
    cumulative = torch.cat([torch.zeros_like(magnitude[:, :1]), magnitude.cumsum(dim=1)], dim=1)
    earlier = cumulative[:, starts]                      # sum over channels < start(c)
    counts = starts.to(magnitude.dtype)
    activity = earlier / counts.clamp(min=1.0).view(1, -1, 1, 1)
    available = (counts > 0).to(magnitude.dtype).view(1, -1, 1, 1).expand_as(magnitude)
    return torch.stack([prev, activity, available], dim=2)


# --- the model ----------------------------------------------------------------------


class ChannelContextEntropyModel(nn.Module):
    """M10K plus causal channel-context planes. See the module docstring."""

    def __init__(self, latent_channels: int = 64, alphabet: int = 16, hidden: int = 32,
                 use_context: bool = True, group_size: int = 1) -> None:
        super().__init__()
        group_starts(latent_channels, group_size)          # validates divisibility
        self.latent_channels = latent_channels
        self.alphabet = alphabet
        self.hidden = hidden
        self.use_context = use_context
        self.group_size = group_size
        self.features = nn.Sequential(
            nn.Conv2d(1 + CONTEXT_PLANES, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.channel_embedding = nn.Embedding(latent_channels, hidden)
        self.head = nn.Conv2d(hidden, alphabet, 1)

    def planes(self, symbols: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        return context_planes(symbols, zero, self.group_size)

    def forward(self, reference: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
        """reference [B, C, H, W], planes [B, C, 3, H, W] -> logits [B, C, A, H, W]."""
        batch, channels, height, width = reference.shape
        if channels != self.latent_channels:
            raise ValueError(f"expected {self.latent_channels} channels, got {channels}")
        if not self.use_context:
            planes = torch.zeros_like(planes)
        stacked = torch.cat([reference.unsqueeze(2), planes], dim=2).reshape(
            batch * channels, 1 + CONTEXT_PLANES, height, width)
        features = self.features(stacked)
        embedding = self.channel_embedding.weight.repeat(batch, 1)
        features = features + embedding[:, :, None, None]
        logits = self.head(features)
        return logits.reshape(batch, channels, self.alphabet, height, width)

    def log_probabilities(self, reference, planes):
        return F.log_softmax(self.forward(reference, planes), dim=2)

    def config_dict(self) -> dict[str, Any]:
        return {"latent_channels": self.latent_channels, "alphabet": self.alphabet,
                "hidden": self.hidden, "use_context": self.use_context,
                "group_size": self.group_size}


def from_m10k(m10k_model, *, use_context: bool = True,
              group_size: int = 1) -> ChannelContextEntropyModel:
    """Warm start: M10K's weights, with the new context inputs' weights at zero.

    At initialisation the output is therefore EXACTLY M10K's (asserted in the
    tests), so the model starts from the strongest current baseline rather than
    re-learning what z_ref already provides.
    """
    model = ChannelContextEntropyModel(m10k_model.latent_channels, m10k_model.alphabet,
                                       m10k_model.hidden, use_context=use_context,
                                       group_size=group_size)
    source = m10k_model.state_dict()
    target = model.state_dict()
    for key, value in source.items():
        if key == "features.0.weight":
            widened = torch.zeros_like(target[key])
            widened[:, :1] = value
            target[key] = widened
        else:
            target[key] = value.clone()
    model.load_state_dict(target)
    return model


def build_model(config: dict[str, Any]) -> ChannelContextEntropyModel:
    return ChannelContextEntropyModel(**config)


def load_model(path, *, device=None):
    checkpoint = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, checkpoint


def model_identity(model: ChannelContextEntropyModel, *, m10k_identity: bytes,
                   calibration_signature: str, bits: int, codebook=None) -> bytes:
    """8-byte identity for the .nvct v2 residual-entropy-model field.

    Binds the weights, the context DEFINITION (group size included), the M10K
    model it was warm-started from, the quantization calibration, the bit depth
    and - when one is used - the codebook's tables. Any mismatch is then a stream
    identity failure rather than a silent rate loss.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps({
        "name": MODEL_NAME, "version": MODEL_VERSION, "bits": bits,
        "context": context_definition_id(model.group_size),
        "use_context": model.use_context,
        "m10k": bytes(m10k_identity).hex(), "calibration": calibration_signature,
    }, sort_keys=True).encode())
    state = model.state_dict()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().numpy().tobytes())
    if codebook is not None:
        digest.update(np.ascontiguousarray(codebook.frequencies).tobytes())
    return digest.digest()[:8]


# --- training -------------------------------------------------------------------------


def _batches(count: int, batch_size: int, *, shuffle: bool, generator=None):
    order = torch.randperm(count, generator=generator) if shuffle else torch.arange(count)
    for start in range(0, count, batch_size):
        yield order[start:start + batch_size]


def nll_bits(model, symbols, references, zero, *, device, batch_size: int = 16,
             optimizer=None, generator=None) -> float:
    """Mean -log2 P(true symbol) over a set; trains when an optimizer is given.

    Training uses the TRUE earlier-channel symbols. That is not teacher forcing
    in the usual approximate sense: coding is lossless, so the decoder holds
    exactly these symbols when it predicts a later group.
    """
    total, count = 0.0, 0
    zero_t = zero.to(device)
    for index in _batches(symbols.shape[0], batch_size, shuffle=optimizer is not None,
                          generator=generator):
        target = symbols[index].to(device)
        reference = references[index].to(device)
        log_probabilities = model.log_probabilities(reference, model.planes(target, zero_t))
        picked = log_probabilities.gather(2, target.unsqueeze(2)).squeeze(2)
        loss = -picked.mean()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) / math.log(2.0) * target.numel()
        count += target.numel()
    return total / count


def train(model, train_set, select_set, zero, *, epochs: int, batch_size: int,
          learning_rate: float, device, seed: int = 42, log=print) -> list[dict[str, Any]]:
    """Pure rate objective. Best weights by the SELECTION split are restored,
    and epoch 0 (the warm start itself) is a legitimate winner."""
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_symbols, train_references = train_set
    select_symbols, select_references = select_set

    model.eval()
    with torch.no_grad():
        start_bits = nll_bits(model, select_symbols, select_references, zero, device=device)
    history = [{"epoch": 0, "train_nll_bits": None, "val_nll_bits": start_bits,
                "val_loss": start_bits, "rate_enabled": True}]
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_bits = start_bits
    log(f"[EPOCH   0] val {start_bits:.5f} bits/symbol (warm start)")
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        train_bits = nll_bits(model, train_symbols, train_references, zero, device=device,
                              batch_size=batch_size, optimizer=optimizer, generator=generator)
        model.eval()
        with torch.no_grad():
            val_bits = nll_bits(model, select_symbols, select_references, zero, device=device)
        history.append({"epoch": epoch, "train_nll_bits": train_bits, "val_nll_bits": val_bits,
                        "val_loss": val_bits, "rate_enabled": True,
                        "elapsed_seconds": time.time() - started})
        if val_bits < best_bits:
            best_bits = val_bits
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"[EPOCH {epoch:>3}] train {train_bits:.5f}  val {val_bits:.5f} bits/symbol  "
            f"{history[-1]['elapsed_seconds']:.0f}s")
    model.load_state_dict(best_state)
    model.eval()
    return history


# --- per-frame coding ----------------------------------------------------------------


def _rows(log_probabilities: torch.Tensor) -> torch.Tensor:
    """[1, C', A, H, W] log-probs -> [C'*H*W, A] probabilities, C-major like the coder."""
    probabilities = log_probabilities.exp()[0]
    alphabet = probabilities.shape[1]
    return probabilities.permute(0, 2, 3, 1).reshape(-1, alphabet)


def _tables_for(rows: torch.Tensor, codebook, mk):
    """Probabilities -> (cumulative block, table indices) for the coder.

    Per-position: one integer table per row, built here. Codebook: the fixed
    prototype tables, and one assignment per row.
    """
    if codebook is None:
        frequencies = mk.probabilities_to_frequencies(rows.double().cpu().numpy())
        cumulative = np.zeros((frequencies.shape[0], frequencies.shape[1] + 1), dtype=np.int64)
        np.cumsum(frequencies, axis=1, out=cumulative[:, 1:])
        return cumulative, None
    return None, codebook.assign_tensor(rows)


@torch.no_grad()
def encode_frame(model, reference: torch.Tensor, symbols: np.ndarray, zero: torch.Tensor, *,
                 bits: int, codebook=None, timings: dict | None = None) -> tuple[bytes, float]:
    """Encoder side: every symbol is known, so ONE pass predicts every channel."""
    mk = _load_script("m10k_learned_entropy")
    device = reference.device
    started = time.perf_counter()
    target = torch.from_numpy(np.asarray(symbols, dtype=np.int64))[None].to(device)
    rows = _rows(model.log_probabilities(reference, model.planes(target, zero.to(device))))
    if device.type == "cuda":
        torch.cuda.synchronize()
    network = time.perf_counter() - started

    started = time.perf_counter()
    cumulative, assigned = _tables_for(rows, codebook, mk)
    if codebook is None:
        table_index = np.arange(cumulative.shape[0], dtype=np.int64)
    else:
        cumulative, table_index = codebook.cumulative, assigned
    tables = time.perf_counter() - started

    flat = np.asarray(symbols, dtype=np.int64).reshape(-1)
    started = time.perf_counter()
    payload = encode_symbols(flat, cumulative, table_index)
    coder = time.perf_counter() - started
    if timings is not None:
        for key, value in (("network", network), ("tables", tables), ("coder", coder)):
            timings[key] = timings.get(key, 0.0) + value
    totals = cumulative[:, -1] if codebook is None else codebook.cumulative[:, -1]
    lower = cumulative[table_index, flat]
    upper = cumulative[table_index, flat + 1]
    ideal = float(-np.log2((upper - lower) / totals[table_index]).sum())
    return payload, ideal


@torch.no_grad()
def decode_frame(model, payload: bytes, reference: torch.Tensor, zero: torch.Tensor, *,
                 bits: int, shape: tuple[int, int, int], codebook=None,
                 timings: dict | None = None) -> np.ndarray:
    """Decoder side: one step per channel GROUP, through the UNMODIFIED coder.

    At step g the planes are built from the channels decoded so far (later
    channels held at their zero symbol, which cannot reach group g), the network
    runs on the full batch so its kernels match the encoder's, group g's
    distributions become tables, and the coder decodes the first
    (g+1) * G * H * W symbols of the payload. `rc_decode` has no resumable
    state, so each step re-decodes the earlier groups - bit-exact, and measured
    separately under "coder".
    """
    mk = _load_script("m10k_learned_entropy")
    channels, height, width = shape
    plane = height * width
    group = model.group_size
    alphabet = 2 ** bits
    device = reference.device
    zero_d = zero.to(device)
    decoded = zero_d.view(channels, 1, 1).expand(channels, height, width).clone()
    if codebook is None:
        cumulative = np.zeros((channels * plane, alphabet + 1), dtype=np.int64)
        table_index = np.arange(channels * plane, dtype=np.int64)
    else:
        cumulative = codebook.cumulative
        table_index = np.zeros(channels * plane, dtype=np.int64)
    stage = {"network": 0.0, "tables": 0.0, "coder": 0.0}
    symbols = None
    for start in range(0, channels, group):
        stop = start + group
        began = time.perf_counter()
        log_probabilities = model.log_probabilities(reference, model.planes(decoded[None], zero_d))
        rows = _rows(log_probabilities[:, start:stop])
        if device.type == "cuda":
            torch.cuda.synchronize()
        stage["network"] += time.perf_counter() - began

        began = time.perf_counter()
        block, assigned = _tables_for(rows, codebook, mk)
        if codebook is None:
            cumulative[start * plane:stop * plane] = block
        else:
            table_index[start * plane:stop * plane] = assigned
        stage["tables"] += time.perf_counter() - began

        began = time.perf_counter()
        count = stop * plane
        # Per-position tables for later groups are not built yet (all-zero rows,
        # which the coder rightly rejects), so it sees only the filled prefix.
        available = cumulative[:count] if codebook is None else cumulative
        symbols = decode_symbols(payload, count, available, table_index[:count])
        stage["coder"] += time.perf_counter() - began
        decoded[start:stop] = torch.from_numpy(
            symbols[start * plane:count].reshape(group, height, width)).to(device)
    if timings is not None:
        for key, value in stage.items():
            timings[key] = timings.get(key, 0.0) + value
    return symbols
