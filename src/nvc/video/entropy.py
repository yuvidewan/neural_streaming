"""The residual entropy model: causal channel context, shared codebook, coding.

Promoted from three research scripts, inference paths only:

  * `scripts/m11_ar_entropy.py`  - `ChannelContextEntropyModel` (M11-G16), its
    causal context planes, and `model_identity`;
  * `scripts/m10l_shared_codebook.py` - `SharedCodebook`, the K=512 prototype
    tables and the tie-safe prototype assignment;
  * `scripts/m13_recalibration.py` - per-frame coding that ASSIGNS a prototype
    with one codebook and CODES with another (M13's recalibrated tables).

Training and fitting code stays in `scripts/`: the product only needs to load a
fitted model and code with it. Layer names, hashing inputs and identity constants
are unchanged, so a model loaded here produces the same 8-byte stream identity as
the research original - that is what lets streams move between the two.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nvc.compression.codec import latent_to_symbols
from nvc.compression.entropy_model import TOTAL_FREQUENCY, EmpiricalEntropyModel
from nvc.compression.quantization import QuantizationParams
from nvc.compression.range_coder import ResumableDecoder, encode_symbols

CONTEXT_PLANES = 3
MAGNITUDE_CLIP = 3
MODEL_VERSION = 1
MODEL_NAME = "m11_channel_autoregressive_v1"
CONTEXT_DEFINITION = {
    "family": "channel", "version": 2,
    "planes": ["prev_group_magnitude", "running_channel_activity", "available"],
    "magnitude_clip": MAGNITUDE_CLIP, "scan_order": "C-major raster",
}

CODEBOOK_VERSION = 1
CODEBOOK_NAME = "m10l_shared_prototype_v1"
CODEBOOK_METRICS = ("code_length", "kl", "l1")


# --- identities ------------------------------------------------------------------------


def context_definition_id(group_size: int = 1) -> str:
    """Identity of the context definition, group size included."""
    payload = dict(CONTEXT_DEFINITION, group_size=int(group_size))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def calibration_signature(residual_params: QuantizationParams, *, bits: int,
                          calibration_frames: int, quant_mode: str = "per_channel") -> str:
    """Fingerprint of the residual grid a context model and codebook were fitted
    to (`scripts/m10l_evaluate.calibration_signature`, same inputs, same digest)."""
    digest = hashlib.sha256()
    digest.update(np.asarray(residual_params.scale.cpu().numpy(), dtype=np.float64).tobytes())
    digest.update(np.asarray(residual_params.zero_point.cpu().numpy(), dtype=np.float64).tobytes())
    digest.update(json.dumps({"bits": bits, "mode": quant_mode,
                              "calibration_frames": calibration_frames},
                             sort_keys=True).encode())
    return digest.hexdigest()[:16]


def zero_symbols(residual_params: QuantizationParams, channels: int) -> np.ndarray:
    """Per-channel symbol a ZERO residual quantizes to."""
    zeros = torch.zeros(1, channels, 1, 1)
    return latent_to_symbols(zeros, residual_params).reshape(channels).astype(np.int64)


# --- the context model -----------------------------------------------------------------


def group_starts(channels: int, group_size: int) -> torch.Tensor:
    """First channel of each channel's decoding group: G * floor(c / G)."""
    if group_size < 1 or channels % group_size:
        raise ValueError(f"group_size {group_size} must divide {channels} channels")
    return (torch.arange(channels) // group_size) * group_size


def context_planes(symbols: torch.Tensor, zero: torch.Tensor, group_size: int = 1) -> torch.Tensor:
    """[B, C, H, W] symbols -> [B, C, 3, H, W] causal planes. Channel c only sees
    channels before start(c) = G * floor(c / G)."""
    magnitude = (symbols - zero.view(1, -1, 1, 1)).abs().clamp(max=MAGNITUDE_CLIP).float()
    magnitude = magnitude / MAGNITUDE_CLIP
    channels = magnitude.shape[1]
    starts = group_starts(channels, group_size).to(magnitude.device)
    prev = torch.zeros_like(magnitude)
    prev[:, group_size:] = magnitude[:, :-group_size]
    cumulative = torch.cat([torch.zeros_like(magnitude[:, :1]), magnitude.cumsum(dim=1)], dim=1)
    earlier = cumulative[:, starts]
    counts = starts.to(magnitude.dtype)
    activity = earlier / counts.clamp(min=1.0).view(1, -1, 1, 1)
    available = (counts > 0).to(magnitude.dtype).view(1, -1, 1, 1).expand_as(magnitude)
    return torch.stack([prev, activity, available], dim=2)


class ChannelContextEntropyModel(nn.Module):
    """P(residual symbol | reference latent, earlier channel groups).

    Architecture and parameter names match `scripts/m11_ar_entropy.py` exactly:
    the state dict is hashed into the stream identity.
    """

    def __init__(self, latent_channels: int = 64, alphabet: int = 16, hidden: int = 32,
                 use_context: bool = True, group_size: int = 1) -> None:
        super().__init__()
        group_starts(latent_channels, group_size)
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


def model_identity(model: ChannelContextEntropyModel, *, m10k_identity: bytes,
                   calibration_signature: str, bits: int, codebook=None) -> bytes:
    """The 8-byte residual-entropy-model identity carried in the `.nvct` header."""
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


# --- the shared codebook ---------------------------------------------------------------


def _torch_argmin_lowest_index(costs: torch.Tensor) -> torch.Tensor:
    """argmin along dim 1 with ties resolved to the LOWEST index on every backend.
    Encoder and decoder disagreeing on one tie would corrupt the stream."""
    minimum = costs.min(dim=1, keepdim=True).values
    index = costs.argmin(dim=1)
    is_minimum = costs == minimum
    tied = torch.nonzero(is_minimum.sum(dim=1) > 1, as_tuple=False).flatten()
    if tied.numel():
        rows = is_minimum[tied]
        columns = torch.arange(costs.shape[1], device=costs.device).expand_as(rows)
        index[tied] = torch.where(rows, columns, costs.shape[1]).min(dim=1).values
    return index


class SharedCodebook:
    """K prototype distributions, pre-quantized into coder frequency tables."""

    def __init__(self, frequencies: np.ndarray, *, bits: int, metric: str = "code_length") -> None:
        if metric not in CODEBOOK_METRICS:
            raise ValueError(f"unknown metric {metric!r}; expected one of {CODEBOOK_METRICS}")
        # EmpiricalEntropyModel enforces the coder invariants (rows total exactly
        # TOTAL_FREQUENCY, no entry below MIN_FREQUENCY).
        self.entropy_model = EmpiricalEntropyModel(np.asarray(frequencies, dtype=np.int64), bits=bits)
        self.bits = bits
        self.metric = metric
        self.frequencies = self.entropy_model.frequencies
        self.cumulative = self.entropy_model.cumulative
        self.probabilities = self.frequencies / float(TOTAL_FREQUENCY)
        self._log2_costs = -np.log2(self.probabilities)
        self._torch_cache: dict[Any, torch.Tensor] = {}

    @property
    def size(self) -> int:
        return int(self.frequencies.shape[0])

    def assign_tensor(self, probabilities: torch.Tensor) -> np.ndarray:
        """[N, A] predicted distributions -> [N] prototype indices."""
        if self.metric == "l1":
            prototypes = self._torch_prototypes(probabilities, "probabilities")
            costs = (probabilities[:, None, :] - prototypes[None, :, :]).abs().sum(dim=2)
        else:
            costs = probabilities @ self._torch_prototypes(probabilities, "log2").T
        return _torch_argmin_lowest_index(costs).to("cpu").numpy().astype(np.int64)

    def _torch_prototypes(self, like: torch.Tensor, kind: str) -> torch.Tensor:
        key = (kind, str(like.device), like.dtype)
        if key not in self._torch_cache:
            source = self._log2_costs if kind == "log2" else self.probabilities
            self._torch_cache[key] = torch.as_tensor(source, dtype=like.dtype, device=like.device)
        return self._torch_cache[key]

    def codebook_id(self, *, model_identity: bytes = b"", calibration_signature: str = "") -> bytes:
        digest = hashlib.sha256()
        digest.update(json.dumps({
            "name": CODEBOOK_NAME, "version": CODEBOOK_VERSION, "bits": self.bits,
            "size": self.size, "metric": self.metric,
            "model_identity": bytes(model_identity).hex(),
            "calibration_signature": calibration_signature,
        }, sort_keys=True).encode("utf-8"))
        digest.update(self.frequencies.tobytes())
        return digest.digest()[:8]


# --- per-frame residual coding ---------------------------------------------------------


def _rows(log_probabilities: torch.Tensor) -> torch.Tensor:
    """[1, C', A, H, W] log-probs -> [C'*H*W, A] probabilities, C-major like the coder."""
    probabilities = log_probabilities.exp()[0]
    alphabet = probabilities.shape[1]
    return probabilities.permute(0, 2, 3, 1).reshape(-1, alphabet)


@torch.no_grad()
def encode_residual_frame(model: ChannelContextEntropyModel, assign_codebook: SharedCodebook,
                          coding_codebook: SharedCodebook, reference: torch.Tensor,
                          symbols: np.ndarray, zero: torch.Tensor) -> bytes:
    """Code one P-frame's residual symbols.

    The encoder knows every symbol, so one network pass predicts every channel.
    Prototype ASSIGNMENT uses `assign_codebook`; the coder reads the (M13
    recalibrated) `coding_codebook` tables. The two are never swapped.
    """
    device = reference.device
    target = torch.from_numpy(np.asarray(symbols, dtype=np.int64))[None].to(device)
    rows = _rows(model.log_probabilities(reference, model.planes(target, zero.to(device))))
    table_index = assign_codebook.assign_tensor(rows)
    flat = np.asarray(symbols, dtype=np.int64).reshape(-1)
    return encode_symbols(flat, coding_codebook.cumulative, table_index)


@torch.no_grad()
def decode_residual_frame(model: ChannelContextEntropyModel, assign_codebook: SharedCodebook,
                          coding_codebook: SharedCodebook, payload: bytes,
                          reference: torch.Tensor, zero: torch.Tensor, *,
                          shape: tuple[int, int, int]) -> np.ndarray:
    """Inverse of `encode_residual_frame`: one network pass per channel group,
    each decoding only the symbols of that group through a resumable decoder."""
    channels, height, width = shape
    plane = height * width
    group = model.group_size
    device = reference.device
    zero_d = zero.to(device)
    decoded = zero_d.view(channels, 1, 1).expand(channels, height, width).clone()
    symbols = np.empty(channels * plane, dtype=np.int64)

    decoder = ResumableDecoder(payload)
    try:
        for start in range(0, channels, group):
            stop = start + group
            log_probabilities = model.log_probabilities(reference, model.planes(decoded[None], zero_d))
            group_table_index = assign_codebook.assign_tensor(_rows(log_probabilities[:, start:stop]))
            group_symbols = decoder.decode_group(coding_codebook.cumulative, group_table_index)
            symbols[start * plane:stop * plane] = group_symbols
            decoded[start:stop] = torch.from_numpy(
                group_symbols.reshape(group, height, width)).to(device)
    finally:
        decoder.close()
    return symbols
