"""Codec bundles: every piece of state one operating point needs, in one file.

The research pipeline never had this. Each milestone script REBUILT its codec
state from training data on every run - grid calibration over 400 TRAIN frames,
M13 table fitting, the M14 motion table - which is fine for experiments and
unusable for a product. A bundle freezes that state once:

    autoencoder weights         the analysis/synthesis transform
    intra grid + tables         I-frame quantizer and per-channel frequency tables
    residual grid               P-frame quantizer (M22's grid, or the M13/M14 one)
    context model weights       M11-G16
    assignment codebook         K=512 prototypes used to ROUTE each position
    coding codebook             M13-recalibrated tables the coder READS
    motion tables               M14's recalibrated dy/dx tables
    identities                  the three 8-byte ids a stream header carries

It is saved with `torch.save` as tensors and plain Python values only, and loaded
with `weights_only=True`, so opening a bundle never executes pickled code.

`verify()` recomputes every identity from the loaded weights and tables and
compares it with the recorded one, and checks the autoencoder's SHA-256. A bundle
whose contents drifted from its record is refused rather than producing streams
that silently fail to decode elsewhere.

Bundles are produced by `scripts/export_codec_bundle.py` from the research state
and proven byte-identical to it by `scripts/verify_promoted_codec.py`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.compression.quantization import QuantizationParams
from nvc.models.autoencoder import BaselineAutoencoder
from nvc.video.entropy import (
    ChannelContextEntropyModel,
    SharedCodebook,
    calibration_signature,
    model_identity,
)
from nvc.video.motion import motion_alphabet_bits

BUNDLE_FORMAT = "nvc-codec-bundle"
BUNDLE_VERSION = 1


class BundleError(ValueError):
    """A bundle that is malformed, unsupported, or does not match its record."""


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    """Order-independent digest of a state dict's names, dtypes, shapes and values."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _as_int64(array) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array, dtype=np.int64))


@dataclass
class CodecBundle:
    """Frozen state for one codec operating point. See the module docstring."""

    name: str
    bits: int
    gop_size: int
    block_size: int
    search_range: int
    calibration_frames: int
    quantization_mode: str
    autoencoder_config: dict[str, Any]
    autoencoder_state: dict[str, torch.Tensor]
    autoencoder_sha256: str
    intra_scale: torch.Tensor
    intra_zero_point: torch.Tensor
    intra_frequencies: torch.Tensor
    residual_scale: torch.Tensor
    residual_zero_point: torch.Tensor
    context_model_config: dict[str, Any]
    context_model_state: dict[str, torch.Tensor]
    assign_codebook_frequencies: torch.Tensor
    assign_codebook_metric: str
    coding_codebook_frequencies: torch.Tensor
    motion_frequencies: torch.Tensor
    m10k_identity: str
    calibration_signature: str
    intra_entropy_model_id: str
    residual_entropy_model_id: str
    motion_entropy_model_id: str
    provenance: dict[str, Any] = field(default_factory=dict)

    # --- derived objects ---------------------------------------------------------------

    def intra_params(self) -> QuantizationParams:
        return QuantizationParams(scale=self.intra_scale.clone(),
                                  zero_point=self.intra_zero_point.clone(),
                                  bits=self.bits, mode=self.quantization_mode)

    def residual_params(self) -> QuantizationParams:
        return QuantizationParams(scale=self.residual_scale.clone(),
                                  zero_point=self.residual_zero_point.clone(),
                                  bits=self.bits, mode=self.quantization_mode)

    def intra_entropy_model(self) -> EmpiricalEntropyModel:
        return EmpiricalEntropyModel(self.intra_frequencies.numpy(), bits=self.bits)

    def motion_entropy_model(self) -> EmpiricalEntropyModel:
        return EmpiricalEntropyModel(self.motion_frequencies.numpy(),
                                     bits=motion_alphabet_bits(self.search_range))

    def assign_codebook(self) -> SharedCodebook:
        return SharedCodebook(self.assign_codebook_frequencies.numpy(), bits=self.bits,
                              metric=self.assign_codebook_metric)

    def coding_codebook(self) -> SharedCodebook:
        return SharedCodebook(self.coding_codebook_frequencies.numpy(), bits=self.bits,
                              metric=self.assign_codebook_metric)

    def context_model(self) -> ChannelContextEntropyModel:
        model = ChannelContextEntropyModel(**self.context_model_config)
        model.load_state_dict(self.context_model_state)
        return model.eval()

    def autoencoder(self) -> BaselineAutoencoder:
        model = BaselineAutoencoder(**self.autoencoder_config)
        model.load_state_dict(self.autoencoder_state)
        return model.eval()

    # --- integrity ---------------------------------------------------------------------

    def computed_identities(self) -> dict[str, str]:
        """Every identity recomputed from the bundle's own contents."""
        return {
            "intra": self.intra_entropy_model().model_id().hex(),
            "motion": self.motion_entropy_model().model_id().hex(),
            "calibration_signature": calibration_signature(
                self.residual_params(), bits=self.bits,
                calibration_frames=self.calibration_frames, quant_mode=self.quantization_mode),
            "residual": model_identity(
                self.context_model(), m10k_identity=bytes.fromhex(self.m10k_identity),
                calibration_signature=self.calibration_signature, bits=self.bits,
                codebook=self.coding_codebook()).hex(),
            "autoencoder_sha256": state_dict_sha256(self.autoencoder_state),
        }

    def verify(self) -> None:
        """Raise `BundleError` unless every recorded identity matches the contents."""
        if self.quantization_mode not in ("per_channel", "global"):
            raise BundleError(f"unknown quantization mode {self.quantization_mode!r}")
        if self.context_model_config.get("alphabet") != 2 ** self.bits:
            raise BundleError(
                f"context model alphabet {self.context_model_config.get('alphabet')} does not "
                f"match {self.bits}-bit symbols")
        if self.context_model_config.get("latent_channels") != self.autoencoder_config.get(
                "latent_channels", 64):
            raise BundleError("context model and autoencoder disagree on latent channels")
        try:
            computed = self.computed_identities()
        except (ValueError, RuntimeError, KeyError) as exc:
            raise BundleError(f"bundle contents are invalid: {exc}") from exc
        recorded = {"intra": self.intra_entropy_model_id, "motion": self.motion_entropy_model_id,
                    "calibration_signature": self.calibration_signature,
                    "residual": self.residual_entropy_model_id,
                    "autoencoder_sha256": self.autoencoder_sha256}
        mismatched = sorted(key for key in recorded if recorded[key] != computed[key])
        if mismatched:
            details = ", ".join(f"{k}: recorded {recorded[k]}, computed {computed[k]}"
                                for k in mismatched)
            raise BundleError(f"bundle '{self.name}' does not match its record ({details})")

    # --- persistence -------------------------------------------------------------------

    def to_payload(self) -> dict[str, Any]:
        payload = {"format": BUNDLE_FORMAT, "version": BUNDLE_VERSION}
        for name in self.__dataclass_fields__:
            payload[name] = getattr(self, name)
        return payload

    def save(self, path: str | Path) -> str:
        """Write the bundle and return the file's SHA-256."""
        self.verify()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), path)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path: str | Path, *, verify: bool = True) -> "CodecBundle":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"codec bundle not found: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:          # a pickle that is not plain data, or not a bundle
            raise BundleError(f"{path} is not a loadable codec bundle: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("format") != BUNDLE_FORMAT:
            raise BundleError(f"{path} is not a codec bundle")
        if payload.get("version") != BUNDLE_VERSION:
            raise BundleError(
                f"unsupported bundle version {payload.get('version')!r}; this build reads "
                f"version {BUNDLE_VERSION}")
        missing = [name for name in cls.__dataclass_fields__ if name not in payload]
        if missing:
            raise BundleError(f"bundle is missing field(s): {', '.join(missing)}")
        bundle = cls(**{name: payload[name] for name in cls.__dataclass_fields__})
        if verify:
            bundle.verify()
        return bundle

    @classmethod
    def build(cls, *, name: str, bits: int, gop_size: int, block_size: int, search_range: int,
              calibration_frames: int, autoencoder: torch.nn.Module,
              intra_params: QuantizationParams, intra_entropy_model: EmpiricalEntropyModel,
              residual_params: QuantizationParams, context_model: torch.nn.Module,
              assign_codebook_frequencies, assign_codebook_metric: str,
              coding_codebook_frequencies, motion_entropy_model: EmpiricalEntropyModel,
              m10k_identity: str, calibration_signature: str,
              residual_entropy_model_id: str,
              provenance: dict[str, Any] | None = None) -> "CodecBundle":
        """Assemble a bundle from live objects (research or package ones)."""
        if intra_params.mode != residual_params.mode:
            raise BundleError("intra and residual grids must share a quantization mode")
        autoencoder_state = {k: v.detach().cpu().clone()
                             for k, v in autoencoder.state_dict().items()}
        bundle = cls(
            name=name, bits=int(bits), gop_size=int(gop_size), block_size=int(block_size),
            search_range=int(search_range), calibration_frames=int(calibration_frames),
            quantization_mode=residual_params.mode,
            autoencoder_config=dict(autoencoder.config_dict()),
            autoencoder_state=autoencoder_state,
            autoencoder_sha256=state_dict_sha256(autoencoder_state),
            intra_scale=intra_params.scale.detach().cpu().clone(),
            intra_zero_point=intra_params.zero_point.detach().cpu().clone(),
            intra_frequencies=_as_int64(intra_entropy_model.frequencies),
            residual_scale=residual_params.scale.detach().cpu().clone(),
            residual_zero_point=residual_params.zero_point.detach().cpu().clone(),
            context_model_config=dict(context_model.config_dict()),
            context_model_state={k: v.detach().cpu().clone()
                                 for k, v in context_model.state_dict().items()},
            assign_codebook_frequencies=_as_int64(assign_codebook_frequencies),
            assign_codebook_metric=assign_codebook_metric,
            coding_codebook_frequencies=_as_int64(coding_codebook_frequencies),
            motion_frequencies=_as_int64(motion_entropy_model.frequencies),
            m10k_identity=m10k_identity, calibration_signature=calibration_signature,
            intra_entropy_model_id=intra_entropy_model.model_id().hex(),
            residual_entropy_model_id=residual_entropy_model_id,
            motion_entropy_model_id=motion_entropy_model.model_id().hex(),
            provenance=dict(provenance or {}))
        bundle.verify()
        return bundle
