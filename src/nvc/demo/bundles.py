"""Which codec the demo runs: trained bundles when present, a stand-in otherwise.

Trained bundles (`outputs/codec_bundles/*.pt`, from `scripts/export_codec_bundle.py`)
are git-ignored, so a fresh clone has none. Without one the demo would have
nothing to run, so `build_standin_bundle` assembles a working codec from what a
clone CAN have: an autoencoder checkpoint and some training frames.

A stand-in is a real, lossless-entropy-coded, decodable codec, and its pictures
come from a trained autoencoder. It is NOT the project's codec: the context
model is untrained, and every codebook table is the residual symbols' overall
histogram, so nothing is predicted from context. Its bitrate is therefore not a
project result, and every stand-in is labelled as one wherever it is shown.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.codec import latent_to_symbols, symbols_to_latent
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.video import entropy as ventropy
from nvc.video.bundle import BundleError, CodecBundle
from nvc.video.motion import (
    deterministic_kernels,
    estimate_block_motion,
    motion_alphabet_bits,
    motion_to_symbols,
    warp_blocks,
)

TRAINED_DIR = Path("outputs/codec_bundles")
STANDIN_DIR = Path("outputs/demo/bundles")
STANDIN_KIND = "standin"
TRAINED_KIND = "trained"

STANDIN_GOP, STANDIN_BLOCK, STANDIN_RANGE = 10, 16, 16      # the deployed codec's structure
STANDIN_GROUP, STANDIN_CODEBOOK = 16, 4
STANDIN_NOTE = ("Stand-in codec: a trained autoencoder with an untrained context model and "
                "a single shared residual table. Its bitrate is not a project result.")


@dataclass(frozen=True)
class BundleInfo:
    """One selectable codec, as the page lists it."""

    id: str
    label: str
    kind: str                      # "trained" or "standin"
    bits: int
    path: str
    gop_size: int
    block_size: int
    search_range: int
    note: str = ""

    def to_json(self) -> dict:
        data = asdict(self)
        del data["path"]
        return data


def _label(bundle: CodecBundle, kind: str, stem: str) -> str:
    if kind == STANDIN_KIND:
        return f"Stand-in, {bundle.bits}-bit"
    names = (str(bundle.provenance.get("arm", "")), bundle.name, stem)
    family = "M22" if any("m22" in name.lower() for name in names) else "Deployed"
    return f"{family}, {bundle.bits}-bit"


def describe(path: Path, kind: str) -> BundleInfo:
    """Load (and verify) one bundle and summarise it. Raises `BundleError` if it
    is not a bundle or does not match its own record."""
    bundle = CodecBundle.load(path)
    return BundleInfo(
        id=path.stem, label=_label(bundle, kind, path.stem), kind=kind, bits=bundle.bits,
        path=str(path),
        gop_size=bundle.gop_size, block_size=bundle.block_size, search_range=bundle.search_range,
        note=STANDIN_NOTE if kind == STANDIN_KIND else "")


def bundle_files(repo_root: str | Path) -> list[tuple[Path, str]]:
    """(path, kind) for every `.pt` in the trained and stand-in bundle folders."""
    repo_root = Path(repo_root)
    files: list[tuple[Path, str]] = []
    for directory, kind in ((repo_root / TRAINED_DIR, TRAINED_KIND),
                            (repo_root / STANDIN_DIR, STANDIN_KIND)):
        if directory.is_dir():
            files.extend((path, kind) for path in sorted(directory.glob("*.pt")))
    return files


def sort_bundles(bundles: Sequence[BundleInfo]) -> list[BundleInfo]:
    """Trained before stand-in, M22 before deployed, then by bits."""
    return sorted(bundles, key=lambda b: (b.kind != TRAINED_KIND,
                                          not b.label.startswith("M22"), b.bits))


def discover_bundles(repo_root: str | Path) -> list[BundleInfo]:
    """Every usable bundle, in display order.

    A file that fails to load or verify is skipped rather than failing the demo -
    a half-copied bundle should not take the page down.
    """
    found: list[BundleInfo] = []
    for path, kind in bundle_files(repo_root):
        try:
            found.append(describe(path, kind))
        except (BundleError, OSError):
            continue
    return sort_bundles(found)


def default_bundle(bundles: Sequence[BundleInfo]) -> BundleInfo | None:
    """The 4-bit point of the best family present (the scoreboard's middle point)."""
    if not bundles:
        return None
    best_kind, best_family = bundles[0].kind, bundles[0].label.split(",")[0]
    family = [b for b in bundles if b.kind == best_kind and b.label.split(",")[0] == best_family]
    return next((b for b in family if b.bits == 4), family[0])


# --- the stand-in ----------------------------------------------------------------------


def _marginal_table(symbols: np.ndarray, *, bits: int, tables: int) -> np.ndarray:
    """`tables` identical frequency tables, each the histogram of all `symbols`."""
    flat = np.asarray(symbols, dtype=np.int64).reshape(1, 1, -1)
    return EmpiricalEntropyModel.from_symbols(
        np.repeat(flat, tables, axis=1), bits=bits, num_tables=tables).frequencies


@torch.no_grad()
def build_standin_bundle(autoencoder: torch.nn.Module, sequences: Sequence[torch.Tensor], *,
                         bits: int, name: str | None = None, gop_size: int = STANDIN_GOP,
                         block_size: int = STANDIN_BLOCK, search_range: int = STANDIN_RANGE,
                         group_size: int = STANDIN_GROUP, codebook_size: int = STANDIN_CODEBOOK,
                         provenance: dict | None = None) -> CodecBundle:
    """Assemble a stand-in bundle from an autoencoder and calibration sequences.

    `sequences` are [N, 3, H, W] float tensors in [0, 1] of consecutive frames
    (N >= 2). Grids and tables are fitted the way the codec will use them: the
    intra grid on the frames' latents, and the residual grid and tables on the
    motion-compensated residual against the previous frame's own quantized
    reconstruction.
    """
    if not sequences or any(s.dim() != 4 or s.shape[0] < 2 for s in sequences):
        raise ValueError("need at least one sequence of two or more frames")
    # Building modules draws from the global RNG (the context model below, and every
    # model `CodecBundle.build` constructs to verify itself). The caller's stream is
    # put back afterwards so building a bundle is not a hidden reseed.
    rng_state = torch.random.get_rng_state()
    try:
        return _build_standin(autoencoder, sequences, bits=bits, name=name, gop_size=gop_size,
                              block_size=block_size, search_range=search_range,
                              group_size=group_size, codebook_size=codebook_size,
                              provenance=provenance)
    finally:
        torch.random.set_rng_state(rng_state)


def _build_standin(autoencoder, sequences, *, bits, name, gop_size, block_size, search_range,
                   group_size, codebook_size, provenance) -> CodecBundle:
    autoencoder = autoencoder.eval()
    channels = int(autoencoder.config_dict().get("latent_channels", 64))
    if channels % group_size:
        group_size = 1

    with deterministic_kernels():
        latents = [autoencoder.encode(sequence) for sequence in sequences]
        intra_params = calibrate_quantization_params(torch.cat(latents), bits=bits, mode="per_channel")
        intra_symbols = np.stack([
            latent_to_symbols(latent[i:i + 1], intra_params).reshape(channels, -1)
            for latent in latents for i in range(latent.shape[0])])
        intra_model = EmpiricalEntropyModel.from_symbols(intra_symbols, bits=bits, num_tables=channels)

        residuals, motion_symbols = [], []
        for sequence, latent in zip(sequences, latents):
            shape = tuple(latent.shape[1:])
            for i in range(1, sequence.shape[0]):
                previous = autoencoder.decode(symbols_to_latent(
                    latent_to_symbols(latent[i - 1:i], intra_params), shape, intra_params)
                    .to(latent.device))
                motion = estimate_block_motion(previous, sequence[i:i + 1], block_size=block_size,
                                               search_range=search_range)
                warped = warp_blocks(previous, motion, block_size=block_size)
                residuals.append(latent[i:i + 1] - autoencoder.encode(warped))
                motion_symbols.append(motion_to_symbols(motion, search_range=search_range)
                                      .reshape(2, -1))
        residual_params = calibrate_quantization_params(torch.cat(residuals), bits=bits,
                                                        mode="per_channel")
        residual_symbols = np.concatenate([latent_to_symbols(r, residual_params) for r in residuals])

    motion_model = EmpiricalEntropyModel.from_symbols(
        np.stack(motion_symbols), bits=motion_alphabet_bits(search_range), num_tables=2)
    codebook = _marginal_table(residual_symbols, bits=bits, tables=codebook_size)

    torch.manual_seed(0)                        # the stand-in must rebuild to the same bytes
    context = ventropy.ChannelContextEntropyModel(channels, 2 ** bits, hidden=32,
                                                  group_size=group_size).eval()

    calibration_frames = sum(int(s.shape[0]) for s in sequences)
    signature = ventropy.calibration_signature(residual_params, bits=bits,
                                               calibration_frames=calibration_frames)
    identity = ventropy.model_identity(
        context, m10k_identity=b"\x00" * 8, calibration_signature=signature, bits=bits,
        codebook=ventropy.SharedCodebook(codebook, bits=bits))
    details = {"kind": STANDIN_KIND, "note": STANDIN_NOTE}
    details.update(provenance or {})
    return CodecBundle.build(
        name=name or f"standin_{bits}bit", bits=bits, gop_size=gop_size, block_size=block_size,
        search_range=search_range, calibration_frames=calibration_frames, autoencoder=autoencoder,
        intra_params=intra_params, intra_entropy_model=intra_model,
        residual_params=residual_params, context_model=context,
        assign_codebook_frequencies=codebook, assign_codebook_metric="code_length",
        coding_codebook_frequencies=codebook, motion_entropy_model=motion_model,
        m10k_identity="00" * 8, calibration_signature=signature,
        residual_entropy_model_id=identity.hex(), provenance=details)
