"""`nvc.video` - the promoted video codec.

The load-bearing tests here are:

  * `test_package_writes_the_same_bytes_as_the_research_codec` - the promotion
    contract. A miniature but structurally complete codec (autoencoder, intra and
    residual grids, G16-style context model, assignment and coding codebooks,
    motion tables) is run through `nvc.video` and through the frozen research path
    (`scripts/m21_refinement.py` at its identity candidate). The streams must be
    byte-identical, and each side must decode the other's stream bit-exactly;
  * `test_identities_match_the_research_definitions` - the 8-byte ids in a stream
    header are computed identically, so streams move between the two;
  * the bundle integrity tests - a bundle whose contents drifted from its record
    is refused.

The full-size proof on DAVIS TEST is `scripts/verify_promoted_codec.py`.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import nvc
from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.codec import latent_to_symbols
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.models.autoencoder import BaselineAutoencoder
from nvc.video import BundleError, CodecBundle, TemporalFormatError, VideoCodec
from nvc.video import entropy as ventropy
from nvc.video.container import (
    TEMPORAL_HEADER_SIZE,
    TemporalStreamHeader,
    TemporalStreamReader,
)
from nvc.video.motion import motion_alphabet_bits

ROOT = Path(__file__).resolve().parents[1]
BITS, GOP, BLOCK, RANGE = 4, 5, 16, 4
LATENT, GROUP, K = 8, 2, 6
SIZE = 32                         # 32x32 frames -> 2x2 latent, 2x2 motion blocks


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frames(count: int = 12, seed: int = 0) -> torch.Tensor:
    """A smooth pattern translating a few pixels per frame, so motion search has
    real work to do and the P-chain is exercised."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.nn.functional.interpolate(torch.rand(1, 3, 8, 8, generator=generator),
                                           size=(SIZE * 2, SIZE * 2), mode="bilinear",
                                           align_corners=False)[0]
    frames = [base[:, 2 * t % SIZE:2 * t % SIZE + SIZE, t % SIZE:t % SIZE + SIZE]
              for t in range(count)]
    return torch.stack(frames).clamp(0, 1).contiguous()


def _table(seed: int, tables: int, bits: int) -> EmpiricalEntropyModel:
    rng = np.random.default_rng(seed)
    symbols = rng.integers(0, 2 ** bits, size=(200, tables))
    return EmpiricalEntropyModel.from_symbols(symbols, bits=bits, num_tables=tables)


def _build_bundle(*, motion_seed: int = 3) -> CodecBundle:
    torch.manual_seed(0)
    autoencoder = BaselineAutoencoder(latent_channels=LATENT, base_channels=4).eval()
    frames = _frames(20, seed=5)
    with torch.no_grad():
        latents = autoencoder.encode(frames)
    intra_params = calibrate_quantization_params(latents, bits=BITS, mode="per_channel")
    intra_symbols = np.stack([latent_to_symbols(latents[i:i + 1], intra_params).reshape(LATENT, -1)
                              for i in range(latents.shape[0])])
    intra_model = EmpiricalEntropyModel.from_symbols(intra_symbols, bits=BITS, num_tables=LATENT)
    residuals = latents[1:] - latents[:-1]
    residual_params = calibrate_quantization_params(residuals, bits=BITS, mode="per_channel")

    torch.manual_seed(1)
    context = ventropy.ChannelContextEntropyModel(LATENT, 2 ** BITS, hidden=8, group_size=GROUP).eval()
    assign = _table(11, K, BITS).frequencies
    coding = _table(12, K, BITS).frequencies
    motion = _table(motion_seed, 2, motion_alphabet_bits(RANGE))
    signature = ventropy.calibration_signature(residual_params, bits=BITS, calibration_frames=20)
    identity = ventropy.model_identity(context, m10k_identity=b"\x00" * 8,
                                       calibration_signature=signature, bits=BITS,
                                       codebook=ventropy.SharedCodebook(coding, bits=BITS))
    return CodecBundle.build(
        name="test", bits=BITS, gop_size=GOP, block_size=BLOCK, search_range=RANGE,
        calibration_frames=20, autoencoder=autoencoder, intra_params=intra_params,
        intra_entropy_model=intra_model, residual_params=residual_params, context_model=context,
        assign_codebook_frequencies=assign, assign_codebook_metric="code_length",
        coding_codebook_frequencies=coding, motion_entropy_model=motion,
        m10k_identity="00" * 8, calibration_signature=signature,
        residual_entropy_model_id=identity.hex(), provenance={"source": "unit test"})


@pytest.fixture(scope="module")
def bundle():
    return _build_bundle()


@pytest.fixture(scope="module")
def codec(bundle):
    return VideoCodec(bundle, device="cpu")


@pytest.fixture(scope="module")
def encoded(codec):
    return codec.encode(_frames(), return_reconstructions=True)


# --- the promotion contract ------------------------------------------------------------


def _research_state(bundle):
    mc = _load_script("m10h_motion_compensation")
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    m13 = _load_script("m13_recalibration")
    m21 = _load_script("m21_refinement")
    cx = _load_script("m11_causal_context")
    model11 = ma.ChannelContextEntropyModel(**bundle.context_model_config)
    model11.load_state_dict(bundle.context_model_state)
    model11.eval()
    residual_params = bundle.residual_params()
    spec = {"model": model11,
            "zero": torch.from_numpy(cx.zero_symbols(residual_params, LATENT)),
            "assign_codebook": ml.SharedCodebook(bundle.assign_codebook_frequencies.numpy(),
                                                 bits=BITS, metric="code_length"),
            "coding_codebook": ml.SharedCodebook(bundle.coding_codebook_frequencies.numpy(),
                                                 bits=BITS, metric="code_length"),
            "identity": bytes.fromhex(bundle.residual_entropy_model_id)}
    return mc, m13, m21, spec


def test_package_writes_the_same_bytes_as_the_research_codec(bundle, codec, encoded, tmp_path):
    mc, m13, m21, spec = _research_state(bundle)
    frames = _frames()
    path = tmp_path / "research.nvct"
    research = m21.encode_sequence_refined(
        mc, m13, bundle.autoencoder(), frames, spec, path, m21.IDENTITY,
        intra_params=bundle.intra_params(), intra_entropy_model=bundle.intra_entropy_model(),
        residual_params=bundle.residual_params(),
        motion_entropy_model=bundle.motion_entropy_model(), bits=BITS, gop_size=GOP,
        block_size=BLOCK, search_range=RANGE)
    stream, reconstructions = encoded

    assert stream.data == path.read_bytes()
    assert torch.equal(reconstructions, research["reconstructions"])
    assert stream.frame_types.count(0) == 3 and stream.frame_types.count(1) == 9

    # Each side decodes the other's stream, bit-exactly.
    assert torch.equal(codec.decode(path), research["reconstructions"])
    decoded = m21.decode_sequence_refined(
        mc, m13, bundle.autoencoder(), stream.data and path, spec, m21.IDENTITY,
        intra_entropy_model=bundle.intra_entropy_model(),
        motion_entropy_model=bundle.motion_entropy_model(), bits=BITS)
    assert torch.equal(decoded["reconstructions"].cpu(), reconstructions)


def test_identities_match_the_research_definitions(bundle):
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    ev = _load_script("m10l_evaluate")
    residual_params = bundle.residual_params()
    model11 = ma.ChannelContextEntropyModel(**bundle.context_model_config)
    model11.load_state_dict(bundle.context_model_state)
    research_coding = ml.SharedCodebook(bundle.coding_codebook_frequencies.numpy(), bits=BITS)
    signature = ev.calibration_signature({"residual_params": residual_params}, bits=BITS,
                                         calibration_frames=20, quant_mode="per_channel")
    assert signature == bundle.calibration_signature
    assert ma.model_identity(model11, m10k_identity=b"\x00" * 8, calibration_signature=signature,
                             bits=BITS, codebook=research_coding).hex() \
        == bundle.residual_entropy_model_id
    assert research_coding.codebook_id() == bundle.coding_codebook().codebook_id()
    assert ma.context_definition_id(GROUP) == ventropy.context_definition_id(GROUP)


def test_header_packs_identically_to_the_research_container(encoded):
    mc = _load_script("m10h_motion_compensation")
    data = encoded[0].data
    ours = TemporalStreamHeader.unpack(data)
    theirs = mc.TemporalStreamHeader.unpack(data)
    assert ours.pack() == theirs.pack() == data[:TEMPORAL_HEADER_SIZE]


# --- round trip and API ----------------------------------------------------------------


def test_decode_reproduces_the_encoder_reconstructions_exactly(codec, encoded):
    stream, reconstructions = encoded
    assert torch.equal(codec.decode(stream.data), reconstructions)


def test_encoding_is_deterministic(codec, encoded):
    assert codec.encode(_frames()).data == encoded[0].data


def test_byte_accounting_closes(encoded):
    stream = encoded[0]
    overhead = TEMPORAL_HEADER_SIZE + 2 * LATENT * 8 + 9 * len(stream.frame_types)
    assert sum(stream.motion_bytes) + sum(stream.residual_bytes) + overhead == stream.total_bytes
    assert all(m == 0 for t, m in zip(stream.frame_types, stream.motion_bytes) if t == 0)


def test_uint8_frames_encode_like_float_frames(codec):
    frames = torch.round(_frames(6) * 255) / 255
    as_uint8 = (frames * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    assert codec.encode(as_uint8).data == codec.encode(frames).data


def test_top_level_encode_and_decode(bundle, codec, encoded, tmp_path):
    path = tmp_path / "bundle.pt"
    bundle.save(path)
    data = nvc.encode(_frames(), path, device="cpu")
    assert data == encoded[0].data
    assert torch.equal(nvc.decode(data, codec), encoded[1])


def test_importing_nvc_does_not_import_torch():
    result = subprocess.run([sys.executable, "-c", "import nvc, sys; print('torch' in sys.modules)"],
                            capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert result.stdout.strip() == "False", result.stderr


@pytest.mark.parametrize("frames, message", [
    (torch.rand(2, 3, 30, 32), "divisible"),
    (torch.rand(2, 1, 32, 32), r"\[N, 3, H, W\]"),
    (torch.rand(2, 3, 32, 32) + 1.0, r"\[0, 1\]"),
    (torch.zeros(0, 3, 32, 32), "empty"),
])
def test_encode_rejects_unusable_frames(codec, frames, message):
    with pytest.raises(ValueError, match=message):
        codec.encode(frames)


# --- stream validation -----------------------------------------------------------------


def test_decode_refuses_a_stream_from_a_different_bundle(encoded):
    other = VideoCodec(_build_bundle(motion_seed=99), device="cpu")
    with pytest.raises(TemporalFormatError, match="motion entropy model mismatch"):
        other.decode(encoded[0].data)


def test_decode_refuses_mismatched_structure(codec, encoded):
    data = bytearray(encoded[0].data)
    data[28] = 8                                        # block_size
    with pytest.raises(TemporalFormatError, match="block_size"):
        codec.decode(bytes(data))


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d[:-3], "Truncated"),
    (lambda d: d + b"\0", "Trailing data"),
    (lambda d: b"XXXX" + d[4:], "Bad magic"),
])
def test_decode_refuses_malformed_streams(codec, encoded, mutate, message):
    with pytest.raises(TemporalFormatError, match=message):
        codec.decode(mutate(encoded[0].data))


def test_reader_checks_quantization_block_size_against_latent_channels(encoded):
    data = bytearray(encoded[0].data)
    data[24:26] = (LATENT + 1).to_bytes(2, "little")   # num_intra_quantization_params
    with pytest.raises(TemporalFormatError, match="quantization block"):
        TemporalStreamReader(bytes(data))


# --- bundle integrity ------------------------------------------------------------------


def test_bundle_round_trips_through_a_file(bundle, tmp_path):
    path = tmp_path / "bundle.pt"
    bundle.save(path)
    loaded = CodecBundle.load(path)
    assert loaded.computed_identities() == bundle.computed_identities()
    assert loaded.provenance == {"source": "unit test"}


def _tampered(bundle, **changes):
    payload = bundle.to_payload()
    payload.update(changes)
    return CodecBundle(**{k: payload[k] for k in CodecBundle.__dataclass_fields__})


def test_bundle_with_drifted_tables_is_refused(bundle):
    coding = bundle.coding_codebook_frequencies.clone()
    coding[0, 0] += 1
    coding[0, 1] -= 1
    with pytest.raises(BundleError, match="residual"):
        _tampered(bundle, coding_codebook_frequencies=coding).verify()


def test_bundle_with_drifted_autoencoder_is_refused(bundle):
    state = {k: v.clone() for k, v in bundle.autoencoder_state.items()}
    first = sorted(state)[0]
    state[first].view(-1)[0] += 1e-3
    with pytest.raises(BundleError, match="autoencoder_sha256"):
        _tampered(bundle, autoencoder_state=state).verify()


def test_loading_a_file_that_is_not_a_bundle_is_refused(tmp_path):
    path = tmp_path / "not_a_bundle.pt"
    torch.save({"format": "something-else"}, path)
    with pytest.raises(BundleError, match="not a codec bundle"):
        CodecBundle.load(path)


def test_bundle_loading_never_unpickles_arbitrary_objects(tmp_path):
    path = tmp_path / "evil.pt"
    torch.save({"format": "nvc-codec-bundle", "payload": Path("x")}, path)
    with pytest.raises(BundleError, match="not a loadable codec bundle"):
        CodecBundle.load(path)


def test_video_package_does_not_depend_on_research_scripts():
    for path in (ROOT / "src" / "nvc" / "video").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "importlib" not in text and "spec_from_file_location" not in text, path.name
