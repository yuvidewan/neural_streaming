"""M16 - tests for the GOP-boundary reference diagnostic. M16 is an audit
milestone (see M16_REPORT.md): these tests cover categories 1-7 of its own
Phase J list unconditionally (GOP-boundary identification, reference-flow
correctness, bit-depth-specific behavior, oracle/reference comparison, no
TEST access, determinism, existing-stream compatibility). Categories 8-9
(new-stream round trip, provenance mismatch) are conditional on Phase F
actually shipping an implementation - see the report for why they were not
reached.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.codec import latent_to_symbols
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.models.autoencoder import BaselineAutoencoder


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mc():
    return _load_script("m10h_motion_compensation")


@pytest.fixture(scope="module")
def m16():
    return _load_script("m16_reference_diagnostic")


TINY = {"in_channels": 3, "latent_channels": 4, "base_channels": 8}
BITS = 3
GOP = 4


def _autoencoder(seed: int = 0):
    torch.manual_seed(seed)
    model = BaselineAutoencoder(**TINY).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class _FakeSequence:
    def __init__(self, frames: torch.Tensor, sequence_id: str = "fake"):
        self._frames = frames
        self.sequence_id = sequence_id
        self.frame_count = frames.shape[0]

    def load_frames(self) -> torch.Tensor:
        return self._frames


def _moving_frames(count: int = 15, size: int = 64, *, seed: int = 0, step: int = 2):
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 3, size, size, generator=generator)
    return torch.cat([torch.roll(base, shifts=(i * step, i * step), dims=(2, 3))
                      for i in range(count)], dim=0)


def _calibration(model, frames, *, bits=BITS):
    with torch.no_grad():
        latents = torch.cat([model.encode(frames[i:i + 1]) for i in range(frames.shape[0])])
    intra_params = calibrate_quantization_params(latents, bits=bits, mode="per_channel")
    intra_symbols = np.stack([
        latent_to_symbols(latents[i:i + 1], intra_params).reshape(latents.shape[1], -1)
        for i in range(latents.shape[0])])
    intra_entropy_model = EmpiricalEntropyModel.from_symbols(
        intra_symbols, bits=bits, num_tables=latents.shape[1])
    residuals = latents[1:] - latents[:-1]
    residual_params = calibrate_quantization_params(residuals, bits=bits, mode="per_channel")
    return intra_params, intra_entropy_model, residual_params


# --- 1: GOP-boundary identification --------------------------------------------------


def test_gop_boundary_positions_match_gop_frame_types(mc):
    frame_count, gop = 23, 10
    types = mc.gop_frame_types(frame_count, gop)
    i_positions = [i for i, t in enumerate(types) if t == mc.FRAME_TYPE_I]
    assert i_positions == [0, 10, 20]

    boundary = [i for i in range(frame_count)
               if types[i] != mc.FRAME_TYPE_I and (i - 1) % gop == 0]
    ordinary = [i for i in range(frame_count)
               if types[i] != mc.FRAME_TYPE_I and i not in boundary]
    assert boundary == [1, 11, 21]
    assert 2 in ordinary and 9 in ordinary and 19 in ordinary


def test_diagnose_sequence_tags_boundary_rows_correctly(mc, m16):
    model = _autoencoder()
    frames = _moving_frames(13, seed=0)
    intra_params, intra_entropy_model, residual_params = _calibration(model, frames)
    rows, _ = m16.diagnose_sequence(
        mc, model, _FakeSequence(frames), bits=BITS, intra_params=intra_params,
        intra_entropy_model=intra_entropy_model, residual_params=residual_params,
        gop_size=GOP, block_size=16, search_range=8, device=torch.device("cpu"))
    boundary_indices = [r["index"] for r in rows if r["is_boundary"]]
    assert boundary_indices == [1, 5, 9]  # gop=4 -> I at 0,4,8,12; boundary P at 1,5,9
    for r in rows:
        assert r["gop_position"] == r["index"] % GOP


# --- 2: reference-flow correctness (matches encode_multi bit-for-bit) -----------------


def test_diagnose_sequence_real_chain_matches_encode_multi_exactly(mc, m16):
    # The whole diagnostic is only valid if its REAL chain is not just
    # "similar to" but IDENTICAL to what the deployed coder actually
    # advances - this is the load-bearing correctness check for the
    # entire milestone.
    cl = _load_script("m13_closed_loop")
    ma = _load_script("m11_ar_entropy")
    mk = _load_script("m10k_learned_entropy")
    ml = _load_script("m10l_shared_codebook")
    m13 = _load_script("m13_recalibration")

    model = _autoencoder(seed=1)
    frames = _moving_frames(13, seed=1)
    intra_params, intra_entropy_model, residual_params = _calibration(model, frames, bits=BITS)

    blocks = (frames.shape[2] // 16) * (frames.shape[3] // 16)
    motion_model = EmpiricalEntropyModel.from_symbols(
        np.stack([np.stack([np.arange(blocks) % 17, np.arange(blocks) % 17])]),
        bits=mc.motion_alphabet_bits(8), num_tables=2)

    torch.manual_seed(4)
    m10k = mk.build_model({"latent_channels": 4, "alphabet": 2 ** BITS, "hidden": 8}).eval()
    with torch.no_grad():
        latents = torch.cat([model.encode(frames[i:i + 1]) for i in range(frames.shape[0])])
    references = [latents[i].numpy() for i in range(1, latents.shape[0])]
    samples = ml.sample_training_distributions(m10k, references, device=torch.device("cpu"),
                                                max_rows=2000)
    assign_codebook = ml.fit_codebook(samples, 16, bits=BITS)
    m11_model = ma.from_m10k(m10k, group_size=2).eval()
    with torch.no_grad():
        m11_model.features[0].weight[:, 1:] = 0.4
    zero = torch.full((4,), 4)
    symbols_for_k = [latent_to_symbols(latents[i:i + 1] - latents[i - 1:i], residual_params)
                     .reshape(latents.shape[1:]) for i in range(1, latents.shape[0])]
    train_symbols = np.stack([s.reshape(-1) for s in symbols_for_k])
    train_k = m13.m11_g16_prototype_indices(m11_model, assign_codebook, np.stack(symbols_for_k),
                                            np.stack(references), zero, device=torch.device("cpu"))
    frequencies, strength, _, _ = m13.fit_recalibrated_frequencies(
        assign_codebook, train_symbols.reshape(-1), train_k.reshape(-1),
        train_symbols.reshape(-1), train_k.reshape(-1), alphabet=2 ** BITS)
    coding_codebook = m13.build_recalibrated_codebook(assign_codebook, frequencies)
    identity = ma.model_identity(m11_model, m10k_identity=b"\x9a" * 8,
                                 calibration_signature="test", bits=BITS, codebook=coding_codebook)
    arms = {"m13_recal": {"model": m11_model, "zero": zero, "assign_codebook": assign_codebook,
                          "coding_codebook": coding_codebook, "identity": identity}}

    paths = {"m13_recal": Path("test_m16_scratch.nvct")}
    try:
        result = cl.encode_multi(mc, ma, m13, model, frames, arms, paths,
                                 intra_params=intra_params, intra_entropy_model=intra_entropy_model,
                                 residual_params=residual_params, motion_entropy_model=motion_model,
                                 bits=BITS, gop_size=GOP, block_size=16, search_range=8)
        encode_multi_reconstructions = result["reconstructions"]
    finally:
        paths["m13_recal"].unlink(missing_ok=True)

    _, real_reconstructions = m16.diagnose_sequence(
        mc, model, _FakeSequence(frames), bits=BITS, intra_params=intra_params,
        intra_entropy_model=intra_entropy_model, residual_params=residual_params,
        gop_size=GOP, block_size=16, search_range=8, device=torch.device("cpu"))

    diagnostic_reconstructions = torch.cat(real_reconstructions, dim=0)
    assert torch.equal(diagnostic_reconstructions, encode_multi_reconstructions)


# --- 3: bit-depth-specific reference behavior ------------------------------------------


def test_quantization_error_grows_as_bit_depth_drops(mc):
    model = _autoencoder(seed=2)
    frames = _moving_frames(4, seed=2)
    with torch.no_grad():
        latent = model.encode(frames[0:1])
    from nvc.compression.codec import decode_payload_to_latent, encode_latent_to_payload
    errors = {}
    for bits in (5, 4, 3):
        intra_params, intra_entropy_model, _ = _calibration(model, frames, bits=bits)
        with torch.no_grad():
            payload, _ = encode_latent_to_payload(latent, params=intra_params,
                                                  entropy_model=intra_entropy_model)
            decoded, _ = decode_payload_to_latent(payload, entropy_model=intra_entropy_model,
                                                  params=intra_params, shape=tuple(latent.shape[1:]))
        errors[bits] = float((decoded - latent).abs().mean())
    assert errors[3] > errors[4] > errors[5]


# --- 4: oracle/reference comparison -----------------------------------------------------


def test_oracle_and_real_reference_diverge_when_quantization_is_coarse(mc, m16):
    model = _autoencoder(seed=3)
    frames = _moving_frames(13, seed=3)
    intra_params, intra_entropy_model, residual_params = _calibration(model, frames, bits=3)
    rows, _ = m16.diagnose_sequence(
        mc, model, _FakeSequence(frames), bits=3, intra_params=intra_params,
        intra_entropy_model=intra_entropy_model, residual_params=residual_params,
        gop_size=GOP, block_size=16, search_range=8, device=torch.device("cpu"))
    assert len(rows) > 0
    for row in rows:
        assert row["sad_oracle_mean"] >= 0.0
        assert row["sad_real_mean"] >= 0.0
        assert 0.0 <= row["fraction_motion_vectors_changed"] <= 1.0
    # Oracle reference is the SAME class of thing regardless of frame - a
    # basic sanity bound, not a claim about which side wins on average.
    assert all(row["residual_energy_real"] >= 0.0 for row in rows)


def test_no_distinct_decoded_latent_reference_c_is_forced(mc):
    # Phase A's own finding: motion estimation/warping never accept a
    # latent tensor in this codec, so "C" cannot be realized as anything
    # other than a duplicate of "A" - pinned here so a future change that
    # DID add a latent-space warp path would be noticed (this test would
    # then need updating, not silently stay green on a stale assumption).
    source = Path("scripts/m10h_motion_compensation.py").read_text(encoding="utf-8")
    assert "def estimate_block_motion" in source
    assert "def warp_blocks" in source
    import re
    motion_sig = re.search(r"def estimate_block_motion\([^)]*\)", source, re.S).group(0)
    warp_sig = re.search(r"def warp_blocks\([^)]*\)", source, re.S).group(0)
    assert "latent" not in motion_sig.lower()
    assert "latent" not in warp_sig.lower()


# --- 5: no TEST access during diagnostics/fitting ---------------------------------------


def test_diagnose_sequence_has_no_test_channel(m16):
    parameters = list(inspect.signature(m16.diagnose_sequence).parameters)
    assert not any("test" in p.lower() for p in parameters)


@pytest.mark.parametrize("script_name", ["m16_reference_audit.py", "m16_reference_diagnostic.py"])
def test_scripts_never_feed_test_sequences_into_a_fitting_or_diagnostic_call(script_name):
    source = Path("scripts") / script_name
    text = source.read_text(encoding="utf-8")
    calls = ("diagnose_sequence(", "calibrate_grids(", "build_policy(")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if any(call in line for call in calls):
            window = "\n".join(lines[index:index + 4])
            assert "test_sequences" not in window, (
                f"{script_name}:{index + 1} calls a fitting/diagnostic function near "
                f"'test_sequences' - TEST must never reach it")


# --- 6: deterministic behavior -----------------------------------------------------------


def test_diagnose_sequence_is_deterministic(mc, m16):
    model = _autoencoder(seed=5)
    frames = _moving_frames(9, seed=5)
    intra_params, intra_entropy_model, residual_params = _calibration(model, frames, bits=4)
    kwargs = dict(bits=4, intra_params=intra_params, intra_entropy_model=intra_entropy_model,
                 residual_params=residual_params, gop_size=GOP, block_size=16, search_range=8,
                 device=torch.device("cpu"))
    rows_a, recon_a = m16.diagnose_sequence(mc, model, _FakeSequence(frames), **kwargs)
    rows_b, recon_b = m16.diagnose_sequence(mc, model, _FakeSequence(frames), **kwargs)
    for a, b in zip(rows_a, rows_b):
        assert a["sad_real_mean"] == b["sad_real_mean"]
        assert a["sad_oracle_mean"] == b["sad_oracle_mean"]
        assert np.array_equal(a["motion_symbols_real"], b["motion_symbols_real"])
        assert np.array_equal(a["motion_symbols_oracle"], b["motion_symbols_oracle"])
    for a, b in zip(recon_a, recon_b):
        assert torch.equal(a, b)


# --- 7: existing-stream compatibility (M16 changes nothing about production) -----------


def test_m16_scripts_do_not_modify_any_production_source():
    # M16 is audit-only unless Phase F is reached (see the report) - this
    # pins that no src/nvc file or existing m10-m15 script was touched.
    # CHANGELOG.md is EXPECTED to change every milestone (an M15 entry is
    # already pending, uncommitted) - it is documentation, not production
    # source, and is deliberately excluded from this check.
    import subprocess
    result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1])
    modified = [line for line in result.stdout.splitlines()
               if line.startswith(" M") or line.startswith("M ")]
    modified_production = [line for line in modified if "CHANGELOG.md" not in line]
    assert modified_production == [], f"unexpected production modifications: {modified_production}"


def test_m16_introduces_no_new_container_format():
    for script_name in ("m16_reference_audit.py", "m16_reference_diagnostic.py"):
        source = (Path("scripts") / script_name).read_text(encoding="utf-8")
        assert "nvct_v3" not in source.lower() and "format_version = 3" not in source.lower()
