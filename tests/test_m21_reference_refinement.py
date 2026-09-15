"""M21 - tests for causal reference refinement.

The three load-bearing tests here are:

  * `test_identity_candidate_reproduces_the_deployed_closed_loop` - every M21
    number is a delta against the identity arm, so if that arm is not
    byte-identical to `m13_closed_loop`'s deployed path, every result is
    measured from the wrong origin;
  * `test_encoder_and_decoder_produce_identical_refined_references` - a
    refinement is only admissible if the decoder can rebuild the reference it
    conditions on, with no side information;
  * `test_refinement_cannot_see_the_current_frame` - the causality property the
    decoder-compatibility argument rests on, pinned by signature so a future
    "small" change cannot slip the target frame in.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def m21():
    return _load_script("m21_refinement")


@pytest.fixture(scope="module")
def m13():
    return _load_script("m13_recalibration")


@pytest.fixture(scope="module")
def cl13():
    return _load_script("m13_closed_loop")


@pytest.fixture(scope="module")
def mc():
    return _load_script("m10h_motion_compensation")


@pytest.fixture(scope="module")
def ma():
    return _load_script("m11_ar_entropy")


@pytest.fixture(scope="module")
def mk():
    return _load_script("m10k_learned_entropy")


@pytest.fixture(scope="module")
def ml():
    return _load_script("m10l_shared_codebook")


@pytest.fixture(scope="module")
def mt():
    return _load_script("m11_train")


C, H, W = 8, 4, 4
ALPHABET = 16
GROUP = 2
ZERO = torch.full((C,), 8)


def _tiny_model(ma, mk, *, seed=0):
    torch.manual_seed(seed)
    m10k = mk.build_model({"latent_channels": C, "alphabet": ALPHABET, "hidden": 8}).eval()
    torch.nn.init.normal_(m10k.channel_embedding.weight)
    model = ma.from_m10k(m10k, group_size=GROUP).eval()
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        model.features[0].weight[:, 1:] = torch.randn(
            model.features[0].weight[:, 1:].shape, generator=generator) * 0.3
    return model


def _frames(count, seed=0):
    generator = torch.Generator().manual_seed(seed)
    references = [torch.randn(C, H, W, generator=generator).numpy().astype(np.float32)
                  for _ in range(count)]
    symbols = [torch.randint(0, ALPHABET, (C, H, W), generator=generator).numpy().astype(np.int64)
               for _ in range(count)]
    return np.stack(symbols), np.stack(references)


@pytest.fixture(scope="module")
def spec(m13, ma, mk, ml, mt):
    """A miniature but structurally real M13 arm: real G16 model, real fitted
    K-prototype assignment codebook, real recalibrated coding codebook."""
    model = _tiny_model(ma, mk)
    symbols, references = _frames(24, seed=10)
    assign_codebook = mt.fit_model_codebook(
        ml, model, (torch.from_numpy(symbols), torch.from_numpy(references)), ZERO,
        bits=4, size=6, rows=2000, device=torch.device("cpu"), seed=0)
    table_index = m13.m11_g16_prototype_indices(model, assign_codebook, symbols, references,
                                                ZERO, device=torch.device("cpu"))
    select_symbols, select_references = _frames(8, seed=20)
    select_k = m13.m11_g16_prototype_indices(model, assign_codebook, select_symbols,
                                             select_references, ZERO, device=torch.device("cpu"))
    frequencies, strength, _, _ = m13.fit_recalibrated_frequencies(
        assign_codebook, symbols.reshape(-1), table_index.reshape(-1),
        select_symbols.reshape(-1), select_k.reshape(-1), alphabet=ALPHABET)
    coding_codebook = m13.build_recalibrated_codebook(assign_codebook, frequencies,
                                                      provenance={"strength": strength})
    identity = ma.model_identity(model, m10k_identity=b"\x00" * 8,
                                 calibration_signature="m21-test", bits=4,
                                 codebook=coding_codebook)
    return {"model": model, "zero": ZERO, "assign_codebook": assign_codebook,
            "coding_codebook": coding_codebook, "identity": identity}


# --- 1: refinement definitions and determinism ---------------------------------------------


def test_candidate_family_is_pre_registered_and_well_formed(m21):
    assert m21.CANDIDATES[0] is m21.IDENTITY
    names = [c.name for c in m21.CANDIDATES]
    assert len(names) == len(set(names))
    for candidate in m21.CANDIDATES:
        assert candidate.domain in ("none", "pixel", "latent")
        assert candidate.definition.strip()
        assert candidate.family in ("control", "A", "B")
    assert {c.domain for c in m21.CANDIDATES} == {"none", "pixel", "latent"}
    assert any(c.family == "A" for c in m21.CANDIDATES)
    assert any(c.family == "B" for c in m21.CANDIDATES)


def test_identity_refinement_is_the_identity(m21):
    x = torch.randn(1, 3, 8, 8)
    assert torch.equal(m21.IDENTITY(x, None), x)
    assert m21.IDENTITY.domain == "none"


@pytest.mark.parametrize("name", [c.name for c in _load_script("m21_refinement").CANDIDATES
                                  if c.domain == "pixel" and "ae_" not in c.name])
def test_pixel_refinements_are_deterministic_and_shape_preserving(m21, name):
    candidate = m21.candidate(name)
    torch.manual_seed(3)
    x = torch.rand(1, 3, 16, 16)
    first, second = candidate(x, None), candidate(x, None)
    assert torch.equal(first, second)
    assert first.shape == x.shape and first.dtype == x.dtype
    assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0


@pytest.mark.parametrize("name", [c.name for c in _load_script("m21_refinement").CANDIDATES
                                  if c.domain == "latent" and "ae_" not in c.name])
def test_latent_refinements_are_deterministic_and_shape_preserving(m21, name):
    candidate = m21.candidate(name)
    torch.manual_seed(4)
    x = torch.randn(1, 8, 6, 6)
    first, second = candidate(x, None), candidate(x, None)
    assert torch.equal(first, second)
    assert first.shape == x.shape and first.dtype == x.dtype


def test_a_refinement_that_changes_shape_is_rejected(m21):
    bad = m21.Refinement(name="bad", domain="pixel", definition="drops a channel",
                         apply=lambda x, m: x[:, :1])
    with pytest.raises(ValueError, match="changed shape/dtype"):
        bad(torch.rand(1, 3, 8, 8), None)


def test_unknown_candidate_is_refused(m21):
    with pytest.raises(ValueError, match="not pre-registered|unknown candidate"):
        m21.candidate("px_sharpen_9000")


def test_median3_matches_a_reference_implementation(m21):
    """The one kernel with an order statistic rather than a linear combination,
    so it is checked against an explicit replicate-padded window."""
    torch.manual_seed(5)
    x = torch.rand(1, 2, 5, 5)
    actual = m21.median3(x)
    padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate")
    for channel in range(2):
        for row in range(5):
            for column in range(5):
                window = padded[0, channel, row:row + 3, column:column + 3].reshape(-1)
                assert torch.isclose(actual[0, channel, row, column], window.median())


def test_blends_interpolate_between_identity_and_the_smoother(m21):
    torch.manual_seed(6)
    x = torch.rand(1, 3, 8, 8)
    zero = m21.blend_box(x, size=3, alpha=0.0)
    full = m21.blend_box(x, size=3, alpha=1.0)
    half = m21.blend_box(x, size=3, alpha=0.5)
    assert torch.allclose(zero, x, atol=1e-7)
    assert torch.allclose(half, (x + full) / 2, atol=1e-6)


def test_unsharp_is_the_sign_flipped_counterpart_of_a_blend(m21):
    torch.manual_seed(7)
    x = torch.rand(1, 3, 8, 8)
    sharp = m21.unsharp(x, size=3, beta=0.5)
    blur = m21.blend_binomial(x, size=3, alpha=0.5)
    assert torch.allclose(sharp + blur, 2 * x, atol=1e-6)


# --- 2: no current-frame information -------------------------------------------------------


def test_refinement_cannot_see_the_current_frame(m21):
    """A refinement is handed exactly one tensor and the frozen model. Pinned by
    signature so the causality argument cannot silently rot."""
    parameters = list(inspect.signature(m21.Refinement.__call__).parameters)
    assert parameters == ["self", "tensor", "model"]
    for candidate in m21.CANDIDATES:
        applied = list(inspect.signature(candidate.apply).parameters)
        assert len(applied) == 2, candidate.name
        assert not any(word in " ".join(applied).lower()
                       for word in ("frame", "target", "latent_current", "symbol", "residual"))


@pytest.mark.parametrize("script", ["m21_refinement", "m21_sweep", "m21_oracle"])
def test_refinement_call_sites_never_pass_the_current_frame(script):
    source = (ROOT / "scripts" / f"{script}.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if "refinement(" in stripped and not stripped.startswith(("#", "*", "-")):
            assert "frame" not in stripped, stripped
            assert "delta" not in stripped and "symbols" not in stripped, stripped


def test_encode_and_decode_apply_the_refinement_at_the_same_point(m21):
    encode = inspect.getsource(m21.encode_sequence_refined)
    decode = inspect.getsource(m21.decode_sequence_refined)
    for source in (encode, decode):
        assert 'refinement(previous, model)\n' in source.replace("  ", "") or \
            'refinement(previous, model)' in source
        assert 'refinement.domain == "pixel"' in source
        assert 'refinement.domain == "latent"' in source
        assert "refinement(reference_latent, model)" in source


# --- 3: the closed loop reproduces the deployed one at identity ----------------------------


def _run_pair(m21, mc, m13, spec, refinement, tmp_path, *, frames, rig):
    path = tmp_path / f"{refinement.name}.nvct"
    encoded = m21.encode_sequence_refined(
        mc, m13, rig["autoencoder"], frames, spec, path, refinement,
        intra_params=rig["intra_params"], intra_entropy_model=rig["intra_entropy_model"],
        residual_params=rig["residual_params"],
        motion_entropy_model=rig["motion_entropy_model"], bits=4, gop_size=rig["gop"],
        block_size=rig["block_size"], search_range=rig["search_range"])
    decoded = m21.decode_sequence_refined(
        mc, m13, rig["autoencoder"], path, spec, refinement,
        intra_entropy_model=rig["intra_entropy_model"],
        motion_entropy_model=rig["motion_entropy_model"], bits=4)
    return encoded, decoded


@pytest.fixture(scope="module")
def rig(mc, spec):
    """A tiny real autoencoder plus real intra/residual/motion coding pieces, so
    the closed loop under test is the real one at a small size."""
    from nvc.compression.entropy_model import EmpiricalEntropyModel
    from nvc.compression.quantization import QuantizationParams
    from nvc.models.autoencoder import BaselineAutoencoder

    torch.manual_seed(11)
    autoencoder = BaselineAutoencoder(in_channels=3, latent_channels=C, base_channels=4).eval()
    # `scale`/`zero_point` must be broadcastable as [1, C, 1, 1] - the shape the
    # real calibration produces.
    intra_params = QuantizationParams(
        bits=4, mode="per_channel", scale=torch.full((1, C, 1, 1), 0.25),
        zero_point=torch.full((1, C, 1, 1), 8.0))
    residual_params = QuantizationParams(
        bits=4, mode="per_channel", scale=torch.full((1, C, 1, 1), 0.5),
        zero_point=torch.full((1, C, 1, 1), 8.0))
    intra_entropy_model = EmpiricalEntropyModel(
        np.full((C, 16), 4096, dtype=np.int64), bits=4)
    search_range = 2
    motion_bits = mc.motion_alphabet_bits(search_range)
    motion_entropy_model = EmpiricalEntropyModel(
        np.full((2, 2 ** motion_bits), 65536 // (2 ** motion_bits), dtype=np.int64),
        bits=motion_bits)
    return {"autoencoder": autoencoder, "intra_params": intra_params,
            "residual_params": residual_params, "intra_entropy_model": intra_entropy_model,
            "motion_entropy_model": motion_entropy_model, "gop": 3,
            "block_size": 8, "search_range": search_range}


@pytest.fixture(scope="module")
def probe_frames():
    generator = torch.Generator().manual_seed(21)
    base = torch.rand(1, 3, 32, 32, generator=generator)
    frames = [base]
    for step in range(5):
        shifted = torch.roll(base, shifts=(step + 1, step + 1), dims=(2, 3))
        frames.append((shifted + 0.02 * torch.rand(
            1, 3, 32, 32, generator=generator)).clamp(0, 1))
    return torch.cat(frames, dim=0)


def test_identity_candidate_reproduces_the_deployed_closed_loop(
        m21, cl13, mc, m13, ma, spec, rig, probe_frames, tmp_path):
    """M21's parallel loop, at the identity candidate, must produce the SAME
    container bytes as `m13_closed_loop.encode_multi`'s deployed arm - the
    origin every M21 delta is measured from."""
    deployed_path = tmp_path / "deployed.nvct"
    deployed = cl13.encode_multi(
        mc, ma, m13, rig["autoencoder"], probe_frames, {"m13_recal": spec},
        {"m13_recal": deployed_path}, intra_params=rig["intra_params"],
        intra_entropy_model=rig["intra_entropy_model"],
        residual_params=rig["residual_params"],
        motion_entropy_model=rig["motion_entropy_model"], bits=4, gop_size=rig["gop"],
        block_size=rig["block_size"], search_range=rig["search_range"])

    encoded, _ = _run_pair(m21, mc, m13, spec, m21.IDENTITY, tmp_path,
                           frames=probe_frames, rig=rig)
    reference = deployed["arms"]["m13_recal"]
    assert encoded["residual_bytes"] == reference["residual_bytes"]
    assert encoded["motion_bytes"] == reference["motion_bytes"]
    assert encoded["container_bytes"] == reference["container_bytes"]
    assert encoded["p_frame_ideal_bits"] == pytest.approx(reference["p_frame_ideal_bits"])
    assert torch.equal(encoded["reconstructions"], deployed["reconstructions"])
    assert (tmp_path / "identity.nvct").read_bytes() == deployed_path.read_bytes()


def test_byte_accounting_closes(m21, mc, m13, spec, rig, probe_frames, tmp_path):
    encoded, _ = _run_pair(m21, mc, m13, spec, m21.IDENTITY, tmp_path,
                           frames=probe_frames, rig=rig)
    assert (encoded["motion_bytes"] + encoded["residual_bytes"]
            + encoded["container_overhead_bytes"]) == encoded["container_bytes"]
    assert (encoded["i_frame_residual_bytes"] + encoded["p_frame_residual_bytes"]
            == encoded["residual_bytes"])


# --- 4: encoder / decoder equality ---------------------------------------------------------


@pytest.mark.parametrize("name", ["identity", "px_median3", "px_box3_a50", "px_unsharp3_b50",
                                  "px_ae_reproject", "lat_box3_a50", "lat_ae_reproject"])
def test_encoder_and_decoder_produce_identical_refined_references(
        m21, mc, m13, spec, rig, probe_frames, tmp_path, name):
    """The decoder-compatibility proof: symbols, reconstruction AND the refined
    reference latents must match exactly, with nothing extra in the stream."""
    refinement = m21.candidate(name)
    encoded, decoded = _run_pair(m21, mc, m13, spec, refinement, tmp_path,
                                 frames=probe_frames, rig=rig)
    assert len(decoded["symbols"]) == len(encoded["symbols"]) > 0
    for produced, recovered in zip(encoded["symbols"], decoded["symbols"]):
        assert np.array_equal(np.asarray(produced).reshape(-1),
                              np.asarray(recovered).reshape(-1))
    for produced, recovered in zip(encoded["reference_latents"], decoded["reference_latents"]):
        assert torch.equal(produced, recovered)
    assert torch.equal(decoded["reconstructions"].cpu(), encoded["reconstructions"])


def test_a_decoder_using_the_wrong_refinement_diverges(
        m21, mc, m13, spec, rig, probe_frames, tmp_path):
    """Negative control: if the refinement were NOT load-bearing, decoding with
    a different one would still work - and the equality test above would prove
    nothing."""
    encoded, _ = _run_pair(m21, mc, m13, spec, m21.candidate("px_box3_a100"), tmp_path,
                           frames=probe_frames, rig=rig)
    wrong = m21.decode_sequence_refined(
        mc, m13, rig["autoencoder"], tmp_path / "px_box3_a100.nvct", spec, m21.IDENTITY,
        intra_entropy_model=rig["intra_entropy_model"],
        motion_entropy_model=rig["motion_entropy_model"], bits=4)
    assert not all(
        torch.equal(a, b) for a, b in
        zip(encoded["reference_latents"], wrong["reference_latents"]))


def test_no_side_information_enters_the_container(
        m21, mc, m13, spec, rig, probe_frames, tmp_path):
    """Every candidate's stream must carry exactly the same FIELDS as the
    deployed one - the header is read, never extended, and the only thing that
    may differ is how many bytes the payloads take."""
    headers = {}
    for name in ("identity", "px_box3_a50", "lat_ae_reproject"):
        _run_pair(m21, mc, m13, spec, m21.candidate(name), tmp_path,
                  frames=probe_frames, rig=rig)
        header = mc.TemporalStreamReader(tmp_path / f"{name}.nvct").header
        headers[name] = (header.format_version, header.gop_size, header.block_size,
                         header.search_range, header.motion_bits, header.reference_mode,
                         header.intra_entropy_model_id, header.residual_entropy_model_id,
                         header.motion_entropy_model_id, header.frame_count)
    assert len(set(headers.values())) == 1, headers


# --- 5: refinement is not a no-op, and latent candidates leave motion alone ----------------


def test_a_real_refinement_actually_changes_the_reference(
        m21, mc, m13, spec, rig, probe_frames, tmp_path):
    """Guards against a sweep of silent no-ops, which would 'prove' refinement
    is harmless without ever exercising it."""
    base, _ = _run_pair(m21, mc, m13, spec, m21.IDENTITY, tmp_path,
                        frames=probe_frames, rig=rig)
    moved, _ = _run_pair(m21, mc, m13, spec, m21.candidate("px_box3_a100"), tmp_path,
                         frames=probe_frames, rig=rig)
    assert not all(torch.equal(a, b) for a, b in
                   zip(base["reference_latents"], moved["reference_latents"]))


def test_latent_candidates_leave_the_motion_field_untouched(
        m21, mc, m13, spec, rig, probe_frames, tmp_path):
    """A latent-domain refinement sits downstream of motion, so motion bytes
    must be bit-identical to the deployed arm's; a pixel-domain one may change
    them, and that difference is part of what the sweep measures."""
    base, _ = _run_pair(m21, mc, m13, spec, m21.IDENTITY, tmp_path,
                        frames=probe_frames, rig=rig)
    latent, _ = _run_pair(m21, mc, m13, spec, m21.candidate("lat_box3_a50"), tmp_path,
                          frames=probe_frames, rig=rig)
    assert latent["motion_bytes"] == base["motion_bytes"]


# --- 6: protocol, gates, and selection rule ------------------------------------------------


def test_declared_protocol_is_fixed(m21):
    assert m21.SCREEN_BITS == 4
    assert tuple(m21.CONFIRM_BITS) == (5, 3)
    assert m21.STAGE2_ADMISSION_PERCENT == -0.25
    assert m21.STAGE2_MINIMUM_CANDIDATES == 3
    assert sorted([m21.SCREEN_BITS, *m21.CONFIRM_BITS], reverse=True) == [5, 4, 3]


def test_gate_thresholds_match_the_project_wide_ones(m21):
    assert (m21.WEAK_BELOW_PERCENT, m21.MEANINGFUL_ABOVE_PERCENT) == (0.5, 1.0)
    assert m21.verdict(0.49) == "weak"
    assert m21.verdict(0.5) == "marginal"
    assert m21.verdict(1.0) == "meaningful"
    assert m21.verdict(-3.0) == "weak"


def test_selection_rule_uses_coded_bytes_not_quality():
    analysis = _load_script("m21_analysis")
    rows = [
        {"candidate": "a", "total_stream_gain_percent": 0.2, "decoder_compatible": True,
         "delta_psnr_db": 2.0, "seconds": 10.0, "domain": "pixel"},
        {"candidate": "b", "total_stream_gain_percent": 0.9, "decoder_compatible": True,
         "delta_psnr_db": -0.01, "seconds": 40.0, "domain": "pixel"},
        {"candidate": "c", "total_stream_gain_percent": 5.0, "decoder_compatible": False,
         "delta_psnr_db": 9.0, "seconds": 1.0, "domain": "pixel"},
    ]
    assert analysis.select_candidate(rows)["candidate"] == "b"


def test_selection_rule_breaks_ties_toward_the_cheaper_candidate():
    analysis = _load_script("m21_analysis")
    rows = [
        {"candidate": "slow", "total_stream_gain_percent": 0.60, "decoder_compatible": True,
         "seconds": 90.0, "domain": "pixel"},
        {"candidate": "fast", "total_stream_gain_percent": 0.6001, "decoder_compatible": True,
         "seconds": 10.0, "domain": "latent"},
    ]
    assert analysis.select_candidate(rows)["candidate"] == "fast"


def test_a_quality_loss_that_buys_bytes_is_flagged():
    analysis = _load_script("m21_analysis")
    assert analysis.rate_distortion_flag({"total_stream_gain_percent": 1.2,
                                          "delta_psnr_db": -0.35,
                                          "delta_msssim": -0.004}) is not None
    assert analysis.rate_distortion_flag({"total_stream_gain_percent": 1.2,
                                          "delta_psnr_db": 0.01,
                                          "delta_msssim": 0.0}) is None


# --- 7: no TEST access ---------------------------------------------------------------------


@pytest.mark.parametrize("script", ["m21_refinement", "m21_baseline", "m21_oracle",
                                    "m21_sweep", "m21_analysis"])
def test_m21_scripts_never_touch_the_test_split(script):
    source = (ROOT / "scripts" / f"{script}.py").read_text(encoding="utf-8")
    assert 'split="test"' not in source
    assert "test_sequences" not in source


def test_val_b_selection_never_returns_test_sequences(m21):
    source = inspect.getsource(m21.val_b_sequences)
    assert 'split="val"' in source and "[1::2]" in source


def test_motion_table_allocation_is_train_only(m21):
    source = inspect.getsource(m21.broad_train_sequences)
    assert 'split="train"' in source


# --- 8: provenance and container compatibility ---------------------------------------------


def test_m21_does_not_modify_any_production_source():
    result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True,
                            cwd=ROOT, check=False)
    # Scoped to src/nvc/ deliberately. An earlier version failed on ANY modified
    # tracked file, so an unrelated in-progress edit (a README change, say) looked
    # identical to this milestone touching production code - a false positive that
    # fires for everyone with a dirty tree.
    modified = [line for line in result.stdout.splitlines()
                if line[:2].strip() in {"M", "A", "D", "R"}]
    modified_production = [line for line in modified if "src/nvc/" in line.replace("\\", "/")]
    assert modified_production == [], f"unexpected production modifications: {modified_production}"


@pytest.mark.parametrize("script", ["m21_refinement", "m21_baseline", "m21_oracle",
                                    "m21_sweep", "m21_analysis"])
def test_m21_introduces_no_new_container_format(script):
    source = (ROOT / "scripts" / f"{script}.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "nvct_v3" not in lowered and "format_version = 3" not in lowered
    assert "TEMPORAL_FORMAT_VERSION =" not in source


def test_m21_reuses_the_deployed_coder_and_container(m21):
    encode = inspect.getsource(m21.encode_sequence_refined)
    decode = inspect.getsource(m21.decode_sequence_refined)
    assert "mc.TemporalStreamWriter(" in encode
    assert "m13.encode_frame_recalibrated(" in encode
    assert "mc.TemporalStreamReader(" in decode
    assert "m13.decode_frame_recalibrated(" in decode


def test_decoder_checks_all_three_entropy_identities(m21):
    source = inspect.getsource(m21.decode_sequence_refined)
    for field in ("residual_entropy_model_id", "intra_entropy_model_id",
                  "motion_entropy_model_id"):
        assert field in source


def test_frozen_baseline_identities_still_match_m19(m21):
    recorded = json.loads((ROOT / "outputs/m19_reference_error_audit/m19_identities.json")
                          .read_text(encoding="utf-8"))
    baseline_path = ROOT / "outputs/m21_reference_refinement/m21_baseline.json"
    if not baseline_path.is_file():
        pytest.skip("Phase 0 has not been run in this working tree")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    for point in baseline["rate_points"]:
        expected = recorded[str(point["bits"])]
        assert point["identities"]["residual"] == expected["residual_identity"]
        assert point["identities"]["assign_codebook"] == expected["assign_codebook_id"]
        assert point["identities"]["coding_codebook"] == expected["coding_codebook_id"]
        assert point["identities"]["motion"] == m21.DEPLOYED_MOTION_IDENTITY
        assert point["bytes_match"] is True


def test_recorded_val_b_baseline_matches_m17():
    baseline = _load_script("m21_baseline")
    assert baseline.RECORDED_VAL_B_P_RESIDUAL_BYTES == {5: 1_333_275, 4: 908_404, 3: 549_083}


# --- 9: independent-process reproducibility ------------------------------------------------


def test_refinements_are_reproducible_in_an_independent_process(tmp_path):
    """Phase 10 in miniature: the refinement kernels must produce identical
    output in a fresh interpreter, not merely twice inside one warm process."""
    script = tmp_path / "probe.py"
    script.write_text(
        "import importlib.util, json, sys\n"
        "import torch\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        f"spec = importlib.util.spec_from_file_location('m21', {str(ROOT / 'scripts' / 'm21_refinement.py')!r})\n"
        "m21 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m21)\n"
        "torch.manual_seed(99)\n"
        "pixel = torch.rand(1, 3, 16, 16)\n"
        "latent = torch.randn(1, 8, 6, 6)\n"
        "out = {}\n"
        "for c in m21.CANDIDATES:\n"
        "    if 'ae_' in c.name or c.domain == 'none':\n"
        "        continue\n"
        "    x = pixel if c.domain == 'pixel' else latent\n"
        "    y = c(x, None)\n"
        "    out[c.name] = [float(y.sum()), float(y.abs().max()), list(y.shape)]\n"
        "print(json.dumps(out, sort_keys=True))\n", encoding="utf-8")
    runs = [subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                           cwd=ROOT, check=True).stdout for _ in range(2)]
    assert json.loads(runs[0]) == json.loads(runs[1])
    assert len(json.loads(runs[0])) >= 8
