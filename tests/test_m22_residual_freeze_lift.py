"""M22 - tests for the residual-quantizer freeze-lift.

The load-bearing tests here are:

  * `test_grid_change_moves_the_calibration_signature` and
    `test_deployed_g16_rejects_a_changed_grid` - the coupling that defines the
    whole milestone. If a grid change did NOT invalidate the downstream stack,
    M22's two-arm design would be unnecessary;
  * `test_deployed_variant_reproduces_calibrate_grids_exactly` - the identity
    control, without which every M22 delta is measured from the wrong origin;
  * `test_no_grid_variant_sees_val_or_test_data` - every variant is fitted on
    the TRAIN residual stack alone.
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
def m22():
    return _load_script("m22_residual")


@pytest.fixture(scope="module")
def ev():
    return _load_script("m10l_evaluate")


@pytest.fixture(scope="module")
def ma():
    return _load_script("m11_ar_entropy")


@pytest.fixture(scope="module")
def residuals():
    """A synthetic residual stack with the shape and rough character of the real
    one: 64 channels, zero-centred, heavy-tailed, with per-channel scale spread."""
    generator = torch.Generator().manual_seed(22)
    scales = torch.linspace(0.2, 4.0, 64).view(1, 64, 1, 1)
    base = torch.randn(48, 64, 16, 16, generator=generator)
    heavy = torch.randn(48, 64, 16, 16, generator=generator) ** 3 * 0.15
    return (base + heavy) * scales


# --- 1: candidate pre-registration ---------------------------------------------------------


def test_grid_variants_are_pre_registered_and_well_formed(m22):
    assert m22.GRID_VARIANTS[0] is m22.DEPLOYED
    names = [v.name for v in m22.GRID_VARIANTS]
    assert len(names) == len(set(names))
    assert m22.DEPLOYED.family == "control"
    for variant in m22.GRID_VARIANTS:
        assert variant.definition.strip()
        assert variant.family in ("control", "A")
        assert callable(variant.build)


def test_unknown_grid_variant_is_refused(m22):
    with pytest.raises(ValueError, match="unknown grid variant|pre-registered"):
        m22.grid_variant("percentile_9000")


def test_declared_protocol_is_fixed(m22):
    assert m22.SCREEN_BITS == 3
    assert tuple(m22.CONFIRM_BITS) == (5, 4)
    assert m22.STAGE2_ADMISSION_PERCENT == -0.25
    assert m22.STAGE2_MINIMUM_CANDIDATES == 2
    assert sorted([m22.SCREEN_BITS, *m22.CONFIRM_BITS], reverse=True) == [5, 4, 3]


def test_gate_thresholds_match_the_project_wide_ones(m22):
    assert (m22.WEAK_BELOW_PERCENT, m22.MEANINGFUL_ABOVE_PERCENT) == (0.5, 1.0)
    assert m22.verdict(0.49) == "weak"
    assert m22.verdict(0.5) == "marginal"
    assert m22.verdict(1.0) == "meaningful"
    assert m22.verdict(-4.0) == "weak"


def test_distortion_guard_is_declared_and_only_ever_disqualifies(m22):
    """A residual grid change is an R/D move by construction, so the acceptable
    regression is a pre-declared constant - and it can only reject a candidate,
    never promote one."""
    assert (m22.MAX_PSNR_REGRESSION_DB, m22.MAX_MSSSIM_REGRESSION) == (0.10, 0.0010)
    assert m22.distortion_regression(0.0, 0.0) is None
    assert m22.distortion_regression(-0.09, -0.0009) is None
    assert m22.distortion_regression(5.0, 5.0) is None          # a gain is never a reason
    assert "PSNR" in m22.distortion_regression(-0.2, 0.0)
    assert "MS-SSIM" in m22.distortion_regression(0.0, -0.002)
    assert set(inspect.signature(m22.distortion_regression).parameters) == {
        "delta_psnr_db", "delta_msssim"}


# --- 2: the identity control ---------------------------------------------------------------


@pytest.mark.parametrize("bits", [5, 4, 3])
def test_deployed_variant_reproduces_calibrate_grids_exactly(m22, residuals, bits):
    """The `deployed` variant must be bit-identical to what
    `calibrate_quantization_params` produces at the frozen percentiles - the
    origin every M22 delta is measured from."""
    from nvc.compression.calibration import calibrate_quantization_params
    expected = calibrate_quantization_params(
        residuals, bits=bits, mode="per_channel",
        lower_percentile=m22.DEPLOYED_LOWER_PERCENTILE,
        upper_percentile=m22.DEPLOYED_UPPER_PERCENTILE)
    actual = m22.DEPLOYED.build(residuals, bits)
    assert torch.equal(actual.scale, expected.scale)
    assert torch.equal(actual.zero_point, expected.zero_point)
    assert m22.grid_signature(actual) == m22.grid_signature(expected)


def test_params_from_range_matches_the_production_derivation(m22, residuals):
    """M22's own affine derivation is used by the symmetric/MAD/MSE variants; it
    must agree with `calibrate_quantization_params` when handed the same range."""
    from nvc.compression.calibration import calibrate_quantization_params
    bits = 4
    values = residuals.permute(1, 0, 2, 3).reshape(64, -1)
    low = torch.quantile(values, 0.001, dim=1).reshape(1, -1, 1, 1)
    high = torch.quantile(values, 0.999, dim=1).reshape(1, -1, 1, 1)
    mine = m22._params_from_range(low, high, bits=bits)
    theirs = calibrate_quantization_params(residuals, bits=bits, mode="per_channel",
                                           lower_percentile=0.1, upper_percentile=99.9)
    assert torch.allclose(mine.scale, theirs.scale, rtol=1e-5, atol=1e-8)
    assert torch.equal(mine.zero_point, theirs.zero_point)


# --- 3: every variant is well-behaved and TRAIN-only ---------------------------------------


@pytest.mark.parametrize("name", [v.name for v in _load_script("m22_residual").GRID_VARIANTS])
@pytest.mark.parametrize("bits", [5, 3])
def test_every_variant_produces_a_usable_grid(m22, residuals, name, bits):
    params = m22.grid_variant(name).build(residuals, bits)
    assert params.bits == bits and params.mode == "per_channel"
    assert params.scale.shape == (1, 64, 1, 1)
    assert params.zero_point.shape == (1, 64, 1, 1)
    assert torch.all(params.scale > 0)
    assert torch.all(torch.isfinite(params.scale))
    # zero_point must be an exact integer: that is what makes residual 0 - the
    # mode of a motion-compensated residual - exactly representable.
    assert torch.equal(params.zero_point, torch.round(params.zero_point))


@pytest.mark.parametrize("name", [v.name for v in _load_script("m22_residual").GRID_VARIANTS])
def test_every_variant_is_deterministic(m22, residuals, name):
    variant = m22.grid_variant(name)
    first, second = variant.build(residuals, 4), variant.build(residuals, 4)
    assert torch.equal(first.scale, second.scale)
    assert torch.equal(first.zero_point, second.zero_point)
    assert m22.grid_signature(first) == m22.grid_signature(second)


def test_variants_actually_differ_from_the_deployed_grid(m22, residuals):
    """Guards against a family of silent no-ops, which would 'prove' the grid
    does not matter without ever changing it."""
    deployed = m22.grid_signature(m22.DEPLOYED.build(residuals, 4))
    others = {m22.grid_signature(v.build(residuals, 4))
              for v in m22.GRID_VARIANTS if v.name != "deployed"}
    assert deployed not in others
    assert len(others) == len(m22.GRID_VARIANTS) - 1


def test_tighter_percentiles_give_a_smaller_step(m22, residuals):
    tight = m22.grid_variant("tight_p1").build(residuals, 4)
    deployed = m22.DEPLOYED.build(residuals, 4)
    broad = m22.grid_variant("broad_p001").build(residuals, 4)
    assert float(tight.scale.mean()) < float(deployed.scale.mean())
    assert float(broad.scale.mean()) > float(deployed.scale.mean())


def test_symmetric_variant_is_symmetric_about_zero(m22, residuals):
    params = m22.grid_variant("symmetric_p01").build(residuals, 4)
    levels = 2 ** 4 - 1
    # The reconstruction range must straddle zero with equal half-widths.
    low = (0 - params.zero_point) * params.scale
    high = (levels - params.zero_point) * params.scale
    assert torch.allclose(low.abs(), high.abs(), rtol=0.2, atol=1e-3)


def test_mse_optimal_variant_beats_the_deployed_grid_on_train_mse(m22, residuals):
    """The MSE-optimal reference point must actually be MSE-optimal on the data
    it was fitted to - otherwise it is not the reference point it claims."""
    deployed = m22.grid_statistics(residuals, m22.DEPLOYED.build(residuals, 4), bits=4)
    optimal = m22.grid_statistics(
        residuals, m22.grid_variant("mse_optimal").build(residuals, 4), bits=4)
    assert optimal["quantization_mse"] <= deployed["quantization_mse"]


@pytest.mark.parametrize("name", [v.name for v in _load_script("m22_residual").GRID_VARIANTS])
def test_no_grid_variant_sees_val_or_test_data(m22, name):
    """A variant is handed the TRAIN residual stack and the bit depth. Pinned by
    signature so a future change cannot slip held-out data in."""
    variant = m22.grid_variant(name)
    parameters = list(inspect.signature(variant.build).parameters)
    assert len(parameters) == 2, name
    assert not any(word in " ".join(parameters).lower()
                   for word in ("val", "test", "holdout", "held"))


def test_verify_deployed_grid_accepts_a_matching_grid_and_rejects_a_drifted_one(m22, residuals):
    """The identity control, enforced. This guard exists because a missing intra
    round trip in the TRAIN residual walk once shifted the rebuilt grid just
    enough to move the 3-bit control's VAL-B total from 723,381 to 897,872 bytes
    while still looking like 'the deployed grid' in the output table."""
    deployed = m22.DEPLOYED.build(residuals, 4)
    m22.verify_deployed_grid(residuals, deployed, bits=4)          # must not raise
    drifted = m22.grid_variant("tight_p05").build(residuals, 4)
    with pytest.raises(ValueError, match="does not reproduce the deployed"):
        m22.verify_deployed_grid(residuals, drifted, bits=4)


def test_train_residual_walk_includes_the_intra_round_trip(m22):
    """The I-frame reference inside the calibration walk must be the INTRA
    ROUND TRIP of the latent, exactly as calibrate_grids does it - not
    model.decode(latent). Pinned by source because the difference is a few
    characters and its effect is a 24% byte error in the control."""
    source = inspect.getsource(m22.collect_train_residuals)
    index = source.index("FRAME_TYPE_I")
    window = source[index:index + 600]
    assert "encode_latent_to_payload(" in window
    assert "decode_payload_to_latent(" in window
    assert "intra_params" in window and "intra_entropy_model" in window
    parameters = set(inspect.signature(m22.collect_train_residuals).parameters)
    assert {"intra_params", "intra_entropy_model"} <= parameters


# --- 4: grid statistics --------------------------------------------------------------------


def test_grid_statistics_are_internally_consistent(m22, residuals):
    stats = m22.grid_statistics(residuals, m22.DEPLOYED.build(residuals, 4), bits=4)
    assert 0.0 <= stats["clipping_fraction"] <= 1.0
    assert 0.0 <= stats["symbol_entropy_bits"] <= 4.0
    assert stats["symbol_entropy_efficiency"] == pytest.approx(
        stats["symbol_entropy_bits"] / 4.0)
    assert stats["distinct_symbols_used"] <= stats["alphabet"] == 16
    assert 0.0 <= stats["level_distance_mean"] <= 0.5
    bands = ("fraction_within_0_05_of_level", "fraction_within_0_10_of_level",
             "fraction_within_0_25_of_level")
    values = [stats[band] for band in bands]
    assert values == sorted(values)
    assert stats["zero_exactly_representable"] is True


def test_tighter_grid_clips_more_and_steps_less(m22, residuals):
    tight = m22.grid_statistics(residuals, m22.grid_variant("tight_p1").build(residuals, 4),
                                bits=4)
    deployed = m22.grid_statistics(residuals, m22.DEPLOYED.build(residuals, 4), bits=4)
    assert tight["clipping_fraction"] > deployed["clipping_fraction"]
    assert tight["step_mean"] < deployed["step_mean"]


# --- 5: the coupling that defines M22 ------------------------------------------------------


def test_grid_change_moves_the_calibration_signature(m22, ev, residuals):
    deployed = m22.calibration_signature_for(ev, m22.DEPLOYED.build(residuals, 4), bits=4,
                                             calibration_frames=400)
    for variant in m22.GRID_VARIANTS:
        if variant.name == "deployed":
            continue
        other = m22.calibration_signature_for(ev, variant.build(residuals, 4), bits=4,
                                              calibration_frames=400)
        assert other != deployed, variant.name


def test_calibration_signature_depends_only_on_the_residual_grid(m22, ev, residuals):
    """The signature hashes residual scale/zero_point, bits, mode and frame count
    - and nothing else. That is precisely why the residual quantizer cannot be
    changed in isolation, and why the intra quantizer (M18's subject) could."""
    source = inspect.getsource(ev.calibration_signature)
    assert 'calibration["residual_params"]' in source
    assert "intra" not in source.replace("# ", "")
    params = m22.DEPLOYED.build(residuals, 4)
    first = m22.calibration_signature_for(ev, params, bits=4, calibration_frames=400)
    second = m22.calibration_signature_for(ev, params, bits=4, calibration_frames=401)
    assert first != second


def test_deployed_g16_rejects_a_changed_grid(m22, ma, residuals, tmp_path):
    """The executable version of the coupling claim: `check_provenance` raises
    when the grid moves, so a new grid cannot be quietly evaluated under the
    deployed entropy stack."""
    ev_m11 = _load_script("m11_evaluate")
    checkpoint = {"bits": 4, "model_config": {"group_size": m22.M11_G16_GROUP_SIZE},
                  "context_definition_id": ma.context_definition_id(m22.M11_G16_GROUP_SIZE),
                  "calibration_signature": "aaaaaaaaaaaaaaaa",
                  "m10k_identity": "00" * 8}
    ev_m11.check_provenance(checkpoint, signature="aaaaaaaaaaaaaaaa", bits=4,
                            group_size=m22.M11_G16_GROUP_SIZE,
                            context_definition_id=ma.context_definition_id(
                                m22.M11_G16_GROUP_SIZE),
                            m10k_identity=bytes.fromhex("00" * 8))
    with pytest.raises(Exception, match="provenance|calibration"):
        ev_m11.check_provenance(checkpoint, signature="bbbbbbbbbbbbbbbb", bits=4,
                                group_size=m22.M11_G16_GROUP_SIZE,
                                context_definition_id=ma.context_definition_id(
                                    m22.M11_G16_GROUP_SIZE),
                                m10k_identity=bytes.fromhex("00" * 8))


def test_symbol_cache_key_does_not_include_the_grid(m22):
    """The hazard `collect_symbols_for_grid` exists to avoid: `load_or_collect`
    would return deployed-grid symbols for a new grid."""
    md = _load_script("m11_data")
    parameters = set(inspect.signature(md.cache_key).parameters)
    assert not any("residual" in name or "grid" in name or "scale" in name
                   for name in parameters)
    source = inspect.getsource(m22.collect_symbols_for_grid)
    assert "residual_params" in source
    assert "collect_training_symbols" in source


def test_refit_reuses_the_original_fitting_functions(m22):
    """Every downstream step must call the code that produced the deployed
    component, not a reimplementation - otherwise the REFIT arm would confound a
    grid change with a fitting change."""
    source = inspect.getsource(m22.refit_downstream)
    for call in ("mk.build_model(", "mk.train_entropy_model(", "ma.from_m10k(", "ma.train(",
                 "mt.fit_model_codebook(", "m13.fit_recalibrated_frequencies(",
                 "m13.build_recalibrated_codebook("):
        assert call in source, call


def test_checkpoint_provenance_records_everything_phase_11_requires(m22):
    """A checkpoint must never be silently evaluated under a different
    quantizer, so the record must pin the quantizer, the architecture, the
    objective, the data and the code that produced it."""
    source = inspect.getsource(m22.save_refit_checkpoint)
    for field in ("quantizer_identity", "architecture_identity", "entropy_identities",
                  "training", "dataset_identity", "code_identity", "sha256",
                  "grid_signature", "calibration_signature", "selected_epoch",
                  "best_val_a_bits", "final_val_a_bits", "seed", "lambda", "gamma"):
        assert field in source, field
    # The digest must be taken AFTER the write, of the artifact a later run loads.
    assert source.index("torch.save(payload, path)") < source.index("hashlib.sha256(path.read_bytes())")


def test_code_identity_covers_every_script_that_can_change_a_refit(m22):
    identity = m22.code_identity()
    for name in ("m22_residual.py", "m10k_learned_entropy.py", "m10l_shared_codebook.py",
                 "m11_ar_entropy.py", "m11_train.py", "m13_recalibration.py",
                 "m10j_conditional_entropy.py"):
        assert name in identity, name
        assert len(identity[name]) == 16
    assert identity == m22.code_identity()


def test_saved_checkpoints_match_their_recorded_digests():
    """Every checkpoint the sweep wrote must still hash to what was recorded.

    The provenance records are committed but the `.pt` files are git-ignored, so a
    fresh checkout (CI) has records without checkpoints. The record fields are
    checked everywhere; digests are checked for every checkpoint that is present.
    """
    import hashlib
    directory = ROOT / "outputs/m22_residual_freeze_lift/checkpoints"
    if not directory.is_dir():
        pytest.skip("the sweep has not been run in this working tree")
    records = sorted(directory.glob("*.provenance.json"))
    if not records:
        pytest.skip("no refit checkpoints recorded")
    for record_path in records:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        checkpoint = ROOT / record["path"]
        if checkpoint.is_file():
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == record["sha256"], \
                record["path"]
        assert record["training"]["lambda"] == 3e-4
        assert record["training"]["gamma"] is None
        assert record["dataset_identity"]["split_discipline"].startswith("TRAIN fits")
        assert "test" not in " ".join(record["dataset_identity"]["train_sequences"]).lower()


# --- 6: split separation and no TEST access ------------------------------------------------


@pytest.mark.parametrize("script", ["m22_residual", "m22_baseline", "m22_diagnostics",
                                    "m22_sweep", "m22_mechanism", "m22_analysis",
                                    "m22_codebook", "m22_reproduce"])
def test_m22_scripts_never_touch_the_test_split(script):
    """Every script that runs BEFORE the candidate lock. The DAVIS TEST runner is
    deliberately not on this list - it is written only after the lock."""
    source = (ROOT / "scripts" / f"{script}.py").read_text(encoding="utf-8")
    assert 'split="test"' not in source
    assert "test_sequences" not in source


def test_val_b_selection_never_returns_test_sequences(m22):
    source = inspect.getsource(m22.val_b_sequences)
    assert 'split="val"' in source and "[1::2]" in source


def test_train_collection_is_train_only(m22):
    for function in (m22.collect_train_residuals, m22.broad_train_sequences):
        source = inspect.getsource(function)
        assert "test" not in source.lower().replace("latest", "")


def test_refit_selects_checkpoints_on_val_a_not_val_b(m22):
    """M11's own convention: the SELECTION split tunes, VAL-B only reports."""
    source = inspect.getsource(m22.refit_downstream)
    assert "select_mask" in source and "select_set" in source
    assert "val_sequence_index) % 2 == 0" in source


# --- 7: container and production-source compatibility --------------------------------------


@pytest.mark.parametrize("script", ["m22_residual", "m22_baseline", "m22_diagnostics"])
def test_m22_introduces_no_new_container_format(script):
    source = (ROOT / "scripts" / f"{script}.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "nvct_v3" not in lowered and "format_version = 3" not in lowered
    assert "TEMPORAL_FORMAT_VERSION =" not in source


def test_m22_does_not_modify_any_production_source():
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


def test_residual_identity_uses_the_unmodified_model_identity(m22):
    source = inspect.getsource(m22.residual_identity_for)
    assert "ma.model_identity(" in source
    assert "codebook=coding_codebook" in source


# --- 8: frozen baseline reproduction -------------------------------------------------------


def test_recorded_val_b_baseline_matches_m17_through_m21():
    baseline = _load_script("m22_baseline")
    assert baseline.RECORDED_VAL_B_P_RESIDUAL_BYTES == {5: 1_333_275, 4: 908_404, 3: 549_083}
    assert baseline.RECORDED_VAL_B_TOTAL_BYTES == {5: 1_625_012, 4: 1_140_526, 3: 723_381}


def test_frozen_baseline_identities_still_match_m19(m22):
    recorded = json.loads((ROOT / "outputs/m19_reference_error_audit/m19_identities.json")
                          .read_text(encoding="utf-8"))
    path = ROOT / "outputs/m22_residual_freeze_lift/m22_baseline.json"
    if not path.is_file():
        pytest.skip("Phase 0 has not been run in this working tree")
    baseline = json.loads(path.read_text(encoding="utf-8"))
    for point in baseline["rate_points"]:
        expected = recorded[str(point["bits"])]
        assert point["identities"]["residual"] == expected["residual_identity"]
        assert point["identities"]["assign_codebook"] == expected["assign_codebook_id"]
        assert point["identities"]["coding_codebook"] == expected["coding_codebook_id"]
        assert point["identities"]["motion"] == m22.DEPLOYED_MOTION_IDENTITY
        assert point["bytes_match"] is True
    assert baseline["status"] == "CONFIRMED"


def test_recorded_coupling_audit_confirms_the_forced_refit():
    path = ROOT / "outputs/m22_residual_freeze_lift/m22_baseline.json"
    if not path.is_file():
        pytest.skip("Phase 0 has not been run in this working tree")
    baseline = json.loads(path.read_text(encoding="utf-8"))
    for point in baseline["rate_points"]:
        audit = point["coupling_audit"]
        assert audit["calibration_signature_moves_with_the_grid"] is True
        assert audit["g16_checkpoint_rejects_the_new_grid"] is True
        assert audit["container_needs_a_new_field"] is False
        assert audit["symbol_cache_key_includes_the_grid"] is False


# --- 9: diagnostics accounting -------------------------------------------------------------


def test_sensitivity_accumulator_partitions_every_position():
    diagnostics = _load_script("m22_diagnostics")
    accumulator = diagnostics.SensitivityAccumulator(4)
    rng = np.random.default_rng(0)
    size = 512
    real = rng.integers(0, 16, size)
    oracle = rng.integers(0, 16, size)
    accumulator.add(real_symbols=real, oracle_symbols=oracle,
                    real_offset=rng.random(size) * 16,
                    reference_shift_steps=rng.normal(size=size),
                    code_len_real=rng.random(size) * 4,
                    code_len_oracle=rng.random(size) * 4)
    summary = accumulator.to_dict()
    assert summary["positions"] == size
    assert pytest.approx(sum(summary["symbol_displacement"]["fraction_of_positions"])) == 1.0
    assert pytest.approx(sum(summary["symbol_displacement"]["share_of_excess_bits"])) == 1.0
    assert summary["symbol_agreement"] + summary["symbol_change_rate"] == pytest.approx(1.0)
    bands = summary["level_bands"]["fraction_of_positions"]
    assert bands == sorted(bands)


def test_level_distance_is_measured_in_steps():
    """0 means exactly ON a reconstruction level (maximally stable), 0.5 means
    exactly on a decision boundary (maximally fragile). The convention matters
    because it is what makes the reference shift - also in steps - directly
    comparable, and because reading it the other way inverts the conclusion."""
    diagnostics = _load_script("m22_diagnostics")
    accumulator = diagnostics.SensitivityAccumulator(4)
    on_level = np.array([3.0, 7.0, 11.0])
    accumulator.add(real_symbols=np.array([3, 7, 11]), oracle_symbols=np.array([3, 7, 11]),
                    real_offset=on_level, reference_shift_steps=np.zeros(3),
                    code_len_real=np.zeros(3), code_len_oracle=np.zeros(3))
    assert accumulator.to_dict()["mean_level_distance_steps"] == pytest.approx(0.0)

    other = diagnostics.SensitivityAccumulator(4)
    on_decision_boundary = np.array([3.5, 7.5, 11.5])
    other.add(real_symbols=np.array([3, 7, 11]), oracle_symbols=np.array([3, 7, 11]),
              real_offset=on_decision_boundary, reference_shift_steps=np.zeros(3),
              code_len_real=np.zeros(3), code_len_oracle=np.zeros(3))
    assert other.to_dict()["mean_level_distance_steps"] == pytest.approx(0.5)


# --- 10: independent-process reproducibility -----------------------------------------------


def test_grid_variants_are_reproducible_in_an_independent_process(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text(
        "import importlib.util, json, sys\n"
        "import torch\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        f"spec = importlib.util.spec_from_file_location('m22', {str(ROOT / 'scripts' / 'm22_residual.py')!r})\n"
        "m22 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m22)\n"
        "g = torch.Generator().manual_seed(7)\n"
        "residuals = torch.randn(24, 64, 16, 16, generator=g) * 2.0\n"
        "out = {}\n"
        "for v in m22.GRID_VARIANTS:\n"
        "    for bits in (5, 4, 3):\n"
        "        p = v.build(residuals, bits)\n"
        "        out[f'{v.name}@{bits}'] = m22.grid_signature(p)\n"
        "print(json.dumps(out, sort_keys=True))\n", encoding="utf-8")
    runs = [subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                           cwd=ROOT, check=True).stdout for _ in range(2)]
    assert json.loads(runs[0]) == json.loads(runs[1])
    assert len(json.loads(runs[0])) == len(_load_script("m22_residual").GRID_VARIANTS) * 3


# --- 15: resuming a crashed sweep ----------------------------------------------------------


@pytest.fixture
def fake_refit_checkpoint(m22, tmp_path):
    """A minimal but REAL refit checkpoint, written in the exact shape
    `save_refit_checkpoint` produces, so the resume guards are exercised against
    a genuine artifact rather than a stub."""
    import hashlib

    from nvc.compression.quantization import QuantizationParams

    ma_mod = _load_script("m11_ar_entropy")
    ml_mod = _load_script("m10l_shared_codebook")

    channels, bits = 8, 3
    model11 = ma_mod.ChannelContextEntropyModel(
        latent_channels=channels, alphabet=2 ** bits, hidden=4, group_size=4)
    frequencies = np.full((4, 2 ** bits), ml_mod.TOTAL_FREQUENCY // 2 ** bits, dtype=np.int64)
    frequencies[:, 0] += ml_mod.TOTAL_FREQUENCY - frequencies[0].sum()
    codebook = ml_mod.SharedCodebook(frequencies, bits=bits)
    params = QuantizationParams(
        scale=torch.linspace(0.5, 2.0, channels).view(1, channels, 1, 1),
        zero_point=torch.zeros(1, channels, 1, 1), bits=bits, mode="per_channel")

    path = tmp_path / "m22_fake_3bit.pt"
    payload = {
        "m22_variant": "fake", "bits": bits,
        "model_state_dict": model11.state_dict(), "model_config": model11.config_dict(),
        "assign_codebook": codebook.to_dict(), "coding_codebook": codebook.to_dict(),
        "residual_scale": params.scale, "residual_zero_point": params.zero_point,
        "quantizer_identity": {"grid_signature": m22.grid_signature(params),
                               "calibration_signature": "sig-fake",
                               "bits": bits, "mode": params.mode},
        "entropy_identities": {"residual": "00" * 8, "m10k": "11" * 8,
                               "assign_codebook": codebook.codebook_id().hex(),
                               "coding_codebook": codebook.codebook_id().hex(),
                               "residual_entropy_model": "22" * 8},
        "training": {"seed": 42, "selected_epoch": 7, "best_val_a_bits": 2.1,
                     "final_val_a_bits": 2.2, "lambda": 3e-4, "gamma": None},
        "dataset_identity": {"train_p_frames_collected": 10, "val_p_frames_collected": 5,
                             "train_sequences": [], "val_sequences": [],
                             "split_discipline": "TRAIN fits, VAL-A selects"},
        "code_identity": m22.code_identity(),
    }
    torch.save(payload, path)
    record = {"path": str(path).replace("\\", "/"),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "bytes": path.stat().st_size,
              **{k: payload[k] for k in ("quantizer_identity", "architecture_identity",
                                         "entropy_identities", "training", "dataset_identity",
                                         "code_identity") if k in payload}}
    path.with_suffix(".provenance.json").write_text(json.dumps(record, indent=2),
                                                    encoding="utf-8")
    return {"path": path, "record": record, "grid_signature": m22.grid_signature(params)}


def _resume(m22, checkpoint, **overrides):
    sweep = _load_script("m22_sweep")
    kwargs = {"grid_sig": checkpoint["grid_signature"], "signature": "sig-fake"}
    kwargs.update(overrides)
    return sweep.resume_refit(
        checkpoint["path"], m22=m22, ma=_load_script("m11_ar_entropy"),
        ml=_load_script("m10l_shared_codebook"), cx=_load_script("m11_causal_context"),
        device=torch.device("cpu"), **kwargs)


def test_resume_refit_accepts_a_checkpoint_fitted_to_this_grid_by_this_code(
        m22, fake_refit_checkpoint):
    loaded = _resume(m22, fake_refit_checkpoint)
    assert set(loaded["spec"]) == {"model", "zero", "assign_codebook", "coding_codebook",
                                   "identity"}
    assert m22.grid_signature(loaded["residual_params"]) == fake_refit_checkpoint[
        "grid_signature"]


def test_resume_refit_refuses_a_checkpoint_fitted_to_a_different_grid(m22,
                                                                     fake_refit_checkpoint):
    """The whole point of the guard: a stack must never be evaluated under a
    quantizer it was not fitted to."""
    with pytest.raises(ValueError, match="grid signature"):
        _resume(m22, fake_refit_checkpoint, grid_sig="deadbeefdeadbeef")


def test_resume_refit_refuses_a_checkpoint_with_a_different_calibration_signature(
        m22, fake_refit_checkpoint):
    with pytest.raises(ValueError, match="calibration signature"):
        _resume(m22, fake_refit_checkpoint, signature="sig-other")


def test_resume_refit_rebuilds_rather_than_aborting_when_the_code_changed(m22,
                                                                          fake_refit_checkpoint):
    """A stale artifact is only a hazard if it is USED. Rebuilding it is always
    safe, so a code-identity mismatch must fall back to refitting - aborting
    would throw away a multi-hour sweep for no correctness gain. (It did, once.)"""
    record_path = fake_refit_checkpoint["path"].with_suffix(".provenance.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["code_identity"]["m22_residual.py"] = "0" * 16
    record_path.write_text(json.dumps(record), encoding="utf-8")
    assert _resume(m22, fake_refit_checkpoint) is None


def test_resume_refit_still_raises_for_a_quantizer_mismatch_not_merely_skips(m22,
                                                                            fake_refit_checkpoint):
    """The distinction that matters: a wrong GRID must abort, because reusing
    that checkpoint would evaluate a stack under a quantizer it never saw."""
    with pytest.raises(ValueError, match="grid signature"):
        _resume(m22, fake_refit_checkpoint, grid_sig="deadbeefdeadbeef")


def test_resume_refit_refuses_a_tampered_checkpoint(m22, fake_refit_checkpoint):
    with fake_refit_checkpoint["path"].open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(ValueError, match="hashes to"):
        _resume(m22, fake_refit_checkpoint)


def test_sweep_persists_the_report_after_every_row():
    """A sweep is hours long; a crash in the last variant must not discard the
    rows already measured (it did, once)."""
    source = (ROOT / "scripts" / "m22_sweep.py").read_text(encoding="utf-8")
    body = source.split("for variant in variants:", 1)[1].split("\n        print(", 1)[0]
    assert "persist()" in body, "the sweep must write its report inside the variant loop"


# --- 16: Phase 15, the codebook incompatibility is quantified, not repaired ------------------


def test_phase15_uses_the_phase17_lock_rather_than_choosing_its_own_candidate():
    """Phase 15 must describe the candidate that was actually locked, so it
    delegates to the selection rule instead of re-picking a winner."""
    codebook = _load_script("m22_codebook")
    source = inspect.getsource(codebook.locked_candidate)
    assert "analysis.select_candidate" in source
    assert '"refit"' in source, "the incompatibility is measured against a refitted stack"


def test_phase15_never_writes_a_codebook_or_a_checkpoint():
    """Section 15 forbids silent recalibration: this script may only measure."""
    source = (ROOT / "scripts" / "m22_codebook.py").read_text(encoding="utf-8")
    for forbidden in ("fit_model_codebook", "fit_recalibrated_frequencies",
                      "build_recalibrated_codebook", "refit_downstream",
                      "save_refit_checkpoint", "torch.save"):
        assert forbidden not in source, f"Phase 15 must not call {forbidden}"


def test_codebook_incompatibility_reports_both_causes():
    """The STALE arm prices the incompatibility in bytes; this decomposes it into
    assignment drift and table mismatch, which are different repairs."""
    sweep = _load_script("m22_sweep")
    source = inspect.getsource(sweep.codebook_incompatibility)
    for field in ("assignment_drift", "refit_bits_per_symbol", "deployed_bits_per_symbol",
                  "deployed_table_penalty_percent", "symbol_entropy_bits"):
        assert field in source, field


# --- 17: Phase 20, independent-process reproduction ----------------------------------------


def test_reproduction_compares_every_field_section_20_names():
    """Bytes, symbols, entropy lengths and metrics - a reproduction that only
    checked the total would miss a compensating pair of errors."""
    reproduce = _load_script("m22_reproduce")
    fields = set(reproduce.EXACT_FIELDS)
    for required in ("total_container_bytes", "total_residual_bytes", "residual_symbols",
                     "p_frame_ideal_bits", "mean_psnr_db", "mean_msssim", "stream_bpp"):
        assert required in fields, required


def test_reproduction_comparison_is_exact_and_reports_every_mismatch():
    reproduce = _load_script("m22_reproduce")
    recorded = {"total_container_bytes": 723_381, "residual_symbols": 4_030_464,
                "mean_psnr_db": 27.9431}
    assert reproduce.compare_exact(dict(recorded), recorded) == []
    # a single ULP is a failure here: determinism is the property under test, and
    # a tolerance would hide exactly the drift Phase 20 exists to catch
    import math
    drifted = {**recorded, "mean_psnr_db": math.nextafter(27.9431, math.inf)}
    assert [d["field"] for d in reproduce.compare_exact(drifted, recorded)] == ["mean_psnr_db"]
    both = reproduce.compare_exact({**recorded, "total_container_bytes": 723_382,
                                    "residual_symbols": 0}, recorded)
    assert {d["field"] for d in both} == {"total_container_bytes", "residual_symbols"}


def test_reproduction_fails_the_process_when_it_does_not_reproduce():
    """Phase 20 must be usable as a gate, not just as a report."""
    source = (ROOT / "scripts" / "m22_reproduce.py").read_text(encoding="utf-8")
    assert "return 0 if reproduced else 1" in source


def test_reproduction_verifies_the_checkpoint_digest_before_loading_it():
    source = (ROOT / "scripts" / "m22_reproduce.py").read_text(encoding="utf-8")
    assert "load_refit_checkpoint" in source
    mech = _load_script("m22_mechanism")
    assert "sha256" in inspect.getsource(mech.load_refit_checkpoint)


def test_report_path_is_never_rebound_by_a_loop_variable():
    """A real bug, caught mid-run: `for path in stream_dir.glob(...)` rebound the
    same name `persist()` closes over, so every incremental report write landed
    in a just-deleted `.nvct` stream file and the sweep produced no JSON at all.
    The report path must not share a name with any loop variable in `main`."""
    import ast
    source = (ROOT / "scripts" / "m22_sweep.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    persist = next(node for node in ast.walk(main)
                   if isinstance(node, ast.FunctionDef) and node.name == "persist")
    written = {n.id for node in ast.walk(persist) if isinstance(node, ast.Attribute)
               for n in ast.walk(node.value) if isinstance(n, ast.Name)}
    assert written, "persist() must write through a named path"
    loop_targets = {n.id for node in ast.walk(main) if isinstance(node, ast.For)
                    for n in ast.walk(node.target) if isinstance(n, ast.Name)}
    collisions = written & loop_targets
    assert not collisions, (
        f"{collisions} is both closed over by persist() and rebound by a loop in main(); "
        "the report would be written to whatever the loop last assigned")


# --- 18: rate/distortion framing -----------------------------------------------------------


def test_bd_rate_reuses_the_project_methodology_rather_than_reimplementing_it():
    analysis = _load_script("m22_analysis")
    source = inspect.getsource(analysis._bd_rate)
    assert "m10b_evaluate" in source and "_bd_rate_linear" in source


def test_bd_rate_separates_a_real_gain_from_a_slide_along_the_same_curve():
    """The central interpretive risk in M22: a coarser grid buys bytes by coding
    at lower quality. BD-rate must read that as ~0, not as a win."""
    analysis = _load_script("m22_analysis")
    control = [(0.321103, 27.9431), (0.506270, 29.1027), (0.721330, 29.4393)]
    slid = [(0.25, 27.30), (0.40, 28.60), (0.60, 29.20)]       # same curve, lower operating point
    better = [(0.30, 27.9431), (0.48, 29.1027), (0.68, 29.4393)]   # same quality, fewer bits
    assert abs(analysis._bd_rate(control, slid)) < 1.0
    assert analysis._bd_rate(control, better) < -3.0


def test_bd_rate_is_none_when_the_curves_do_not_overlap():
    """Undefined is not zero - reporting 0 for non-overlapping curves would
    invent a result."""
    analysis = _load_script("m22_analysis")
    assert analysis._bd_rate([(0.3, 27.0), (0.5, 28.0)], [(0.3, 40.0), (0.5, 41.0)]) is None


def test_bd_rate_table_needs_at_least_two_rate_points():
    analysis = _load_script("m22_analysis")
    def _row(label, bpp, psnr):
        return {"label": label, "aggregate": {"stream_bpp": bpp, "mean_psnr_db": psnr,
                                              "mean_msssim": psnr / 30.0}}
    single = {"rate_points": [{"candidates": [_row("deployed/stale", 0.32, 27.9),
                                              _row("broad_p001/refit", 0.25, 27.3)]}]}
    assert analysis.bd_rate_table(single) == []
