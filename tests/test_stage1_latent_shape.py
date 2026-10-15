"""Tests for `scripts/stage1_latent_shape.py`.

The expensive parts (400 TRAIN latents, 719 DAVIS frames, the arithmetic coder)
are out of scope here. What is tested is the arithmetic the conclusions rest on,
plus the one finding about the production code that this script exists to pin:
`allocate_bits_per_channel` is a no-op at its default `max_bits`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def shape():
    spec = importlib.util.spec_from_file_location(
        "stage1_latent_shape", ROOT / "scripts" / "stage1_latent_shape.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the finding about production code -------------------------------------------


def test_the_library_bit_allocator_is_a_no_op_at_its_default_max_bits():
    """`allocate_bits_per_channel` caps every channel at `average_bits` AND
    rebalances the realized average back to `average_bits`. Those two constraints
    together are only satisfiable by giving every channel exactly `average_bits`,
    so the function cannot reallocate anything at its default.

    This is why the diagnostic carries its own un-rebalanced version. The test
    pins it so that if the default is ever fixed, this fails and says so rather
    than the diagnostic quietly measuring something else.
    """
    from nvc.compression.calibration import allocate_bits_per_channel

    generator = torch.Generator().manual_seed(0)
    # Wildly different per-channel variances: if reallocation can ever happen,
    # it happens here.
    scales = torch.tensor([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4])
    latents = torch.randn(32, 8, 4, 4, generator=generator) * scales.reshape(1, -1, 1, 1)

    allocation = allocate_bits_per_channel(latents, average_bits=4)

    assert set(allocation) == {4}, (
        f"expected the documented no-op, got {sorted(set(allocation))} - if the "
        "default max_bits was raised, the diagnostic's own arm should be revisited")


def test_raising_max_bits_lets_the_library_allocator_actually_reallocate():
    """The companion to the above: the formula itself works, it is the default cap
    that neutralizes it. Establishes that the no-op is a configuration artifact
    and not a broken implementation."""
    from nvc.compression.calibration import allocate_bits_per_channel

    generator = torch.Generator().manual_seed(0)
    scales = torch.tensor([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4])
    latents = torch.randn(32, 8, 4, 4, generator=generator) * scales.reshape(1, -1, 1, 1)

    allocation = allocate_bits_per_channel(latents, average_bits=4, max_bits=8)

    assert len(set(allocation)) > 1, "still uniform even with headroom"
    # Monotone in variance: the quietest channel never gets more than the busiest.
    assert allocation[0] <= allocation[-1]


# --- the diagnostic's own arithmetic ---------------------------------------------


def test_textbook_allocation_gives_quiet_channels_fewer_bits(shape):
    variances = np.array([1e-4, 1e-2, 1.0, 100.0, 1e4])

    allocation = shape._textbook_allocation(variances, average_bits=4)

    assert allocation[0] < allocation[-1], "not ordered by variance"
    assert max(allocation) <= 4, "a channel exceeded the table's bit depth"
    assert min(allocation) >= 1
    # The point of dropping the rebalance: the average must be free to fall.
    assert float(np.mean(allocation)) < 4.0


def test_textbook_allocation_is_uniform_when_every_channel_is_equal(shape):
    """No reallocation to make when there is no variance spread - the formula's
    log term is zero everywhere."""
    allocation = shape._textbook_allocation(np.full(16, 3.5), average_bits=4)

    assert set(allocation) == {4}


def test_textbook_allocation_survives_an_exactly_zero_variance_channel(shape):
    """A dead channel has variance 0 and log2(0) is -inf. Real latents have them:
    the full run found 58 of Stage 1's 192 channels below 1% of the strongest."""
    allocation = shape._textbook_allocation(
        np.array([0.0, 0.0, 1.0, 100.0]), average_bits=4)

    assert all(np.isfinite(allocation))
    assert min(allocation) >= 1


def test_cost_versus_energy_reports_decoupling_when_cost_is_flat(shape):
    """The script's central claim is that near-idle channels still pay nearly
    full price. This checks the statistic actually detects that pattern: energies
    spanning four orders of magnitude against an almost flat cost."""
    energy = np.logspace(-4, 0, 64)
    bits_per_element = np.full(64, 3.5)

    relation = shape.cost_versus_energy(
        energy, bits_per_element, np.full(64, 3.0), bits=4)

    assert relation["energy_ratio_busiest_over_quietest"] > 100
    assert relation["cost_ratio_busiest_over_quietest"] == pytest.approx(1.0)
    assert relation["quietest_quartile"]["fraction_of_full_bit_depth"] == pytest.approx(0.875)


def test_cost_versus_energy_reports_coupling_when_cost_tracks_energy(shape):
    """The negative control: a stack that DOES price quiet channels cheaply must
    not be reported as decoupled, or the statistic proves nothing."""
    energy = np.logspace(-4, 0, 64)
    bits_per_element = np.linspace(0.05, 4.0, 64)  # cost rises with energy

    relation = shape.cost_versus_energy(
        energy, bits_per_element, np.full(64, 3.0), bits=4)

    assert relation["spearman_energy_vs_bits"] == pytest.approx(1.0, abs=1e-6)
    # linspace(0.05, 4.0) puts the quartile means at ~0.52 and ~3.5, so ~6.8x.
    assert relation["cost_ratio_busiest_over_quietest"] > 5
    assert relation["quietest_quartile"]["fraction_of_full_bit_depth"] < 0.15


def test_profile_summary_counts_the_channels_carrying_the_energy(shape):
    """Four channels hold everything, sixty hold nothing."""
    energy = np.concatenate([np.full(4, 100.0), np.zeros(60)])

    summary = shape.profile_summary(energy)

    assert summary["channels"] == 64
    assert summary["channels_for_50pct_energy"] == 2
    assert summary["channels_for_90pct_energy"] == 4
    assert summary["channels_below_1pct_of_max"] == 60


def test_profile_summary_on_a_flat_profile_needs_most_of_the_channels(shape):
    summary = shape.profile_summary(np.full(64, 1.0))

    assert summary["channels_for_90pct_energy"] == 58
    assert summary["channels_below_1pct_of_max"] == 0
    assert summary["max_over_median_energy"] == pytest.approx(1.0)


# --- the shape claim the recommendation rests on ---------------------------------


def test_a_64_channel_stage_1_keeps_most_of_the_capacity():
    """The recommendation is to narrow the latent, not shrink the model. It only
    holds if the parameters live in the body rather than the latent projection."""
    from nvc.models import ResidualGDNAutoencoder

    wide = ResidualGDNAutoencoder(latent_channels=192, base_channels=192,
                                  residual_blocks=1)
    narrow = ResidualGDNAutoencoder(latent_channels=64, base_channels=192,
                                    residual_blocks=1)

    assert narrow.num_parameters() > 0.8 * wide.num_parameters(), (
        "narrowing the latent gutted the model, so the two are not comparable")
    # And it is still an order of magnitude past the model it has to beat.
    from nvc.models import BaselineAutoencoder
    assert narrow.num_parameters() > 10 * BaselineAutoencoder(
        latent_channels=64).num_parameters()


def test_a_64_channel_latent_restores_symbol_parity_with_the_baseline():
    """The whole rate penalty is symbol count: same stride, 3x the channels. This
    pins the arithmetic the conclusion is built on."""
    import torch

    from nvc.models import BaselineAutoencoder, ResidualGDNAutoencoder

    image = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        baseline = BaselineAutoencoder(latent_channels=64).eval().encode(image)
        wide = ResidualGDNAutoencoder(latent_channels=192, base_channels=192,
                                      residual_blocks=1).eval().encode(image)
        narrow = ResidualGDNAutoencoder(latent_channels=64, base_channels=192,
                                        residual_blocks=1).eval().encode(image)

    assert wide.numel() == 3 * baseline.numel()
    assert narrow.numel() == baseline.numel()
    # Same spatial stride in all three - the channel count is the only difference.
    assert baseline.shape[-2:] == wide.shape[-2:] == narrow.shape[-2:]


def test_the_diagnostic_modifies_no_production_source():
    """Repo convention for a diagnostic (see tests/test_m22_*). Scoped to src/nvc/
    so an unrelated dirty file elsewhere is not a false positive."""
    import subprocess

    result = subprocess.run(["git", "status", "--short"], capture_output=True,
                            text=True, cwd=ROOT, check=False)
    modified = [line for line in result.stdout.splitlines()
                if line[:2].strip() in {"M", "A", "D", "R"}]
    production = [line for line in modified if "src/nvc/" in line.replace("\\", "/")]

    assert production == [], f"unexpected production modifications: {production}"
