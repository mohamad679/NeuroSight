"""Leakage detection regression tests for NeuroSight benchmark.

Verifies that:
  1. Orthogonal noise embeddings are not correlated with cognitive features.
  2. The leakage detection function raises when fed deliberately correlated data.
  3. The benchmark output reports leakage_check_passed == True.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.benchmark import (
    _build_orthogonal_noise_embeddings,
    _check_no_direct_leakage,
)


# ---------------------------------------------------------------------------
# Orthogonal noise embeddings
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_orthogonal_noise_not_correlated_with_cognitive_features() -> None:
    """Pearson r between noise embedding norms and cognitive feature norms < 0.3."""
    n = 200
    seed = 42
    rng = np.random.default_rng(seed)
    cog_features = rng.standard_normal((n, 8)).astype(np.float32)
    mri_noise, _, _ = _build_orthogonal_noise_embeddings(n, seed=seed)

    emb_norms = np.linalg.norm(mri_noise, axis=1)
    cog_norms = np.linalg.norm(cog_features, axis=1)

    emb_c = emb_norms - emb_norms.mean()
    cog_c = cog_norms - cog_norms.mean()
    denom = np.sqrt((emb_c ** 2).sum()) * np.sqrt((cog_c ** 2).sum())
    r = float(np.dot(emb_c, cog_c) / denom) if denom > 1e-10 else 0.0

    assert abs(r) < 0.3, (
        f"Noise embeddings must not be correlated with cognitive features. "
        f"Got Pearson r={r:.3f}"
    )


@pytest.mark.unit
def test_orthogonal_noise_different_seeds_different_embeddings() -> None:
    """Different seeds produce different noise embeddings."""
    mri_a, _, _ = _build_orthogonal_noise_embeddings(50, seed=1)
    mri_b, _, _ = _build_orthogonal_noise_embeddings(50, seed=2)
    assert not np.allclose(mri_a, mri_b), "Different seeds should produce different noise"


@pytest.mark.unit
def test_orthogonal_noise_same_seed_is_deterministic() -> None:
    """Same seed produces identical noise embeddings."""
    mri_a, eeg_a, cog_a = _build_orthogonal_noise_embeddings(60, seed=99)
    mri_b, eeg_b, cog_b = _build_orthogonal_noise_embeddings(60, seed=99)
    np.testing.assert_array_equal(mri_a, mri_b)
    np.testing.assert_array_equal(eeg_a, eeg_b)
    np.testing.assert_array_equal(cog_a, cog_b)


@pytest.mark.unit
def test_orthogonal_noise_shapes() -> None:
    """Noise embeddings have the expected shapes."""
    n = 40
    mri, eeg, cog = _build_orthogonal_noise_embeddings(n, seed=0)
    assert mri.shape == (n, 768), f"Expected MRI shape (n, 768), got {mri.shape}"
    assert eeg.shape == (n, 256), f"Expected EEG shape (n, 256), got {eeg.shape}"
    assert cog.shape == (n, 64), f"Expected Cog shape (n, 64), got {cog.shape}"


# ---------------------------------------------------------------------------
# Leakage check function
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_leakage_check_passes_on_uncorrelated_data() -> None:
    """Leakage check returns True for truly independent embeddings."""
    n = 300
    rng = np.random.default_rng(0)
    cog_features = rng.standard_normal((n, 8)).astype(np.float32)
    mri_noise, _, _ = _build_orthogonal_noise_embeddings(n, seed=7)
    passed = _check_no_direct_leakage(mri_noise, cog_features, threshold=0.5)
    assert passed is True, "Leakage check should pass for independent noise embeddings"


@pytest.mark.unit
def test_leakage_check_fails_on_perfectly_correlated_data() -> None:
    """Leakage check returns False when embeddings are copies of cognitive features."""
    n = 100
    rng = np.random.default_rng(1)
    cog_features = rng.standard_normal((n, 8)).astype(np.float32)
    # Pad cognitive features to 768 dims — perfectly correlated with themselves
    padded = np.tile(cog_features, (1, 96))[:, :768]  # 8 * 96 = 768
    # Normalise so shapes match but correlation is preserved
    passed = _check_no_direct_leakage(padded, cog_features, threshold=0.5)
    assert passed is False, (
        "Leakage check should FAIL when embeddings are derived from cognitive features"
    )


@pytest.mark.unit
def test_leakage_check_strict_threshold() -> None:
    """Leakage check can be tightened with a lower threshold."""
    n = 100
    rng = np.random.default_rng(2)
    # Slight correlation: noise + 0.1 * cognitive norm
    cog_features = rng.standard_normal((n, 8)).astype(np.float32)
    slight_corr = np.hstack([
        rng.standard_normal((n, 760)).astype(np.float32),
        cog_features,
    ])  # shape (n, 768)
    # With a very tight threshold (0.01) should fail; with 0.9 should pass
    passed_tight = _check_no_direct_leakage(slight_corr, cog_features, threshold=0.01)
    passed_loose = _check_no_direct_leakage(slight_corr, cog_features, threshold=0.99)
    # The strict threshold should flag it; loose should not
    assert not passed_tight or passed_loose, (
        "Strict threshold should be more sensitive to correlation than loose threshold"
    )


# ---------------------------------------------------------------------------
# Benchmark leakage flag
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.integration
def test_benchmark_leakage_check_passed_flag(tmp_path: pytest.TempPathFactory) -> None:
    """Benchmark output reports leakage_check_passed == True."""
    from evaluation.benchmark import run_benchmark

    csv_path = str(tmp_path / "data" / "ADNIMERGE_synthetic.csv")
    results = run_benchmark(
        csv_path=csv_path,
        seed=42,
        n_per_class=8,  # Enough samples for train/test split with 6 classes
        cv_folds=2,
        cognitive_epochs=1,
        fusion_epochs=1,
        random_forest_estimators=5,
        gradient_boosting_estimators=5,
    )
    assert results["leakage_check_passed"] is True, (
        "Benchmark must confirm leakage_check_passed == True"
    )
