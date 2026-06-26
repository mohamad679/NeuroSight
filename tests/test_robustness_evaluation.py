"""Smoke tests for NeuroSight robustness evaluation functions."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.robustness import (
    evaluate_class_imbalance,
    evaluate_missing_fields,
    evaluate_noisy_inputs,
    evaluate_out_of_range,
    evaluate_site_shift,
)


def _build_logistic_regression():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(solver="lbfgs", max_iter=100, random_state=0)


def _small_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a small synthetic dataset for smoke tests."""
    rng = np.random.default_rng(seed)
    n_per_class = 6
    n_classes = 6
    features_list = []
    labels_list = []
    for cls_id in range(n_classes):
        x = rng.normal(loc=float(cls_id), scale=0.5, size=(n_per_class, 8)).astype(np.float32)
        features_list.append(x)
        labels_list.extend([cls_id] * n_per_class)
    return np.vstack(features_list), np.array(labels_list, dtype=np.int64)


@pytest.mark.unit
def test_noisy_inputs_returns_valid_structure() -> None:
    """evaluate_noisy_inputs returns required keys with valid AUC."""
    x, y = _small_dataset()
    result = evaluate_noisy_inputs(
        x, y,
        noise_levels=[0.1, 0.5],
        model_builder=_build_logistic_regression,
        seed=0,
    )
    assert result["synthetic_data"] is True
    assert result["clinical_validity"] is False
    assert result["test_type"] == "noisy_inputs"
    assert "results" in result
    assert "noise_0.0" in result["results"]  # baseline key uses f"noise_{std:.1f}" format
    for key in ("noise_0.10", "noise_0.50"):
        assert key in result["results"], f"Missing key: {key}"
        auc = result["results"][key]["mean_auc"]
        assert isinstance(auc, float), f"AUC should be float for {key}"


@pytest.mark.unit
def test_missing_fields_returns_valid_structure() -> None:
    """evaluate_missing_fields returns required keys and imputation label."""
    x, y = _small_dataset()
    result = evaluate_missing_fields(
        x, y,
        missing_fractions=[0.2, 0.5],
        model_builder=_build_logistic_regression,
        seed=0,
    )
    assert result["synthetic_data"] is True
    assert result["imputation"] == "median"
    assert "noise_0.0" not in result["results"]  # key format is missing_*
    assert "missing_0.20" in result["results"]


@pytest.mark.unit
def test_out_of_range_returns_clean_and_perturbed() -> None:
    """evaluate_out_of_range returns both 'clean' and 'out_of_range' result keys."""
    x, y = _small_dataset()
    result = evaluate_out_of_range(x, y, model_builder=_build_logistic_regression, seed=0)
    assert result["synthetic_data"] is True
    assert result["test_type"] == "out_of_range_values"
    assert "clean" in result["results"]
    assert "out_of_range" in result["results"]
    # Both should yield valid float AUCs
    for k in ("clean", "out_of_range"):
        auc = result["results"][k]["mean_auc"]
        assert isinstance(auc, float), f"AUC for '{k}' must be float"
        assert not (auc < 0 or auc > 1.1), f"AUC out of expected range: {auc}"


@pytest.mark.unit
def test_class_imbalance_per_ratio_has_n_samples() -> None:
    """evaluate_class_imbalance results include n_samples per ratio."""
    x, y = _small_dataset()
    result = evaluate_class_imbalance(
        x, y,
        imbalance_ratios=[1.0, 0.5],
        model_builder=_build_logistic_regression,
        seed=0,
    )
    assert result["synthetic_data"] is True
    assert result["test_type"] == "class_imbalance"
    for key in result["results"]:
        assert "n_samples" in result["results"][key], (
            f"Expected 'n_samples' in results['{key}']"
        )


@pytest.mark.unit
def test_site_shift_returns_per_site_results() -> None:
    """evaluate_site_shift returns one result per hold-out site."""
    x, y = _small_dataset()
    n_sites = 3
    result = evaluate_site_shift(
        x, y,
        model_builder=_build_logistic_regression,
        n_sites=n_sites,
        seed=0,
    )
    assert result["synthetic_data"] is True
    assert result["test_type"] == "site_shift"
    assert result["n_sites"] == n_sites
    for site_id in range(n_sites):
        key = f"hold_out_site_{site_id}"
        assert key in result["results"], f"Missing site result: {key}"


@pytest.mark.unit
def test_robustness_functions_produce_provenance_metadata() -> None:
    """All robustness functions include synthetic_data and clinical_validity."""
    x, y = _small_dataset()
    results = [
        evaluate_noisy_inputs(x, y, [0.1], _build_logistic_regression, seed=0),
        evaluate_missing_fields(x, y, [0.2], _build_logistic_regression, seed=0),
        evaluate_out_of_range(x, y, _build_logistic_regression, seed=0),
        evaluate_class_imbalance(x, y, [1.0], _build_logistic_regression, seed=0),
        evaluate_site_shift(x, y, _build_logistic_regression, n_sites=2, seed=0),
    ]
    for r in results:
        assert r.get("synthetic_data") is True, "Missing synthetic_data field"
        assert r.get("clinical_validity") is False, "Missing clinical_validity field"
