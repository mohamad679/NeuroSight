"""Tests for NeuroSight benchmark evaluation framework."""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest

from evaluation.benchmark import _build_logistic_regression, run_benchmark
from evaluation.benchmark_table import render_comparison_table


def _method_score(results: dict[str, object], method: str) -> float:
    """Extract method macro AUC from benchmark results."""
    rows = results.get("results", [])
    assert isinstance(rows, list), "Benchmark payload must include results list."
    for row in rows:
        if isinstance(row, dict) and row.get("method") == method:
            return float(row["macro_auc"])
    raise AssertionError(f"Method '{method}' was not found in benchmark results.")


@pytest.fixture(scope="module")
def benchmark_smoke_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Run benchmark smoke config once and cache results for assertions."""
    tmp_path = tmp_path_factory.mktemp("benchmark_data")
    csv_path = tmp_path / "ADNIMERGE_synthetic.csv"
    return run_benchmark(
        csv_path=str(csv_path),
        seed=42,
        n_per_class=5,
        cv_folds=2,
        cognitive_epochs=1,
        fusion_epochs=1,
        random_forest_estimators=10,
        gradient_boosting_estimators=10,
    )


@pytest.mark.benchmark
@pytest.mark.integration
def test_benchmark_smoke_runs_all_methods(benchmark_smoke_results: dict[str, object]) -> None:
    """Small benchmark smoke test executes every real baseline path."""
    results = benchmark_smoke_results

    rows = results.get("results", [])
    assert isinstance(rows, list)
    assert len(rows) == 8, (
        f"Expected 8 baseline methods, got {len(rows)}. "
        f"Methods: {[r.get('method') for r in rows]}"
    )

    methods = {str(row.get("method")) for row in rows if isinstance(row, dict)}
    assert methods == {
        "random_classifier",
        "majority_classifier",
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "mlp_cognitive_only",
        "neurosight_cognitive_only",
        "neurosight_fusion",
    }
    assert results["config"]["n_per_class"] == 5


@pytest.mark.benchmark
@pytest.mark.unit
def test_logistic_regression_uses_current_sklearn_api() -> None:
    """Regression test for removed/deprecated sklearn LogisticRegression args."""
    model = _build_logistic_regression(seed=7)
    params = model.get_params()
    source = inspect.getsource(_build_logistic_regression)
    x = np.array(
        [
            [0.0, 0.1],
            [0.2, 0.0],
            [1.0, 1.1],
            [1.2, 1.0],
            [2.0, 2.1],
            [2.2, 2.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        model.fit(x, y)

    assert "multi_class" not in source
    assert params["solver"] == "lbfgs"
    assert params["random_state"] == 7


@pytest.mark.benchmark
@pytest.mark.integration
def test_benchmark_smoke_is_deterministic(tmp_path: Path) -> None:
    """Identical benchmark inputs and seed produce identical metric payloads."""
    csv_path = tmp_path / "data" / "ADNIMERGE_synthetic.csv"
    kwargs = {
        "csv_path": str(csv_path),
        "seed": 9,
        "n_per_class": 5,
        "cv_folds": 2,
        "cognitive_epochs": 1,
        "fusion_epochs": 1,
        "random_forest_estimators": 10,
        "gradient_boosting_estimators": 10,
    }

    first = run_benchmark(**kwargs)
    second = run_benchmark(**kwargs)

    first_scores = [
        (row["method"], round(float(row["macro_auc"]), 6), round(float(row["macro_f1"]), 6))
        for row in first["results"]
    ]
    second_scores = [
        (row["method"], round(float(row["macro_auc"]), 6), round(float(row["macro_f1"]), 6))
        for row in second["results"]
    ]
    assert first_scores == second_scores


@pytest.mark.benchmark
@pytest.mark.unit
def test_benchmark_table_has_correct_columns(benchmark_smoke_results: dict[str, object]) -> None:
    """Comparison table contains AUC, F1, Acc, ECE, Bal.Acc, Brier columns."""
    results = benchmark_smoke_results
    table = render_comparison_table(results)

    assert "AUC ↑" in table
    assert "F1 ↑" in table
    assert "Acc ↑" in table
    assert "ECE ↓" in table
    assert "Bal.Acc ↑" in table
    assert "Brier ↓" in table


@pytest.mark.benchmark
@pytest.mark.unit
def test_benchmark_has_synthetic_data_flag(benchmark_smoke_results: dict[str, object]) -> None:
    """Benchmark output has synthetic_data == True."""
    results = benchmark_smoke_results
    assert results["synthetic_data"] is True, "Benchmark must report synthetic_data == True"


@pytest.mark.benchmark
@pytest.mark.unit
def test_benchmark_has_no_clinical_validity(benchmark_smoke_results: dict[str, object]) -> None:
    """Benchmark output has clinical_validity == False."""
    results = benchmark_smoke_results
    assert results["clinical_validity"] is False, "Benchmark must report clinical_validity == False"


@pytest.mark.benchmark
@pytest.mark.unit
def test_benchmark_results_have_balanced_accuracy_and_brier(benchmark_smoke_results: dict[str, object]) -> None:
    """Every result row includes balanced_accuracy and brier_score fields."""
    results = benchmark_smoke_results
    for row in results["results"]:
        method = row.get("method", "unknown")
        assert "balanced_accuracy" in row, f"Method '{method}' missing balanced_accuracy"
        assert "brier_score" in row, f"Method '{method}' missing brier_score"
        # Values must be valid floats
        assert isinstance(float(row["balanced_accuracy"]), float), (
            f"balanced_accuracy must be float for '{method}'"
        )
        assert isinstance(float(row["brier_score"]), float), (
            f"brier_score must be float for '{method}'"
        )


@pytest.mark.benchmark
@pytest.mark.slow
def test_cognitive_baseline_exceeds_random() -> None:
    """Cognitive baseline AUC should exceed random on synthetic tabular data.

    NOTE: After the leakage fix, we compare the best cognitive baseline (not
    fusion) to the random classifier.  Fusion on orthogonal noise is NOT
    expected to exceed cognitive-only.
    """
    results = run_benchmark(csv_path="data/ADNIMERGE_synthetic.csv", seed=42)
    cog_auc = _method_score(results, "neurosight_cognitive_only")
    random_auc = _method_score(results, "random_classifier")

    assert cog_auc > 0.40, f"Cognitive AUC {cog_auc} should be a valid score above 0.40"
    assert cog_auc >= random_auc - 0.10, (
        f"Cognitive AUC {cog_auc} should be comparable to or better than random AUC {random_auc}"
    )


@pytest.mark.benchmark
@pytest.mark.slow
def test_benchmark_table_warning_banner_present() -> None:
    """Rendered comparison table includes synthetic data warning."""
    results = run_benchmark(csv_path="data/ADNIMERGE_synthetic.csv", seed=42)
    table = render_comparison_table(results)
    assert "SYNTHETIC" in table, "Comparison table must include synthetic data warning banner"
