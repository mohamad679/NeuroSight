"""Smoke-mode benchmark tests — fast, deterministic, and comprehensive.

All tests in this module use n_per_class=3, seed=42, 1 epoch, 2 CV folds.
Expected runtime: < 30s on CPU for the full module.

Tests
-----
- test_benchmark_smoke_runs_fast          — completes under 60s wall time
- test_benchmark_smoke_outputs_json       — JSON is valid and has required keys
- test_benchmark_smoke_contains_synthetic_disclosure — flags present and correct
- test_benchmark_smoke_contains_required_metrics     — test_metrics block present
- test_benchmark_smoke_is_deterministic   — two identical runs produce same scores
- test_benchmark_smoke_does_not_claim_clinical_validity — no forbidden flags
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from evaluation.benchmark import run_benchmark


# ---------------------------------------------------------------------------
# Shared smoke fixture — runs once per module, cached for all tests
# ---------------------------------------------------------------------------

_SMOKE_KWARGS: dict[str, Any] = dict(
    seed=42,
    n_per_class=3,
    cv_folds=2,
    cognitive_epochs=1,
    fusion_epochs=1,
    random_forest_estimators=10,
    gradient_boosting_estimators=10,
)

_REQUIRED_TOP_LEVEL_KEYS = (
    "synthetic_data",
    "clinical_validity",
    "trained_on_real_data",
    "leakage_checked",
    "leakage_check_passed",
    "warning",
    "methodology",
    "dependency_versions",
    "test_metrics",
    "limitations",
    "warnings",
    "seed",
    "config",
    "results",
    "winner",
)

_REQUIRED_METRIC_KEYS = ("accuracy", "macro_f1", "macro_auc", "ece")

_REQUIRED_METHODS = {
    "random_classifier",
    "majority_classifier",
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "mlp_cognitive_only",
    "neurosight_cognitive_only",
    "neurosight_fusion",
}


@pytest.fixture(scope="module")
def smoke_csv(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Return a temp path for the auto-generated smoke CSV."""
    tmp = tmp_path_factory.mktemp("smoke_csv")
    return str(tmp / "ADNIMERGE_smoke.csv")


@pytest.fixture(scope="module")
def smoke_results(smoke_csv: str) -> dict[str, Any]:
    """Run benchmark smoke config once and cache results for all tests in module."""
    return run_benchmark(csv_path=smoke_csv, **_SMOKE_KWARGS)


# ---------------------------------------------------------------------------
# Test 1 — runs fast
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.unit
def test_benchmark_smoke_runs_fast(tmp_path: Path) -> None:
    """Smoke benchmark must complete in under 60 seconds on CPU.

    Uses n_per_class=3, 1 epoch, 10 trees — the minimal real pipeline run.
    """
    csv_path = str(tmp_path / "ADNIMERGE_fast.csv")
    start = time.perf_counter()
    results = run_benchmark(csv_path=csv_path, **_SMOKE_KWARGS)
    elapsed = time.perf_counter() - start

    assert elapsed < 60.0, (
        f"Smoke benchmark took {elapsed:.1f}s — must complete in under 60s on CPU."
    )
    # Also verify the run produced valid results (not an empty/errored dict)
    assert isinstance(results.get("results"), list), "Benchmark must return a results list"
    assert len(results["results"]) == 8, (
        f"Expected 8 methods, got {len(results['results'])}"
    )


# ---------------------------------------------------------------------------
# Test 2 — outputs valid JSON with all required keys
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.unit
def test_benchmark_smoke_outputs_json(smoke_results: dict[str, Any], tmp_path: Path) -> None:
    """Smoke benchmark output is JSON-serializable and contains all required top-level keys."""
    # Verify the dict is JSON-serializable (no numpy types, no non-serializable objects)
    try:
        serialized = json.dumps(smoke_results, indent=2)
    except (TypeError, ValueError) as exc:
        pytest.fail(f"Benchmark output is not JSON-serializable: {exc}")

    # Round-trip: parse back and verify structure
    parsed = json.loads(serialized)
    assert isinstance(parsed, dict), "Benchmark output must be a dict"

    missing_keys = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in parsed]
    assert not missing_keys, (
        f"Required top-level keys missing from benchmark output: {missing_keys}"
    )

    # Verify results list structure
    results_list = parsed["results"]
    assert isinstance(results_list, list), "results must be a list"
    assert len(results_list) > 0, "results list must not be empty"

    methods_found = {row["method"] for row in results_list if isinstance(row, dict)}
    assert methods_found == _REQUIRED_METHODS, (
        f"Method set mismatch. Expected: {_REQUIRED_METHODS}. Got: {methods_found}"
    )

    for row in results_list:
        assert isinstance(row, dict), f"Each result row must be a dict, got {type(row)}"
        for key in ("method", "macro_auc", "macro_f1", "accuracy", "balanced_accuracy", "brier_score"):
            assert key in row, f"Result row missing key '{key}': {row}"

    # Verify winner is a dict with method and macro_auc
    winner = parsed["winner"]
    assert isinstance(winner, dict), "winner must be a dict"
    assert "method" in winner, "winner must have a method key"
    assert "macro_auc" in winner, "winner must have a macro_auc key"


# ---------------------------------------------------------------------------
# Test 3 — synthetic disclosure flags present and correct
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.unit
def test_benchmark_smoke_contains_synthetic_disclosure(smoke_results: dict[str, Any]) -> None:
    """Smoke output must contain all required synthetic/disclosure flags."""
    assert smoke_results.get("synthetic_data") is True, (
        "synthetic_data must be True in benchmark output"
    )
    assert smoke_results.get("clinical_validity") is False, (
        "clinical_validity must be False in benchmark output"
    )
    assert smoke_results.get("trained_on_real_data") is False, (
        "trained_on_real_data must be False in benchmark output"
    )
    assert smoke_results.get("leakage_checked") is True, (
        "leakage_checked must be True in benchmark output"
    )
    assert smoke_results.get("leakage_check_passed") is True, (
        "leakage_check_passed must be True — orthogonal noise embeddings should not correlate with cognitive features"
    )

    # Verify warning text contains the synthetic disclosure
    warning = smoke_results.get("warning", "")
    assert isinstance(warning, str) and len(warning) > 10, (
        "warning field must be a non-empty string"
    )
    assert "SYNTHETIC" in warning.upper(), (
        f"warning must contain 'SYNTHETIC'. Got: {warning!r}"
    )
    assert "NOT CLINICAL" in warning.upper() or "not clinical" in warning.lower(), (
        f"warning must contain 'not clinical'. Got: {warning!r}"
    )

    # Limitations and warnings must be non-empty lists
    limitations = smoke_results.get("limitations", [])
    assert isinstance(limitations, list) and len(limitations) > 0, (
        "limitations must be a non-empty list"
    )
    warnings_list = smoke_results.get("warnings", [])
    assert isinstance(warnings_list, list) and len(warnings_list) > 0, (
        "warnings must be a non-empty list"
    )


# ---------------------------------------------------------------------------
# Test 4 — test_metrics block present at top level with required keys
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.unit
def test_benchmark_smoke_contains_required_metrics(smoke_results: dict[str, Any]) -> None:
    """Smoke output must have a test_metrics dict with accuracy, macro_f1, macro_auc, ece."""
    assert "test_metrics" in smoke_results, (
        "test_metrics key missing from benchmark output (required for quality gate compatibility)"
    )

    test_metrics = smoke_results["test_metrics"]
    assert isinstance(test_metrics, dict), (
        f"test_metrics must be a dict, got {type(test_metrics)}"
    )

    missing_metrics = [k for k in _REQUIRED_METRIC_KEYS if k not in test_metrics]
    assert not missing_metrics, (
        f"test_metrics missing required keys: {missing_metrics}"
    )

    for key in _REQUIRED_METRIC_KEYS:
        val = test_metrics[key]
        assert isinstance(val, (int, float)), (
            f"test_metrics['{key}'] must be numeric, got {type(val)}: {val!r}"
        )
        assert not (val != val), f"test_metrics['{key}'] must not be NaN"  # NaN check

    # Sanity range checks — smoke results are from synthetic data but must be plausible
    assert 0.0 <= test_metrics["accuracy"] <= 1.0, (
        f"accuracy out of range: {test_metrics['accuracy']}"
    )
    assert 0.0 <= test_metrics["macro_f1"] <= 1.0, (
        f"macro_f1 out of range: {test_metrics['macro_f1']}"
    )
    assert 0.0 <= test_metrics["macro_auc"] <= 1.0, (
        f"macro_auc out of range: {test_metrics['macro_auc']}"
    )
    assert 0.0 <= test_metrics["ece"] <= 1.0, (
        f"ece out of range: {test_metrics['ece']}"
    )


# ---------------------------------------------------------------------------
# Test 5 — deterministic: identical inputs → identical outputs
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.unit
def test_benchmark_smoke_is_deterministic(tmp_path: Path) -> None:
    """Two smoke runs with identical seed and config must produce identical metric values."""
    csv_path = str(tmp_path / "ADNIMERGE_det.csv")

    first = run_benchmark(csv_path=csv_path, **_SMOKE_KWARGS)
    second = run_benchmark(csv_path=csv_path, **_SMOKE_KWARGS)

    def _extract_scores(result: dict[str, Any]) -> list[tuple[str, float, float]]:
        return sorted(
            [
                (
                    str(row["method"]),
                    round(float(row["macro_auc"]), 6),
                    round(float(row["macro_f1"]), 6),
                )
                for row in result["results"]
                if isinstance(row, dict)
            ],
            key=lambda t: t[0],
        )

    first_scores = _extract_scores(first)
    second_scores = _extract_scores(second)

    assert first_scores == second_scores, (
        "Benchmark is not deterministic — results differ between identical runs.\n"
        f"First:  {first_scores}\n"
        f"Second: {second_scores}"
    )

    # Also verify test_metrics are identical
    assert first["test_metrics"] == second["test_metrics"], (
        "test_metrics differ between identical runs — non-determinism detected.\n"
        f"First:  {first['test_metrics']}\n"
        f"Second: {second['test_metrics']}"
    )


# ---------------------------------------------------------------------------
# Test 6 — does not claim clinical validity
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.unit
def test_benchmark_smoke_does_not_claim_clinical_validity(smoke_results: dict[str, Any]) -> None:
    """Smoke output must not make any clinical validity or regulatory claims."""
    # Direct flag assertions
    assert smoke_results.get("clinical_validity") is False, (
        "clinical_validity must be False — never True for this benchmark"
    )
    assert smoke_results.get("trained_on_real_data") is False, (
        "trained_on_real_data must be False"
    )

    # Serialize and check for forbidden strings in the entire output
    serialized = json.dumps(smoke_results).lower()
    forbidden_claims = [
        "fda approved",
        "clinically validated model",
        "ready for clinical use",
        "diagnostic accuracy on real patients",
        "autonomous diagnosis",
        "clinical_validity\": true",
        "trained_on_real_data\": true",
    ]
    found_claims = [claim for claim in forbidden_claims if claim in serialized]
    assert not found_claims, (
        f"Forbidden clinical claims found in benchmark output: {found_claims}"
    )

    # methodology must be present and mention synthetic data
    methodology = smoke_results.get("methodology", "")
    assert "synthetic" in methodology.lower(), (
        "methodology field must mention 'synthetic' data"
    )

    # winner must not make clinical claims
    winner = smoke_results.get("winner", {})
    assert isinstance(winner, dict), "winner must be a dict"
    # winner.macro_auc should be in [0, 1] — a simple sanity bound, not a clinical claim
    winner_auc = float(winner.get("macro_auc", -1))
    assert 0.0 <= winner_auc <= 1.0, (
        f"winner.macro_auc must be in [0,1], got {winner_auc}"
    )
