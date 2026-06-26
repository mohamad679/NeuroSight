"""Mathematically-grounded unit tests for NeuroSight evaluation metrics.

Each test verifies a known analytical property of the metric function,
not just that it runs without error.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import (
    compute_auc_roc,
    compute_balanced_accuracy,
    compute_brier_score,
    compute_confusion_matrix,
    compute_ece,
    compute_per_class_metrics,
)


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


def test_ece_perfect_calibration_is_near_zero() -> None:
    """Calibration where confidence = 1.0 and every prediction is correct → ECE = 0."""
    n = 100
    n_classes = 3
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, n_classes, size=n)
    # Probability = 1.0 for the true class → confidence = 1.0, accuracy = 1.0 → ECE = 0
    y_prob = np.zeros((n, n_classes), dtype=np.float64)
    y_prob[np.arange(n), y_true] = 1.0
    ece = compute_ece(y_true, y_prob)
    assert ece < 0.01, f"Perfect predictions with confidence=1.0 should yield ECE≈0, got {ece}"


def test_ece_overconfident_wrong_predictions_is_high() -> None:
    """All-wrong predictions with confidence=1.0 yields ECE near 1."""
    n_classes = 4
    n = 100
    y_true = np.zeros(n, dtype=np.int64)        # true class is 0
    y_prob = np.zeros((n, n_classes), dtype=np.float64)
    y_prob[:, 1] = 1.0                           # always predict class 1
    ece = compute_ece(y_true, y_prob)
    assert ece >= 0.85, f"Overconfident wrong predictions should yield ECE≥0.85, got {ece}"


def test_ece_in_valid_range() -> None:
    """ECE is always in [0, 1]."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 6, size=300)
    y_prob = rng.dirichlet(np.ones(6), size=300)
    ece = compute_ece(y_true, y_prob)
    assert 0.0 <= ece <= 1.0, f"ECE must be in [0,1], got {ece}"


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


def test_brier_score_perfect_is_zero() -> None:
    """Brier score is 0 when predicted probability = 1 for the true class."""
    n = 50
    n_classes = 4
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, n_classes, size=n)
    y_prob = np.zeros((n, n_classes), dtype=np.float64)
    y_prob[np.arange(n), y_true] = 1.0
    brier = compute_brier_score(y_true, y_prob)
    assert brier == pytest.approx(0.0, abs=1e-9), f"Perfect probs → Brier=0, got {brier}"


def test_brier_score_uniform_is_approximately_inv_c_times_c_minus_1() -> None:
    """Uniform probability predictions yield macro per-class Brier ≈ (C-1)/C².

    Our compute_brier_score computes macro-averaged per-class squared error:
      E[(p_i - y_i)^2] averaged over classes.
    For class i with uniform p=1/C:
      - For the correct class: (1/C - 1)^2 = ((C-1)/C)^2
      - For incorrect classes: (1/C - 0)^2 = (1/C)^2
    Per-class average = (1/N) * sum over samples of [(1/C - indicator)^2]
    Macro average ≈ (C-1)/C² when classes are balanced.
    """
    n = 6000
    n_classes = 6
    rng = np.random.default_rng(3)
    # Balanced dataset
    y_true = np.repeat(np.arange(n_classes), n // n_classes).astype(np.int64)
    y_prob = np.full((n, n_classes), 1.0 / n_classes, dtype=np.float64)
    brier = compute_brier_score(y_true, y_prob)
    # For balanced uniform: per-class Brier = mean over N of (1/C - indicator)^2
    # = (1 - 1/C)*(1/C)^2 + (1/C)*(1 - 1/C)^2  (by complement)
    # = (1/C)^2 * (1-1/C) + (1-1/C)^2 * (1/C)
    inv_c = 1.0 / n_classes
    expected = inv_c * (inv_c ** 2) * (n_classes - 1) + inv_c * ((1 - inv_c) ** 2)
    assert brier == pytest.approx(expected, abs=0.02), (
        f"Uniform probs → macro per-class Brier ≈ {expected:.4f}, got {brier:.4f}"
    )


def test_brier_score_non_negative() -> None:
    """Brier score is always non-negative."""
    rng = np.random.default_rng(4)
    y_true = rng.integers(0, 3, size=200)
    y_prob = rng.dirichlet(np.ones(3), size=200)
    brier = compute_brier_score(y_true, y_prob)
    assert brier >= 0.0, f"Brier score must be ≥0, got {brier}"


# ---------------------------------------------------------------------------
# Balanced accuracy
# ---------------------------------------------------------------------------


def test_balanced_accuracy_perfect_is_one() -> None:
    """Balanced accuracy is 1.0 when all predictions are correct."""
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    ba = compute_balanced_accuracy(y_true, y_pred)
    assert ba == pytest.approx(1.0), f"Perfect predictions → balanced_accuracy=1, got {ba}"


def test_balanced_accuracy_random_is_near_inv_c() -> None:
    """Balanced accuracy of a uniform random classifier ≈ 1/C."""
    rng = np.random.default_rng(5)
    n_classes = 6
    n = 3000
    y_true = rng.integers(0, n_classes, size=n)
    y_pred = rng.integers(0, n_classes, size=n)
    ba = compute_balanced_accuracy(y_true, y_pred)
    assert abs(ba - (1.0 / n_classes)) < 0.05, (
        f"Random classifier balanced_accuracy ≈ 1/{n_classes}, got {ba:.3f}"
    )


def test_balanced_accuracy_penalises_majority_bias() -> None:
    """Balanced accuracy is lower than accuracy when predicting only majority class."""
    # 90 class-0, 10 class-1 — predict all class-0
    y_true = np.array([0] * 90 + [1] * 10, dtype=np.int64)
    y_pred = np.zeros(100, dtype=np.int64)
    acc_naive = float((y_pred == y_true).mean())   # 0.90
    ba = compute_balanced_accuracy(y_true, y_pred)
    # Recall for class-1 is 0, so balanced acc = (1 + 0)/2 = 0.5
    assert ba < acc_naive, "Balanced accuracy should penalise majority-class bias"
    assert ba == pytest.approx(0.5, abs=0.01), f"Expected balanced_acc≈0.5, got {ba:.3f}"


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------


def test_auroc_perfect_classifier_is_one() -> None:
    """Perfect discriminating classifier scores macro AUC = 1.0."""
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_prob = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    result = compute_auc_roc(y_true, y_prob)
    assert result["macro"] == pytest.approx(1.0), (
        f"Perfect classifier AUC should be 1.0, got {result['macro']}"
    )


def test_auroc_random_classifier_near_half() -> None:
    """Random probability predictions produce macro AUC ≈ 0.5."""
    rng = np.random.default_rng(6)
    n = 600
    n_classes = 6
    y_true = rng.integers(0, n_classes, size=n)
    y_prob = rng.dirichlet(np.ones(n_classes), size=n)
    result = compute_auc_roc(y_true, y_prob)
    assert abs(result["macro"] - 0.5) < 0.1, (
        f"Random classifier AUC should be near 0.5, got {result['macro']:.3f}"
    )


# ---------------------------------------------------------------------------
# Per-class metrics consistency
# ---------------------------------------------------------------------------


def test_per_class_metrics_macro_f1_consistent() -> None:
    """Macro F1 equals the mean of per-class F1 scores."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 1, 2, 0, 2], dtype=np.int64)
    result = compute_per_class_metrics(y_true, y_pred)
    per_class_f1s = [result["per_class"][f"class_{i}"]["f1"] for i in range(3)]
    expected_macro = float(np.mean(per_class_f1s))
    assert result["macro_f1"] == pytest.approx(expected_macro, abs=1e-6), (
        f"macro_f1={result['macro_f1']} should equal mean per-class F1={expected_macro}"
    )


def test_per_class_metrics_accuracy_consistent() -> None:
    """Accuracy matches fraction of correct predictions."""
    y_true = np.array([0, 0, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 2], dtype=np.int64)
    result = compute_per_class_metrics(y_true, y_pred)
    expected_acc = 3.0 / 4.0
    assert result["accuracy"] == pytest.approx(expected_acc, abs=1e-6), (
        f"Accuracy should be {expected_acc}, got {result['accuracy']}"
    )


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def test_confusion_matrix_diagonal_on_perfect_predictions() -> None:
    """Perfect predictions produce a diagonal confusion matrix."""
    y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    cm = compute_confusion_matrix(y_true, y_pred, n_classes=3)
    for i in range(3):
        for j in range(3):
            expected = 2 if i == j else 0
            assert cm[i][j] == expected, (
                f"cm[{i}][{j}] should be {expected}, got {cm[i][j]}"
            )


def test_confusion_matrix_sum_equals_n() -> None:
    """Total confusion matrix sum equals number of samples."""
    rng = np.random.default_rng(7)
    n = 100
    y_true = rng.integers(0, 4, size=n)
    y_pred = rng.integers(0, 4, size=n)
    cm = compute_confusion_matrix(y_true, y_pred, n_classes=4)
    total = sum(cm[i][j] for i in range(4) for j in range(4))
    assert total == n, f"Confusion matrix sum={total} should equal n={n}"
