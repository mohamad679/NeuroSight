"""Robustness evaluation for NeuroSight cognitive classifiers.

Evaluates model performance under distribution shift conditions using only
synthetic data.  No results claim clinical robustness or real-world validity.

All functions return structured dictionaries with ``synthetic_data: true``
and ``clinical_validity: false`` in their outputs.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np


_N_CLASSES = 6
_FEATURE_NAMES = ["MMSE", "MOCA", "CDRSB", "ADAS11", "RAVLT_immediate", "RAVLT_learning", "FAQ", "AGE"]

# Feature valid ranges for out-of-range injection
_FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "MMSE": (0.0, 30.0),
    "MOCA": (0.0, 30.0),
    "CDRSB": (0.0, 18.0),
    "ADAS11": (0.0, 70.0),
    "RAVLT_immediate": (0.0, 75.0),
    "RAVLT_learning": (-15.0, 15.0),
    "FAQ": (0.0, 30.0),
    "AGE": (0.0, 120.0),
}


def _run_model_cv(
    x: np.ndarray,
    y: np.ndarray,
    model_builder: Callable[[], Any],
    seed: int,
    cv_folds: int = 3,
) -> dict[str, float]:
    """Run stratified K-fold CV and return mean AUC/F1."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    from evaluation.metrics import compute_auc_roc, compute_per_class_metrics

    _, class_counts = np.unique(y, return_counts=True)
    n_splits = max(2, min(cv_folds, int(class_counts.min())))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    auc_values: list[float] = []
    f1_values: list[float] = []

    for fold_i, (train_idx, val_idx) in enumerate(splitter.split(x, y), start=1):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_val = scaler.transform(x[val_idx])
        y_train = y[train_idx]
        y_val = y[val_idx]

        model = model_builder()
        model.fit(x_train, y_train)

        if hasattr(model, "predict_proba"):
            y_prob = np.asarray(model.predict_proba(x_val), dtype=np.float64)
            # Pad to n_classes if model only saw a subset
            if y_prob.shape[1] < _N_CLASSES:
                full = np.zeros((y_prob.shape[0], _N_CLASSES), dtype=np.float64)
                classes_seen = list(np.unique(y_train))
                for col_i, cls_i in enumerate(classes_seen):
                    if cls_i < _N_CLASSES:
                        full[:, cls_i] = y_prob[:, col_i]
                row_sums = full.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1.0, row_sums)
                y_prob = full / row_sums
        else:
            y_pred = np.asarray(model.predict(x_val), dtype=np.int64)
            y_prob = np.eye(_N_CLASSES, dtype=np.float64)[y_pred]

        try:
            auc = float(compute_auc_roc(y_val, y_prob).get("macro", float("nan")))
        except Exception:
            auc = float("nan")

        try:
            y_pred_cls = np.argmax(y_prob, axis=1)
            f1 = float(compute_per_class_metrics(y_val, y_pred_cls)["macro_f1"])
        except Exception:
            f1 = float("nan")

        auc_values.append(auc)
        f1_values.append(f1)

    valid_aucs = [v for v in auc_values if not np.isnan(v)]
    valid_f1s = [v for v in f1_values if not np.isnan(v)]
    return {
        "mean_auc": float(np.mean(valid_aucs)) if valid_aucs else float("nan"),
        "mean_f1": float(np.mean(valid_f1s)) if valid_f1s else float("nan"),
    }


def evaluate_noisy_inputs(
    x: np.ndarray,
    y: np.ndarray,
    noise_levels: list[float],
    model_builder: Callable[[], Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate classifier performance under additive Gaussian noise.

    Args:
        x: Feature matrix ``(N, 8)`` of normalised cognitive scores.
        y: Integer class labels ``(N,)``.
        noise_levels: List of noise standard deviations (e.g. [0.1, 0.5, 1.0]).
        model_builder: Callable returning a fresh sklearn-compatible estimator.
        seed: Reproducibility seed.

    Returns:
        Structured dict with per noise-level AUC/F1 and provenance metadata.
    """
    rng = np.random.default_rng(seed)
    per_level: dict[str, dict[str, float]] = {}

    baseline = _run_model_cv(x, y, model_builder, seed=seed)
    per_level["noise_0.0"] = baseline

    for std in noise_levels:
        noisy_x = x + rng.normal(0.0, std, size=x.shape).astype(np.float32)
        metrics = _run_model_cv(noisy_x, y, model_builder, seed=seed)
        per_level[f"noise_{std:.2f}"] = metrics

    return {
        "synthetic_data": True,
        "clinical_validity": False,
        "test_type": "noisy_inputs",
        "noise_levels": noise_levels,
        "results": per_level,
    }


def evaluate_missing_fields(
    x: np.ndarray,
    y: np.ndarray,
    missing_fractions: list[float],
    model_builder: Callable[[], Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate classifier under random feature masking (median imputation).

    Args:
        x: Feature matrix ``(N, 8)``.
        y: Integer class labels ``(N,)``.
        missing_fractions: Fractions of values to zero-out (e.g. [0.1, 0.3, 0.5]).
        model_builder: Callable returning a fresh estimator.
        seed: Reproducibility seed.

    Returns:
        Structured dict with per-fraction AUC/F1 and provenance metadata.
    """
    rng = np.random.default_rng(seed)
    per_fraction: dict[str, dict[str, float]] = {}

    baseline = _run_model_cv(x, y, model_builder, seed=seed)
    per_fraction["missing_0.0"] = baseline

    feature_medians = np.median(x, axis=0)

    for frac in missing_fractions:
        masked_x = x.copy()
        n_total = masked_x.shape[0] * masked_x.shape[1]
        n_missing = int(frac * n_total)
        flat_indices = rng.choice(n_total, size=n_missing, replace=False)
        row_idx = flat_indices // masked_x.shape[1]
        col_idx = flat_indices % masked_x.shape[1]
        masked_x[row_idx, col_idx] = feature_medians[col_idx]

        metrics = _run_model_cv(masked_x, y, model_builder, seed=seed)
        per_fraction[f"missing_{frac:.2f}"] = metrics

    return {
        "synthetic_data": True,
        "clinical_validity": False,
        "test_type": "missing_fields",
        "imputation": "median",
        "missing_fractions": missing_fractions,
        "results": per_fraction,
    }


def evaluate_out_of_range(
    x: np.ndarray,
    y: np.ndarray,
    model_builder: Callable[[], Any],
    seed: int = 42,
    fraction: float = 0.2,
) -> dict[str, Any]:
    """Evaluate classifier with a fraction of features set to extreme values.

    Injects values at 1.5× the maximum valid range (boundary violation) to
    verify that the pipeline does not crash and produces valid probabilities.

    Args:
        x: Feature matrix ``(N, 8)``.
        y: Integer class labels ``(N,)``.
        model_builder: Callable returning a fresh estimator.
        seed: Reproducibility seed.
        fraction: Fraction of samples to perturb with out-of-range values.

    Returns:
        Structured dict with AUC/F1 under clean and perturbed data.
    """
    rng = np.random.default_rng(seed)
    perturbed_x = x.copy()
    n_perturb = max(1, int(fraction * x.shape[0]))
    perturb_idx = rng.choice(x.shape[0], size=n_perturb, replace=False)

    for col_i, (_, (_, hi)) in enumerate(zip(_FEATURE_NAMES, _FEATURE_RANGES.values())):
        perturbed_x[perturb_idx, col_i] = hi * 1.5  # Intentional boundary violation

    baseline = _run_model_cv(x, y, model_builder, seed=seed)
    perturbed = _run_model_cv(perturbed_x, y, model_builder, seed=seed)

    return {
        "synthetic_data": True,
        "clinical_validity": False,
        "test_type": "out_of_range_values",
        "perturb_fraction": fraction,
        "results": {
            "clean": baseline,
            "out_of_range": perturbed,
        },
    }


def evaluate_class_imbalance(
    x: np.ndarray,
    y: np.ndarray,
    imbalance_ratios: list[float],
    model_builder: Callable[[], Any],
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate classifier under simulated class imbalance by subsampling.

    Reduces minority class representation to the given ratio relative to the
    majority class to test balanced accuracy sensitivity.

    Args:
        x: Feature matrix ``(N, 8)``.
        y: Integer class labels ``(N,)``.
        imbalance_ratios: Minority/majority ratios to test (e.g. [1.0, 0.5, 0.1]).
        model_builder: Callable returning a fresh estimator.
        seed: Reproducibility seed.

    Returns:
        Structured dict with AUC/F1 per imbalance ratio.
    """
    from evaluation.metrics import compute_balanced_accuracy

    rng = np.random.default_rng(seed)
    per_ratio: dict[str, dict[str, float]] = {}

    unique_classes, class_counts = np.unique(y, return_counts=True)
    majority_count = int(class_counts.max())

    for ratio in imbalance_ratios:
        indices_to_keep: list[int] = []
        target_minority = max(1, int(majority_count * ratio))

        for cls_label in unique_classes:
            cls_indices = np.where(y == cls_label)[0]
            if len(cls_indices) > target_minority:
                chosen = rng.choice(cls_indices, size=target_minority, replace=False)
            else:
                chosen = cls_indices
            indices_to_keep.extend(chosen.tolist())

        indices_arr = np.array(indices_to_keep, dtype=np.int64)
        x_imb = x[indices_arr]
        y_imb = y[indices_arr]

        if len(np.unique(y_imb)) < 2:
            per_ratio[f"ratio_{ratio:.2f}"] = {
                "mean_auc": float("nan"),
                "mean_f1": float("nan"),
                "n_samples": len(y_imb),
            }
            continue

        metrics = _run_model_cv(x_imb, y_imb, model_builder, seed=seed)
        metrics["n_samples"] = len(y_imb)
        per_ratio[f"ratio_{ratio:.2f}"] = metrics

    return {
        "synthetic_data": True,
        "clinical_validity": False,
        "test_type": "class_imbalance",
        "imbalance_ratios": imbalance_ratios,
        "results": per_ratio,
    }


def evaluate_site_shift(
    x: np.ndarray,
    y: np.ndarray,
    model_builder: Callable[[], Any],
    n_sites: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate model under simulated scanner/site distribution shift.

    Partitions data into ``n_sites`` groups.  Each site receives an additive
    bias and multiplicative scale perturbation simulating scanner differences.
    Train on all sites except one, evaluate on the held-out site.

    Args:
        x: Feature matrix ``(N, 8)``.
        y: Integer class labels ``(N,)``.
        model_builder: Callable returning a fresh estimator.
        n_sites: Number of simulated acquisition sites.
        seed: Reproducibility seed.

    Returns:
        Structured dict with per hold-out-site AUC/F1 and provenance metadata.
    """
    from sklearn.preprocessing import StandardScaler
    from evaluation.metrics import compute_auc_roc, compute_per_class_metrics

    rng = np.random.default_rng(seed)
    n_samples = x.shape[0]
    site_assignments = rng.integers(0, n_sites, size=n_samples)

    # Per-site random bias and scale
    site_biases = rng.normal(0.0, 0.3, size=(n_sites, x.shape[1])).astype(np.float32)
    site_scales = rng.uniform(0.8, 1.2, size=(n_sites, x.shape[1])).astype(np.float32)

    x_shifted = x.copy()
    for site_id in range(n_sites):
        mask = site_assignments == site_id
        x_shifted[mask] = x[mask] * site_scales[site_id] + site_biases[site_id]

    per_site: dict[str, dict[str, float]] = {}

    for hold_out in range(n_sites):
        train_mask = site_assignments != hold_out
        test_mask = site_assignments == hold_out
        x_train = x_shifted[train_mask]
        y_train = y[train_mask]
        x_test = x_shifted[test_mask]
        y_test = y[test_mask]

        if len(np.unique(y_train)) < 2 or len(x_test) == 0:
            per_site[f"hold_out_site_{hold_out}"] = {
                "mean_auc": float("nan"),
                "mean_f1": float("nan"),
            }
            continue

        scaler = StandardScaler()
        x_train_sc = scaler.fit_transform(x_train)
        x_test_sc = scaler.transform(x_test)

        model = model_builder()
        model.fit(x_train_sc, y_train)

        if hasattr(model, "predict_proba"):
            y_prob = np.asarray(model.predict_proba(x_test_sc), dtype=np.float64)
            if y_prob.shape[1] < _N_CLASSES:
                full = np.zeros((y_prob.shape[0], _N_CLASSES), dtype=np.float64)
                for col_i, cls_i in enumerate(np.unique(y_train)):
                    if cls_i < _N_CLASSES:
                        full[:, cls_i] = y_prob[:, col_i]
                row_sums = full.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums == 0, 1.0, row_sums)
                y_prob = full / row_sums
        else:
            y_pred = np.asarray(model.predict(x_test_sc), dtype=np.int64)
            y_prob = np.eye(_N_CLASSES, dtype=np.float64)[y_pred]

        try:
            auc = float(compute_auc_roc(y_test, y_prob).get("macro", float("nan")))
        except Exception:
            auc = float("nan")

        try:
            y_pred_cls = np.argmax(y_prob, axis=1)
            f1 = float(compute_per_class_metrics(y_test, y_pred_cls)["macro_f1"])
        except Exception:
            f1 = float("nan")

        per_site[f"hold_out_site_{hold_out}"] = {
            "mean_auc": auc,
            "mean_f1": f1,
            "n_test_samples": int(test_mask.sum()),
        }

    return {
        "synthetic_data": True,
        "clinical_validity": False,
        "test_type": "site_shift",
        "n_sites": n_sites,
        "site_shift_method": "random additive bias + multiplicative scale per feature",
        "results": per_site,
    }
