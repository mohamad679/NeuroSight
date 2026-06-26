"""Evaluation metrics for NeuroSight multiclass classification.

All functions operate on NumPy arrays. No clinical claims are made or implied.
These metrics are correct for any multiclass classification problem; their
interpretation in a medical context requires expert review.

Functions
---------
compute_auc_roc          — per-class and macro AUROC
compute_ece              — expected calibration error (max-confidence variant)
compute_per_class_metrics — accuracy, macro P/R/F1, per-class breakdown
compute_balanced_accuracy — macro-averaged recall (balanced accuracy)
compute_brier_score      — macro-averaged multiclass Brier score
compute_confusion_matrix  — integer confusion matrix as nested list
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import numpy as np


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def compute_auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute per-class and macro one-vs-rest AUROC.

    Args:
        y_true: Ground-truth integer class labels of shape ``(N,)``.
        y_prob: Predicted class probabilities of shape ``(N, C)``.

    Returns:
        Dictionary with ``class_<i>`` keys and a ``"macro"`` aggregate.
        Classes with only one distinct label receive ``float("nan")``.
    """
    from sklearn.metrics import roc_auc_score

    n_classes = y_prob.shape[1]
    result: dict[str, float] = {}
    for i in range(n_classes):
        binary = (y_true == i).astype(int)
        if binary.sum() > 0 and binary.sum() < len(binary):
            try:
                result[f"class_{i}"] = float(roc_auc_score(binary, y_prob[:, i]))
            except ValueError:
                result[f"class_{i}"] = 0.5
        else:
            result[f"class_{i}"] = float("nan")
    valid = [v for v in result.values() if not np.isnan(v)]
    result["macro"] = float(np.mean(valid)) if valid else float("nan")
    return result


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Compute expected calibration error using max-confidence binning.

    Args:
        y_true: Ground-truth integer class labels of shape ``(N,)``.
        y_prob: Predicted class probabilities of shape ``(N, C)``.
        n_bins: Number of equal-width confidence bins.

    Returns:
        ECE scalar in ``[0, 1]``.
    """
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for low, high in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > low) & (confidences <= high)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.mean() * abs(bin_acc - bin_conf)
    return float(ece)


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute accuracy, macro P/R/F1, and per-class breakdown.

    Args:
        y_true: Ground-truth integer class labels of shape ``(N,)``.
        y_pred: Predicted integer class labels of shape ``(N,)``.

    Returns:
        Dictionary with ``"accuracy"``, ``"macro_f1"``, ``"macro_precision"``,
        ``"macro_recall"``, and ``"per_class"`` mapping from class key to
        ``{precision, recall, f1, support}``.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            f"class_{i}": {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(p))
        },
        "macro_f1": float(f1.mean()),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
    }


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy (macro-averaged per-class recall).

    This metric is robust to class imbalance. On balanced datasets it equals
    ordinary accuracy; on imbalanced datasets it penalises majority-class bias.

    Args:
        y_true: Ground-truth integer class labels of shape ``(N,)``.
        y_pred: Predicted integer class labels of shape ``(N,)``.

    Returns:
        Balanced accuracy in ``[0, 1]``.
    """
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y_true, y_pred))


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute macro-averaged multiclass Brier score.

    Lower is better. A random uniform classifier on C classes scores
    approximately ``1 - 1/C``.

    Args:
        y_true: Ground-truth integer class labels of shape ``(N,)``.
        y_prob: Predicted class probabilities of shape ``(N, C)``.

    Returns:
        Macro-averaged Brier score in ``[0, 2]`` (theoretically bounded).
    """
    n_classes = y_prob.shape[1]
    one_hot = np.zeros((y_true.shape[0], n_classes), dtype=np.float64)
    one_hot[np.arange(y_true.shape[0]), y_true.astype(int)] = 1.0
    per_class_brier = np.mean((y_prob.astype(np.float64) - one_hot) ** 2, axis=0)
    return float(np.mean(per_class_brier))


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> list[list[int]]:
    """Compute an integer confusion matrix without external dependencies.

    Args:
        y_true: Ground-truth integer class labels of shape ``(N,)``.
        y_pred: Predicted integer class labels of shape ``(N,)``.
        n_classes: Total number of classes.

    Returns:
        Nested list of shape ``(n_classes, n_classes)`` where
        ``matrix[true][pred]`` is the count.
    """
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, prediction in zip(y_true.tolist(), y_pred.tolist()):
        matrix[int(truth), int(prediction)] += 1
    return cast(list[list[int]], matrix.tolist())


# ---------------------------------------------------------------------------
# Ablation helper
# ---------------------------------------------------------------------------


def compute_modality_ablation(model: Any, test_data: list[dict[str, Any]]) -> dict[str, float]:
    """Compute macro AUROC under different modality-availability configurations.

    Args:
        model: Fusion model callable accepting ``mri``, ``eeg``, ``cog``
            keyword tensors (any may be ``None``).
        test_data: List of sample dicts with keys ``mri``, ``eeg``, ``cog``,
            ``label``.

    Returns:
        Dictionary with AUROC per config and ``_delta`` entries relative to
        the all-modality baseline.
    """
    import torch

    model.eval()
    configs: dict[str, dict[str, bool]] = {
        "all": {"use_mri": True, "use_eeg": True, "use_cog": True},
        "no_mri": {"use_mri": False, "use_eeg": True, "use_cog": True},
        "no_eeg": {"use_mri": True, "use_eeg": False, "use_cog": True},
        "no_cognitive": {"use_mri": True, "use_eeg": True, "use_cog": False},
    }
    baseline_auc: float | None = None
    results: dict[str, float] = {}

    with torch.no_grad():
        for config_name, cfg in configs.items():
            all_probs: list[np.ndarray] = []
            all_labels: list[int] = []
            skipped = False
            for sample in test_data:
                mri = sample["mri"] if cfg["use_mri"] else None
                eeg = sample["eeg"] if cfg["use_eeg"] else None
                cog = sample["cog"] if cfg["use_cog"] else None
                if mri is None and eeg is None and cog is None:
                    skipped = True
                    break
                out = model(mri=mri, eeg=eeg, cog=cog)
                all_probs.append(out["probs"].numpy())
                all_labels.append(sample["label"])
            if skipped or not all_probs:
                results[config_name] = float("nan")
                continue
            y_prob = np.vstack(all_probs)
            y_true = np.array(all_labels, dtype=np.int64)
            auc = compute_auc_roc(y_true, y_prob)["macro"]
            results[config_name] = auc
            if config_name == "all":
                baseline_auc = auc

    if baseline_auc is not None and not np.isnan(baseline_auc):
        for k in ("no_mri", "no_eeg", "no_cognitive"):
            if k in results and not np.isnan(results[k]):
                results[f"{k}_delta"] = round(results[k] - baseline_auc, 4)
            else:
                results[f"{k}_delta"] = float("nan")
    return results


# ---------------------------------------------------------------------------
# MLflow / fallback reporting
# ---------------------------------------------------------------------------


@dataclass
class EvaluationReport:
    """Structured container for evaluation results and provenance."""

    metrics: dict[str, Any]
    timestamp: str
    model_checkpoint: str


def log_to_mlflow(report: EvaluationReport, experiment_name: str) -> None:
    """Log metrics to MLflow, falling back to JSONL if unavailable.

    Args:
        report: Evaluation report container.
        experiment_name: MLflow experiment name.
    """
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_metrics(
                {k: v for k, v in report.metrics.items() if isinstance(v, float)}
            )
            mlflow.log_param("model_checkpoint", report.model_checkpoint)
            mlflow.log_param("timestamp", report.timestamp)
    except ImportError:
        import pathlib

        log_path = pathlib.Path("logs/mlflow_fallback.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(
                json.dumps({"experiment": experiment_name, **report.__dict__}) + "\n"
            )
