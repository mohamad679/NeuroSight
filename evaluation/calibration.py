"""Calibration analysis utilities for probabilistic classification outputs.

IMPORTANT: Calibration analysis in this module operates on synthetic benchmark
data only.  Results do NOT estimate calibration quality on real patient
populations.  Temperature scaling is a post-hoc calibration method that may
improve ECE on held-out synthetic data but is not validated for clinical use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


class CalibrationAnalyzer:
    """Analyze calibration behavior for multiclass probabilistic predictions.

    All analysis is valid for any multiclass classifier.  Interpretation in
    a medical context requires expert review.  This class does not make
    clinical validity claims.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
        class_names: Optional[list[str]] = None,
    ) -> None:
        """Initialize calibration analyzer with predictions and labels.

        Args:
            y_true: Ground-truth class labels of shape ``(N,)``.
            y_prob: Predicted class probabilities of shape ``(N, C)``.
            n_bins: Number of reliability bins.
            class_names: Optional names for classes in plotting/reporting.
        """
        if y_true.ndim != 1:
            raise ValueError("y_true must be a 1D array of class labels.")
        if y_prob.ndim != 2:
            raise ValueError("y_prob must be a 2D array of class probabilities.")
        if y_true.shape[0] != y_prob.shape[0]:
            raise ValueError("y_true and y_prob must have the same number of rows.")
        if n_bins < 2:
            raise ValueError("n_bins must be at least 2.")

        self.y_true = y_true.astype(np.int64)
        self.y_prob = y_prob.astype(np.float64)
        self.n_bins = int(n_bins)
        self.n_classes = int(y_prob.shape[1])

        if class_names is not None and len(class_names) != self.n_classes:
            raise ValueError("class_names length must match number of classes.")
        self.class_names = (
            class_names
            if class_names is not None
            else [f"class_{index}" for index in range(self.n_classes)]
        )

    def reliability_diagram(self, save_path: str) -> None:
        """Save multiclass reliability diagram image.

        Args:
            save_path: Output path for the generated figure.
        """
        import matplotlib.pyplot as plt

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.plot(
            [0.0, 1.0], [0.0, 1.0],
            linestyle="--", linewidth=1.5, color="black", label="Perfect"
        )

        for class_index, class_name in enumerate(self.class_names):
            probs = self.y_prob[:, class_index]
            targets = (self.y_true == class_index).astype(np.float64)
            confidence_points: list[float] = []
            accuracy_points: list[float] = []

            for low, high in zip(bin_edges[:-1], bin_edges[1:]):
                if high == 1.0:
                    mask = (probs >= low) & (probs <= high)
                else:
                    mask = (probs >= low) & (probs < high)
                if not np.any(mask):
                    continue
                confidence_points.append(float(np.mean(probs[mask])))
                accuracy_points.append(float(np.mean(targets[mask])))

            if confidence_points:
                axis.plot(confidence_points, accuracy_points, marker="o", label=class_name)

        axis.set_title("Reliability Diagram (SYNTHETIC DATA — not clinical)")
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Empirical accuracy")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.3)
        axis.legend(loc="best")

        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(output_path, dpi=200)
        plt.close(figure)

    def compute_ece_per_class(self) -> dict[str, float]:
        """Compute expected calibration error for each class.

        Returns:
            Mapping from class name to ECE score.
        """
        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece_by_class: dict[str, float] = {}

        for class_index, class_name in enumerate(self.class_names):
            probs = self.y_prob[:, class_index]
            targets = (self.y_true == class_index).astype(np.float64)
            class_ece = 0.0

            for low, high in zip(bin_edges[:-1], bin_edges[1:]):
                if high == 1.0:
                    mask = (probs >= low) & (probs <= high)
                else:
                    mask = (probs >= low) & (probs < high)
                if not np.any(mask):
                    continue

                confidence = float(np.mean(probs[mask]))
                accuracy = float(np.mean(targets[mask]))
                weight = float(np.mean(mask))
                class_ece += weight * abs(accuracy - confidence)

            ece_by_class[class_name] = float(class_ece)

        return ece_by_class

    def compute_brier_score(self) -> float:
        """Compute macro-averaged multiclass Brier score.

        Returns:
            Macro-averaged Brier score (lower = better calibrated).
        """
        one_hot = np.zeros_like(self.y_prob, dtype=np.float64)
        one_hot[np.arange(self.y_true.shape[0]), self.y_true] = 1.0
        per_class_brier = np.mean((self.y_prob - one_hot) ** 2, axis=0)
        return float(np.mean(per_class_brier))

    def compute_ece_before_after_scaling(
        self, temperature: float
    ) -> dict[str, float]:
        """Compute ECE before and after temperature scaling.

        Temperature scaling divides logit-equivalent probabilities by a scalar
        temperature T before softmax.  This is approximated here by applying
        the sharpening/flattening effect directly to the probability simplex
        via a re-normalised power transform (valid for small T adjustments).

        Args:
            temperature: Temperature scalar > 0.  T < 1 sharpens, T > 1 flattens.

        Returns:
            Dict with ``"ece_before"``, ``"ece_after"``, and ``"temperature"``.
        """
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")

        from evaluation.metrics import compute_ece

        ece_before = compute_ece(self.y_true, self.y_prob.astype(np.float32))

        # Power-law temperature approximation on probability simplex
        scaled = np.power(np.clip(self.y_prob, 1e-10, 1.0), 1.0 / temperature)
        scaled_sum = scaled.sum(axis=1, keepdims=True)
        scaled_prob = (scaled / scaled_sum).astype(np.float32)
        ece_after = compute_ece(self.y_true, scaled_prob)

        return {
            "ece_before": float(ece_before),
            "ece_after": float(ece_after),
            "temperature": float(temperature),
        }

    def summary_report(self) -> dict[str, Any]:
        """Return summary calibration report with provenance metadata.

        Returns:
            Dictionary containing class-wise and aggregate calibration metrics,
            plus ``synthetic_data: true`` and ``clinical_validity: false``.
        """
        ece_per_class = self.compute_ece_per_class()
        macro_ece = float(np.mean(list(ece_per_class.values())))
        brier_score = self.compute_brier_score()
        return {
            "synthetic_data": True,
            "clinical_validity": False,
            "warning": (
                "Calibration metrics are from synthetic benchmark data only. "
                "They do not estimate calibration quality on real patient populations."
            ),
            "n_samples": int(self.y_true.shape[0]),
            "n_classes": int(self.n_classes),
            "ece_per_class": ece_per_class,
            "macro_ece": macro_ece,
            "brier_score": brier_score,
        }
