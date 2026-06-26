"""Evaluation package exports for NeuroSight.

All evaluation utilities operate on synthetic data unless real cohort data
is explicitly provided and documented.  No output of this package makes
clinical validity claims.
"""

from evaluation.ablation import run_ablation_benchmark
from evaluation.benchmark import run_benchmark
from evaluation.benchmark_table import render_comparison_table, render_report_markdown
from evaluation.calibration import CalibrationAnalyzer
from evaluation.cross_validation import run_kfold_cv
from evaluation.metrics import (
    compute_auc_roc,
    compute_balanced_accuracy,
    compute_brier_score,
    compute_confusion_matrix,
    compute_ece,
    compute_per_class_metrics,
)
from evaluation.report import generate_full_report, save_json_report, save_markdown_report
from evaluation.robustness import (
    evaluate_class_imbalance,
    evaluate_missing_fields,
    evaluate_noisy_inputs,
    evaluate_out_of_range,
    evaluate_site_shift,
)

__all__ = [
    # Benchmark
    "run_benchmark",
    # Ablation
    "run_ablation_benchmark",
    # Table & report rendering
    "render_comparison_table",
    "render_report_markdown",
    # Calibration
    "CalibrationAnalyzer",
    # Cross-validation
    "run_kfold_cv",
    # Metrics
    "compute_auc_roc",
    "compute_balanced_accuracy",
    "compute_brier_score",
    "compute_confusion_matrix",
    "compute_ece",
    "compute_per_class_metrics",
    # Report generation
    "generate_full_report",
    "save_json_report",
    "save_markdown_report",
    # Robustness
    "evaluate_class_imbalance",
    "evaluate_missing_fields",
    "evaluate_noisy_inputs",
    "evaluate_out_of_range",
    "evaluate_site_shift",
]
