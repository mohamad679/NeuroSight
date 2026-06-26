"""Rendering helpers for benchmark comparison outputs.

All rendered tables and reports carry a mandatory warning banner reminding
readers that results are from synthetic data and carry no clinical validity.
"""

from __future__ import annotations

from typing import Any


_REPORT_WARNING = (
    "> ⚠️  **SYNTHETIC BENCHMARK — NOT CLINICAL PERFORMANCE**\n"
    ">\n"
    "> All results in this report are computed on synthetically generated data.\n"
    "> They do NOT represent real-world accuracy, clinical reliability, or\n"
    "> validated medical performance on any patient population.\n"
    "> `synthetic_data: true` | `clinical_validity: false`"
)


def _format_metric(value: Any, digits: int = 3) -> str:
    """Format numeric metric values for markdown table output."""
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    import math
    if math.isnan(numeric):
        return "NaN"
    return f"{numeric:.{digits}f}"


def render_comparison_table(results: dict[str, Any]) -> str:
    """Render benchmark results as a Markdown table with warning banner.

    Args:
        results: Benchmark results dictionary returned by ``run_benchmark``.

    Returns:
        Markdown string including warning banner and comparison table.
    """
    rows = results.get("results", []) if isinstance(results, dict) else []
    if not isinstance(rows, list):
        raise ValueError("results['results'] must be a list.")

    header = (
        "| Method | Modality | AUC ↑ | F1 ↑ | Acc ↑ | Bal.Acc ↑ | Brier ↓ | ECE ↓ | Train Time |\n"
        "|--------|----------|--------|------|-------|-----------|---------|-------|------------|"
    )
    rendered_rows: list[str] = [_REPORT_WARNING, "", header]

    for row in rows:
        if not isinstance(row, dict):
            continue
        method = str(row.get("method", "unknown"))
        modality = str(row.get("modality", "-"))
        auc = _format_metric(row.get("macro_auc"))
        f1 = _format_metric(row.get("macro_f1"))
        acc = _format_metric(row.get("accuracy"))
        bal_acc = _format_metric(row.get("balanced_accuracy"))
        brier = _format_metric(row.get("brier_score"))
        ece = _format_metric(row.get("ece"))

        train_time = row.get("train_time_seconds")
        try:
            if train_time is None:
                raise TypeError
            train_time_str = f"{float(train_time):.1f}s"
        except (TypeError, ValueError):
            train_time_str = "—"

        if method == "neurosight_fusion":
            method_cell = "**NeuroSight Fusion**"
            modality_cell = f"**{modality}**"
        elif method == "random_classifier":
            method_cell = "Random Classifier (chance)"
            modality_cell = "—"
        elif method == "majority_classifier":
            method_cell = "Majority Classifier"
            modality_cell = "—"
        elif method == "logistic_regression":
            method_cell = "Logistic Regression"
            modality_cell = modality
        elif method == "random_forest":
            method_cell = "Random Forest"
            modality_cell = modality
        elif method == "gradient_boosting":
            method_cell = "Gradient Boosting"
            modality_cell = modality
        elif method == "mlp_cognitive_only":
            method_cell = "MLP (Cognitive only)"
            modality_cell = modality
        elif method == "neurosight_cognitive_only":
            method_cell = "NeuroSight (Cognitive, calibrated)"
            modality_cell = modality
        else:
            method_cell = method.replace("_", " ").title()
            modality_cell = modality

        rendered_rows.append(
            f"| {method_cell} | {modality_cell} | {auc} | {f1} | {acc} | {bal_acc} | {brier} | {ece} | {train_time_str} |"
        )

    seed = results.get("seed", "unknown")
    rendered_rows.append("")
    rendered_rows.append(
        f"*Seed: {seed} | Dataset: synthetic ADNI-like tabular data | Clinical validity: false*"
    )
    return "\n".join(rendered_rows)


def render_report_markdown(results: dict[str, Any]) -> str:
    """Generate a full Markdown evaluation report with methodology and limitations.

    Args:
        results: Benchmark results dictionary returned by ``run_benchmark``.

    Returns:
        Full Markdown report string including warning, methodology, table,
        and limitations sections.
    """
    seed = results.get("seed", "unknown")
    dataset_path = results.get("dataset_path", "unknown")
    methodology = results.get("methodology", "Not documented.")
    dep_versions = results.get("dependency_versions", {})
    config = results.get("config", {})
    leakage_ok = results.get("leakage_check_passed", "unknown")
    winner = results.get("winner", {})
    command_used = results.get("command_used") or "PYTHONPATH=. python3 scripts/run_benchmark.py --mode full"
    model_status = results.get("model_status") or "Demonstration and portfolio research scaffold (no clinical validation, not for diagnostic use)"

    dep_str = "\n".join(
        f"- **{pkg}**: {ver}" for pkg, ver in dep_versions.items()
    )
    config_str = "\n".join(
        f"- **{k}**: {v}" for k, v in config.items()
    )

    comparison_table = render_comparison_table(results)
    winner_method = winner.get("method", "unknown") if isinstance(winner, dict) else "unknown"
    winner_auc = _format_metric(winner.get("macro_auc") if isinstance(winner, dict) else None)

    return f"""# NeuroSight Synthetic Benchmark Report

{_REPORT_WARNING}

---

## Overview

This report documents the results of the NeuroSight synthetic benchmark evaluation.
**These are engineering validation results, not clinical performance estimates.**

| Field | Value |
|-------|-------|
| Seed | `{seed}` |
| Dataset | `{dataset_path}` |
| Leakage check passed | `{leakage_ok}` |
| `synthetic_data` | `true` |
| `clinical_validity` | `false` |
| Best method (by AUC) | `{winner_method}` (AUC: {winner_auc}) |
| Model Status | `{model_status}` |
| Command Used | `{command_used}` |

---

## Methodology

{methodology}

### Run Configuration

{config_str}

### Dependency Versions

{dep_str}

---

## Reproducibility Details

To reproduce these benchmark results, run the exact command:
```bash
{command_used}
```
Ensure all dependency versions match the run configuration listed above and that the dataset at `{dataset_path}` has been generated with seed `{seed}`.

---

## Baseline Comparison

{comparison_table}

---

## Metrics Legend

| Metric | Description |
|--------|-------------|
| AUC ↑ | Macro one-vs-rest AUROC (higher = better) |
| F1 ↑ | Macro F1 score (higher = better) |
| Acc ↑ | Accuracy (higher = better) |
| Bal.Acc ↑ | Balanced accuracy — macro recall (better for imbalanced sets) |
| Brier ↓ | Macro Brier score — probabilistic calibration (lower = better) |
| ECE ↓ | Expected calibration error (lower = better) |

---

## Limitations

- All results are from synthetically generated data with known class profiles.
  Performance on real clinical cohorts may differ substantially.
- MRI and EEG modalities in this benchmark are independent Gaussian noise tensors.
  They do not simulate real imaging or electrophysiology signal.
- The fusion model's performance advantage over cognitive baselines is NOT expected
  on orthogonal noise inputs. A lower fusion AUC than cognitive-only is the
  honest result.
- Calibration metrics (ECE, Brier) on synthetic data do not estimate calibration
  quality on real patients.
- No subgroup analysis, fairness evaluation, or demographic breakdown is performed.
- No external validation has been conducted.
- **This benchmark is not for clinical use and does not represent clinical validation.**

## Future Work

- Real-cohort evaluation requires authorized ADNI/OASIS access (see
  `scripts/prepare_adni_like_dataset.py` for expected input format).
- External validation on independent clinical sites.
- Fairness and subgroup analysis on real demographic distributions.
- Scanner-shift robustness evaluation with real multi-site imaging data.
"""
