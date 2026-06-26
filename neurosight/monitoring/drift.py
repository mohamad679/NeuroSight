"""Data drift monitoring helpers for NeuroSight.

The monitor compares a baseline cohort against a current cohort using
interpretable, dependency-light statistics. It is designed for research/demo
governance rather than clinical release decisions.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

COGNITIVE_FEATURES: tuple[str, ...] = (
    "MMSE",
    "MOCA",
    "CDRSB",
    "ADAS11",
    "RAVLT_immediate",
    "RAVLT_learning",
    "FAQ",
    "AGE",
)


@dataclass(frozen=True)
class DriftThresholds:
    """Thresholds used to classify drift severity."""

    psi_warning: float = 0.10
    psi_drift: float = 0.25
    ks_warning: float = 0.15
    ks_drift: float = 0.30
    mean_z_warning: float = 0.50
    mean_z_drift: float = 1.00
    missing_rate_warning: float = 0.05
    missing_rate_drift: float = 0.15


def utc_now() -> str:
    """Return current UTC time as an ISO-8601 instant."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_float_array(values: list[Any]) -> np.ndarray:
    parsed: list[float] = []
    for value in values:
        if value is None or value == "":
            parsed.append(np.nan)
            continue
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            parsed.append(np.nan)
    return np.asarray(parsed, dtype=np.float64)


def _finite_values(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    return finite.astype(np.float64)


def _missing_rate(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0
    return float(np.mean(~np.isfinite(values)))


def _safe_mean(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _safe_std(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    if finite.size < 2:
        return None
    return float(np.std(finite, ddof=1))


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = _finite_values(values)
    if finite.size == 0:
        return np.asarray([], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, max(2, int(n_bins) + 1))
    edges = np.quantile(finite, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        center = float(edges[0]) if edges.size else 0.0
        edges = np.asarray([center - 0.5, center + 0.5], dtype=np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(
    baseline: np.ndarray,
    current: np.ndarray,
    *,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Calculate Population Stability Index using baseline quantile bins."""
    baseline_finite = _finite_values(baseline)
    current_finite = _finite_values(current)
    if baseline_finite.size == 0 or current_finite.size == 0:
        return 0.0

    edges = _quantile_bins(baseline_finite, n_bins)
    baseline_counts, _ = np.histogram(baseline_finite, bins=edges)
    current_counts, _ = np.histogram(current_finite, bins=edges)

    baseline_pct = baseline_counts / max(1, baseline_counts.sum())
    current_pct = current_counts / max(1, current_counts.sum())
    baseline_pct = np.clip(baseline_pct, epsilon, None)
    current_pct = np.clip(current_pct, epsilon, None)
    return float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))


def ks_statistic(baseline: np.ndarray, current: np.ndarray) -> float:
    """Calculate a two-sample Kolmogorov-Smirnov distance without scipy."""
    baseline_finite = np.sort(_finite_values(baseline))
    current_finite = np.sort(_finite_values(current))
    if baseline_finite.size == 0 or current_finite.size == 0:
        return 0.0

    combined = np.sort(np.unique(np.concatenate([baseline_finite, current_finite])))
    baseline_cdf = np.searchsorted(baseline_finite, combined, side="right") / baseline_finite.size
    current_cdf = np.searchsorted(current_finite, combined, side="right") / current_finite.size
    return float(np.max(np.abs(baseline_cdf - current_cdf)))


def classify_feature(
    *,
    psi: float,
    ks: float,
    mean_z_shift: float,
    missing_rate_delta: float,
    thresholds: DriftThresholds,
) -> str:
    """Classify one feature as ok, warning, or drift."""
    if (
        psi >= thresholds.psi_drift
        or ks >= thresholds.ks_drift
        or abs(mean_z_shift) >= thresholds.mean_z_drift
        or abs(missing_rate_delta) >= thresholds.missing_rate_drift
    ):
        return "drift"
    if (
        psi >= thresholds.psi_warning
        or ks >= thresholds.ks_warning
        or abs(mean_z_shift) >= thresholds.mean_z_warning
        or abs(missing_rate_delta) >= thresholds.missing_rate_warning
    ):
        return "warning"
    return "ok"


def feature_drift_report(
    feature: str,
    baseline: np.ndarray,
    current: np.ndarray,
    thresholds: DriftThresholds,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Build a drift report for one numeric feature."""
    baseline_mean = _safe_mean(baseline)
    current_mean = _safe_mean(current)
    baseline_std = _safe_std(baseline)
    current_std = _safe_std(current)
    missing_delta = _missing_rate(current) - _missing_rate(baseline)

    if baseline_mean is None or current_mean is None:
        mean_delta = None
        mean_z_shift = 0.0
    else:
        mean_delta = current_mean - baseline_mean
        denominator = baseline_std if baseline_std and baseline_std > 1e-9 else 1.0
        mean_z_shift = mean_delta / denominator

    psi = population_stability_index(baseline, current, n_bins=n_bins)
    ks = ks_statistic(baseline, current)
    severity = classify_feature(
        psi=psi,
        ks=ks,
        mean_z_shift=mean_z_shift,
        missing_rate_delta=missing_delta,
        thresholds=thresholds,
    )
    return {
        "feature": feature,
        "severity": severity,
        "metrics": {
            "psi": round(float(psi), 6),
            "ks_statistic": round(float(ks), 6),
            "mean_delta": round(float(mean_delta), 6) if mean_delta is not None else None,
            "mean_z_shift": round(float(mean_z_shift), 6),
            "missing_rate_delta": round(float(missing_delta), 6),
        },
        "baseline": {
            "count": int(_finite_values(baseline).size),
            "missing_rate": round(_missing_rate(baseline), 6),
            "mean": round(float(baseline_mean), 6) if baseline_mean is not None else None,
            "std": round(float(baseline_std), 6) if baseline_std is not None else None,
        },
        "current": {
            "count": int(_finite_values(current).size),
            "missing_rate": round(_missing_rate(current), 6),
            "mean": round(float(current_mean), 6) if current_mean is not None else None,
            "std": round(float(current_std), 6) if current_std is not None else None,
        },
    }


def overall_status(feature_reports: list[dict[str, Any]]) -> str:
    """Summarize feature severities into one report status."""
    severities = {str(report.get("severity")) for report in feature_reports}
    if "drift" in severities:
        return "drift"
    if "warning" in severities:
        return "warning"
    return "ok"


def build_drift_report(
    baseline_records: list[dict[str, Any]],
    current_records: list[dict[str, Any]],
    *,
    features: tuple[str, ...] = COGNITIVE_FEATURES,
    thresholds: DriftThresholds | None = None,
    n_bins: int = 10,
    source: str = "synthetic_demo",
) -> dict[str, Any]:
    """Build a JSON-safe drift report for baseline and current cohorts."""
    applied_thresholds = thresholds or DriftThresholds()
    feature_reports: list[dict[str, Any]] = []
    for feature in features:
        baseline = _as_float_array([row.get(feature) for row in baseline_records])
        current = _as_float_array([row.get(feature) for row in current_records])
        feature_reports.append(
            feature_drift_report(
                feature,
                baseline,
                current,
                applied_thresholds,
                n_bins=n_bins,
            )
        )

    status = overall_status(feature_reports)
    drifted_features = [
        report["feature"]
        for report in feature_reports
        if report.get("severity") == "drift"
    ]
    warning_features = [
        report["feature"]
        for report in feature_reports
        if report.get("severity") == "warning"
    ]
    return {
        "project": "NeuroSight",
        "generated_at": utc_now(),
        "monitor": "cognitive_input_drift",
        "status": status,
        "source": source,
        "cohorts": {
            "baseline_rows": len(baseline_records),
            "current_rows": len(current_records),
        },
        "thresholds": asdict(applied_thresholds),
        "summary": {
            "drifted_features": drifted_features,
            "warning_features": warning_features,
            "ok_features": [
                report["feature"]
                for report in feature_reports
                if report.get("severity") == "ok"
            ],
        },
        "features": feature_reports,
        "recommended_actions": recommended_actions(status, drifted_features, warning_features),
        "clinical_boundary": (
            "Drift monitoring detects distribution changes in demo/research inputs. "
            "It does not validate clinical performance or authorize deployment."
        ),
    }


def recommended_actions(status: str, drifted_features: list[str], warning_features: list[str]) -> list[str]:
    """Return action text for a report status."""
    if status == "drift":
        features = ", ".join(drifted_features)
        return [
            f"Block silent promotion and review drifted features: {features}.",
            "Compare data collection, preprocessing, and cohort definitions before retraining.",
            "Run evaluation and fairness checks before using a new checkpoint.",
        ]
    if status == "warning":
        features = ", ".join(warning_features)
        return [
            f"Continue monitoring warning features: {features}.",
            "Inspect cohort composition and recent upload/preprocessing changes.",
            "Run a lightweight validation pass before model promotion.",
        ]
    return [
        "No material input drift detected under configured thresholds.",
        "Continue routine monitoring and keep baseline snapshots versioned.",
    ]


def generate_synthetic_cohorts(
    *,
    scenario: str = "warning",
    n_baseline: int = 240,
    n_current: int = 80,
    seed: int = 42,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Generate Git-safe baseline/current cohorts for drift demos."""
    rng = np.random.default_rng(seed)
    baseline = {
        "MMSE": rng.normal(25.5, 2.4, n_baseline),
        "MOCA": rng.normal(22.0, 2.8, n_baseline),
        "CDRSB": rng.choice([0.0, 0.5, 1.0, 2.0], n_baseline, p=[0.22, 0.34, 0.32, 0.12]),
        "ADAS11": rng.normal(15.0, 5.0, n_baseline),
        "RAVLT_immediate": rng.normal(40.0, 10.0, n_baseline),
        "RAVLT_learning": rng.normal(4.0, 3.0, n_baseline),
        "FAQ": rng.normal(4.0, 3.0, n_baseline),
        "AGE": rng.normal(71.0, 6.5, n_baseline),
    }

    if scenario == "stable":
        current = {
            feature: _stratified_sample_like(
                values,
                n_current,
                rng,
                noise_scale=0.02 if feature != "CDRSB" else 0.0,
            )
            for feature, values in baseline.items()
        }
        return _clip_and_pack(baseline), _clip_and_pack(current)

    if scenario == "drift":
        shift = {
            "MMSE": -3.2,
            "MOCA": -3.0,
            "CDRSB": 1.0,
            "ADAS11": 8.0,
            "RAVLT_immediate": -9.0,
            "RAVLT_learning": -3.0,
            "FAQ": 5.0,
            "AGE": 4.0,
        }
    else:
        shift = {
            "MMSE": -0.65,
            "MOCA": -0.65,
            "CDRSB": 0.0,
            "ADAS11": 1.5,
            "RAVLT_immediate": -1.5,
            "RAVLT_learning": -0.5,
            "FAQ": 0.8,
            "AGE": 1.0,
        }

    current = {
        feature: _stratified_sample_like(
            values,
            n_current,
            rng,
            noise_scale=0.05 if feature != "CDRSB" else 0.0,
        )
        for feature, values in baseline.items()
    }
    for feature, delta in shift.items():
        current[feature] = current[feature] + delta

    return _clip_and_pack(baseline), _clip_and_pack(current)


def _stratified_sample_like(
    values: np.ndarray,
    size: int,
    rng: np.random.Generator,
    *,
    noise_scale: float,
) -> np.ndarray:
    """Sample values across the baseline distribution to make a stable cohort."""
    sorted_values = np.sort(values.astype(np.float64))
    positions = ((np.arange(size) + 0.5) * sorted_values.size / max(1, size)).astype(int)
    positions = np.clip(positions, 0, sorted_values.size - 1)
    sampled = sorted_values[positions].copy()
    if noise_scale > 0:
        spread = float(np.std(sorted_values, ddof=1)) if sorted_values.size > 1 else 0.0
        sampled = sampled + rng.normal(0.0, max(spread * noise_scale, 1e-6), size)
    rng.shuffle(sampled)
    return sampled


def _clip_and_pack(columns: dict[str, np.ndarray]) -> list[dict[str, float]]:
    """Clip generated cognitive values and pack column arrays into records."""
    clipped = {
        "MMSE": np.clip(columns["MMSE"], 0, 30),
        "MOCA": np.clip(columns["MOCA"], 0, 30),
        "CDRSB": np.clip(columns["CDRSB"], 0, 18),
        "ADAS11": np.clip(columns["ADAS11"], 0, 70),
        "RAVLT_immediate": np.clip(columns["RAVLT_immediate"], 0, 75),
        "RAVLT_learning": np.clip(columns["RAVLT_learning"], -15, 15),
        "FAQ": np.clip(columns["FAQ"], 0, 30),
        "AGE": np.clip(columns["AGE"], 0, 120),
    }
    length = len(next(iter(clipped.values())))
    rows: list[dict[str, float]] = []
    for index in range(length):
        rows.append({feature: float(clipped[feature][index]) for feature in COGNITIVE_FEATURES})
    return rows


def load_feature_csv(path: str | Path, features: tuple[str, ...] = COGNITIVE_FEATURES) -> list[dict[str, Any]]:
    """Load a CSV file containing feature columns."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV path not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(features) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"CSV missing required feature columns: {', '.join(missing)}")
        return [{feature: row.get(feature) for feature in features} for row in reader]


def drift_report_to_json(report: dict[str, Any]) -> str:
    """Serialize a drift report with stable formatting."""
    return json.dumps(report, indent=2, sort_keys=True)


def write_drift_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a drift report to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(drift_report_to_json(report) + "\n", encoding="utf-8")
    return path
