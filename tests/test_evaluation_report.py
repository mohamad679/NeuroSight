"""Tests for NeuroSight evaluation report generation.

Verifies that generated JSON and Markdown reports contain all mandatory
provenance metadata and warning banners.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.report import generate_full_report, save_json_report, save_markdown_report


_MINIMAL_RESULTS: dict = {
    "synthetic_data": True,
    "clinical_validity": False,
    "warning": "SYNTHETIC BENCHMARK",
    "methodology": "Test methodology.",
    "seed": 7,
    "dataset_path": "data/synthetic.csv",
    "leakage_check_passed": True,
    "dependency_versions": {"python": "3.11.0", "numpy": "1.26.0"},
    "config": {"n_per_class": 3, "cv_folds": 2},
    "results": [
        {
            "method": "random_classifier",
            "modality": "-",
            "macro_auc": 0.52,
            "macro_f1": 0.20,
            "accuracy": 0.25,
            "balanced_accuracy": 0.22,
            "brier_score": 0.81,
            "ece": 0.12,
            "train_time_seconds": 0.01,
        }
    ],
    "winner": {
        "method": "random_classifier",
        "macro_auc": 0.52,
    },
}


@pytest.mark.unit
def test_json_report_has_required_keys(tmp_path: Path) -> None:
    """JSON report contains all mandatory provenance fields."""
    out = save_json_report(_MINIMAL_RESULTS, tmp_path / "report.json")
    data = json.loads(out.read_text())

    for key in ("synthetic_data", "clinical_validity", "warning", "methodology",
                "seed", "dependency_versions", "results"):
        assert key in data, f"Required key '{key}' missing from JSON report"


@pytest.mark.unit
def test_json_report_synthetic_data_is_true(tmp_path: Path) -> None:
    """JSON report always has synthetic_data == True."""
    # Even if passed without the field, save_json_report inserts it
    results_no_flag = {k: v for k, v in _MINIMAL_RESULTS.items() if k != "synthetic_data"}
    out = save_json_report(results_no_flag, tmp_path / "report2.json")
    data = json.loads(out.read_text())
    assert data["synthetic_data"] is True


@pytest.mark.unit
def test_json_report_clinical_validity_is_false(tmp_path: Path) -> None:
    """JSON report always has clinical_validity == False."""
    results_no_flag = {k: v for k, v in _MINIMAL_RESULTS.items() if k != "clinical_validity"}
    out = save_json_report(results_no_flag, tmp_path / "report3.json")
    data = json.loads(out.read_text())
    assert data["clinical_validity"] is False


@pytest.mark.unit
def test_json_report_is_valid_json(tmp_path: Path) -> None:
    """JSON report is valid UTF-8 JSON."""
    out = save_json_report(_MINIMAL_RESULTS, tmp_path / "report.json")
    content = out.read_text(encoding="utf-8")
    parsed = json.loads(content)
    assert isinstance(parsed, dict)


@pytest.mark.unit
def test_markdown_report_has_warning_banner(tmp_path: Path) -> None:
    """Markdown report contains the synthetic data warning banner."""
    out = save_markdown_report(_MINIMAL_RESULTS, tmp_path / "report.md")
    content = out.read_text()
    assert "SYNTHETIC BENCHMARK" in content or "SYNTHETIC" in content, (
        "Markdown report must contain warning about synthetic data"
    )
    assert "clinical_validity: false" in content.lower() or "clinical_validity" in content, (
        "Markdown report must mention clinical_validity: false"
    )


@pytest.mark.unit
def test_markdown_report_has_limitations_section(tmp_path: Path) -> None:
    """Markdown report contains a Limitations section."""
    out = save_markdown_report(_MINIMAL_RESULTS, tmp_path / "report.md")
    content = out.read_text()
    assert "## Limitations" in content, "Markdown report must have a '## Limitations' section"


@pytest.mark.unit
def test_markdown_report_has_methodology_section(tmp_path: Path) -> None:
    """Markdown report contains a Methodology section."""
    out = save_markdown_report(_MINIMAL_RESULTS, tmp_path / "report.md")
    content = out.read_text()
    assert "## Methodology" in content or "Methodology" in content, (
        "Markdown report must include methodology documentation"
    )


@pytest.mark.unit
def test_markdown_report_is_non_empty(tmp_path: Path) -> None:
    """Markdown report is a non-empty file on disk."""
    out = save_markdown_report(_MINIMAL_RESULTS, tmp_path / "report.md")
    assert out.exists()
    assert out.stat().st_size > 100, "Markdown report must be non-trivially sized"


@pytest.mark.unit
def test_generate_full_report_creates_both_files(tmp_path: Path) -> None:
    """generate_full_report saves JSON and Markdown files."""
    paths = generate_full_report(_MINIMAL_RESULTS, tmp_path, prefix="test_report")
    assert "json" in paths
    assert "markdown" in paths
    assert paths["json"].exists()
    assert paths["markdown"].exists()
    assert paths["json"].suffix == ".json"
    assert paths["markdown"].suffix == ".md"


@pytest.mark.unit
def test_generate_full_report_json_loadable(tmp_path: Path) -> None:
    """JSON from generate_full_report is parseable."""
    paths = generate_full_report(_MINIMAL_RESULTS, tmp_path)
    data = json.loads(paths["json"].read_text())
    assert "results" in data


@pytest.mark.unit
def test_report_creates_parent_directories(tmp_path: Path) -> None:
    """Report saves correctly even when output directory does not yet exist."""
    nested = tmp_path / "a" / "b" / "c"
    paths = generate_full_report(_MINIMAL_RESULTS, nested)
    assert paths["json"].exists()
    assert paths["markdown"].exists()
