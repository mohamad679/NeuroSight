"""Automated governance and compliance verification tests."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from neurosight.governance.model_card import (
    build_model_card_report,
    REQUIRED_ARTIFACT_LINKS,
    REQUIRED_SECTIONS,
    DISCLOSURE_TERMS,
    FORBIDDEN_CLAIMS,
)
from neurosight.governance.quality_gate import run_quality_gate
from evaluation.report import save_markdown_report


@pytest.mark.unit
def test_model_card_check_passes() -> None:
    """Validate that MODEL_CARD.md is fully compliant and passes checker."""
    root = Path(__file__).resolve().parents[1]
    report = build_model_card_report(root=root)
    failed = [c for c in report.get("checks", []) if not c["passed"]]
    assert report.get("status") == "passed", (
        f"Model card check failed. Failed checks:\n"
        + "\n".join(f"  - {c['check_id']}: {c['message']} | evidence: {c['evidence']}" for c in failed)
    )


@pytest.mark.unit
def test_quality_gate_strict_passes() -> None:
    """Validate that the strict quality gate check passes."""
    root = Path(__file__).resolve().parents[1]
    report = run_quality_gate(root=root)
    failed = [c for c in report.get("checks", []) if not c["passed"]]
    assert report.get("status") == "passed", (
        f"Quality gate checks failed. Failed checks:\n"
        + "\n".join(f"  - {c['gate_id']}: {c['message']} | evidence: {c['evidence']}" for c in failed)
    )


@pytest.mark.unit
def test_evaluation_results_schema() -> None:
    """Validate that evaluation/results.json matches the required schema."""
    root = Path(__file__).resolve().parents[1]
    results_path = root / "evaluation" / "results.json"
    assert results_path.exists(), "evaluation/results.json does not exist"

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Required top-level keys from task specification
    expected_keys = [
        "synthetic_data",
        "clinical_validity",
        "trained_on_real_data",
        "leakage_checked",
        "metrics",
        "accuracy",
        "macro_f1",
        "brier_score",
        "confusion_matrix",
        "class_names",
        "seed",
        "created_at",
        "dependency_versions",
        "limitations",
        "warnings",
        "command_used",
    ]
    for key in expected_keys:
        assert key in data, f"Required schema key '{key}' missing from results.json"

    # Ensure correct validation flag values
    assert data["synthetic_data"] is True, "synthetic_data must be true"
    assert data["clinical_validity"] is False, "clinical_validity must be false"
    assert data["trained_on_real_data"] is False, "trained_on_real_data must be false"
    assert data["leakage_checked"] is True, "leakage_checked must be true"

    # Verify test_metrics and metrics blocks have required keys
    for metrics_key in ("test_metrics", "metrics"):
        assert metrics_key in data, f"'{metrics_key}' key is missing from results.json"
        m = data[metrics_key]
        for metric in ("accuracy", "macro_f1", "macro_auc", "ece"):
            assert metric in m, f"Metric '{metric}' missing in '{metrics_key}'"

    # Verify top-level summary metrics
    for metric in ("accuracy", "macro_f1", "ece"):
        assert isinstance(data.get(metric), (int, float)), f"Top-level metric '{metric}' must be a number"

    # Must have macro_auc or macro_auroc at top level
    assert "macro_auc" in data or "macro_auroc" in data, (
        "Either macro_auc or macro_auroc must be present at top level"
    )

    # Warnings and limitations must be non-empty lists
    assert isinstance(data["limitations"], list) and len(data["limitations"]) > 0, (
        "limitations must be a non-empty list"
    )
    assert isinstance(data["warnings"], list) and len(data["warnings"]) > 0, (
        "warnings must be a non-empty list"
    )


@pytest.mark.unit
def test_no_fake_clinical_claims() -> None:
    """Validate that no fake clinical validation or regulatory approval claims exist."""
    root = Path(__file__).resolve().parents[1]

    results_path = root / "evaluation" / "results.json"
    with open(results_path, "r", encoding="utf-8") as f:
        results_content = f.read().lower()

    card_path = root / "MODEL_CARD.md"
    card_content = card_path.read_text(encoding="utf-8").lower()

    for claim in FORBIDDEN_CLAIMS:
        assert claim not in results_content, f"Forbidden claim '{claim}' found in results.json"
        assert claim not in card_content, f"Forbidden claim '{claim}' found in MODEL_CARD.md"


@pytest.mark.unit
def test_synthetic_disclosure_present() -> None:
    """Validate that mandatory synthetic disclosures are present in MODEL_CARD.md."""
    root = Path(__file__).resolve().parents[1]

    card_content = (root / "MODEL_CARD.md").read_text(encoding="utf-8").lower()
    for disclosure in DISCLOSURE_TERMS:
        assert disclosure in card_content, (
            f"Required disclosure '{disclosure}' missing from MODEL_CARD.md"
        )


@pytest.mark.unit
def test_evaluation_report_contains_limitations(tmp_path: Path) -> None:
    """Validate that a generated Markdown report contains safety disclosures and limitations."""
    root = Path(__file__).resolve().parents[1]
    results_path = root / "evaluation" / "results.json"

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    report_path = tmp_path / "temp_report.md"
    save_markdown_report(results, report_path)

    assert report_path.exists(), "Markdown report was not generated"
    content = report_path.read_text(encoding="utf-8")

    # Mandatory banner and section assertions
    assert "SYNTHETIC BENCHMARK" in content, "Warning banner missing from report"
    assert "## Limitations" in content, "Limitations section missing from report"
    assert "## Methodology" in content or "Methodology" in content, (
        "Methodology section missing from report"
    )
    assert "reproduce" in content.lower(), "Reproducibility instructions missing from report"
    assert "command" in content.lower(), "Command used info missing from report"


@pytest.mark.unit
def test_artifact_links_exist_or_are_removed() -> None:
    """Validate that all artifact links listed in MODEL_CARD.md exist on disk."""
    root = Path(__file__).resolve().parents[1]
    card_text = (root / "MODEL_CARD.md").read_text(encoding="utf-8")

    missing_from_card = [link for link in REQUIRED_ARTIFACT_LINKS if link not in card_text]
    missing_on_disk = [link for link in REQUIRED_ARTIFACT_LINKS if not (root / link).exists()]

    assert not missing_from_card, (
        f"Required artifact links missing from MODEL_CARD.md: {missing_from_card}"
    )
    assert not missing_on_disk, (
        f"Artifact links referenced in MODEL_CARD.md do not exist on disk: {missing_on_disk}"
    )


@pytest.mark.unit
def test_model_card_required_sections_present() -> None:
    """Validate every required section heading appears in MODEL_CARD.md."""
    import re
    root = Path(__file__).resolve().parents[1]
    card_text = (root / "MODEL_CARD.md").read_text(encoding="utf-8")

    def normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    headings = set()
    for line in card_text.splitlines():
        m = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if m:
            headings.add(normalize(m.group(1)))

    missing = [s for s in REQUIRED_SECTIONS if normalize(s) not in headings]
    assert not missing, (
        f"Required sections missing from MODEL_CARD.md: {missing}"
    )
