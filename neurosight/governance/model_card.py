"""Model-card quality checks for NeuroSight documentation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REQUIRED_SECTIONS: tuple[str, ...] = (
    "Status Snapshot",
    "Model Status",
    "Model Details",
    "Architecture And Pipeline",
    "Input And Output Contract",
    "Intended Use",
    "Out-Of-Scope Use",
    "Training Data Status",
    "Evaluation Data Status",
    "Data And Training Scope",
    "Synthetic Benchmark Disclosure",
    "Clinical Validation Status",
    "Evaluation Results",
    "Safety Limitations",
    "Bias And Fairness Limitations",
    "Data Privacy Limitations",
    "Human Oversight Requirement",
    "Known Failure Modes",
    "Known Limitations",
    "Responsible Use",
    "Reviewer Interpretation",
    "Reproducibility",
    "Explainability And Reporting",
    "Safety And Governance",
    "Bias, Fairness, And Representation",
    "Limitations",
    "Deployment And Monitoring",
    "Evidence And Artifacts",
    "Versioning And Artifacts",
    "Clinical Disclaimer",
    "Citation",
)

DISCLOSURE_TERMS: tuple[str, ...] = (
    "synthetic adni-like",
    "not clinical performance",
    "not a medical device",
    "specialist review",
    "no real patient data",
    "cognitive-only",
    "not clinically validated",
    "not approved for diagnosis",
    "external validation is not yet performed",
    "synthetic benchmark results are not medical evidence",
)

FORBIDDEN_CLAIMS: tuple[str, ...] = (
    "fda approved",
    "clinically validated model",
    "ready for clinical use",
    "diagnostic accuracy on real patients",
    "autonomous diagnosis",
)

REQUIRED_ARTIFACT_LINKS: tuple[str, ...] = (
    "evaluation/results.json",
    "docs/MONAI_PIPELINE.md",
    "docs/MLFLOW_REGISTRY.md",
    "docs/DVC_PROVENANCE.md",
    "docs/AI_SAFETY_OWASP_GENAI.md",
    "docs/OPENTELEMETRY_OBSERVABILITY.md",
    "docs/DRIFT_MONITORING.md",
    "docs/ONNX_RUNTIME_EXPORT.md",
)


@dataclass(frozen=True)
class ModelCardCheck:
    """One model-card quality check result."""

    check_id: str
    passed: bool
    message: str
    evidence: dict[str, Any] | None = None


def utc_now() -> str:
    """Return current UTC time as an ISO-8601 instant."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_heading(text: str) -> str:
    """Normalize a markdown heading for robust comparison."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def extract_headings(markdown: str) -> list[str]:
    """Extract markdown heading titles without the leading hash marks."""
    headings: list[str] = []
    for line in markdown.splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if match:
            headings.append(match.group(1).strip())
    return headings


def load_metrics(results_path: str | Path) -> dict[str, float]:
    """Load the evaluation metrics used by the model card."""
    path = Path(results_path)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    metrics = raw.get("test_metrics", {})
    if not isinstance(metrics, dict):
        return {}
    return {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "macro_auc": float(metrics.get("macro_auc", 0.0)),
        "ece": float(metrics.get("ece", 0.0)),
    }


def _metric_strings(metrics: dict[str, float]) -> dict[str, str]:
    return {
        "accuracy_percent": f"{metrics.get('accuracy', 0.0) * 100:.1f}%",
        "macro_f1": f"{metrics.get('macro_f1', 0.0):.3f}",
        "macro_auc": f"{metrics.get('macro_auc', 0.0):.3f}",
        "ece": f"{metrics.get('ece', 0.0):.3f}",
    }


def _check_required_sections(headings: list[str]) -> ModelCardCheck:
    normalized = {normalize_heading(heading) for heading in headings}
    missing = [
        section
        for section in REQUIRED_SECTIONS
        if normalize_heading(section) not in normalized
    ]
    return ModelCardCheck(
        check_id="required_sections",
        passed=not missing,
        message="All required model-card sections are present." if not missing else "Required sections are missing.",
        evidence={"missing": missing, "required_count": len(REQUIRED_SECTIONS)},
    )


def _check_disclosures(markdown_lower: str) -> ModelCardCheck:
    missing = [term for term in DISCLOSURE_TERMS if term not in markdown_lower]
    return ModelCardCheck(
        check_id="disclosures",
        passed=not missing,
        message="Clinical-use and synthetic-data disclosures are present." if not missing else "Disclosure terms are missing.",
        evidence={"missing_terms": missing},
    )


def _check_forbidden_claims(markdown_lower: str) -> ModelCardCheck:
    found = [claim for claim in FORBIDDEN_CLAIMS if claim in markdown_lower]
    return ModelCardCheck(
        check_id="forbidden_claims",
        passed=not found,
        message="No forbidden clinical or regulatory claims found." if not found else "Forbidden claims found.",
        evidence={"found": found},
    )


def _check_metrics(markdown: str, metrics: dict[str, float]) -> ModelCardCheck:
    expected = _metric_strings(metrics)
    missing = [label for label, value in expected.items() if value not in markdown]
    return ModelCardCheck(
        check_id="metrics_consistency",
        passed=bool(metrics) and not missing,
        message="Model-card metrics match evaluation/results.json." if metrics and not missing else "Model-card metrics are incomplete or inconsistent.",
        evidence={"expected_strings": expected, "missing": missing},
    )


def _check_artifact_links(markdown: str, root: Path) -> ModelCardCheck:
    missing_from_card = [link for link in REQUIRED_ARTIFACT_LINKS if link not in markdown]
    missing_on_disk = [link for link in REQUIRED_ARTIFACT_LINKS if not (root / link).exists()]
    return ModelCardCheck(
        check_id="artifact_links",
        passed=not missing_from_card and not missing_on_disk,
        message="Required evidence links are present and exist on disk." if not missing_from_card and not missing_on_disk else "Evidence links are missing.",
        evidence={
            "missing_from_card": missing_from_card,
            "missing_on_disk": missing_on_disk,
        },
    )


def build_model_card_report(
    model_card_path: str | Path = "MODEL_CARD.md",
    results_path: str | Path = "evaluation/results.json",
    *,
    root: str | Path = ".",
) -> dict[str, Any]:
    """Build a JSON-safe model-card quality report."""
    root_path = Path(root)
    card_path = root_path / model_card_path
    results_full_path = root_path / results_path
    markdown = card_path.read_text(encoding="utf-8")
    markdown_lower = markdown.lower()
    headings = extract_headings(markdown)
    metrics = load_metrics(results_full_path)

    checks = [
        _check_required_sections(headings),
        _check_disclosures(markdown_lower),
        _check_forbidden_claims(markdown_lower),
        _check_metrics(markdown, metrics),
        _check_artifact_links(markdown, root_path),
    ]
    passed_count = sum(1 for check in checks if check.passed)
    status = "passed" if passed_count == len(checks) else "failed"
    return {
        "generated_at": utc_now(),
        "status": status,
        "model_card": str(model_card_path),
        "results_artifact": str(results_path),
        "summary": {
            "checks_total": len(checks),
            "checks_passed": passed_count,
            "checks_failed": len(checks) - passed_count,
            "heading_count": len(headings),
        },
        "metrics_source": metrics,
        "checks": [asdict(check) for check in checks],
        "clinical_boundary": (
            "This report validates documentation quality and disclosure coverage. "
            "It does not validate model performance, clinical safety, or regulatory readiness."
        ),
    }


def report_to_json(report: dict[str, Any]) -> str:
    """Serialize a model-card report with stable formatting."""
    return json.dumps(report, indent=2, sort_keys=True)


def write_model_card_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write the model-card quality report to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_to_json(report) + "\n", encoding="utf-8")
    return path
