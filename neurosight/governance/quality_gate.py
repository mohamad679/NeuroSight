"""Offline CI/CD quality gate for NeuroSight release readiness."""

from __future__ import annotations

import json
import py_compile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

GateSeverity = Literal["blocker", "warning", "info"]

REQUIRED_REPO_FILES: tuple[str, ...] = (
    "README.md",
    "PROJECT_STATUS.md",
    "MODEL_CARD.md",
    "SECURITY.md",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "frontend/package.json",
)

ROADMAP_DOCS: tuple[str, ...] = (
    "docs/IMPLEMENTED_VS_PLANNED.md",
    "docs/ARCHITECTURE_OVERVIEW.md",
    "docs/API_CONTRACT_CHECKS.md",
    "docs/GITHUB_RELEASE_READINESS.md",
    "docs/PUBLIC_REPOSITORY_GUIDE.md",
    "docs/MONAI_PIPELINE.md",
    "docs/MLFLOW_REGISTRY.md",
    "docs/DVC_PROVENANCE.md",
    "docs/OPENTELEMETRY_OBSERVABILITY.md",
    "docs/FHIR_EXPORT.md",
    "docs/DICOM_DICOMWEB.md",
    "docs/LANGGRAPH_AGENT_WORKFLOW.md",
    "docs/DRIFT_MONITORING.md",
    "docs/ONNX_RUNTIME_EXPORT.md",
    "docs/SECURITY_SUPPLY_CHAIN.md",
    "docs/AI_SAFETY_OWASP_GENAI.md",
    "docs/MODEL_CARD_POLISH.md",
    "docs/CI_CD_QUALITY_GATE.md",
    "docs/DEMO_SCRIPT.md",
    "docs/PORTFOLIO_CHECKLIST.md",
)

RUNNABLE_SCRIPTS: tuple[str, ...] = (
    "scripts/smoke_backend.py",
    "scripts/api_contract_check.py",
    "scripts/github_readiness.py",
    "scripts/mlflow_registry.py",
    "scripts/dvc_provenance.py",
    "scripts/otel_probe.py",
    "scripts/fhir_export.py",
    "scripts/dicomweb_manifest.py",
    "scripts/langgraph_workflow.py",
    "scripts/drift_monitor.py",
    "scripts/onnx_export.py",
    "scripts/supply_chain_audit.py",
    "scripts/ai_safety_eval.py",
    "scripts/model_card_check.py",
    "scripts/quality_gate.py",
    "scripts/portfolio_check.py",
)

REQUIRED_WORKFLOWS: tuple[str, ...] = (
    ".github/workflows/ci.yml",
    ".github/workflows/security_supply_chain.yml",
    ".github/workflows/deploy_spaces.yml",
    ".github/workflows/quality_gate.yml",
)

REQUIRED_MAKE_TARGETS: tuple[str, ...] = (
    "smoke-backend:",
    "api-contract-check:",
    "mlflow-registry:",
    "dvc-provenance:",
    "otel-probe:",
    "fhir-export:",
    "dicomweb-manifest:",
    "langgraph-workflow:",
    "drift-monitor:",
    "onnx-export:",
    "supply-chain-audit:",
    "ai-safety-eval:",
    "model-card-check:",
    "quality-gate:",
    "github-readiness:",
    "portfolio-check:",
)

REQUIRED_METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "macro_f1",
    "macro_auc",
    "ece",
)


@dataclass(frozen=True)
class GateCheck:
    """One quality-gate check result."""

    gate_id: str
    passed: bool
    severity: GateSeverity
    message: str
    evidence: dict[str, Any] | None = None


def utc_now() -> str:
    """Return current UTC time as an ISO-8601 instant."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _missing_paths(root: Path, paths: tuple[str, ...]) -> list[str]:
    return [path for path in paths if not (root / path).exists()]


def _check_required_files(root: Path) -> GateCheck:
    missing = _missing_paths(root, REQUIRED_REPO_FILES)
    return GateCheck(
        gate_id="required_repo_files",
        passed=not missing,
        severity="blocker",
        message="Required repository files are present." if not missing else "Required repository files are missing.",
        evidence={"missing": missing, "required_count": len(REQUIRED_REPO_FILES)},
    )


def _check_roadmap_artifacts(root: Path) -> GateCheck:
    missing_docs = _missing_paths(root, ROADMAP_DOCS)
    missing_scripts = _missing_paths(root, RUNNABLE_SCRIPTS)
    return GateCheck(
        gate_id="roadmap_artifacts",
        passed=not missing_docs and not missing_scripts,
        severity="blocker",
        message="Roadmap docs and runnable scripts are present." if not missing_docs and not missing_scripts else "Roadmap docs or runnable scripts are missing.",
        evidence={
            "missing_docs": missing_docs,
            "missing_scripts": missing_scripts,
            "docs_count": len(ROADMAP_DOCS),
            "scripts_count": len(RUNNABLE_SCRIPTS),
        },
    )


def _python_files_for_compile(root: Path) -> list[Path]:
    files: list[Path] = []
    for folder in ("neurosight", "scripts"):
        folder_path = root / folder
        if not folder_path.exists():
            continue
        files.extend(
            path
            for path in sorted(folder_path.rglob("*.py"))
            if "__pycache__" not in path.parts
        )
    return files


def _check_python_syntax(root: Path) -> GateCheck:
    failures: list[dict[str, str]] = []
    files = _python_files_for_compile(root)
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(
                {
                    "path": str(path.relative_to(root)),
                    "error": str(exc),
                }
            )
    return GateCheck(
        gate_id="python_syntax",
        passed=not failures,
        severity="blocker",
        message="Python files compile successfully." if not failures else "Python syntax failures detected.",
        evidence={"file_count": len(files), "failures": failures[:10]},
    )


def _check_evaluation_artifact(root: Path) -> GateCheck:
    path = root / "evaluation" / "results.json"
    if not path.exists():
        return GateCheck(
            gate_id="evaluation_artifact",
            passed=False,
            severity="blocker",
            message="evaluation/results.json is missing.",
            evidence={"path": "evaluation/results.json"},
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return GateCheck(
            gate_id="evaluation_artifact",
            passed=False,
            severity="blocker",
            message="evaluation/results.json is not valid JSON.",
            evidence={"error": str(exc)},
        )
    metrics = raw.get("test_metrics", {})
    missing_metrics = [
        key
        for key in REQUIRED_METRIC_KEYS
        if key not in metrics or not isinstance(metrics.get(key), (int, float))
    ]
    
    synthetic_data = raw.get("synthetic_data")
    clinical_validity = raw.get("clinical_validity")
    trained_on_real_data = raw.get("trained_on_real_data")
    leakage_checked = raw.get("leakage_checked")

    issues = []
    if synthetic_data is not True:
        issues.append("synthetic_data must be true")
    if clinical_validity is not False:
        issues.append("clinical_validity must be false")
    if trained_on_real_data is not False:
        issues.append("trained_on_real_data must be false")
    if leakage_checked is not True:
        issues.append("leakage_checked must be true")

    # Required top-level provenance fields
    required_provenance = ["seed", "created_at", "command_used", "dependency_versions", "limitations", "warnings"]
    missing_provenance = [k for k in required_provenance if k not in raw]
    if missing_provenance:
        issues.append(f"missing provenance fields: {missing_provenance}")

    passed = (not missing_metrics) and (not issues)

    msg_parts = []
    if missing_metrics:
        msg_parts.append("incomplete metrics")
    if issues:
        msg_parts.append("; ".join(issues))

    message = "Evaluation metrics artifact is valid." if passed else f"Evaluation metrics artifact is invalid: {'; '.join(msg_parts)}."

    return GateCheck(
        gate_id="evaluation_artifact",
        passed=passed,
        severity="blocker",
        message=message,
        evidence={
            "missing_metrics": missing_metrics,
            "metric_keys": sorted(metrics.keys()) if isinstance(metrics, dict) else [],
            "issues": issues,
            "synthetic_data": synthetic_data,
            "clinical_validity": clinical_validity,
            "trained_on_real_data": trained_on_real_data,
            "leakage_checked": leakage_checked,
        },
    )


def _check_model_card(root: Path) -> GateCheck:
    from neurosight.governance.model_card import build_model_card_report

    report = build_model_card_report(root=root)
    return GateCheck(
        gate_id="model_card",
        passed=report.get("status") == "passed",
        severity="blocker",
        message="Model-card disclosure check passed." if report.get("status") == "passed" else "Model-card disclosure check failed.",
        evidence=report.get("summary"),
    )


def _check_ai_safety() -> GateCheck:
    from neurosight.governance.ai_safety import run_ai_safety_evaluation

    report = run_ai_safety_evaluation()
    return GateCheck(
        gate_id="ai_safety",
        passed=report.get("status") == "passed",
        severity="blocker",
        message="OWASP GenAI safety regression suite passed." if report.get("status") == "passed" else "OWASP GenAI safety regression suite failed.",
        evidence=report.get("summary"),
    )


def _check_supply_chain(root: Path) -> GateCheck:
    from neurosight.security.supply_chain import build_supply_chain_report

    report = build_supply_chain_report(root)
    summary = report.get("summary", {})
    counts = summary.get("severity_counts", {})
    critical_count = int(counts.get("critical", 0)) if isinstance(counts, dict) else 0
    return GateCheck(
        gate_id="supply_chain_critical",
        passed=critical_count == 0,
        severity="blocker",
        message="No critical supply-chain findings detected." if critical_count == 0 else "Critical supply-chain findings detected.",
        evidence={
            "status": report.get("status"),
            "severity_counts": counts,
            "finding_count": summary.get("finding_count"),
        },
    )


def _check_workflows(root: Path) -> GateCheck:
    missing = _missing_paths(root, REQUIRED_WORKFLOWS)
    workflow_issues: list[dict[str, str]] = []
    for workflow in REQUIRED_WORKFLOWS:
        path = root / workflow
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if "permissions:" not in text or "contents: read" not in text:
            workflow_issues.append({"path": workflow, "issue": "missing minimal permissions"})
        if "actions/checkout" in text and "persist-credentials: false" not in text:
            workflow_issues.append({"path": workflow, "issue": "checkout credentials persist"})
    return GateCheck(
        gate_id="workflow_baseline",
        passed=not missing and not workflow_issues,
        severity="blocker",
        message="GitHub workflow baseline is present and hardened." if not missing and not workflow_issues else "Workflow baseline is incomplete.",
        evidence={"missing": missing, "issues": workflow_issues},
    )


def _check_make_targets(root: Path) -> GateCheck:
    path = root / "Makefile"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    missing = [target for target in REQUIRED_MAKE_TARGETS if target not in text]
    return GateCheck(
        gate_id="make_targets",
        passed=not missing,
        severity="warning",
        message="Makefile exposes roadmap commands." if not missing else "Makefile is missing roadmap commands.",
        evidence={"missing": missing},
    )


def run_quality_gate(root: str | Path = ".") -> dict[str, Any]:
    """Run the offline quality gate and return a JSON-safe report."""
    root_path = Path(root).resolve()
    checks = [
        _check_required_files(root_path),
        _check_roadmap_artifacts(root_path),
        _check_python_syntax(root_path),
        _check_evaluation_artifact(root_path),
        _check_model_card(root_path),
        _check_ai_safety(),
        _check_supply_chain(root_path),
        _check_workflows(root_path),
        _check_make_targets(root_path),
    ]
    failed_blockers = [
        check.gate_id
        for check in checks
        if not check.passed and check.severity == "blocker"
    ]
    failed_warnings = [
        check.gate_id
        for check in checks
        if not check.passed and check.severity == "warning"
    ]
    status = "passed" if not failed_blockers else "failed"
    return {
        "project": "NeuroSight",
        "generated_at": utc_now(),
        "status": status,
        "repository_root": str(root_path),
        "summary": {
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check.passed),
            "checks_failed": sum(1 for check in checks if not check.passed),
            "failed_blockers": failed_blockers,
            "failed_warnings": failed_warnings,
        },
        "checks": [asdict(check) for check in checks],
        "quality_gate_boundary": (
            "This gate checks offline repository readiness for CI/CD. It does not "
            "replace full dependency installation, unit tests, vulnerability database "
            "scans, external deployment validation, or clinical validation."
        ),
    }


def report_to_json(report: dict[str, Any]) -> str:
    """Serialize a quality-gate report with stable formatting."""
    return json.dumps(report, indent=2, sort_keys=True)


def write_quality_gate_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write the quality-gate report to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_to_json(report) + "\n", encoding="utf-8")
    return path
