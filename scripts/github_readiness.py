#!/usr/bin/env python3
"""Audit NeuroSight's GitHub portfolio release readiness."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "logs/github_readiness/neurosight_github_readiness_report.json"
Severity = Literal["blocker", "warning"]

REQUIRED_PUBLIC_FILES: tuple[str, ...] = (
    "README.md",
    "PROJECT_STATUS.md",
    "MODEL_CARD.md",
    "SECURITY.md",
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "frontend/package.json",
)

REQUIRED_EVIDENCE_FILES: tuple[str, ...] = (
    "docs/ARCHITECTURE_OVERVIEW.md",
    "docs/IMPLEMENTED_VS_PLANNED.md",
    "docs/API_CONTRACT_CHECKS.md",
    "docs/DEMO_SCRIPT.md",
    "docs/PORTFOLIO_CHECKLIST.md",
    "docs/PUBLIC_REPOSITORY_GUIDE.md",
    "docs/MONAI_PIPELINE.md",
    "docs/LANGGRAPH_AGENT_WORKFLOW.md",
    "docs/AI_SAFETY_OWASP_GENAI.md",
    "docs/CI_CD_QUALITY_GATE.md",
    "scripts/portfolio_check.py",
    "scripts/api_contract_check.py",
    "scripts/quality_gate.py",
    "scripts/ai_safety_eval.py",
    "scripts/model_card_check.py",
    "scripts/langgraph_workflow.py",
)

REQUIRED_README_HEADINGS: tuple[str, ...] = (
    "## 🚀 Live Demo",
    "## 📌 Highlights",
    "## What This Project Demonstrates",
    "## Current Limitations",
    "## 🏗️ Architecture",
    "## 🚀 Quick Start",
    "## 🔌 API Reference",
    "## 📁 Project Structure",
    "## 📚 Datasets",
    "## ⚠️ Detailed Limitations",
    "## 📄 License",
)

REQUIRED_WORKFLOWS: tuple[str, ...] = (
    ".github/workflows/ci.yml",
    ".github/workflows/security_supply_chain.yml",
    ".github/workflows/deploy_spaces.yml",
    ".github/workflows/quality_gate.yml",
)

REQUIRED_GITIGNORE_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "frontend/.env",
    "CLAUDE.md",
    ".claude/",
    ".codex/",
    ".cursor/",
    ".windsurf/",
    "logs/",
    "frontend/node_modules/",
    "frontend/.next/",
    "checkpoints/",
    "data/",
    "*.pt",
    "*.onnx",
)

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("OpenAI-style API key", re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{24,}\b")),
    ("NeuroSight API key", re.compile(r"\bns-[A-Za-z0-9_-]{24,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{24,}\b")),
)

SECRET_SCAN_EXCLUDED_DIRS: set[str] = {
    ".git",
    ".ruff_cache",
    ".next",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "logs",
    "data",
    "checkpoints",
    ".deploy",
}

SECRET_SCAN_EXCLUDED_FILES: set[str] = {
    "CLAUDE.md",
    ".env",
    ".env.local",
    "frontend/.env",
    "frontend/.env.local",
    "package-lock.json",
}

PRIVATE_WORKING_PATTERNS: tuple[str, ...] = (
    "CLAUDE.md",
    ".claude/",
    ".codex/",
    ".cursor/",
    ".windsurf/",
)

TEXT_SUFFIXES: set[str] = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class ReadinessCheck:
    """One GitHub readiness audit check."""

    check_id: str
    passed: bool
    severity: Severity
    message: str
    evidence: dict[str, Any]


def utc_now() -> str:
    """Return current UTC time as an ISO-8601 instant."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit NeuroSight for GitHub portfolio release readiness."
    )
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help=f"Output JSON path. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--stdout", action="store_true", help="Print JSON to stdout instead of writing a file.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if blocker checks fail.")
    parser.add_argument("--fail-warnings", action="store_true", help="Treat warnings as strict-mode failures.")
    return parser.parse_args()


def _read(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def _missing(paths: tuple[str, ...]) -> list[str]:
    return [path for path in paths if not (PROJECT_ROOT / path).exists()]


def _check_required_public_files() -> ReadinessCheck:
    missing = _missing(REQUIRED_PUBLIC_FILES)
    return ReadinessCheck(
        check_id="required_public_files",
        passed=not missing,
        severity="blocker",
        message="Required public-facing repository files are present." if not missing else "Required public-facing repository files are missing.",
        evidence={"missing": missing, "required_count": len(REQUIRED_PUBLIC_FILES)},
    )


def _check_evidence_files() -> ReadinessCheck:
    missing = _missing(REQUIRED_EVIDENCE_FILES)
    return ReadinessCheck(
        check_id="evidence_files",
        passed=not missing,
        severity="blocker",
        message="Reviewer evidence docs and scripts are present." if not missing else "Reviewer evidence docs or scripts are missing.",
        evidence={"missing": missing, "required_count": len(REQUIRED_EVIDENCE_FILES)},
    )


def _check_readme_structure() -> ReadinessCheck:
    text = _read("README.md") if (PROJECT_ROOT / "README.md").exists() else ""
    missing = [heading for heading in REQUIRED_README_HEADINGS if heading not in text]
    return ReadinessCheck(
        check_id="readme_structure",
        passed=not missing,
        severity="blocker",
        message="README contains the expected reviewer sections." if not missing else "README is missing expected reviewer sections.",
        evidence={"missing_headings": missing},
    )


def _check_positioning_disclosures() -> ReadinessCheck:
    sources = ["README.md", "PROJECT_STATUS.md", "MODEL_CARD.md"]
    combined = "\n".join(_read(path).lower() for path in sources if (PROJECT_ROOT / path).exists())
    required_terms = {
        "synthetic": "synthetic",
        "not_clinical": "not clinical",
        "not_medical_device": "not a medical device",
        "research_prototype": "research prototype",
        "specialist_review": "specialist review",
    }
    missing = [
        label
        for label, term in required_terms.items()
        if term not in combined
    ]
    return ReadinessCheck(
        check_id="positioning_disclosures",
        passed=not missing,
        severity="blocker",
        message="Synthetic-data and non-clinical disclosures are visible." if not missing else "Core safety/positioning disclosures are missing.",
        evidence={"missing_disclosure_terms": missing, "sources": sources},
    )


def _check_gitignore_hygiene() -> ReadinessCheck:
    text = _read(".gitignore") if (PROJECT_ROOT / ".gitignore").exists() else ""
    lines = {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}
    missing = [pattern for pattern in REQUIRED_GITIGNORE_PATTERNS if pattern not in lines]
    return ReadinessCheck(
        check_id="gitignore_hygiene",
        passed=not missing,
        severity="blocker",
        message="Git ignore policy protects secrets and generated artifacts." if not missing else "Git ignore policy is missing important patterns.",
        evidence={"missing_patterns": missing},
    )


def _check_private_working_file_policy() -> ReadinessCheck:
    text = _read(".gitignore") if (PROJECT_ROOT / ".gitignore").exists() else ""
    lines = {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}
    missing = [pattern for pattern in PRIVATE_WORKING_PATTERNS if pattern not in lines]
    present_locally = [
        pattern
        for pattern in PRIVATE_WORKING_PATTERNS
        if (PROJECT_ROOT / pattern.rstrip("/")).exists()
    ]
    return ReadinessCheck(
        check_id="private_working_file_policy",
        passed=not missing,
        severity="blocker",
        message="Local assistant and IDE working files are ignored by policy." if not missing else "Local assistant or IDE working-file ignore rules are missing.",
        evidence={
            "missing_ignore_patterns": missing,
            "present_locally_but_ignored": present_locally,
        },
    )


def _candidate_text_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT_ROOT)
        relative_posix = relative.as_posix()
        if any(part in SECRET_SCAN_EXCLUDED_DIRS for part in relative.parts):
            continue
        if relative_posix in SECRET_SCAN_EXCLUDED_FILES:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.stat().st_size > 1_000_000:
            continue
        files.append(path)
    return files


def _check_secret_scan() -> ReadinessCheck:
    findings: list[dict[str, str]] = []
    scanned = 0
    for path in _candidate_text_files():
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(
                    {
                        "path": path.relative_to(PROJECT_ROOT).as_posix(),
                        "kind": label,
                    }
                )
    return ReadinessCheck(
        check_id="secret_scan_public_files",
        passed=not findings,
        severity="blocker",
        message="No obvious API tokens detected in public candidate files." if not findings else "Potential secrets detected in public candidate files.",
        evidence={"findings": findings[:20], "scanned_files": scanned},
    )


def _check_visual_evidence() -> ReadinessCheck:
    figure_count = len(list((PROJECT_ROOT / "docs" / "figures").glob("*.png")))
    screenshot_count = len(list((PROJECT_ROOT / "docs" / "screenshots").glob("*.png")))
    passed = figure_count >= 5 and screenshot_count >= 3
    return ReadinessCheck(
        check_id="visual_evidence",
        passed=passed,
        severity="warning",
        message="Figures and screenshots are available for reviewer scanning." if passed else "Add more figures/screenshots for reviewer scanning.",
        evidence={"figure_png_count": figure_count, "screenshot_png_count": screenshot_count},
    )


def _check_workflows() -> ReadinessCheck:
    missing = _missing(REQUIRED_WORKFLOWS)
    return ReadinessCheck(
        check_id="github_workflows",
        passed=not missing,
        severity="blocker",
        message="Expected GitHub Actions workflows are present." if not missing else "Expected GitHub Actions workflows are missing.",
        evidence={"missing": missing, "required_count": len(REQUIRED_WORKFLOWS)},
    )


def _check_frontend_scripts() -> ReadinessCheck:
    path = PROJECT_ROOT / "frontend" / "package.json"
    missing: list[str] = []
    scripts: dict[str, Any] = {}
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        scripts = raw.get("scripts", {}) if isinstance(raw, dict) else {}
        for script_name in ("dev", "build", "type-check"):
            if script_name not in scripts:
                missing.append(script_name)
    else:
        missing = ["frontend/package.json"]
    return ReadinessCheck(
        check_id="frontend_scripts",
        passed=not missing,
        severity="warning",
        message="Frontend exposes dev/build/type-check scripts." if not missing else "Frontend scripts are incomplete.",
        evidence={"missing": missing, "scripts": sorted(scripts.keys())},
    )


def _check_license_alignment() -> ReadinessCheck:
    readme = _read("README.md") if (PROJECT_ROOT / "README.md").exists() else ""
    license_text = _read("LICENSE") if (PROJECT_ROOT / "LICENSE").exists() else ""
    passed = "MIT License" in license_text and "MIT License" in readme
    return ReadinessCheck(
        check_id="license_alignment",
        passed=passed,
        severity="blocker",
        message="README license statement matches the repository LICENSE file." if passed else "README/license alignment is incomplete.",
        evidence={
            "readme_mentions_mit": "MIT License" in readme,
            "license_file_mentions_mit": "MIT License" in license_text,
        },
    )


def run_github_readiness() -> dict[str, Any]:
    """Run the GitHub readiness audit."""
    checks = [
        _check_required_public_files(),
        _check_evidence_files(),
        _check_readme_structure(),
        _check_positioning_disclosures(),
        _check_gitignore_hygiene(),
        _check_private_working_file_policy(),
        _check_secret_scan(),
        _check_visual_evidence(),
        _check_workflows(),
        _check_frontend_scripts(),
        _check_license_alignment(),
    ]
    failed_blockers = [
        check.check_id
        for check in checks
        if not check.passed and check.severity == "blocker"
    ]
    failed_warnings = [
        check.check_id
        for check in checks
        if not check.passed and check.severity == "warning"
    ]
    return {
        "project": "NeuroSight",
        "generated_at": utc_now(),
        "status": "passed" if not failed_blockers else "failed",
        "summary": {
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check.passed),
            "checks_failed": sum(1 for check in checks if not check.passed),
            "failed_blockers": failed_blockers,
            "failed_warnings": failed_warnings,
        },
        "checks": [asdict(check) for check in checks],
        "readiness_boundary": (
            "This audit checks GitHub portfolio hygiene, evidence discoverability, "
            "secret/artifact guardrails, and public positioning. It does not prove "
            "clinical validity, deployment uptime, or external dataset performance."
        ),
    }


def report_to_json(report: dict[str, Any]) -> str:
    """Serialize the report with stable formatting."""
    return json.dumps(report, indent=2, sort_keys=True)


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write the report to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_to_json(report) + "\n", encoding="utf-8")
    return path


def summarize_report(report: dict[str, Any], output_path: Path | None, *, stream: TextIO = sys.stdout) -> None:
    """Print a compact human-readable summary."""
    summary = report.get("summary", {})
    print("GITHUB READINESS CHECK COMPLETE", file=stream)
    print(f"Status: {report.get('status')}", file=stream)
    print(
        "Checks: "
        f"passed={summary.get('checks_passed', 0)} "
        f"failed={summary.get('checks_failed', 0)} "
        f"total={summary.get('checks_total', 0)}",
        file=stream,
    )
    print(f"Failed blockers: {', '.join(summary.get('failed_blockers', [])) or 'none'}", file=stream)
    print(f"Failed warnings: {', '.join(summary.get('failed_warnings', [])) or 'none'}", file=stream)
    if output_path is not None:
        print(f"Wrote: {output_path}", file=stream)


def main() -> int:
    args = parse_args()
    report = run_github_readiness()
    if args.stdout:
        print(report_to_json(report))
        summarize_report(report, None, stream=sys.stderr)
    else:
        output_path = write_report(report, args.out)
        summarize_report(report, output_path)

    failed_warnings = report.get("summary", {}).get("failed_warnings", [])
    should_fail_for_warnings = args.fail_warnings and bool(failed_warnings)
    should_fail_for_blockers = report.get("status") != "passed"
    return 1 if args.strict and (should_fail_for_blockers or should_fail_for_warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
