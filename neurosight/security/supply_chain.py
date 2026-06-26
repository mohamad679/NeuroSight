"""Static security and supply-chain audit helpers for NeuroSight.

The audit is intentionally dependency-light. It does not replace tools such as
pip-audit, npm audit, CodeQL, or Trivy; it gives the repository a runnable,
reviewable baseline that works in local and CI environments without network
access.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".deploy",
        ".next",
        ".venv",
        "__pycache__",
        "build",
        "checkpoints",
        "data",
        "dist",
        "logs",
        "node_modules",
        "venv",
    }
)
TEXT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".cfg",
        ".css",
        ".env",
        ".example",
        ".html",
        ".ini",
        ".js",
        ".json",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
    }
)
TEXT_NAMES: frozenset[str] = frozenset(
    {
        ".dockerignore",
        ".env",
        ".env.example",
        ".env.local",
        ".env.local.example",
        ".gitignore",
        "Dockerfile",
        "Makefile",
    }
)
SEVERITY_ORDER: dict[str, int] = {
    "info": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}
SECRET_ASSIGNMENT_RE = re.compile(
    r"""(?ix)
    \b(?P<name>[A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)\b
    \s*[:=]\s*
    (?P<value>["']?[^"'\s#]+["']?)
    """
)
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)
ENV_LIKE_SUFFIXES: frozenset[str] = frozenset({".env", ".yaml", ".yml"})
ENV_LIKE_NAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".env.example",
        ".env.local",
        ".env.local.example",
        "docker-compose.yml",
        "docker-compose.yaml",
    }
)
LOCAL_SECRET_PATHS: frozenset[str] = frozenset(
    {
        ".env",
        ".env.local",
        "frontend/.env",
        "frontend/.env.local",
    }
)


@dataclass(frozen=True)
class Finding:
    """One redacted audit finding."""

    severity: str
    category: str
    path: str
    message: str
    recommendation: str
    line: int | None = None
    fingerprint: str | None = None
    evidence: dict[str, Any] | None = None


def utc_now() -> str:
    """Return current UTC time as an ISO-8601 instant."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _is_excluded(path: Path, root: Path, excluded_dirs: frozenset[str] = DEFAULT_EXCLUDED_DIRS) -> bool:
    relative_parts = path.relative_to(root).parts if path.is_absolute() else path.parts
    return any(part in excluded_dirs for part in relative_parts)


def _is_text_candidate(path: Path) -> bool:
    return path.name in TEXT_NAMES or path.suffix.lower() in TEXT_SUFFIXES


def iter_repo_files(root: str | Path) -> list[Path]:
    """Return repository files, excluding generated/heavy directories."""
    root_path = Path(root).resolve()
    files: list[Path] = []
    import os
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Prune excluded directories in-place to prevent os.walk from descending into them
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDED_DIRS]
        for filename in filenames:
            file_path = Path(dirpath) / filename
            files.append(file_path)
    return sorted(files)


def read_text_file(path: Path, *, max_bytes: int = 1_000_000) -> str | None:
    """Read a likely text file safely."""
    if not _is_text_candidate(path):
        return None
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def _is_placeholder_secret(value: str) -> bool:
    normalized = value.strip().strip('"').strip("'").lower()
    if not normalized:
        return True
    placeholder_fragments = (
        "${",
        "<",
        "changeme",
        "change-me",
        "example",
        "placeholder",
        "secret_",
        "secrets.",
        "your_",
        "your-",
    )
    return any(fragment in normalized for fragment in placeholder_fragments)


def _is_assignment_secret_candidate(path: Path, value: str) -> bool:
    """Return whether a generic secret assignment looks like a real scalar secret."""
    raw = value.strip()
    normalized = raw.strip('"').strip("'")
    if _is_placeholder_secret(normalized) or len(normalized) < 8:
        return False
    lower = normalized.lower()
    if lower in {"false", "none", "null", "true", "undefined"}:
        return False
    if any(fragment in normalized for fragment in ("(", ")", "[", "]", "{", "}")):
        return False
    if normalized.startswith(("os.", "process.", "settings.", "self.", "nn.")):
        return False
    if path.name in ENV_LIKE_NAMES or path.suffix.lower() in ENV_LIKE_SUFFIXES:
        return True
    return raw.startswith(("'", '"')) and raw.endswith(("'", '"'))


def _secret_fingerprint(kind: str, value: str) -> str:
    digest = hashlib.sha256(f"{kind}:{value}".encode()).hexdigest()
    return digest[:16]


def scan_secrets(root: str | Path, *, include_local_secret_files: bool = False) -> list[Finding]:
    """Scan repository text files for secret-like values without exposing them."""
    root_path = Path(root)
    findings: list[Finding] = []
    for path in iter_repo_files(root_path):
        rel_path = _relative(path, root_path)
        if not include_local_secret_files and rel_path in LOCAL_SECRET_PATHS:
            continue
        text = read_text_file(path)
        if text is None:
            continue
        is_example = "example" in path.name.lower()
        for line_number, line in enumerate(text.splitlines(), start=1):
            for kind, pattern in SECRET_PATTERNS:
                for match in pattern.finditer(line):
                    value = match.group(0)
                    if _is_placeholder_secret(value):
                        continue
                    findings.append(
                        Finding(
                            severity="critical" if not is_example else "low",
                            category="secret_scan",
                            path=rel_path,
                            line=line_number,
                            message=f"Potential {kind} value detected. The value is redacted.",
                            recommendation="Rotate the secret if real, remove it from Git, and keep only an example placeholder.",
                            fingerprint=_secret_fingerprint(kind, value),
                        )
                    )
            assignment = SECRET_ASSIGNMENT_RE.search(line)
            if assignment:
                name = assignment.group("name")
                value = assignment.group("value").strip().strip('"').strip("'")
                if not _is_assignment_secret_candidate(path, value):
                    continue
                findings.append(
                    Finding(
                        severity="critical" if not is_example else "low",
                        category="secret_scan",
                        path=rel_path,
                        line=line_number,
                        message=f"Potential secret assignment for `{name}` detected. The value is redacted.",
                        recommendation="Move this value to a secret manager or local ignored env file, then rotate it if it was exposed.",
                        fingerprint=_secret_fingerprint(name, value),
                    )
                )
    return dedupe_findings(findings)


def dedupe_findings(findings: list[Finding]) -> list[Finding]:
    """Deduplicate findings by category/path/line/fingerprint."""
    seen: set[tuple[str, str, int | None, str | None]] = set()
    unique: list[Finding] = []
    for finding in findings:
        key = (finding.category, finding.path, finding.line, finding.fingerprint)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique


def parse_requirement_file(path: Path) -> list[dict[str, Any]]:
    """Parse a pip requirements file into a lightweight inventory."""
    if not path.exists():
        return []
    packages: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        cleaned = line.split("#", maxsplit=1)[0].strip()
        match = re.match(r"(?P<name>[A-Za-z0-9_.-]+)\s*(?P<specifier>.*)", cleaned)
        if not match:
            continue
        specifier = match.group("specifier").strip()
        packages.append(
            {
                "name": match.group("name").lower(),
                "specifier": specifier,
                "line": line_number,
                "pinned": "==" in specifier or "===" in specifier,
            }
        )
    return packages


def parse_pyproject_dependencies(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Parse Poetry dependency groups from pyproject.toml."""
    if not path.exists():
        return {}
    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    poetry = parsed.get("tool", {}).get("poetry", {})
    groups: dict[str, list[dict[str, Any]]] = {}
    main_deps = poetry.get("dependencies", {})
    if isinstance(main_deps, dict):
        groups["main"] = _poetry_dependency_rows(main_deps)
    group_deps = poetry.get("group", {})
    if isinstance(group_deps, dict):
        for group_name, group_value in group_deps.items():
            dependencies = group_value.get("dependencies", {}) if isinstance(group_value, dict) else {}
            if isinstance(dependencies, dict):
                groups[str(group_name)] = _poetry_dependency_rows(dependencies)
    return groups


def _poetry_dependency_rows(dependencies: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, specifier in sorted(dependencies.items()):
        spec_text = json.dumps(specifier, sort_keys=True) if isinstance(specifier, dict) else str(specifier)
        rows.append(
            {
                "name": str(name).lower(),
                "specifier": spec_text,
                "pinned": spec_text.startswith("==") or spec_text.count(".") >= 2 and spec_text[0].isdigit(),
            }
        )
    return rows


def parse_package_json(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Parse npm dependency groups from package.json."""
    if not path.exists():
        return {}
    parsed = json.loads(path.read_text(encoding="utf-8"))
    groups: dict[str, list[dict[str, Any]]] = {}
    for group_name in ("dependencies", "devDependencies"):
        deps = parsed.get(group_name, {})
        if not isinstance(deps, dict):
            continue
        groups[group_name] = [
            {
                "name": str(name),
                "specifier": str(specifier),
                "pinned": str(specifier)[0].isdigit(),
            }
            for name, specifier in sorted(deps.items())
        ]
    return groups


def dependency_inventory(root: str | Path) -> dict[str, Any]:
    """Collect dependency manifests, hashes, package counts, and lock status."""
    root_path = Path(root)
    manifest_paths = [
        root_path / "pyproject.toml",
        root_path / "requirements.txt",
        root_path / "requirements-dev.txt",
        root_path / "requirements.lock",
        root_path / "requirements_ui.txt",
        root_path / "hf_space" / "requirements.txt",
        root_path / "hf_space" / "requirements_backend.txt",
        root_path / "frontend" / "package.json",
        root_path / "frontend" / "package-lock.json",
    ]
    manifests = [
        {
            "path": _relative(path, root_path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        }
        for path in manifest_paths
    ]
    python_requirement_files = {
        _relative(path, root_path): parse_requirement_file(path)
        for path in manifest_paths
        if path.name.startswith("requirements") and path.exists()
    }
    pyproject_groups = parse_pyproject_dependencies(root_path / "pyproject.toml")
    npm_groups = parse_package_json(root_path / "frontend" / "package.json")
    lockfiles = {
        "python_canonical": "pip_requirements_lock",
        "requirements_lock": (root_path / "requirements.lock").exists(),
        "poetry_lock": (root_path / "poetry.lock").exists(),
        "frontend_package_lock": (root_path / "frontend" / "package-lock.json").exists(),
        "npm_shrinkwrap": (root_path / "frontend" / "npm-shrinkwrap.json").exists(),
    }
    return {
        "manifests": manifests,
        "lockfiles": lockfiles,
        "python": {
            "pyproject_groups": _summarize_dependency_groups(pyproject_groups),
            "requirements_files": {
                path: _summarize_package_rows(rows)
                for path, rows in python_requirement_files.items()
            },
        },
        "npm": {
            "package_json_groups": _summarize_dependency_groups(npm_groups),
        },
    }


def _summarize_dependency_groups(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        group_name: _summarize_package_rows(rows)
        for group_name, rows in groups.items()
    }


def _summarize_package_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unpinned = [row for row in rows if not row.get("pinned")]
    return {
        "count": len(rows),
        "pinned_count": len(rows) - len(unpinned),
        "unpinned_count": len(unpinned),
        "unpinned_examples": [row["name"] for row in unpinned[:8]],
    }


def dependency_findings(inventory: dict[str, Any]) -> list[Finding]:
    """Generate dependency hygiene findings from the inventory."""
    findings: list[Finding] = []
    lockfiles = inventory.get("lockfiles", {})
    python_has_lock = bool(lockfiles.get("requirements_lock") or lockfiles.get("poetry_lock"))
    if not python_has_lock:
        findings.append(
            Finding(
                severity="high",
                category="dependency_locking",
                path="requirements.txt",
                message="Python dependencies are declared, but no canonical Python lockfile is present.",
                recommendation="Commit requirements.lock for pip installs or poetry.lock if Poetry becomes canonical.",
            )
        )
    if not lockfiles.get("frontend_package_lock"):
        findings.append(
            Finding(
                severity="high",
                category="dependency_locking",
                path="frontend/package.json",
                message="Frontend dependencies are declared, but package-lock.json is missing.",
                recommendation="Run npm install in frontend/ and commit package-lock.json.",
            )
        )

    python_files = inventory.get("python", {}).get("requirements_files", {})
    for file_path, summary in python_files.items():
        if summary.get("unpinned_count", 0) > 0:
            findings.append(
                Finding(
                    severity="medium",
                    category="dependency_pinning",
                    path=file_path,
                    message=f"{summary['unpinned_count']} requirement entries are range-pinned or unpinned.",
                    recommendation="Prefer exact pins for deploy images and use scheduled dependency updates.",
                    evidence={"examples": summary.get("unpinned_examples", [])},
                )
            )

    npm_groups = inventory.get("npm", {}).get("package_json_groups", {})
    for group_name, summary in npm_groups.items():
        if summary.get("unpinned_count", 0) > 0:
            findings.append(
                Finding(
                    severity="medium",
                    category="dependency_pinning",
                    path="frontend/package.json",
                    message=f"{summary['unpinned_count']} npm {group_name} entries are range-pinned or unpinned.",
                    recommendation="Prefer exact pins or rely on package-lock.json plus Dependabot review.",
                    evidence={"examples": summary.get("unpinned_examples", [])},
                )
            )
    return findings


def workflow_findings(root: str | Path) -> list[Finding]:
    """Audit GitHub Actions workflow hygiene."""
    root_path = Path(root)
    workflow_dir = root_path / ".github" / "workflows"
    findings: list[Finding] = []
    if not workflow_dir.exists():
        return findings
    for path in sorted(workflow_dir.glob("*.y*ml")):
        text = path.read_text(encoding="utf-8")
        rel_path = _relative(path, root_path)
        if "permissions:" not in text:
            findings.append(
                Finding(
                    severity="medium",
                    category="github_actions",
                    path=rel_path,
                    message="Workflow does not declare minimal permissions.",
                    recommendation="Add top-level `permissions: contents: read` unless the workflow needs broader access.",
                )
            )
        if "actions/checkout" in text and "persist-credentials: false" not in text:
            findings.append(
                Finding(
                    severity="low",
                    category="github_actions",
                    path=rel_path,
                    message="actions/checkout does not disable persisted credentials.",
                    recommendation="Set `persist-credentials: false` for jobs that do not need GitHub token credentials after checkout.",
                )
            )
        unpinned_actions = sorted(set(re.findall(r"uses:\s*([^@\s]+@[vV][0-9][^\s]*)", text)))
        if unpinned_actions:
            findings.append(
                Finding(
                    severity="low",
                    category="github_actions",
                    path=rel_path,
                    message="Workflow actions are pinned by tag rather than immutable SHA.",
                    recommendation="For high-assurance releases, pin third-party actions to commit SHA and update via Dependabot.",
                    evidence={"actions": unpinned_actions[:8]},
                )
            )
    return findings


def docker_findings(root: str | Path) -> list[Finding]:
    """Audit Dockerfile hygiene."""
    root_path = Path(root)
    findings: list[Finding] = []
    for path in sorted(root_path.rglob("Dockerfile")):
        if _is_excluded(path, root_path):
            continue
        text = path.read_text(encoding="utf-8")
        rel_path = _relative(path, root_path)
        from_lines = [line.strip() for line in text.splitlines() if line.strip().upper().startswith("FROM ")]
        digestless = [line for line in from_lines if "@sha256:" not in line]
        if digestless:
            findings.append(
                Finding(
                    severity="low",
                    category="container_supply_chain",
                    path=rel_path,
                    message="Base image is not digest-pinned.",
                    recommendation="Pin production container base images by digest after selecting an approved image.",
                    evidence={"from": digestless},
                )
            )
        if "apt-get install -y" in text and "--no-install-recommends" not in text:
            findings.append(
                Finding(
                    severity="low",
                    category="container_hardening",
                    path=rel_path,
                    message="apt-get install does not use --no-install-recommends.",
                    recommendation="Use --no-install-recommends to reduce image size and attack surface.",
                )
            )
        if not re.search(r"(?m)^\s*USER\s+\S+", text):
            findings.append(
                Finding(
                    severity="medium",
                    category="container_hardening",
                    path=rel_path,
                    message="Container image does not switch to a non-root user.",
                    recommendation="Create and use a non-root runtime user for production images.",
                )
            )
    return findings


def severity_counts(findings: list[Finding]) -> dict[str, int]:
    """Count findings by severity."""
    counts = {severity: 0 for severity in SEVERITY_ORDER}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    return counts


def overall_status(findings: list[Finding]) -> str:
    """Return ok, warning, high_risk, or critical."""
    counts = severity_counts(findings)
    if counts.get("critical", 0) > 0:
        return "critical"
    if counts.get("high", 0) > 0:
        return "high_risk"
    if counts.get("medium", 0) > 0 or counts.get("low", 0) > 0:
        return "warning"
    return "ok"


def recommended_actions(findings: list[Finding]) -> list[str]:
    """Return prioritized remediation guidance."""
    categories = {finding.category for finding in findings}
    actions: list[str] = []
    if "secret_scan" in categories:
        actions.append("Rotate any real secret-like values, remove them from repository history, and keep only placeholders in examples.")
    if "dependency_locking" in categories:
        actions.append("Commit deterministic lockfiles for package ecosystems used by deployable applications.")
    if "dependency_pinning" in categories:
        actions.append("Prefer exact pins for deploy images and let Dependabot open reviewed update pull requests.")
    if "github_actions" in categories:
        actions.append("Use minimal GitHub Actions permissions and avoid persisted checkout credentials unless required.")
    if "container_hardening" in categories or "container_supply_chain" in categories:
        actions.append("Harden production containers with non-root users, smaller apt installs, and digest-pinned base images.")
    if not actions:
        actions.append("No supply-chain hygiene issues detected by this local static audit.")
    return actions


def build_supply_chain_report(
    root: str | Path = ".",
    *,
    include_local_secret_files: bool = False,
) -> dict[str, Any]:
    """Build a JSON-safe supply-chain and security hygiene report."""
    root_path = Path(root).resolve()
    inventory = dependency_inventory(root_path)
    findings = (
        scan_secrets(root_path, include_local_secret_files=include_local_secret_files)
        + dependency_findings(inventory)
        + workflow_findings(root_path)
        + docker_findings(root_path)
    )
    findings = sorted(
        findings,
        key=lambda item: (-SEVERITY_ORDER.get(item.severity, 0), item.category, item.path, item.line or 0),
    )
    return {
        "project": "NeuroSight",
        "generated_at": utc_now(),
        "task": "security_supply_chain_audit",
        "status": overall_status(findings),
        "repository_root": str(root_path),
        "dependency_inventory": inventory,
        "secret_scan": {
            "redacted": True,
            "excluded_dirs": sorted(DEFAULT_EXCLUDED_DIRS),
            "include_local_secret_files": include_local_secret_files,
            "skipped_local_secret_files": []
            if include_local_secret_files
            else sorted(
                path
                for path in LOCAL_SECRET_PATHS
                if (root_path / path).exists()
            ),
            "finding_count": len([finding for finding in findings if finding.category == "secret_scan"]),
        },
        "summary": {
            "finding_count": len(findings),
            "severity_counts": severity_counts(findings),
            "categories": sorted({finding.category for finding in findings}),
        },
        "findings": [asdict(finding) for finding in findings],
        "recommended_actions": recommended_actions(findings),
        "audit_boundary": (
            "This static audit checks repository hygiene and redacted secret patterns. "
            "It does not query vulnerability databases or prove clinical/security compliance."
        ),
    }


def report_to_json(report: dict[str, Any]) -> str:
    """Serialize the supply-chain report with stable formatting."""
    return json.dumps(report, indent=2, sort_keys=True)


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a supply-chain report to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_to_json(report) + "\n", encoding="utf-8")
    return path
