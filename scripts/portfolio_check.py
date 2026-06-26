#!/usr/bin/env python3
"""Run NeuroSight's reviewer-facing portfolio proof checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "logs/portfolio/neurosight_portfolio_check_report.json"
DEFAULT_BACKEND_URL = "http://localhost:8000"
BackendSmokeMode = Literal["auto", "skip", "required"]
CheckStatus = Literal["passed", "failed", "skipped"]


@dataclass(frozen=True)
class PortfolioCheckResult:
    """One portfolio proof check result."""

    name: str
    status: CheckStatus
    required: bool
    command: list[str]
    duration_seconds: float
    returncode: int | None
    stdout_tail: str
    stderr_tail: str
    artifact: str | None = None
    reason: str | None = None


def utc_now() -> str:
    """Return current UTC time as an ISO-8601 instant."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NeuroSight portfolio proof path for reviewers."
    )
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help=f"Output JSON path. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--stdout", action="store_true", help="Print JSON to stdout instead of writing a file.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if required checks fail.")
    parser.add_argument(
        "--backend-smoke",
        choices=["auto", "skip", "required"],
        default="auto",
        help="Run backend smoke test when available, skip it, or require it.",
    )
    parser.add_argument(
        "--backend-base-url",
        default=DEFAULT_BACKEND_URL,
        help=f"Backend base URL for smoke test. Default: {DEFAULT_BACKEND_URL}",
    )
    parser.add_argument("--api-key", default=None, help="Optional API key forwarded to smoke_backend.py.")
    return parser.parse_args()


def _tail(text: str, *, max_chars: int = 1800) -> str:
    if len(text) <= max_chars:
        return text.strip()
    return text[-max_chars:].strip()


def run_command(
    name: str,
    command: list[str],
    *,
    required: bool,
    artifact: str | None = None,
    timeout: int = 180,
) -> PortfolioCheckResult:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        duration = round(time.perf_counter() - started, 3)
        return PortfolioCheckResult(
            name=name,
            status="passed" if completed.returncode == 0 else "failed",
            required=required,
            command=command,
            duration_seconds=duration,
            returncode=completed.returncode,
            stdout_tail=_tail(completed.stdout),
            stderr_tail=_tail(completed.stderr),
            artifact=artifact,
        )
    except subprocess.TimeoutExpired as exc:
        duration = round(time.perf_counter() - started, 3)
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return PortfolioCheckResult(
            name=name,
            status="failed",
            required=required,
            command=command,
            duration_seconds=duration,
            returncode=None,
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr),
            artifact=artifact,
            reason=f"Command timed out after {timeout} seconds.",
        )


def skipped_check(name: str, reason: str, command: list[str], *, required: bool) -> PortfolioCheckResult:
    return PortfolioCheckResult(
        name=name,
        status="skipped",
        required=required,
        command=command,
        duration_seconds=0.0,
        returncode=None,
        stdout_tail="",
        stderr_tail="",
        reason=reason,
    )


def backend_is_available(base_url: str, *, timeout: float = 0.75) -> bool:
    url = base_url.rstrip("/") + "/healthz"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= int(response.status) < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def build_backend_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/smoke_backend.py",
        "--base-url",
        str(args.backend_base_url),
        "--json",
    ]
    if args.api_key:
        command.extend(["--api-key", str(args.api_key)])
    return command


def run_portfolio_checks(args: argparse.Namespace) -> dict[str, object]:
    portfolio_dir = Path("logs/portfolio")
    checks: list[PortfolioCheckResult] = []
    python = sys.executable

    mandatory_commands: list[tuple[str, list[str], str]] = [
        (
            "quality_gate",
            [python, "scripts/quality_gate.py", "--strict", "--out", str(portfolio_dir / "quality_gate_report.json")],
            str(portfolio_dir / "quality_gate_report.json"),
        ),
        (
            "api_contract_check",
            [python, "scripts/api_contract_check.py", "--strict", "--out", str(portfolio_dir / "api_contract_report.json")],
            str(portfolio_dir / "api_contract_report.json"),
        ),
        (
            "github_readiness",
            [python, "scripts/github_readiness.py", "--strict", "--out", str(portfolio_dir / "github_readiness_report.json")],
            str(portfolio_dir / "github_readiness_report.json"),
        ),
        (
            "model_card_check",
            [python, "scripts/model_card_check.py", "--strict", "--out", str(portfolio_dir / "model_card_report.json")],
            str(portfolio_dir / "model_card_report.json"),
        ),
        (
            "ai_safety_eval",
            [python, "scripts/ai_safety_eval.py", "--strict", "--out", str(portfolio_dir / "ai_safety_report.json")],
            str(portfolio_dir / "ai_safety_report.json"),
        ),
        (
            "langgraph_normal_trace",
            [python, "scripts/langgraph_workflow.py", "--scenario", "normal", "--out", str(portfolio_dir / "langgraph_normal_trace.json")],
            str(portfolio_dir / "langgraph_normal_trace.json"),
        ),
        (
            "langgraph_blocked_trace",
            [python, "scripts/langgraph_workflow.py", "--scenario", "blocked", "--out", str(portfolio_dir / "langgraph_blocked_trace.json")],
            str(portfolio_dir / "langgraph_blocked_trace.json"),
        ),
    ]

    for name, command, artifact in mandatory_commands:
        checks.append(run_command(name, command, required=True, artifact=artifact))

    backend_command = build_backend_command(args)
    if args.backend_smoke == "skip":
        checks.append(skipped_check("backend_smoke", "Skipped by --backend-smoke skip.", backend_command, required=False))
    elif args.backend_smoke == "auto" and not backend_is_available(args.backend_base_url):
        checks.append(
            skipped_check(
                "backend_smoke",
                f"No backend detected at {args.backend_base_url}; start FastAPI and rerun with --backend-smoke required for live API proof.",
                backend_command,
                required=False,
            )
        )
    else:
        checks.append(run_command("backend_smoke", backend_command, required=args.backend_smoke == "required", artifact=None, timeout=240))

    failed_required = [check.name for check in checks if check.required and check.status == "failed"]
    failed_optional = [check.name for check in checks if not check.required and check.status == "failed"]
    skipped = [check.name for check in checks if check.status == "skipped"]
    status = "passed" if not failed_required else "failed"

    return {
        "project": "NeuroSight",
        "generated_at": utc_now(),
        "status": status,
        "summary": {
            "checks_total": len(checks),
            "checks_passed": sum(1 for check in checks if check.status == "passed"),
            "checks_failed": sum(1 for check in checks if check.status == "failed"),
            "checks_skipped": len(skipped),
            "failed_required": failed_required,
            "failed_optional": failed_optional,
            "skipped": skipped,
        },
        "checks": [asdict(check) for check in checks],
        "reviewer_boundary": (
            "This portfolio proof checks repository evidence and offline workflows. "
            "It does not validate clinical performance, real-patient behavior, or production deployment."
        ),
    }


def report_to_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


def write_report(report: dict[str, object], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_to_json(report) + "\n", encoding="utf-8")
    return path


def summarize_report(report: dict[str, object], output_path: Path | None, *, stream: TextIO = sys.stdout) -> None:
    summary = report.get("summary", {})
    print("PORTFOLIO CHECK COMPLETE", file=stream)
    print(f"Status: {report.get('status')}", file=stream)
    print(
        "Checks: "
        f"passed={summary.get('checks_passed', 0)} "
        f"failed={summary.get('checks_failed', 0)} "
        f"skipped={summary.get('checks_skipped', 0)} "
        f"total={summary.get('checks_total', 0)}",
        file=stream,
    )
    print(f"Failed required: {', '.join(summary.get('failed_required', [])) or 'none'}", file=stream)
    print(f"Failed optional: {', '.join(summary.get('failed_optional', [])) or 'none'}", file=stream)
    print(f"Skipped: {', '.join(summary.get('skipped', [])) or 'none'}", file=stream)
    if output_path is not None:
        print(f"Wrote: {output_path}", file=stream)


def main() -> int:
    args = parse_args()
    report = run_portfolio_checks(args)
    if args.stdout:
        print(report_to_json(report))
        summarize_report(report, None, stream=sys.stderr)
    else:
        output_path = write_report(report, args.out)
        summarize_report(report, output_path)
    return 1 if args.strict and report.get("status") != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
