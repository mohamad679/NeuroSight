#!/usr/bin/env python3
"""Run in-process FastAPI contract checks for NeuroSight."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TextIO

from fastapi.testclient import TestClient

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DISABLE_MRI_WARMUP", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app  # noqa: E402

DEFAULT_OUTPUT = "logs/api_contract/neurosight_api_contract_report.json"
CheckStatus = Literal["passed", "failed"]
ResponseKind = Literal["dict", "list", "sse"]


@dataclass(frozen=True)
class ContractCase:
    """One API contract case."""

    name: str
    method: str
    path: str
    expected_status: int
    response_kind: ResponseKind
    json_body: dict[str, object] | None = None
    params: dict[str, object] | None = None
    required_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContractResult:
    """Result for one API contract case."""

    name: str
    status: CheckStatus
    method: str
    path: str
    expected_status: int
    actual_status: int | None
    duration_seconds: float
    issues: list[str]
    observed_keys: list[str]
    response_excerpt: str


def utc_now() -> str:
    """Return current UTC time as an ISO-8601 instant."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight FastAPI contract checks without starting a server."
    )
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help=f"Output JSON path. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--stdout", action="store_true", help="Print JSON to stdout instead of writing a file.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any contract check fails.")
    return parser.parse_args()


def _excerpt(text: str, *, max_chars: int = 700) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "..."


def _required_key_issues(payload: object, required_keys: tuple[str, ...]) -> list[str]:
    if not required_keys:
        return []
    if not isinstance(payload, dict):
        return ["required-key validation expected a JSON object"]
    missing = [key for key in required_keys if key not in payload]
    return [f"missing required key: {key}" for key in missing]


def _json_payload(response_text: str) -> tuple[object | None, list[str]]:
    try:
        return json.loads(response_text), []
    except json.JSONDecodeError as exc:
        return None, [f"response is not valid JSON: {exc}"]


def _run_json_case(client: TestClient, case: ContractCase) -> ContractResult:
    started = time.perf_counter()
    issues: list[str] = []
    observed_keys: list[str] = []
    response = client.request(
        case.method,
        case.path,
        json=case.json_body,
        params=case.params,
        headers={"X-Request-ID": f"contract-{case.name}"},
    )
    duration = round(time.perf_counter() - started, 3)

    if response.status_code != case.expected_status:
        issues.append(f"expected status {case.expected_status}, got {response.status_code}")
    if not response.headers.get("X-Request-ID"):
        issues.append("missing X-Request-ID response header")
    if not response.headers.get("X-Process-Time"):
        issues.append("missing X-Process-Time response header")

    payload, json_issues = _json_payload(response.text)
    issues.extend(json_issues)
    if payload is not None:
        if case.response_kind == "dict" and not isinstance(payload, dict):
            issues.append("expected JSON object response")
        if case.response_kind == "list" and not isinstance(payload, list):
            issues.append("expected JSON list response")
        if isinstance(payload, dict):
            observed_keys = sorted(str(key) for key in payload.keys())
        issues.extend(_required_key_issues(payload, case.required_keys))

    return ContractResult(
        name=case.name,
        status="passed" if not issues else "failed",
        method=case.method,
        path=case.path,
        expected_status=case.expected_status,
        actual_status=response.status_code,
        duration_seconds=duration,
        issues=issues,
        observed_keys=observed_keys,
        response_excerpt=_excerpt(response.text),
    )


def _parse_sse_events(response_text: str) -> tuple[list[dict[str, object]], bool, list[str]]:
    events: list[dict[str, object]] = []
    saw_done = False
    issues: list[str] = []
    for line in response_text.splitlines():
        if not line.startswith("data: "):
            continue
        raw = line[6:].strip()
        if raw == "[DONE]":
            saw_done = True
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            issues.append(f"invalid SSE JSON event: {exc}")
            continue
        if isinstance(event, dict):
            events.append(event)
        else:
            issues.append("SSE event was not a JSON object")
    return events, saw_done, issues


def _run_sse_case(client: TestClient, case: ContractCase) -> ContractResult:
    started = time.perf_counter()
    response = client.request(
        case.method,
        case.path,
        json=case.json_body,
        params=case.params,
        headers={"X-Request-ID": f"contract-{case.name}"},
    )
    duration = round(time.perf_counter() - started, 3)
    issues: list[str] = []

    if response.status_code != case.expected_status:
        issues.append(f"expected status {case.expected_status}, got {response.status_code}")

    events, saw_done, sse_issues = _parse_sse_events(response.text)
    issues.extend(sse_issues)
    if not saw_done:
        issues.append("SSE stream did not emit [DONE]")
    if not events:
        issues.append("SSE stream emitted no JSON events")

    final_events = [
        event
        for event in events
        if event.get("agent") == "complete" and event.get("status") == "done"
    ]
    if not final_events:
        issues.append("SSE stream did not emit final complete/done event")
    else:
        final_event = final_events[-1]
        issues.extend(_required_key_issues(final_event, case.required_keys))

    agent_names = sorted(
        {
            str(event.get("agent"))
            for event in events
            if event.get("agent") is not None
        }
    )

    return ContractResult(
        name=case.name,
        status="passed" if not issues else "failed",
        method=case.method,
        path=case.path,
        expected_status=case.expected_status,
        actual_status=response.status_code,
        duration_seconds=duration,
        issues=issues,
        observed_keys=agent_names,
        response_excerpt=_excerpt(response.text),
    )


def _run_auth_contract(client: TestClient) -> ContractResult:
    started = time.perf_counter()
    previous_env = os.environ.get("APP_ENV")
    previous_key = os.environ.get("NEUROSIGHT_API_KEY")
    issues: list[str] = []
    try:
        os.environ["APP_ENV"] = "production"
        os.environ["NEUROSIGHT_API_KEY"] = "contract-test-key"
        missing_key = client.get("/v1/data/status")
        with_key = client.get(
            "/v1/data/status",
            headers={"X-API-Key": "contract-test-key"},
        )
    finally:
        if previous_env is None:
            os.environ.pop("APP_ENV", None)
        else:
            os.environ["APP_ENV"] = previous_env
        if previous_key is None:
            os.environ.pop("NEUROSIGHT_API_KEY", None)
        else:
            os.environ["NEUROSIGHT_API_KEY"] = previous_key

    duration = round(time.perf_counter() - started, 3)
    if missing_key.status_code != 401:
        issues.append(f"missing key should return 401, got {missing_key.status_code}")
    if with_key.status_code != 200:
        issues.append(f"configured key should return 200, got {with_key.status_code}")

    return ContractResult(
        name="api_key_enforcement",
        status="passed" if not issues else "failed",
        method="GET",
        path="/v1/data/status",
        expected_status=200,
        actual_status=with_key.status_code,
        duration_seconds=duration,
        issues=issues,
        observed_keys=["missing_key_status", "with_key_status"],
        response_excerpt=(
            f"missing_key={missing_key.status_code}; with_key={with_key.status_code}"
        ),
    )


def contract_cases() -> list[ContractCase]:
    """Return the lightweight backend contract cases."""
    cognitive_scores = {
        "mmse": 24.0,
        "moca": 20.0,
        "cdrsb": 4.5,
        "adas11": 18.0,
        "ravlt_immediate": 28.0,
        "ravlt_learning": 2.0,
        "faq": 6.0,
        "age": 70.0,
    }
    return [
        ContractCase("root", "GET", "/", 200, "dict", required_keys=("name", "status", "capabilities")),
        ContractCase("healthz", "GET", "/healthz", 200, "dict", required_keys=("status", "version", "runtime", "capabilities")),
        ContractCase("data_status", "GET", "/v1/data/status", 200, "dict", required_keys=("status", "privacy", "recommended_patient_id")),
        ContractCase("demo_patients", "GET", "/v1/data/demo-patients", 200, "dict", params={"limit": 3}, required_keys=("patients", "count", "privacy")),
        ContractCase("modalities_status", "GET", "/v1/modalities/status", 200, "dict", required_keys=("mri", "eeg", "cognitive")),
        ContractCase("governance_status", "GET", "/v1/governance/status", 200, "dict", required_keys=("privacy", "security", "scientific_disclosure")),
        ContractCase("demo_readiness", "GET", "/v1/demo/readiness", 200, "dict", required_keys=("status", "checks", "recommended_ui_flow")),
        ContractCase("upload_cognitive", "POST", "/v1/upload/cognitive", 200, "dict", json_body={"scores": cognitive_scores}, required_keys=("status", "embedding_dim", "embedding", "unimodal_probs")),
        ContractCase("risk_profile_cognitive", "POST", "/v1/risk-profile", 200, "dict", json_body={"patient_id": "CONTRACT_DEMO_001", "query": "Summarize non-clinical risk profile.", "cognitive_scores": cognitive_scores}, required_keys=("diagnosis", "confidence", "requires_review", "report_text")),
        ContractCase("legacy_diagnose_cognitive", "POST", "/v1/diagnose", 200, "dict", json_body={"patient_id": "CONTRACT_DEMO_001", "query": "Summarize non-clinical risk profile.", "cognitive_scores": cognitive_scores}, required_keys=("diagnosis", "confidence", "requires_review", "report_text")),
        ContractCase("risk_profile_stream", "POST", "/v1/risk-profile/stream", 200, "sse", json_body={"patient_id": "CONTRACT_STREAM_001", "query": "Summarize non-clinical risk profile.", "cognitive_scores": cognitive_scores}, required_keys=("diagnosis", "confidence", "requires_review", "report_text")),
        ContractCase("legacy_diagnose_stream", "POST", "/v1/diagnose/stream", 200, "sse", json_body={"patient_id": "CONTRACT_STREAM_001", "query": "Summarize non-clinical risk profile.", "cognitive_scores": cognitive_scores}, required_keys=("diagnosis", "confidence", "requires_review", "report_text")),
        ContractCase("kg_query", "POST", "/v1/kg/query", 200, "dict", json_body={"patient_id": "SYN_0001", "query_type": "history"}, required_keys=("patient_id", "query_type", "results", "count")),
        ContractCase("kg_history", "GET", "/v1/kg/patient/SYN_0001/history", 200, "dict", required_keys=("patient_id", "history", "count")),
        ContractCase("kg_similar", "GET", "/v1/kg/patient/SYN_0001/similar", 200, "list"),
        ContractCase("eval_metrics", "GET", "/v1/eval/metrics", 200, "dict"),
        ContractCase("eval_history", "GET", "/v1/eval/history", 200, "list"),
        ContractCase("eval_report", "GET", "/v1/eval/report", 200, "dict", required_keys=("checkpoint", "evaluation", "model_card", "scientific_claims")),
        ContractCase("models", "GET", "/v1/models", 200, "list"),
        ContractCase("production_model", "GET", "/v1/models/production", 200, "dict"),
        ContractCase("checkpoint_status", "GET", "/v1/models/checkpoint/status", 200, "dict", required_keys=("checkpoint", "loading", "evaluation", "model_card")),
        ContractCase("xai_status", "GET", "/v1/xai/status", 200, "dict", required_keys=("methods", "interpretation_policy", "endpoint")),
        ContractCase("xai_cognitive", "GET", "/v1/xai/CONTRACT_DEMO_001", 200, "dict", params={"modality": "cognitive"}, required_keys=("patient_id", "modality", "feature_importance", "xai_available", "method_contract")),
    ]


def run_contract_checks() -> dict[str, object]:
    """Run all contract checks and return a JSON-safe report."""
    client = TestClient(app)
    results: list[ContractResult] = []
    for case in contract_cases():
        if case.response_kind == "sse":
            results.append(_run_sse_case(client, case))
        else:
            results.append(_run_json_case(client, case))
    results.append(_run_auth_contract(client))

    failed = [result.name for result in results if result.status == "failed"]
    return {
        "project": "NeuroSight",
        "generated_at": utc_now(),
        "status": "passed" if not failed else "failed",
        "summary": {
            "checks_total": len(results),
            "checks_passed": sum(1 for result in results if result.status == "passed"),
            "checks_failed": len(failed),
            "failed": failed,
        },
        "checks": [asdict(result) for result in results],
        "contract_boundary": (
            "This script exercises the FastAPI app in-process with APP_ENV=test. "
            "It proves route contracts, response shapes, streaming events, middleware "
            "headers, and API-key behavior. It does not prove clinical validity, real "
            "deployment health, or expensive MRI/EEG inference performance."
        ),
    }


def report_to_json(report: dict[str, object]) -> str:
    """Serialize the report with stable formatting."""
    return json.dumps(report, indent=2, sort_keys=True)


def write_report(report: dict[str, object], output_path: str | Path) -> Path:
    """Write the report to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_to_json(report) + "\n", encoding="utf-8")
    return path


def summarize_report(report: dict[str, object], output_path: Path | None, *, stream: TextIO = sys.stdout) -> None:
    """Print a compact human-readable summary."""
    summary = report.get("summary", {})
    print("API CONTRACT CHECK COMPLETE", file=stream)
    print(f"Status: {report.get('status')}", file=stream)
    print(
        "Checks: "
        f"passed={summary.get('checks_passed', 0)} "
        f"failed={summary.get('checks_failed', 0)} "
        f"total={summary.get('checks_total', 0)}",
        file=stream,
    )
    print(f"Failed: {', '.join(summary.get('failed', [])) or 'none'}", file=stream)
    if output_path is not None:
        print(f"Wrote: {output_path}", file=stream)


def main() -> int:
    args = parse_args()
    report = run_contract_checks()
    if args.stdout:
        print(report_to_json(report))
        summarize_report(report, None, stream=sys.stderr)
    else:
        output_path = write_report(report, args.out)
        summarize_report(report, output_path)
    return 1 if args.strict and report.get("status") != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
