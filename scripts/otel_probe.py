#!/usr/bin/env python3
"""Probe NeuroSight request observability.

This script is intentionally small and reviewer-friendly. It checks that the
FastAPI app returns request correlation headers and, when OpenTelemetry is
enabled, exposes a trace ID plus observability status in `/healthz`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_API_KEY = "dev-key"
OBSERVABILITY_HEADERS = (
    "X-Request-ID",
    "X-Trace-ID",
    "X-Process-Time",
    "X-Observability",
)


class ProbeFailure(Exception):
    """Raised when a probe request fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check NeuroSight request IDs, trace IDs, and OTel health metadata."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("NEUROSIGHT_BACKEND_URL", DEFAULT_BASE_URL),
        help=f"FastAPI base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NEUROSIGHT_API_KEY", DEFAULT_API_KEY),
        help="API key for protected /v1 routes. Default: dev-key",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--skip-protected",
        action="store_true",
        help="Only call /healthz and skip the protected diagnosis probe.",
    )
    return parser.parse_args()


def decode_json(response: requests.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ProbeFailure(f"response was not JSON: {response.text[:200]}") from exc
    if not isinstance(payload, dict):
        raise ProbeFailure("response JSON was not an object")
    return payload


def request_json(
    method: str,
    base_url: str,
    path: str,
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> tuple[requests.Response, dict[str, Any]]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        response = requests.request(
            method,
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise ProbeFailure(f"{method} {path} failed: {exc}") from exc

    body = decode_json(response)
    if not response.ok:
        detail = body.get("detail", response.text[:200])
        raise ProbeFailure(f"{method} {path} returned {response.status_code}: {detail}")
    return response, body


def selected_headers(response: requests.Response) -> dict[str, str]:
    return {
        header: response.headers.get(header, "absent")
        for header in OBSERVABILITY_HEADERS
    }


def print_probe(name: str, response: requests.Response, payload: dict[str, Any]) -> None:
    print(f"\n== {name} ==")
    print(f"status: {response.status_code}")
    print("headers:")
    for header, value in selected_headers(response).items():
        print(f"  {header}: {value}")

    observability = payload.get("observability")
    if isinstance(observability, dict):
        print("observability:")
        print(json.dumps(observability, indent=2, sort_keys=True))


def run_health_probe(base_url: str, timeout: float) -> None:
    response, payload = request_json(
        "GET",
        base_url,
        "/healthz",
        timeout=timeout,
    )
    print_probe("healthz", response, payload)


def run_diagnosis_probe(base_url: str, api_key: str, timeout: float) -> None:
    payload = {
        "query": "OpenTelemetry probe: run a synthetic cognitive risk profile.",
        "cognitive_scores": {
            "mmse": 24.0,
            "moca": 20.0,
            "cdrsb": 4.5,
            "adas11": 18.0,
            "ravlt_immediate": 28.0,
            "ravlt_learning": 2.0,
            "faq": 6.0,
            "age": 70.0,
        },
    }
    response, body = request_json(
        "POST",
        base_url,
        "/v1/risk-profile",
        timeout=timeout,
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        payload=payload,
    )
    print_probe("risk-profile", response, body)
    print("risk_profile:")
    print(
        json.dumps(
            {
                "diagnosis": body.get("diagnosis"),
                "confidence": body.get("confidence"),
                "requires_review": body.get("requires_review"),
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> int:
    args = parse_args()
    try:
        run_health_probe(args.base_url, args.timeout)
        if not args.skip_protected:
            run_diagnosis_probe(args.base_url, args.api_key, args.timeout)
    except ProbeFailure as exc:
        print(f"OBSERVABILITY PROBE FAILED: {exc}", file=sys.stderr)
        return 1

    print("\nOBSERVABILITY PROBE PASSED")
    print(
        "Enable full tracing with NEUROSIGHT_OTEL_ENABLED=true; "
        "X-Trace-ID appears when OpenTelemetry is active."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
