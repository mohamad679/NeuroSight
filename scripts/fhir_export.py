#!/usr/bin/env python3
"""Generate a FHIR R4 Bundle from a NeuroSight risk-profile result.

By default this runs offline with synthetic, public-demo-safe values. With
`--from-backend`, it calls the local/deployed FastAPI `/v1/risk-profile` route and
exports the actual response returned by the backend.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_API_KEY = "dev-key"
DEFAULT_OUTPUT = "logs/fhir/neurosight_demo_bundle.json"
DIAGNOSIS_CHOICES = ("normal", "mci", "ad", "ftd", "lbd", "vd")
DEFAULT_COGNITIVE_SCORES: dict[str, float] = {
    "mmse": 24.0,
    "moca": 20.0,
    "cdrsb": 4.5,
    "adas11": 18.0,
    "ravlt_immediate": 28.0,
    "ravlt_learning": 2.0,
    "faq": 6.0,
    "age": 70.0,
}
DEFAULT_REPORT = (
    "NeuroSight demo export: synthetic cognitive inputs were converted into a "
    "FHIR R4 diagnostic bundle. This output is for interoperability testing "
    "only and is not clinical evidence."
)


class FhirExportError(Exception):
    """Raised when backend export preparation fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a NeuroSight risk-profile result as a FHIR R4 Bundle."
    )
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help=f"Output JSON path. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--stdout", action="store_true", help="Print the bundle JSON to stdout.")
    parser.add_argument("--patient-id", default="NS-DEMO-0001", help="Pseudonymized patient identifier.")
    parser.add_argument(
        "--diagnosis",
        default="mci",
        choices=DIAGNOSIS_CHOICES,
        help="Offline demo diagnosis label.",
    )
    parser.add_argument("--confidence", type=float, default=0.68, help="Offline demo confidence in [0, 1].")
    parser.add_argument("--age", type=float, default=70.0, help="Demo patient age extension value.")
    parser.add_argument("--sex", default="unknown", choices=["male", "female", "other", "unknown"])
    parser.add_argument(
        "--model-status",
        default="demo_untrained",
        help="Model status tag written to Device.property.",
    )
    parser.add_argument(
        "--source",
        default="offline-synthetic-demo",
        help="Provenance source label.",
    )
    parser.add_argument(
        "--from-backend",
        action="store_true",
        help="Call POST /v1/risk-profile and export the backend response.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("NEUROSIGHT_BACKEND_URL", DEFAULT_BASE_URL),
        help=f"FastAPI base URL for --from-backend. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NEUROSIGHT_API_KEY", DEFAULT_API_KEY),
        help="API key for --from-backend. Default: dev-key",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="Backend request timeout in seconds.")
    return parser.parse_args()


def backend_diagnosis(base_url: str, api_key: str, timeout: float) -> dict[str, Any]:
    """Call NeuroSight `/v1/risk-profile` with synthetic cognitive scores."""
    request_payload = {
        "query": "FHIR export probe: create an interoperable research report.",
        "cognitive_scores": DEFAULT_COGNITIVE_SCORES,
    }
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/risk-profile",
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            json=request_payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise FhirExportError(f"Backend request failed: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise FhirExportError(f"Backend response was not JSON: {response.text[:200]}") from exc

    if not response.ok:
        detail = payload.get("detail") if isinstance(payload, dict) else response.text[:200]
        raise FhirExportError(f"Backend returned {response.status_code}: {detail}") from None
    if not isinstance(payload, dict):
        raise FhirExportError("Backend response JSON was not an object.")
    return payload


def build_bundle_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build the bundle from CLI args or a backend response."""
    from neurosight.interop.fhir_export import build_diagnosis_bundle

    diagnosis = args.diagnosis
    confidence = args.confidence
    report_text = DEFAULT_REPORT
    requires_review = True
    source = args.source

    if args.from_backend:
        payload = backend_diagnosis(args.base_url, args.api_key, args.timeout)
        diagnosis = str(payload.get("diagnosis", diagnosis))
        confidence = float(payload.get("confidence", confidence))
        report_text = str(payload.get("report_text", report_text))
        requires_review = bool(payload.get("requires_review", True))
        source = f"backend:{args.base_url.rstrip('/')}/v1/risk-profile"

    return build_diagnosis_bundle(
        patient_id=args.patient_id,
        diagnosis=diagnosis,
        confidence=confidence,
        report_text=report_text,
        requires_review=requires_review,
        cognitive_scores=DEFAULT_COGNITIVE_SCORES,
        age=args.age,
        sex=args.sex,
        model_status=args.model_status,
        source=source,
    )


def summarize_bundle(bundle: dict[str, Any], output_path: Path | None, *, stream: Any = sys.stdout) -> None:
    """Print a concise export summary."""
    resources = [
        entry.get("resource", {}).get("resourceType")
        for entry in bundle.get("entry", [])
        if isinstance(entry, dict)
    ]
    print("FHIR EXPORT PASSED", file=stream)
    print("FHIR version tag: 4.0.1", file=stream)
    print(f"Bundle id: {bundle.get('id')}", file=stream)
    print(f"Resources: {', '.join(str(resource) for resource in resources)}", file=stream)
    if output_path is not None:
        print(f"Wrote: {output_path}", file=stream)


def main() -> int:
    from neurosight.interop.fhir_export import bundle_to_json, validate_bundle_shape, write_bundle

    args = parse_args()
    try:
        bundle = build_bundle_from_args(args)
        errors = validate_bundle_shape(bundle)
        if errors:
            raise FhirExportError("; ".join(errors))

        output_path: Path | None = None
        if args.stdout:
            print(bundle_to_json(bundle))
            summarize_bundle(bundle, None, stream=sys.stderr)
        else:
            output_path = write_bundle(bundle, args.out)
            summarize_bundle(bundle, output_path)
    except FhirExportError as exc:
        print(f"FHIR EXPORT FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
