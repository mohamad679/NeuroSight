"""Integration tests for completed Phase 4 API endpoints."""

from __future__ import annotations

import json
import os
import sys

from fastapi.testclient import TestClient

os.environ.setdefault("APP_ENV", "test")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.main import app
from neurosight.models.xai import SHAPExplainer

client = TestClient(app)

VALID_COGNITIVE_SCORES = {
    "MMSE": 26,
    "MOCA": 23,
    "CDRSB": 0.5,
    "ADAS11": 10.0,
    "RAVLT_immediate": 40.0,
    "RAVLT_learning": 4.0,
    "FAQ": 2.0,
    "AGE": 72,
}


def test_cognitive_upload_returns_embedding() -> None:
    """Cognitive upload with canonical scores returns 64-dim embedding."""
    response = client.post("/v1/upload/cognitive", json={"scores": VALID_COGNITIVE_SCORES})
    assert response.status_code == 200, "Cognitive upload should return HTTP 200."

    payload = response.json()
    assert payload["status"] == "ok", "Cognitive upload must return status='ok'."
    assert payload["embedding_dim"] == 64, "Cognitive embedding dimension must be 64."
    assert len(payload["embedding"]) == 64, "Cognitive embedding vector length must be 64."
    assert "unimodal_probs" in payload, "Response must include unimodal class probabilities."


def test_cognitive_upload_missing_all_scores_fails_field_specific() -> None:
    """Empty scores dict is rejected because the cognitive schema is explicit."""
    response = client.post("/v1/upload/cognitive", json={"scores": {}})
    assert response.status_code == 422

    payload = response.json()
    fields = {error["field"] for error in payload["detail"]}
    assert {"MMSE", "MOCA", "CDRSB", "ADAS11", "RAVLT_immediate", "RAVLT_learning", "FAQ", "AGE"} <= fields


def test_xai_cognitive_returns_feature_importance() -> None:
    """XAI endpoint for cognitive modality returns 8 named features."""
    response = client.get("/v1/xai/PATIENT_XAI_001?modality=cognitive")
    assert response.status_code == 200, "Cognitive XAI endpoint should return HTTP 200."

    payload = response.json()
    expected_features = set(SHAPExplainer.FEATURE_NAMES)

    assert payload["xai_available"] is True, "Cognitive XAI should be marked available."
    assert payload["modality"] == "cognitive", "Returned modality should be 'cognitive'."
    assert payload["method"] == "gradient_x_input", "Cognitive XAI method should be gradient_x_input."
    assert set(payload["feature_importance"].keys()) == expected_features, (
        "Cognitive XAI must return exactly the expected 8 cognitive feature names."
    )


def test_stream_endpoint_completes_with_done() -> None:
    """Streaming endpoint returns [DONE] and diagnosis result."""
    response = client.post(
        "/v1/diagnose/stream",
        json={"patient_id": "SYN_STREAM_0001", "query": "Provide diagnosis summary."},
    )
    assert response.status_code == 200, "Streaming endpoint should return HTTP 200."

    body = response.text
    assert "data: [DONE]" in body, "SSE stream must terminate with [DONE] marker."

    events: list[dict[str, object]] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload_text = line[6:]
        if payload_text == "[DONE]":
            continue
        events.append(json.loads(payload_text))

    final_events = [
        event
        for event in events
        if event.get("agent") == "complete" and event.get("status") == "done"
    ]
    assert final_events, "SSE stream must include a final completion event with diagnosis payload."
    final_event = final_events[-1]
    assert "diagnosis" in final_event, "Completion event must include diagnosis."
    assert "confidence" in final_event, "Completion event must include confidence."


def test_diagnose_with_cognitive_scores() -> None:
    """Full /v1/diagnose with cognitive_scores dict returns diagnosis."""
    response = client.post(
        "/v1/diagnose",
        json={
            "patient_id": "SYN_DIAG_0001",
            "query": "Assess likely diagnosis.",
            "cognitive_scores": VALID_COGNITIVE_SCORES,
        },
    )
    assert response.status_code == 200, "/v1/diagnose should return HTTP 200 for cognitive scores."

    payload = response.json()
    assert "diagnosis" in payload, "Diagnosis response must include diagnosis label."
    assert "confidence" in payload, "Diagnosis response must include confidence score."
    assert "requires_review" in payload, "Diagnosis response must include review requirement."
    assert payload["trained_on_real_data"] is False
    assert payload["clinical_validated"] is False
    assert payload["requires_expert_review"] is True
    assert "disclaimer" in payload
    assert payload["warnings"]
    assert "report_text" in payload, "Diagnosis response must include report text."
    assert payload["requires_review"] is True, "Demo-mode diagnosis must always require review."
    assert "Demo mode" in payload["report_text"], "Diagnosis report must disclose demo/untrained mode."


def test_protected_endpoint_requires_api_key(monkeypatch) -> None:
    """Protected endpoints reject missing API keys outside test mode."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("NEUROSIGHT_API_KEY", "secret-test-key")

    missing_key = client.post("/v1/upload/cognitive", json={"scores": VALID_COGNITIVE_SCORES})
    assert missing_key.status_code == 401, "Protected endpoint must reject missing API key."

    with_key = client.post(
        "/v1/upload/cognitive",
        headers={"X-API-Key": "secret-test-key"},
        json={"scores": VALID_COGNITIVE_SCORES},
    )
    assert with_key.status_code == 200, "Protected endpoint must accept the configured API key."
