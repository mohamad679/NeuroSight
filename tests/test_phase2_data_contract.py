"""Phase 2 demo/ADNI-style data contract tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault("APP_ENV", "test")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app
from neurosight.data.synthetic import generate_structured_synthetic


client = TestClient(app)


def test_data_status_uses_synthetic_demo_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Demo mode should report the synthetic ADNI-like CSV when no private path is set."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEUROSIGHT_RUNTIME_MODE", "demo")
    monkeypatch.delenv("NEUROSIGHT_PATIENT_CSV_PATH", raising=False)
    generate_structured_synthetic(n_per_class=4, seed=101)

    response = client.get("/v1/data/status")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["source_kind"] == "synthetic_adni_like_demo"
    assert payload["privacy"]["public_demo_safe"] is True
    assert payload["summary"]["row_count"] == 24
    assert payload["summary"]["adni_style_count"] == 12
    assert payload["summary"]["synthetic_demo_only_count"] == 12
    assert payload["schema"]["missing_columns"] == []


def test_demo_patients_are_bounded_and_privacy_scoped(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Demo patient endpoint should return a small, sanitized list of sample rows."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEUROSIGHT_RUNTIME_MODE", "demo")
    monkeypatch.delenv("NEUROSIGHT_PATIENT_CSV_PATH", raising=False)
    generate_structured_synthetic(n_per_class=3, seed=102)

    response = client.get("/v1/data/demo-patients?limit=5")
    assert response.status_code == 200

    payload = response.json()
    assert payload["count"] == 5
    assert payload["recommended_patient_id"] == payload["patients"][0]["patient_id"]
    assert payload["privacy"]["private_adni_in_repo"] is False
    assert set(payload["patients"][0]["scores"]) == {
        "MMSE",
        "MOCA",
        "CDRSB",
        "ADAS11",
        "RAVLT_immediate",
        "RAVLT_learning",
        "FAQ",
        "AGE",
    }
    assert "modalities" in payload["patients"][0]


def test_patient_diagnosis_runs_from_demo_csv_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Patient diagnosis should work in public demo mode without private ADNI files."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEUROSIGHT_RUNTIME_MODE", "demo")
    monkeypatch.delenv("NEUROSIGHT_PATIENT_CSV_PATH", raising=False)
    generate_structured_synthetic(n_per_class=3, seed=103)

    response = client.post(
        "/v1/diagnose/patient/100000",
        json={"query": "Run the demo patient diagnosis."},
    )
    assert response.status_code == 200

    payload = response.json()
    assert payload["requires_review"] is True
    assert "diagnosis" in payload
    assert "Configured dataset patient 100000" in payload["report_text"]
    assert "Demo mode" in payload["report_text"]
