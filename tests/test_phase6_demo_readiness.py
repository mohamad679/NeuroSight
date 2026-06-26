import os
import sys

from fastapi.testclient import TestClient

os.environ.setdefault("APP_ENV", "test")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app  # noqa: E402


def test_demo_readiness_endpoint_reports_launch_contract(monkeypatch) -> None:
    monkeypatch.setenv("NEUROSIGHT_RUNTIME_MODE", "demo")
    monkeypatch.setenv("NEUROSIGHT_CLASS_MODE", "six_class_demo")
    client = TestClient(app)

    response = client.get("/v1/demo/readiness")

    assert response.status_code == 200
    payload = response.json()
    check_ids = {item["id"] for item in payload["checks"]}

    assert payload["status"] in {"demo_ready", "demo_ready_with_warnings"}
    assert payload["clinical_use_allowed"] is False
    assert payload["recommended_patient_id"]
    assert "data_contract" in check_ids
    assert "privacy_governance" in check_ids
    assert payload["adni_style_private_run"]["recommended_class_mode"] == "three_class_adni"
    assert payload["recommended_ui_flow"][0]["view"] == "Overview"


def test_demo_readiness_is_in_health_root_and_capabilities() -> None:
    client = TestClient(app)

    health = client.get("/healthz").json()
    root = client.get("/").json()
    capability_ids = {item["id"] for item in health["capabilities"]["items"]}

    assert "demo_readiness" in health
    assert health["demo_readiness"]["clinical_use_allowed"] is False
    assert "demo_readiness" in capability_ids
    assert root["demo_readiness_status"] == "/v1/demo/readiness"


def test_demo_readiness_flags_missing_data(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("NEUROSIGHT_RUNTIME_MODE", "adni_style")
    monkeypatch.setenv("NEUROSIGHT_PATIENT_CSV_PATH", str(tmp_path / "missing.csv"))
    client = TestClient(app)

    payload = client.get("/v1/demo/readiness").json()
    checks = {item["id"]: item for item in payload["checks"]}

    assert payload["status"] == "needs_attention"
    assert checks["data_contract"]["status"] == "action_required"
    assert checks["data_contract"]["blocking"] is True
