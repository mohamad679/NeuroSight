import os
import sys

from fastapi.testclient import TestClient

os.environ.setdefault("APP_ENV", "test")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app  # noqa: E402


def test_xai_status_discloses_methods_without_clinical_claims() -> None:
    client = TestClient(app)
    response = client.get("/v1/xai/status")

    assert response.status_code == 200
    payload = response.json()
    methods = {item["modality"]: item for item in payload["methods"]}

    assert payload["interpretation_policy"]["clinical_use_allowed"] is False
    assert payload["interpretation_policy"]["causal_claims_allowed"] is False
    assert methods["cognitive"]["status"] == "implemented"
    assert methods["mri"]["validated_for_clinical_use"] is False
    assert methods["eeg"]["requires_uploaded_data"] is True


def test_governance_status_reports_privacy_and_security_controls() -> None:
    client = TestClient(app)
    response = client.get("/v1/governance/status")

    assert response.status_code == 200
    payload = response.json()

    assert payload["privacy"]["public_demo_safe"] is True
    assert payload["privacy"]["private_adni_in_repository"] is False
    assert payload["privacy"]["clinical_use_allowed"] is False
    assert payload["security"]["api_key_header"] == "X-API-Key"
    assert payload["security"]["upload_controls"]["numpy_pickle_disabled"] is True
    assert payload["scientific_disclosure"]["validated_clinically"] is False


def test_cognitive_xai_response_includes_interpretation_and_privacy() -> None:
    client = TestClient(app)
    response = client.get("/v1/xai/PATIENT_XAI_001?modality=cognitive")

    assert response.status_code == 200
    payload = response.json()

    assert payload["xai_available"] is True
    assert payload["method_contract"]["modality"] == "cognitive"
    assert payload["interpretation_policy"]["requires_human_review"] is True
    assert payload["privacy"]["patient_data_persisted_by_xai_endpoint"] is False
    assert "target_label" in payload


def test_health_and_root_include_phase5_contracts() -> None:
    client = TestClient(app)
    health = client.get("/healthz").json()
    root = client.get("/").json()

    capability_ids = {item["id"] for item in health["capabilities"]["items"]}

    assert "xai" in health
    assert "governance" in health
    assert "xai_status" in capability_ids
    assert "governance_status" in capability_ids
    assert root["xai_status"] == "/v1/xai/status"
    assert root["governance_status"] == "/v1/governance/status"
