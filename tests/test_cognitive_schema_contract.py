from __future__ import annotations

import re
from pathlib import Path
import numpy as np

from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import app
from neurosight.schemas.cognitive import COGNITIVE_FEATURES, CognitiveSchema

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PAYLOAD = {
    "MMSE": 26,
    "MOCA": 23,
    "CDRSB": 0.5,
    "ADAS11": 10.0,
    "RAVLT_immediate": 40.0,
    "RAVLT_learning": 4.0,
    "FAQ": 2.0,
    "AGE": 72,
}
OBSOLETE_FIELDS = ("trail_a", "trail_b", "verbal_fluency", "sex", "cdr")


def test_cognitive_schema_order_and_lowercase_canonical_aliases() -> None:
    schema = CognitiveSchema.model_validate(
        {
            "mmse": 26,
            "moca": 23,
            "cdrsb": 0.5,
            "adas11": 10.0,
            "ravlt_immediate": 40.0,
            "ravlt_learning": 4.0,
            "faq": 2.0,
            "age": 72,
        }
    )

    assert tuple(schema.to_features_dict()) == COGNITIVE_FEATURES
    assert list(schema.to_features_dict().values()) == [26, 23, 0.5, 10.0, 40.0, 4.0, 2.0, 72]


def test_cognitive_schema_rejects_impossible_values_and_obsolete_fields() -> None:
    invalid = dict(CANONICAL_PAYLOAD, MMSE=31)
    try:
        CognitiveSchema.model_validate(invalid)
    except ValidationError as exc:
        fields = {error["loc"][0] for error in exc.errors()}
        assert "MMSE" in fields
    else:
        raise AssertionError("MMSE outside [0, 30] must fail validation.")

    for field in OBSOLETE_FIELDS:
        stale = dict(CANONICAL_PAYLOAD, **{field: 1})
        try:
            CognitiveSchema.model_validate(stale)
        except ValidationError as exc:
            assert field in str(exc)
        else:
            raise AssertionError(f"Obsolete field {field} must be rejected.")


def test_api_valid_and_invalid_cognitive_payloads_are_field_specific() -> None:
    client = TestClient(app)
    ok = client.post("/v1/diagnose", json={"cognitive_scores": CANONICAL_PAYLOAD})
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["trained_on_real_data"] is False
    assert payload["clinical_validated"] is False
    assert payload["requires_expert_review"] is True

    bad = client.post("/v1/diagnose", json={"cognitive_scores": dict(CANONICAL_PAYLOAD, AGE=130)})
    assert bad.status_code == 422
    assert any(error["field"] == "AGE" for error in bad.json()["detail"])


def test_frontend_payload_schema_uses_exact_backend_fields() -> None:
    api_source = (ROOT / "frontend/src/lib/api.ts").read_text()
    type_source = (ROOT / "frontend/src/lib/types.ts").read_text()
    store_source = (ROOT / "frontend/src/lib/store.ts").read_text()
    panel_source = (ROOT / "frontend/src/components/DiagnosisPanel.tsx").read_text()

    for field in COGNITIVE_FEATURES:
        assert field in api_source
        assert field in type_source
        assert field in store_source
        assert field in panel_source

    checked_sources = api_source + type_source + store_source + panel_source
    for stale in OBSOLETE_FIELDS:
        assert re.search(rf"\b{re.escape(stale)}\b", checked_sources) is None


def test_gradio_argument_order_matches_app_contract() -> None:
    # 1. Verify frontend payload data array construction order in api.ts
    api_source = (ROOT / "frontend/src/lib/api.ts").read_text()
    match = re.search(r"data:\s*\[(.*?)\]\s*\}", api_source, flags=re.S)
    assert match is not None
    items = [item.strip() for item in match.group(1).split(",") if item.strip()]
    assert items == [
        "req.MMSE",
        "req.MOCA",
        "req.CDRSB",
        "req.ADAS11",
        "req.RAVLT_immediate",
        "req.RAVLT_learning",
        "req.FAQ",
        "req.AGE",
        "gradioInputValue(mriUpload)",
        "gradioInputValue(eegUpload)",
        "req.query ?? \"\"",
    ]

    # 2. Verify backend function signature and click handler inputs order in app.py
    app_source = (ROOT / "app.py").read_text()
    
    ui_def_match = re.search(r"def run_diagnosis_ui\((.*?)\) ->", app_source, flags=re.S)
    assert ui_def_match is not None
    ui_args = [arg.split(":")[0].strip() for arg in ui_def_match.group(1).split(",") if arg.strip()]
    assert ui_args == [
        "mmse",
        "moca",
        "cdrsb",
        "adas11",
        "ravlt_immediate",
        "ravlt_learning",
        "faq",
        "age",
        "mri_file",
        "eeg_file",
        "query",
    ]

    render_def_match = re.search(r"def _run_diagnosis_and_render\((.*?)\) ->", app_source, flags=re.S)
    assert render_def_match is not None
    render_args = [arg.split(":")[0].strip() for arg in render_def_match.group(1).split(",") if arg.strip()]
    assert render_args == [
        "mmse",
        "moca",
        "cdrsb",
        "adas11",
        "ravlt_immediate",
        "ravlt_learning",
        "faq",
        "age",
        "mri_file",
        "eeg_file",
        "query",
    ]

    click_match = re.search(r"analyze_btn\.click\(.*?inputs=\[(.*?)\]", app_source, flags=re.S)
    assert click_match is not None
    click_inputs = [x.strip() for x in click_match.group(1).split(",") if x.strip()]
    assert click_inputs == [
        "mmse",
        "moca",
        "cdrsb",
        "adas11",
        "ravlt_immediate",
        "ravlt_learning",
        "faq",
        "age",
        "mri_file",
        "eeg_file",
        "query",
    ]


def test_cognitive_schema_age_and_faq_mappings() -> None:
    # Validate mapping correctness for age and faq
    schema = CognitiveSchema.model_validate(
        {
            "mmse": 26,
            "moca": 23,
            "cdrsb": 0.5,
            "adas11": 10.0,
            "ravlt_immediate": 40.0,
            "ravlt_learning": 4.0,
            "faq": 2.0,
            "age": 72.0,
        }
    )
    assert schema.AGE == 72.0
    assert schema.FAQ == 2.0

    features_dict = schema.to_features_dict()
    assert features_dict["AGE"] == 72.0
    assert features_dict["FAQ"] == 2.0
    
    # Verify that obsolete fields are completely absent from fallback payloads
    for obsolete_field in OBSOLETE_FIELDS:
        assert obsolete_field not in features_dict
        assert obsolete_field.upper() not in features_dict


def test_backend_tensor_construction_aligns_values() -> None:
    from neurosight.models.service import ModelService
    
    schema = CognitiveSchema.model_validate(
        {
            "mmse": 20.0,
            "moca": 18.0,
            "cdrsb": 3.0,
            "adas11": 15.0,
            "ravlt_immediate": 30.0,
            "ravlt_learning": 1.0,
            "faq": 5.0,
            "age": 75.0,
        }
    )
    
    service = ModelService()
    tensor = service.preprocess_cognitive(schema)
    assert tensor.shape == (1, 8)
    
    # Reconstruct standard scaling behavior: (value - mean) / std
    expected_values = np.array([20.0, 18.0, 3.0, 15.0, 30.0, 1.0, 5.0, 75.0], dtype=np.float32)
    expected_normalized = (expected_values - service.cog_means) / (service.cog_stds + 1e-8)
    
    actual_normalized = tensor.cpu().numpy()[0]
    assert np.allclose(actual_normalized, expected_normalized, atol=1e-6)
