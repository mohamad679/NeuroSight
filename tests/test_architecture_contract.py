"""Phase 1 architecture contract tests."""

from __future__ import annotations

import os
import sys

from fastapi.testclient import TestClient

os.environ.setdefault("APP_ENV", "test")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app


client = TestClient(app)


def test_health_exposes_runtime_scope_and_capabilities(monkeypatch) -> None:
    """Health payload should explain runtime mode, class scope, and UI coverage."""
    monkeypatch.setenv("NEUROSIGHT_RUNTIME_MODE", "demo")
    monkeypatch.setenv("NEUROSIGHT_CLASS_MODE", "six_class_demo")

    response = client.get("/healthz")
    assert response.status_code == 200

    payload = response.json()
    runtime = payload["runtime"]
    capabilities = payload["capabilities"]

    assert runtime["runtime_mode"] == "demo"
    assert runtime["class_mode"] == "six_class_demo"
    assert runtime["adni_style_classes"] == ["normal", "mci", "ad"]
    assert runtime["synthetic_demo_only_classes"] == ["ftd", "lbd", "vd"]
    assert runtime["data_policy"]["private_adni_data"] == "not_included"

    assert capabilities["summary"]["total"] >= 10
    assert capabilities["summary"]["implemented"] == capabilities["summary"]["total"]
    assert capabilities["summary"]["ui_coverage_percent"] > 0
    assert any(item["id"] == "diagnosis" for item in capabilities["items"])

    assert payload["model_status"]["trained_checkpoint_loaded"] is False
    assert payload["upload_limits"]["max_upload_bytes"] > 0


def test_root_exposes_compact_public_contract(monkeypatch) -> None:
    """Root payload should be enough for a landing page or proxy to discover routes."""
    monkeypatch.setenv("NEUROSIGHT_RUNTIME_MODE", "adni_style")
    monkeypatch.setenv("NEUROSIGHT_CLASS_MODE", "three_class_adni")

    response = client.get("/")
    assert response.status_code == 200

    payload = response.json()
    assert payload["name"] == "NeuroSight Backend API"
    assert payload["runtime"]["runtime_mode"] == "adni_style"
    assert payload["runtime"]["classes"] == ["normal", "mci", "ad"]
    assert payload["health"] == "/healthz"
    assert payload["uploads"]["mri"] == "/v1/upload/mri"
    assert payload["knowledge_graph"] == "/v1/kg/query"
    assert payload["capabilities"]["summary"]["ui_exposed"] > 0
