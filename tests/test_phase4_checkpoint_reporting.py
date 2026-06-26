"""Phase 4 checkpoint/evaluation reporting contract tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault("APP_ENV", "test")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app


client = TestClient(app)


def test_checkpoint_status_reports_artifact_without_claiming_loaded(monkeypatch, tmp_path: Path) -> None:
    """Artifact presence and runtime load must be separate states."""
    checkpoint_path = tmp_path / "best_fusion.pt"
    checkpoint_path.write_bytes(b"fake checkpoint bytes")
    registry_path = tmp_path / "model_registry.json"
    registry_path.write_text(
        json.dumps(
            [
                {
                    "run_id": "run-a",
                    "status": "staging",
                    "checkpoint_path": str(checkpoint_path),
                    "metrics": {"val_auc": 0.91, "val_f1": 0.72},
                }
            ]
        ),
        encoding="utf-8",
    )
    eval_path = tmp_path / "eval_results.json"
    eval_path.write_text(
        json.dumps({"macro_auc": 0.9, "ece": 0.2, "per_class_f1": {"class_0": 0.8}}),
        encoding="utf-8",
    )
    model_card_path = tmp_path / "MODEL_CARD.md"
    model_card_path.write_text("# Model Card\n", encoding="utf-8")

    monkeypatch.setenv("NEUROSIGHT_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("NEUROSIGHT_MODEL_REGISTRY_PATH", str(registry_path))
    monkeypatch.setenv("NEUROSIGHT_EVAL_RESULTS_PATH", str(eval_path))
    monkeypatch.setenv("NEUROSIGHT_MODEL_CARD_PATH", str(model_card_path))
    monkeypatch.delenv("NEUROSIGHT_LOAD_CHECKPOINT", raising=False)
    app.state.trained_checkpoint_loaded = False
    app.state.checkpoint_load_error = None

    response = client.get("/v1/models/checkpoint/status")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "available"
    assert payload["checkpoint"]["exists"] is True
    assert payload["loading"]["enabled"] is False
    assert payload["loading"]["loaded"] is False
    assert payload["registry"]["run_count"] == 1
    assert payload["registry"]["best_run"]["run_id"] == "run-a"
    assert payload["evaluation"]["available"] is True
    assert payload["scientific_claims"]["clinical_validation"] is False
    assert payload["scientific_claims"]["public_metrics_are_clinical_claims"] is False


def test_eval_report_exposes_scientific_claims(monkeypatch, tmp_path: Path) -> None:
    """Evaluation report endpoint should always include limitations disclosure."""
    checkpoint_path = tmp_path / "missing.pt"
    eval_path = tmp_path / "eval_results.json"
    eval_path.write_text(
        json.dumps({"macro_auc": 0.81, "ece": 0.33, "modality_ablation": {"all": 0.81}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("NEUROSIGHT_CHECKPOINT_PATH", str(checkpoint_path))
    monkeypatch.setenv("NEUROSIGHT_MODEL_REGISTRY_PATH", str(tmp_path / "registry.json"))
    monkeypatch.setenv("NEUROSIGHT_EVAL_RESULTS_PATH", str(eval_path))
    monkeypatch.setenv("NEUROSIGHT_MODEL_CARD_PATH", str(tmp_path / "MODEL_CARD.md"))
    app.state.trained_checkpoint_loaded = False
    app.state.checkpoint_load_error = None

    response = client.get("/v1/eval/report")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["evaluation"]["metrics"]["macro_auc"] == 0.81
    assert payload["scientific_claims"]["clinical_use_allowed"] is False
    assert payload["scientific_claims"]["training_data"] == "synthetic_adni_like"


def test_health_model_status_keeps_default_demo_mode(monkeypatch, tmp_path: Path) -> None:
    """Default health must not imply a trained checkpoint is loaded."""
    monkeypatch.setenv("NEUROSIGHT_CHECKPOINT_PATH", str(tmp_path / "missing.pt"))
    monkeypatch.delenv("NEUROSIGHT_LOAD_CHECKPOINT", raising=False)
    app.state.trained_checkpoint_loaded = False
    app.state.checkpoint_load_error = None

    response = client.get("/healthz")
    assert response.status_code == 200

    payload = response.json()
    assert payload["model_status"]["mode"] == "demo_untrained"
    assert payload["model_status"]["trained_checkpoint_loaded"] is False
    assert payload["checkpoint"]["loading"]["loaded"] is False
