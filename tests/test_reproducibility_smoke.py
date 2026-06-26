"""CPU-friendly reproducibility and smoke coverage."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

os.environ.setdefault("APP_ENV", "test")

from api.main import app  # noqa: E402
from neurosight.models.fusion import CrossModalAttentionFusion  # noqa: E402
from neurosight.tracking.model_registry import ModelRegistry  # noqa: E402
from neurosight.utils.seed import set_global_seed  # noqa: E402


@pytest.mark.unit
def test_model_forward_pass_with_missing_modalities_is_stable() -> None:
    """Fusion model handles missing MRI/EEG without random crashes."""
    set_global_seed(101)
    model = CrossModalAttentionFusion(num_classes=6, d_model=64)
    model.eval()
    cog = torch.randn(2, 64)

    with torch.no_grad():
        output = model(mri=None, eeg=None, cog=cog)

    assert output["logits"].shape == (2, 6)
    assert output["probs"].shape == (2, 6)
    assert torch.allclose(output["probs"].sum(dim=1), torch.ones(2), atol=1e-5)
    assert set(output["modality_weights"]) == {"mri", "eeg", "cog"}


@pytest.mark.unit
def test_model_forward_pass_is_deterministic_for_identical_seed() -> None:
    """Identical model/input seeds produce identical outputs."""
    set_global_seed(202)
    first_model = CrossModalAttentionFusion(num_classes=6, d_model=64).eval()
    first_input = torch.randn(1, 64)
    with torch.no_grad():
        first_probs = first_model(mri=None, eeg=None, cog=first_input)["probs"]

    set_global_seed(202)
    second_model = CrossModalAttentionFusion(num_classes=6, d_model=64).eval()
    second_input = torch.randn(1, 64)
    with torch.no_grad():
        second_probs = second_model(mri=None, eeg=None, cog=second_input)["probs"]

    assert torch.allclose(first_probs, second_probs, atol=1e-7)


@pytest.mark.integration
def test_api_startup_health_contract() -> None:
    """FastAPI app starts in-process and exposes health contract."""
    client = TestClient(app)
    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "healthy"}
    assert payload["demo_readiness"]["clinical_use_allowed"] is False


@pytest.mark.integration
def test_demo_cognitive_path_is_deterministic() -> None:
    """Identical demo inputs return identical API outputs."""
    client = TestClient(app)
    payload = {
        "scores": {
            "MMSE": 26,
            "MOCA": 23,
            "CDRSB": 0.5,
            "ADAS11": 10.0,
            "RAVLT_immediate": 40.0,
            "RAVLT_learning": 4.0,
            "FAQ": 2.0,
            "AGE": 72,
        }
    }

    first = client.post("/v1/upload/cognitive", json=payload)
    second = client.post("/v1/upload/cognitive", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["embedding"] == second_payload["embedding"]
    assert first_payload["unimodal_probs"] == second_payload["unimodal_probs"]


@pytest.mark.unit
def test_model_registry_register_and_promote(tmp_path: Path) -> None:
    """Local registry persists entries and production promotion semantics."""
    registry = ModelRegistry(registry_path=str(tmp_path / "registry.json"))
    registry.register_model(
        run_id="run-a",
        model_name="fusion",
        checkpoint_path="checkpoints/a.pt",
        metrics={"val_auc": 0.7},
        config={"seed": 1},
    )
    registry.register_model(
        run_id="run-b",
        model_name="fusion",
        checkpoint_path="checkpoints/b.pt",
        metrics={"val_auc": 0.8},
        config={"seed": 2},
    )

    assert [run["run_id"] for run in registry.list_runs()] == ["run-b", "run-a"]
    promoted = registry.promote_to_production("run-a")
    assert promoted["status"] == "production"
    assert registry.get_production_model()["run_id"] == "run-a"


@pytest.mark.unit
def test_repository_hygiene_script_passes(tmp_path: Path) -> None:
    """Hygiene script remains runnable in CI."""
    import shutil
    repo_root = Path(__file__).resolve().parents[1]

    # Copy the repository files to tmp_path while ignoring cache/checkpoint directories
    ignore_patterns = shutil.ignore_patterns(
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "checkpoints",
        "outputs",
        "logs",
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        ".next",
        "out",
        ".deploy",
        "coverage",
        "htmlcov",
        "mlruns",
        "mlartifacts",
        "__MACOSX"
    )
    clean_dir = tmp_path / "clean_repo"
    shutil.copytree(repo_root, clean_dir, ignore=ignore_patterns, symlinks=True)

    result = subprocess.run(
        [sys.executable, "scripts/check_repo_hygiene.py"],
        cwd=clean_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout.lower()
