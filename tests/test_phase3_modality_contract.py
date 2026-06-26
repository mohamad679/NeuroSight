"""Phase 3 MRI/EEG preprocessing contract tests."""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import torch
from fastapi.testclient import TestClient

os.environ.setdefault("APP_ENV", "test")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api import main as api_main
from api.main import app
from neurosight.data.modality_contract import inspect_eeg_array, inspect_mri_array


client = TestClient(app)


def test_modalities_status_exposes_shapes_formats_and_notice() -> None:
    """Modalities status should describe real-data-shaped preprocessing."""
    response = client.get("/v1/modalities/status")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["mri"]["model_tensor_shape"] == [1, 1, 96, 96, 96]
    assert ".nii.gz" in payload["mri"]["supported_formats"]
    assert "DICOM .zip" in payload["mri"]["supported_formats"]
    assert payload["eeg"]["model_tensor_shape"] == [1, 19, 1024]
    assert ".edf" in payload["eeg"]["supported_formats"]
    assert payload["cognitive"]["embedding_dim"] == 64
    assert "unvalidated" in payload["scientific_notice"]


def test_modality_inspectors_report_expected_layouts() -> None:
    """Pure inspectors should classify MRI/EEG input layouts without model inference."""
    mri_payload = inspect_mri_array(np.zeros((32, 48, 40), dtype=np.float32), "scan.nii.gz")
    eeg_payload = inspect_eeg_array(np.zeros((1024, 19), dtype=np.float32), "signal.npy")

    assert mri_payload["source_format"] == "nifti_gz"
    assert mri_payload["accepted_layout"] is True
    assert mri_payload["input_layout"] == "volume_dhw"
    assert mri_payload["model_tensor_shape"] == [1, 1, 96, 96, 96]

    assert eeg_payload["source_format"] == "numpy"
    assert eeg_payload["accepted_layout"] is True
    assert eeg_payload["input_layout"] == "time_channels_transposed"
    assert eeg_payload["model_tensor_shape"] == [1, 19, 1024]


def test_eeg_upload_returns_preprocessing_metadata(monkeypatch) -> None:
    """EEG upload should include preprocessing details alongside the embedding."""
    fake_eeg_model = type(
        "FakeEegModel",
        (),
        {"encoder": staticmethod(lambda tensor: torch.zeros((tensor.shape[0], 256)))},
    )()
    monkeypatch.setattr(api_main, "_get_eeg_model", lambda: fake_eeg_model)

    arr = np.random.randn(1024, 19).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)

    response = client.post(
        "/v1/upload/eeg",
        files={"file": ("phase3_eeg.npy", buf, "application/octet-stream")},
    )
    assert response.status_code == 200

    payload = response.json()
    preprocessing = payload["preprocessing"]
    assert payload["embedding_dim"] == 256
    assert preprocessing["source_shape"] == [1024, 19]
    assert preprocessing["input_layout"] == "time_channels_transposed"
    assert preprocessing["prepared_tensor_shape"] == [1, 19, 1024]
