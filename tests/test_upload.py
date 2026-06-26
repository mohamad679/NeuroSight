import io
import os
import sys
import zipfile

import numpy as np
from fastapi.testclient import TestClient

os.environ.setdefault("APP_ENV", "test")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api import main as api_main
from api.main import app

client = TestClient(app)


def test_mri_upload_npy() -> None:
    """Test MRI upload with synthetic `.npy` file."""
    arr = np.random.randn(96, 96, 96).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)

    res = client.post(
        "/v1/upload/mri",
        files={"file": ("test.npy", buf, "application/octet-stream")},
    )

    assert res.status_code == 200, "MRI upload should return HTTP 200."
    data = res.json()
    assert data["status"] == "ok", "MRI upload response should indicate success."
    assert data["embedding_dim"] == 768, "MRI embedding dimension must be 768."
    assert len(data["embedding"]) == 768, "MRI embedding vector length must be 768."


def test_eeg_upload_npy() -> None:
    """Test EEG upload with synthetic `.npy` file."""
    arr = np.random.randn(19, 1024).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)

    res = client.post(
        "/v1/upload/eeg",
        files={"file": ("test.npy", buf, "application/octet-stream")},
    )

    assert res.status_code == 200, "EEG upload should return HTTP 200."
    data = res.json()
    assert data["status"] == "ok", "EEG upload response should indicate success."
    assert data["embedding_dim"] == 256, "EEG embedding dimension must be 256."
    assert len(data["embedding"]) == 256, "EEG embedding vector length must be 256."


def test_upload_rejects_oversized_file(monkeypatch) -> None:
    """Upload endpoints reject payloads over the configured byte limit."""
    monkeypatch.setattr(api_main, "MAX_UPLOAD_BYTES", 16)
    buf = io.BytesIO(b"x" * 17)

    res = client.post(
        "/v1/upload/eeg",
        files={"file": ("too_large.npy", buf, "application/octet-stream")},
    )

    assert res.status_code == 413, "Oversized uploads should return HTTP 413."
    assert "maximum size" in res.json()["detail"]


def test_mri_upload_rejects_unsafe_zip_member() -> None:
    """DICOM zip uploads reject path traversal entries before extraction/parsing."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as archive:
        archive.writestr("../evil.dcm", b"not a dicom file")
    buf.seek(0)

    res = client.post(
        "/v1/upload/mri",
        files={"file": ("dicom.zip", buf, "application/zip")},
    )

    assert res.status_code == 422, "Unsafe zip paths should be rejected."
    assert "Unsafe zip member path" in res.json()["detail"]
