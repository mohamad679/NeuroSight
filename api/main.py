# api/main.py
import sys
import os
import time
import json
import asyncio
import csv
import io
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, Depends, HTTPException, Request, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ValidationError
from typing import Dict, Any, Optional, Tuple, List

try:
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address
    from slowapi import _rate_limit_exceeded_handler
    SLOWAPI_AVAILABLE = True
except ModuleNotFoundError:
    SLOWAPI_AVAILABLE = False

    class RateLimitExceeded(Exception):
        """Fallback rate-limit exception type when slowapi is unavailable."""

    class Limiter:
        """No-op limiter fallback to preserve runtime compatibility."""

        def __init__(self, key_func: Any):
            self.key_func = key_func

        def limit(self, _: str) -> Any:
            def _decorator(func: Any) -> Any:
                return func

            return _decorator

    def get_remote_address(request: Request) -> str:
        """Fallback remote address resolver."""
        return request.client.host if request.client is not None else "unknown"

    def _rate_limit_exceeded_handler(request: Request, exc: Exception) -> Response:
        """Fallback rate-limit response handler."""
        del request
        del exc
        return Response(content="Rate limit exceeded", status_code=429)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from neurosight.contracts import Diagnosis, DiagnoseRequest, DiagnoseResponse
from neurosight.schemas.cognitive import COGNITIVE_FEATURES, CognitiveSchema
from neurosight.observability.otel import (
    current_trace_id,
    observability_status,
    setup_observability,
    start_span,
)

limiter = Limiter(key_func=get_remote_address)

COGNITIVE_FEATURE_ORDER: tuple[str, ...] = COGNITIVE_FEATURES

COGNITIVE_DEFAULTS: dict[str, float] = {
    "MMSE": 27.0,
    "MOCA": 25.0,
    "CDRSB": 0.5,
    "ADAS11": 10.0,
    "RAVLT_immediate": 40.0,
    "RAVLT_learning": 4.0,
    "FAQ": 2.0,
    "AGE": 70.0,
}

MAX_UPLOAD_BYTES = int(os.environ.get("NEUROSIGHT_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
MAX_DICOM_ZIP_MEMBERS = int(os.environ.get("NEUROSIGHT_MAX_DICOM_ZIP_MEMBERS", "2000"))
MAX_DICOM_ZIP_UNCOMPRESSED_BYTES = int(
    os.environ.get("NEUROSIGHT_MAX_DICOM_ZIP_UNCOMPRESSED_BYTES", str(512 * 1024 * 1024))
)
MAX_MRI_VALUES = int(os.environ.get("NEUROSIGHT_MAX_MRI_VALUES", str(64 * 1024 * 1024)))
MAX_EEG_VALUES = int(os.environ.get("NEUROSIGHT_MAX_EEG_VALUES", str(64 * 1024 * 1024)))
APP_VERSION = "0.3.0"
FRONTEND_DIST_DIR = Path(
    os.environ.get(
        "NEUROSIGHT_FRONTEND_DIR",
        str(Path(__file__).resolve().parents[1] / "frontend" / "out"),
    )
)


def _frontend_available() -> bool:
    """Return whether a static frontend build is available to serve."""
    return (FRONTEND_DIST_DIR / "index.html").exists()


def _frontend_file(frontend_path: str) -> Path:
    """Resolve a requested frontend path within the static export directory."""
    cleaned = frontend_path.strip().lstrip("/")
    candidate = (FRONTEND_DIST_DIR / cleaned).resolve()
    root = FRONTEND_DIST_DIR.resolve()
    if root != candidate and root not in candidate.parents:
        raise HTTPException(status_code=404, detail="Static asset not found.")
    return candidate
MODEL_STATUS_MODE = "demo_untrained"
MODEL_STATUS_NOTICE = (
    "Demo mode: this runtime initializes model weights in process and does not "
    "load a validated clinical checkpoint. Predictions are not clinically meaningful."
)
CHECKPOINT_STATUS_NOTICE = (
    "Checkpoint artifacts may be trained on synthetic ADNI-like data, but they are "
    "not clinically validated and must not be interpreted as medical evidence."
)
RUNTIME_MODES: dict[str, dict[str, str]] = {
    "demo": {
        "label": "Public demo mode",
        "description": (
            "Synthetic/mock-data safe mode for GitHub and Hugging Face demos. "
            "No private ADNI data is required or bundled."
        ),
    },
    "adni_style": {
        "label": "ADNI-style ingestion mode",
        "description": (
            "Accepts ADNI-style metadata and modality files supplied by an authorized operator. "
            "The repository does not include private ADNI records."
        ),
    },
    "research": {
        "label": "Research checkpoint mode",
        "description": (
            "Reserved for externally trained checkpoints and validated datasets supplied "
            "outside the public repository."
        ),
    },
}
CLASS_MODES: dict[str, dict[str, Any]] = {
    "three_class_adni": {
        "label": "3-class ADNI-style workflow",
        "classes": ["normal", "mci", "ad"],
        "adni_style_classes": ["normal", "mci", "ad"],
        "synthetic_demo_only_classes": [],
        "description": (
            "Scientifically cleaner ADNI-style scope. Real validation should focus on "
            "Normal/MCI/AD unless additional datasets are added."
        ),
    },
    "six_class_demo": {
        "label": "6-class demo workflow",
        "classes": [diagnosis.value for diagnosis in Diagnosis],
        "adni_style_classes": ["normal", "mci", "ad"],
        "synthetic_demo_only_classes": ["ftd", "lbd", "vd"],
        "description": (
            "Current prototype output space. FTD/LBD/VD are synthetic demo placeholders "
            "until separate public/authorized datasets and trained checkpoints are added."
        ),
    },
}
API_CAPABILITIES: tuple[dict[str, Any], ...] = (
    {
        "id": "health_runtime",
        "label": "Health, runtime mode, and scientific scope",
        "endpoint": "GET /healthz",
        "ui_view": "Overview",
        "status": "implemented",
        "protected": False,
    },
    {
        "id": "risk_profile",
        "label": "Synthetic risk profiling from cognitive scores or modality embeddings",
        "endpoint": "POST /v1/risk-profile",
        "legacy_endpoint": "POST /v1/diagnose",
        "ui_view": "Risk Profiling",
        "status": "implemented",
        "protected": True,
    },
    {
        "id": "diagnosis",
        "label": "Legacy diagnosis-compatible risk profiling endpoint",
        "endpoint": "POST /v1/diagnose",
        "ui_view": "Risk Profiling",
        "status": "implemented",
        "protected": True,
    },
    {
        "id": "patient_risk_profile",
        "label": "ADNI-style patient lookup and risk profiling",
        "endpoint": "POST /v1/risk-profile/patient/{patient_id}",
        "legacy_endpoint": "POST /v1/diagnose/patient/{patient_id}",
        "ui_view": "Risk Profiling",
        "status": "implemented",
        "protected": True,
    },
    {
        "id": "data_readiness",
        "label": "Demo/ADNI-style data readiness and sample cohort",
        "endpoint": "GET /v1/data/status",
        "ui_view": "Data",
        "status": "implemented",
        "protected": True,
    },
    {
        "id": "modality_preprocessing",
        "label": "MRI/EEG preprocessing contract and upload readiness",
        "endpoint": "GET /v1/modalities/status",
        "ui_view": "Uploads",
        "status": "implemented",
        "protected": True,
    },
    {
        "id": "streaming_risk_profile",
        "label": "Streaming multi-agent evaluation flow",
        "endpoint": "POST /v1/risk-profile/stream",
        "legacy_endpoint": "POST /v1/diagnose/stream",
        "ui_view": "Streaming",
        "status": "implemented",
        "protected": True,
    },
    {
        "id": "mri_upload",
        "label": "MRI volume upload and embedding",
        "endpoint": "POST /v1/upload/mri",
        "ui_view": "Uploads",
        "status": "implemented",
        "protected": True,
        "formats": [".npy", ".nii", ".nii.gz", "DICOM .zip"],
    },
    {
        "id": "eeg_upload",
        "label": "EEG signal upload and embedding",
        "endpoint": "POST /v1/upload/eeg",
        "ui_view": "Uploads",
        "status": "implemented",
        "protected": True,
        "formats": [".npy", ".edf"],
    },
    {
        "id": "cognitive_upload",
        "label": "Cognitive score encoding",
        "endpoint": "POST /v1/upload/cognitive",
        "ui_view": "Uploads",
        "status": "implemented",
        "protected": True,
    },
    {
        "id": "knowledge_graph",
        "label": "Temporal knowledge graph query",
        "endpoint": "POST /v1/kg/query",
        "ui_view": "Knowledge Graph",
        "status": "implemented",
        "protected": True,
    },
    {
        "id": "explainability",
        "label": "Modality-specific explainability payloads",
        "endpoint": "GET /v1/xai/{patient_id}",
        "ui_view": "XAI",
        "status": "implemented",
        "protected": True,
    },
    {
        "id": "xai_status",
        "label": "Explainability method availability and interpretation policy",
        "endpoint": "GET /v1/xai/status",
        "ui_view": "XAI",
        "status": "implemented",
        "protected": True,
    },
    {
        "id": "governance_status",
        "label": "Privacy, security, and scientific disclosure contract",
        "endpoint": "GET /v1/governance/status",
        "ui_view": "Trust",
        "status": "implemented",
        "protected": True,
    },
    {
        "id": "demo_readiness",
        "label": "Final demo launch readiness checklist",
        "endpoint": "GET /v1/demo/readiness",
        "ui_view": "Demo",
        "status": "implemented",
        "protected": True,
    },
    {
        "id": "evaluation",
        "label": "Metrics, benchmark, cross-validation, and evaluation history",
        "endpoint": "GET/POST /v1/eval/*",
        "ui_view": "Evaluation",
        "status": "implemented",
        "protected": True,
    },
    {
        "id": "evaluation_report",
        "label": "Model-card and evaluation artifact report",
        "endpoint": "GET /v1/eval/report",
        "ui_view": "Evaluation",
        "status": "implemented",
        "protected": True,
    },
    {
        "id": "model_registry",
        "label": "Model runs, production metadata, and promotion",
        "endpoint": "GET/POST /v1/models*",
        "ui_view": "Models",
        "status": "implemented",
        "protected": True,
    },
    {
        "id": "checkpoint_status",
        "label": "Checkpoint availability and runtime load status",
        "endpoint": "GET /v1/models/checkpoint/status",
        "ui_view": "Models",
        "status": "implemented",
        "protected": True,
    },
)

DIAGNOSIS_TO_COGNITIVE_TEMPLATE: dict[str, dict[str, float]] = {
    "normal": {
        "MMSE": 29.0,
        "MOCA": 28.0,
        "CDRSB": 0.2,
        "ADAS11": 6.5,
        "RAVLT_immediate": 45.0,
        "RAVLT_learning": 6.0,
        "FAQ": 1.5,
        "AGE": 71.5,
    },
    "mci": {
        "MMSE": 25.5,
        "MOCA": 23.5,
        "CDRSB": 1.0,
        "ADAS11": 13.0,
        "RAVLT_immediate": 31.0,
        "RAVLT_learning": 4.0,
        "FAQ": 6.0,
        "AGE": 74.0,
    },
    "ad": {
        "MMSE": 20.5,
        "MOCA": 18.0,
        "CDRSB": 3.5,
        "ADAS11": 26.5,
        "RAVLT_immediate": 16.0,
        "RAVLT_learning": 1.5,
        "FAQ": 19.0,
        "AGE": 77.0,
    },
    "ftd": {
        "MMSE": 22.5,
        "MOCA": 19.0,
        "CDRSB": 2.0,
        "ADAS11": 22.0,
        "RAVLT_immediate": 20.0,
        "RAVLT_learning": 2.0,
        "FAQ": 14.5,
        "AGE": 64.0,
    },
    "lbd": {
        "MMSE": 23.0,
        "MOCA": 20.0,
        "CDRSB": 2.5,
        "ADAS11": 21.0,
        "RAVLT_immediate": 21.0,
        "RAVLT_learning": 3.0,
        "FAQ": 15.0,
        "AGE": 72.0,
    },
    "vd": {
        "MMSE": 23.5,
        "MOCA": 21.0,
        "CDRSB": 2.5,
        "ADAS11": 19.0,
        "RAVLT_immediate": 23.5,
        "RAVLT_learning": 3.0,
        "FAQ": 12.5,
        "AGE": 75.0,
    },
}


class UploadTooLarge(Exception):
    """Raised when an uploaded payload exceeds configured limits."""


def _env_choice(name: str, default: str, choices: dict[str, Any]) -> str:
    """Read an enum-like environment setting with a deterministic fallback."""
    value = os.environ.get(name, default).strip().lower()
    return value if value in choices else default


def _runtime_contract() -> dict[str, Any]:
    """Return the public runtime and scientific-scope contract."""
    runtime_mode = _env_choice("NEUROSIGHT_RUNTIME_MODE", "demo", RUNTIME_MODES)
    class_mode = _env_choice("NEUROSIGHT_CLASS_MODE", "six_class_demo", CLASS_MODES)
    runtime_payload = RUNTIME_MODES[runtime_mode]
    class_payload = CLASS_MODES[class_mode]

    return {
        "runtime_mode": runtime_mode,
        "runtime_label": runtime_payload["label"],
        "runtime_description": runtime_payload["description"],
        "class_mode": class_mode,
        "class_label": class_payload["label"],
        "class_description": class_payload["description"],
        "classes": list(class_payload["classes"]),
        "adni_style_classes": list(class_payload["adni_style_classes"]),
        "synthetic_demo_only_classes": list(class_payload["synthetic_demo_only_classes"]),
        "data_policy": {
            "public_repo_data": "synthetic_or_mock_only",
            "private_adni_data": "not_included",
            "clinical_use": "not_allowed",
        },
        "scientific_scope_notice": (
            "ADNI-style workflows are credible for Normal/MCI/AD structure. "
            "FTD/LBD/VD require additional datasets and validation before real claims."
        ),
    }


def _api_capability_payload() -> dict[str, Any]:
    """Return API capability metadata shared by docs, health, and UI."""
    capabilities = [dict(capability) for capability in API_CAPABILITIES]
    total = len(capabilities)
    ui_exposed = sum(1 for capability in capabilities if capability.get("ui_view"))
    implemented = sum(
        1 for capability in capabilities if capability.get("status") == "implemented"
    )
    return {
        "summary": {
            "total": total,
            "implemented": implemented,
            "ui_exposed": ui_exposed,
            "ui_coverage_percent": round((ui_exposed / total) * 100, 1) if total else 0.0,
        },
        "items": capabilities,
    }


def _format_bytes(n_bytes: int) -> str:
    """Format bytes into a compact human-readable string."""
    if n_bytes >= 1024 * 1024:
        return f"{n_bytes / (1024 * 1024):.0f} MB"
    if n_bytes >= 1024:
        return f"{n_bytes / 1024:.0f} KB"
    return f"{n_bytes} bytes"


def _safe_float(value: Any, default: float) -> float:
    """Safely parse numeric values with a deterministic fallback.

    Args:
        value: Raw input value that may be string/numeric/None.
        default: Fallback numeric value when parsing fails.

    Returns:
        Parsed float or fallback.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _cognitive_validation_detail(exc: Exception) -> list[dict[str, str]]:
    """Convert schema validation failures into clear field-specific API errors."""
    if isinstance(exc, ValidationError):
        details: list[dict[str, str]] = []
        for error in exc.errors():
            loc = ".".join(str(part) for part in error.get("loc", ())) or "cognitive_scores"
            details.append(
                {
                    "field": loc,
                    "message": str(error.get("msg", "Invalid cognitive score.")),
                    "type": str(error.get("type", "value_error")),
                }
            )
        return details
    return [
        {
            "field": "cognitive_scores",
            "message": str(exc),
            "type": exc.__class__.__name__,
        }
    ]


def _extract_latest_diagnosis(history: List[Dict[str, Any]]) -> Optional[str]:
    """Extract the latest diagnosis label from KG history records.

    Args:
        history: KG history records returned by `get_patient_history`.

    Returns:
        Lowercase diagnosis label when available, otherwise `None`.
    """
    latest_date = ""
    latest_diagnosis: Optional[str] = None

    for entry in history:
        node_payload = entry.get("node", {}) if isinstance(entry, dict) else {}
        edge_payload = entry.get("edge", {}) if isinstance(entry, dict) else {}

        diagnosis_value = (
            node_payload.get("diagnosis")
            if isinstance(node_payload, dict)
            else None
        )
        if diagnosis_value is None:
            continue

        edge_date = str(edge_payload.get("date", ""))
        if edge_date >= latest_date:
            latest_date = edge_date
            latest_diagnosis = str(diagnosis_value).strip().lower()

    return latest_diagnosis


def _synthesize_cognitive_scores_from_kg(kg: Any, patient_id: str) -> tuple[dict[str, float], bool]:
    """Generate cognitive input from KG context with robust defaults.

    Args:
        kg: Knowledge graph instance.
        patient_id: Patient identifier used for KG retrieval.

    Returns:
        Tuple of (`scores`, `has_history`) where `scores` contains all 8
        required cognitive features.
    """
    synthesized = dict(COGNITIVE_DEFAULTS)
    history: List[Dict[str, Any]] = []

    if kg is not None and hasattr(kg, "get_patient_history"):
        try:
            raw_history = kg.get_patient_history(patient_id)
            if isinstance(raw_history, list):
                history = [entry for entry in raw_history if isinstance(entry, dict)]
        except (TypeError, ValueError, KeyError):
            history = []

    has_history = len(history) > 0
    latest_diagnosis = _extract_latest_diagnosis(history)
    if latest_diagnosis in DIAGNOSIS_TO_COGNITIVE_TEMPLATE:
        synthesized.update(DIAGNOSIS_TO_COGNITIVE_TEMPLATE[latest_diagnosis])

    if (
        kg is not None
        and hasattr(kg, "graph")
        and hasattr(kg.graph, "__contains__")
        and patient_id in kg.graph
    ):
        patient_node = dict(kg.graph.nodes[patient_id])
        record = patient_node.get("record")
        if record is not None:
            synthesized["AGE"] = _safe_float(
                getattr(record, "age", synthesized["AGE"]),
                synthesized["AGE"],
            )

    return synthesized, has_history


def _build_cognitive_vector(input_scores: Dict[str, Any]) -> dict[str, float]:
    """Create normalized 8-feature cognitive vector from partial user input.

    Args:
        input_scores: User-provided cognitive score mapping.

    Returns:
        Dictionary containing all required model features.
    """
    schema = CognitiveSchema.model_validate(input_scores)
    return schema.to_features_dict()


def _tensor_from_cognitive_features(features: dict[str, float]) -> torch.Tensor:
    """Convert 8-feature cognitive dictionary into model tensor.

    Args:
        features: Cognitive features keyed by canonical feature names.

    Returns:
        Tensor of shape `(1, 8)` and dtype float32.
    """
    schema = CognitiveSchema.model_validate(features)
    return _get_model_service().preprocess_cognitive(schema)


def _cognitive_summary(
    feature_importance: dict[str, float],
    diagnosis_label: str,
    from_history: bool,
) -> str:
    """Generate human-readable cognitive XAI summary sentence.

    Args:
        feature_importance: Feature importance mapping.
        diagnosis_label: Target diagnosis label.
        from_history: Whether KG history informed synthetic feature generation.

    Returns:
        Concise summary string.
    """
    ranked = sorted(
        feature_importance.items(),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    top_features = ranked[:3] if ranked else []
    top_text = ", ".join(
        f"{name} ({float(value):.2f})" for name, value in top_features
    )
    source_text = "KG-derived cognitive profile" if from_history else "neutral default cognitive profile"
    return (
        f"Gradient x input explanation for {diagnosis_label}: strongest contributors are "
        f"{top_text}. Source profile: {source_text}."
    )


def _load_numpy_from_bytes(raw_bytes: bytes) -> np.ndarray:
    """Read a `.npy` array from raw bytes.

    Args:
        raw_bytes: Byte stream containing serialized NumPy array data.

    Returns:
        Loaded NumPy array converted to float32.

    Raises:
        ValueError: If bytes are empty or not valid `.npy` content.
    """
    if not raw_bytes:
        raise ValueError("Uploaded file is empty.")
    try:
        array = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
    except (ValueError, OSError, EOFError) as exc:
        raise ValueError(f"Unable to parse .npy file: {exc}") from exc
    return np.asarray(array, dtype=np.float32)


def _read_upload_bytes(file: UploadFile, max_bytes: Optional[int] = None) -> bytes:
    """Read upload bytes with an explicit maximum size.

    Args:
        file: Upload object to read from its underlying file handle.
        max_bytes: Maximum accepted byte count.

    Returns:
        Uploaded bytes.

    Raises:
        UploadTooLarge: If the upload exceeds `max_bytes`.
    """
    limit = MAX_UPLOAD_BYTES if max_bytes is None else int(max_bytes)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = file.file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise UploadTooLarge(
                f"Uploaded file exceeds the maximum size of {_format_bytes(limit)}."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _load_numpy_from_upload(file: UploadFile) -> np.ndarray:
    """Read a `.npy` upload safely from memory.

    Args:
        file: Uploaded file object expected to contain a NumPy array.

    Returns:
        Loaded NumPy array.

    Raises:
        ValueError: If bytes are missing or cannot be parsed as `.npy`.
    """
    raw_bytes = _read_upload_bytes(file)
    return _load_numpy_from_bytes(raw_bytes)


def _validate_array_size(array: np.ndarray, label: str, max_values: int) -> None:
    """Reject arrays that are too large to process safely."""
    if int(array.size) > max_values:
        raise UploadTooLarge(
            f"{label} contains {array.size:,} values, exceeding the limit of {max_values:,}."
        )


def _load_nifti_from_bytes(raw_bytes: bytes, suffix: str) -> np.ndarray:
    """Load a NIfTI volume from uploaded bytes.

    Args:
        raw_bytes: Raw file bytes from `.nii` or `.nii.gz`.
        suffix: Upload filename suffix used for temporary file creation.

    Returns:
        NIfTI volume as float32 NumPy array.

    Raises:
        ValueError: If nibabel is unavailable or file cannot be parsed.
    """
    if not raw_bytes:
        raise ValueError("Uploaded file is empty.")
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise ValueError("nibabel is required to process NIfTI uploads.") from exc

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(raw_bytes)
        temp_path = temp_file.name

    try:
        nifti_image = nib.load(temp_path)
        data = nifti_image.get_fdata(dtype=np.float32)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to parse NIfTI upload: {exc}") from exc
    finally:
        try:
            os.remove(temp_path)
        except OSError as remove_error:
            _ = remove_error

    nifti_array = np.asarray(data, dtype=np.float32)
    _validate_array_size(nifti_array, "NIfTI volume", MAX_MRI_VALUES)
    return nifti_array


def _validate_filename(filename: str) -> str:
    """Validate upload filename against traversal and injection attacks.
    
    Args:
        filename: The uploaded filename.
        
    Returns:
        The validated filename.
        
    Raises:
        ValueError: If filename contains unsafe path traversal, null bytes, or separators.
    """
    if not filename:
        raise ValueError("Filename is empty.")
    if "\x00" in filename:
        raise ValueError("Null bytes are not allowed in filename.")
    if ".." in filename or "/" in filename or "\\" in filename:
        raise ValueError("Path traversal components are not allowed in filename.")
    return filename


def _load_dicom_zip_from_bytes(raw_bytes: bytes) -> np.ndarray:
    """Load a DICOM series from zipped upload bytes.

    Args:
        raw_bytes: Raw bytes from uploaded zip archive.

    Returns:
        Volume array stacked along slice axis with dtype float32.

    Raises:
        ValueError: If pydicom is unavailable, archive is invalid, or no slices are found.
    """
    if not raw_bytes:
        raise ValueError("Uploaded file is empty.")

    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            if len(members) > MAX_DICOM_ZIP_MEMBERS:
                raise UploadTooLarge(
                    "DICOM zip contains too many files "
                    f"({len(members):,}; limit {MAX_DICOM_ZIP_MEMBERS:,})."
                )

            total_uncompressed = 0
            for info in members:
                if not _is_safe_zip_member(info.filename):
                    raise ValueError(f"Unsafe zip member path: {info.filename!r}.")
                if info.flag_bits & 0x1:
                    raise ValueError("Encrypted zip uploads are not supported.")

                # Reject nested archives by extension check
                ext = os.path.splitext(info.filename.lower())[1]
                if ext in {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tgz", ".tbz2"}:
                    raise ValueError(f"Nested archive file types are not permitted: {info.filename}")

                # Reject nested archives by reading magic numbers (first 4 bytes)
                with archive.open(info) as f:
                    magic = f.read(4)
                if (
                    magic.startswith(b"PK\x03\x04") or
                    magic.startswith(b"\x1f\x8b") or
                    magic.startswith(b"Rar!") or
                    magic.startswith(b"7z\xbc\xaf\x27\x1c")
                ):
                    raise ValueError(f"Nested archive magic bytes detected: {info.filename}")

                # Zip bomb check for individual file compression ratio
                if info.compress_size > 0:
                    ratio = info.file_size / info.compress_size
                    if ratio > 100.0:
                        raise ValueError(f"Individual file compression ratio too high ({ratio:.1f}x): {info.filename}")

                total_uncompressed += int(info.file_size)
                if total_uncompressed > MAX_DICOM_ZIP_UNCOMPRESSED_BYTES:
                    raise UploadTooLarge(
                        "DICOM zip expands beyond the maximum allowed size of "
                        f"{_format_bytes(MAX_DICOM_ZIP_UNCOMPRESSED_BYTES)}."
                    )

            # Zip bomb check for total compression ratio
            if len(raw_bytes) > 0:
                total_ratio = total_uncompressed / len(raw_bytes)
                if total_ratio > 100.0:
                    raise ValueError(f"Total zip compression ratio too high ({total_ratio:.1f}x). Potential zip bomb.")

            try:
                import pydicom
                from pydicom.errors import InvalidDicomError
            except ModuleNotFoundError as exc:
                raise ValueError("pydicom is required to process DICOM zip uploads.") from exc

            dicom_slices: list[tuple[float, np.ndarray]] = []
            fallback_index = 0.0
            for info in sorted(members, key=lambda item: item.filename):
                try:
                    with archive.open(info) as member:
                        dataset = pydicom.dcmread(member, force=True)
                except (InvalidDicomError, OSError, RuntimeError, zipfile.BadZipFile):
                    continue

                if not hasattr(dataset, "pixel_array"):
                    continue

                pixel_array = np.asarray(dataset.pixel_array, dtype=np.float32)
                if pixel_array.ndim != 2:
                    continue

                if hasattr(dataset, "ImagePositionPatient"):
                    position = dataset.ImagePositionPatient
                    if isinstance(position, (list, tuple)) and len(position) >= 3:
                        try:
                            sort_key = float(position[2])
                        except (TypeError, ValueError):
                            sort_key = fallback_index
                    else:
                        sort_key = fallback_index
                elif hasattr(dataset, "InstanceNumber"):
                    try:
                        sort_key = float(dataset.InstanceNumber)
                    except (TypeError, ValueError):
                        sort_key = fallback_index
                else:
                    sort_key = fallback_index

                dicom_slices.append((sort_key, pixel_array))
                fallback_index += 1.0

            if not dicom_slices:
                raise ValueError("No readable DICOM slices found in uploaded zip archive.")

            dicom_slices.sort(key=lambda item: item[0])
            first_shape = dicom_slices[0][1].shape
            if any(slice_array.shape != first_shape for _, slice_array in dicom_slices):
                raise ValueError("Inconsistent DICOM slice shapes detected in archive.")

            stacked = np.stack([slice_array for _, slice_array in dicom_slices], axis=0)
            _validate_array_size(stacked, "DICOM volume", MAX_MRI_VALUES)
            return stacked.astype(np.float32)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Unable to parse zip archive: {exc}") from exc


def _is_safe_zip_member(filename: str) -> bool:
    """Return whether a zip member path is safe to process."""
    normalized = filename.replace("\\", "/")
    if not normalized or "\x00" in normalized:
        return False
    if normalized.startswith("/"):
        return False
    parts = normalized.split("/")
    for part in parts:
        if part == "." or part == "..":
            return False
    return True


def _load_mri_array_from_upload(file: UploadFile) -> np.ndarray:
    """Load MRI array from supported upload formats.

    Supported formats:
    - `.npy`
    - `.nii` / `.nii.gz`
    - `.zip` (DICOM series)

    Args:
        file: Uploaded MRI file object.

    Returns:
        MRI volume array as float32 NumPy array.

    Raises:
        ValueError: If extension is unsupported or parsing fails.
    """
    if not file.filename:
        raise ValueError("MRI upload requires a filename.")

    filename = file.filename.lower()
    raw_bytes = _read_upload_bytes(file)

    if filename.endswith(".npy"):
        mri_array = _load_numpy_from_bytes(raw_bytes)
        _validate_array_size(mri_array, "MRI array", MAX_MRI_VALUES)
        return mri_array
    if filename.endswith(".nii.gz"):
        return _load_nifti_from_bytes(raw_bytes, suffix=".nii.gz")
    if filename.endswith(".nii"):
        return _load_nifti_from_bytes(raw_bytes, suffix=".nii")
    if filename.endswith(".zip"):
        return _load_dicom_zip_from_bytes(raw_bytes)

    raise ValueError("MRI upload expects .npy, .nii, .nii.gz, or .zip file.")


def _prepare_mri_tensor(mri_array: np.ndarray) -> torch.Tensor:
    """Validate and convert MRI volume into model-ready tensor.

    Args:
        mri_array: Raw MRI numpy array, expected `(D,H,W)` or `(1,D,H,W)`.

    Returns:
        Tensor of shape `(1, 1, 96, 96, 96)` for MRI encoder input.

    Raises:
        ValueError: If array dimensionality is unsupported.
    """
    from neurosight.models.mri import get_mri_transforms

    if mri_array.ndim == 4 and mri_array.shape[0] == 1:
        mri_array = np.squeeze(mri_array, axis=0)
    elif mri_array.ndim == 4 and mri_array.shape[-1] == 1:
        mri_array = np.squeeze(mri_array, axis=-1)
    elif mri_array.ndim != 3:
        raise ValueError(
            f"MRI input must have shape (D,H,W) or (1,D,H,W), got {mri_array.shape}."
        )

    transforms = get_mri_transforms()
    transformed = transforms(mri_array)
    if transformed.ndim != 4:
        raise ValueError(
            f"Transformed MRI tensor must be 4D (C,D,H,W), got {tuple(transformed.shape)}."
        )
    return transformed.unsqueeze(0).to(dtype=torch.float32)


def _prepare_eeg_tensor(eeg_array: np.ndarray) -> torch.Tensor:
    """Normalize EEG array and map it to encoder shape.

    Args:
        eeg_array: Raw EEG numpy array from `.npy` or preprocessed `.edf`.

    Returns:
        Tensor of shape `(1, 19, 1024)` ready for EEG encoder.

    Raises:
        ValueError: If array cannot be interpreted as EEG channels x time.
    """
    if eeg_array.ndim == 3:
        if eeg_array.shape[0] <= 0:
            raise ValueError("EEG epochs array is empty.")
        channel_time = eeg_array.mean(axis=0)
    elif eeg_array.ndim == 2:
        channel_time = eeg_array
    else:
        raise ValueError(
            f"EEG input must have shape (channels,time) or (epochs,channels,time), got {eeg_array.shape}."
        )

    if channel_time.shape[0] != 19 and channel_time.shape[1] == 19:
        channel_time = channel_time.T

    if channel_time.shape[0] != 19:
        raise ValueError(
            f"EEG input must contain 19 channels, got shape {channel_time.shape}."
        )

    n_time = int(channel_time.shape[1])
    if n_time <= 0:
        raise ValueError("EEG time dimension must be positive.")
    if n_time < 1024:
        pad_width = 1024 - n_time
        channel_time = np.pad(channel_time, ((0, 0), (0, pad_width)), mode="constant")
    elif n_time > 1024:
        channel_time = channel_time[:, :1024]

    eeg_tensor = torch.tensor(channel_time, dtype=torch.float32).unsqueeze(0)
    return eeg_tensor


def _extract_embedding_payload(
    embedding_tensor: torch.Tensor,
    expected_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert encoder output tensor into JSON-safe payload.

    Args:
        embedding_tensor: Encoder embedding tensor, expected `(1, D)` or `(D,)`.
        expected_dim: Optional expected size for embedding validation.

    Returns:
        Dictionary with `embedding_dim` and `embedding` values.

    Raises:
        ValueError: If `expected_dim` is provided and output size mismatches.
    """
    flattened = embedding_tensor.squeeze(0).flatten().to(dtype=torch.float32)
    embedding_list = flattened.tolist()
    if expected_dim is not None and len(embedding_list) != expected_dim:
        raise ValueError(
            f"Unexpected embedding size {len(embedding_list)} (expected {expected_dim})."
        )
    return {"embedding_dim": len(embedding_list), "embedding": embedding_list}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models on startup + warm-up dummy forward pass."""
    import torch
    from neurosight.models.service import ModelService
    from knowledge_graph import NeuroKnowledgeGraph

    checkpoint_path = _checkpoint_path() if _checkpoint_loading_enabled() else None
    app.state.model_service = ModelService(checkpoint_path=checkpoint_path)
    app.state.mri_model = app.state.model_service.mri_model
    app.state.eeg_model = app.state.model_service.eeg_model
    app.state.cognitive_model = app.state.model_service.cognitive_model
    app.state.fusion_model = app.state.model_service.fusion_model

    app.state.trained_checkpoint_loaded = app.state.model_service.checkpoint_loaded
    app.state.checkpoint_load_error = app.state.model_service.checkpoint_error
    app.state.checkpoint_loaded_metadata = app.state.model_service.checkpoint_metadata

    app.state.kg = NeuroKnowledgeGraph()
    app.state.started_at = time.time()
    try:
        app.state.kg.load()
        app.state.kg_loaded = True
    except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError):
        app.state.kg_loaded = False

    with torch.no_grad():
        import spaces_config
        if not spaces_config.DISABLE_MRI_WARMUP:
            app.state.mri_model(torch.randn(1, 1, 96, 96, 96))
        app.state.eeg_model(torch.randn(1, 19, 1024))
        app.state.cognitive_model(torch.randn(1, 8))
    yield
    # shutdown: nothing to clean up


app = FastAPI(title="NeuroSight API", lifespan=lifespan)
app.state.limiter = limiter
setup_observability(app, service_name="neurosight-api", service_version=APP_VERSION)


def _get_model_service() -> Any:
    """Get or initialize ModelService lazily."""
    model_service = getattr(app.state, "model_service", None)
    if model_service is None:
        from neurosight.models.service import ModelService
        checkpoint_path = _checkpoint_path() if _checkpoint_loading_enabled() else None
        model_service = ModelService(checkpoint_path=checkpoint_path)
        app.state.model_service = model_service
        app.state.mri_model = model_service.mri_model
        app.state.eeg_model = model_service.eeg_model
        app.state.cognitive_model = model_service.cognitive_model
        app.state.fusion_model = model_service.fusion_model
        app.state.trained_checkpoint_loaded = model_service.checkpoint_loaded
        app.state.checkpoint_load_error = model_service.checkpoint_error
        app.state.checkpoint_loaded_metadata = model_service.checkpoint_metadata
    return model_service


def _get_cognitive_model() -> torch.nn.Module:
    return _get_model_service().cognitive_model


def _get_fusion_model() -> torch.nn.Module:
    return _get_model_service().fusion_model


def _get_mri_model() -> torch.nn.Module:
    return _get_model_service().mri_model


def _get_eeg_model() -> torch.nn.Module:
    return _get_model_service().eeg_model


def _get_kg() -> Any:
    from knowledge_graph import NeuroKnowledgeGraph

    kg = getattr(app.state, "kg", None)
    if kg is None:
        kg = NeuroKnowledgeGraph()
        app.state.kg = kg
    return kg


if SLOWAPI_AVAILABLE:
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _evaluation_csv_path() -> str:
    """Resolve evaluation CSV path from environment with a safe fallback."""
    return os.environ.get("NEUROSIGHT_DATA_CSV_PATH", "data/ADNIMERGE_synthetic.csv")


def _checkpoint_path() -> Path:
    """Resolve the optional model checkpoint path."""
    return Path(os.environ.get("NEUROSIGHT_CHECKPOINT_PATH", "checkpoints/best_fusion.pt"))


def _registry_path() -> Path:
    """Resolve model registry metadata path."""
    return Path(os.environ.get("NEUROSIGHT_MODEL_REGISTRY_PATH", "logs/model_registry.json"))


def _eval_results_path() -> Path:
    """Resolve persisted evaluation result path."""
    return Path(os.environ.get("NEUROSIGHT_EVAL_RESULTS_PATH", "logs/eval_results.json"))


def _model_card_path() -> Path:
    """Resolve model card path."""
    return Path(os.environ.get("NEUROSIGHT_MODEL_CARD_PATH", "MODEL_CARD.md"))


def _checkpoint_loading_enabled() -> bool:
    """Return whether runtime checkpoint loading is explicitly enabled."""
    return os.environ.get("NEUROSIGHT_LOAD_CHECKPOINT", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _checkpoint_status_payload() -> dict[str, Any]:
    """Return checkpoint availability and scientific reporting metadata."""
    from neurosight.tracking.checkpoint_contract import build_checkpoint_contract

    return build_checkpoint_contract(
        checkpoint_path=_checkpoint_path(),
        registry_path=_registry_path(),
        eval_results_path=_eval_results_path(),
        model_card_path=_model_card_path(),
        load_enabled=_checkpoint_loading_enabled(),
        loaded=bool(getattr(app.state, "trained_checkpoint_loaded", False)),
        load_error=getattr(app.state, "checkpoint_load_error", None),
    )


def _demo_csv_path() -> Path:
    """Resolve the public synthetic ADNI-like demo CSV path."""
    return Path(os.environ.get("NEUROSIGHT_DEMO_CSV_PATH", "data/ADNIMERGE_synthetic.csv"))


def _patient_csv_path() -> Path:
    """Resolve patient metadata CSV path for demo or authorized ADNI-style data."""
    runtime_mode = _env_choice("NEUROSIGHT_RUNTIME_MODE", "demo", RUNTIME_MODES)
    demo_path = _demo_csv_path()
    if runtime_mode == "demo":
        return demo_path

    configured_path = os.environ.get("NEUROSIGHT_PATIENT_CSV_PATH", "").strip()
    if configured_path:
        return Path(configured_path)
    return Path("data/ADNIMERGE.csv")


def _patient_mri_dir() -> Path:
    """Resolve the directory containing Space-hosted MRI `.npy` volumes."""
    return Path(os.environ.get("NEUROSIGHT_MRI_DIR", "data/mri"))


def _patient_eeg_dir() -> Path:
    """Resolve the directory containing Space-hosted EEG `.npy` arrays."""
    return Path(os.environ.get("NEUROSIGHT_EEG_DIR", "data/eeg"))


def _data_status_payload(sample_limit: int = 0) -> dict[str, Any]:
    """Return public-demo/ADNI-style data readiness metadata."""
    from neurosight.data.demo_contract import build_data_contract

    return build_data_contract(
        csv_path=_patient_csv_path(),
        synthetic_csv_path=_demo_csv_path(),
        mri_dir=_patient_mri_dir(),
        eeg_dir=_patient_eeg_dir(),
        runtime=_runtime_contract(),
        sample_limit=sample_limit,
    )


def _modality_status_payload() -> dict[str, Any]:
    """Return MRI/EEG/cognitive preprocessing readiness metadata."""
    from neurosight.data.modality_contract import build_modality_contract

    return build_modality_contract(
        _upload_limits_payload()
    )


def _upload_limits_payload() -> dict[str, int]:
    """Return upload limits in one reusable JSON-safe shape."""
    return {
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_mri_values": MAX_MRI_VALUES,
        "max_eeg_values": MAX_EEG_VALUES,
        "max_dicom_zip_members": MAX_DICOM_ZIP_MEMBERS,
    }


def _xai_status_payload() -> dict[str, Any]:
    """Return explainability availability and interpretation metadata."""
    from neurosight.governance.trust_contract import build_xai_contract

    return build_xai_contract(runtime=_runtime_contract())


def _governance_status_payload() -> dict[str, Any]:
    """Return privacy, security, and scientific-disclosure metadata."""
    from neurosight.governance.trust_contract import build_governance_contract

    return build_governance_contract(
        runtime=_runtime_contract(),
        upload_limits=_upload_limits_payload(),
        allowed_origins=allowed_origins,
        rate_limiting_available=SLOWAPI_AVAILABLE,
    )


def _demo_readiness_payload() -> dict[str, Any]:
    """Return final launch-readiness metadata for the public/local demo."""
    from neurosight.governance.demo_readiness import build_demo_readiness_contract

    runtime = _runtime_contract()
    data = _data_status_payload(sample_limit=1)
    modalities = _modality_status_payload()
    checkpoint = _checkpoint_status_payload()
    xai = _xai_status_payload()
    governance = _governance_status_payload()
    capabilities = _api_capability_payload()
    return build_demo_readiness_contract(
        runtime=runtime,
        data=data,
        modalities=modalities,
        checkpoint=checkpoint,
        xai=xai,
        governance=governance,
        capabilities=capabilities,
    )


def _xai_method_contract(modality: str) -> dict[str, Any]:
    """Return the XAI method contract for one modality."""
    status = _xai_status_payload()
    for method in status.get("methods", []):
        if method.get("modality") == modality:
            return dict(method)
    return {}


def _find_patient_row(patient_id: str) -> Dict[str, Any]:
    """Load a patient row from the configured demo or ADNI-style metadata CSV."""
    csv_path = _patient_csv_path()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Patient metadata CSV not found at {csv_path}. "
            "Upload data to the Space or set NEUROSIGHT_PATIENT_CSV_PATH."
        )

    normalized_patient_id = str(patient_id).strip()
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rid = str(row.get("RID", "")).strip()
            if rid.endswith(".0") and rid[:-2].isdigit():
                rid = rid[:-2]
            if rid == normalized_patient_id:
                return dict(row)

    raise KeyError(f"Patient ID {normalized_patient_id!r} was not found in {csv_path}.")


def _cognitive_scores_from_patient_row(row: Dict[str, Any]) -> dict[str, float]:
    """Map ADNI-style CSV fields into the API cognitive score schema."""
    return _build_cognitive_vector(
        {
            "MMSE": row.get("MMSE"),
            "MOCA": row.get("MOCA"),
            "CDRSB": row.get("CDRSB"),
            "ADAS11": row.get("ADAS11"),
            "RAVLT_immediate": row.get("RAVLT_immediate"),
            "RAVLT_learning": row.get("RAVLT_learning"),
            "FAQ": row.get("FAQ"),
            "AGE": row.get("AGE"),
        }
    )


def _optional_patient_array(directory: Path, patient_id: str) -> Optional[np.ndarray]:
    """Load an optional Space-hosted `.npy` modality array by patient ID."""
    array_path = directory / f"{patient_id}.npy"
    if not array_path.exists():
        return None
    return np.load(array_path, allow_pickle=False).astype(np.float32)


def _build_diagnose_response(
    fusion_model: torch.nn.Module,
    mri_emb: Optional[torch.Tensor],
    eeg_emb: Optional[torch.Tensor],
    cog_emb: Optional[torch.Tensor],
    report_prefix: str,
) -> DiagnoseResponse:
    """Run fusion and format the shared diagnosis response contract."""
    diagnoses = list(Diagnosis)
    with torch.no_grad():
        out = fusion_model(mri=mri_emb, eeg=eeg_emb, cog=cog_emb)

    probs = out["probs"][0].tolist()
    pred_idx = int(out["probs"].argmax(dim=1).item())
    confidence = float(probs[pred_idx])
    diagnosis = diagnoses[pred_idx]

    model_service = _get_model_service()
    status_meta = model_service.get_status_metadata()

    notice = MODEL_STATUS_NOTICE if status_meta["model_mode"] == "demo_untrained" else CHECKPOINT_STATUS_NOTICE
    report_text = (
        f"{report_prefix}: {diagnosis.value} "
        f"(confidence: {confidence:.1%}). "
        f"{notice} {status_meta['disclaimer']} Human specialist review is required for every output."
    )

    return DiagnoseResponse(
        diagnosis=diagnosis,
        confidence=confidence,
        requires_review=True,
        report_text=report_text,
        model_mode=status_meta["model_mode"],
        checkpoint_id=status_meta["checkpoint_id"],
        trained_on_real_data=False,
        clinical_validated=False,
        requires_expert_review=True,
        disclaimer=status_meta["disclaimer"],
        warnings=[
            "Demo/research output only; not clinical software.",
            notice,
            "Outputs require expert review before any real-world interpretation.",
        ],
    )


def _load_checkpoint_into_app_state(app_instance: FastAPI, checkpoint_path: Path) -> dict[str, Any]:
    """Load a NeuroSight checkpoint into initialized app-state models."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary containing model state dictionaries.")

    required_keys = {"model_state"}
    missing_required = sorted(required_keys - set(checkpoint.keys()))
    if missing_required:
        raise ValueError(f"Checkpoint missing required keys: {', '.join(missing_required)}")

    if "mri_state" in checkpoint:
        app_instance.state.mri_model.load_state_dict(checkpoint["mri_state"])
    if "eeg_state" in checkpoint:
        app_instance.state.eeg_model.load_state_dict(checkpoint["eeg_state"])
    if "cog_state" in checkpoint:
        app_instance.state.cognitive_model.load_state_dict(checkpoint["cog_state"])
    app_instance.state.fusion_model.load_state_dict(checkpoint["model_state"])

    return {
        "path": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "val_auc": checkpoint.get("val_auc"),
    }


def _model_status_payload() -> dict[str, Any]:
    """Return current model-status payload with checkpoint load disclosure."""
    trained_checkpoint_loaded = bool(getattr(app.state, "trained_checkpoint_loaded", False))
    checkpoint_status = _checkpoint_status_payload()
    return {
        "mode": "checkpoint_loaded" if trained_checkpoint_loaded else MODEL_STATUS_MODE,
        "trained_checkpoint_loaded": trained_checkpoint_loaded,
        "notice": CHECKPOINT_STATUS_NOTICE if trained_checkpoint_loaded else MODEL_STATUS_NOTICE,
        "checkpoint": {
            "status": checkpoint_status["status"],
            "path": checkpoint_status["checkpoint"]["path"],
            "exists": checkpoint_status["checkpoint"]["exists"],
            "load_enabled": checkpoint_status["loading"]["enabled"],
            "load_error": checkpoint_status["loading"]["error"],
        },
    }


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    with start_span(
        "neurosight.http_request",
        {
            "http.method": request.method,
            "http.target": request.url.path,
            "neurosight.request_id": request_id,
        },
    ) as span:
        try:
            response = await call_next(request)
        except Exception as exc:
            span.record_exception(exc)
            raise

        process_time = time.time() - start_time
        span.set_attribute("http.status_code", response.status_code)
        span.set_attribute("neurosight.process_time_seconds", process_time)

        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        trace_id = current_trace_id()
        if trace_id:
            response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Observability"] = (
            "opentelemetry" if observability_status().get("available") else "request-id"
        )
        return response


async def verify_api_key(request: Request):
    app_env = os.environ.get("APP_ENV", "local").strip().lower()
    if app_env == "test":
        return  # skip auth in test mode

    expected = os.environ.get("NEUROSIGHT_API_KEY", "").strip()
    if not expected and app_env in {"local", "dev", "development"}:
        expected = "dev-key"
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="NEUROSIGHT_API_KEY must be configured before using protected endpoints.",
        )
    if request.headers.get("X-API-Key") != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@app.get("/healthz")
async def healthz():
    from neurosight.tracking.model_registry import ModelRegistry

    def _count_params(model: Optional[torch.nn.Module]) -> int:
        if model is None:
            return 0
        return int(sum(parameter.numel() for parameter in model.parameters()))

    mri_model = getattr(app.state, "mri_model", None)
    eeg_model = getattr(app.state, "eeg_model", None)
    cognitive_model = getattr(app.state, "cognitive_model", None)
    fusion_model = getattr(app.state, "fusion_model", None)
    kg = getattr(app.state, "kg", None)

    kg_nodes = len(kg.graph.nodes) if kg is not None and hasattr(kg, "graph") else 0
    kg_edges = len(kg.graph.edges) if kg is not None and hasattr(kg, "graph") else 0

    production_payload: Dict[str, Any] = {}
    try:
        production_model = ModelRegistry().get_production_model()
        if production_model:
            metrics = production_model.get("metrics", {})
            production_payload = {
                "run_id": production_model.get("run_id"),
                "val_auc": float(metrics.get("val_auc"))
                if isinstance(metrics, dict) and "val_auc" in metrics
                else None,
            }
    except (ValueError, TypeError):
        production_payload = {}

    started_at = float(getattr(app.state, "started_at", time.time()))
    uptime_seconds = max(0.0, time.time() - started_at)

    return {
        "status": "ok",
        "version": APP_VERSION,
        "runtime": _runtime_contract(),
        "capabilities": _api_capability_payload(),
        "data": _data_status_payload(sample_limit=0),
        "modalities": _modality_status_payload(),
        "checkpoint": _checkpoint_status_payload(),
        "xai": _xai_status_payload(),
        "governance": _governance_status_payload(),
        "demo_readiness": _demo_readiness_payload(),
        "models": {
            "mri_classifier": {
                "loaded": mri_model is not None,
                "params": _count_params(mri_model),
            },
            "eeg_classifier": {
                "loaded": eeg_model is not None,
                "params": _count_params(eeg_model),
            },
            "cognitive_classifier": {
                "loaded": cognitive_model is not None,
                "params": _count_params(cognitive_model),
            },
            "fusion": {
                "loaded": fusion_model is not None,
                "params": _count_params(fusion_model),
            },
        },
        "kg": {
            "loaded": bool(getattr(app.state, "kg_loaded", kg is not None)),
            "nodes": int(kg_nodes),
            "edges": int(kg_edges),
        },
        "model_status": _model_status_payload(),
        "upload_limits": _upload_limits_payload(),
        "observability": observability_status(),
        "production_model": production_payload,
        "uptime_seconds": float(round(uptime_seconds, 4)),
    }


@app.get("/")
async def root() -> Any:
    """Return a small landing payload for the Hugging Face Space App tab."""
    if _frontend_available():
        return FileResponse(FRONTEND_DIST_DIR / "index.html")
    return {
        "name": "NeuroSight Backend API",
        "status": "ok",
        "version": APP_VERSION,
        "runtime": _runtime_contract(),
        "capabilities": _api_capability_payload(),
        "data": _data_status_payload(sample_limit=0),
        "modalities": _modality_status_payload(),
        "checkpoint": _checkpoint_status_payload(),
        "xai": _xai_status_payload(),
        "governance": _governance_status_payload(),
        "demo_readiness": _demo_readiness_payload(),
        "observability": observability_status(),
        "health": "/healthz",
        "docs": "/docs",
        "data_status": "/v1/data/status",
        "demo_patients": "/v1/data/demo-patients",
        "modalities_status": "/v1/modalities/status",
        "checkpoint_status": "/v1/models/checkpoint/status",
        "evaluation_report": "/v1/eval/report",
        "xai_status": "/v1/xai/status",
        "governance_status": "/v1/governance/status",
        "demo_readiness_status": "/v1/demo/readiness",
        "risk_profile": "/v1/risk-profile",
        "patient_risk_profile": "/v1/risk-profile/patient/{patient_id}",
        "streaming_risk_profile": "/v1/risk-profile/stream",
        "legacy_diagnose": "/v1/diagnose",
        "legacy_patient_diagnose": "/v1/diagnose/patient/{patient_id}",
        "legacy_streaming_diagnose": "/v1/diagnose/stream",
        "uploads": {
            "mri": "/v1/upload/mri",
            "eeg": "/v1/upload/eeg",
            "cognitive": "/v1/upload/cognitive",
        },
        "knowledge_graph": "/v1/kg/query",
        "evaluation": "/v1/eval/*",
        "models": "/v1/models",
        "xai": "/v1/xai/{patient_id}",
        "notice": MODEL_STATUS_NOTICE,
    }


@app.get("/v1/data/status")
async def data_status(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    """Return demo/ADNI-style data readiness without exposing private records."""
    _ = auth
    return _data_status_payload(sample_limit=0)


@app.get("/v1/data/demo-patients")
async def demo_patients(
    limit: int = 12,
    auth: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Return a bounded list of safe demo patient rows for the local UI."""
    _ = auth
    safe_limit = max(1, min(int(limit), 50))
    payload = _data_status_payload(sample_limit=safe_limit)
    return {
        "status": payload["status"],
        "source_kind": payload["source_kind"],
        "privacy": payload["privacy"],
        "recommended_patient_id": payload["recommended_patient_id"],
        "count": len(payload["samples"]),
        "patients": payload["samples"],
    }


@app.get("/v1/modalities/status")
async def modalities_status(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    """Return MRI/EEG/cognitive preprocessing and upload readiness metadata."""
    _ = auth
    return _modality_status_payload()


@app.get("/v1/governance/status")
async def governance_status(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    """Return privacy, security, and scientific-disclosure metadata."""
    _ = auth
    return _governance_status_payload()


@app.get("/v1/demo/readiness")
async def demo_readiness(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    """Return final public-demo launch readiness and recommended demo flow."""
    _ = auth
    return _demo_readiness_payload()


@app.post("/v1/diagnose", response_model=DiagnoseResponse, deprecated=True)
@app.post("/v1/risk-profile", response_model=DiagnoseResponse)
async def diagnose(req: dict, auth: None = Depends(verify_api_key)):
    """Run synthetic risk profiling on input modalities.

    This endpoint uses PyTorch fusion models to perform pattern matching on
    simulated or research-demo inputs. `/v1/risk-profile` is the preferred
    route. `/v1/diagnose` is retained as legacy naming for backward
    compatibility. Not intended for clinical or diagnostic use.
    """
    _ = auth
    import torch
    with start_span(
        "neurosight.diagnose",
        {
            "neurosight.has_mri_embedding": bool(req.get("mri_embedding")),
            "neurosight.has_eeg_embedding": bool(req.get("eeg_embedding")),
            "neurosight.has_cog_embedding": bool(req.get("cog_embedding")),
            "neurosight.has_cognitive_scores": isinstance(req.get("cognitive_scores"), dict),
        },
    ) as span:
        fusion_model = _get_fusion_model()

        # Extract embeddings from request if provided, else use None
        # (None = missing modality → fusion uses learned mask tokens)
        mri_emb = None
        eeg_emb = None
        cog_emb = None

        if req.get("mri_embedding"):
            mri_emb = torch.tensor([req["mri_embedding"]], dtype=torch.float32)  # (1, 768)
        if req.get("eeg_embedding"):
            eeg_emb = torch.tensor([req["eeg_embedding"]], dtype=torch.float32)  # (1, 256)
        if req.get("cog_embedding"):
            cog_emb = torch.tensor([req["cog_embedding"]], dtype=torch.float32)  # (1, 64)

        # If no embeddings provided at all, use cognitive scores as a simple proxy
        if mri_emb is None and eeg_emb is None and cog_emb is None:
            cog_scores = req.get("cognitive_scores", {})
            try:
                schema = CognitiveSchema.model_validate(cog_scores)
            except Exception as exc:
                raise HTTPException(status_code=422, detail=_cognitive_validation_detail(exc))
            model_service = _get_model_service()
            cognitive_tensor = model_service.preprocess_cognitive(schema)
            with torch.no_grad():
                _, cog_emb = model_service.cognitive_model(cognitive_tensor)

        response = _build_diagnose_response(
            fusion_model=fusion_model,
            mri_emb=mri_emb,
            eeg_emb=eeg_emb,
            cog_emb=cog_emb,
            report_prefix="Multimodal fusion non-clinical risk-profile demo",
        )
        span.set_attribute("neurosight.diagnosis", response.diagnosis.value)
        span.set_attribute("neurosight.confidence", response.confidence)
        return response


@app.post("/v1/diagnose/patient/{patient_id}", response_model=DiagnoseResponse, deprecated=True)
@app.post("/v1/risk-profile/patient/{patient_id}", response_model=DiagnoseResponse)
async def diagnose_patient(
    patient_id: str,
    req: dict | None = None,
    auth: None = Depends(verify_api_key),
):
    """Run synthetic risk profiling for a patient from configured demo or ADNI-style data.

    `/v1/risk-profile/patient/{patient_id}` is preferred. The `/v1/diagnose`
    route is retained as legacy naming. Not intended for diagnostic or clinical
    decision support. The endpoint expects:
    - metadata CSV: `NEUROSIGHT_PATIENT_CSV_PATH`, demo fallback, or `data/ADNIMERGE.csv`
    - optional MRI volume: `NEUROSIGHT_MRI_DIR/{RID}.npy` or `data/mri/{RID}.npy`
    - optional EEG array: `NEUROSIGHT_EEG_DIR/{RID}.npy` or `data/eeg/{RID}.npy`
    """
    _ = auth
    _ = req or {}

    normalized_patient_id = str(patient_id).strip()
    if not normalized_patient_id:
        raise HTTPException(status_code=422, detail="patient_id is required.")

    try:
        row = await asyncio.to_thread(_find_patient_row, normalized_patient_id)
        cognitive_scores = _cognitive_scores_from_patient_row(row)
        cognitive_tensor = _tensor_from_cognitive_features(cognitive_scores)

        with torch.no_grad():
            _, cog_emb = _get_cognitive_model()(cognitive_tensor)

        mri_emb: Optional[torch.Tensor] = None
        eeg_emb: Optional[torch.Tensor] = None

        mri_array = await asyncio.to_thread(
            _optional_patient_array,
            _patient_mri_dir(),
            normalized_patient_id,
        )
        if mri_array is not None:
            _validate_array_size(mri_array, "MRI array", MAX_MRI_VALUES)
            mri_tensor = await asyncio.to_thread(_prepare_mri_tensor, mri_array)
            with torch.no_grad():
                mri_emb = await asyncio.to_thread(_get_mri_model().encoder, mri_tensor)

        eeg_array = await asyncio.to_thread(
            _optional_patient_array,
            _patient_eeg_dir(),
            normalized_patient_id,
        )
        if eeg_array is not None:
            _validate_array_size(eeg_array, "EEG array", MAX_EEG_VALUES)
            eeg_tensor = _prepare_eeg_tensor(eeg_array)
            with torch.no_grad():
                eeg_emb = _get_eeg_model().encoder(eeg_tensor)

    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UploadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    modality_status = [
        "cognitive=loaded",
        f"mri={'loaded' if mri_emb is not None else 'missing'}",
        f"eeg={'loaded' if eeg_emb is not None else 'missing'}",
    ]
    return _build_diagnose_response(
        fusion_model=_get_fusion_model(),
        mri_emb=mri_emb,
        eeg_emb=eeg_emb,
        cog_emb=cog_emb,
        report_prefix=(
            f"Configured dataset patient {normalized_patient_id} demo inference "
            f"({', '.join(modality_status)})"
        ),
    )


@app.post("/v1/diagnose/stream", deprecated=True)
@app.post("/v1/risk-profile/stream")
async def diagnose_stream(req: dict, auth: None = Depends(verify_api_key)):
    """Stream multi-agent synthetic evaluation and risk profiling events.

    `/v1/risk-profile/stream` is preferred. `/v1/diagnose/stream` is retained
    as legacy naming. This endpoint utilizes LangGraph agent orchestration to
    coordinate cognitive, MRI, and EEG analyst subagents. All outputs are
    non-clinical research demo outputs.
    """
    _ = auth

    async def event_generator():
        from knowledge_graph import MockPatientRecord
        from neurosight.agents.orchestrator import (
            build_diagnosis_graph,
            build_initial_state,
            build_report_from_state,
        )

        patient_id = str(req.get("patient_id") or f"REQ_{uuid.uuid4().hex[:8]}")
        patient = MockPatientRecord(patient_id)
        raw_scores = req.get("cognitive_scores")
        if raw_scores and isinstance(raw_scores, dict):
            merged_scores = _build_cognitive_vector(raw_scores)
            patient.cognitive = dict(merged_scores)

        query = req.get("query", "What should the research/demo workflow inspect?")
        graph = build_diagnosis_graph(None, getattr(app.state, "kg", None))
        initial_state = build_initial_state(patient, query)
        final_state: Dict[str, Any] | None = None

        for event in graph.stream(initial_state):
            for agent_name, state in event.items():
                yield f"data: {json.dumps({'agent': agent_name, 'status': 'running'})}\n\n"
                await asyncio.sleep(0.01)
                completed_payload = {
                    "agent": agent_name,
                    "status": "completed",
                    "next_agent": state.get("next_agent"),
                    "iteration_count": state.get("iteration_count"),
                    "requires_review": bool(state.get("requires_review", False)),
                    "safety_flags": state.get("safety_flags", []),
                }
                yield f"data: {json.dumps(completed_payload)}\n\n"
                final_state = state
                await asyncio.sleep(0.01)

        if final_state is None:
            raise RuntimeError("LangGraph stream produced no events.")

        report = build_report_from_state(final_state)
        status_meta = _get_model_service().get_status_metadata()

        result_payload = {
            "agent": "complete",
            "status": "done",
            "diagnosis": report.final_diagnosis.value,
            "confidence": float(report.confidence),
            "requires_review": bool(report.requires_review),
            "blocked": bool(report.blocked_by_safety),
            "report_text": str(report.report_text)[:500],
            "model_mode": status_meta["model_mode"],
            "checkpoint_id": status_meta["checkpoint_id"],
            "trained_on_real_data": False,
            "clinical_validated": False,
            "requires_expert_review": True,
            "disclaimer": status_meta["disclaimer"],
            "warnings": [
                "Demo/research output only; not clinical software.",
                "Outputs require expert review before interpretation.",
            ],
        }
        yield f"data: {json.dumps(result_payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/v1/upload/mri")
@limiter.limit("10/minute")
async def upload_mri(
    request: Request,
    file: UploadFile = File(...),
    auth: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Upload MRI volume and return encoder embedding.

    Args:
        request: FastAPI request object required for rate limiting.
        file: Uploaded MRI file (`.npy`, `.nii`, `.nii.gz`, or DICOM `.zip`).

    Returns:
        JSON payload with embedding vector and dimension.
    """
    _ = request
    _ = auth

    with start_span(
        "neurosight.upload_mri",
        {
            "neurosight.upload.filename": file.filename or "",
            "neurosight.upload.content_type": file.content_type or "",
        },
    ) as span:
        try:
            _validate_filename(file.filename)
            mri_array = await asyncio.to_thread(_load_mri_array_from_upload, file)
            span.set_attribute("neurosight.mri.raw_shape", str(tuple(mri_array.shape)))
            from neurosight.data.modality_contract import inspect_mri_array

            preprocessing_payload = inspect_mri_array(mri_array, file.filename)
            mri_tensor = await asyncio.to_thread(_prepare_mri_tensor, mri_array)
            preprocessing_payload["prepared_tensor_shape"] = [
                int(value) for value in tuple(mri_tensor.shape)
            ]
            span.set_attribute("neurosight.mri.tensor_shape", str(tuple(mri_tensor.shape)))
            with torch.no_grad():
                embedding = await asyncio.to_thread(_get_mri_model().encoder, mri_tensor)
            payload = _extract_embedding_payload(embedding, expected_dim=768)
            span.set_attribute("neurosight.embedding_dim", payload["embedding_dim"])
        except UploadTooLarge as exc:
            span.record_exception(exc)
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ValueError as exc:
            span.record_exception(exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            file.file.close()

    return {
        "status": "ok",
        "embedding_dim": payload["embedding_dim"],
        "embedding": payload["embedding"],
        "preprocessing": preprocessing_payload,
    }


@app.post("/v1/upload/eeg")
@limiter.limit("10/minute")
async def upload_eeg(
    request: Request,
    file: UploadFile = File(...),
    auth: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Upload EEG data and return encoder embedding.

    Args:
        request: FastAPI request object required for rate limiting.
        file: Uploaded `.npy` or `.edf` EEG file.

    Returns:
        JSON payload with embedding vector and dimension.
    """
    _ = request
    _ = auth
    from neurosight.models.eeg import preprocess_eeg
    from neurosight.data.modality_contract import inspect_eeg_array

    if not file.filename:
        raise HTTPException(status_code=422, detail="EEG upload requires a filename.")

    suffix = os.path.splitext(file.filename.lower())[1]
    if suffix not in {".npy", ".edf"}:
        raise HTTPException(
            status_code=422,
            detail="EEG upload expects a .npy or .edf file.",
        )

    with start_span(
        "neurosight.upload_eeg",
        {
            "neurosight.upload.filename": file.filename or "",
            "neurosight.upload.content_type": file.content_type or "",
            "neurosight.upload.suffix": suffix,
        },
    ) as span:
        try:
            _validate_filename(file.filename)
            if suffix == ".npy":
                eeg_array = _load_numpy_from_upload(file)
            else:
                raw_bytes = _read_upload_bytes(file)
                if not raw_bytes:
                    raise ValueError("Uploaded EEG file is empty.")
                with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as raw_tmp:
                    raw_tmp.write(raw_bytes)
                    raw_path = raw_tmp.name
                with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as out_tmp:
                    out_path = out_tmp.name
                try:
                    eeg_array = preprocess_eeg(raw_path, out_path)
                finally:
                    try:
                        os.remove(raw_path)
                    except OSError as remove_error:
                        _ = remove_error
                    try:
                        os.remove(out_path)
                    except OSError as remove_error:
                        _ = remove_error

            span.set_attribute("neurosight.eeg.raw_shape", str(tuple(eeg_array.shape)))
            _validate_array_size(eeg_array, "EEG array", MAX_EEG_VALUES)
            preprocessing_payload = inspect_eeg_array(eeg_array, file.filename)
            eeg_tensor = _prepare_eeg_tensor(eeg_array)
            preprocessing_payload["prepared_tensor_shape"] = [
                int(value) for value in tuple(eeg_tensor.shape)
            ]
            span.set_attribute("neurosight.eeg.tensor_shape", str(tuple(eeg_tensor.shape)))
            with torch.no_grad():
                embedding = _get_eeg_model().encoder(eeg_tensor)
            payload = _extract_embedding_payload(embedding, expected_dim=256)
            span.set_attribute("neurosight.embedding_dim", payload["embedding_dim"])
        except UploadTooLarge as exc:
            span.record_exception(exc)
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ValueError as exc:
            span.record_exception(exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            file.file.close()

    return {
        "status": "ok",
        "embedding_dim": payload["embedding_dim"],
        "embedding": payload["embedding"],
        "preprocessing": preprocessing_payload,
    }


@app.post("/v1/upload/cognitive")
@limiter.limit("10/minute")
async def upload_cognitive(
    request: Request,
    req: dict,
    auth: None = Depends(verify_api_key),
):
    _ = request
    _ = auth

    if not isinstance(req, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object.")

    input_scores = req.get("scores", req)
    if not isinstance(input_scores, dict):
        raise HTTPException(status_code=422, detail="`scores` must be an object when provided.")

    try:
        schema = CognitiveSchema.model_validate(input_scores)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=_cognitive_validation_detail(exc))

    model_service = _get_model_service()
    cognitive_tensor = model_service.preprocess_cognitive(schema)

    with torch.no_grad():
        logits, embedding_tensor = model_service.cognitive_model(cognitive_tensor)
        probabilities = torch.softmax(logits, dim=-1)[0]

    diagnoses = list(Diagnosis)
    probability_values = probabilities.detach().cpu().numpy().astype(np.float32)
    unimodal_probs = {
        diagnoses[index].value: float(probability_values[index])
        for index in range(len(diagnoses))
    }
    top_prediction_index = int(probability_values.argmax())
    top_prediction = diagnoses[top_prediction_index].value
    confidence = float(probability_values[top_prediction_index])

    embedding_payload = _extract_embedding_payload(embedding_tensor, expected_dim=64)
    status_meta = model_service.get_status_metadata()
    return {
        "status": "ok",
        "embedding_dim": embedding_payload["embedding_dim"],
        "embedding": embedding_payload["embedding"],
        "unimodal_probs": unimodal_probs,
        "top_prediction": top_prediction,
        "confidence": confidence,
        "note": "Unimodal prediction only. Use /v1/risk-profile for multimodal fusion. /v1/diagnose remains as legacy naming.",
        "model_mode": status_meta["model_mode"],
        "checkpoint_id": status_meta["checkpoint_id"],
        "disclaimer": status_meta["disclaimer"],
    }


@app.post("/v1/kg/query")
async def kg_query(req: dict, auth: None = Depends(verify_api_key)):
    _ = auth
    kg = _get_kg()
    patient_id = req.get("patient_id")
    query_type = req.get("query_type", "history")  # history | similar | snapshot | progression
    target_date = req.get("target_date")

    if not patient_id:
        raise HTTPException(status_code=422, detail="patient_id is required")

    if query_type == "history":
        results = kg.get_patient_history(patient_id, before_date=target_date)
        return {"patient_id": patient_id, "query_type": query_type,
                "results": results, "count": len(results)}

    elif query_type == "similar":
        top_k = int(req.get("top_k", 5))
        similar = kg.find_similar_patients(patient_id, top_k=top_k)
        results = [{"patient_id": s.patient_id, "similarity_score": s.similarity_score,
                     "shared_features": s.shared_features} for s in similar]
        return {"patient_id": patient_id, "query_type": query_type,
                "results": results, "count": len(results)}

    elif query_type == "snapshot":
        if not target_date:
            raise HTTPException(status_code=422,
                                detail="target_date required for snapshot query")
        snapshot = kg.query_at_date(patient_id, target_date)
        return {"patient_id": patient_id, "query_type": query_type,
                "target_date": target_date, "results": snapshot}

    elif query_type == "progression":
        progression = kg.get_disease_progression(patient_id)
        return {"patient_id": patient_id, "query_type": query_type,
                "results": progression, "count": len(progression)}

    else:
        raise HTTPException(status_code=422,
                            detail=f"Unknown query_type: {query_type}. "
                                   f"Use: history | similar | snapshot | progression")


@app.get("/v1/kg/patient/{patient_id}/history")
async def get_history(
    patient_id: str,
    auth: None = Depends(verify_api_key),
):
    _ = auth
    kg = _get_kg()
    history = kg.get_patient_history(patient_id)
    return {"patient_id": patient_id, "history": history, "count": len(history)}


@app.get("/v1/kg/patient/{patient_id}/similar")
async def get_similar(
    patient_id: str,
    auth: None = Depends(verify_api_key),
):
    _ = auth
    kg = _get_kg()
    results = kg.find_similar_patients(patient_id, top_k=5)
    return [{"patient_id": r.patient_id, "score": r.similarity_score} for r in results]


@app.get("/v1/eval/metrics")
async def eval_metrics(auth: None = Depends(verify_api_key)):
    _ = auth
    import pathlib
    log_path = pathlib.Path("logs/mlflow_fallback.jsonl")
    if log_path.exists():
        with open(log_path) as f:
            lines = f.readlines()
        if lines:
            latest = json.loads(lines[-1])
            return latest.get("metrics", latest)
    return {"status": "no evaluations run yet"}


@app.get("/v1/eval/history")
async def eval_history(auth: None = Depends(verify_api_key)) -> List[Dict[str, Any]]:
    """Return the latest evaluation runs from fallback MLflow JSONL logs.

    Args:
        None.

    Returns:
        List containing up to the last 10 evaluation entries with
        `timestamp`, `metrics`, and `checkpoint` fields.
    """
    _ = auth
    import pathlib

    log_path = pathlib.Path("logs/mlflow_fallback.jsonl")
    if not log_path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for raw in lines[-10:]:
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue

        records.append(
            {
                "timestamp": entry.get("timestamp"),
                "metrics": entry.get("metrics", {}),
                "checkpoint": entry.get("model_checkpoint"),
            }
        )

    return records


@app.get("/v1/eval/report")
async def eval_report(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    """Return persisted evaluation/model-card reporting status."""
    _ = auth
    checkpoint_status = _checkpoint_status_payload()
    return {
        "status": "ok",
        "checkpoint": checkpoint_status["checkpoint"],
        "evaluation": checkpoint_status["evaluation"],
        "model_card": checkpoint_status["model_card"],
        "scientific_claims": checkpoint_status["scientific_claims"],
    }


@app.post("/v1/eval/run")
async def run_eval(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    """Run honest baseline evaluation on random synthetic multimodal data.

    Args:
        None.

    Returns:
        Evaluation payload with baseline metrics and modality ablation results.
    """
    _ = auth
    import torch
    import numpy as np
    from datetime import datetime
    from evaluation.metrics import (
        compute_auc_roc, compute_ece,
        compute_per_class_metrics, compute_modality_ablation,
        EvaluationReport, log_to_mlflow,
    )

    fusion_model = _get_fusion_model()
    fusion_model.eval()
    diagnoses = list(Diagnosis)
    n_classes = len(diagnoses)

    # Honest synthetic baseline: random features and balanced labels.
    # This intentionally contains no label signal and reflects untrained behavior.
    samples_per_class = 10
    n_samples = n_classes * samples_per_class
    torch.manual_seed(42)
    np.random.seed(42)

    all_probs = []
    all_preds = []
    all_labels = []
    ablation_data: List[Dict[str, Any]] = []

    with torch.no_grad():
        for i in range(n_samples):
            true_class = i % n_classes
            mri_emb = torch.randn(1, 768)
            eeg_emb = torch.randn(1, 256)
            cog_emb = torch.randn(1, 64)

            out = fusion_model(mri=mri_emb, eeg=eeg_emb, cog=cog_emb)
            probs = out["probs"][0].numpy()
            pred = int(probs.argmax())

            all_probs.append(probs)
            all_preds.append(pred)
            all_labels.append(true_class)
            ablation_data.append(
                {
                    "mri": mri_emb,
                    "eeg": eeg_emb,
                    "cog": cog_emb,
                    "label": true_class,
                }
            )

    y_prob = np.array(all_probs)   # (n_samples, 6)
    y_pred = np.array(all_preds)   # (n_samples,)
    y_true = np.array(all_labels)  # (n_samples,)

    auc_results = compute_auc_roc(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)
    per_class = compute_per_class_metrics(y_true, y_pred)

    metrics = {
        "auc_macro":  auc_results.get("macro", float("nan")),
        "ece":        ece,
        "accuracy":   per_class["accuracy"],
        "macro_f1":   per_class["macro_f1"],
        "n_samples":  float(n_samples),
    }
    metrics.update({f"auc_{k}": v for k, v in auc_results.items() if k != "macro"})
    ablation = compute_modality_ablation(fusion_model, ablation_data)
    ablation_summary = {
        "all": float(ablation.get("all", float("nan"))),
        "no_mri": float(ablation.get("no_mri", float("nan"))),
        "no_eeg": float(ablation.get("no_eeg", float("nan"))),
        "no_cognitive": float(ablation.get("no_cognitive", float("nan"))),
    }

    report = EvaluationReport(
        metrics=metrics,
        timestamp=datetime.utcnow().isoformat(),
        model_checkpoint="baseline (untrained model, random initialization)",
    )
    log_to_mlflow(report, experiment_name="neurosight_eval")

    return {
        "status": "evaluation complete",
        "evaluation_type": "baseline (untrained model)",
        "n_samples": n_samples,
        "metrics": metrics,
        "ablation": ablation_summary,
        "note": (
            "These are baseline metrics on synthetic data with randomly initialized "
            "weights. For real performance, train on ADNI/OASIS-3 datasets."
        ),
    }


@app.post("/v1/eval/cv")
async def run_cv_eval(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    """Run 5-fold cross-validation on synthetic data.

    NOTE: This runs in the background and may take several minutes on
    real hardware. On synthetic data with small models it completes quickly.
    """
    _ = auth
    from evaluation.cross_validation import run_kfold_cv

    cv_config: Dict[str, Any] = {
        "seed": 42,
        "csv_path": _evaluation_csv_path(),
        "cv_epochs": 3,
        "batch_size": 16,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "n_classes": 6,
        "split_seed": 42,
    }

    try:
        cv_results = await asyncio.to_thread(run_kfold_cv, cv_config, 5)
    except (ModuleNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Cross-validation failed: {exc}") from exc

    return {"status": "ok", "cv_results": cv_results}


@app.post("/v1/eval/benchmark")
async def run_benchmark_eval(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    """Run full baseline comparison benchmark on structured synthetic data.

    Returns:
        Payload containing benchmark metrics, markdown table, and winner entry.
    """
    _ = auth
    from evaluation.benchmark import run_benchmark
    from evaluation.benchmark_table import render_comparison_table

    benchmark_seed = 42
    csv_path = _evaluation_csv_path()
    try:
        benchmark_results = await asyncio.to_thread(
            run_benchmark,
            csv_path,
            benchmark_seed,
        )
        comparison_table = render_comparison_table(benchmark_results)
    except (ModuleNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Benchmark evaluation failed: {exc}") from exc

    return {
        "status": "ok",
        "results": benchmark_results.get("results", []),
        "winner": benchmark_results.get("winner", {}),
        "comparison_table": comparison_table,
        "note": "Benchmark executes on structured synthetic data and may take 30-60 seconds.",
    }


@app.get("/v1/models")
async def list_models(auth: None = Depends(verify_api_key)) -> List[Dict[str, Any]]:
    """List all registered model versions from the model registry."""
    _ = auth
    from neurosight.tracking.model_registry import ModelRegistry

    registry = ModelRegistry()
    return registry.list_runs()


@app.get("/v1/models/production")
async def get_production_model(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    """Return metadata of the current production model."""
    _ = auth
    from neurosight.tracking.model_registry import ModelRegistry

    registry = ModelRegistry()
    production = registry.get_production_model()
    if production:
        return production
    return {
        "status": "no_production_model",
        "message": "No production model is currently registered.",
    }


@app.get("/v1/models/checkpoint/status")
async def checkpoint_status(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    """Return checkpoint availability, registry, and runtime load status."""
    _ = auth
    return _checkpoint_status_payload()


@app.post("/v1/models/{run_id}/promote")
async def promote_model(
    run_id: str,
    auth: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Promote a model run to production status."""
    _ = auth
    from neurosight.tracking.model_registry import ModelRegistry

    registry = ModelRegistry()
    try:
        promoted = registry.promote_to_production(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "status": "ok",
        "production_model": promoted,
    }


@app.get("/v1/xai/status")
async def xai_status(auth: None = Depends(verify_api_key)) -> Dict[str, Any]:
    """Return explainability method availability and interpretation policy."""
    _ = auth
    return _xai_status_payload()


@app.get("/v1/xai/{patient_id}")
async def get_xai(
    patient_id: str,
    modality: str = "cognitive",
    target_class: Optional[int] = None,
    auth: None = Depends(verify_api_key),
):
    _ = auth
    from neurosight.models.xai import SHAPExplainer, XAIEngine

    modality_key = modality.strip().lower()
    xai_status_payload = _xai_status_payload()
    method_contract = _xai_method_contract(modality_key)
    if modality_key in {"mri", "eeg"}:
        return {
            "patient_id": patient_id,
            "modality": modality_key,
            "method": method_contract.get("method", "runtime_limited"),
            "feature_importance": {},
            "text_summary": "",
            "xai_available": False,
            "method_contract": method_contract,
            "interpretation_policy": xai_status_payload["interpretation_policy"],
            "privacy": {
                "patient_data_persisted_by_xai_endpoint": False,
                "clinical_use_allowed": False,
            },
            "note": (
                "Upload MRI/EEG via /v1/upload/{mri,eeg} first, then pass the embedding "
                "to /v1/risk-profile"
            ),
        }

    if modality_key != "cognitive":
        raise HTTPException(
            status_code=422,
            detail="modality must be one of: mri, eeg, cognitive.",
        )

    diagnoses = list(Diagnosis)
    cognitive_model = _get_cognitive_model()

    kg = getattr(app.state, "kg", None)
    cognitive_scores, has_history = _synthesize_cognitive_scores_from_kg(kg, patient_id)
    cognitive_tensor = _tensor_from_cognitive_features(cognitive_scores)

    with torch.no_grad():
        _, cognitive_embedding = cognitive_model(cognitive_tensor)
        fusion_out = _get_fusion_model()(mri=None, eeg=None, cog=cognitive_embedding)
        inferred_target_class = int(fusion_out["probs"].argmax(dim=1).item())

    resolved_target_class = inferred_target_class if target_class is None else int(target_class)
    if resolved_target_class < 0 or resolved_target_class >= len(diagnoses):
        raise HTTPException(
            status_code=422,
            detail=f"target_class must be in range [0, {len(diagnoses) - 1}].",
        )

    xai_engine = XAIEngine(cognitive_model=cognitive_model)
    explanation = xai_engine.explain_cognitive(
        cognitive_tensor,
        target_class=resolved_target_class,
        cognitive_model=cognitive_model,
    )
    raw_importance = (
        explanation.saliency if isinstance(explanation.saliency, dict) else {}
    )
    feature_importance = {
        feature_name: float(raw_importance.get(feature_name, 0.0))
        for feature_name in SHAPExplainer.FEATURE_NAMES
    }
    diagnosis_label = diagnoses[resolved_target_class].value
    summary = _cognitive_summary(
        feature_importance=feature_importance,
        diagnosis_label=diagnosis_label,
        from_history=has_history,
    )

    return {
        "patient_id": patient_id,
        "modality": "cognitive",
        "method": "gradient_x_input",
        "target_class": resolved_target_class,
        "target_label": diagnosis_label,
        "input_source": "kg_history" if has_history else "default_cognitive_profile",
        "feature_importance": feature_importance,
        "text_summary": summary,
        "xai_available": True,
        "method_contract": method_contract,
        "interpretation_policy": xai_status_payload["interpretation_policy"],
        "privacy": {
            "patient_data_persisted_by_xai_endpoint": False,
            "clinical_use_allowed": False,
        },
    }


@app.get("/{frontend_path:path}", include_in_schema=False)
async def serve_frontend(frontend_path: str) -> Any:
    """Serve the static Next.js export when bundled into the Docker Space."""
    if not _frontend_available():
        raise HTTPException(status_code=404, detail="Frontend build is not bundled.")
    first_segment = frontend_path.split("/", 1)[0]
    if first_segment in {"v1", "docs", "redoc", "openapi.json", "healthz"}:
        raise HTTPException(status_code=404, detail="Route not found.")
    requested = _frontend_file(frontend_path)
    if requested.exists() and requested.is_file():
        return FileResponse(requested)
    return FileResponse(FRONTEND_DIST_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
