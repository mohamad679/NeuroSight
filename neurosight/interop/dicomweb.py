"""DICOM and DICOMweb awareness helpers for NeuroSight.

This module does not implement a DICOMweb server. It documents the mapping
between NeuroSight's current DICOM ZIP ingestion and a future standards-aligned
DICOMweb boundary.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DICOMWEB_BASE_PATH = "/dicomweb"
NEUROSIGHT_MRI_UPLOAD_PATH = "/v1/upload/mri"
DICOM_STANDARD_BASE = "https://www.dicomstandard.org/using/dicomweb"

SAFE_DICOM_FIELDS: tuple[str, ...] = (
    "Modality",
    "SOPClassUID",
    "TransferSyntaxUID",
    "Rows",
    "Columns",
    "NumberOfFrames",
    "PixelSpacing",
    "SliceThickness",
    "InstanceNumber",
    "ImagePositionPatient",
)

IDENTIFIER_FIELDS: tuple[str, ...] = (
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
)

DICOMWEB_SERVICES: tuple[dict[str, Any], ...] = (
    {
        "service": "QIDO-RS",
        "name": "Query based on ID for DICOM Objects",
        "method": "GET",
        "scope": "Query studies, series, and instances by metadata without moving pixels.",
        "standard": "DICOM PS3.18 10.6",
        "routes": [
            f"{DICOMWEB_BASE_PATH}/studies?PatientID=...",
            f"{DICOMWEB_BASE_PATH}/studies/{{StudyInstanceUID}}/series",
            f"{DICOMWEB_BASE_PATH}/studies/{{StudyInstanceUID}}/series/{{SeriesInstanceUID}}/instances",
        ],
        "neurosight_status": "roadmap_not_implemented",
    },
    {
        "service": "WADO-RS",
        "name": "Web Access to DICOM Objects",
        "method": "GET",
        "scope": "Retrieve DICOM instances, frames, rendered images, or metadata.",
        "standard": "DICOM PS3.18 10.4",
        "routes": [
            f"{DICOMWEB_BASE_PATH}/studies/{{StudyInstanceUID}}",
            f"{DICOMWEB_BASE_PATH}/studies/{{StudyInstanceUID}}/metadata",
            f"{DICOMWEB_BASE_PATH}/studies/{{StudyInstanceUID}}/series/{{SeriesInstanceUID}}/instances/{{SOPInstanceUID}}",
        ],
        "neurosight_status": "roadmap_not_implemented",
    },
    {
        "service": "STOW-RS",
        "name": "STore Over the Web",
        "method": "POST",
        "scope": "Store DICOM instances through HTTP multipart payloads.",
        "standard": "DICOM PS3.18 10.5",
        "routes": [
            f"{DICOMWEB_BASE_PATH}/studies",
            f"{DICOMWEB_BASE_PATH}/studies/{{StudyInstanceUID}}",
        ],
        "neurosight_status": "roadmap_not_implemented",
    },
    {
        "service": "RS Capabilities",
        "name": "Retrieve Capabilities",
        "method": "OPTIONS",
        "scope": "Expose supported DICOMweb resources, media types, and methods.",
        "standard": "DICOM PS3.18 8.9",
        "routes": [
            DICOMWEB_BASE_PATH,
            f"{DICOMWEB_BASE_PATH}/studies",
        ],
        "neurosight_status": "roadmap_not_implemented",
    },
)


class DicomInspectionError(Exception):
    """Raised when DICOM inspection cannot be completed."""


def utc_now() -> str:
    """Return current UTC time as an ISO-8601 instant."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(value: object, *, length: int = 16) -> str | None:
    """Return a short deterministic hash for potentially identifying values."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _import_pydicom() -> Any:
    try:
        import pydicom
    except ModuleNotFoundError as exc:
        raise DicomInspectionError(
            "pydicom is required to inspect DICOM files. Install project dependencies first."
        ) from exc
    return pydicom


def _safe_get(dataset: Any, keyword: str) -> Any:
    if keyword == "TransferSyntaxUID":
        file_meta = getattr(dataset, "file_meta", None)
        return getattr(file_meta, "TransferSyntaxUID", None) if file_meta is not None else None
    return getattr(dataset, keyword, None)


def _serialize_dicom_value(value: Any) -> Any:
    """Convert common pydicom values into JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, Iterable):
        values = list(value)
        if len(values) > 8:
            return [str(item) for item in values[:8]] + ["..."]
        return [str(item) for item in values]
    return str(value)


def dataset_to_safe_record(
    dataset: Any,
    *,
    source_name: str,
    include_uids: bool = False,
) -> dict[str, Any]:
    """Convert a pydicom Dataset into a PHI-minimized metadata record."""
    identifiers: dict[str, Any] = {}
    for keyword in IDENTIFIER_FIELDS:
        value = _safe_get(dataset, keyword)
        identifiers[f"{keyword}_hash"] = stable_hash(value)
        if include_uids and keyword != "PatientID" and value is not None:
            identifiers[keyword] = str(value)

    metadata: dict[str, Any] = {}
    for keyword in SAFE_DICOM_FIELDS:
        value = _safe_get(dataset, keyword)
        serialized = _serialize_dicom_value(value)
        if serialized is not None:
            metadata[keyword] = serialized

    return {
        "source": source_name,
        "identifiers": identifiers,
        "metadata": metadata,
        "redactions": {
            "PatientName": "omitted",
            "PatientBirthDate": "omitted",
            "StudyDate": "presence_only",
            "SeriesDescription": "presence_only",
        },
        "presence": {
            "StudyDate": bool(_safe_get(dataset, "StudyDate")),
            "SeriesDescription": bool(_safe_get(dataset, "SeriesDescription")),
            "PixelData": "PixelData" in dataset,
        },
    }


def _read_dataset_from_bytes(raw: bytes, source_name: str, include_uids: bool) -> dict[str, Any] | None:
    pydicom = _import_pydicom()
    try:
        dataset = pydicom.dcmread(io.BytesIO(raw), stop_before_pixels=True, force=True)
    except Exception:
        return None
    return dataset_to_safe_record(dataset, source_name=source_name, include_uids=include_uids)


def _iter_file_bytes(input_path: Path, max_instances: int) -> Iterable[tuple[str, bytes]]:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            yielded = 0
            for info in sorted(archive.infolist(), key=lambda item: item.filename):
                if info.is_dir():
                    continue
                if yielded >= max_instances:
                    break
                with archive.open(info) as handle:
                    yielded += 1
                    yield info.filename, handle.read()
        return

    if input_path.is_file():
        yield input_path.name, input_path.read_bytes()
        return

    yielded = 0
    for child in sorted(path for path in input_path.rglob("*") if path.is_file()):
        if yielded >= max_instances:
            break
        yielded += 1
        yield str(child.relative_to(input_path)), child.read_bytes()


def inspect_dicom_input(
    input_path: str | Path,
    *,
    max_instances: int = 50,
    include_uids: bool = False,
) -> dict[str, Any]:
    """Inspect a DICOM file, directory, or ZIP without retaining pixel data."""
    path = Path(input_path)
    if not path.exists():
        raise DicomInspectionError(f"DICOM input path not found: {path}")

    _import_pydicom()
    records: list[dict[str, Any]] = []
    unreadable = 0
    scanned = 0
    for source_name, raw in _iter_file_bytes(path, max_instances=max_instances):
        scanned += 1
        record = _read_dataset_from_bytes(raw, source_name, include_uids)
        if record is None:
            unreadable += 1
            continue
        records.append(record)

    return {
        "source": str(path),
        "source_kind": "zip" if path.is_file() and path.suffix.lower() == ".zip" else "file_or_directory",
        "max_instances": int(max_instances),
        "files_scanned": scanned,
        "dicom_instances": len(records),
        "unreadable_files": unreadable,
        "records": records,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize inspected DICOM records."""
    modalities = Counter(
        str(record.get("metadata", {}).get("Modality", "unknown"))
        for record in records
    )
    study_hashes = {
        str(record.get("identifiers", {}).get("StudyInstanceUID_hash"))
        for record in records
        if record.get("identifiers", {}).get("StudyInstanceUID_hash")
    }
    series_hashes = {
        str(record.get("identifiers", {}).get("SeriesInstanceUID_hash"))
        for record in records
        if record.get("identifiers", {}).get("SeriesInstanceUID_hash")
    }
    transfer_syntaxes = Counter(
        str(record.get("metadata", {}).get("TransferSyntaxUID", "unknown"))
        for record in records
    )

    return {
        "instances": len(records),
        "studies": len(study_hashes),
        "series": len(series_hashes),
        "modalities": dict(sorted(modalities.items())),
        "transfer_syntaxes": dict(sorted(transfer_syntaxes.items())),
    }


def example_instance_record() -> dict[str, Any]:
    """Return a clearly labeled synthetic example record for no-input manifests."""
    return {
        "source": "synthetic-example-only",
        "identifiers": {
            "PatientID_hash": stable_hash("NS-DEMO-0001"),
            "StudyInstanceUID_hash": stable_hash("1.2.826.0.1.3680043.10.543.1"),
            "SeriesInstanceUID_hash": stable_hash("1.2.826.0.1.3680043.10.543.1.1"),
            "SOPInstanceUID_hash": stable_hash("1.2.826.0.1.3680043.10.543.1.1.1"),
        },
        "metadata": {
            "Modality": "MR",
            "Rows": 256,
            "Columns": 256,
            "TransferSyntaxUID": "1.2.840.10008.1.2.1",
        },
        "redactions": {
            "PatientName": "omitted",
            "PatientBirthDate": "omitted",
            "StudyDate": "presence_only",
            "SeriesDescription": "presence_only",
        },
        "presence": {
            "StudyDate": True,
            "SeriesDescription": True,
            "PixelData": True,
        },
    }


def build_dicomweb_awareness_manifest(
    *,
    input_summary: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a DICOM/DICOMweb awareness manifest for documentation and review."""
    generated = generated_at or utc_now()
    records = []
    scan_payload: dict[str, Any] = {
        "status": "not_scanned",
        "note": "Run scripts/dicomweb_manifest.py --input <dicom-file-or-zip> to inspect local DICOM metadata.",
        "example_record": example_instance_record(),
    }
    if input_summary is not None:
        records = list(input_summary.get("records", []))
        scan_payload = {
            **input_summary,
            "summary": summarize_records(records),
        }

    return {
        "project": "NeuroSight",
        "generated_at": generated,
        "status": "dicom_ingest_supported_dicomweb_roadmap",
        "current_support": {
            "dicom_file_format": "supported as zipped DICOM series for MRI upload",
            "neurosight_endpoint": NEUROSIGHT_MRI_UPLOAD_PATH,
            "accepted_container": "DICOM .zip",
            "processing_boundary": (
                "DICOM slices are read with pydicom, stacked into a 3D volume, "
                "then passed through the MRI preprocessing/model path."
            ),
            "not_currently_supported": [
                "PACS storage",
                "QIDO-RS search server",
                "WADO-RS retrieval server",
                "STOW-RS multipart store endpoint",
                "OHIF viewer integration",
            ],
        },
        "dicomweb_services": list(DICOMWEB_SERVICES),
        "security_and_privacy": {
            "public_demo_phi_policy": "Do not upload PHI to public demos.",
            "manifest_default": "hashes patient/study/series/instance identifiers and omits direct PHI fields",
            "pixel_data_retention": "No pixel data is written to the manifest.",
            "recommended_runtime_boundary": "Use an external DICOMweb server/PACS for storage; NeuroSight should consume scoped, authorized studies.",
        },
        "dicom_standard_references": {
            "dicomweb_overview": DICOM_STANDARD_BASE,
            "restful_structure": f"{DICOM_STANDARD_BASE}/restful-structure",
            "web_services": "https://www.dicomstandard.org/standards/view/web-services",
        },
        "input_scan": scan_payload,
    }


def manifest_to_json(manifest: dict[str, Any]) -> str:
    """Serialize a manifest with stable formatting."""
    return json.dumps(manifest, indent=2, sort_keys=True)


def write_manifest(manifest: dict[str, Any], output_path: str | Path) -> Path:
    """Write a manifest to disk and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest_to_json(manifest) + "\n", encoding="utf-8")
    return path
