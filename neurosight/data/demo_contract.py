"""Public demo and ADNI-style data readiness helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from neurosight.data.synthetic import ADNI_COLUMNS


ADNI_STYLE_CLASSES: tuple[str, ...] = ("normal", "mci", "ad")
SYNTHETIC_DEMO_ONLY_CLASSES: tuple[str, ...] = ("ftd", "lbd", "vd")


def normalize_patient_id(rid_value: Any) -> str:
    """Normalize ADNI-style RID values for file lookup and API display."""
    patient_id = str(rid_value or "").strip()
    if patient_id.endswith(".0"):
        trimmed = patient_id[:-2]
        if trimmed.isdigit():
            patient_id = trimmed
    return patient_id


def map_dx_to_label(dx_raw: Any) -> str:
    """Map ADNI-style diagnosis labels to NeuroSight's canonical labels."""
    normalized = str(dx_raw or "").strip().upper().replace(" ", "")
    if normalized in {"CN", "NORMAL", "NC"}:
        return "normal"
    if "MCI" in normalized:
        return "mci"
    if normalized in {"DEMENTIA", "AD", "ALZHEIMER", "ALZHEIMERSDISEASE"}:
        return "ad"
    if "FTD" in normalized:
        return "ftd"
    if "LBD" in normalized or "LEWYBODY" in normalized:
        return "lbd"
    if normalized == "VD" or "VASCULAR" in normalized:
        return "vd"
    return "unknown"


def _safe_float(value: Any) -> float | None:
    """Parse a float value for public demo summaries."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_csv(csv_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read an ADNI-style CSV as dictionaries."""
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = [str(column) for column in (reader.fieldnames or [])]
        return columns, [dict(row) for row in reader]


def _row_to_demo_patient(
    row: dict[str, Any],
    mri_dir: Path,
    eeg_dir: Path,
) -> dict[str, Any]:
    """Convert one source row into a privacy-conscious demo patient payload."""
    patient_id = normalize_patient_id(row.get("RID"))
    diagnosis = map_dx_to_label(row.get("DX_bl"))
    class_scope = (
        "adni_style"
        if diagnosis in ADNI_STYLE_CLASSES
        else "synthetic_demo_only"
        if diagnosis in SYNTHETIC_DEMO_ONLY_CLASSES
        else "unknown"
    )

    mri_path = mri_dir / f"{patient_id}.npy"
    eeg_path = eeg_dir / f"{patient_id}.npy"
    return {
        "patient_id": patient_id,
        "source_dx": str(row.get("DX_bl", "")).strip(),
        "diagnosis_label": diagnosis,
        "class_scope": class_scope,
        "age": _safe_float(row.get("AGE")),
        "sex": str(row.get("PTGENDER", "")).strip() or None,
        "scores": {
            "MMSE": _safe_float(row.get("MMSE")),
            "MOCA": _safe_float(row.get("MOCA")),
            "CDRSB": _safe_float(row.get("CDRSB")),
            "ADAS11": _safe_float(row.get("ADAS11")),
            "RAVLT_immediate": _safe_float(row.get("RAVLT_immediate")),
            "RAVLT_learning": _safe_float(row.get("RAVLT_learning")),
            "FAQ": _safe_float(row.get("FAQ")),
            "AGE": _safe_float(row.get("AGE")),
        },
        "modalities": {
            "cognitive": True,
            "mri_npy": mri_path.exists(),
            "eeg_npy": eeg_path.exists(),
        },
    }


def build_data_contract(
    csv_path: Path,
    synthetic_csv_path: Path,
    mri_dir: Path,
    eeg_dir: Path,
    runtime: dict[str, Any],
    sample_limit: int = 6,
) -> dict[str, Any]:
    """Build a data readiness payload for API responses and the local UI."""
    csv_path = Path(csv_path)
    synthetic_csv_path = Path(synthetic_csv_path)
    mri_dir = Path(mri_dir)
    eeg_dir = Path(eeg_dir)

    source_kind = (
        "synthetic_adni_like_demo"
        if csv_path.resolve() == synthetic_csv_path.resolve()
        else "operator_supplied_adni_style"
    )
    base_payload: dict[str, Any] = {
        "status": "missing",
        "source_kind": source_kind,
        "privacy": {
            "public_demo_safe": source_kind == "synthetic_adni_like_demo",
            "private_adni_in_repo": False,
            "notice": (
                "Synthetic ADNI-like demo data only."
                if source_kind == "synthetic_adni_like_demo"
                else "Operator-supplied ADNI-style data; verify data-use approval and never commit private records."
            ),
        },
        "paths": {
            "csv": str(csv_path),
            "synthetic_csv": str(synthetic_csv_path),
            "mri_dir": str(mri_dir),
            "eeg_dir": str(eeg_dir),
        },
        "files": {
            "csv_exists": csv_path.exists(),
            "mri_dir_exists": mri_dir.exists(),
            "eeg_dir_exists": eeg_dir.exists(),
        },
        "schema": {
            "required_columns": list(ADNI_COLUMNS),
            "columns": [],
            "missing_columns": list(ADNI_COLUMNS),
        },
        "summary": {
            "row_count": 0,
            "label_distribution": {},
            "adni_style_count": 0,
            "synthetic_demo_only_count": 0,
            "unknown_label_count": 0,
            "mri_file_count": 0,
            "eeg_file_count": 0,
        },
        "runtime": {
            "runtime_mode": runtime.get("runtime_mode"),
            "class_mode": runtime.get("class_mode"),
            "classes": runtime.get("classes", []),
            "adni_style_classes": runtime.get("adni_style_classes", []),
            "synthetic_demo_only_classes": runtime.get("synthetic_demo_only_classes", []),
        },
        "supported_upload_formats": {
            "mri": [".npy", ".nii", ".nii.gz", "DICOM .zip"],
            "eeg": [".npy", ".edf"],
            "cognitive": ["JSON scores"],
        },
        "samples": [],
        "recommended_patient_id": None,
    }

    if not csv_path.exists():
        return base_payload

    columns, rows = _read_csv(csv_path)
    missing_columns = sorted(set(ADNI_COLUMNS) - set(columns))
    base_payload["schema"] = {
        "required_columns": list(ADNI_COLUMNS),
        "columns": columns,
        "missing_columns": missing_columns,
    }
    if missing_columns:
        base_payload["status"] = "invalid_schema"
        return base_payload

    label_distribution: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    mri_file_count = 0
    eeg_file_count = 0

    for row in rows:
        patient = _row_to_demo_patient(row, mri_dir=mri_dir, eeg_dir=eeg_dir)
        diagnosis = str(patient["diagnosis_label"])
        label_distribution[diagnosis] = label_distribution.get(diagnosis, 0) + 1
        if patient["modalities"]["mri_npy"]:
            mri_file_count += 1
        if patient["modalities"]["eeg_npy"]:
            eeg_file_count += 1
        if len(samples) < max(0, int(sample_limit)):
            samples.append(patient)

    base_payload["status"] = "ready"
    base_payload["summary"] = {
        "row_count": len(rows),
        "label_distribution": label_distribution,
        "adni_style_count": sum(
            label_distribution.get(label, 0) for label in ADNI_STYLE_CLASSES
        ),
        "synthetic_demo_only_count": sum(
            label_distribution.get(label, 0) for label in SYNTHETIC_DEMO_ONLY_CLASSES
        ),
        "unknown_label_count": label_distribution.get("unknown", 0),
        "mri_file_count": mri_file_count,
        "eeg_file_count": eeg_file_count,
    }
    base_payload["files"] = {
        "csv_exists": True,
        "mri_dir_exists": mri_dir.exists(),
        "eeg_dir_exists": eeg_dir.exists(),
    }
    base_payload["samples"] = samples
    base_payload["recommended_patient_id"] = samples[0]["patient_id"] if samples else None
    return base_payload
