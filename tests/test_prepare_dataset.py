"""Tests for the NeuroSight real-data preparation and validation interface."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

from scripts.prepare_adni_like_dataset import (
    DataValidationError,
    ValidationResult,
    generate_sample_schema_json,
    print_schema_documentation,
    validate_csv,
)


def _write_csv(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


# Minimal valid CSV matching the schema
_VALID_CSV = """\
RID,DX_bl,AGE,MMSE,MOCA,CDRSB,ADAS11,RAVLT_immediate,RAVLT_learning,FAQ
1,CN,72.0,29,27,0.0,6.0,45,3,0
2,MCI,68.5,24,22,2.5,14.0,32,-1,4
3,Dementia,75.3,18,14,8.0,30.0,18,-4,12
4,CN,65.0,30,28,0.0,4.0,50,4,0
5,MCI,70.1,22,19,3.5,18.0,28,-2,6
"""


@pytest.mark.unit
def test_schema_validation_accepts_valid_csv(tmp_path: Path) -> None:
    """Valid CSV passes schema validation without errors."""
    csv_file = _write_csv(tmp_path / "valid.csv", _VALID_CSV)
    result = validate_csv(csv_file)
    assert result.valid is True, f"Expected valid=True, got errors: {result.errors}"
    assert result.n_rows == 5


@pytest.mark.unit
def test_schema_validation_rejects_missing_required_column(tmp_path: Path) -> None:
    """CSV missing 'MMSE' column fails validation with a clear error."""
    csv_no_mmse = """\
RID,DX_bl,AGE,MOCA,CDRSB,ADAS11,RAVLT_immediate,RAVLT_learning,FAQ
1,CN,72.0,27,0.0,6.0,45,3,0
"""
    csv_file = _write_csv(tmp_path / "no_mmse.csv", csv_no_mmse)
    result = validate_csv(csv_file)
    assert result.valid is False
    missing_errors = [e for e in result.errors if "MMSE" in e]
    assert len(missing_errors) >= 1, (
        f"Expected error mentioning MMSE, got errors: {result.errors}"
    )


@pytest.mark.unit
def test_schema_validation_rejects_out_of_range_mmse(tmp_path: Path) -> None:
    """CSV with MMSE=35 (>30) fails validation with a range error."""
    csv_oob = """\
RID,DX_bl,AGE,MMSE,MOCA,CDRSB,ADAS11,RAVLT_immediate,RAVLT_learning,FAQ
1,CN,72.0,35,27,0.0,6.0,45,3,0
2,MCI,68.5,24,22,2.5,14.0,32,-1,4
"""
    csv_file = _write_csv(tmp_path / "oob_mmse.csv", csv_oob)
    result = validate_csv(csv_file)
    assert result.valid is False
    mmse_errors = [e for e in result.errors if "MMSE" in e]
    assert len(mmse_errors) >= 1, (
        f"Expected range error for MMSE, got errors: {result.errors}"
    )


@pytest.mark.unit
def test_schema_validation_rejects_invalid_dx_label(tmp_path: Path) -> None:
    """CSV with invalid DX_bl label (e.g. 'UNKNOWN') fails validation."""
    csv_bad_label = """\
RID,DX_bl,AGE,MMSE,MOCA,CDRSB,ADAS11,RAVLT_immediate,RAVLT_learning,FAQ
1,UNKNOWN,72.0,29,27,0.0,6.0,45,3,0
"""
    csv_file = _write_csv(tmp_path / "bad_label.csv", csv_bad_label)
    result = validate_csv(csv_file)
    assert result.valid is False
    label_errors = [e for e in result.errors if "DX_bl" in e or "UNKNOWN" in e or "unexpected" in e.lower()]
    assert len(label_errors) >= 1, (
        f"Expected label error for DX_bl, got errors: {result.errors}"
    )


@pytest.mark.unit
def test_schema_validation_warns_on_low_class_support(tmp_path: Path) -> None:
    """CSV with fewer than 5 samples for a class produces a warning."""
    # Only 1 sample for 'Dementia'
    csv_low_support = """\
RID,DX_bl,AGE,MMSE,MOCA,CDRSB,ADAS11,RAVLT_immediate,RAVLT_learning,FAQ
1,CN,72.0,29,27,0.0,6.0,45,3,0
2,CN,68.0,28,26,0.0,5.0,47,3,0
3,CN,70.0,30,28,0.0,4.0,50,4,0
4,CN,66.0,27,25,0.5,7.0,42,2,1
5,CN,74.0,29,27,0.0,6.0,44,3,0
6,Dementia,75.0,18,14,8.0,30.0,18,-4,12
"""
    csv_file = _write_csv(tmp_path / "low_support.csv", csv_low_support)
    result = validate_csv(csv_file)
    # May be valid but should have warnings
    dementia_warnings = [w for w in result.warnings if "Dementia" in w]
    assert len(dementia_warnings) >= 1, (
        f"Expected warning for low Dementia support, got warnings: {result.warnings}"
    )


@pytest.mark.unit
def test_schema_validation_file_not_found_raises() -> None:
    """validate_csv raises FileNotFoundError for non-existent file."""
    with pytest.raises(FileNotFoundError):
        validate_csv("/this/path/does/not/exist/data.csv")


@pytest.mark.unit
def test_schema_documentation_prints_without_error(capsys: pytest.CaptureFixture) -> None:
    """print_schema_documentation runs without exceptions."""
    print_schema_documentation()
    captured = capsys.readouterr()
    assert "MMSE" in captured.out
    assert "DX_bl" in captured.out
    assert len(captured.out) > 100


@pytest.mark.unit
def test_generate_sample_schema_json_structure() -> None:
    """generate_sample_schema_json returns valid schema dict."""
    schema = generate_sample_schema_json()
    assert "columns" in schema
    assert "clinical_validity" in schema
    assert schema["clinical_validity"] is False
    col_names = [c["name"] for c in schema["columns"]]
    for required in ("MMSE", "MOCA", "CDRSB", "DX_bl", "AGE"):
        assert required in col_names, f"Required column '{required}' missing from schema"


@pytest.mark.unit
def test_generate_sample_schema_json_saves_to_disk(tmp_path: Path) -> None:
    """generate_sample_schema_json writes a valid JSON file when path given."""
    import json

    out = tmp_path / "schema.json"
    generate_sample_schema_json(out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert "columns" in data
    assert data["clinical_validity"] is False
