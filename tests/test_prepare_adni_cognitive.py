"""Synthetic tests for the authorized ADNI-style cognitive adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_adni_cognitive import (
    CANONICAL_LABEL,
    CANONICAL_SUBJECT_ID,
    AdapterValidationError,
    prepare_dataset,
)


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _synthetic_csv() -> str:
    return """\
RID,DX_bl,EXAMDATE,AGE,MMSE,MOCA,CDRSB,ADAS11,RAVLT_immediate,RAVLT_learning,FAQ
S001,CN,2024-01-01,70,29,28,0,5,50,4,0
S002,CN,2024-01-02,71,28,27,0.5,6,48,3,1
S003,CN,2024-01-03,72,27,26,1,7,46,2,1
S004,MCI,2024-01-04,73,24,22,2,14,34,0,5
S005,MCI,2024-01-05,74,23,21,2.5,15,32,-1,6
S006,MCI,2024-01-06,75,22,20,3,16,30,-2,7
S007,Dementia,2024-01-07,76,18,15,7,30,20,-4,12
S008,Dementia,2024-01-08,77,17,14,8,32,18,-5,14
S009,Dementia,2024-01-09,78,16,13,9,34,16,-6,15
"""


@pytest.mark.unit
def test_prepare_adni_cognitive_writes_subject_disjoint_splits(tmp_path: Path) -> None:
    input_csv = _write_csv(tmp_path / "synthetic_adni.csv", _synthetic_csv())
    output_dir = tmp_path / "processed"

    prepared = prepare_dataset(input_csv, output_dir=output_dir, seed=7)

    assert prepared.train_path.exists()
    assert prepared.val_path.exists()
    assert prepared.test_path.exists()
    assert prepared.summary_path.exists()

    split_subjects = []
    for split_path in (prepared.train_path, prepared.val_path, prepared.test_path):
        frame = pd.read_csv(split_path)
        assert CANONICAL_SUBJECT_ID in frame.columns
        assert CANONICAL_LABEL in frame.columns
        split_subjects.append(set(frame[CANONICAL_SUBJECT_ID].astype(str)))

    assert split_subjects[0].isdisjoint(split_subjects[1])
    assert split_subjects[0].isdisjoint(split_subjects[2])
    assert split_subjects[1].isdisjoint(split_subjects[2])

    summary = json.loads(prepared.summary_path.read_text(encoding="utf-8"))
    assert summary["clinical_validity"] is False
    assert summary["raw_data_committed"] is False
    assert summary["leakage_checks"]["subject_disjoint_splits"] is True
    assert summary["n_subjects"] == 9


@pytest.mark.unit
def test_prepare_adni_cognitive_rejects_missing_required_column(tmp_path: Path) -> None:
    input_csv = _write_csv(
        tmp_path / "missing.csv",
        """\
RID,DX_bl,AGE,MMSE,MOCA,CDRSB,ADAS11,RAVLT_immediate,RAVLT_learning
S001,CN,70,29,28,0,5,50,4
""",
    )

    with pytest.raises(AdapterValidationError, match="Missing required columns: FAQ"):
        prepare_dataset(input_csv, output_dir=tmp_path / "processed")


@pytest.mark.unit
def test_prepare_adni_cognitive_rejects_invalid_values(tmp_path: Path) -> None:
    input_csv = _write_csv(
        tmp_path / "bad_values.csv",
        """\
RID,DX_bl,AGE,MMSE,MOCA,CDRSB,ADAS11,RAVLT_immediate,RAVLT_learning,FAQ
S001,CN,70,31,28,0,5,50,4,0
S002,MCI,71,24,22,2,14,34,0,5
S003,Dementia,72,18,15,7,30,20,-4,12
""",
    )

    with pytest.raises(AdapterValidationError, match="MMSE"):
        prepare_dataset(input_csv, output_dir=tmp_path / "processed")


@pytest.mark.unit
def test_prepare_adni_cognitive_blocks_label_leakage_columns(tmp_path: Path) -> None:
    input_csv = _write_csv(
        tmp_path / "leakage.csv",
        """\
RID,DX_bl,DXCHANGE,AGE,MMSE,MOCA,CDRSB,ADAS11,RAVLT_immediate,RAVLT_learning,FAQ
S001,CN,stable,70,29,28,0,5,50,4,0
S002,MCI,progressed,71,24,22,2,14,34,0,5
S003,Dementia,stable,72,18,15,7,30,20,-4,12
""",
    )

    with pytest.raises(AdapterValidationError, match="label-leakage"):
        prepare_dataset(input_csv, output_dir=tmp_path / "processed")


@pytest.mark.unit
def test_prepare_adni_cognitive_blocks_post_diagnosis_columns(tmp_path: Path) -> None:
    input_csv = _write_csv(
        tmp_path / "future.csv",
        """\
RID,DX_bl,months_to_conversion,AGE,MMSE,MOCA,CDRSB,ADAS11,RAVLT_immediate,RAVLT_learning,FAQ
S001,CN,0,70,29,28,0,5,50,4,0
S002,MCI,12,71,24,22,2,14,34,0,5
S003,Dementia,0,72,18,15,7,30,20,-4,12
""",
    )

    with pytest.raises(AdapterValidationError, match="post-diagnosis"):
        prepare_dataset(input_csv, output_dir=tmp_path / "processed")
