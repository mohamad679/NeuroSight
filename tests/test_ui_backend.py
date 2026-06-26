"""Backend tests for Gradio UI helper functions."""

from __future__ import annotations

import os
import pytest

import pandas as pd

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DISABLE_MRI_WARMUP", "1")

from app import (
    query_kg_ui,
    run_ablation_ui,
    run_calibration_ui,
    run_diagnosis_ui,
    seed_demo_kg_ui,
)


@pytest.mark.integration
@pytest.mark.frontend
def test_diagnosis_ui_normal_profile_returns_dict() -> None:
    """Normal cognitive profile returns prediction with confidence > 0."""
    diagnosis, confidence, requires_review, modality_weights, feature_importance, report_text, _ = (
        run_diagnosis_ui(
            mmse=29.0,
            moca=28.0,
            cdrsb=0.0,
            adas11=5.0,
            ravlt_immediate=50.0,
            ravlt_learning=10.0,
            faq=0.0,
            age=65.0,
            mri_file=None,
            eeg_file=None,
            query="",
        )
    )

    assert isinstance(diagnosis, str) and diagnosis, "Diagnosis should be a non-empty string."
    assert confidence > 0.0, "Diagnosis confidence should be positive."
    assert isinstance(requires_review, bool), "requires_review should be a boolean value."
    assert isinstance(modality_weights, dict), "Modality weights must be returned as a dictionary."
    assert isinstance(feature_importance, dict), "Feature importance must be returned as a dictionary."
    assert isinstance(report_text, str) and report_text, "Clinical report text should be non-empty."


@pytest.mark.integration
@pytest.mark.frontend
def test_diagnosis_ui_ad_profile_requires_review() -> None:
    """AD-like cognitive profile (MMSE=19) has requires_review or high confidence."""
    _, confidence, requires_review, _, _, _, _ = run_diagnosis_ui(
        mmse=19.0,
        moca=16.0,
        cdrsb=4.0,
        adas11=25.0,
        ravlt_immediate=20.0,
        ravlt_learning=2.0,
        faq=15.0,
        age=76.0,
        mri_file=None,
        eeg_file=None,
        query="What is the differential diagnosis?",
    )

    assert (
        requires_review or confidence >= 0.75
    ), "AD-like profile should either require review or produce high confidence prediction."


@pytest.mark.integration
@pytest.mark.frontend
def test_ablation_ui_returns_dataframe_with_4_rows() -> None:
    """Ablation study returns DataFrame with 4 configuration rows."""
    ablation_df, _ = run_ablation_ui()
    assert isinstance(ablation_df, pd.DataFrame), "Ablation output must be a pandas DataFrame."
    assert len(ablation_df) == 4, "Ablation DataFrame must contain exactly four configuration rows."


@pytest.mark.integration
@pytest.mark.frontend
def test_calibration_ui_returns_ece_float() -> None:
    """Calibration check returns ECE as float in [0, 1]."""
    ece_value, _ = run_calibration_ui()
    assert isinstance(ece_value, float), "ECE output must be a float value."
    assert 0.0 <= ece_value <= 1.0, "ECE should be bounded between 0 and 1."


@pytest.mark.integration
@pytest.mark.frontend
def test_kg_ui_query_history_returns_dataframe() -> None:
    """KG history query returns a DataFrame (may be empty for unknown patient)."""
    frame = query_kg_ui(patient_id="UNKNOWN_PATIENT", query_type="history", target_date="")
    assert isinstance(frame, pd.DataFrame), "KG query output should be a pandas DataFrame."


@pytest.mark.integration
@pytest.mark.frontend
def test_seed_kg_ui_returns_success_string() -> None:
    """Seed function returns non-empty success message."""
    message = seed_demo_kg_ui()
    assert isinstance(message, str) and message.strip(), "Seed KG output should be a non-empty status message."
