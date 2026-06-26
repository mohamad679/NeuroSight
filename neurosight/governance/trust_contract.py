"""Read-only trust, explainability, privacy, and security contracts."""

from __future__ import annotations

from typing import Any


def build_xai_contract(runtime: dict[str, Any]) -> dict[str, Any]:
    """Return an honest explainability capability contract."""
    return {
        "status": "ok",
        "endpoint": "/v1/xai/{patient_id}",
        "runtime_mode": runtime.get("runtime_mode", "demo"),
        "class_mode": runtime.get("class_mode", "six_class_demo"),
        "methods": [
            {
                "modality": "cognitive",
                "status": "implemented",
                "method": "gradient_x_input",
                "artifact": "feature_importance",
                "requires_uploaded_data": False,
                "source": "KG-derived or default cognitive feature vector",
                "validated_for_clinical_use": False,
                "limitations": [
                    "Explains the current model response, not disease causality.",
                    "Uses synthetic/default cognitive profiles when patient history is unavailable.",
                    "Feature importance should be read with the model-status disclosure.",
                ],
            },
            {
                "modality": "mri",
                "status": "architecture_supported_runtime_limited",
                "method": "gradcam_plus_plus",
                "artifact": "saliency_volume",
                "requires_uploaded_data": True,
                "source": "MRI tensor supplied by an authorized operator",
                "validated_for_clinical_use": False,
                "limitations": [
                    "The public patient XAI endpoint does not persist uploaded MRI tensors.",
                    "Real MRI saliency requires a trained checkpoint and image-space QC.",
                    "Public demo outputs are not radiology evidence.",
                ],
            },
            {
                "modality": "eeg",
                "status": "architecture_supported_runtime_limited",
                "method": "attention_rollout",
                "artifact": "temporal_importance",
                "requires_uploaded_data": True,
                "source": "EEG tensor supplied by an authorized operator",
                "validated_for_clinical_use": False,
                "limitations": [
                    "The public patient XAI endpoint does not persist uploaded EEG tensors.",
                    "Real EEG explanation requires montage, artifact, and sampling-rate QC.",
                    "Attention maps are model diagnostics, not clinical biomarkers.",
                ],
            },
            {
                "modality": "fusion",
                "status": "model_outputs_available",
                "method": "cross_modal_attention",
                "artifact": "modality_weighting",
                "requires_uploaded_data": False,
                "source": "fusion model forward pass",
                "validated_for_clinical_use": False,
                "limitations": [
                    "Attention weights summarize model routing pressure only.",
                    "Missing-modality tokens can strongly affect modality weights.",
                ],
            },
        ],
        "interpretation_policy": {
            "clinical_use_allowed": False,
            "causal_claims_allowed": False,
            "requires_human_review": True,
            "patient_level_claims_allowed": False,
            "primary_notice": (
                "Explainability payloads are research-demo diagnostics of model behavior. "
                "They are not clinical evidence and must not be used for medical decisions."
            ),
        },
    }


def build_governance_contract(
    runtime: dict[str, Any],
    upload_limits: dict[str, int],
    allowed_origins: list[str],
    rate_limiting_available: bool,
) -> dict[str, Any]:
    """Return privacy, security, and scientific-disclosure metadata."""
    return {
        "status": "ok",
        "runtime_mode": runtime.get("runtime_mode", "demo"),
        "class_mode": runtime.get("class_mode", "six_class_demo"),
        "privacy": {
            "public_demo_safe": True,
            "private_adni_in_repository": False,
            "demo_data_policy": "synthetic_or_mock_only",
            "operator_supplied_data_policy": (
                "Authorized operators may point the runtime at ADNI-style files outside "
                "the public repository."
            ),
            "phi_policy": "Do not upload PHI to public demo deployments.",
            "upload_retention": (
                "Uploaded MRI/EEG files are read in memory for embedding generation. "
                "Temporary NIfTI files are deleted after parsing."
            ),
            "clinical_use_allowed": False,
        },
        "security": {
            "protected_endpoint_scope": "/v1/* endpoints require X-API-Key outside APP_ENV=test.",
            "api_key_header": "X-API-Key",
            "cors_allowed_origins": list(allowed_origins),
            "rate_limiting_available": bool(rate_limiting_available),
            "request_observability_headers": [
                "X-Request-ID",
                "X-Trace-ID",
                "X-Process-Time",
                "X-Observability",
            ],
            "upload_controls": {
                "max_upload_bytes": int(upload_limits.get("max_upload_bytes", 0)),
                "max_mri_values": int(upload_limits.get("max_mri_values", 0)),
                "max_eeg_values": int(upload_limits.get("max_eeg_values", 0)),
                "max_dicom_zip_members": int(upload_limits.get("max_dicom_zip_members", 0)),
                "zip_path_traversal_guard": True,
                "encrypted_zip_rejected": True,
                "numpy_pickle_disabled": True,
            },
        },
        "scientific_disclosure": {
            "validated_clinically": False,
            "adni_style_validated_classes": ["normal", "mci", "ad"],
            "synthetic_placeholder_classes": ["ftd", "lbd", "vd"],
            "label_policy": (
                "Normal/MCI/AD match the ADNI-style workflow. FTD/LBD/VD remain "
                "synthetic demo placeholders until additional datasets and validation exist."
            ),
            "required_before_real_claims": [
                "authorized real cohort ingestion",
                "trained checkpoint loading",
                "external validation",
                "subgroup/fairness audit",
                "model-card update",
                "clinical expert review",
            ],
        },
    }
