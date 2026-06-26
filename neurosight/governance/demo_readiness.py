"""Final demo-readiness contract for NeuroSight."""

from __future__ import annotations

from typing import Any


def _check(
    check_id: str,
    label: str,
    status: str,
    detail: str,
    *,
    action: str = "",
    blocking: bool = False,
) -> dict[str, Any]:
    """Create one launch-readiness checklist row."""
    return {
        "id": check_id,
        "label": label,
        "status": status,
        "detail": detail,
        "action": action,
        "blocking": bool(blocking),
    }


def _overall_status(checks: list[dict[str, Any]]) -> str:
    """Summarize checklist rows into one launch status."""
    if any(item["status"] == "action_required" for item in checks):
        return "needs_attention"
    if any(item["status"] == "warning" for item in checks):
        return "demo_ready_with_warnings"
    return "demo_ready"


def _counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    """Count readiness states."""
    return {
        "ready": sum(1 for item in checks if item["status"] == "ready"),
        "warning": sum(1 for item in checks if item["status"] == "warning"),
        "action_required": sum(1 for item in checks if item["status"] == "action_required"),
        "total": len(checks),
    }


def build_demo_readiness_contract(
    runtime: dict[str, Any],
    data: dict[str, Any],
    modalities: dict[str, Any],
    checkpoint: dict[str, Any],
    xai: dict[str, Any],
    governance: dict[str, Any],
    capabilities: dict[str, Any],
) -> dict[str, Any]:
    """Build a single launch-readiness view over existing read-only contracts."""
    data_status = str(data.get("status") or "missing")
    data_summary = data.get("summary", {})
    data_privacy = data.get("privacy", {})
    recommended_patient_id = data.get("recommended_patient_id")
    checkpoint_artifact = checkpoint.get("checkpoint", {})
    checkpoint_loading = checkpoint.get("loading", {})
    xai_methods = xai.get("methods", [])
    xai_policy = xai.get("interpretation_policy", {})
    governance_privacy = governance.get("privacy", {})
    governance_security = governance.get("security", {})
    upload_controls = governance_security.get("upload_controls", {})
    capability_summary = capabilities.get("summary", {})

    row_count = int(data_summary.get("row_count") or 0)
    ui_coverage = float(capability_summary.get("ui_coverage_percent") or 0.0)
    checks = [
        _check(
            "runtime_scope",
            "Runtime and class scope",
            "ready",
            (
                f"{runtime.get('runtime_mode', 'unknown')} runtime with "
                f"{runtime.get('class_mode', 'unknown')} class policy."
            ),
        ),
        _check(
            "data_contract",
            "Demo/ADNI-style data contract",
            "ready" if data_status == "ready" and row_count > 0 else "action_required",
            f"{row_count} metadata rows from {data.get('source_kind', 'unknown source')}.",
            action="Load or generate an ADNI-style CSV before presenting the demo.",
            blocking=data_status != "ready" or row_count <= 0,
        ),
        _check(
            "demo_patient",
            "Demo patient lookup",
            "ready" if recommended_patient_id else "action_required",
            (
                f"Recommended patient ID: {recommended_patient_id}."
                if recommended_patient_id
                else "No demo patient ID is available."
            ),
            action="Use /v1/data/demo-patients after the CSV is ready.",
            blocking=not recommended_patient_id,
        ),
        _check(
            "modality_contract",
            "Upload and preprocessing contract",
            "ready" if modalities.get("status") in {"ok", "ready"} else "warning",
            "MRI, EEG, and cognitive upload contracts are exposed.",
            action="Review optional dependency badges before demonstrating MRI/EEG uploads.",
        ),
        _check(
            "checkpoint_disclosure",
            "Checkpoint disclosure",
            "ready" if checkpoint_artifact.get("exists") else "warning",
            (
                "Checkpoint artifact exists; runtime loading is "
                f"{'enabled' if checkpoint_loading.get('enabled') else 'disabled'}."
                if checkpoint_artifact.get("exists")
                else "No checkpoint artifact found at the configured path."
            ),
            action="For trained demos, set NEUROSIGHT_CHECKPOINT_PATH and NEUROSIGHT_LOAD_CHECKPOINT=true.",
        ),
        _check(
            "xai_disclosure",
            "Explainability disclosure",
            "ready" if xai_methods and xai_policy.get("clinical_use_allowed") is False else "action_required",
            f"{len(xai_methods)} XAI method contracts exposed; clinical use is blocked.",
            action="Expose /v1/xai/status before presenting XAI results.",
            blocking=not xai_methods,
        ),
        _check(
            "privacy_governance",
            "Privacy and clinical-use guardrails",
            "ready"
            if governance_privacy.get("public_demo_safe") and governance_privacy.get("clinical_use_allowed") is False
            else "action_required",
            (
                "Public demo is synthetic/mock-data safe and blocks clinical use."
                if governance_privacy.get("public_demo_safe")
                else "Public demo safety was not confirmed."
            ),
            action="Keep private ADNI files outside the repository and public Space.",
            blocking=not governance_privacy.get("public_demo_safe"),
        ),
        _check(
            "api_ui_coverage",
            "UI capability coverage",
            "ready" if ui_coverage >= 90.0 else "warning",
            f"{ui_coverage:.1f}% of implemented API capabilities are exposed in the local console.",
            action="Expose any remaining API-only capability in the local UI if needed.",
        ),
    ]

    return {
        "status": _overall_status(checks),
        "counts": _counts(checks),
        "recommended_patient_id": recommended_patient_id,
        "runtime_mode": runtime.get("runtime_mode"),
        "class_mode": runtime.get("class_mode"),
        "public_demo_safe": bool(data_privacy.get("public_demo_safe")),
        "clinical_use_allowed": False,
        "upload_limit_bytes": int(upload_controls.get("max_upload_bytes") or 0),
        "checks": checks,
        "recommended_ui_flow": [
            {
                "order": 1,
                "view": "Overview",
                "action": "Check Backend",
                "expected": "Runtime, class scope, capability coverage, checkpoint, XAI, and trust contracts load.",
            },
            {
                "order": 2,
                "view": "Demo",
                "action": "Load Demo Readiness",
                "expected": "Checklist reports demo_ready or demo_ready_with_warnings.",
            },
            {
                "order": 3,
                "view": "Data",
                "action": "Load Data Status and Load Demo Patients",
                "expected": "Synthetic or authorized ADNI-style rows appear with a safe patient ID.",
            },
            {
                "order": 4,
                "view": "Diagnosis",
                "action": "Run the recommended patient ID or cognitive-score demo",
                "expected": "Diagnosis payload includes confidence and requires specialist review.",
            },
            {
                "order": 5,
                "view": "XAI",
                "action": "Load XAI Status, then Generate XAI for cognitive modality",
                "expected": "Feature importance chart appears with clinical-use and causality limits.",
            },
            {
                "order": 6,
                "view": "Trust",
                "action": "Load Trust Status",
                "expected": "Privacy, security, upload controls, and scientific guardrails are visible.",
            },
        ],
        "adni_style_private_run": {
            "recommended_class_mode": "three_class_adni",
            "required_env": [
                "NEUROSIGHT_RUNTIME_MODE=adni_style",
                "NEUROSIGHT_CLASS_MODE=three_class_adni",
                "NEUROSIGHT_PATIENT_CSV_PATH=/secure/path/ADNIMERGE.csv",
                "NEUROSIGHT_MRI_DIR=/secure/path/mri",
                "NEUROSIGHT_EEG_DIR=/secure/path/eeg",
            ],
            "optional_checkpoint_env": [
                "NEUROSIGHT_CHECKPOINT_PATH=/secure/path/best_fusion.pt",
                "NEUROSIGHT_LOAD_CHECKPOINT=true",
            ],
            "warning": "Authorized ADNI/OASIS-style files must remain outside the public repository.",
        },
    }
