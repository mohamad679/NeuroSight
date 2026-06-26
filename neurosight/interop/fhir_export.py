"""FHIR R4 export helpers for NeuroSight diagnosis payloads.

The exporter builds a small, self-contained Bundle that can be inspected by
reviewers or imported into FHIR tooling. It does not claim EHR production
conformance; profile validation belongs in a downstream implementation guide.
"""

from __future__ import annotations

import base64
import json
import re
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FHIR_VERSION = "4.0.1"
NEUROSIGHT_SYSTEM = "https://github.com/mohi679/neurosight/fhir/CodeSystem/neurosight"
NEUROSIGHT_IDENTIFIER_SYSTEM = "https://github.com/mohi679/neurosight/patient-id"
UCUM_SYSTEM = "http://unitsofmeasure.org"

DEFAULT_COGNITIVE_SCORES: dict[str, float] = {
    "MMSE": 24.0,
    "MOCA": 20.0,
    "CDRSB": 1.0,
    "ADAS11": 18.0,
    "RAVLT_immediate": 34.0,
    "RAVLT_learning": 2.0,
    "FAQ": 6.0,
    "AGE": 70.0,
}

COGNITIVE_LABELS: dict[str, str] = {
    "MMSE": "Mini-Mental State Examination",
    "MOCA": "Montreal Cognitive Assessment",
    "CDRSB": "Clinical Dementia Rating Sum of Boxes",
    "ADAS11": "Alzheimer's Disease Assessment Scale 11",
    "RAVLT_immediate": "Rey Auditory Verbal Learning Test Immediate",
    "RAVLT_learning": "Rey Auditory Verbal Learning Test Learning",
    "FAQ": "Functional Activities Questionnaire",
    "AGE": "Age",
}


def utc_now() -> str:
    """Return current UTC time as a FHIR-compatible instant."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fhir_id(prefix: str, *parts: object) -> str:
    """Build a stable FHIR id from arbitrary values."""
    raw = "|".join(str(part) for part in parts if part is not None)
    digest = uuid.uuid5(uuid.NAMESPACE_URL, raw or prefix).hex[:24]
    safe_prefix = re.sub(r"[^A-Za-z0-9.-]+", "-", prefix).strip("-") or "resource"
    return f"{safe_prefix}-{digest}"[:64]


def codeable(code: str, display: str, system: str = NEUROSIGHT_SYSTEM) -> dict[str, Any]:
    """Return a compact CodeableConcept."""
    return {
        "coding": [coding(code, display, system)],
        "text": display,
    }


def coding(code: str, display: str, system: str = NEUROSIGHT_SYSTEM) -> dict[str, str]:
    """Return a compact Coding."""
    return {
        "system": system,
        "code": code,
        "display": display,
    }


def reference(resource_type: str, resource_id: str, display: str | None = None) -> dict[str, str]:
    """Return a FHIR Reference object."""
    payload = {"reference": f"{resource_type}/{resource_id}"}
    if display:
        payload["display"] = display
    return payload


def make_patient(patient_id: str, *, age: float | None = None, sex: str | None = None) -> dict[str, Any]:
    """Create a pseudonymized Patient resource."""
    resource_id = fhir_id("patient", patient_id)
    patient: dict[str, Any] = {
        "resourceType": "Patient",
        "id": resource_id,
        "identifier": [
            {
                "system": NEUROSIGHT_IDENTIFIER_SYSTEM,
                "value": patient_id,
            }
        ],
        "meta": {
            "tag": [
                coding("research-demo", "Research demo export"),
                coding("no-phi", "No PHI in public export"),
            ]
        },
    }
    if sex:
        normalized = sex.strip().lower()
        if normalized in {"male", "female", "other", "unknown"}:
            patient["gender"] = normalized
    if age is not None:
        patient["extension"] = [
            {
                "url": f"{NEUROSIGHT_SYSTEM}/StructureDefinition/demo-age",
                "valueDecimal": float(age),
            }
        ]
    return patient


def make_device(model_status: str, generated_at: str) -> dict[str, Any]:
    """Create a Device resource representing the NeuroSight runtime."""
    return {
        "resourceType": "Device",
        "id": fhir_id("device", "neurosight", model_status),
        "deviceName": [
            {
                "name": "NeuroSight Research Prototype",
                "type": "user-friendly-name",
            }
        ],
        "type": codeable("ai-research-prototype", "AI research prototype"),
        "version": [
            {
                "value": "0.3.0",
                "type": codeable("software-version", "Software version"),
            }
        ],
        "property": [
            {
                "type": codeable("model-status", "Model status"),
                "valueCode": [codeable(model_status, model_status.replace("_", " "))],
            },
            {
                "type": codeable("generated-at", "Generated at"),
                "valueCode": [codeable("generated", generated_at)],
            },
        ],
    }


def make_organization() -> dict[str, Any]:
    """Create an Organization resource for the demo producer."""
    return {
        "resourceType": "Organization",
        "id": "neurosight-research-demo",
        "name": "NeuroSight Research Demo",
        "type": [codeable("research", "Research organization")],
    }


def make_diagnosis_observation(
    patient_id: str,
    diagnosis: str,
    generated_at: str,
    device_id: str,
) -> dict[str, Any]:
    """Create an Observation for the predicted diagnosis label."""
    obs_id = fhir_id("obs-diagnosis", patient_id, diagnosis, generated_at)
    return {
        "resourceType": "Observation",
        "id": obs_id,
        "status": "final",
        "category": [codeable("ai-output", "AI model output")],
        "code": codeable("predicted-class", "Predicted class"),
        "subject": reference("Patient", fhir_id("patient", patient_id)),
        "effectiveDateTime": generated_at,
        "issued": generated_at,
        "device": reference("Device", device_id, "NeuroSight Research Prototype"),
        "valueCodeableConcept": codeable(diagnosis, diagnosis.upper()),
        "note": [
            {
                "text": (
                    "Research prototype output only. This is not a clinical diagnosis "
                    "and requires human specialist review."
                )
            }
        ],
    }


def make_confidence_observation(
    patient_id: str,
    confidence: float,
    generated_at: str,
    device_id: str,
) -> dict[str, Any]:
    """Create an Observation for model confidence as a percent."""
    bounded = max(0.0, min(float(confidence), 1.0))
    return {
        "resourceType": "Observation",
        "id": fhir_id("obs-confidence", patient_id, bounded, generated_at),
        "status": "final",
        "category": [codeable("ai-output", "AI model output")],
        "code": codeable("model-confidence", "Model confidence"),
        "subject": reference("Patient", fhir_id("patient", patient_id)),
        "effectiveDateTime": generated_at,
        "issued": generated_at,
        "device": reference("Device", device_id, "NeuroSight Research Prototype"),
        "valueQuantity": {
            "value": round(bounded * 100.0, 3),
            "unit": "%",
            "system": UCUM_SYSTEM,
            "code": "%",
        },
        "note": [{"text": "Confidence is a model score, not calibrated clinical evidence."}],
    }


def make_review_observation(
    patient_id: str,
    requires_review: bool,
    generated_at: str,
    device_id: str,
) -> dict[str, Any]:
    """Create an Observation for the human-review requirement."""
    return {
        "resourceType": "Observation",
        "id": fhir_id("obs-review", patient_id, requires_review, generated_at),
        "status": "final",
        "category": [codeable("safety", "Safety flag")],
        "code": codeable("requires-human-review", "Requires human review"),
        "subject": reference("Patient", fhir_id("patient", patient_id)),
        "effectiveDateTime": generated_at,
        "issued": generated_at,
        "device": reference("Device", device_id, "NeuroSight Research Prototype"),
        "valueBoolean": bool(requires_review),
    }


def make_cognitive_observation(
    patient_id: str,
    cognitive_scores: dict[str, float],
    generated_at: str,
) -> dict[str, Any]:
    """Create a component Observation for cognitive input scores."""
    components: list[dict[str, Any]] = []
    for key, value in cognitive_scores.items():
        label = COGNITIVE_LABELS.get(key, key.replace("_", " ").title())
        components.append(
            {
                "code": codeable(key, label),
                "valueQuantity": {
                    "value": float(value),
                    "unit": "score",
                    "system": NEUROSIGHT_SYSTEM,
                    "code": "score",
                },
            }
        )

    return {
        "resourceType": "Observation",
        "id": fhir_id("obs-cognitive", patient_id, generated_at),
        "status": "final",
        "category": [codeable("survey", "Cognitive assessment input")],
        "code": codeable("cognitive-panel", "Cognitive assessment panel"),
        "subject": reference("Patient", fhir_id("patient", patient_id)),
        "effectiveDateTime": generated_at,
        "issued": generated_at,
        "component": components,
    }


def make_diagnostic_report(
    patient_id: str,
    diagnosis: str,
    confidence: float,
    report_text: str,
    requires_review: bool,
    generated_at: str,
    observation_ids: Iterable[str],
    organization_id: str,
) -> dict[str, Any]:
    """Create the DiagnosticReport that groups model observations."""
    report_id = fhir_id("diagnostic-report", patient_id, diagnosis, generated_at)
    disclosure = (
        "NeuroSight is a research prototype. This export demonstrates FHIR "
        "interoperability and is not intended for clinical decision-making."
    )
    presented_text = f"{report_text}\n\n{disclosure}"
    return {
        "resourceType": "DiagnosticReport",
        "id": report_id,
        "status": "final",
        "category": [codeable("neurology-ai", "Neurology AI research")],
        "code": codeable("neurosight-diagnosis-report", "NeuroSight diagnosis report"),
        "subject": reference("Patient", fhir_id("patient", patient_id)),
        "effectiveDateTime": generated_at,
        "issued": generated_at,
        "performer": [reference("Organization", organization_id, "NeuroSight Research Demo")],
        "result": [reference("Observation", obs_id) for obs_id in observation_ids],
        "conclusion": (
            f"Predicted class: {diagnosis.upper()}; confidence: {confidence:.1%}; "
            f"requires human review: {requires_review}. {disclosure}"
        ),
        "presentedForm": [
            {
                "contentType": "text/plain",
                "data": base64.b64encode(presented_text.encode("utf-8")).decode("ascii"),
                "title": "NeuroSight research demo report",
                "creation": generated_at,
            }
        ],
    }


def make_provenance(
    report_id: str,
    observation_ids: Iterable[str],
    device_id: str,
    organization_id: str,
    generated_at: str,
    source: str,
) -> dict[str, Any]:
    """Create Provenance for generated report resources."""
    targets = [reference("DiagnosticReport", report_id)]
    targets.extend(reference("Observation", obs_id) for obs_id in observation_ids)
    return {
        "resourceType": "Provenance",
        "id": fhir_id("provenance", report_id, generated_at),
        "target": targets,
        "recorded": generated_at,
        "activity": codeable("ai-assisted-export", "AI-assisted research export"),
        "agent": [
            {
                "type": codeable("assembler", "Assembler"),
                "who": reference("Device", device_id, "NeuroSight Research Prototype"),
            },
            {
                "type": codeable("author", "Author"),
                "who": reference("Organization", organization_id, "NeuroSight Research Demo"),
            },
        ],
        "entity": [
            {
                "role": "source",
                "what": {
                    "identifier": {
                        "system": NEUROSIGHT_SYSTEM,
                        "value": source,
                    },
                    "display": source,
                },
            }
        ],
    }


def build_diagnosis_bundle(
    *,
    patient_id: str,
    diagnosis: str,
    confidence: float,
    report_text: str,
    requires_review: bool = True,
    cognitive_scores: dict[str, float] | None = None,
    age: float | None = None,
    sex: str | None = None,
    model_status: str = "demo_untrained",
    generated_at: str | None = None,
    source: str = "local-demo",
) -> dict[str, Any]:
    """Build a self-contained FHIR R4 Bundle for one NeuroSight result."""
    generated = generated_at or utc_now()
    normalized_diagnosis = diagnosis.strip().lower()
    scores = dict(DEFAULT_COGNITIVE_SCORES)
    if cognitive_scores:
        scores.update({str(key): float(value) for key, value in cognitive_scores.items()})

    patient = make_patient(patient_id, age=age, sex=sex)
    organization = make_organization()
    device = make_device(model_status=model_status, generated_at=generated)
    device_id = str(device["id"])

    observations = [
        make_diagnosis_observation(patient_id, normalized_diagnosis, generated, device_id),
        make_confidence_observation(patient_id, confidence, generated, device_id),
        make_review_observation(patient_id, requires_review, generated, device_id),
        make_cognitive_observation(patient_id, scores, generated),
    ]
    observation_ids = [str(observation["id"]) for observation in observations]
    report = make_diagnostic_report(
        patient_id=patient_id,
        diagnosis=normalized_diagnosis,
        confidence=confidence,
        report_text=report_text,
        requires_review=requires_review,
        generated_at=generated,
        observation_ids=observation_ids,
        organization_id=str(organization["id"]),
    )
    provenance = make_provenance(
        report_id=str(report["id"]),
        observation_ids=observation_ids,
        device_id=device_id,
        organization_id=str(organization["id"]),
        generated_at=generated,
        source=source,
    )

    resources = [patient, organization, device, *observations, report, provenance]
    bundle_id = fhir_id("bundle", patient_id, normalized_diagnosis, generated)
    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": "collection",
        "timestamp": generated,
        "meta": {
            "tag": [
                coding("fhir-r4", f"FHIR R4 {FHIR_VERSION}"),
                coding("research-demo", "Research demo export"),
            ]
        },
        "entry": [
            {
                "fullUrl": f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, resource['resourceType'] + '/' + resource['id'])}",
                "resource": resource,
            }
            for resource in resources
        ],
    }


def validate_bundle_shape(bundle: dict[str, Any]) -> list[str]:
    """Return lightweight structural validation errors for the generated Bundle."""
    errors: list[str] = []
    if bundle.get("resourceType") != "Bundle":
        errors.append("Bundle.resourceType must be Bundle")
    if bundle.get("type") != "collection":
        errors.append("Bundle.type must be collection")

    entries = bundle.get("entry")
    if not isinstance(entries, list) or not entries:
        errors.append("Bundle.entry must be a non-empty list")
        return errors

    resource_types: set[str] = set()
    resource_refs: set[str] = set()
    for entry in entries:
        resource = entry.get("resource") if isinstance(entry, dict) else None
        if not isinstance(resource, dict):
            errors.append("Every Bundle.entry must contain a resource object")
            continue
        resource_type = resource.get("resourceType")
        resource_id = resource.get("id")
        if not isinstance(resource_type, str) or not isinstance(resource_id, str):
            errors.append("Every resource must include resourceType and id")
            continue
        resource_types.add(resource_type)
        resource_refs.add(f"{resource_type}/{resource_id}")

    required = {"Patient", "DiagnosticReport", "Observation", "Device", "Organization", "Provenance"}
    missing = sorted(required - resource_types)
    if missing:
        errors.append(f"Missing resource types: {', '.join(missing)}")

    for entry in entries:
        resource = entry.get("resource") if isinstance(entry, dict) else None
        if isinstance(resource, dict) and resource.get("resourceType") == "DiagnosticReport":
            for result_ref in resource.get("result", []):
                ref_value = result_ref.get("reference") if isinstance(result_ref, dict) else None
                if isinstance(ref_value, str) and ref_value not in resource_refs:
                    errors.append(f"DiagnosticReport.result references missing resource: {ref_value}")

    return errors


def bundle_to_json(bundle: dict[str, Any]) -> str:
    """Serialize a bundle with stable formatting."""
    return json.dumps(bundle, indent=2, sort_keys=True)


def write_bundle(bundle: dict[str, Any], output_path: str | Path) -> Path:
    """Write a FHIR Bundle to disk and return the resolved path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bundle_to_json(bundle) + "\n", encoding="utf-8")
    return path
