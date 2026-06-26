# FHIR Export

NeuroSight can export a model-generated risk profile as a self-contained HL7 FHIR R4
Bundle. This is a portfolio-grade interoperability proof: it shows how the
backend result can be mapped into healthcare data exchange resources without
pretending the demo model is clinically validated.

Official FHIR references used for this mapping:

- FHIR R4 `DiagnosticReport`: https://hl7.org/fhir/R4/diagnosticreport.html
- FHIR R4 `Observation`: https://hl7.org/fhir/R4/observation.html

## Files

| File | Purpose |
|------|---------|
| `neurosight/interop/fhir_export.py` | Pure-Python FHIR Bundle builder and lightweight structural validator |
| `scripts/fhir_export.py` | Runnable CLI that writes a demo Bundle or exports a backend risk-profiling response |
| `logs/fhir/neurosight_demo_bundle.json` | Default generated output path; ignored by Git through `logs/` |

No new runtime dependency is required. The exporter uses plain dictionaries so
the output is easy to inspect, diff, and validate with external FHIR tooling.

## Resource Mapping

| NeuroSight concept | FHIR resource | Notes |
|-------------------|---------------|-------|
| Pseudonymized demo subject | `Patient` | Uses a NeuroSight identifier system and no PHI |
| Model/runtime identity | `Device` | Records the NeuroSight prototype and model status |
| Demo producer | `Organization` | Represents the research-demo source |
| Predicted class | `Observation` | Local code `predicted-class`, value is the predicted class label |
| Confidence | `Observation` | Percent quantity using UCUM `%`; this is a model score, not clinical evidence |
| Human-review requirement | `Observation` | Boolean safety flag |
| Cognitive input panel | `Observation.component[]` | MMSE, MOCA, CDRSB, ADAS11, RAVLT_immediate, RAVLT_learning, FAQ, AGE |
| Report wrapper | `DiagnosticReport` | Groups the Observations and carries a human-readable conclusion |
| Audit trail | `Provenance` | Links generated resources to NeuroSight as the assembler/source |

FHIR does not contain a universal standard code for this exact prototype output,
so project-local codes are intentionally used under:

```text
https://github.com/mohi679/neurosight/fhir/CodeSystem/neurosight
```

That is more honest than inventing fake LOINC/SNOMED mappings.

## Offline Demo Export

Generate a Git-safe synthetic Bundle:

```bash
python3 scripts/fhir_export.py
```

Default output:

```text
logs/fhir/neurosight_demo_bundle.json
```

Print JSON to stdout:

```bash
python3 scripts/fhir_export.py --stdout
```

With Poetry:

```bash
make fhir-export
```

## Backend-Connected Export

Start the backend:

```bash
uvicorn api.main:app --reload --port 8000
```

Then export the actual `/v1/risk-profile` response:

```bash
python3 scripts/fhir_export.py --from-backend --base-url http://localhost:8000 --api-key dev-key
```

The script sends synthetic cognitive scores to the protected risk-profiling route,
receives the backend response payload, and wraps that returned result in a FHIR
Bundle. This proves the exporter is connected to the real API contract instead
of being a UI-only demonstration.

## Example Summary

The CLI prints a concise summary:

```text
FHIR EXPORT PASSED
FHIR version tag: 4.0.1
Bundle id: bundle-...
Resources: Patient, Organization, Device, Observation, Observation, Observation, Observation, DiagnosticReport, Provenance
Wrote: logs/fhir/neurosight_demo_bundle.json
```

The generated `DiagnosticReport.conclusion` always includes the research-demo
boundary:

```text
Predicted class: MCI; confidence: 68.0%; requires human review: True.
NeuroSight is a research prototype. This export demonstrates FHIR
interoperability and is not intended for clinical decision-making.
```

## What This Proves

This feature demonstrates:

- Healthcare interoperability awareness.
- FHIR resource modeling, not just JSON dumping.
- Auditability through `Provenance`.
- Safe pseudonymized export defaults.
- Separation between model output and clinical claims.
- A runnable script that recruiters/reviewers can execute locally.

## What It Does Not Claim

This is not a certified EHR integration and not a medical-device workflow. Real
deployment would still require:

- FHIR profile/Implementation Guide selection.
- Validation against a FHIR validator.
- Terminology review for LOINC/SNOMED/UCUM codes.
- Authentication and consent workflow.
- PHI handling, retention, and audit policies.
- Clinical validation of any exported model result.
