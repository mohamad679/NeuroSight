# NeuroSight Running Modes

NeuroSight is a research demo. The public GitHub/Hugging Face version must be reproducible without private medical data, while still showing how an ADNI-style pipeline would be wired.

## Phase 1 Contract

Phase 1 makes the runtime and capability contract explicit:

- `GET /healthz` reports backend health, model load state, runtime mode, class scope, capability coverage, upload limits, KG status, and production model metadata.
- `GET /` reports the same public backend contract in a compact landing payload.
- The local UI reads this metadata and shows which backend capabilities are implemented and exposed.
- `GET /v1/data/status` and `GET /v1/data/demo-patients` expose safe demo/ADNI-style data readiness.
- No backend logic, model architecture, API request body, routing behavior, or data flow is changed by this contract.

## Runtime Modes

| Mode | Env value | Purpose |
|---|---|---|
| Public demo | `NEUROSIGHT_RUNTIME_MODE=demo` | Default GitHub-safe mode. Uses synthetic/mock/public-demo data only. |
| ADNI-style ingestion | `NEUROSIGHT_RUNTIME_MODE=adni_style` | Operator-supplied ADNI-style CSV and modality files. Private data remains outside the repo. |
| Research checkpoint | `NEUROSIGHT_RUNTIME_MODE=research` | Future externally trained checkpoints and validated datasets. |

These modes are metadata and operating constraints in Phase 1. They do not change prediction logic yet.

## Class Modes

| Mode | Env value | Scientific meaning |
|---|---|---|
| 3-class ADNI-style | `NEUROSIGHT_CLASS_MODE=three_class_adni` | Normal, MCI, and AD are the credible ADNI-style target workflow. |
| 6-class demo | `NEUROSIGHT_CLASS_MODE=six_class_demo` | Current prototype output space: Normal, MCI, AD, FTD, LBD, VD. FTD/LBD/VD are synthetic demo placeholders unless additional datasets and checkpoints are added. |

The current backend model still has six output classes. Phase 1 documents the scientific scope honestly instead of pretending that ADNI validates every class.

## Data Policy

The public repository should include only synthetic, mock, or legally shareable public demo data. Real ADNI records require formal access approval and must not be committed.

Expected optional paths for authorized local or private deployments:

```text
data/ADNIMERGE.csv
data/mri/{RID}.npy
data/eeg/{RID}.npy
```

In demo mode, patient lookup falls back to `data/ADNIMERGE_synthetic.csv` unless `NEUROSIGHT_PATIENT_CSV_PATH` is explicitly configured. See [`DATA_DEMO_PIPELINE.md`](DATA_DEMO_PIPELINE.md) for the Phase 2 data contract.

## Upload Formats

MRI upload endpoint:

- `.npy`: 3D volume shaped `(D,H,W)` or `(1,D,H,W)`
- `.nii` or `.nii.gz`: NIfTI volume loaded with nibabel
- `.zip`: DICOM series, one study per zip archive

EEG upload endpoint:

- `.npy`: `(channels,time)` or `(epochs,channels,time)`
- `.edf`: preprocessed with MNE when available

Cognitive endpoint:

- JSON object with the required canonical cognitive schema: MMSE, MOCA, CDRSB, ADAS11, RAVLT_immediate, RAVLT_learning, FAQ, and AGE. Missing or obsolete values are rejected with field-specific validation errors.

## Deployment Shape

```text
Browser
  -> local_ui static app served by app_local.py
  -> app_local.py proxy with X-API-Key
  -> Hugging Face/FastAPI backend
  -> model encoders, fusion, KG, evaluation, registry, XAI
```

The local frontend intentionally avoids heavy ML dependencies. The backend owns model execution and data processing.

## Definition Of Done For Phase 1

- Runtime and class scope are visible from backend health.
- Capability metadata is visible in the local UI.
- Environment variables are documented in `.env.example`.
- Tests assert the health/root contract.
- The project remains in demo/untrained mode unless a future phase adds real checkpoint loading.
