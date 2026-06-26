# NeuroSight Demo Data Pipeline

Phase 2 makes the data layer runnable and honest for a public GitHub demo.

## Goal

NeuroSight supports two data stories:

- A reproducible synthetic ADNI-like demo that anyone can run.
- An ADNI-style local adapter path for authorized users who supply their own private files.

The repository must not contain real private ADNI records. Public demo data is synthetic and cannot support clinical claims.

## Public Demo Data

Default demo CSV:

```text
data/ADNIMERGE_synthetic.csv
```

The CSV follows an ADNI-style tabular schema:

```text
RID, DX_bl, AGE, PTGENDER, MMSE, CDRSB, ADAS11, RAVLT_immediate, RAVLT_learning, FAQ, MOCA
```

In `NEUROSIGHT_RUNTIME_MODE=demo`, the backend uses this synthetic CSV for patient lookup unless `NEUROSIGHT_PATIENT_CSV_PATH` is set.

## Authorized ADNI-Style Data

For private/authorized preprocessing, place operator-supplied files under
`data/private/` or another secure local path and run:

```bash
python3 scripts/prepare_adni_cognitive.py \
  --input-csv data/private/adni/ADNIMERGE.csv \
  --output-dir data/processed/adni_cognitive
```

The adapter validates required cognitive columns, normalizes common ADNI-style
names, blocks direct label-leakage and future/post-diagnosis fields, creates
subject-disjoint train/validation/test splits, and writes
`dataset_summary.json`. Do not commit generated outputs from real cohorts.
See [`PRIVATE_DATA_ADAPTERS.md`](PRIVATE_DATA_ADAPTERS.md).

For private/authorized deployments, point the backend at operator-supplied files:

```bash
export NEUROSIGHT_RUNTIME_MODE=adni_style
export NEUROSIGHT_PATIENT_CSV_PATH=/secure/path/ADNIMERGE.csv
export NEUROSIGHT_MRI_DIR=/secure/path/mri
export NEUROSIGHT_EEG_DIR=/secure/path/eeg
```

Expected modality layout:

```text
{NEUROSIGHT_MRI_DIR}/{RID}.npy
{NEUROSIGHT_EEG_DIR}/{RID}.npy
```

MRI arrays should be 3D volumes shaped `(D,H,W)` or `(1,D,H,W)`. EEG arrays should be `(channels,time)` or `(epochs,channels,time)`.

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/data/status` | Reports CSV readiness, schema validity, row counts, label distribution, modality file counts, privacy notice, and recommended demo patient ID. |
| `GET /v1/data/demo-patients?limit=12` | Returns a bounded, privacy-conscious list of demo rows for UI selection. |
| `POST /v1/risk-profile/patient/{patient_id}` | Preferred risk profiling evaluation from the configured CSV plus optional MRI/EEG files. |
| `POST /v1/diagnose/patient/{patient_id}` | Legacy/deprecated alias retained for backward compatibility. |

All `/v1/data/*` endpoints require `X-API-Key` outside test mode.

## Class Scope

ADNI-style workflows should focus on:

- normal
- mci
- ad

The public six-class demo also contains:

- ftd
- lbd
- vd

Those three classes are synthetic placeholders in the public demo. True validation would require additional public or authorized datasets and trained checkpoints.

## What Phase 2 Does Not Do

Phase 2 does not train a clinical checkpoint, add private data, validate real-world performance, or change the prediction logic. It makes the data mode explicit, inspectable, and runnable.

Phase 3 extends this with an explicit MRI/EEG upload and preprocessing contract in [`MODALITY_PREPROCESSING.md`](MODALITY_PREPROCESSING.md).
