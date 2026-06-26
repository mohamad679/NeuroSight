# NeuroSight Demo Runbook

This runbook is the final Phase 6 path for presenting NeuroSight as a polished, honest research demo.

## Public Synthetic Demo

Use this mode for GitHub, portfolio review, and public Hugging Face demos. It does not require private ADNI data.

```bash
export APP_ENV=local
export NEUROSIGHT_RUNTIME_MODE=demo
export NEUROSIGHT_CLASS_MODE=six_class_demo
export NEUROSIGHT_API_KEY=dev-key

uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
export BACKEND_URL=http://127.0.0.1:8000
export NEUROSIGHT_API_KEY=dev-key
export PORT=7863
python3 app_local.py
```

Open:

```text
http://127.0.0.1:7863
```

Recommended presentation flow:

1. Overview: click `Check Backend`.
2. Demo: click `Load Demo Readiness`.
3. Data: click `Load Data Status`, then `Load Demo Patients`.
4. Diagnosis: paste the recommended patient ID or use cognitive scores, then click `Analyze`.
5. XAI: click `Load XAI Status`, then generate cognitive XAI.
6. Evaluation: click `Load Report`.
7. Models: click `Load Checkpoint`.
8. Trust: click `Load Trust Status`.

Expected result: the demo should be `demo_ready` or `demo_ready_with_warnings`. Warnings are acceptable when they honestly disclose missing private ADNI data or disabled checkpoint loading.

## ADNI-style Private Demo

Use this mode only after data-use approval. Keep all real data outside the repository.

```bash
export APP_ENV=local
export NEUROSIGHT_RUNTIME_MODE=adni_style
export NEUROSIGHT_CLASS_MODE=three_class_adni
export NEUROSIGHT_API_KEY=replace-with-a-real-secret
export NEUROSIGHT_PATIENT_CSV_PATH=/secure/path/ADNIMERGE.csv
export NEUROSIGHT_MRI_DIR=/secure/path/mri
export NEUROSIGHT_EEG_DIR=/secure/path/eeg
```

Optional trained-checkpoint mode:

```bash
export NEUROSIGHT_CHECKPOINT_PATH=/secure/path/best_fusion.pt
export NEUROSIGHT_LOAD_CHECKPOINT=true
```

The ADNI-style scientifically credible scope is Normal/MCI/AD. FTD/LBD/VD remain placeholders unless additional datasets and validation are added.

## Final QA Checklist

- `/healthz` returns runtime, data, modalities, checkpoint, XAI, governance, and demo-readiness contracts.
- `/v1/demo/readiness` reports a recommended patient ID and launch checklist.
- The local UI loads `index.html` and `app.js` from `app_local.py`.
- Diagnosis responses always require specialist review.
- XAI and evaluation screens show no clinical-use claims.
- The model card, README, and trust docs disclose synthetic data and limitations.
- No private ADNI files are committed.
