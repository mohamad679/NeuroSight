# NeuroSight Checkpoint And Evaluation Contract

Phase 4 makes checkpoint and evaluation artifacts explicit without turning synthetic-demo metrics into clinical claims.

## Runtime Loading

By default, NeuroSight initializes model weights in process and reports demo/untrained mode. This keeps public demos lightweight and honest.

To opt into checkpoint loading in a private or research deployment:

```bash
export NEUROSIGHT_CHECKPOINT_PATH=checkpoints/best_fusion.pt
export NEUROSIGHT_LOAD_CHECKPOINT=true
```

The backend will attempt to load:

- `mri_state`
- `eeg_state`
- `cog_state`
- `model_state`

If loading fails, `/healthz` and `/v1/models/checkpoint/status` report the error instead of silently claiming success.

## Reporting Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/models/checkpoint/status` | Reports checkpoint existence, size, runtime load state, load error, registry summary, evaluation artifact, model-card artifact, and scientific-claims policy. |
| `GET /v1/eval/report` | Reports persisted evaluation/model-card status and synthetic-data claims disclosure. |
| `GET /v1/models` | Lists registered model runs. |
| `GET /v1/models/production` | Reports the production registry entry, if one is promoted. |

## Scientific Claim Policy

Current public artifacts are portfolio evidence, not medical evidence:

- Training/evaluation data is synthetic ADNI-like data.
- Real private ADNI data is not included.
- MRI/EEG branches are structurally present, but public training remains cognitive/synthetic-centered.
- Metrics must not be described as clinical performance.
- Every output still requires specialist review.

## Definition Of Done For Phase 4

- Checkpoint availability is visible from API health and model endpoints.
- Runtime load status is explicit and cannot be confused with artifact presence.
- Evaluation/model-card artifacts are surfaced with limitations.
- The UI shows checkpoint status, registry status, and evaluation report status.
- Tests protect the default no-clinical-claims behavior.
