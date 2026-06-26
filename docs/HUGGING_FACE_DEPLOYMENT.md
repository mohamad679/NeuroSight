# Hugging Face Full-Stack Space Deployment

NeuroSight deploys to Hugging Face as a Docker Space. The Docker image builds
the Next.js frontend, serves the static export from the FastAPI backend, and
keeps all API routes available on the same Space origin.

## Security Boundary

Do not commit Hugging Face tokens, API keys, private patient data, checkpoints,
or generated real-cohort outputs. The deployment script reads `HF_TOKEN` from
the local environment or from the Hugging Face token cache and never writes it
to the repository.

If a token has been pasted into chat, terminal history, or logs, rotate it in
Hugging Face before treating the deployment as secure.

## Deploy

```bash
export HF_TOKEN=...
export HF_USERNAME=mohi679
export HF_SPACE_NAME=neurosight
python3 scripts/deploy_hf_space.py
```

The script stages only public-safe files under `.deploy/hf_space_stage`, creates
or updates `mohi679/neurosight` as a Docker Space, sets non-secret demo
variables, and uploads the staged folder.

Expected URLs:

```text
Space repo: https://huggingface.co/spaces/mohi679/neurosight
App URL:    https://mohi679-neurosight.hf.space
Health:     https://mohi679-neurosight.hf.space/healthz
API docs:   https://mohi679-neurosight.hf.space/docs
```

## What Runs In The Space

- `GET /` serves the static Next.js frontend when bundled.
- `GET /healthz` reports backend health and model/data status.
- `POST /v1/risk-profile` is the preferred non-clinical demo endpoint.
- `POST /v1/risk-profile/stream` serves SSE workflow events.
- Legacy `/v1/diagnose*` routes remain available for backward compatibility.

The Space runs in public demo mode with synthetic data only. It is not a
clinical product and does not contain real patient data.
