# API Contract Checks

Phase 4 adds an in-process FastAPI contract check for reviewers and CI-quality
hardening. The goal is to prove that NeuroSight has a real backend surface with
testable response contracts, not only a frontend presentation layer.

## Runnable Command

```bash
python3 scripts/api_contract_check.py --strict
```

With Poetry:

```bash
make api-contract-check
```

Default report:

```text
logs/api_contract/neurosight_api_contract_report.json
```

The report is generated under `logs/`, which is intentionally ignored by Git.

## What It Checks

The script imports the FastAPI app directly and exercises it with
`fastapi.testclient.TestClient` under `APP_ENV=test`.

| Area | Contract evidence |
|------|-------------------|
| Health/runtime | `GET /`, `GET /healthz` |
| Data readiness | `GET /v1/data/status`, `GET /v1/data/demo-patients` |
| Modalities | `GET /v1/modalities/status` |
| Governance | `GET /v1/governance/status`, `GET /v1/demo/readiness` |
| Risk profile | `POST /v1/upload/cognitive`, `POST /v1/risk-profile`, legacy `POST /v1/diagnose` |
| Streaming | `POST /v1/risk-profile/stream` and legacy `POST /v1/diagnose/stream` with SSE event parsing and `[DONE]` validation |
| Knowledge graph | `POST /v1/kg/query`, history, and similar-patient routes |
| Evaluation | metrics, history, and report routes |
| Models | registry, production model, and checkpoint status routes |
| XAI | status and cognitive explanation routes |
| Security | production-mode `X-API-Key` enforcement |
| Observability | `X-Request-ID` and `X-Process-Time` middleware headers |

## Why This Matters

For a GitHub portfolio reviewer, this answers three practical questions:

1. Does the backend actually import and run?
2. Do the key endpoints return stable JSON/SSE shapes?
3. Are governance, auth, and observability concerns represented in code?

This gives more credible evidence than screenshots alone.

## What It Does Not Prove

The contract check intentionally avoids overclaiming. It does not prove:

- clinical validity,
- real-patient performance,
- production deployment health,
- GPU inference throughput,
- expensive MRI/EEG upload performance,
- regulatory readiness.

MRI and EEG are covered through architecture, preprocessing contracts, modality
status, and dedicated model files. Full MRI/EEG performance proof still requires
authorized cohorts, trained checkpoints, and external validation.

## Relationship To Portfolio Proof

`scripts/portfolio_check.py` runs this API contract check as a required
sub-check. That makes Phase 4 part of the one-command reviewer path:

```bash
python3 scripts/portfolio_check.py --strict
```

If the FastAPI server is also running, the portfolio check can additionally
require live HTTP smoke proof:

```bash
python3 scripts/portfolio_check.py --strict --backend-smoke required
```

The in-process contract check and live smoke check are complementary:

| Check | Best for |
|-------|----------|
| `api_contract_check.py` | Fast, deterministic route and response-shape proof |
| `smoke_backend.py` | Real network/deployment connectivity proof |
