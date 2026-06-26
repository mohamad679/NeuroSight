# NeuroSight Demo Script

This demo script is for portfolio review, PhD scholarship interviews, and AI
engineering conversations. It keeps the presentation honest: NeuroSight is a
production-shaped research prototype with synthetic public data, real system
components, and explicit medical-AI limits.

## Demo Goal

Show that NeuroSight is more than a UI shell:

- FastAPI backend routes exist for risk profiling, upload, KG, eval, model registry,
  XAI, system health, and governance.
- The route contracts are checked in-process with a runnable FastAPI contract
  script, including SSE and auth behavior.
- The final GitHub package is audited for public-file hygiene, disclosures,
  ignored artifacts, obvious secret patterns, workflows, visuals, and license
  alignment.
- Model modules exist for MRI, EEG, cognitive scoring, fusion, and
  explainability.
- Agent orchestration is implemented with a LangGraph-style workflow and a
  deterministic safety path.
- The project includes MLOps, LLMOps, interoperability, observability, safety,
  and CI/CD evidence.
- Clinical claims are intentionally limited because public data is synthetic.

## Five-Minute Reviewer Path

### 1. Position the Project

Suggested wording:

> NeuroSight is a student research and portfolio project. It demonstrates an
> end-to-end medical-AI system architecture across multimodal ML, FastAPI,
> LangGraph orchestration, XAI, MLOps, LLMOps, and governance. It is not a
> clinical device, and the public metrics are synthetic-data evidence only.

Open:

- `PROJECT_STATUS.md`
- `docs/IMPLEMENTED_VS_PLANNED.md`
- `docs/ARCHITECTURE_OVERVIEW.md`

### 2. Run Offline Portfolio Proof

Use this when a reviewer wants evidence without starting servers:

```bash
python3 scripts/portfolio_check.py --strict
```

Expected behavior:

- Runs the quality gate.
- Runs the API contract check across the backend route surface.
- Runs the GitHub release-readiness audit.
- Checks the model-card disclosure contract.
- Runs the OWASP GenAI safety regression script.
- Generates two LangGraph workflow traces.
- Skips backend smoke automatically if no local API is running.

Generated report:

```text
logs/portfolio/neurosight_portfolio_check_report.json
```

### 3. Show Architecture Evidence

Open these files and connect them to the demo:

| Topic | Evidence |
|-------|----------|
| MRI/MONAI | `docs/MONAI_PIPELINE.md`, `neurosight/models/mri.py` |
| EEG | `neurosight/models/eeg.py`, `docs/MODALITY_PREPROCESSING.md` |
| Fusion | `neurosight/models/fusion.py` |
| Agents | `docs/LANGGRAPH_AGENT_WORKFLOW.md`, `neurosight/agents/orchestrator.py` |
| Backend | `api/main.py`, `docs/API_EXAMPLES.md` |
| API contracts | `scripts/api_contract_check.py`, `docs/API_CONTRACT_CHECKS.md` |
| GitHub readiness | `scripts/github_readiness.py`, `docs/GITHUB_RELEASE_READINESS.md`, `LICENSE` |
| Safety | `docs/AI_SAFETY_OWASP_GENAI.md`, `MODEL_CARD.md` |
| CI/CD | `docs/CI_CD_QUALITY_GATE.md`, `.github/workflows/quality_gate.yml` |

### 4. Optional Live Backend Proof

Start the API in one terminal:

```bash
uvicorn api.main:app --reload --port 8000
```

Then run:

```bash
python3 scripts/smoke_backend.py
python3 scripts/portfolio_check.py --strict --backend-smoke required
```

Use this to prove backend connectivity. If the backend is not running, keep the
demo in offline mode and explicitly say the live API path is optional.

### 5. UI Walkthrough

Use the UI only after explaining the backend proof. The UI is a dashboard for
the backend and project workflow, not the core evidence by itself.

| Tab | What to Show | What to Say |
|-----|--------------|-------------|
| Risk Profiling | Cognitive sliders, optional MRI/EEG uploads, class, confidence, report | Demonstrates the risk profiling request and response contract |
| Stream | Agent timeline and event log | Demonstrates orchestration and streaming behavior |
| Patient | Demo patient workflow | Shows dataset lookup and patient-centered result rendering |
| Eval | Metrics, history, eval report | Shows evaluation surface and explicit synthetic-data warning |
| Models | Registry and checkpoint status | Shows model lifecycle awareness |
| XAI | XAI availability and patient explanation | Shows explainability contract and modality limits |
| System | Health, data, modalities, governance | Shows operational readiness thinking |
| KG | Query, history, similar patients | Shows temporal graph and retrieval concepts |

## What Each Main Risk Profiling Mosaic Means

| Mosaic | Purpose |
|--------|---------|
| Predicted Class | The model output label for the current request |
| Confidence | Demo model score, not clinical certainty |
| Modality Attention | Relative model weighting across cognitive, MRI, and EEG inputs |
| Cognitive Feature Importance | Which tabular cognitive features influenced the model most |
| Clinical Report | Human-readable research report with limitations and safety notes |
| Demo Notice | Explicit boundary that the output is research/demo only |

## Talking Points For Interviews

- I separated what is implemented from what is planned to avoid overclaiming.
- I added a portfolio check so reviewers can verify evidence quickly.
- I included model-card, safety, supply-chain, and CI/CD gates because medical
  AI needs governance, not only model code.
- I kept clinical performance claims conservative because the public dataset is
  synthetic and the MRI/EEG paths need authorized real cohorts and checkpoints.
- I used the UI to explain the system, but the defensible evidence lives in the
  backend modules, scripts, docs, and generated reports.

## Do Not Claim

- Do not claim clinical diagnostic accuracy.
- Do not claim the model is trained on private ADNI records unless authorized
  data and reproducible evidence are added.
- Do not claim MRI or EEG predictions are clinically validated.
- Do not claim LangGraph agents perform autonomous clinical reasoning.
- Do not claim regulatory readiness.

## If A Reviewer Asks About Real Medical Value

Suggested answer:

> The current repository is a research prototype and portfolio demonstration.
> It proves that I understand the system architecture, data contracts, model
> lifecycle, safety controls, and deployment path. To make clinical claims, the
> next step would be authorized real cohorts, locked preprocessing, external
> validation, calibrated uncertainty, clinical review, and regulatory controls.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Backend smoke skipped | API is not running | Start `uvicorn api.main:app --reload --port 8000` |
| Backend smoke fails | Missing API key or route failure | Check `/healthz` and `NEUROSIGHT_API_KEY` |
| UI shows synthetic warning | Expected public demo mode | Explain the demo boundary |
| MRI/EEG result looks uncertain | No validated multimodal checkpoint | Explain current limitation and MONAI/EEG path evidence |
| Quality gate fails | Missing required docs, scripts, or workflows | Open the JSON report under `logs/` |
