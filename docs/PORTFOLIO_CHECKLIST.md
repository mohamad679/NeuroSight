# Portfolio Checklist

This checklist prepares NeuroSight for GitHub resume review. The goal is to make
the project defensible to a reviewer who asks, "Is this a real AI systems
project, or only a polished interface?"

## One-Command Proof

Run the portfolio check before publishing or sending the repository:

```bash
python3 scripts/portfolio_check.py --strict
```

Optional live backend proof:

```bash
uvicorn api.main:app --reload --port 8000
python3 scripts/portfolio_check.py --strict --backend-smoke required
```

Main artifact:

```text
logs/portfolio/neurosight_portfolio_check_report.json
```

This artifact is intentionally ignored by Git. Generate it fresh for demos,
interviews, or release screenshots.

## Reviewer Evidence Map

| Reviewer question | Evidence to show |
|-------------------|------------------|
| Is there a real backend? | `api/main.py`, `scripts/api_contract_check.py`, `scripts/smoke_backend.py`, `docs/API_EXAMPLES.md` |
| Is MONAI actually represented? | `neurosight/models/mri.py`, `docs/MONAI_PIPELINE.md` |
| Is EEG represented? | `neurosight/models/eeg.py`, `docs/MODALITY_PREPROCESSING.md` |
| Is there agent orchestration? | `neurosight/agents/orchestrator.py`, `scripts/langgraph_workflow.py` |
| Are there safety controls? | `docs/AI_SAFETY_OWASP_GENAI.md`, `scripts/ai_safety_eval.py` |
| Are claims honest? | `PROJECT_STATUS.md`, `MODEL_CARD.md`, `docs/IMPLEMENTED_VS_PLANNED.md` |
| Is there a safe real-data path? | `scripts/prepare_adni_cognitive.py`, `docs/PRIVATE_DATA_ADAPTERS.md`, `tests/test_prepare_adni_cognitive.py` |
| Is the repo publishable? | `scripts/github_readiness.py`, `docs/GITHUB_RELEASE_READINESS.md`, `LICENSE` |
| Is there MLOps maturity? | `docs/MLFLOW_REGISTRY.md`, `docs/DVC_PROVENANCE.md`, `docs/ONNX_RUNTIME_EXPORT.md` |
| Is clone-and-run reproducible? | `requirements.lock`, `Makefile`, `.github/workflows/ci.yml`, `Dockerfile`, `docs/REPRODUCIBILITY.md` |
| Is there operational thinking? | `docs/OPENTELEMETRY_OBSERVABILITY.md`, `docs/CI_CD_QUALITY_GATE.md` |
| Is there interoperability awareness? | `docs/FHIR_EXPORT.md`, `docs/DICOM_DICOMWEB.md` |

## GitHub Readiness Checklist

- [ ] `README.md` clearly states that public data is synthetic.
- [ ] `PROJECT_STATUS.md` separates real implementation from demo limits.
- [ ] `docs/IMPLEMENTED_VS_PLANNED.md` does not overclaim future work.
- [ ] `MODEL_CARD.md` states intended use, limitations, data provenance, and
      safety boundaries.
- [ ] `docs/DEMO_SCRIPT.md` can guide a five-minute interview walkthrough.
- [ ] `python3 scripts/api_contract_check.py --strict` passes.
- [ ] `make install` completes in a fresh Python 3.11 virtual environment.
- [ ] `make verify` passes from a clean clone.
- [ ] `python3 scripts/github_readiness.py --strict` passes.
- [ ] `python3 scripts/portfolio_check.py --strict` passes.
- [ ] `.github/workflows/quality_gate.yml` is present.
- [ ] Screenshots show warnings and system surfaces, not only attractive UI.
- [ ] No API keys, Hugging Face tokens, patient data, or private dataset files
      are committed.
- [ ] `data/private/` remains gitignored, and real-cohort outputs from
      `data/processed/` are not committed.
- [ ] Local assistant/IDE planning files such as `CLAUDE.md`, `.claude/`,
      `.codex/`, `.cursor/`, and `.windsurf/` are ignored or removed from Git tracking.
- [ ] The repository description uses "research prototype" or "portfolio
      project", not "clinical diagnosis product".

## Suggested GitHub README Framing

Strong:

> Production-shaped medical-AI research prototype demonstrating multimodal ML,
> FastAPI, LangGraph orchestration, XAI, MLOps, LLMOps, safety, and governance
> with synthetic public data and explicit clinical limitations.

Weak:

> AI doctor for neurological diagnosis.

The first version is defensible. The second creates ethical, technical, and
regulatory risk.

## Optional Screenshots To Include

Use screenshots only when they are captured from the running local app:

| Screenshot | Why it helps |
|------------|--------------|
| Diagnosis result with warning | Shows the user-facing safety boundary |
| Stream timeline | Shows agent workflow behavior |
| Eval tab | Shows metrics and synthetic-data disclosure |
| Models tab | Shows registry/checkpoint lifecycle thinking |
| XAI tab | Shows explainability and method availability |
| System tab | Shows health, data, modality, and governance status |
| Portfolio check terminal output | Shows reproducible verification |

## Interview Walkthrough Order

1. Start with `PROJECT_STATUS.md`.
2. Run `python3 scripts/portfolio_check.py --strict`.
3. Open `docs/ARCHITECTURE_OVERVIEW.md`.
4. Show `api/main.py` and one route group.
5. Show `scripts/api_contract_check.py` and its generated JSON report.
6. Show `scripts/github_readiness.py` and explain what it protects before publishing.
7. Show `neurosight/models/mri.py`, `neurosight/models/eeg.py`, and
   `neurosight/models/fusion.py`.
8. Show `scripts/langgraph_workflow.py` output.
9. Open the UI last as the visual control surface.

## What To Improve Next

The strongest next changes are:

- Add a small real open-data MRI or EEG preprocessing example with documented
  source and license.
- Extend the private adapter family only with local workflows that keep raw
  clinical data outside the public repository.
- Add unit tests for upload routes and diagnosis response normalization.
- Add a reproducible training/evaluation command that creates a small artifact
  from synthetic data in CI.
- Add frontend integration tests for the diagnosis and stream flows.
- Add a short video demo linked from the README.

## Defensible Final Position

NeuroSight should be presented as:

> A GitHub-safe AI systems portfolio project that demonstrates architecture,
> implementation discipline, safety, and roadmap maturity for medical-AI
> research, without claiming clinical readiness.
