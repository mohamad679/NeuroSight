# Public Repository Guide

This guide defines what should be visible in the public NeuroSight GitHub
repository and how the project should describe its AI capabilities.

The goal is simple: make the repository impressive, honest, and clean for
scholarship, PhD, internship, and job review.

## Keep Public

These files help reviewers understand the project and should remain public:

| Area | Keep public |
|------|-------------|
| Project overview | `README.md`, `PROJECT_STATUS.md`, `MODEL_CARD.md` |
| Architecture | `docs/ARCHITECTURE_OVERVIEW.md`, `docs/IMPLEMENTED_VS_PLANNED.md` |
| Backend proof | `docs/API_CONTRACT_CHECKS.md`, `scripts/api_contract_check.py` |
| Portfolio proof | `docs/DEMO_SCRIPT.md`, `docs/PORTFOLIO_CHECKLIST.md`, `scripts/portfolio_check.py` |
| Safety and governance | `docs/AI_SAFETY_OWASP_GENAI.md`, `SECURITY.md`, `scripts/ai_safety_eval.py` |
| Medical-AI evidence | `docs/MONAI_PIPELINE.md`, `docs/MODALITY_PREPROCESSING.md`, `MODEL_CARD.md` |
| MLOps evidence | MLflow, DVC, drift, ONNX, OpenTelemetry, CI/CD docs and scripts |
| Public examples | synthetic/demo data contracts and generated figures; screenshots only when captured from the running app |
| Legal/project files | `LICENSE`, `.env.example`, `.github/`, `CONTRIBUTING.md` |

## Keep Local Or Ignored

These files are useful during development but should not be part of the public
GitHub story:

| Item | Why it should stay local |
|------|--------------------------|
| `CLAUDE.md` | Assistant-specific planning notes; not reviewer-facing project evidence |
| `.claude/`, `.codex/`, `.cursor/`, `.windsurf/` | Local AI assistant and IDE state |
| `.env`, `.env.local`, `frontend/.env` | Secrets and machine-specific configuration |
| `logs/` | Generated reports; reviewers can regenerate them |
| `checkpoints/` | Large local model artifacts and weights |
| `data/` | Local generated or private data; public repo should not contain patient data |
| `frontend/node_modules/`, `frontend/.next/` | Build/dependency artifacts |
| `__pycache__/`, `.ruff_cache/`, `.DS_Store` | Local caches and OS artifacts |

These are covered by `.gitignore`. If a file was already tracked before being
ignored, remove it from Git tracking with:

```bash
git rm --cached CLAUDE.md
git rm --cached -r logs checkpoints data frontend/.next frontend/node_modules
```

Only run those commands inside a real Git repository after checking `git status`.

## AI Wording Standard

Use precise, defensible language:

| Use this | Avoid this |
|----------|------------|
| medical-AI research prototype | AI doctor |
| neurodiagnostic research workflow | clinical diagnosis product |
| synthetic ADNI-like public data | real ADNI results |
| model score or demo confidence | clinical certainty |
| model-behavior diagnostics | causal biomarker |
| research report | medical report |
| requires specialist review | autonomous decision |
| backend/API contract proof | fully production-ready healthcare system |

Good one-sentence description:

> NeuroSight is a multimodal medical-AI research prototype demonstrating MRI,
> EEG, cognitive fusion, FastAPI, LangGraph orchestration, XAI, MLOps, LLMOps,
> and governance with synthetic public data and explicit clinical limitations.

## Pre-Publish Commands

Run these before pushing a public branch:

```bash
python3 scripts/github_readiness.py --strict
python3 scripts/portfolio_check.py --strict
```

Optional, when the backend is running:

```bash
python3 scripts/portfolio_check.py --strict --backend-smoke required
```

## Public Claims Boundary

It is safe to claim:

- the backend routes are implemented and contract-checked,
- the MRI, EEG, cognitive, fusion, and XAI modules exist,
- the project demonstrates MONAI-aware and EEG-aware architecture,
- the public evaluation uses synthetic ADNI-like data,
- the project includes MLOps, LLMOps, safety, and governance evidence,
- the project is suitable for portfolio and research discussion.

Do not claim:

- clinical diagnostic accuracy,
- validation on real patients,
- ADNI/OASIS results unless authorized data and reproducible evidence are added,
- reliable MRI/EEG clinical interpretation,
- medical-device readiness,
- autonomous clinical decision-making.

## Reviewer-Friendly Structure

When a reviewer opens the repository, they should quickly see:

1. What NeuroSight is.
2. What is implemented.
3. What is synthetic or demo-limited.
4. How to run proof scripts.
5. Why the project is not making clinical claims.

The best path is:

```text
README.md
PROJECT_STATUS.md
docs/ARCHITECTURE_OVERVIEW.md
docs/API_CONTRACT_CHECKS.md
docs/PORTFOLIO_CHECKLIST.md
MODEL_CARD.md
```
