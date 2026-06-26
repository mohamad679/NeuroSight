# NeuroSight Project Status

NeuroSight is a student research and portfolio project. It is designed to show
knowledge of multimodal AI systems, medical-AI engineering constraints, LLM
orchestration, MLOps, LLMOps, safety, and documentation discipline.

It is not clinical software. It is not a medical device. It must not be used for
diagnosis, treatment, triage, prescribing, or patient management.

## One-Sentence Status

NeuroSight is a production-shaped research prototype with real code paths,
synthetic public data, explicit safety/governance controls, and clear clinical
limitations.

## What Is Real

| Area | Current status | Evidence |
|------|----------------|----------|
| Backend API | Implemented FastAPI routes for risk profiling, legacy diagnosis naming, upload, KG, XAI, eval, models, health, and governance | `api/main.py`, `docs/API_EXAMPLES.md` |
| API contract proof | Implemented in-process FastAPI route, SSE, auth, and middleware checks | `scripts/api_contract_check.py`, `docs/API_CONTRACT_CHECKS.md` |
| GitHub release readiness | Implemented public-file, disclosure, artifact, secret-pattern, workflow, visual-evidence, and license audit | `scripts/github_readiness.py`, `docs/GITHUB_RELEASE_READINESS.md` |
| Public repository hygiene | Implemented public/private file policy and AI wording standards | `docs/PUBLIC_REPOSITORY_GUIDE.md`, `.gitignore` |
| Multimodal model architecture | Implemented MRI, EEG, cognitive, fusion, and XAI modules | `neurosight/models/` |
| MONAI MRI awareness | Implemented/documented 3D MRI pipeline shape and constraints | `docs/MONAI_PIPELINE.md`, `neurosight/models/mri.py` |
| EEG pathway | Implemented/documented EEG encoder and upload pathway | `neurosight/models/eeg.py`, `docs/MODALITY_PREPROCESSING.md` |
| Cognitive model path | Implemented and used in synthetic public evaluation | `neurosight/models/cognitive.py`, `evaluation/results.json` |
| LangGraph workflow | Implemented supervisor-routed graph with report and safety nodes | `neurosight/agents/orchestrator.py`, `scripts/langgraph_workflow.py` |
| Knowledge graph concept | Implemented temporal KG demo and similar-patient retrieval | `knowledge_graph.py`, KG endpoints |
| Safety/governance | Implemented model card, OWASP GenAI safety checks, supply-chain audit, quality gate | `MODEL_CARD.md`, `docs/AI_SAFETY_OWASP_GENAI.md`, `scripts/quality_gate.py` |
| Interoperability awareness | Implemented FHIR export and DICOM/DICOMweb manifest tooling | `docs/FHIR_EXPORT.md`, `docs/DICOM_DICOMWEB.md` |
| MLOps/LLMOps roadmap | Implemented runnable docs/scripts for MLflow, DVC, ONNX, drift, OTEL, CI gates | `docs/`, `scripts/` |

## What Is Synthetic Or Demo-Limited

| Area | Current limitation |
|------|--------------------|
| Dataset | Public training/evaluation uses synthetic ADNI-like data only |
| Clinical validity | No external validation on real patient cohorts |
| MRI/EEG performance | Public metrics do not prove real MRI/EEG diagnostic performance |
| Six-class output | FTD, LBD, and VD are synthetic demo placeholders in the public six-class setup |
| Confidence | ECE remains high; confidence is a demo model score, not clinical certainty |
| Hugging Face demo | Public demo may use random or demo weights depending on runtime configuration |
| Agent intelligence | LangGraph workflow is real orchestration, but not an autonomous clinical reasoning system |

## What This Project Is Meant To Prove

- Ability to design an end-to-end AI system rather than a notebook-only model.
- Understanding of multimodal ML: MRI, EEG, cognitive features, fusion, and missing-modality handling.
- Familiarity with medical-AI constraints: data privacy, model-card disclosure, no clinical overclaims.
- LLMOps awareness: LangGraph orchestration, prompt safety, traceability, and governance gates.
- MLOps awareness: MLflow-style registry, DVC-style provenance, ONNX export, drift monitoring, CI/CD.
- Backend/frontend integration: FastAPI, upload endpoints, local UI, and dashboard surfaces.
- Professional documentation: architecture, safety, model cards, quality gates, and reproducible scripts.

## What This Project Does Not Claim

- It does not claim clinical accuracy.
- It does not claim diagnostic validity on real patients.
- It does not claim regulatory readiness.
- It does not claim that MRI/EEG uploaded files are clinically interpreted correctly.
- It does not claim that synthetic-data metrics transfer to real-world cohorts.
- It does not replace specialist review.

## How A Reviewer Can Verify The Project

Run the one-command portfolio proof:

```bash
python3 scripts/portfolio_check.py --strict
```

This writes a reviewer-facing aggregate report to
`logs/portfolio/neurosight_portfolio_check_report.json`.

The underlying offline checks can also be run individually:

```bash
python3 scripts/api_contract_check.py --strict
python3 scripts/github_readiness.py --strict
python3 scripts/quality_gate.py --strict
python3 scripts/model_card_check.py --strict
python3 scripts/ai_safety_eval.py --strict
python3 scripts/langgraph_workflow.py --scenario blocked
```

Optional backend smoke path:

```bash
python3 scripts/smoke_backend.py
python3 scripts/portfolio_check.py --strict --backend-smoke required
```

The generated reports are written under `logs/`, which is intentionally ignored
by Git.

## Current Defensible Positioning

The strongest way to present NeuroSight is:

> A production-shaped, GitHub-safe medical-AI research prototype that shows
> multimodal architecture, agent orchestration, MLOps, LLMOps, safety, and
> scientific honesty, while explicitly avoiding clinical claims.

This positioning is appropriate for PhD scholarship applications, research
internships, AI engineering roles, and ML/LLMOps portfolio review.
