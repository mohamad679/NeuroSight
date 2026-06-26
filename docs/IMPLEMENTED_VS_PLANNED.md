# Implemented Vs Planned

This document separates what NeuroSight currently implements from what is
planned for future research. It exists so reviewers can evaluate the project
fairly and avoid confusing production-shaped architecture with clinical
readiness.

## Status Labels

| Label | Meaning |
|-------|---------|
| Implemented | Code exists and can be inspected or run locally |
| Implemented, demo-limited | Code exists, but public data/checkpoints are synthetic or incomplete |
| Documented roadmap | Design is documented, but not yet a complete production implementation |
| Out of scope | Not claimed by this student project |

## Core AI And Backend

| Component | Status | Evidence | What it demonstrates | Next defensible step |
|-----------|--------|----------|----------------------|----------------------|
| FastAPI backend | Implemented | `api/main.py`, `docs/API_EXAMPLES.md` | API design, contracts, upload handling, service boundaries | Add broader integration tests and deployment smoke checks |
| API contract checks | Implemented | `scripts/api_contract_check.py`, `docs/API_CONTRACT_CHECKS.md` | In-process route, SSE, auth, and middleware verification | Add frontend integration and browser smoke checks |
| GitHub release readiness | Implemented | `scripts/github_readiness.py`, `docs/GITHUB_RELEASE_READINESS.md`, `LICENSE` | Portfolio publishability, secret/artifact hygiene, and reviewer evidence checks | Add release tag and recorded demo |
| Public repository hygiene | Implemented | `docs/PUBLIC_REPOSITORY_GUIDE.md`, `.gitignore` | Local/private file policy and professional AI wording standard | Remove any already-tracked local assistant files before publishing |
| Cognitive model path | Implemented | `neurosight/models/cognitive.py`, `evaluation/results.json` | Train/evaluate loop and tabular neuropsychology features | Validate against authorized real cohort |
| MRI model architecture | Implemented, demo-limited | `neurosight/models/mri.py`, `docs/MONAI_PIPELINE.md` | MONAI-aware 3D imaging path plus compact CPU-safe 3D CNN fallback | Add real MRI preprocessing QC and trained checkpoint |
| EEG model architecture | Implemented, demo-limited | `neurosight/models/eeg.py`, `docs/MODALITY_PREPROCESSING.md` | EEG encoder and upload-aware pathway | Add montage/artifact QC and trained EEG validation |
| Fusion model | Implemented, demo-limited | `neurosight/models/fusion.py` | Cross-modal attention and missing-modality strategy | Evaluate with real multimodal cohorts |
| XAI methods | Implemented, demo-limited | `neurosight/models/xai.py`, `docs/TRUST_EXPLAINABILITY.md` | Explainability API and interpretation boundaries | Add validated MRI/EEG explanation workflows |

## Agentic AI And LLMOps

| Component | Status | Evidence | What it demonstrates | Next defensible step |
|-----------|--------|----------|----------------------|----------------------|
| LangGraph workflow | Implemented | `neurosight/agents/orchestrator.py`, `docs/LANGGRAPH_AGENT_WORKFLOW.md` | Supervisor-routed multi-agent state machine | Add persistence, replay, and graph checkpoints |
| Streaming workflow events | Implemented | `/v1/risk-profile/stream`, legacy `/v1/diagnose/stream`, frontend stream components | Real-time agent execution UI pattern | Store trace IDs and event logs |
| Deterministic safety guardian | Implemented | `neurosight/governance/ai_safety.py` | AI safety checks before report finalization | Add prompt red-team fixture library |
| External LLM reporting | Not implemented in this release | Provider-agnostic/offline report-generation hooks only | Deterministic demo path works without external providers | Add provider config, safety evaluation rubric, and explicit opt-in integration |
| Long-term agent memory | Documented roadmap | KG and future memory notes | Awareness of memory needs | Add vector/KG retrieval evaluation |

## Data, Interoperability, And Medical-AI Awareness

| Component | Status | Evidence | What it demonstrates | Next defensible step |
|-----------|--------|----------|----------------------|----------------------|
| Synthetic ADNI-like data | Implemented | `neurosight/data/synthetic.py`, `data/ADNIMERGE_synthetic.csv` when generated | GitHub-safe demo data generation | Add private-data adapter docs for approved research |
| ADNI/OASIS positioning | Documented roadmap | `docs/DATA_DEMO_PIPELINE.md`, `docs/RUNNING_MODES.md` | Understanding of controlled medical datasets | Add real cohort validation after access approval |
| FHIR export | Implemented | `scripts/fhir_export.py`, `docs/FHIR_EXPORT.md` | Healthcare interoperability awareness | Validate exported resources with FHIR validator |
| DICOM/DICOMweb awareness | Implemented, demo-limited | `scripts/dicomweb_manifest.py`, `docs/DICOM_DICOMWEB.md` | Imaging interoperability and metadata safety | Add QIDO/WADO integration in a private test stack |
| Knowledge graph | Implemented, demo-limited | `knowledge_graph.py`, KG endpoints | Temporal patient context and similar-patient retrieval | Move to typed schema and graph DB if scaling |

## MLOps, Governance, And Deployment

| Component | Status | Evidence | What it demonstrates | Next defensible step |
|-----------|--------|----------|----------------------|----------------------|
| MLflow registry bridge | Implemented as local artifact | `scripts/mlflow_registry.py`, `docs/MLFLOW_REGISTRY.md` | Model lifecycle and promotion concepts | Add real MLflow server and artifact store |
| DVC provenance | Implemented as manifest/stage | `scripts/dvc_provenance.py`, `dvc.yaml`, `docs/DVC_PROVENANCE.md` | Data/model hash tracking | Add remote storage in private environment |
| ONNX export | Implemented for cognitive classifier | `scripts/onnx_export.py`, `docs/ONNX_RUNTIME_EXPORT.md` | Deployment/runtime portability | Add parity tests in CI with optional ONNX deps |
| Drift monitoring | Implemented | `scripts/drift_monitor.py`, `docs/DRIFT_MONITORING.md` | PSI/KS monitoring concepts | Connect to real inference logs |
| OpenTelemetry | Implemented as probe/docs | `scripts/otel_probe.py`, `docs/OPENTELEMETRY_OBSERVABILITY.md` | Trace/observability awareness | Add always-on service tracing and dashboards |
| Supply-chain audit | Implemented | `scripts/supply_chain_audit.py`, `docs/SECURITY_SUPPLY_CHAIN.md` | Security hygiene and secret scanning | Add pip-audit, Trivy, CodeQL, gitleaks in CI |
| Model card checker | Implemented | `scripts/model_card_check.py`, `MODEL_CARD.md` | Disclosure and scientific-claim discipline | Add release checklist tied to model versions |
| CI/CD quality gate | Implemented | `scripts/quality_gate.py`, `.github/workflows/quality_gate.yml` | Repository readiness automation | Add frontend and integration gates |
| Portfolio proof path | Implemented | `scripts/portfolio_check.py`, `docs/DEMO_SCRIPT.md`, `docs/PORTFOLIO_CHECKLIST.md` | Reviewer-facing verification and interview readiness | Add recorded demo and fresh UI screenshots generated from a running app |

## Frontend And Demo UX

| Component | Status | Evidence | What it demonstrates | Next defensible step |
|-----------|--------|----------|----------------------|----------------------|
| Gradio/Hugging Face demo | Implemented | `app.py`, `hf_space/` | Public demo packaging | Add stable demo scenario script |
| React dashboard | Implemented, demo-limited | `frontend/` | Larger product UI and API client design | Add frontend CI, Playwright smoke tests |
| Static local UI | Implemented | `local_ui/` | Lightweight local presentation layer | Keep aligned with backend contracts |
| Screenshots/GIF | Not included | Placeholder files were removed | Avoids fake visual evidence | Add fresh screenshots for major tabs only after generating them from a running app |

## Explicitly Out Of Scope

| Item | Reason |
|------|--------|
| Clinical diagnosis | Requires real cohorts, clinical validation, regulatory process, and expert oversight |
| Treatment or prescribing | Unsafe and outside the purpose of the project |
| Medical-device claims | Not supported by synthetic data or current validation |
| Public hosting of private patient data | Privacy and data-use agreements prohibit this |
| Fully autonomous clinical agents | Not appropriate for this domain or project scope |

## Recommended Interview Framing

If asked what the project proves, say:

> NeuroSight is not a clinical model. It is a research and portfolio prototype
> showing that I can design a multimodal AI system with medical-AI constraints,
> backend APIs, agent orchestration, safety checks, MLOps/LLMOps artifacts, and
> honest scientific documentation.

If asked what is missing, say:

> The main missing piece is real-cohort validation. The architecture is shaped
> for MRI, EEG, and cognitive data, but public GitHub metrics are synthetic and
> cognitive-centered. I intentionally disclose that instead of overclaiming.
