# NeuroSight

**Full-stack multimodal medical AI research scaffold for MRI, EEG, and cognitive-score fusion.**

> **Not for clinical use:** NeuroSight is a portfolio and research engineering project. It is not validated for clinical use, and it is not a medical device, triage tool, or diagnostic system. Public outputs are software-demo outputs and require expert review.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3-red.svg)](https://monai.io/)

NeuroSight demonstrates how a senior AI engineer might structure a medically sensitive AI system without overstating its maturity. The repository includes model modules, API contracts, safety disclosures, synthetic evaluation artifacts, local UI surfaces, and reviewer-oriented checks. It does not include real ADNI, OASIS, hospital, or patient data.

## 🎯 Intended Use / Not Intended Use

### Intended Use
- **Research & Engineering Scaffold:** Demonstrating systems architecture, data pipeline engineering, and governance design.
- **Multimodal Engineering Demo:** Showcasing how to handle disparate modalities (tabular, time-series, and 3D volumes) in a unified model stack.
- **Developer & Educational Sandbox:** Sandbox for testing safety agents, calibration checks, and model card automation in ML pipelines.

### Not Intended Use
- **No Clinical Decisions:** Not for clinical use, patient triage, clinical decision-making, diagnosis, or treatment planning.
- **No Diagnostic Claims:** Must not be used as a medical device or diagnostic system.
- **No Clinical Data:** Not intended for use with actual patient datasets without real-world validation and institutional clearance.

## 💼 Portfolio Positioning

This repository is designed to demonstrate senior AI systems engineering and ML architecture:
- **Multimodal Model Architecture:** Designing custom PyTorch fusion modules with missing modality imputation and modality dropout.
- **API Engineering:** Implementing Pydantic validation contracts, route-level asynchronous operations, and Server-Sent Events (SSE).
- **Safety and Governance:** Building LangGraph-based supervisor and checker agents with dynamic, deterministic guardrails.
- **Reproducible Synthetic Evaluation:** Standardizing baseline comparisons and automated benchmark table updates.
- **MLOps Readiness:** Integrating ONNX export pathways, registry patterns, and automated model card validation checks.

## 🚀 Live Demo

You can interact with a live mockup of the NeuroSight interface to see the multimodal evaluation helper, the EEG/MRI viewer, and the LangGraph-based safety guardian in action:
- **FastAPI backend**: API documentation is interactive via Swagger UI at `/docs`.
- **Gradio frontend**: Start `python3 app.py` to open the interactive Gradio risk profiling panel.

## 📌 Highlights

- **Leakage-Free Baselines**: Robust baseline benchmark ensuring that noise embeddings do not inflate fusion model performance.
- **Multimodal Fusion**: Custom PyTorch fusion network supporting missing modalities (MRI/EEG) via zero-masked imputation.
- **Explainable AI (XAI)**: Grad-CAM++ visualization for MRI heatmaps and attention weight maps for cognitive scores.
- **LangGraph Agentic Guardians**: Supervisor agent with integrated safety checks preventing risk profile generation for invalid/risky input profiles.

## Current Status

| Area | Status |
|------|--------|
| Project version | 0.3.0 research scaffold |
| Public data | Synthetic ADNI-like tabular demo data only |
| Private data adapters | Optional local ADNI-style cognitive adapter for authorized users; no raw data is included |
| Public model weights | No clinical or validated checkpoint is shipped |
| MRI/EEG | Engineering pathways for ingestion and encoder architecture; no validated clinical interpretation |
| Cognitive branch | Implemented and used by the synthetic demo/evaluation path |
| Evaluation | Synthetic/demo metrics only, not real-world medical evidence |
| Deployment | Local FastAPI, Gradio, static local UI, frontend code, and a full-stack Hugging Face Docker Space path are present; no uptime claim is made |
| External LLM providers | Not implemented; only provider-agnostic/offline report-generation hooks are present |

## Implemented

- PyTorch modules for MRI, EEG, cognitive encoding, fusion, and XAI interpretations.
- MRI encoder uses MONAI ViT when available and a compact CPU-safe 3D CNN fallback
  for demo/CI environments; this public repo provides no clinical validation for either path.
- FastAPI backend with risk profiling, upload, streaming, evaluation, model-registry, KG, XAI, and governance routes.
- Local Gradio and UI code for demonstrating the request/response flow.
- Synthetic ADNI-like data generation and demo contracts.
- Local-only private-data adapter interface for authorized ADNI-style cognitive CSVs.
- Synthetic evaluation figures in `docs/figures/`.
- LangGraph workflow with deterministic safety-guardian behavior.
- Provider-agnostic/offline report-generation hooks exist, but no external
  Gemini/OpenAI/Anthropic provider is configured in this release. LLMs must not
  be used as diagnostic models.
- Governance, supply-chain, model-card, API-contract, drift, ONNX, and portfolio check scripts.
- Tests covering data contracts, backend behavior, model components, XAI boundaries, and demo readiness.

## Planned

- External validation on approved ADNI/OASIS-style or institutional cohorts.
- Trained and documented multimodal checkpoints with full provenance.
- Frontend integration tests and fresh real screenshots generated from a running local app.
- Production-grade auth, audit logging, deployment hardening, and monitoring.

See [docs/IMPLEMENTED_VS_PLANNED.md](docs/IMPLEMENTED_VS_PLANNED.md) for the detailed matrix.

## Synthetic Benchmark, Not Medical Evidence

> ⚠️ **`synthetic_data: true` | `clinical_validity: false`**
>
> All results below are from synthetically generated ADNI-like tabular data.
> They show that the training, evaluation, and reporting pipelines run
> end-to-end. They do **not** estimate real-world clinical accuracy,
> robustness, or utility on any patient population.

**Leakage fix (v0.3.0):** Prior to v0.3.0, synthetic MRI/EEG embeddings
were derived as linear projections of cognitive features, artificially
inflating fusion model AUC. From v0.3.0, MRI and EEG embeddings are
**independent Gaussian noise** (zero cognitive-feature correlation), making
the benchmark leakage-free and fusion performance honestly uninformative.

Because public smoke benchmarks intentionally use uninformative MRI/EEG noise,
fusion collapse or fusion underperformance is expected. It should be read as a
sanity check that noise modalities are not boosting metrics, not as evidence of
clinical model failure or success. Signal-bearing MRI/EEG evaluation requires
authorized data, locked preprocessing, and subject-disjoint validation; this
repository does not claim that real modalities will automatically improve
performance.

### Baseline Comparison — Cognitive Features Only (Leakage-Checked)

| Method | Modality | AUC ↑ | F1 ↑ | Acc ↑ | Bal.Acc ↑ | Brier ↓ | ECE ↓ | Train Time |
|--------|----------|--------|------|-------|-----------|---------|-------|------------|
| Random Classifier (chance) | — | 0.600 | 0.222 | 0.333 | 0.333 | 0.134 | 0.321 | 0.0s |
| Majority Classifier | — | 0.500 | 0.048 | 0.167 | 0.167 | 0.139 | 0.000 | 0.0s |
| Logistic Regression | Cognitive | 0.800 | 0.361 | 0.500 | 0.500 | 0.102 | 0.404 | 0.0s |
| Random Forest | Cognitive | 0.783 | 0.417 | 0.500 | 0.500 | 0.094 | 0.333 | 0.0s |
| Gradient Boosting | Cognitive | 0.833 | 0.556 | 0.667 | 0.667 | 0.090 | 0.268 | 0.1s |
| MLP (Cognitive only) | Cognitive | 0.567 | 0.194 | 0.333 | 0.333 | 0.136 | 0.127 | 0.1s |
| NeuroSight (Cognitive, calibrated) | Cognitive | 0.600 | 0.167 | 0.167 | 0.167 | 0.137 | 0.179 | 0.0s |
| **NeuroSight Fusion** | **Noise-MRI+Noise-EEG+Cog** | 0.433 | 0.048 | 0.167 | 0.167 | 0.251 | 0.750 | 0.8s |

*Run `scripts/run_benchmark.py --mode full` to populate this table with current numbers.*
*Fusion on noise embeddings is NOT expected to exceed cognitive-only; this is the correct honest result.*

### Archived Inflated Metrics (v0.2.0)

These pre-leakage-audit reference metrics are archived for historical transparency. They were obtained prior to the correlation leakage fix, where synthetic MRI/EEG embeddings were artificially derived from cognitive features:
- Accuracy: 71.4% (Stale / Leakage-inflated)
- Macro AUC: 0.945 (Stale / Leakage-inflated)
- Macro F1: 0.694 (Stale / Leakage-inflated)
- ECE: 0.255 (Stale / Leakage-inflated)

No real-world medical performance is measured in this repository. Any real
claim would require authorized data access, locked preprocessing, trained
checkpoints, external validation, calibration review, subgroup analysis,
expert review, and appropriate regulatory controls.

## Authorized Private-Data Adapter

Authorized users who already have access to external datasets can run a local
adapter without adding private data to the repository:

```bash
python3 scripts/prepare_adni_cognitive.py \
  --input-csv data/private/adni/ADNIMERGE.csv \
  --output-dir data/processed/adni_cognitive
```

The adapter validates the cognitive schema, normalizes common ADNI-style
column names, blocks known leakage fields, creates subject-disjoint
train/validation/test splits, and writes a dataset summary JSON. Generated
outputs from real cohorts may still be sensitive and must not be committed.
See [docs/PRIVATE_DATA_ADAPTERS.md](docs/PRIVATE_DATA_ADAPTERS.md).

## Visual Artifacts

This repository keeps only real generated evaluation figures:

- [confusion_matrix.png](docs/figures/confusion_matrix.png)
- [roc_curves.png](docs/figures/roc_curves.png)
- [training_curves.png](docs/figures/training_curves.png)
- [ablation.png](docs/figures/ablation.png)
- [attention_weights.png](docs/figures/attention_weights.png)
- [calibration_reliability.png](docs/figures/calibration_reliability.png)
- [federated_convergence.png](docs/figures/federated_convergence.png)

Placeholder screenshots and the previous 1x1 demo GIF were removed. To inspect the UI honestly, run it locally using the commands below.

## 🏗️ Architecture

NeuroSight is structured as a decoupled, multi-tier multimodal AI system:
- **Tabular & Signal Encoders**: Custom PyTorch networks for tabular cognitive, EEG time-series, and 3D MRI volume encoding.
- **Attention Fusion Classifier**: Aggregates latent embeddings using scaled dot-product cross-modality attention.
- **FastAPI Core Service**: Validates schemas using strict Pydantic contracts and serves predictions asynchronously.
- **Agentic Logic Layer**: An orchestrator supervisor implemented in LangGraph coordinating domain analyst subagents.

## 🚀 Quick Start

To facilitate quick reviewer verification, the repository includes a master target that runs repository hygiene, strict model-card and quality gates, fast Python tests, safety red-team tests, a synthetic benchmark smoke pipeline, and frontend checks when frontend dependencies are installed.

To run the complete verification check, execute:

```bash
make install
make verify
```

Individual developer and verification targets include:

* **Clean Installation**: `make install` installs Python 3.11 dependencies using `requirements.lock`.
* **Test Suite**: `make test` runs the default non-slow Python suite with timeout protection.
* **Fast Test Suite**: `make test-fast` runs non-slow, non-benchmark, non-safety Python tests on CPU.
* **Quality Gate Check**: `make quality` performs ruff check, mypy checks, repository hygiene, model card correctness, and strict quality gate compliance.
* **Safety Red-Team Tests**: `make safety` runs the safety and red-team regression suite.
* **Benchmark Smoke Test**: `make benchmark-smoke` runs the fast benchmark smoke pipeline CLI.
* **Frontend Check**: `make frontend-check` runs type-checking when `frontend/node_modules` exists, and runs the Next.js build only in CI.

Hugging Face full-stack Space deployment is documented in
[docs/HUGGING_FACE_DEPLOYMENT.md](docs/HUGGING_FACE_DEPLOYMENT.md).

## Reviewer Verification

Use the one-command reviewer path:

```bash
make verify
```

`make verify` requires a working Python 3.11 interpreter. If `python3` does not
resolve to Python 3.11 on your machine, run it as `make verify
PYTHON=/path/to/python3.11`.

Equivalent individual checks:

```bash
python3 scripts/check_repo_hygiene.py
python3 scripts/model_card_check.py --strict
python3 scripts/quality_gate.py --strict
APP_ENV=test python3 -m pytest tests/ -v -m "not slow and not benchmark and not safety" --tb=short
APP_ENV=test python3 -m pytest tests/test_safety_redteam.py -v -m "safety" --tb=short
APP_ENV=test python3 scripts/run_benchmark.py --mode smoke
npm --prefix frontend run type-check
```

The frontend command requires `frontend/node_modules`; install it with `npm --prefix frontend ci` in environments where Node dependencies are available. `make frontend-check` skips cleanly when those dependencies are absent, preserving a clean GitHub archive.

## How To Reproduce

### Installation and Reproduction

```bash
python3.11 -m venv /tmp/neurosight-venv
source /tmp/neurosight-venv/bin/activate
python scripts/check_python_version.py
make install
```

Run the hygiene check (use `--allow-dev-caches` to ignore local caches and frontend dependencies in a working environment):

```bash
python3 scripts/check_repo_hygiene.py --allow-dev-caches
```

For CI-equivalent setup and test details, see
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

Run focused project checks:

```bash
make test-fast
make test-benchmark
python3 scripts/model_card_check.py --strict
python3 scripts/api_contract_check.py
python3 scripts/ai_safety_eval.py --strict
```

Start the local FastAPI backend:

```bash
APP_ENV=test NEUROSIGHT_API_KEY=change-me-for-local-dev uvicorn api.main:app --reload --port 8000
```

Start the Gradio demo:

```bash
python3 app.py
```

Run synthetic training and figure generation:

```bash
python3 scripts/train.py training.warmup_epochs=3 training.finetune_epochs=5 data.num_workers=0
python3 scripts/evaluate.py checkpoints/best_fusion.pt
python3 scripts/generate_phase2_figures.py
```

Run the synthetic benchmark (generates JSON + Markdown reports with provenance):

```bash
# Smoke mode (fast pipeline check)
APP_ENV=test python3 scripts/run_benchmark.py --mode smoke --output outputs/

# Full mode (30 samples/class)
APP_ENV=test python3 scripts/run_benchmark.py --mode full --output outputs/

# Inspect the reports
cat outputs/benchmark_report.json
open outputs/benchmark_report.md
```

Inspect or validate real-data schema (no protected data required or accessed):

```bash
python3 scripts/prepare_adni_like_dataset.py --schema
python3 scripts/prepare_adni_like_dataset.py --generate-schema docs/adni_schema.json
```

The training commands create local checkpoints and logs. Those artifacts are intentionally ignored by Git unless they are deliberately reviewed, documented, and released.

## Developer Commands

```bash
make install        # Python 3.11 check + pip install with constraints
make hygiene        # repository artifact and claim hygiene
make lint           # ruff + mypy
make test-fast      # CPU-safe tests, excludes slow marker
make test-benchmark # tiny synthetic benchmark smoke
make test           # full test suite
make smoke-api      # backend smoke path without upload-heavy checks
```

Evaluation-specific commands:

```bash
# Benchmark: smoke (fast) or full
APP_ENV=test python3 scripts/run_benchmark.py --mode smoke

# Real-data schema documentation
python3 scripts/prepare_adni_like_dataset.py --schema

# Metrics correctness unit tests
APP_ENV=test python3 -m pytest tests/test_metrics_correctness.py -v

# Leakage regression tests
APP_ENV=test python3 -m pytest tests/test_leakage_detection.py -v
```

The fast suite is intended to complete in a few minutes on CPU. Slow/release
checks are available with `make test-slow`.

## 🔌 API Reference

All `/v1/*` routes require `X-API-Key` outside test mode. `/healthz` is public for local service health.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/healthz` | Health, runtime mode, model status, upload limits |
| `POST` | `/v1/risk-profile` | Preferred non-clinical demo risk-profile response from supplied inputs |
| `POST` | `/v1/risk-profile/stream` | Preferred SSE workflow events for the risk profiling path |
| `POST` | `/v1/diagnose` | Legacy/deprecated alias retained for backward compatibility |
| `POST` | `/v1/upload/mri` | MRI upload parsing and embedding path |
| `POST` | `/v1/upload/eeg` | EEG upload parsing and embedding path |
| `POST` | `/v1/upload/cognitive` | Cognitive-score validation and embedding path |
| `GET` | `/v1/eval/metrics` | Synthetic evaluation metrics |
| `GET` | `/v1/xai/status` | XAI method availability and interpretation policy |
| `GET` | `/v1/governance/status` | Safety, privacy, and disclosure status |

Canonical cognitive payloads use exactly eight required fields:
`MMSE`, `MOCA`, `CDRSB`, `ADAS11`, `RAVLT_immediate`,
`RAVLT_learning`, `FAQ`, and `AGE`. Obsolete UI fields are rejected rather
than silently remapped.

See [docs/API_EXAMPLES.md](docs/API_EXAMPLES.md) for request examples.

## What This Project Demonstrates

- Honest medical-AI positioning: explicit separation between scaffold, demo metrics, and clinical validation.
- Multimodal model design: modality encoders, missing-modality handling, fusion, calibration, and XAI interpretations.
- Backend engineering: typed contracts, protected routes, uploads, streaming, health checks, and route-level tests.
- MLOps thinking: model registry, DVC/provenance design, ONNX export path, drift monitoring, and quality gates.
- Agentic workflow design: LangGraph orchestration with deterministic safety boundaries.
- Repository maturity: model card, security notes, CI, release-readiness docs, and hygiene checks.

## Current Limitations

## 📚 Datasets

NeuroSight is evaluated on a synthetic ADNI-like dataset that mirrors the demographic, cognitive, and imaging data distribution of the Alzheimer's Disease Neuroimaging Initiative:
- **Cognitive Profile**: 8 features (MMSE, MoCA, CDR-SB, ADAS-11, FAQ, RAVLT Immediate, RAVLT Learning, Age).
- **Tabular Data**: Generated using `scripts/prepare_adni_like_dataset.py`.
- **Imaging Modalities**: Synthetic 3D MRI tensors and EEG signal profiles representing normal controls, mild cognitive impairment (MCI), and Alzheimer's disease (AD).

## ⚠️ Detailed Limitations

For detailed system and clinical limitations, refer to [MODEL_CARD.md](MODEL_CARD.md) and [docs/IMPLEMENTED_VS_PLANNED.md](docs/IMPLEMENTED_VS_PLANNED.md). Key points include:
- **Synthetic Evaluation**: Not tested on real clinical patients.
- **Safety Guardians**: LangGraph agents rely on strict threshold heuristics rather than deep clinical reasoning.
- **Calibration**: Brier scores indicate the model's confidence is not calibrated for real-world interpretation.

- No real patient data is bundled.
- No public checkpoint in this repository is validated for medical use.
- Public metrics are synthetic/demo metrics and may not transfer to real cohorts.
- MRI and EEG code paths are real engineering surfaces, but the public model has no validated MRI/EEG risk-profiling capability.
- FTD, LBD, and VD labels are synthetic six-class demo labels, not validated disease classifiers.
- Confidence scores are model scores, not clinical certainty; the disclosed ECE shows calibration remains weak even in the synthetic setting.
- Explainability outputs are model-behavior diagnostics, not biomarkers or disease-causality evidence.
- The local demo security model is not a production healthcare security architecture.

## Repository Hygiene

Run:

```bash
python3 scripts/check_repo_hygiene.py
```

The check fails on local secrets, virtual environments, caches, logs, local deployment folders, build artifacts, tiny placeholder images, unsupported paper references, and unsupported clinical claims. Details are in [docs/REPOSITORY_HYGIENE.md](docs/REPOSITORY_HYGIENE.md).

## 📁 Project Structure

```text
api/                    FastAPI backend
neurosight/             Core models, data contracts, governance, tracking, interop
evaluation/             Synthetic metrics and benchmark helpers
federated/              Federated-learning simulation
frontend/               React/Next UI source
local_ui/               Static local UI
scripts/                Training, evaluation, governance, hygiene, and release checks
tests/                  Unit, integration, and smoke tests
docs/                   Architecture, governance, runbooks, and generated figures
MODEL_CARD.md           Medical-use boundaries and evaluation disclosure
```

## Citation

```bibtex
@software{neurosight2026,
  title  = {NeuroSight: Full-Stack Multimodal Medical AI Research Scaffold},
  author = {NeuroSight Contributors},
  year   = {2026},
  note   = {Portfolio and research engineering scaffold using synthetic public data; not for clinical use}
}
```

## 📄 License

[MIT License](LICENSE). The license does not grant clinical, regulatory, or medical-device approval.
