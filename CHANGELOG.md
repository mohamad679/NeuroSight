# Changelog

## [0.3.0] — 2026-05-23

### Added
- ADNI-style private cognitive adapter for authorized local users, with schema
  normalization, subject-disjoint train/validation/test splits, leakage checks,
  and dataset summary output.
- Locked pip dependency strategy (`requirements.lock`) and Makefile verification
  path for reviewer reproducibility.
- Gradient clipping in PyTorch training loops using
  `torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)`, including safe gradient
  norm logging in the main training script.
- Timeout protection for pytest-based smoke and safety checks.

### Changed
- Hardened MRI fallback behavior for CPU/demo environments and clarified that
  public MRI/EEG paths are engineering scaffolds, not validated clinical
  interpretation.
- Improved benchmark and baseline wording after the leakage audit: public
  synthetic MRI/EEG inputs are intentionally uninformative, so fusion collapse
  or underperformance is expected and should not be interpreted as clinical
  failure or success.
- Updated documentation and safety wording to consistently state that public
  results are synthetic-only unless an authorized user runs private validation.
- Hardened Docker to run as a non-root user and avoid baking secrets into the
  image.

### Fixed
- Repository hygiene for release review by removing accidental local test
  artifacts and excluding Python caches, pytest caches, Hydra outputs,
  benchmark outputs, frontend build caches, and local build artifacts.
- Frontend release tree hygiene by excluding `.next/`, `node_modules/`, `out/`,
  build directories, and common frontend cache files.

### Not Implemented
- External Gemini/OpenAI/Anthropic report generation is not implemented in this
  release. The LangGraph demo remains deterministic/offline unless explicitly
  extended in a future release.

## [0.2.0] — 2026-04-03

### Added
- Gradio web interface with three tabs (Diagnosis, Ablation, KG Explorer)
- HuggingFace Spaces deployment configuration
- Benchmark framework with 5 baseline comparisons
- Pre-run Jupyter notebook with embedded outputs
- Technical report (`docs/TECHNICAL_REPORT.md`)

## [0.1.0] — 2026-04-02

### Added
- CrossModalAttentionFusion with missing-modality tokens
- Temperature calibration on all three unimodal classifiers
- GradCAM++, AttentionRollout, SHAP XAI per modality
- LangGraph multi-agent pipeline with deterministic safety guardian
- Temporal Knowledge Graph with bi-temporal queries
- Federated Learning simulation (FedAvg, 3 hospital clients)
- ADNI-compatible data pipeline with stratified splits
- Training script with Hydra, early stopping, MLflow tracking
- CI/CD with GitHub Actions (ruff + mypy + pytest)
- DICOM/NIfTI MRI upload support
- Model registry with staging/production/archived lifecycle
