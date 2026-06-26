# Model Card: NeuroSight

NeuroSight is a full-stack multimodal medical AI research scaffold for MRI, EEG, and cognitive-score fusion. It is a portfolio and research engineering project, not software validated for clinical use.

> **⚠ Not for clinical use. Not a medical device. Outputs require specialist review. No real patient data is included.**

---

## Status Snapshot

| Field | Current status |
|-------|----------------|
| Version | 0.3.0 research scaffold |
| Intended setting | Research, education, portfolio review, and local software demonstrations |
| Public data | Synthetic ADNI-like tabular demo data only |
| Real patient data | Not included |
| Public checkpoint | No validated clinical checkpoint is shipped |
| Current public training | Cognitive/demo path on synthetic data |
| MRI/EEG status | Ingestion and model architecture paths exist; no validated clinical interpretation |
| Evaluation | Synthetic held-out split only |
| Clinical use | Out of scope |

## Model Status

This model is a **research scaffold** at version 0.3.0. It is:

- **Not clinically validated.** No clinical validation study has been conducted.
- **Not a medical device.** It has not been reviewed or approved by any regulatory body (FDA, CE, etc.).
- **Not approved for diagnosis, treatment, triage, or emergency use.**
- **Not externally validated.** External validation on independent real-world patient cohorts has not been performed.
- **Not production-ready.** The repository is a portfolio and engineering demonstration.

The active model pipeline runs a cognitive-only modality path on synthetic ADNI-like tabular data. MRI and EEG ingestion architectures exist in the codebase but have no validated clinical checkpoints.

## Model Details

- **Model Identifier:** NeuroSight Multimodal Fusion Model (`CrossModalAttentionFusion`)
- **Developer:** NeuroSight Contributors (AI engineering portfolio project)
- **Model Type:** Multimodal attention-based fusion classifier
- **Task:** 6-class cognitive classification (Normal, MCI, AD, FTD, LBD, VD) using tabular demographic/cognitive scores as the primary active modality, with architectures defined for MRI and EEG embeddings.
- **License:** MIT License

## Architecture And Pipeline

- **Cognitive Encoder:** Multi-layer Perceptron (MLP) mapping 8 canonical features (MMSE, MoCA, CDRSB, ADAS11, RAVLT_immediate, RAVLT_learning, FAQ, AGE) to a shared embedding space.
- **MRI Encoder:** MONAI ViT path for research environments when MONAI is installed, plus a compact PyTorch 3D CNN fallback for CPU demo/CI environments. Neither path is clinically validated in this public repository.
- **EEG Encoder:** Conv1D and transformer-based sequential architecture to process raw EEG time-series signals.
- **Fusion Layer:** Cross-modal attention layer that fuses the active modality representations, using masking to handle missing inputs.
- **Explainability:** Post-hoc attention weight heatmaps showing the contribution of each modality per sample.

## Input And Output Contract

- **Inputs:**
  - Tabular Cognitive/Demographic Vector: Shape `(batch_size, 8)` — fields: MMSE, MoCA, CDRSB, ADAS11, RAVLT_immediate, RAVLT_learning, FAQ, AGE
  - MRI Volume (Optional): Shape `(batch_size, 1, 96, 96, 96)`
  - EEG Signal (Optional): Shape `(batch_size, channels, sequence_length)`
- **Outputs:**
  - Diagnosis Probabilities: Shape `(batch_size, 6)` mapping to class likelihoods (Normal, MCI, AD, FTD, LBD, VD).
  - Attention Weights: Shape `(batch_size, 3)` indicating attention weights for MRI, EEG, and Cognitive features.
  - Every API response includes a hardcoded clinical disclaimer. Outputs are not medical findings.

## Intended Use

Appropriate uses are limited to:
- Reviewing multimodal AI system architecture,
- Experimenting with synthetic/demo data contracts,
- Studying FastAPI, UI, LangGraph, and MLOps integration patterns,
- Demonstrating responsible medical-AI documentation and repository hygiene,
- Local software demos where every output is treated as non-clinical.

Outputs may be useful for engineering discussion, but they are not medical findings.
Provider-agnostic/offline report-generation hooks exist, but no external
Gemini/OpenAI/Anthropic provider is configured in this release. LLMs must not
be used as diagnostic models.

## Out-Of-Scope Use

Do not use NeuroSight for:
- Clinical diagnosis,
- Treatment decisions,
- Medication or prescribing decisions,
- Triage or emergency use,
- Patient management,
- Autonomous clinical report generation,
- Medical-device or regulatory submissions,
- Claims about disease detection on real cohorts,
- Processing identifiable patient records in public deployments.

Every output requires expert review and must be treated as a software-demo artifact.

## Model Status

See above for current status. In summary: not clinically validated, not a medical device, not approved for any clinical purpose.

## Training Data Status

- **Tabular Dataset:** ADNIMERGE-like synthetic data generated via class-conditional Gaussian models. **No real ADNI participants are included.**
- **Imaging Dataset:** Dummy T1-weighted MRI volumes generated from random noise distributions. No real patient MRI data.
- **MRI fallback status:** The non-MONAI fallback is an efficient 3D CNN engineering path for demos and tests; it is not evidence of clinical MRI interpretation capability.
- **EEG Dataset:** Dummy signal time series generated from sine/cosine functions with random noise. No real patient EEG data.
- **Real Patient Cohorts:** No real patient cohorts (ADNI, OASIS, or any other) were used to train or optimize the model.
- **Data Provenance:** The synthetic dataset is fully reproducible with fixed seed. See `data/ADNIMERGE_synthetic.csv`.

## Evaluation Data Status

- **Evaluation Dataset:** Held-out synthetic split of the same ADNI-like synthetic tabular dataset.
- **No real clinical evaluation** data is included in this repository.
- **ADNI/OASIS real-data results are not claimed.** Any comparison to ADNI benchmarks would require a separate authorized real-data evaluation, which has not been performed.
- **External Validation:** External validation on independent real-world patient cohorts has not been performed.
- **Evaluation scope:** Synthetic benchmark only. Results reflect pipeline correctness, not clinical performance.

## Data And Training Scope

The training scope is strictly limited to synthetic/demo data:
- **Tabular Dataset:** ADNIMERGE-like synthetic data generated via class-conditional Gaussian models (no real ADNI participants).
- **Imaging Dataset:** Dummy T1-weighted MRI volumes generated from random noise distributions.
- **EEG Dataset:** Dummy signal time series generated from sine/cosine functions with random noise.
- **Real Patient Cohorts:** No real patient cohorts (ADNI, OASIS, etc.) were used to train or optimize the model.

## Synthetic Benchmark Disclosure

> **SYNTHETIC BENCHMARK — NOT CLINICAL PERFORMANCE**
>
> All evaluation metrics in this model card and in `evaluation/results.json` come from a held-out synthetic benchmark only.
>
> - These results are not medical evidence.
> - They do not represent real-world diagnostic accuracy.
> - They do not validate clinical performance.
> - Synthetic benchmark results cannot substitute for prospective clinical validation.
> - No ADNI/OASIS real-data results have been computed or are claimed in this repository.
> - The project is not clinically validated.

## Clinical Validation Status

- **Status:** Not clinically validated.
- **Regulatory approval:** None. This project is not a medical device and has not been submitted for regulatory review.
- **Prospective clinical study:** Not conducted.
- **External validation:** Not performed.
- **Real patient data evaluation:** Not performed.
- **Clinical performance:** This is not clinical performance.

## Evaluation Results

The metrics below come from the test split evaluation of the model checkpoint, as saved in `evaluation/results.json`. **These are synthetic benchmark results only and are not clinical performance.**

- Accuracy: 66.7%
- Macro F1: 0.556
- Macro AUC: 0.833
- ECE: 0.268

These results show that the training and evaluation pipeline can run. They do not estimate clinical accuracy, robustness, fairness, or real-world utility.

## Safety Limitations

- The model has no approved clinical use and must not be used in patient-facing applications.
- Safety checks in the LangGraph agent workflow reject out-of-bounds inputs, but this does not constitute clinical-grade safety validation.
- The system has not been tested for adversarial robustness, edge-case failure modes, or demographic subgroup failures on real cohorts.
- Supply chain vulnerability auditing is run as part of CI/CD, but cannot substitute for a full security audit.
- No healthcare-grade authentication, audit logging, or data access controls are present in the demo deployment.
- The model is not approved for emergency or life-critical use under any circumstances.

## Bias And Fairness Limitations

- The synthetic data is balanced across classes but does not represent real-world clinical covariate distributions or demographic variations.
- No subgroup fairness analysis on real human populations has been conducted.
- Any deployment on real patients without extensive bias evaluation would be inappropriate and unethical.
- Demographic covariates (age, sex, ethnicity) present in real ADNI cohorts are not present in the synthetic training data.
- Bias assessments are limited to synthetic checks only.

## Data Privacy Limitations

- No real patient data is included in this repository.
- The synthetic ADNI-like dataset contains no identifiable information.
- The repository must not be used to process, store, or transmit real patient health data without appropriate data governance, HIPAA/GDPR controls, and IRB oversight.
- The public API demo does not implement healthcare-grade de-identification, access controls, or audit logging.

## Human Oversight Requirement

All model outputs require mandatory specialist review before any action is taken:

- No output from NeuroSight may be acted upon without expert clinical review.
- The system is designed for research exploration, not autonomous clinical decision-making.
- Hardcoded disclaimers are returned with every API response to enforce this requirement.
- Any deployment context that bypasses human oversight would violate the intended use of this system.

## Known Failure Modes

- **Synthetic-to-real distribution shift:** The model is trained on synthetic data and will fail when applied to real patient cohorts.
- **Fusion/class collapse:** Public smoke benchmarks intentionally use uninformative MRI/EEG noise or placeholders, so fusion underperformance or single-class predictions can occur and do not establish clinical failure or success.
- **Missing modality fallback:** MRI and EEG are architecturally supported but not clinically validated; missing-modality predictions degrade gracefully but are not clinically reliable.
- **Out-of-range inputs:** Inputs outside the expected clinical ranges may produce unpredictable outputs.
- **Calibration failure:** ECE of 0.268 indicates significant miscalibration on the synthetic evaluation set.
- **No edge-case coverage:** Rare or ambiguous clinical presentations are not represented in the synthetic data.

## Known Limitations

- No public clinical validation.
- No authorized real-cohort evaluation in this repository.
- No validated MRI or EEG diagnostic checkpoint.
- No claim of calibration adequacy.
- No subgroup fairness analysis on real populations.
- No production healthcare authentication, audit, or monitoring guarantees.
- No regulatory approval.
- External validation is not yet performed.

## Responsible Use

Users and contributors are expected to:
- Never deploy this system in patient-facing clinical contexts without regulatory approval and clinical validation.
- Clearly communicate to any audience that outputs are from a synthetic research scaffold.
- Never present model outputs as clinical findings or medical evidence.
- Follow all applicable laws, ethical guidelines, and institutional policies for AI in healthcare.
- Maintain this model card and all disclosures when extending or adapting the system.

## Reviewer Interpretation

For GitHub and portfolio reviewers:
- This repository demonstrates engineering architecture, documentation quality, safety governance, and MLOps integration patterns.
- Evaluation metrics are intentionally low because the model is trained on synthetic data with no real-world signal.
- The low accuracy (66.7%) is expected and is not a defect — it reflects training on synthetic class-conditional Gaussian data without meaningful clinical patterns.
- The goal is demonstrating responsible AI engineering practices, not clinical accuracy.
- All safety claims, disclosures, and governance artifacts should be evaluated on their engineering merit and completeness, not on clinical performance.

## Portfolio Positioning

This project is a non-clinical research demo scaffold designed to demonstrate senior AI systems engineering and ML architecture capability. It highlights:
- **Multimodal Model Architecture:** Engineering custom PyTorch models that fuse tabular, 1D sequential (EEG), and 3D volume (MRI) modalities with zero-masked imputation fallback logic.
- **API Engineering:** Building production-shaped, async FastAPI endpoints with strict Pydantic data schemas and Server-Sent Event (SSE) streaming updates.
- **Safety and Governance:** Constructing agentic supervisor workflows using LangGraph with deterministic checks, prompt injection detection, and dynamic risk profile restrictions.
- **Reproducible Synthetic Evaluation:** Standardizing rigorous evaluation protocols with majority, random, and linear baselines, calibrated metrics, and automated reporting.
- **MLOps Readiness:** Implementing containerized deployments, DVC-based tracking schemas, ONNX runtime conversion paths, and automated model card validation checks.

## Reproducibility

To run the pipeline and reproduce these metrics locally, execute:

```bash
# 1. Prepare synthetic dataset
python3 scripts/prepare_adni_like_dataset.py

# 2. Run model training
python3 scripts/train.py training.warmup_epochs=2 training.finetune_epochs=5

# 3. Generate figures and evaluation results.json
python3 scripts/generate_phase2_figures.py
```

See also `evaluation/results.json` for seed, dependency versions, and command metadata.

## Explainability And Reporting

- Attentions are extracted dynamically during the forward pass of `CrossModalAttentionFusion`.
- XAI Attributions: Modality weights are visualized per patient to help developers verify if the model is ignoring/utilizing specific modalities as expected.
- Reporting: The API generates a structured evaluation report including data distribution details, training hyperparameters, and model output calibration.

## Safety And Governance

- Deterministic safety checks are implemented via a LangGraph agent workflow to reject inputs with out-of-bounds parameters.
- Hardcoded clinical disclaimers are returned with every API response to prevent diagnostic misuse.
- Supply chain vulnerability auditing is run as part of CI/CD.

## Bias, Fairness, And Representation

- The synthetic data is balanced across classes but does not represent real-world clinical covariate distributions or clinical demographic variations.
- Bias assessments are limited to demographic checks on synthetic data; no subgroup fairness analysis on real human populations has been conducted.

## Limitations

This project has the following hard limitations:

1. **Not clinically validated** — no clinical study has been conducted.
2. **Not a medical device** — not registered with any regulatory authority.
3. **Synthetic data only** — no real patient data is used or included.
4. **No external validation** — results apply only to the synthetic benchmark.
5. **No regulatory approval** — must not be used in regulated clinical settings.
6. **Miscalibrated** — ECE of 0.268 indicates predictions are not reliable confidence estimates.
7. **No fairness analysis** — demographic subgroup performance on real cohorts is unknown.
8. **No production security** — not suitable for deployment without full security review.

## Deployment And Monitoring

- **FastAPI Endpoint:** Deployed locally or as a containerized microservice.
- **Drift Monitoring:** Logged prediction distributions are monitored for shift against the synthetic training dataset.
- **Observability:** OpenTelemetry traces and metrics capture handler execution times and safety gate rejects.

## Evidence And Artifacts

The following project artifacts and documentation files are maintained and verified:
- [evaluation/results.json](evaluation/results.json)
- [docs/MONAI_PIPELINE.md](docs/MONAI_PIPELINE.md)
- [docs/MLFLOW_REGISTRY.md](docs/MLFLOW_REGISTRY.md)
- [docs/DVC_PROVENANCE.md](docs/DVC_PROVENANCE.md)
- [docs/AI_SAFETY_OWASP_GENAI.md](docs/AI_SAFETY_OWASP_GENAI.md)
- [docs/OPENTELEMETRY_OBSERVABILITY.md](docs/OPENTELEMETRY_OBSERVABILITY.md)
- [docs/DRIFT_MONITORING.md](docs/DRIFT_MONITORING.md)
- [docs/ONNX_RUNTIME_EXPORT.md](docs/ONNX_RUNTIME_EXPORT.md)

## Versioning And Artifacts

See [Evidence And Artifacts](#evidence-and-artifacts) for artifact links. Version 0.3.0 is the current research scaffold. No production release has been made.

## Clinical Disclaimer

NeuroSight is a research/educational portfolio project, **not a medical device** and not intended for clinical use. It is trained on a **synthetic adni-like** dataset only. We disclose that the results represent **not clinical performance** but software pipeline validation on synthetic data. There is **no real patient data** used in the model checkpoint training or evaluations in this repository. The current active model pathway is **cognitive-only** for demonstration. All outputs are non-clinical, do not establish diagnostic utility, and require expert **specialist review** prior to any interpretation.

**This is not clinical performance. This is not a medical device. The project is not clinically validated. Outputs require specialist review. No real patient data is included. Synthetic benchmark results are not medical evidence. The model is not approved for diagnosis, treatment, triage, or emergency use. External validation is not yet performed.**

## Citation

```bibtex
@software{neurosight2026,
  title  = {NeuroSight: Full-Stack Multimodal Medical AI Research Scaffold},
  author = {NeuroSight Contributors},
  year   = {2026},
  note   = {Portfolio and research engineering scaffold using synthetic public data; not for clinical use}
}
```
