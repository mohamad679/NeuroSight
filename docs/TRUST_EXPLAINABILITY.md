# NeuroSight Trust And Explainability Contract

Phase 5 adds read-only trust surfaces that make the demo easier to inspect without changing backend inference logic.

## Explainability Scope

| Modality | Public runtime status | Method | Important boundary |
|---|---|---|---|
| Cognitive | Implemented | `gradient_x_input` | Explains the current model response over cognitive features only. |
| MRI | Architecture-supported, runtime-limited | `gradcam_plus_plus` | Requires uploaded image tensors, trained checkpoints, and image QC before real claims. |
| EEG | Architecture-supported, runtime-limited | `attention_rollout` | Requires uploaded EEG tensors, montage/QC review, and trained checkpoints before real claims. |
| Fusion | Model output available | `cross_modal_attention` | Attention weights are model diagnostics, not biomarkers. |

Explainability payloads must be read as model-behavior diagnostics. They do not prove disease causality and are not clinical evidence.

## Privacy And Security Scope

The public repository remains safe for GitHub and Hugging Face demos:

- No private ADNI records are bundled.
- Public demo data is synthetic or mock only.
- `/v1/*` endpoints require `X-API-Key` outside `APP_ENV=test`.
- MRI/EEG uploads are size-limited.
- NumPy uploads use `allow_pickle=False`.
- DICOM zip uploads reject path traversal, encrypted archives, excessive member counts, and excessive expansion.
- NIfTI uploads use temporary files that are deleted after parsing.

Operators with legitimate data access may point environment variables at external ADNI-style files. Those files should stay outside the public repository.

## Reporting Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/xai/status` | Reports XAI method availability, method boundaries, and interpretation policy. |
| `GET /v1/xai/{patient_id}` | Returns patient-level XAI payloads with method and clinical-use disclosure. |
| `GET /v1/governance/status` | Reports privacy, security, upload controls, and scientific guardrails. |
| `GET /healthz` | Includes XAI and governance summaries for UI readiness checks. |

## Definition Of Done For Phase 5

- XAI methods are visible before a user runs an explanation.
- XAI responses include interpretation and privacy disclosures.
- Privacy/security posture is visible in the API and local UI.
- The UI has a Trust view for repository safety, upload controls, and scientific limits.
- Tests protect the no-clinical-claims behavior.
