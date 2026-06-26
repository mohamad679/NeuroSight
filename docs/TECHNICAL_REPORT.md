# NeuroSight Technical Report

## 1. Scope

NeuroSight is a multimodal AI engineering scaffold for MRI, EEG, and cognitive-score fusion. This public repository is a portfolio and research-engineering project, not a clinical product, not a medical device, and not a diagnostic system.

The current public benchmark is a synthetic-demo smoke evaluation. It is designed to verify that the pipeline runs end to end and that leakage controls behave as expected. It does not estimate real-world clinical accuracy, robustness, utility, or safety.

## 2. System Summary

The codebase includes:

- MRI encoder path with a MONAI ViT branch when MONAI is available and a compact CPU-safe PyTorch 3D CNN fallback.
- EEG encoder path with temporal convolution, Transformer encoding, and attention pooling.
- Cognitive encoder path using the canonical 8-feature schema: `MMSE`, `MOCA`, `CDRSB`, `ADAS11`, `RAVLT_immediate`, `RAVLT_learning`, `FAQ`, and `AGE`.
- Cross-modal attention fusion with learnable missing-modality tokens.
- FastAPI routes, local UI surfaces, model-card checks, safety checks, and benchmark/reporting scripts.
- LangGraph-style provider-agnostic/offline report-generation hooks. No external Gemini/OpenAI/Anthropic provider is configured in this release.

LLMs must not be used as diagnostic models. Any generated text in this repository is a non-clinical research demo output and requires expert review.

## 3. Public Benchmark Design

The public benchmark uses generated ADNI-like tabular cognitive data and synthetic modality placeholders. MRI and EEG inputs in the public benchmark are intentionally uninformative independent noise, not real imaging or EEG signals.

The benchmark validates:

- Pipeline integrity: data generation, model invocation, metric computation, and report writing run without hidden local files.
- Baseline comparison: chance, majority, classical ML, cognitive neural, and fusion paths are compared under the same synthetic split.
- Leakage controls: MRI/EEG noise is checked so it does not carry cognitive or label-derived information.
- Reporting behavior: generated JSON and Markdown artifacts include synthetic-data warnings and limitation text.
- Collapse detection: collapsed prediction distributions are flagged instead of hidden.

The benchmark does not validate:

- Clinical diagnosis.
- Real-world disease prediction.
- Scanner, acquisition, demographic, or site robustness.
- Calibration for clinical use.
- Safety for patient care.

## 4. Evaluation Artifact

The metrics below are synchronized with [`evaluation/results.json`](../evaluation/results.json). The artifact records:

- `synthetic_data: true`
- `clinical_validity: false`
- `trained_on_real_data: false`
- `leakage_checked: true`
- `leakage_check_passed: true`
- Seed: `42`
- Smoke config: `n_per_class=3`, `cv_folds=2`, `cognitive_epochs=1`, `fusion_epochs=1`

## 5. Smoke Benchmark Results

Top-level metrics in `evaluation/results.json` correspond to the best synthetic smoke baseline in the artifact, `gradient_boosting`:

| Metric | Value |
|---|---:|
| Macro AUC | 0.8333333333333334 |
| Macro F1 | 0.5555555555555555 |
| Accuracy | 0.6666666666666666 |
| Brier score | 0.08973670509274677 |
| ECE | 0.2676041785694544 |

These are smoke-test metrics on tiny synthetic data. They are useful for checking that the benchmark code executes and produces plausible artifacts. They are not medical evidence.

### Baseline Comparison

| Method | Modality | Macro AUC | Macro F1 | Accuracy | Balanced Accuracy | Brier | ECE | Collapsed |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `random_classifier` | none | 0.6000000000000000 | 0.2222222222222222 | 0.3333333333333333 | 0.3333333333333333 | 0.13428632350959976 | 0.32061447764291573 | false |
| `majority_classifier` | none | 0.5000000000000000 | 0.047619047619047616 | 0.16666666666666666 | 0.16666666666666666 | 0.13888888888888887 | 0.0000000000000000555 | true |
| `logistic_regression` | cognitive | 0.7999999999999999 | 0.3611111111111111 | 0.5000000000000000 | 0.5000000000000000 | 0.10211049418502688 | 0.40389157524014224 | false |
| `random_forest` | cognitive | 0.7833333333333333 | 0.4166666666666667 | 0.5000000000000000 | 0.5000000000000000 | 0.09444444444444446 | 0.3333333333333333 | false |
| `gradient_boosting` | cognitive | 0.8333333333333334 | 0.5555555555555555 | 0.6666666666666666 | 0.6666666666666666 | 0.08973670509274677 | 0.2676041785694544 | false |
| `mlp_cognitive_only` | cognitive | 0.5666666666666668 | 0.19444444444444442 | 0.3333333333333333 | 0.3333333333333333 | 0.13611559373986876 | 0.1274730215469996 | false |
| `neurosight_cognitive_only` | cognitive | 0.6000000000000000 | 0.16666666666666666 | 0.16666666666666666 | 0.16666666666666666 | 0.1365559218782965 | 0.17880639682213464 | false |
| `neurosight_fusion` | noise MRI + noise EEG + cognitive | 0.4333333333333333 | 0.047619047619047616 | 0.16666666666666666 | 0.16666666666666666 | 0.25144462747268004 | 0.7497312426567078 | true |

## 6. Fusion Collapse Interpretation

The public fusion result underperforms cognitive-only baselines and is flagged as collapsed in the smoke artifact. This is expected in the current public benchmark because MRI and EEG are deliberately uninformative noise. The desired behavior is that noise modalities do not inflate performance.

This result should not be interpreted as clinical failure or clinical success. It only shows that the public smoke setup does not manufacture a multimodal gain from synthetic noise. Signal-bearing MRI/EEG evaluation would require authorized datasets, locked preprocessing, subject-disjoint train/validation/test splits, and external validation. This repository does not claim that real modalities will automatically improve performance.

## 7. Figures

The repository includes generated figures under `docs/figures/` for reviewer inspection:

- [`confusion_matrix.png`](figures/confusion_matrix.png)
- [`roc_curves.png`](figures/roc_curves.png)
- [`training_curves.png`](figures/training_curves.png)
- [`ablation.png`](figures/ablation.png)
- [`attention_weights.png`](figures/attention_weights.png)
- [`calibration_reliability.png`](figures/calibration_reliability.png)
- [`federated_convergence.png`](figures/federated_convergence.png)

These figures are demo artifacts. They should be read with the same synthetic-only limitations as the JSON benchmark artifact.

## 8. Private-Data Adapter Path

Authorized users can run a local ADNI-style cognitive adapter with their own legitimate dataset access:

```bash
python3 scripts/prepare_adni_cognitive.py \
  --input-csv data/private/adni/ADNIMERGE.csv \
  --output-dir data/processed/adni_cognitive
```

The adapter validates required columns, normalizes common ADNI-style names, rejects missing/out-of-range values, blocks leakage fields, creates subject-disjoint splits, and writes a dataset summary JSON. Raw data and real-cohort outputs must not be committed.

## 9. Limitations

- Public metrics are synthetic-demo smoke metrics only.
- No real patient data is included.
- No public clinical validation has been performed.
- No trained clinical checkpoint is shipped.
- MRI/EEG public benchmark inputs are not real signals.
- Synthetic class labels and distributions do not represent real clinical cohorts.
- Calibration metrics are not suitable for clinical confidence use.
- External Gemini/OpenAI/Anthropic report generation is not implemented.

## 10. Reproducibility

Recommended verification:

```bash
make install
make verify
```

The canonical Python dependency strategy for this release is pip with `requirements.lock` on Python 3.11. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for setup, Docker, and CI details.
