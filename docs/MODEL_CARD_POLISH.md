# Model Card Polish

This item turns the root `MODEL_CARD.md` into a stronger portfolio artifact and
adds a runnable check so the disclosure quality is protected over time.

## Files

| File | Purpose |
|------|---------|
| `MODEL_CARD.md` | Polished model card with status snapshot, evidence links, metrics, limitations, and clinical boundary |
| `neurosight/governance/model_card.py` | Dependency-free model-card quality checks |
| `scripts/model_card_check.py` | CLI that writes a JSON model-card report |
| `logs/model_card/neurosight_model_card_report.json` | Default generated report path; ignored by Git through `logs/` |

## What Changed

The model card now separates:

- implemented engineering from clinical validation,
- cognitive-only public evaluation from MRI/EEG architectural support,
- synthetic metrics from real-world performance claims,
- explainability diagnostics from disease causality,
- deployment/MLOps evidence from medical-device readiness.

It also links to supporting artifacts such as:

- `evaluation/results.json`
- `docs/MONAI_PIPELINE.md`
- `docs/MLFLOW_REGISTRY.md`
- `docs/DVC_PROVENANCE.md`
- `docs/AI_SAFETY_OWASP_GENAI.md`
- `docs/OPENTELEMETRY_OBSERVABILITY.md`
- `docs/DRIFT_MONITORING.md`
- `docs/ONNX_RUNTIME_EXPORT.md`

## Run The Check

```bash
python3 scripts/model_card_check.py
```

With Poetry:

```bash
make model-card-check
```

Strict mode for CI:

```bash
python3 scripts/model_card_check.py --strict
```

The generated report checks:

- required model-card sections,
- synthetic-data and clinical-use disclosures,
- forbidden clinical/regulatory claims,
- consistency with `evaluation/results.json`,
- evidence links that exist on disk.

## Why This Matters

A polished model card helps a reviewer quickly understand what is real in the
project:

- the multimodal architecture is implemented,
- the public metrics are synthetic,
- MRI/EEG upload paths are real-data-shaped but not validated,
- safety/governance controls are explicit,
- reproducibility commands are available,
- the project is honest about clinical limits.

That combination is much stronger for GitHub than a UI-only demo or a loose
README claim.

## Boundary

This check validates documentation quality and disclosure coverage. It does not
validate model performance, clinical safety, regulatory readiness, or real-world
medical utility.
