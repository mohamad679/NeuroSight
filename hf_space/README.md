---
title: NeuroSight
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
python_version: "3.11"
app_port: 7860
pinned: false
license: mit
short_description: NeuroSight risk demo
---

# 🧠 NeuroSight — Multimodal Neurological Risk Profiling Demo Scaffold

NeuroSight is a research-grade multimodal neurological risk profiling demo scaffold for MRI, EEG, and cognitive-assessment workflows. This Space is a public portfolio demonstration that uses synthetic ADNI-like data and must not be used for clinical decision making, diagnosis, or triage.

> Research prototype demonstrating synthetic pattern matching and risk profiling.
> Trained on **synthetic ADNI-like data** — NOT a medical device and not clinically validated.

## Features

- **6-class synthetic pattern matching:** Normal · MCI · AD · FTD · LBD · VD
- **Cross-modal attention fusion** with learnable missing-modality tokens
- **Explainability:** GradCAM++ · Attention rollout · SHAP-style feature importance
- **Agent orchestration:** LangGraph multi-agent pipeline with safety guardian
- **Knowledge graph:** Temporal patient history tracking

## Tabs

| Tab | Description |
|-----|-------------|
| 🧠 **Risk Profiling** | Enter cognitive scores → get risk profile, confidence, XAI, draft report |
| 📊 **Benchmark** | Run modality ablation and calibration analysis |
| 🔬 **Knowledge Graph** | Explore patient timelines and similar patient retrieval |

## Quick Links

- 📖 [Technical Report](https://github.com/mohi679/neurosight/blob/main/docs/TECHNICAL_REPORT.md)
- 📋 [Model Card](https://github.com/mohi679/neurosight/blob/main/MODEL_CARD.md)
- 💻 [Source Code](https://github.com/mohi679/neurosight)

⚠️ **Disclaimer:** Research prototype only. Not for diagnosis, treatment, or triage. All outputs require specialist review.

## Container Security

The Docker Space builds the Next.js frontend, serves it from the FastAPI backend
on port 7860, does not bake API keys or provider credentials into the image,
and runs the API process as a non-root `neurosight` user.
