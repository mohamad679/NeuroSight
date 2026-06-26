# ONNX Runtime Export

NeuroSight now includes an ONNX export path for the cognitive classifier. This
is the safest first export target because the cognitive branch is small,
deterministic, and directly connected to the current risk-profile workflow.

The exporter produces:

- an ONNX model artifact,
- an ONNX checker result,
- optional ONNX Runtime CPU validation,
- a JSON manifest describing model inputs, outputs, dependency state, checkpoint
  loading, and clinical boundaries.

## Files

| File | Purpose |
|------|---------|
| `neurosight/deployment/onnx_export.py` | Cognitive classifier ONNX export and ONNX Runtime validation helpers |
| `scripts/onnx_export.py` | Runnable CLI for export, validation, and manifest generation |
| `docs/ONNX_RUNTIME_EXPORT.md` | Export contract and usage notes |
| `logs/onnx/neurosight_cognitive_classifier.onnx` | Default generated ONNX artifact path; ignored by Git |
| `logs/onnx/neurosight_onnx_export_manifest.json` | Default generated manifest path; ignored by Git through `logs/` |

Generated `.onnx` and `.ort` files are ignored because they are binary model
artifacts, not source code.

## Optional Dependencies

ONNX export dependencies are isolated in a Poetry group so the public backend
does not become heavier unless export work is requested:

```bash
poetry install --with onnx
```

For pip-based workflows:

```bash
pip install onnx onnxruntime
```

The script still runs without these packages. In that case it writes a manifest
with `status: missing_dependencies` and tells the operator what to install. Use
`--strict` in CI when missing ONNX dependencies should fail the job.

## Export Command

Default export:

```bash
python3 scripts/onnx_export.py
```

With Poetry:

```bash
make onnx-export
```

Export from a trained NeuroSight checkpoint:

```bash
python3 scripts/onnx_export.py \
  --checkpoint checkpoints/best_fusion.pt \
  --out logs/onnx/neurosight_cognitive_classifier.onnx
```

If the checkpoint contains `cog_state`, the exporter loads it. If no checkpoint
exists, the manifest honestly records that the ONNX file came from freshly
initialized weights.

## Tensor Contract

Input:

| Name | Shape | Dtype |
|------|-------|-------|
| `cognitive_features` | `(batch, 8)` | `float32` |

Feature order:

```text
MMSE, MOCA, CDRSB, ADAS11, RAVLT_immediate, RAVLT_learning, FAQ, AGE
```

Outputs:

| Name | Shape | Meaning |
|------|-------|---------|
| `logits` | `(batch, 6)` | Temperature-scaled diagnosis logits |
| `probabilities` | `(batch, 6)` | Softmax probabilities over diagnosis classes |
| `embedding` | `(batch, 64)` | Cognitive latent embedding used by the fusion stack |

Diagnosis class order:

```text
normal, mci, ad, ftd, lbd, vd
```

The batch dimension is dynamic in the exported ONNX graph.

## Runtime Validation

When `onnxruntime` is installed, the exporter runs the ONNX graph on CPU and
compares each output against the PyTorch reference output:

```text
logits max_abs_diff <= 1e-4
probabilities max_abs_diff <= 1e-4
embedding max_abs_diff <= 1e-4
```

Skip runtime validation only when producing an artifact in a constrained build
environment:

```bash
python3 scripts/onnx_export.py --skip-runtime-validation
```

## Manifest

The generated manifest records:

- dependency availability and versions,
- checkpoint path and whether `cog_state` was loaded,
- ONNX output path and file size,
- ONNX checker status,
- ONNX Runtime validation status,
- exact input/output tensor contracts,
- clinical boundary text.

Print the manifest to stdout:

```bash
python3 scripts/onnx_export.py --stdout
```

## Why Cognitive First

The fusion model currently returns a Python dictionary and includes modality
optional branches plus attention artifacts. That is useful for the FastAPI demo
but less clean as a first ONNX target. The cognitive classifier gives a
reviewer a real deployment artifact with a simple and verifiable contract.

The next export step would be a dedicated fusion-serving wrapper that accepts
three embedding tensors plus modality-presence masks and returns only tensor
outputs. That wrapper should avoid Python dictionaries and numpy conversion so
it remains ONNX-friendly.

## What This Proves

This item demonstrates:

- PyTorch-to-ONNX export design,
- dynamic batch axes,
- ONNX checker validation,
- ONNX Runtime CPU parity validation,
- checkpoint-aware export,
- deployment artifact manifests,
- honest optional dependency handling.

## Clinical Boundary

ONNX export validates deployment mechanics only. It does not validate clinical
performance, improve model quality, or make the exported model suitable for
medical use. NeuroSight remains a research prototype and every output remains
review-required.
