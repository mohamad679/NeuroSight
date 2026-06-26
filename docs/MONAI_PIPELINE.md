# MONAI MRI Pipeline

NeuroSight includes a real MONAI-backed MRI inference path. The public demo
still uses synthetic or unvalidated weights, so this document describes the
engineering pipeline and its limits rather than claiming clinical performance.

## Purpose

The MRI branch converts a 3D brain volume into a fixed 768-dimensional embedding
that can be fused with EEG and cognitive features.

```text
MRI upload / dataset volume
  -> format loader
  -> shape validation
  -> MONAI transforms
  -> MONAI 3D ViT encoder
  -> 768-d MRI embedding
  -> multimodal fusion model
```

Relevant code:

- `neurosight/models/mri.py` - MONAI ViT encoder and transforms
- `api/main.py` - upload parsing and runtime tensor preparation
- `neurosight/data/adni_dataset.py` - ADNI-style dataset MRI loading
- `docs/MODALITY_PREPROCESSING.md` - upload contract and safety notes

## Accepted MRI Inputs

The FastAPI upload endpoint accepts:

| Format | Loader path | Notes |
| --- | --- | --- |
| `.npy` | NumPy array loader | Expected 3D array shaped `(D,H,W)` or channelized `(1,D,H,W)` |
| `.nii` | nibabel | NIfTI file parsed into a float32 volume |
| `.nii.gz` | nibabel | Compressed NIfTI path |
| `.zip` | pydicom | DICOM series zip, stacked into a 3D volume |

Endpoint:

```text
POST /v1/upload/mri
```

The endpoint returns an MRI embedding payload:

```json
{
  "status": "ok",
  "embedding_dim": 768,
  "embedding": [0.01, -0.03, "..."],
  "preprocessing": {
    "prepared_tensor_shape": [1, 1, 96, 96, 96]
  }
}
```

## Tensor Contract

The runtime MRI preparation path enforces the following contract:

| Stage | Shape |
| --- | --- |
| Raw volume | `(D,H,W)` or `(1,D,H,W)` |
| MONAI transform output | `(1,96,96,96)` |
| Encoder input | `(1,1,96,96,96)` |
| Encoder output | `(1,768)` |

Invalid dimensionality is rejected before inference.

## MONAI Transform Chain

When MONAI is installed, `get_mri_transforms()` returns:

```python
Compose([
    EnsureChannelFirst(channel_dim="no_channel"),
    ScaleIntensity(),
    Resize((96, 96, 96)),
    ToTensor(),
])
```

This gives the model a consistent single-channel 3D volume. It is useful for
portfolio and research prototyping, but it is not a complete clinical MRI
preprocessing pipeline.

## MONAI Encoder

`MRIEncoder` uses `monai.networks.nets.ViT` when MONAI is available:

```python
ViT(
    in_channels=1,
    img_size=(96, 96, 96),
    patch_size=(16, 16, 16),
    pos_embed="conv",
    classification=True,
    num_classes=768,
)
```

The model output is treated as the MRI embedding. The classifier wrapper adds a
linear head for six demo classes and a learnable temperature parameter for
calibration experiments.

## Fallback Behavior

If MONAI is unavailable, the MRI encoder falls back to a compact 3D CNN stem
implemented in PyTorch:

```text
96x96x96 volume
  -> Conv3D + GroupNorm + SiLU blocks
  -> AdaptiveAvgPool3d(1)
  -> LayerNorm + Linear(64, 768)
  -> 768-d MRI embedding
```

This replaces the previous flattened-volume projection
`Linear(96 * 96 * 96, 768)`, which required 679,478,016 parameters. The fallback
now has fewer than 250,000 parameters and is suitable for CPU demo, CI, and
Hugging Face Spaces smoke paths. It is still a demo/engineering fallback, not a
clinically validated imaging model.

This distinction matters for GitHub reviewers:

- Local/backend research mode should use the MONAI ViT path when MONAI is available.
- Lightweight public demo mode may use the compact CNN fallback path.
- Clinical claims require a trained checkpoint and validated preprocessing.

## Fusion With Other Modalities

After MRI upload, `/v1/risk-profile` can receive:

```json
{
  "mri_embedding": [768 values],
  "eeg_embedding": [256 values],
  "cog_embedding": [64 values],
  "cognitive_scores": {
    "MMSE": 24,
    "MOCA": 20,
    "CDRSB": 1,
    "ADAS11": 18,
    "RAVLT_immediate": 34,
    "RAVLT_learning": 2,
    "FAQ": 6,
    "AGE": 70
  }
}
```

The fusion model projects modality embeddings into a shared latent space and
uses missing-modality tokens when MRI or EEG is unavailable.

## Smoke Test

The backend proof script generates a synthetic `.npy` MRI volume, uploads it,
and verifies that the backend returns an embedding:

```bash
make smoke-backend
```

For a cognitive-only proof run:

```bash
python scripts/smoke_backend.py --skip-uploads
```

If the backend is using a non-default API key:

```bash
python scripts/smoke_backend.py --api-key <matching-backend-key>
```

## What This Proves

This pipeline demonstrates that the repository contains more than a UI shell:

- medical-imaging input parsing,
- MONAI transform wiring,
- a 3D ViT encoder path,
- embedding extraction,
- FastAPI upload endpoints,
- multimodal fusion integration,
- clear runtime disclosures.

## What This Does Not Prove

The current public/demo setup does not prove clinical diagnostic performance.
The following are still required before any real clinical claim:

- authorized MRI cohort,
- scanner/protocol metadata review,
- orientation and spacing harmonization,
- skull stripping or equivalent brain masking,
- sequence/protocol filtering,
- trained and versioned checkpoint,
- external validation,
- bias and calibration evaluation,
- clinical model-card update.

## Reviewer-Friendly Summary

Use this wording in GitHub or interviews:

> NeuroSight includes a MONAI-backed 3D MRI encoder path using a ViT over
> `96^3` volumes and produces 768-dimensional embeddings for multimodal fusion.
> The public demo uses synthetic/unvalidated data for safety, but the backend
> includes real MRI upload parsing, MONAI transforms, and FastAPI inference
> contracts.
