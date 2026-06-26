# NeuroSight Modality Preprocessing Contract

Phase 3 makes MRI and EEG ingestion inspectable before real-data-style uploads.

## MRI

Supported upload formats:

- `.npy`
- `.nii`
- `.nii.gz`
- DICOM series as `.zip`

Accepted array layouts:

- `(D,H,W)`
- `(1,D,H,W)`
- `(D,H,W,1)`

Backend preprocessing before MRI encoder inference:

1. Parse the uploaded source file.
2. Validate that the payload is a 3D volume.
3. Ensure channel-first layout.
4. Scale intensity.
5. Resize to `96 x 96 x 96`.
6. Convert to a float32 tensor shaped `(1,1,96,96,96)`.

The MRI encoder returns a 768-dimensional embedding. Public demo mode does not perform clinical-grade MRI QC such as orientation harmonization, spacing validation, skull stripping, sequence/protocol filtering, or scanner-domain checks.

## EEG

Supported upload formats:

- `.npy`
- `.edf`

Accepted array layouts:

- `(19,time)`
- `(time,19)`
- `(epochs,19,time)`

Backend preprocessing before EEG encoder inference:

1. Parse the uploaded source file.
2. Validate 19 EEG channels.
3. Average epochs when an epoch dimension is present.
4. Transpose time-by-channel arrays when needed.
5. Pad or truncate to 1024 timepoints.
6. Convert to a float32 tensor shaped `(1,19,1024)`.

For `.edf`, the backend uses MNE when available:

1. Load EDF.
2. Bandpass filter 1-40 Hz.
3. Notch filter 50 and 60 Hz.
4. Apply average reference.
5. Create fixed-length 4-second epochs.
6. Apply epoch-wise z-score normalization.

The EEG encoder returns a 256-dimensional embedding. Public demo mode does not perform clinical-grade artifact rejection, bad-channel review, montage normalization, ICA, or sampling-rate audit.

## API

| Endpoint | Purpose |
|---|---|
| `GET /v1/modalities/status` | Reports supported formats, accepted shapes, model tensor shapes, optional dependency availability, and upload limits. |
| `POST /v1/upload/mri` | Returns MRI embedding plus preprocessing metadata for the uploaded file. |
| `POST /v1/upload/eeg` | Returns EEG embedding plus preprocessing metadata for the uploaded file. |

All `/v1/*` endpoints require `X-API-Key` outside test mode.

## Scientific Scope

This phase makes ingestion real-data-shaped, not clinically validated. Real claims require authorized cohorts, trained checkpoints, preprocessing QC, model-card updates, and external validation.
