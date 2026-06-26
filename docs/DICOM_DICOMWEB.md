# DICOM / DICOMweb Awareness

NeuroSight now includes a concrete DICOM/DICOMweb awareness artifact. This does
not turn the project into a PACS, but it proves that the MRI upload path is
designed with medical-imaging interoperability in mind.

Official DICOM references used for this mapping:

- DICOMweb overview: https://www.dicomstandard.org/using/dicomweb
- DICOMweb RESTful resources: https://www.dicomstandard.org/using/dicomweb/restful-structure
- DICOM PS3.18 Web Services: https://www.dicomstandard.org/standards/view/web-services

## Current Support

NeuroSight currently supports DICOM at the ingestion boundary:

| Capability | Status | Where |
|------------|--------|-------|
| DICOM ZIP upload | Implemented | `POST /v1/upload/mri` |
| DICOM slice parsing | Implemented | `api/main.py`, `_load_dicom_zip_from_bytes` |
| Pixel array stacking | Implemented | DICOM slices become a 3D MRI volume |
| Model preprocessing | Implemented | Volume goes through the MRI tensor path |
| QIDO-RS server | Roadmap | Not implemented |
| WADO-RS server | Roadmap | Not implemented |
| STOW-RS server | Roadmap | Not implemented |
| OHIF/PACS storage | Roadmap | Not implemented |

That distinction matters. The current backend can consume a zipped DICOM series
for model inference, but it does not store studies, query a PACS, or expose a
DICOMweb service.

## New Files

| File | Purpose |
|------|---------|
| `neurosight/interop/dicomweb.py` | DICOMweb route plan, safe DICOM metadata inspector, manifest builder |
| `scripts/dicomweb_manifest.py` | Runnable script for generating a DICOM/DICOMweb awareness manifest |
| `logs/dicomweb/neurosight_dicomweb_manifest.json` | Default generated output path; ignored by Git through `logs/` |

No new dependency was added. The project already lists `pydicom` for DICOM
ingestion. If `pydicom` is not installed, the no-input manifest still works and
the script gives a clear message when asked to inspect DICOM files.

## Run The Manifest

Generate a standards-awareness manifest without local DICOM input:

```bash
python3 scripts/dicomweb_manifest.py
```

Default output:

```text
logs/dicomweb/neurosight_dicomweb_manifest.json
```

With Poetry:

```bash
make dicomweb-manifest
```

Print JSON to stdout:

```bash
python3 scripts/dicomweb_manifest.py --stdout
```

## Inspect A Local DICOM File Or ZIP

Inspect a DICOM file:

```bash
python3 scripts/dicomweb_manifest.py --input path/to/image.dcm
```

Inspect a DICOM series ZIP:

```bash
python3 scripts/dicomweb_manifest.py --input path/to/dicom_series.zip
```

Inspect a folder:

```bash
python3 scripts/dicomweb_manifest.py --input path/to/dicom_folder --max-instances 100
```

The manifest omits direct PHI fields by default:

- `PatientName`: omitted
- `PatientBirthDate`: omitted
- `StudyDate`: presence only
- `SeriesDescription`: presence only
- Pixel data: never written to the manifest
- Patient/study/series/instance identifiers: hashed by default

Raw Study/Series/SOP Instance UIDs can be included for private debugging:

```bash
python3 scripts/dicomweb_manifest.py --input path/to/dicom_series.zip --include-uids
```

`PatientID` remains hashed even with `--include-uids`.

## DICOMweb Route Plan

The manifest lists the DICOMweb services that would be needed for a real
interoperability boundary:

| Service | Method | Purpose | Standard |
|---------|--------|---------|----------|
| QIDO-RS | `GET` | Query studies, series, and instances by metadata | DICOM PS3.18 10.6 |
| WADO-RS | `GET` | Retrieve studies, instances, frames, rendered images, or metadata | DICOM PS3.18 10.4 |
| STOW-RS | `POST` | Store DICOM instances over HTTP multipart payloads | DICOM PS3.18 10.5 |
| RS Capabilities | `OPTIONS` | Discover supported services and media types | DICOM PS3.18 8.9 |

Representative future routes:

```text
GET    /dicomweb/studies?...                                      # QIDO-RS
GET    /dicomweb/studies/{StudyInstanceUID}/metadata              # WADO-RS metadata
GET    /dicomweb/studies/{StudyInstanceUID}/series/{SeriesUID}    # WADO-RS retrieve
POST   /dicomweb/studies                                          # STOW-RS
OPTIONS /dicomweb/studies                                         # capabilities
```

## Recommended Future Architecture

For a public portfolio project, the strongest architecture is:

1. Keep NeuroSight focused on AI inference and reporting.
2. Use Orthanc, dcm4chee, or another dedicated DICOMweb server for storage.
3. Add a thin NeuroSight adapter that receives authorized Study/Series UIDs.
4. Use QIDO-RS for lookup and WADO-RS for scoped retrieval.
5. Convert retrieved instances into the existing MRI tensor path.
6. Keep raw DICOM and PHI out of Git and public demos.

That is more credible than trying to make the research API pretend to be a full
PACS.

## What This Proves

This item demonstrates:

- DICOM file-format awareness.
- DICOMweb service awareness: QIDO-RS, WADO-RS, STOW-RS, capabilities.
- Safe handling of identifiers and PHI-sensitive fields in generated artifacts.
- A real script that can inspect actual DICOM files or ZIPs.
- A realistic architecture for connecting AI inference to clinical imaging
  infrastructure later.

## Clinical Boundary

DICOM compatibility does not imply clinical validation. NeuroSight remains a
research prototype. DICOM metadata inspection and DICOMweb route planning prove
engineering awareness, not diagnostic accuracy or medical-device readiness.
