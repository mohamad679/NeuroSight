# Private Data Adapters

NeuroSight is public-safe by default. The repository does not include real
ADNI, OASIS, hospital, or patient data, and public results remain
synthetic-only unless an authorized user runs private validation locally.

## Scope

The private-data adapter path is for users who already have legitimate access
to external datasets such as ADNI or OASIS. The project does not download,
redistribute, infer, or expose those records.

Current adapter:

```bash
python3 scripts/prepare_adni_cognitive.py \
  --input-csv data/private/adni/ADNIMERGE.csv \
  --output-dir data/processed/adni_cognitive
```

## Expected Input

Place private raw files outside the repository or under `data/private/`.
That directory is gitignored and must remain private.

The cognitive CSV may use common ADNI-style names. The adapter normalizes them
to this canonical schema:

| Canonical column | Required | Notes |
|---|---:|---|
| `subject_id` | yes | Participant identifier, normalized from names such as `RID`, `PTID`, or `participant_id`. |
| `label` | yes | One of `CN`, `MCI`, or `Dementia`; normalized from `DX_bl`, `diagnosis`, or similar baseline labels. |
| `MMSE` | yes | Range `[0, 30]`. |
| `MOCA` | yes | Range `[0, 30]`. |
| `CDRSB` | yes | Range `[0, 18]`. |
| `ADAS11` | yes | Range `[0, 70]`. |
| `RAVLT_immediate` | yes | Range `[0, 75]`. |
| `RAVLT_learning` | yes | Range `[-15, 15]`. |
| `FAQ` | yes | Range `[0, 30]`. |
| `AGE` | yes | Range `[0, 120]`. |
| `visit_date` | optional | Parsed and normalized to `YYYY-MM-DD` when present. |

Rows with missing values, invalid labels, non-numeric features, or out-of-range
feature values are rejected.

## Leakage Checks

The adapter blocks common leakage columns before writing outputs, including:

- Direct label/outcome leakage such as `DXCHANGE`, `final_dx`, `prediction`,
  `target`, `outcome`, and `conversion_status`.
- Post-diagnosis or future-state leakage such as `conversion_date`,
  `months_to_conversion`, `future_dx`, `next_dx`, `followup_dx`, and
  `progression_date`.
- Subject leakage across train, validation, and test splits.

Splits are made by subject ID, not by row, so repeat visits for the same
subject stay in the same split.

## Outputs

The adapter writes:

```text
data/processed/adni_cognitive/train.csv
data/processed/adni_cognitive/val.csv
data/processed/adni_cognitive/test.csv
data/processed/adni_cognitive/dataset_summary.json
```

The summary includes row counts, subject counts, label distributions, split
sizes, and the leakage checks that were applied.

`data/processed/` is available for generated artifacts, but generated outputs
from real cohorts can still be sensitive and must not be committed. The
repository only keeps `data/processed/.gitkeep` so reviewers can see the
intended directory structure.

## What Is Never Committed

Never commit:

- Raw external dataset files.
- Real subject identifiers or visit dates.
- MRI, EEG, DICOM, NIfTI, derived embeddings, or other patient-level files.
- Processed train/validation/test CSVs generated from real cohorts.
- Any metric, figure, or model card text that implies public clinical
  validation unless the validation package is separately approved for release.

## Limitations

This adapter is a preprocessing and validation interface only. It does not
make NeuroSight clinically validated, does not train a validated clinical
checkpoint, and does not establish real-world performance. Any private
validation requires institutional approval, locked preprocessing, provenance,
calibration review, subgroup analysis, expert clinical review, and appropriate
regulatory controls.
