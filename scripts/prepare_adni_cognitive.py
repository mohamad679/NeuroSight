#!/usr/bin/env python3
"""Prepare authorized ADNI-style cognitive CSVs for local validation.

This adapter is intentionally local-only. It never downloads external data and
it never requires credentials. Authorized users provide a CSV under
``data/private/`` or another secure local path, then this script validates the
schema, blocks common leakage columns, creates subject-disjoint splits, and
writes sanitized processed CSVs plus a summary JSON.

The output may still be sensitive if the input contains real participant data.
Do not commit generated outputs from real cohorts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neurosight.schemas.cognitive import COGNITIVE_FEATURES  # noqa: E402

CANONICAL_SUBJECT_ID = "subject_id"
CANONICAL_LABEL = "label"
CANONICAL_VISIT_DATE = "visit_date"
ALLOWED_LABELS = ("CN", "MCI", "Dementia")
REQUIRED_COLUMNS = (CANONICAL_SUBJECT_ID, CANONICAL_LABEL, *COGNITIVE_FEATURES)
OPTIONAL_COLUMNS = (CANONICAL_VISIT_DATE,)
OUTPUT_COLUMNS = (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)

COLUMN_ALIASES: dict[str, str] = {
    "rid": CANONICAL_SUBJECT_ID,
    "subject": CANONICAL_SUBJECT_ID,
    "subjectid": CANONICAL_SUBJECT_ID,
    "subject_id": CANONICAL_SUBJECT_ID,
    "participant_id": CANONICAL_SUBJECT_ID,
    "ptid": CANONICAL_SUBJECT_ID,
    "dx_bl": CANONICAL_LABEL,
    "diagnosis": CANONICAL_LABEL,
    "baseline_diagnosis": CANONICAL_LABEL,
    "label": CANONICAL_LABEL,
    "examdate": CANONICAL_VISIT_DATE,
    "exam_date": CANONICAL_VISIT_DATE,
    "visitdate": CANONICAL_VISIT_DATE,
    "visit_date": CANONICAL_VISIT_DATE,
    "date": CANONICAL_VISIT_DATE,
    "mmse": "MMSE",
    "moca": "MOCA",
    "cdrsb": "CDRSB",
    "adas11": "ADAS11",
    "ravlt_immediate": "RAVLT_immediate",
    "ravlt_learning": "RAVLT_learning",
    "faq": "FAQ",
    "age": "AGE",
}

LABEL_ALIASES = {
    "normal": "CN",
    "cn": "CN",
    "cognitively normal": "CN",
    "mci": "MCI",
    "mild cognitive impairment": "MCI",
    "dementia": "Dementia",
    "ad": "Dementia",
    "alzheimers disease": "Dementia",
    "alzheimer's disease": "Dementia",
}

FEATURE_RANGES = {
    "MMSE": (0.0, 30.0),
    "MOCA": (0.0, 30.0),
    "CDRSB": (0.0, 18.0),
    "ADAS11": (0.0, 70.0),
    "RAVLT_immediate": (0.0, 75.0),
    "RAVLT_learning": (-15.0, 15.0),
    "FAQ": (0.0, 30.0),
    "AGE": (0.0, 120.0),
}

LABEL_LEAKAGE_FIELDS = {
    "dxchange",
    "dx_change",
    "dx_current",
    "dx",
    "current_diagnosis",
    "final_dx",
    "final_diagnosis",
    "consensus_diagnosis",
    "predicted_dx",
    "prediction",
    "target",
    "outcome",
    "conversion_status",
}

POST_DIAGNOSIS_LEAKAGE_FIELDS = {
    "conversion_date",
    "converter",
    "progression",
    "progression_date",
    "months_to_conversion",
    "time_to_conversion",
    "future_dx",
    "next_dx",
    "followup_dx",
    "post_diagnosis_treatment",
    "death_date",
}


class AdapterValidationError(ValueError):
    """Raised when a private-data adapter input is not safe to process."""


@dataclass(frozen=True)
class PreparedDataset:
    """Paths and summary emitted by the adapter."""

    train_path: Path
    val_path: Path
    test_path: Path
    summary_path: Path
    summary: dict[str, Any]


def _normalize_column_name(name: str) -> str:
    compact = "".join(ch for ch in name.strip().lower() if ch.isalnum() or ch == "_")
    compact = compact.replace("__", "_")
    return COLUMN_ALIASES.get(compact, name.strip())


def _normalize_label(value: object) -> str:
    normalized = str(value).strip()
    mapped = LABEL_ALIASES.get(normalized.lower(), normalized)
    if mapped not in ALLOWED_LABELS:
        raise AdapterValidationError(
            f"Invalid label {value!r}. Allowed labels: {', '.join(ALLOWED_LABELS)}."
        )
    return mapped


def _read_and_normalize(input_csv: Path) -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pandas is required to run the ADNI adapter.") from exc

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if input_csv.suffix.lower() != ".csv":
        raise AdapterValidationError("Only CSV input is supported by this adapter.")

    raw = pd.read_csv(input_csv)
    if raw.empty:
        raise AdapterValidationError("Input CSV is empty.")

    normalized_columns = [_normalize_column_name(col) for col in raw.columns]
    if len(normalized_columns) != len(set(normalized_columns)):
        duplicates = sorted({col for col in normalized_columns if normalized_columns.count(col) > 1})
        raise AdapterValidationError(f"Column aliases collide after normalization: {duplicates}.")

    normalized_original = {
        "".join(ch for ch in str(col).strip().lower() if ch.isalnum() or ch == "_")
        for col in raw.columns
    }
    leakage = sorted(normalized_original & LABEL_LEAKAGE_FIELDS)
    post_diagnosis = sorted(normalized_original & POST_DIAGNOSIS_LEAKAGE_FIELDS)
    if leakage:
        raise AdapterValidationError(
            "Potential label-leakage columns are present and must be removed before "
            f"processing: {', '.join(leakage)}."
        )
    if post_diagnosis:
        raise AdapterValidationError(
            "Potential post-diagnosis leakage columns are present and must be removed "
            f"before processing: {', '.join(post_diagnosis)}."
        )

    df = raw.copy()
    df.columns = normalized_columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise AdapterValidationError(f"Missing required columns: {', '.join(missing)}.")

    df = df[[col for col in OUTPUT_COLUMNS if col in df.columns]].copy()
    if CANONICAL_VISIT_DATE not in df.columns:
        df[CANONICAL_VISIT_DATE] = ""

    for col in REQUIRED_COLUMNS:
        if df[col].isna().any():
            raise AdapterValidationError(f"Column {col!r} contains missing values.")

    df[CANONICAL_SUBJECT_ID] = df[CANONICAL_SUBJECT_ID].astype(str).str.strip()
    if (df[CANONICAL_SUBJECT_ID] == "").any():
        raise AdapterValidationError("subject_id contains blank values.")

    df[CANONICAL_LABEL] = df[CANONICAL_LABEL].map(_normalize_label)
    for feature in COGNITIVE_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        if df[feature].isna().any():
            raise AdapterValidationError(f"Feature {feature!r} contains non-numeric or missing values.")
        low, high = FEATURE_RANGES[feature]
        invalid = (df[feature] < low) | (df[feature] > high)
        if invalid.any():
            raise AdapterValidationError(
                f"Feature {feature!r} has values outside [{low}, {high}]."
            )

    if CANONICAL_VISIT_DATE in df.columns and (df[CANONICAL_VISIT_DATE].astype(str).str.strip() != "").any():
        parsed = pd.to_datetime(df[CANONICAL_VISIT_DATE], errors="coerce")
        if parsed.isna().any():
            raise AdapterValidationError("visit_date contains invalid date values.")
        df[CANONICAL_VISIT_DATE] = parsed.dt.strftime("%Y-%m-%d")

    return df.sort_values([CANONICAL_SUBJECT_ID, CANONICAL_VISIT_DATE]).reset_index(drop=True)


def _split_subjects(df: Any, seed: int, val_fraction: float, test_fraction: float) -> dict[str, set[str]]:
    import numpy as np

    if not 0.0 < val_fraction < 0.5 or not 0.0 < test_fraction < 0.5:
        raise AdapterValidationError("val_fraction and test_fraction must be between 0 and 0.5.")
    if val_fraction + test_fraction >= 0.8:
        raise AdapterValidationError("val_fraction + test_fraction must leave enough training data.")

    label_counts_by_subject = df.groupby(CANONICAL_SUBJECT_ID)[CANONICAL_LABEL].nunique()
    inconsistent = label_counts_by_subject[label_counts_by_subject > 1]
    if not inconsistent.empty:
        examples = sorted(inconsistent.index.astype(str).tolist())[:5]
        raise AdapterValidationError(
            "Subjects with conflicting labels are not supported by this baseline "
            f"adapter: {examples}."
        )

    subjects = df[[CANONICAL_SUBJECT_ID, CANONICAL_LABEL]].drop_duplicates(CANONICAL_SUBJECT_ID)
    if len(subjects) < 3:
        raise AdapterValidationError("At least 3 unique subjects are required for train/val/test splits.")

    rng = np.random.default_rng(seed)
    train: set[str] = set()
    val: set[str] = set()
    test: set[str] = set()

    for _, group in subjects.groupby(CANONICAL_LABEL):
        ids = group[CANONICAL_SUBJECT_ID].astype(str).to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        if n >= 3:
            n_test = max(1, int(round(n * test_fraction)))
            n_val = max(1, int(round(n * val_fraction)))
            if n_test + n_val >= n:
                n_test = 1
                n_val = 1
            test.update(ids[:n_test])
            val.update(ids[n_test : n_test + n_val])
            train.update(ids[n_test + n_val :])
        else:
            train.update(ids)

    remaining = set(subjects[CANONICAL_SUBJECT_ID].astype(str)) - train - val - test
    train.update(remaining)
    if not train or not val or not test:
        raise AdapterValidationError(
            "Could not create non-empty train/val/test splits. Provide more subjects per label."
        )
    return {"train": train, "val": val, "test": test}


def _assert_no_subject_overlap(splits: dict[str, set[str]]) -> None:
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in pairs:
        overlap = splits[left] & splits[right]
        if overlap:
            raise AdapterValidationError(
                f"Subject leakage across {left}/{right}: {sorted(overlap)[:5]}."
            )


def prepare_dataset(
    input_csv: str | Path,
    output_dir: str | Path = ROOT / "data" / "processed" / "adni_cognitive",
    *,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> PreparedDataset:
    """Validate a local authorized CSV and write subject-disjoint splits."""

    df = _read_and_normalize(Path(input_csv))
    splits = _split_subjects(df, seed=seed, val_fraction=val_fraction, test_fraction=test_fraction)
    _assert_no_subject_overlap(splits)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    split_frames = {}
    for name, subject_ids in splits.items():
        split_df = df[df[CANONICAL_SUBJECT_ID].isin(subject_ids)].copy()
        split_frames[name] = split_df
        split_df.to_csv(output / f"{name}.csv", index=False)

    label_counts = {
        split: frame[CANONICAL_LABEL].value_counts().sort_index().astype(int).to_dict()
        for split, frame in split_frames.items()
    }
    summary = {
        "adapter": "prepare_adni_cognitive",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "public_repo_safe": True,
        "contains_real_data": "unknown_user_supplied",
        "clinical_validity": False,
        "source_file_name": Path(input_csv).name,
        "raw_data_committed": False,
        "output_policy": "Do not commit generated outputs from real cohorts.",
        "n_rows": int(len(df)),
        "n_subjects": int(df[CANONICAL_SUBJECT_ID].nunique()),
        "required_columns": list(REQUIRED_COLUMNS),
        "optional_columns": list(OPTIONAL_COLUMNS),
        "label_counts": df[CANONICAL_LABEL].value_counts().sort_index().astype(int).to_dict(),
        "split_rows": {split: int(len(frame)) for split, frame in split_frames.items()},
        "split_subjects": {split: int(len(ids)) for split, ids in splits.items()},
        "split_label_counts": label_counts,
        "leakage_checks": {
            "subject_disjoint_splits": True,
            "label_leakage_fields_blocked": sorted(LABEL_LEAKAGE_FIELDS),
            "post_diagnosis_fields_blocked": sorted(POST_DIAGNOSIS_LEAKAGE_FIELDS),
        },
    }

    summary_path = output / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return PreparedDataset(
        train_path=output / "train.csv",
        val_path=output / "val.csv",
        test_path=output / "test.csv",
        summary_path=summary_path,
        summary=summary,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an authorized local ADNI-style cognitive CSV. "
            "No data is downloaded and no raw data should be committed."
        )
    )
    parser.add_argument("--input-csv", required=True, help="Local authorized CSV path.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "processed" / "adni_cognitive"),
        help="Directory for processed split CSVs and dataset_summary.json.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        prepared = prepare_dataset(
            args.input_csv,
            args.output_dir,
            seed=args.seed,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
        )
    except (AdapterValidationError, FileNotFoundError) as exc:
        print(f"Adapter validation failed: {exc}", file=sys.stderr)
        return 1

    print("Prepared ADNI-style cognitive dataset.")
    print(f"  train:   {prepared.train_path}")
    print(f"  val:     {prepared.val_path}")
    print(f"  test:    {prepared.test_path}")
    print(f"  summary: {prepared.summary_path}")
    print("Reminder: do not commit generated outputs from real cohorts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
