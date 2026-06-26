"""Real-data preparation and validation interface for NeuroSight.

This script defines the expected input format for ADNI-like or OASIS-like
tabular datasets and validates a user-provided CSV against that schema.

IMPORTANT
---------
This script does NOT download, fetch, or access any protected clinical database.
Users must obtain ADNI, OASIS, or any other dataset through the appropriate
authorized channels (e.g., ADNI at https://adni.loni.usc.edu).

This script ONLY:
  - Documents the expected CSV schema.
  - Validates a local CSV file against the schema.
  - Generates a sample schema JSON for reference.
  - Does NOT require any API key or credentials.
  - Does NOT contact any remote server.

Usage
-----
Print schema documentation:

    python3 scripts/prepare_adni_like_dataset.py --schema

Validate an existing CSV:

    python3 scripts/prepare_adni_like_dataset.py --validate /path/to/data.csv

Generate a sample schema JSON:

    python3 scripts/prepare_adni_like_dataset.py --generate-schema docs/adni_schema.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

@dataclass
class _ColumnSpec:
    """Schema specification for one CSV column."""
    name: str
    dtype: str          # "float", "int", or "categorical"
    required: bool
    min_val: float | None = None
    max_val: float | None = None
    allowed_values: list[str] | None = None
    description: str = ""


# Canonical 8-feature cognitive schema + required metadata columns
ADNI_LIKE_SCHEMA: list[_ColumnSpec] = [
    _ColumnSpec(
        "RID", "int", required=True,
        description="Participant/record ID (any unique integer identifier).",
    ),
    _ColumnSpec(
        "DX_bl", "categorical", required=True,
        allowed_values=["CN", "MCI", "Dementia", "FTD", "LBD", "VD"],
        description=(
            "Baseline diagnosis label. "
            "CN=cognitively normal, MCI=mild cognitive impairment, "
            "Dementia=AD-type dementia, FTD=frontotemporal, "
            "LBD=Lewy body, VD=vascular dementia."
        ),
    ),
    _ColumnSpec(
        "AGE", "float", required=True, min_val=0.0, max_val=120.0,
        description="Patient age in years at baseline visit.",
    ),
    _ColumnSpec(
        "PTGENDER", "categorical", required=False,
        allowed_values=["Male", "Female"],
        description="Patient sex (optional, not used in 8-feature canonical schema).",
    ),
    _ColumnSpec(
        "MMSE", "float", required=True, min_val=0.0, max_val=30.0,
        description="Mini-Mental State Examination score (0=worst, 30=best).",
    ),
    _ColumnSpec(
        "MOCA", "float", required=True, min_val=0.0, max_val=30.0,
        description="Montreal Cognitive Assessment score (0=worst, 30=best).",
    ),
    _ColumnSpec(
        "CDRSB", "float", required=True, min_val=0.0, max_val=18.0,
        description="Clinical Dementia Rating Sum of Boxes (0=normal, 18=severe).",
    ),
    _ColumnSpec(
        "ADAS11", "float", required=True, min_val=0.0, max_val=70.0,
        description="ADAS-Cog 11-item score (0=best, 70=worst).",
    ),
    _ColumnSpec(
        "RAVLT_immediate", "float", required=True, min_val=0.0, max_val=75.0,
        description="Rey AVLT immediate recall total (0=worst, 75=best).",
    ),
    _ColumnSpec(
        "RAVLT_learning", "float", required=True, min_val=-15.0, max_val=15.0,
        description="Rey AVLT learning score (Trial 5 minus Trial 1).",
    ),
    _ColumnSpec(
        "FAQ", "float", required=True, min_val=0.0, max_val=30.0,
        description="Functional Activities Questionnaire (0=normal, 30=total dependence).",
    ),
]

_REQUIRED_COLUMNS = [s.name for s in ADNI_LIKE_SCHEMA if s.required]
_ALL_COLUMNS = [s.name for s in ADNI_LIKE_SCHEMA]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Structured result from CSV schema validation."""
    valid: bool
    n_rows: int
    n_columns_found: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DataValidationError(ValueError):
    """Raised when a CSV file fails schema validation."""


def validate_csv(path: str | Path) -> ValidationResult:
    """Validate a CSV file against the ADNI-like schema.

    Checks for required columns, value types, and value ranges.  Does NOT
    load imaging or EEG data.  Does NOT contact any remote service.

    Args:
        path: Path to the CSV file to validate.

    Returns:
        ``ValidationResult`` with per-field errors and warnings.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        DataValidationError: If critical schema requirements are not met.
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pandas is required for CSV validation.") from exc

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    errors: list[str] = []
    warnings: list[str] = []

    # Required columns
    for col_name in _REQUIRED_COLUMNS:
        if col_name not in df.columns:
            errors.append(f"Missing required column: '{col_name}'")

    if errors:
        return ValidationResult(
            valid=False,
            n_rows=len(df),
            n_columns_found=len(df.columns),
            errors=errors,
            warnings=warnings,
        )

    # Per-column checks
    for spec in ADNI_LIKE_SCHEMA:
        if spec.name not in df.columns:
            if spec.required:
                errors.append(f"Missing required column: '{spec.name}'")
            else:
                warnings.append(f"Optional column '{spec.name}' not present.")
            continue

        col = df[spec.name]
        n_null = int(col.isna().sum())
        if n_null > 0:
            warnings.append(f"Column '{spec.name}' has {n_null} null values.")

        if spec.dtype == "float" and spec.min_val is not None and spec.max_val is not None:
            numeric_col = pd.to_numeric(col, errors="coerce")
            out_of_range = numeric_col.notna() & (
                (numeric_col < spec.min_val) | (numeric_col > spec.max_val)
            )
            n_oob = int(out_of_range.sum())
            if n_oob > 0:
                errors.append(
                    f"Column '{spec.name}': {n_oob} values outside valid range "
                    f"[{spec.min_val}, {spec.max_val}]."
                )

        if spec.dtype == "categorical" and spec.allowed_values is not None:
            non_null = col.dropna()
            invalid = non_null[~non_null.isin(spec.allowed_values)]
            if len(invalid) > 0:
                unique_invalid = invalid.unique().tolist()[:5]
                errors.append(
                    f"Column '{spec.name}': found unexpected values {unique_invalid}. "
                    f"Allowed: {spec.allowed_values}."
                )

    # Label distribution check
    if "DX_bl" in df.columns:
        label_counts = df["DX_bl"].value_counts()
        for label in label_counts.index:
            if label_counts[label] < 5:
                warnings.append(
                    f"Class '{label}' has only {label_counts[label]} samples. "
                    "Cross-validation may be unstable."
                )

    return ValidationResult(
        valid=len(errors) == 0,
        n_rows=len(df),
        n_columns_found=len(df.columns),
        errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Schema documentation
# ---------------------------------------------------------------------------

def print_schema_documentation() -> None:
    """Print expected ADNI-like CSV schema to stdout."""
    print("\nNeuroSight — Expected ADNI-like CSV Schema")
    print("=" * 60)
    print(
        "\nThis schema defines the tabular features expected by NeuroSight.\n"
        "Users must obtain real datasets through authorized channels.\n"
        "See https://adni.loni.usc.edu for ADNI access.\n"
    )
    print(f"{'Column':<22} {'Type':<12} {'Required':<10} {'Range/Values'}")
    print("-" * 80)
    for spec in ADNI_LIKE_SCHEMA:
        if spec.dtype == "categorical" and spec.allowed_values:
            range_str = ", ".join(spec.allowed_values)
        elif spec.min_val is not None and spec.max_val is not None:
            range_str = f"[{spec.min_val}, {spec.max_val}]"
        else:
            range_str = "—"
        req_str = "yes" if spec.required else "no"
        print(f"  {spec.name:<20} {spec.dtype:<12} {req_str:<10} {range_str}")
    print()
    for spec in ADNI_LIKE_SCHEMA:
        print(f"  {spec.name}: {spec.description}")
    print()
    print("IMPORTANT: NeuroSight does NOT download protected data automatically.")
    print("Users must provide CSV files from authorized data sources.\n")


def generate_sample_schema_json(output_path: str | Path | None = None) -> dict[str, Any]:
    """Generate and optionally save a sample schema JSON document.

    Args:
        output_path: If provided, save the JSON to this path.

    Returns:
        Schema dictionary.
    """
    schema: dict[str, Any] = {
        "schema_version": "1.0",
        "description": (
            "Expected ADNI-like CSV schema for NeuroSight real-data evaluation. "
            "This file documents format requirements only. "
            "No protected data is included."
        ),
        "clinical_validity": False,
        "data_source_required": (
            "Users must obtain datasets from authorized providers. "
            "NeuroSight does not distribute or download protected clinical data."
        ),
        "columns": [],
    }
    for spec in ADNI_LIKE_SCHEMA:
        col_doc: dict[str, Any] = {
            "name": spec.name,
            "dtype": spec.dtype,
            "required": spec.required,
            "description": spec.description,
        }
        if spec.min_val is not None:
            col_doc["min_val"] = spec.min_val
        if spec.max_val is not None:
            col_doc["max_val"] = spec.max_val
        if spec.allowed_values:
            col_doc["allowed_values"] = spec.allowed_values
        schema["columns"].append(col_doc)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    return schema


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeuroSight real-data schema validation and documentation."
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Print expected CSV schema documentation.",
    )
    parser.add_argument(
        "--validate",
        metavar="CSV_PATH",
        help="Validate an existing CSV file against the schema.",
    )
    parser.add_argument(
        "--generate-schema",
        metavar="OUTPUT_PATH",
        help="Save a sample schema JSON to the given path.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.schema:
        print_schema_documentation()

    if args.generate_schema:
        schema = generate_sample_schema_json(args.generate_schema)
        print(f"Schema saved to: {args.generate_schema}")

    if args.validate:
        result = validate_csv(args.validate)
        print(f"\nValidation result for: {args.validate}")
        print(f"  Rows:    {result.n_rows}")
        print(f"  Columns: {result.n_columns_found}")
        print(f"  Valid:   {result.valid}")
        if result.errors:
            print("\nERRORS:")
            for e in result.errors:
                print(f"  ✗ {e}")
        if result.warnings:
            print("\nWARNINGS:")
            for w in result.warnings:
                print(f"  ! {w}")
        if not result.valid:
            sys.exit(1)

    if not any([args.schema, args.generate_schema, args.validate]):
        print_schema_documentation()


if __name__ == "__main__":
    main()
