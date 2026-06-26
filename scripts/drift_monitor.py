#!/usr/bin/env python3
"""Run NeuroSight input drift monitoring.

The default mode generates Git-safe synthetic baseline/current cohorts so the
workflow is runnable on any machine. Operators can also pass CSV snapshots with
the required cognitive feature columns to compare real research cohorts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUTPUT = "logs/drift/neurosight_drift_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and current NeuroSight input cohorts for drift."
    )
    parser.add_argument(
        "--baseline-csv",
        help="Optional baseline CSV with cognitive feature columns.",
    )
    parser.add_argument(
        "--current-csv",
        help="Optional current CSV with cognitive feature columns.",
    )
    parser.add_argument(
        "--scenario",
        choices=["stable", "warning", "drift"],
        default="warning",
        help="Synthetic scenario used when CSV files are not provided.",
    )
    parser.add_argument(
        "--baseline-size",
        type=int,
        default=240,
        help="Synthetic baseline row count.",
    )
    parser.add_argument(
        "--current-size",
        type=int,
        default=80,
        help="Synthetic current row count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic cohorts.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of baseline quantile bins for PSI.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout instead of writing a file.",
    )
    return parser.parse_args()


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    from neurosight.monitoring import drift

    has_baseline = bool(args.baseline_csv)
    has_current = bool(args.current_csv)
    if has_baseline != has_current:
        raise ValueError("--baseline-csv and --current-csv must be provided together.")

    if has_baseline and has_current:
        baseline_records = drift.load_feature_csv(args.baseline_csv)
        current_records = drift.load_feature_csv(args.current_csv)
        source = f"csv:{args.baseline_csv}:{args.current_csv}"
    else:
        baseline_records, current_records = drift.generate_synthetic_cohorts(
            scenario=args.scenario,
            n_baseline=max(1, int(args.baseline_size)),
            n_current=max(1, int(args.current_size)),
            seed=int(args.seed),
        )
        source = f"synthetic_demo:{args.scenario}"

    return drift.build_drift_report(
        baseline_records,
        current_records,
        n_bins=max(2, int(args.bins)),
        source=source,
    )


def _format_features(features: object) -> str:
    if not isinstance(features, list) or not features:
        return "none"
    return ", ".join(str(feature) for feature in features)


def summarize_report(
    report: dict[str, Any],
    output_path: Path | None,
    *,
    stream: Any = sys.stdout,
) -> None:
    summary = report.get("summary", {})
    cohorts = report.get("cohorts", {})
    print("DRIFT MONITOR PASSED", file=stream)
    print(f"Status: {report.get('status')}", file=stream)
    print(
        "Rows: "
        f"baseline={cohorts.get('baseline_rows')} "
        f"current={cohorts.get('current_rows')}",
        file=stream,
    )
    print(f"Drifted features: {_format_features(summary.get('drifted_features'))}", file=stream)
    print(f"Warning features: {_format_features(summary.get('warning_features'))}", file=stream)
    print("Clinical boundary: research monitoring only, not clinical validation.", file=stream)
    if output_path is not None:
        print(f"Wrote: {output_path}", file=stream)


def main() -> int:
    from neurosight.monitoring import drift

    args = parse_args()
    try:
        report = build_report(args)
        if args.stdout:
            print(drift.drift_report_to_json(report))
            summarize_report(report, None, stream=sys.stderr)
        else:
            output_path = drift.write_drift_report(report, args.out)
            summarize_report(report, output_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"DRIFT MONITOR FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
