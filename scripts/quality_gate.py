#!/usr/bin/env python3
"""Run NeuroSight's offline CI/CD quality gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUTPUT = "logs/quality/neurosight_quality_gate_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an offline CI/CD quality gate for NeuroSight repository readiness."
    )
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help=f"Output JSON path. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--stdout", action="store_true", help="Print JSON to stdout instead of writing a file.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when blocker gates fail.")
    return parser.parse_args()


def summarize_report(report: dict[str, Any], output_path: Path | None, *, stream: Any = sys.stdout) -> None:
    summary = report.get("summary", {})
    failed_blockers = summary.get("failed_blockers", [])
    failed_warnings = summary.get("failed_warnings", [])
    print("QUALITY GATE COMPLETE", file=stream)
    print(f"Status: {report.get('status')}", file=stream)
    print(
        "Checks: "
        f"passed={summary.get('checks_passed', 0)} "
        f"failed={summary.get('checks_failed', 0)} "
        f"total={summary.get('checks_total', 0)}",
        file=stream,
    )
    print(f"Failed blockers: {', '.join(failed_blockers) or 'none'}", file=stream)
    print(f"Failed warnings: {', '.join(failed_warnings) or 'none'}", file=stream)
    if output_path is not None:
        print(f"Wrote: {output_path}", file=stream)


def main() -> int:
    from neurosight.governance.quality_gate import (
        report_to_json,
        run_quality_gate,
        write_quality_gate_report,
    )

    args = parse_args()
    report = run_quality_gate(PROJECT_ROOT)
    if args.stdout:
        print(report_to_json(report))
        summarize_report(report, None, stream=sys.stderr)
    else:
        output_path = write_quality_gate_report(report, args.out)
        summarize_report(report, output_path)
    if args.strict and report.get("status") != "passed":
        print("Strict mode failed: CI/CD quality gate did not pass.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
