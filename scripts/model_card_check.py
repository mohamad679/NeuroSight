#!/usr/bin/env python3
"""Validate NeuroSight's model card disclosures and evidence links."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUTPUT = "logs/model_card/neurosight_model_card_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check NeuroSight's model card for required sections, disclosures, and metric consistency."
    )
    parser.add_argument("--model-card", default="MODEL_CARD.md", help="Model-card markdown path.")
    parser.add_argument("--results", default="evaluation/results.json", help="Evaluation results JSON path.")
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help=f"Output JSON path. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--stdout", action="store_true", help="Print JSON to stdout instead of writing a file.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any model-card check fails.")
    return parser.parse_args()


def summarize_report(report: dict[str, Any], output_path: Path | None, *, stream: Any = sys.stdout) -> None:
    summary = report.get("summary", {})
    print("MODEL CARD CHECK COMPLETE", file=stream)
    print(f"Status: {report.get('status')}", file=stream)
    print(
        "Checks: "
        f"passed={summary.get('checks_passed', 0)} "
        f"failed={summary.get('checks_failed', 0)} "
        f"total={summary.get('checks_total', 0)}",
        file=stream,
    )
    if output_path is not None:
        print(f"Wrote: {output_path}", file=stream)


def main() -> int:
    from neurosight.governance.model_card import (
        build_model_card_report,
        report_to_json,
        write_model_card_report,
    )

    args = parse_args()
    report = build_model_card_report(
        args.model_card,
        args.results,
        root=PROJECT_ROOT,
    )
    if args.stdout:
        print(report_to_json(report))
        summarize_report(report, None, stream=sys.stderr)
    else:
        output_path = write_model_card_report(report, args.out)
        summarize_report(report, output_path)
    if args.strict and report.get("status") != "passed":
        print("Strict mode failed: model-card checks did not pass.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
