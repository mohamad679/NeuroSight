#!/usr/bin/env python3
"""Run NeuroSight's local security and supply-chain hygiene audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUTPUT = "logs/security/neurosight_supply_chain_report.json"
STRICT_CHOICES = ("none", "critical", "high", "medium", "low")
STRICT_ORDER = {
    "none": 999,
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a redacted local security and supply-chain audit for NeuroSight."
    )
    parser.add_argument(
        "--root",
        default=str(PROJECT_ROOT),
        help="Repository root to scan.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON report path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--strict-on",
        choices=STRICT_CHOICES,
        default="none",
        help="Exit non-zero when findings at or above this severity are present.",
    )
    parser.add_argument(
        "--include-local-secrets",
        action="store_true",
        help="Also scan ignored local env files such as frontend/.env and .env.local.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout instead of writing a file.",
    )
    return parser.parse_args()


def _strict_failure(report: dict[str, Any], strict_on: str) -> bool:
    threshold = STRICT_ORDER[strict_on]
    counts = report.get("summary", {}).get("severity_counts", {})
    return any(
        int(counts.get(severity, 0)) > 0 and level >= threshold
        for severity, level in {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
        }.items()
    )


def summarize_report(
    report: dict[str, Any],
    output_path: Path | None,
    *,
    stream: Any = sys.stdout,
) -> None:
    summary = report.get("summary", {})
    counts = summary.get("severity_counts", {})
    print("SECURITY SUPPLY CHAIN AUDIT COMPLETE", file=stream)
    print(f"Status: {report.get('status')}", file=stream)
    print(
        "Findings: "
        f"critical={counts.get('critical', 0)} "
        f"high={counts.get('high', 0)} "
        f"medium={counts.get('medium', 0)} "
        f"low={counts.get('low', 0)}",
        file=stream,
    )
    print(f"Categories: {', '.join(summary.get('categories', [])) or 'none'}", file=stream)
    print("Secret values are redacted; report includes fingerprints only.", file=stream)
    if output_path is not None:
        print(f"Wrote: {output_path}", file=stream)


def main() -> int:
    from neurosight.security import supply_chain

    args = parse_args()
    report = supply_chain.build_supply_chain_report(
        args.root,
        include_local_secret_files=bool(args.include_local_secrets),
    )
    if args.stdout:
        print(supply_chain.report_to_json(report))
        summarize_report(report, None, stream=sys.stderr)
    else:
        output_path = supply_chain.write_report(report, args.out)
        summarize_report(report, output_path)
    if _strict_failure(report, args.strict_on):
        print(f"Strict mode failed at threshold: {args.strict_on}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
