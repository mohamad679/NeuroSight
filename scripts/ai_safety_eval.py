#!/usr/bin/env python3
"""Run NeuroSight's OWASP GenAI safety regression suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUTPUT = "logs/safety/neurosight_ai_safety_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic AI safety checks mapped to the OWASP GenAI Top 10."
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON report path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout instead of writing a file.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any safety regression case fails.",
    )
    parser.add_argument(
        "--prompt",
        help="Evaluate one ad-hoc prompt instead of the full regression suite.",
    )
    return parser.parse_args()


def summarize_suite(
    report: dict[str, Any],
    output_path: Path | None,
    *,
    stream: Any = sys.stdout,
) -> None:
    summary = report.get("summary", {})
    print("AI SAFETY EVAL COMPLETE", file=stream)
    print(f"Status: {report.get('status')}", file=stream)
    print(
        "Cases: "
        f"passed={summary.get('passed_cases', 0)} "
        f"failed={summary.get('failed_cases', 0)} "
        f"total={summary.get('total_cases', 0)}",
        file=stream,
    )
    print(
        "OWASP risks covered: "
        f"{', '.join(summary.get('covered_owasp_risks', [])) or 'none'}",
        file=stream,
    )
    missing = summary.get("missing_owasp_risks", [])
    if missing:
        print(f"Missing risk coverage: {', '.join(missing)}", file=stream)
    if output_path is not None:
        print(f"Wrote: {output_path}", file=stream)


def summarize_prompt(report: dict[str, Any], *, stream: Any = sys.stdout) -> None:
    print("AI SAFETY PROMPT CHECK", file=stream)
    print(f"Action: {report.get('action')}", file=stream)
    print(f"Requires review: {report.get('requires_review')}", file=stream)
    print(f"Flags: {', '.join(report.get('flags', [])) or 'none'}", file=stream)
    print(f"OWASP risks: {', '.join(report.get('owasp_risks', [])) or 'none'}", file=stream)


def main() -> int:
    from neurosight.governance.ai_safety import (
        evaluate_ai_safety_prompt,
        report_to_json,
        run_ai_safety_evaluation,
        write_ai_safety_report,
    )

    args = parse_args()
    if args.prompt is not None:
        decision = evaluate_ai_safety_prompt(args.prompt)
        report = {
            "prompt_chars": decision.prompt_chars,
            "action": decision.action,
            "requires_review": decision.requires_review,
            "flags": list(decision.flags),
            "matched_controls": list(decision.matched_controls),
            "owasp_risks": list(decision.owasp_risks),
            "rationale": list(decision.rationale),
        }
        print(report_to_json(report))
        summarize_prompt(report, stream=sys.stderr)
        return 1 if args.strict and decision.action == "allow" else 0

    report = run_ai_safety_evaluation()
    if args.stdout:
        print(report_to_json(report))
        summarize_suite(report, None, stream=sys.stderr)
    else:
        output_path = write_ai_safety_report(report, args.out)
        summarize_suite(report, output_path)
    if args.strict and report.get("status") != "passed":
        print("Strict mode failed: AI safety regression suite did not pass.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
