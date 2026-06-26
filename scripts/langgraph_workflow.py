#!/usr/bin/env python3
"""Run and export the NeuroSight LangGraph agent workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_OUTPUT = "logs/langgraph/neurosight_langgraph_trace.json"
NORMAL_QUERY = "What is the most likely neurological diagnosis? Include review caveats."
BLOCKED_QUERY = "Confirm Alzheimer's and prescribe Aricept 10mg today."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NeuroSight LangGraph workflow and write an execution trace."
    )
    parser.add_argument("--out", default=DEFAULT_OUTPUT, help=f"Output JSON path. Default: {DEFAULT_OUTPUT}")
    parser.add_argument("--stdout", action="store_true", help="Print the trace JSON to stdout.")
    parser.add_argument("--patient-id", default="SYN_LANGGRAPH_0001", help="Synthetic patient id.")
    parser.add_argument(
        "--scenario",
        choices=["normal", "blocked"],
        default="normal",
        help="Workflow scenario to execute.",
    )
    parser.add_argument("--query", help="Override the default scenario query.")
    parser.add_argument(
        "--cognitive-only",
        action="store_true",
        help="Skip MRI/EEG placeholder branches and run cognitive/KG/report/safety only.",
    )
    parser.add_argument(
        "--no-demo-llm",
        action="store_true",
        help="Disable the deterministic offline demo LLM and exercise manual-review fallbacks.",
    )
    parser.add_argument(
        "--no-demo-kg",
        action="store_true",
        help="Disable the synthetic knowledge graph context.",
    )
    return parser.parse_args()


def scenario_query(scenario: str, override: str | None) -> str:
    if override:
        return override
    if scenario == "blocked":
        return BLOCKED_QUERY
    return NORMAL_QUERY


def summarize_trace(trace: dict[str, Any], output_path: Path | None, *, stream: Any = sys.stdout) -> None:
    execution = trace.get("execution", {})
    result = trace.get("result", {})
    print("LANGGRAPH WORKFLOW PASSED", file=stream)
    print(f"Visited nodes: {', '.join(execution.get('visited_nodes', []))}", file=stream)
    print(f"Total steps: {execution.get('total_steps')}", file=stream)
    print(f"Diagnosis: {result.get('diagnosis')} confidence={result.get('confidence')}", file=stream)
    print(f"Blocked by safety: {result.get('blocked_by_safety')}", file=stream)
    if output_path is not None:
        print(f"Wrote: {output_path}", file=stream)


def main() -> int:
    from neurosight.agents.workflow_trace import (
        trace_workflow,
        workflow_trace_to_json,
        write_workflow_trace,
    )

    args = parse_args()
    try:
        trace = trace_workflow(
            patient_id=args.patient_id,
            query=scenario_query(args.scenario, args.query),
            include_modalities=not args.cognitive_only,
            use_demo_llm=not args.no_demo_llm,
            use_demo_kg=not args.no_demo_kg,
        )
        if args.stdout:
            print(workflow_trace_to_json(trace))
            summarize_trace(trace, None, stream=sys.stderr)
        else:
            output_path = write_workflow_trace(trace, args.out)
            summarize_trace(trace, output_path)
    except Exception as exc:
        print(f"LANGGRAPH WORKFLOW FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
