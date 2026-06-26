"""Trace and manifest helpers for the NeuroSight LangGraph workflow."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from knowledge_graph import MockPatientRecord, NeuroKnowledgeGraph
from neurosight.agents.orchestrator import (
    build_diagnosis_graph,
    build_initial_state,
    build_report_from_state,
)

WORKFLOW_NODES: tuple[str, ...] = (
    "supervisor",
    "mri_analyst",
    "eeg_analyst",
    "cognitive_analyst",
    "kg_retriever",
    "report_writer",
    "safety_guardian",
)

WORKFLOW_EDGES: tuple[dict[str, str], ...] = (
    {"from": "supervisor", "to": "mri_analyst", "condition": "MRI data present and not analyzed"},
    {"from": "supervisor", "to": "eeg_analyst", "condition": "EEG data present and not analyzed"},
    {"from": "supervisor", "to": "cognitive_analyst", "condition": "cognitive data present and not analyzed"},
    {"from": "supervisor", "to": "kg_retriever", "condition": "KG context missing"},
    {"from": "supervisor", "to": "report_writer", "condition": "draft report missing"},
    {"from": "supervisor", "to": "safety_guardian", "condition": "final report missing"},
    {"from": "supervisor", "to": "__end__", "condition": "final report complete"},
    {"from": "mri_analyst", "to": "supervisor", "condition": "return control"},
    {"from": "eeg_analyst", "to": "supervisor", "condition": "return control"},
    {"from": "cognitive_analyst", "to": "supervisor", "condition": "return control"},
    {"from": "kg_retriever", "to": "supervisor", "condition": "return control"},
    {"from": "report_writer", "to": "supervisor", "condition": "return control"},
    {"from": "safety_guardian", "to": "supervisor", "condition": "return control"},
)


class DemoMessage:
    """Small response object compatible with LangChain-style `.content` access."""

    def __init__(self, content: str) -> None:
        self.content = content


class DeterministicDemoLLM:
    """Offline LLM stand-in used only for deterministic workflow demonstrations."""

    def invoke(self, messages: list[Any]) -> DemoMessage:
        system_prompt = str(getattr(messages[0], "content", "")).lower() if messages else ""
        if "neuroradiologist" in system_prompt:
            return DemoMessage(
                "MRI analyst: synthetic placeholder MRI context reviewed; medial temporal "
                "atrophy pattern is listed as a workflow example, not a clinical finding."
            )
        if "eeg specialist" in system_prompt:
            return DemoMessage(
                "EEG analyst: synthetic placeholder EEG context reviewed; mild diffuse "
                "theta slowing is listed as a workflow example, not a clinical finding."
            )
        if "neuropsychologist" in system_prompt:
            return DemoMessage(
                "Cognitive analyst: MMSE/MoCA/CDR profile is compatible with an MCI-like "
                "demo pattern. Estimated confidence: 62%."
            )
        if "format the final clinical report" in system_prompt:
            return DemoMessage(
                "Demographics: synthetic patient for workflow demonstration.\n"
                "Clinical History: cognitive profile and optional modality placeholders supplied.\n"
                "Modality Findings: MRI/EEG examples are workflow placeholders; cognitive scores drive the demo.\n"
                "Final Diagnosis: MCI. Confidence: 62%.\n"
                "Recommendations: human specialist review required; not clinical software."
            )
        if "adversarial critic" in system_prompt:
            return DemoMessage("REVIEW_NEEDED: research demo output requires human specialist review.")
        return DemoMessage("No deterministic response configured.")


def utc_now() -> str:
    """Return current UTC time as an ISO-8601 instant."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    """Convert graph payloads into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "__dict__"):
        return json_safe(vars(value))
    return str(value)


def build_demo_patient(patient_id: str, *, include_modalities: bool = True) -> MockPatientRecord:
    """Build a synthetic patient that can exercise the workflow."""
    patient = MockPatientRecord(patient_id)
    if include_modalities:
        patient.mri = {"source": "synthetic-placeholder", "shape": "96x96x96"}
        patient.eeg = {"source": "synthetic-placeholder", "channels": 19, "samples": 1024}
    return patient


def build_demo_kg(patient_id: str) -> NeuroKnowledgeGraph:
    """Build a tiny synthetic KG for agent workflow tracing."""
    kg = NeuroKnowledgeGraph()
    patient = MockPatientRecord(patient_id)
    kg.add_patient(patient)
    kg.add_diagnosis(patient_id, "mci", "2025-01-15", 0.62, "synthetic_demo")
    kg.add_biomarker(patient_id, "hippocampal_volume_z", -1.1, "z-score", "2025-01-15")
    kg.add_similarity(patient_id, "SYN_SIM_0007", 0.81, ["age_band", "moca_range", "cdrsb"])
    return kg


def _event_summary(node_name: str, state: dict[str, Any]) -> dict[str, Any]:
    """Extract a compact per-node summary from a LangGraph state."""
    return {
        "node": node_name,
        "next_agent": state.get("next_agent"),
        "iteration_count": state.get("iteration_count"),
        "has_mri_findings": bool(state.get("mri_findings")),
        "has_eeg_findings": bool(state.get("eeg_findings")),
        "has_cognitive_findings": bool(state.get("cognitive_findings")),
        "has_kg_context": bool(state.get("kg_context")),
        "has_draft_report": bool(state.get("draft_report")),
        "has_final_report": bool(state.get("final_report")),
        "requires_review": bool(state.get("requires_review")),
        "safety_flags": json_safe(state.get("safety_flags", [])),
    }


def trace_workflow(
    *,
    patient_id: str,
    query: str,
    include_modalities: bool = True,
    use_demo_llm: bool = True,
    use_demo_kg: bool = True,
) -> dict[str, Any]:
    """Run the LangGraph workflow and return a JSON-safe execution trace."""
    patient = build_demo_patient(patient_id, include_modalities=include_modalities)
    llm_client = DeterministicDemoLLM() if use_demo_llm else None
    kg = build_demo_kg(patient_id) if use_demo_kg else None
    graph = build_diagnosis_graph(llm_client, kg)
    initial_state = build_initial_state(patient, query)

    events: list[dict[str, Any]] = []
    final_state: dict[str, Any] | None = None
    started = time.perf_counter()
    for step_index, event in enumerate(graph.stream(initial_state), start=1):
        for node_name, state in event.items():
            safe_state = json_safe(state)
            node_event = {
                "step": step_index,
                "status": "completed",
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                **_event_summary(node_name, safe_state),
            }
            events.append(node_event)
            final_state = safe_state

    if final_state is None:
        raise RuntimeError("LangGraph workflow produced no events.")

    report = build_report_from_state(final_state)
    visited_nodes = [event["node"] for event in events]
    return {
        "generated_at": utc_now(),
        "framework": {
            "name": "LangGraph",
            "pinned_dependency": "langgraph==0.1.1",
            "graph_type": "StateGraph",
        },
        "workflow": {
            "nodes": list(WORKFLOW_NODES),
            "edges": list(WORKFLOW_EDGES),
            "entry_point": "supervisor",
            "terminal": "__end__",
            "circuit_breaker_iterations": 10,
        },
        "input": {
            "patient_id": patient_id,
            "query": query,
            "include_modalities": include_modalities,
            "demo_llm": use_demo_llm,
            "demo_kg": use_demo_kg,
            "data_policy": "synthetic_or_operator_supplied_only",
        },
        "execution": {
            "visited_nodes": visited_nodes,
            "events": events,
            "total_steps": len(events),
            "blocked_by_safety": bool(report.blocked_by_safety),
            "requires_review": bool(report.requires_review),
        },
        "result": {
            "patient_id": report.patient_id,
            "diagnosis": report.final_diagnosis.value,
            "confidence": float(report.confidence),
            "requires_review": bool(report.requires_review),
            "blocked_by_safety": bool(report.blocked_by_safety),
            "report_text": report.report_text,
        },
        "clinical_boundary": (
            "This LangGraph trace demonstrates orchestration and safety routing only. "
            "It is not a clinical diagnosis or medical-device validation."
        ),
    }


def workflow_trace_to_json(trace: dict[str, Any]) -> str:
    """Serialize a workflow trace with stable formatting."""
    return json.dumps(trace, indent=2, sort_keys=True)


def write_workflow_trace(trace: dict[str, Any], output_path: str | Path) -> Path:
    """Write a workflow trace to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(workflow_trace_to_json(trace) + "\n", encoding="utf-8")
    return path
