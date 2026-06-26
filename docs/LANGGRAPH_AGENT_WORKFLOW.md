# LangGraph Agent Workflow

NeuroSight uses a LangGraph `StateGraph` to turn model/KG context into a
structured research report with deterministic safety review. This is not a fake
frontend animation: the backend builds and runs a real graph in
`neurosight/agents/orchestrator.py`, and the streaming API now emits events from
the actual graph execution.

## Files

| File | Purpose |
|------|---------|
| `neurosight/agents/orchestrator.py` | Real LangGraph graph definition and legacy diagnosis runner used for non-clinical risk-profile demos |
| `neurosight/agents/workflow_trace.py` | JSON-safe workflow tracing and deterministic offline demo LLM |
| `scripts/langgraph_workflow.py` | Runnable CLI that writes a graph execution trace |
| `logs/langgraph/neurosight_langgraph_trace.json` | Default generated trace path; ignored by Git through `logs/` |

The project pins:

```text
langgraph==0.1.1
langchain==0.2.0
```

## Graph Shape

```text
supervisor
  -> mri_analyst
  -> eeg_analyst
  -> cognitive_analyst
  -> kg_retriever
  -> report_writer
  -> safety_guardian
  -> __end__
```

The graph is supervisor-routed. Every worker returns control to `supervisor`,
which decides the next node based on what is already present in state.

| Node | Responsibility |
|------|----------------|
| `supervisor` | Chooses the next agent and enforces a 10-iteration circuit breaker |
| `mri_analyst` | Writes MRI findings when MRI context is present |
| `eeg_analyst` | Writes EEG findings when EEG context is present |
| `cognitive_analyst` | Writes cognitive findings when cognitive scores are present |
| `kg_retriever` | Retrieves temporal KG context and similar-patient context |
| `report_writer` | Produces the structured draft report |
| `safety_guardian` | Blocks unsafe treatment/directive or overconfident confirmation patterns |

## Safety Behavior

The safety guardian has deterministic rules before any LLM response is trusted:

- Blocks medication dosage or prescribing requests, for example `10mg` or
  `prescribe`.
- Blocks requests that ask the system to confirm a specific diagnosis with
  overconfidence.
- Forces specialist review when blocked.

This means the safety path works even without external LLM APIs. Provider-agnostic/offline report-generation hooks exist, but no external Gemini/OpenAI/Anthropic provider is configured in this release. LLMs must not be used as diagnostic models.

## Run The Workflow Trace

Normal path:

```bash
python3 scripts/langgraph_workflow.py
```

Blocked safety path:

```bash
python3 scripts/langgraph_workflow.py --scenario blocked
```

Print JSON to stdout:

```bash
python3 scripts/langgraph_workflow.py --stdout
```

With Poetry:

```bash
make langgraph-workflow
```

The CLI writes:

```text
logs/langgraph/neurosight_langgraph_trace.json
```

Example summary:

```text
LANGGRAPH WORKFLOW PASSED
Visited nodes: supervisor, mri_analyst, supervisor, eeg_analyst, supervisor, cognitive_analyst, supervisor, kg_retriever, supervisor, report_writer, supervisor, safety_guardian, supervisor
Total steps: 13
Diagnosis: mci confidence=0.62
Blocked by safety: False
```

## Offline Demo LLM

The trace script uses a deterministic offline demo LLM by default. It exists so
reviewers can see a complete report without configuring Gemini/OpenAI/Anthropic.
It is explicitly not clinical intelligence and not a hidden model dependency.
External Gemini, OpenAI, or Anthropic report generation is not implemented in
this release.

To exercise fallback behavior with no LLM:

```bash
python3 scripts/langgraph_workflow.py --no-demo-llm
```

To run only cognitive/KG/report/safety branches:

```bash
python3 scripts/langgraph_workflow.py --cognitive-only
```

## Backend Streaming

The FastAPI endpoint:

```text
POST /v1/risk-profile/stream
```

now streams events from `graph.stream(initial_state)`. Each SSE event includes:

```json
{
  "agent": "cognitive_analyst",
  "status": "completed",
  "next_agent": "cognitive_analyst",
  "iteration_count": 1,
  "requires_review": false,
  "safety_flags": []
}
```

The final event keeps the same frontend contract:

```json
{
  "agent": "complete",
  "status": "done",
  "diagnosis": "mci",
  "confidence": 0.62,
  "requires_review": true,
  "blocked": false,
  "report_text": "..."
}
```

## What This Proves

This item demonstrates:

- Real LangGraph `StateGraph` usage.
- Supervisor-routed multi-agent control flow.
- Node-level execution traces.
- Streaming events tied to actual graph execution.
- Deterministic safety blocking independent of LLM providers.
- Clean offline demo behavior for portfolio review.

## Clinical Boundary

The LangGraph workflow is an orchestration pattern. It can organize findings,
retrieve context, and apply safety rules, but it does not make the underlying
model clinically valid. NeuroSight remains a research prototype; every report
must be treated as non-clinical and review-required.
