# OpenTelemetry Observability

NeuroSight includes a lightweight OpenTelemetry integration for the FastAPI
backend. It is designed to be safe for a public portfolio repository:

- Request correlation always works through `X-Request-ID`.
- Full tracing is opt-in through environment variables.
- The app still starts if OpenTelemetry packages are missing.
- Traces describe engineering behavior only, not clinical correctness.

## What It Instruments

The backend creates spans around the most important demo paths:

| Span | Purpose |
|------|---------|
| `neurosight.http_request` | One wrapper span for every HTTP request |
| `neurosight.diagnose` | Diagnosis request with modality-presence attributes |
| `neurosight.upload_mri` | MRI upload parsing, tensor preparation, and embedding generation |
| `neurosight.upload_eeg` | EEG upload parsing, tensor preparation, and embedding generation |

The middleware also returns observability headers:

| Header | Meaning |
|--------|---------|
| `X-Request-ID` | Request correlation ID; generated when the caller does not supply one |
| `X-Trace-ID` | OpenTelemetry trace ID when tracing is active |
| `X-Process-Time` | Backend request duration in seconds |
| `X-Observability` | `request-id` when basic mode is active, `opentelemetry` when tracing is active |

The `/healthz` and `/` payloads include an `observability` object showing whether
OpenTelemetry is enabled, available, and which exporter is configured.

## Local Console Tracing

Install the project dependencies, then start the backend with tracing enabled:

```bash
poetry install
# or: python3 -m pip install -r requirements.txt
export NEUROSIGHT_OTEL_ENABLED=true
export NEUROSIGHT_OTEL_EXPORTER=console
uvicorn api.main:app --reload --port 8000
```

In another terminal, run the probe:

```bash
poetry run python scripts/otel_probe.py
```

or:

```bash
python3 scripts/otel_probe.py
```

or:

```bash
make otel-probe
```

Expected behavior:

- `healthz` returns `X-Request-ID`, `X-Process-Time`, and `X-Observability`.
- `X-Trace-ID` appears when OpenTelemetry is active.
- The backend terminal prints JSON-like spans from `ConsoleSpanExporter`.
- The diagnosis probe creates both `neurosight.http_request` and
  `neurosight.diagnose` spans.

## OTLP Exporter

To send traces to an OpenTelemetry collector, Jaeger, Grafana Tempo, Honeycomb,
or another OTLP-compatible backend:

```bash
export NEUROSIGHT_OTEL_ENABLED=true
export NEUROSIGHT_OTEL_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
uvicorn api.main:app --reload --port 8000
```

Then run:

```bash
poetry run python scripts/otel_probe.py
```

If the OTLP exporter package is unavailable, the integration falls back to the
console exporter and reports the fallback reason in `/healthz`.

## Example Health Payload

When tracing is disabled, `/healthz` includes:

```json
{
  "observability": {
    "enabled": false,
    "available": false,
    "exporter": "none",
    "service_name": "neurosight-api",
    "reason": "disabled by NEUROSIGHT_OTEL_ENABLED"
  }
}
```

When tracing is enabled and dependencies are installed:

```json
{
  "observability": {
    "enabled": true,
    "available": true,
    "exporter": "console",
    "service_name": "neurosight-api",
    "service_version": "0.3.0",
    "reason": "instrumented"
  }
}
```

## Portfolio Value

This is intentionally more than a UI badge. Reviewers can inspect and run:

- `neurosight/observability/otel.py` for optional tracing setup.
- `api/main.py` for real middleware and domain spans.
- `scripts/otel_probe.py` for a runnable verification script.
- `/healthz` for runtime observability status.

That demonstrates production-minded API work: correlation IDs, latency headers,
structured tracing, exporter configuration, and graceful degradation when an
optional observability stack is not installed.

## Clinical Boundary

Tracing proves that requests moved through the backend and which code paths ran.
It does not prove medical accuracy, model validation, or clinical safety. The
NeuroSight public demo remains a research prototype and every model output must
be treated as non-clinical.
