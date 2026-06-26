"""Optional OpenTelemetry integration for NeuroSight.

The application should run even when OpenTelemetry packages are not installed.
When `NEUROSIGHT_OTEL_ENABLED=true`, this module attempts to instrument FastAPI
and export spans through console or OTLP exporters.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional


_OBSERVABILITY_STATUS: Dict[str, Any] = {
    "enabled": False,
    "available": False,
    "exporter": "none",
    "service_name": "neurosight-api",
    "reason": "not initialized",
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def observability_status() -> Dict[str, Any]:
    """Return a JSON-safe observability status payload."""
    return dict(_OBSERVABILITY_STATUS)


def setup_observability(
    app: Any,
    *,
    service_name: str = "neurosight-api",
    service_version: str = "unknown",
) -> Dict[str, Any]:
    """Configure optional OpenTelemetry instrumentation for FastAPI.

    Env vars:
        NEUROSIGHT_OTEL_ENABLED=true
        NEUROSIGHT_OTEL_EXPORTER=console|otlp
        OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    """
    enabled = _env_bool("NEUROSIGHT_OTEL_ENABLED", default=False)
    exporter_name = os.environ.get("NEUROSIGHT_OTEL_EXPORTER", "console").strip().lower()

    _OBSERVABILITY_STATUS.update(
        {
            "enabled": enabled,
            "available": False,
            "exporter": "none",
            "service_name": service_name,
            "service_version": service_version,
            "reason": "disabled by NEUROSIGHT_OTEL_ENABLED",
        }
    )

    if not enabled:
        app.state.observability = observability_status()
        return observability_status()

    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ModuleNotFoundError as exc:
        _OBSERVABILITY_STATUS.update(
            {
                "enabled": True,
                "available": False,
                "exporter": "none",
                "reason": f"missing optional dependency: {exc.name}",
            }
        )
        app.state.observability = observability_status()
        return observability_status()

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
            "deployment.environment": os.environ.get("APP_ENV", "local"),
        }
    )

    provider = TracerProvider(resource=resource)
    status_reason = "instrumented"
    if exporter_name == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            exporter = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
            resolved_exporter = "otlp"
        except ModuleNotFoundError as exc:
            exporter = ConsoleSpanExporter()
            resolved_exporter = "console"
            exporter_name = "console"
            status_reason = f"instrumented with console fallback; OTLP exporter unavailable: {exc.name}"
    else:
        exporter = ConsoleSpanExporter()
        resolved_exporter = "console"

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)

    _OBSERVABILITY_STATUS.update(
        {
            "enabled": True,
            "available": True,
            "exporter": resolved_exporter,
            "service_name": service_name,
            "service_version": service_version,
            "reason": status_reason,
            "otlp_endpoint": os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
        }
    )
    app.state.observability = observability_status()
    return observability_status()


class NoopSpan:
    """Minimal span-like object used when OpenTelemetry is unavailable."""

    def set_attribute(self, key: str, value: Any) -> None:
        del key
        del value

    def record_exception(self, exc: BaseException) -> None:
        del exc

    def set_status(self, status: Any) -> None:
        del status


@contextmanager
def start_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
    """Start an OpenTelemetry span if available, else yield a no-op span."""
    if not _OBSERVABILITY_STATUS.get("available"):
        yield NoopSpan()
        return

    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("neurosight")
    except Exception:
        yield NoopSpan()
        return

    span_manager: Any = None
    try:
        span_manager = tracer.start_as_current_span(name)
        span = span_manager.__enter__()
        set_span_attributes(span, attributes or {})
    except Exception:
        if span_manager is not None:
            try:
                span_manager.__exit__(*sys.exc_info())
            except Exception:
                pass
        yield NoopSpan()
        return

    try:
        yield span
    except BaseException:
        exc_info = sys.exc_info()
        should_suppress = span_manager.__exit__(*exc_info)
        if not should_suppress:
            raise
    else:
        span_manager.__exit__(None, None, None)


def set_span_attributes(span: Any, attributes: Dict[str, Any]) -> None:
    """Set JSON-safe attributes on a span-like object."""
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, str(value))


def current_trace_id() -> Optional[str]:
    """Return the current OpenTelemetry trace ID as hex, if one is active."""
    if not _OBSERVABILITY_STATUS.get("available"):
        return None
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        context = span.get_span_context()
        if not context or not context.is_valid:
            return None
        return format(context.trace_id, "032x")
    except Exception:
        return None
