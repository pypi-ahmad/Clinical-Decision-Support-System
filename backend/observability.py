"""Optional OpenTelemetry + Prometheus instrumentation.

Everything here is best-effort: if the optional deps aren't installed, the
helpers become no-ops so the app still boots. Enable by setting
``MEDISCAN_OTEL=1`` and/or ``MEDISCAN_PROMETHEUS=1``.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import TYPE_CHECKING, Any

from backend.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI

_logger = get_logger(__name__)

# Header used for request correlation. Accepted on ingress, echoed on egress,
# and bound into structlog contextvars so every log line within a request
# carries the same ``request_id``.
REQUEST_ID_HEADER = "X-Request-ID"


def _flag(name: str) -> bool:
    return os.environ.get(name, "0") in {"1", "true", "True", "yes"}


# --- Prometheus --------------------------------------------------------------

_PROM_METRICS: dict[str, Any] = {}


def _init_prometheus() -> dict[str, Any]:
    if _PROM_METRICS:
        return _PROM_METRICS
    try:
        from prometheus_client import Counter, Histogram
    except Exception:  # pragma: no cover - optional dep
        return _PROM_METRICS

    _PROM_METRICS["request_latency"] = Histogram(
        "mediscan_request_latency_seconds",
        "HTTP request latency.",
        labelnames=("method", "path", "status"),
    )
    _PROM_METRICS["llm_tokens"] = Counter(
        "mediscan_llm_tokens_total",
        "LLM tokens used by provider/model (input + output combined).",
        labelnames=("provider", "model", "kind"),
    )
    _PROM_METRICS["node_duration"] = Histogram(
        "mediscan_graph_node_duration_seconds",
        "Extraction-graph node durations.",
        labelnames=("node",),
    )
    _PROM_METRICS["llm_call_duration"] = Histogram(
        "mediscan_llm_call_duration_seconds",
        "LLM provider call latency (end-to-end wall time including retries).",
        labelnames=("provider", "model", "status"),
    )
    return _PROM_METRICS


def record_llm_tokens(provider: str, model: str, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
    metrics = _init_prometheus()
    counter = metrics.get("llm_tokens")
    if counter is None:
        return
    if input_tokens:
        counter.labels(provider=provider, model=model, kind="input").inc(input_tokens)
    if output_tokens:
        counter.labels(provider=provider, model=model, kind="output").inc(output_tokens)


def record_llm_call(provider: str, model: str, duration: float, *, status: str = "ok") -> None:
    """Record an LLM provider call's wall-clock duration + status.

    ``status`` is ``"ok"`` on success or ``"error"`` on final failure.
    Always emits a structured log line (useful when Prometheus is disabled).
    """
    metrics = _init_prometheus()
    hist = metrics.get("llm_call_duration")
    if hist is not None:
        hist.labels(provider=provider, model=model, status=status).observe(max(0.0, float(duration)))
    _logger.info(
        "llm_call",
        provider=provider,
        model=model,
        duration_ms=int(max(0.0, float(duration)) * 1000),
        status=status,
    )


def record_node_duration(node: str, seconds: float) -> None:
    metrics = _init_prometheus()
    hist = metrics.get("node_duration")
    if hist is not None:
        hist.labels(node=node).observe(max(0.0, float(seconds)))


def instrument_app(app: FastAPI) -> None:
    """Attach Prometheus /metrics + OpenTelemetry FastAPI instrumentation.

    Safe to call unconditionally; each feature is guarded by its env flag and
    by an import check for the optional dependency.
    """
    if _flag("MEDISCAN_PROMETHEUS"):
        try:
            from fastapi import Depends
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
            from starlette.responses import Response

            from backend.security import require_api_key

            _init_prometheus()

            @app.get(
                "/metrics",
                include_in_schema=False,
                dependencies=[Depends(require_api_key)],
            )
            async def _metrics() -> Response:  # pragma: no cover - I/O only
                return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

            _logger.info("prometheus_enabled")
        except Exception as exc:  # pragma: no cover - optional dep
            _logger.warning("prometheus_init_failed", reason=str(exc))

    if _flag("MEDISCAN_OTEL"):
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
            _logger.info("otel_enabled")
        except Exception as exc:  # pragma: no cover - optional dep
            _logger.warning("otel_init_failed", reason=str(exc))


# --- Request context middleware ---------------------------------------------


def install_request_context_middleware(app: FastAPI) -> None:
    """Attach an ASGI middleware that adds a per-request correlation ID.

    Behavior:
      * Accept an inbound ``X-Request-ID`` header (sanitized) or generate a
        UUID4 when absent.
      * Bind ``request_id``, ``method``, and ``path`` into ``structlog``'s
        contextvars so every log line produced during the request carries
        the correlation ID automatically.
      * Echo ``X-Request-ID`` on the response for client-side correlation.
      * Observe request latency into the Prometheus ``request_latency``
        histogram when Prometheus is enabled (no-op otherwise).
    """
    try:
        import structlog
    except Exception:  # pragma: no cover - structlog is a required dep
        structlog = None  # type: ignore[assignment]

    @app.middleware("http")
    async def _request_context(request, call_next):  # type: ignore[no-untyped-def]
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        # Only accept safe characters; otherwise generate a fresh id.
        request_id = (
            incoming
            if incoming and all(c.isalnum() or c in "-_" for c in incoming) and len(incoming) <= 128
            else uuid.uuid4().hex
        )

        if structlog is not None:
            structlog.contextvars.clear_contextvars()
            structlog.contextvars.bind_contextvars(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
            )

        start = time.perf_counter()
        status_code = 500
        response = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration = time.perf_counter() - start
            hist = _PROM_METRICS.get("request_latency") if _PROM_METRICS else None
            if hist is not None:
                try:
                    hist.labels(
                        method=request.method,
                        path=request.url.path,
                        status=str(status_code),
                    ).observe(max(0.0, duration))
                except Exception:  # pragma: no cover - defensive
                    pass
            if response is not None:
                response.headers[REQUEST_ID_HEADER] = request_id
            if structlog is not None:
                structlog.contextvars.clear_contextvars()
