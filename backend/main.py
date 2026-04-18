"""FastAPI entry point.

This file is intentionally narrow — route handlers parse inputs, delegate to
``backend.extract`` / ``backend.workflows`` / ``backend.logic``, and translate
failures into sanitized HTTP errors. Notable changes over the original:

* CORS is configurable; defaults to the Streamlit origin. Wildcards only via
  ``MEDISCAN_ALLOWED_ORIGINS``.
* Every mutating endpoint requires ``X-API-Key`` (or explicit
  ``MEDISCAN_ALLOW_ANONYMOUS=1`` opt-in).
* Uploads are validated (extension + magic bytes + byte cap) and streamed
  via :func:`backend.security.write_upload_with_limit`.
* ``paddle_service_url`` is no longer client-controlled — it is read from
  ``PADDLE_SERVICE_URL`` on the server.
* API keys for external providers are read from the server environment; the
  per-request form fields were removed.
* ``HTTPException`` details never echo raw pipeline output; callers receive a
  correlation ID and the details are logged server-side.
* Artifact serving goes through an authenticated endpoint that enforces path
  canonicalisation; the static mount is no longer exposed on disk roots.
* Blocking I/O (pdf2image, sqlite, sync LLM clients) is offloaded via
  ``run_in_threadpool`` — the event loop never stalls.
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiofiles
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Form as FormFieldInfo
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from backend.artifacts import ensure_upload_root
from backend.database import init_db, record_audit_event, save_record
from backend.extract import process_document_pipeline, run_document_ocr
from backend.lineage import capture_lineage
from backend.logic import analyze_medical_logic, check_insurance_coverage
from backend.logging_config import configure_logging, get_logger
from backend.models import MedicalRecord, MedicalRecordStrict
from backend.retrieval import (
    build_chunks_from_ocr_payload,
    build_chunks_from_text,
    create_vector_store,
    hash_identifier,
)
from backend.retrieval.chunking import sanitize_retrieved_text
from backend.security import (
    MAX_UPLOAD_BYTES,
    require_api_key,
    resolve_artifact_path,
    sanitize_filename,
    validate_outbound_url,
    validate_upload_or_raise,
    write_upload_with_limit,
)
from backend.workflows.agentic_extraction import run_agentic_extraction_workflow
from backend.workflows.extraction_graph import run_extraction_graph


configure_logging()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: DB bootstrap + upload root
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    ensure_upload_root()
    init_db()
    # Security: refuse to run with the well-known placeholder key from
    # docker-compose.yml. Operators must supply a real secret.
    if os.environ.get("MEDISCAN_API_KEY", "").strip().lower() in {"changeme", "change-me"}:
        raise RuntimeError(
            "MEDISCAN_API_KEY is set to the placeholder 'changeme'. Generate a "
            "real secret (e.g. `python -c 'import secrets; print(secrets.token_urlsafe(32))'`) "
            "and set it via the environment before starting the server."
        )
    # Warm up expensive singletons so the first request doesn't pay cold start.
    try:
        from backend.workflows.extraction_graph import _get_compiled_graph as _get_granular
        from backend.workflows.agentic_extraction import _get_compiled_graph as _get_agentic

        _get_granular()
        _get_agentic()
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("graph_warmup_failed", reason=str(exc))
    try:
        # Touch the vector store factory so any misconfiguration surfaces at
        # boot instead of during the first /analyze call.
        create_vector_store()
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("vector_store_warmup_failed", reason=str(exc))
    yield


# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------


def _build_app() -> FastAPI:
    disable_docs = os.environ.get("MEDISCAN_ENABLE_DOCS", "0") != "1"
    return FastAPI(
        lifespan=_lifespan,
        docs_url=None if disable_docs else "/docs",
        redoc_url=None if disable_docs else "/redoc",
        openapi_url=None if disable_docs else "/openapi.json",
    )


app = _build_app()

# Optional OpenTelemetry + Prometheus instrumentation (no-op unless env flags set).
try:  # pragma: no cover - optional dependency wiring
    from backend.observability import install_request_context_middleware, instrument_app

    instrument_app(app)
    install_request_context_middleware(app)
except Exception as _obs_exc:  # pragma: no cover
    logger.warning("observability_init_skipped", reason=str(_obs_exc))


# ---------------------------------------------------------------------------
# Rate limiting (slowapi) — optional dependency; no-op when not installed or
# when the operator explicitly disables it with ``MEDISCAN_RATE_LIMIT=0``.
# ---------------------------------------------------------------------------

try:  # pragma: no cover - optional dependency wiring
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    _rate_limit_enabled = os.environ.get("MEDISCAN_RATE_LIMIT", "1") != "0"
    if _rate_limit_enabled:
        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[os.environ.get("MEDISCAN_DEFAULT_RATE", "60/minute")],
        )
        app.state.limiter = limiter
        app.add_middleware(SlowAPIMiddleware)

        async def _rate_limit_handler(request, exc):
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})

        app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
except Exception:  # pragma: no cover - slowapi optional
    pass


def _allowed_origins() -> list[str]:
    raw = os.environ.get("MEDISCAN_ALLOWED_ORIGINS")
    if not raw:
        # Default to the local Streamlit dev server only.
        return ["http://localhost:8501", "http://127.0.0.1:8501"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_form_value(value, fallback=None):
    if isinstance(value, FormFieldInfo):
        return value.default if value.default is not None else fallback
    if value is None:
        return fallback
    # Treat empty strings from multipart forms as unset so downstream code can
    # rely on ``None`` instead of matching "" everywhere.
    if isinstance(value, str) and value.strip() == "":
        return fallback
    return value


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _correlation_id() -> str:
    return uuid.uuid4().hex


def _fail(correlation_id: str, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
    """Raise a sanitised HTTPException and log the internal detail."""
    logger.warning("request_failed", correlation_id=correlation_id, reason=message, status=status_code)
    raise HTTPException(
        status_code=status_code,
        detail={"error": "Request failed", "correlation_id": correlation_id},
    )


def _api_key_for(provider: str, user_supplied: str | None) -> str | None:
    """Resolve the provider API key.

    Server-side environment variables always take precedence to prevent a
    caller from burning somebody else's quota.  ``user_supplied`` is honoured
    only when explicitly permitted by ``MEDISCAN_ALLOW_USER_API_KEYS=1``.
    """
    normalized = (provider or "").strip().lower()
    mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_name = mapping.get(normalized)
    env_value = os.environ.get(env_name) if env_name else None
    if env_value:
        return env_value
    if user_supplied and os.environ.get("MEDISCAN_ALLOW_USER_API_KEYS") == "1":
        return user_supplied
    return None


def _resolve_paddle_service_url() -> str | None:
    """Paddle service URL is server-configured only (SSRF guard).

    Read exclusively from the ``PADDLE_SERVICE_URL`` environment variable and
    validated through :func:`validate_outbound_url`. There is no client-facing
    input path for this value — the previous ``paddle_service_url`` form
    field has been removed.
    """
    url = os.environ.get("PADDLE_SERVICE_URL")
    if not url:
        return None
    # Validate eagerly so misconfiguration fails fast.
    return validate_outbound_url(url, allow_loopback=True)


async def _save_upload(upload: UploadFile, declared_mime: str) -> tuple[Path, int]:
    """Stream an uploaded file to the upload root with validation.

    Returns ``(absolute_path, size_in_bytes)``.
    """
    safe_name = sanitize_filename(upload.filename)
    destination = Path(ensure_upload_root()) / f"{uuid.uuid4().hex}_{safe_name}"
    written = await write_upload_with_limit(
        upload,
        destination,
        declared_mime=declared_mime,
    )
    return destination, written


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/analyze", dependencies=[Depends(require_api_key)])
async def analyze_medical_doc(
    file: UploadFile = File(...),
    provider: str = Form("Ollama"),
    model: str = Form("glm-4.7-flash"),
    api_key: str | None = Form(None),
    structuring_provider: str | None = Form(None),
    structuring_model: str | None = Form(None),
    structuring_api_key: str | None = Form(None),
    reasoning_provider: str | None = Form(None),
    reasoning_model: str | None = Form(None),
    reasoning_api_key: str | None = Form(None),
    ocr_backend: str = Form("ollama"),
    ocr_model: str | None = Form(None),
    ocr_mode: str = Form("text"),
    use_gpu: bool = Form(True),
    agentic_mode: bool = Form(False),
    extraction_graph_mode: bool = Form(False),
):
    correlation_id = _correlation_id()

    provider = _resolve_form_value(provider, "Ollama")
    model = _resolve_form_value(model, "glm-4.7-flash")
    api_key = _resolve_form_value(api_key)
    structuring_provider = _resolve_form_value(structuring_provider, provider)
    structuring_model = _resolve_form_value(structuring_model, model)
    structuring_api_key = _resolve_form_value(structuring_api_key, api_key)
    reasoning_provider = _resolve_form_value(reasoning_provider, provider)
    reasoning_model = _resolve_form_value(reasoning_model, model)
    reasoning_api_key = _resolve_form_value(reasoning_api_key, api_key)
    ocr_backend = _resolve_form_value(ocr_backend, "ollama")
    ocr_model = _resolve_form_value(ocr_model)
    ocr_mode = _resolve_form_value(ocr_mode, "text")
    use_gpu = _coerce_bool(_resolve_form_value(use_gpu, True))
    paddle_service_url = _resolve_paddle_service_url()
    agentic_mode = _coerce_bool(_resolve_form_value(agentic_mode, False))
    extraction_graph_mode = _coerce_bool(_resolve_form_value(extraction_graph_mode, False))

    structuring_api_key = _api_key_for(structuring_provider, structuring_api_key)
    reasoning_api_key = _api_key_for(reasoning_provider, reasoning_api_key)

    validate_upload_or_raise(file)
    try:
        file_path, _ = await _save_upload(file, file.content_type or "application/octet-stream")
    except HTTPException:
        raise
    except Exception as exc:
        _fail(correlation_id, f"upload_failed: {exc}")

    absolute_path = str(file_path)

    if extraction_graph_mode:
        graph_result = await run_in_threadpool(
            run_extraction_graph,
            file_path=absolute_path,
            structuring_provider=structuring_provider,
            structuring_model=structuring_model,
            structuring_api_key=structuring_api_key,
            reasoning_provider=reasoning_provider,
            reasoning_model=reasoning_model,
            reasoning_api_key=reasoning_api_key,
            ocr_backend=ocr_backend,
            ocr_model=ocr_model,
            ocr_prompt_mode=ocr_mode,
            use_gpu=use_gpu,
            paddle_service_url=paddle_service_url,
        )
        if graph_result.get("error"):
            _fail(correlation_id, graph_result["error"])

        current_data = graph_result.get("structured_data") or {}
        ocr_payload = graph_result.get("ocr") or {}
        past_data = graph_result.get("past_data")
        analysis = graph_result.get("analysis") or {"summary": "Analysis failed", "alerts": [], "trends": []}
        requires_human_review = graph_result.get("requires_human_review", False)
        vector_index_status = graph_result.get("vector_index_status")
    elif agentic_mode:
        workflow_result = await run_in_threadpool(
            run_agentic_extraction_workflow,
            file_path=absolute_path,
            structuring_provider=structuring_provider,
            structuring_model=structuring_model,
            structuring_api_key=structuring_api_key,
            reasoning_provider=reasoning_provider,
            reasoning_model=reasoning_model,
            reasoning_api_key=reasoning_api_key,
            ocr_backend=ocr_backend,
            ocr_model=ocr_model,
            ocr_prompt_mode=ocr_mode,
            use_gpu=use_gpu,
            paddle_service_url=paddle_service_url,
        )
        if workflow_result.get("error"):
            _fail(correlation_id, workflow_result["error"])

        current_data = workflow_result.get("structured_data") or {}
        ocr_payload = workflow_result.get("ocr") or {}
        past_data = workflow_result.get("past_data")
        analysis = workflow_result.get("analysis") or {"summary": "Analysis failed", "alerts": [], "trends": []}
        requires_human_review = workflow_result.get("requires_human_review", False)
        vector_index_status = workflow_result.get("vector_index_status")
    else:
        current_result = await run_in_threadpool(
            _call_document_pipeline,
            absolute_path,
            structuring_provider,
            structuring_model,
            structuring_api_key,
            ocr_backend=ocr_backend,
            ocr_model=ocr_model,
            ocr_prompt_mode=ocr_mode,
            use_gpu=use_gpu,
            paddle_service_url=paddle_service_url,
            return_details=True,
            structuring_provider=structuring_provider,
            structuring_model=structuring_model,
            structuring_api_key=structuring_api_key,
        )
        if "error" in current_result:
            _fail(correlation_id, current_result["error"])

        if "structured_data" in current_result:
            current_data = current_result["structured_data"]
            ocr_payload = current_result.get("ocr") or {}
        else:
            current_data = current_result
            ocr_payload = {}

        mrn = current_data.get("patient", {}).get("mrn")
        past_data = await run_in_threadpool(_load_history, mrn) if mrn else None
        retrieved_context = await run_in_threadpool(_retrieve_patient_context, current_data, ocr_payload)
        analysis = await run_in_threadpool(
            analyze_medical_logic,
            current_data,
            past_data,
            reasoning_provider,
            reasoning_model,
            reasoning_api_key,
            retrieved_context=retrieved_context,
        )
        requires_human_review = False
        vector_index_status = await run_in_threadpool(
            _index_for_retrieval,
            absolute_path,
            current_data,
            ocr_payload,
        )

    if not current_data:
        _fail(correlation_id, "extraction produced no structured data")

    ocr_backend_normalized = (ocr_backend or "").lower().replace("-", "").replace("_", "")
    lineage = await run_in_threadpool(
        capture_lineage,
        ocr_backend=ocr_backend,
        ocr_model=ocr_model,
        structuring_provider=structuring_provider,
        structuring_model=structuring_model,
        reasoning_provider=reasoning_provider,
        reasoning_model=reasoning_model,
    )
    response = {
        "extracted": current_data,
        "analysis": analysis,
        "history_available": bool(past_data),
        "file_path": absolute_path,
        "file_url": ocr_payload.get("artifact_manifest", {}).get("original_file_url"),
        "ocr": ocr_payload,
        "bounding_boxes": ocr_payload.get("bounding_boxes", []),
        "annotated_pdf_path": ocr_payload.get("artifact_manifest", {}).get("annotated_pdf_path"),
        "annotated_pdf_url": ocr_payload.get("artifact_manifest", {}).get("annotated_pdf_url"),
        "annotated_image_paths": ocr_payload.get("artifact_manifest", {}).get("annotated_image_paths", []),
        "annotated_image_urls": ocr_payload.get("artifact_manifest", {}).get("annotated_image_urls", []),
        "page_image_urls": ocr_payload.get("artifact_manifest", {}).get("page_image_urls", []),
        "requires_human_review": requires_human_review,
        "vector_index_status": vector_index_status,
        "retrieval_enabled": vector_index_status is not None and vector_index_status.get("indexed", False),
        "ocr_supports_bboxes": ocr_backend_normalized in ("paddle", "paddleocr", "paddleocrvl", "paddleocrvl15"),
        "correlation_id": correlation_id,
        "lineage": lineage,
    }

    mrn = current_data.get("patient", {}).get("mrn") if isinstance(current_data, dict) else None
    await run_in_threadpool(
        record_audit_event,
        "analyze_complete",
        mrn_hash=hash_identifier(mrn),
        correlation_id=correlation_id,
        payload={"requires_human_review": bool(requires_human_review)},
    )
    return response


@app.post("/check_insurance", dependencies=[Depends(require_api_key)])
async def check_insurance(
    policy_file: UploadFile = File(...),
    medical_json: str = Form(...),
    provider: str = Form("Ollama"),
    model: str = Form("glm-4.7-flash"),
    api_key: str | None = Form(None),
    reasoning_provider: str | None = Form(None),
    reasoning_model: str | None = Form(None),
    reasoning_api_key: str | None = Form(None),
    ocr_backend: str = Form("ollama"),
    ocr_model: str | None = Form(None),
    ocr_mode: str = Form("text"),
    use_gpu: bool = Form(True),
    policy_ocr: bool = Form(False),
):
    correlation_id = _correlation_id()

    provider = _resolve_form_value(provider, "Ollama")
    model = _resolve_form_value(model, "glm-4.7-flash")
    api_key = _resolve_form_value(api_key)
    reasoning_provider = _resolve_form_value(reasoning_provider, provider)
    reasoning_model = _resolve_form_value(reasoning_model, model)
    reasoning_api_key = _resolve_form_value(reasoning_api_key, api_key)
    ocr_backend = _resolve_form_value(ocr_backend, "ollama")
    ocr_model = _resolve_form_value(ocr_model)
    ocr_mode = _resolve_form_value(ocr_mode, "text")
    use_gpu = _coerce_bool(_resolve_form_value(use_gpu, True))
    paddle_service_url = _resolve_paddle_service_url()
    policy_ocr = _coerce_bool(_resolve_form_value(policy_ocr, False))

    reasoning_api_key = _api_key_for(reasoning_provider, reasoning_api_key)

    # Validate the upload (suffix allow-list) and stream it to disk with the
    # same byte-cap + magic-byte checks as /analyze. Plain-text policies are
    # also allowed via the explicit ".txt" suffix.
    validate_upload_or_raise(
        policy_file,
        allowed_suffixes=frozenset({".pdf", ".png", ".jpg", ".jpeg", ".txt"}),
    )
    try:
        policy_path, _ = await _save_upload(
            policy_file, policy_file.content_type or "application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as exc:
        _fail(correlation_id, f"policy_upload_failed: {exc}")

    suffix = Path(policy_path).suffix.lower()
    policy_text = ""
    if policy_ocr or suffix != ".txt":
        ocr_payload = await run_in_threadpool(
            run_document_ocr,
            str(policy_path),
            ocr_backend=ocr_backend,
            ocr_model=ocr_model,
            ocr_prompt_mode=ocr_mode,
            use_gpu=use_gpu,
            paddle_service_url=paddle_service_url,
        )
        if "error" in ocr_payload:
            _fail(correlation_id, ocr_payload["error"])
        policy_text = ocr_payload.get("markdown") or ocr_payload.get("raw_text") or ""
    else:
        # Plaintext policy — read from disk with a hard size guard already
        # enforced by _save_upload.
        async with aiofiles.open(policy_path, "rb") as fh:
            raw_bytes = await fh.read()
        try:
            policy_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            _fail(
                correlation_id,
                "Policy file is not valid UTF-8 text. Pass policy_ocr=true for PDFs/images.",
                status.HTTP_400_BAD_REQUEST,
            )

    if not policy_text.strip():
        _fail(correlation_id, "Policy text is empty after extraction.", 422)

    try:
        medical_raw = json.loads(medical_json)
    except json.JSONDecodeError:
        _fail(correlation_id, "invalid medical_json payload", status.HTTP_400_BAD_REQUEST)

    try:
        medical_data = MedicalRecord.model_validate(medical_raw).model_dump(exclude_none=True)
    except Exception:
        _fail(correlation_id, "medical_json failed schema validation", status.HTTP_400_BAD_REQUEST)

    policy_chunks = await run_in_threadpool(
        _retrieve_policy_context,
        medical_data,
        policy_text,
        policy_file.filename or "policy",
    )
    result = await run_in_threadpool(
        _call_insurance_check,
        medical_data,
        policy_text,
        reasoning_provider,
        reasoning_model,
        reasoning_api_key,
        relevant_policy_chunks=policy_chunks,
    )
    mrn = medical_data.get("patient", {}).get("mrn") if isinstance(medical_data, dict) else None
    await run_in_threadpool(
        record_audit_event,
        "insurance_check",
        mrn_hash=hash_identifier(mrn),
        correlation_id=correlation_id,
        payload={"policy_ocr": bool(policy_ocr)},
    )
    return result


@app.post("/confirm", dependencies=[Depends(require_api_key)])
async def confirm_record(data: dict):
    correlation_id = _correlation_id()
    if not isinstance(data, dict):
        # Starlette renamed HTTP_422_UNPROCESSABLE_ENTITY to _CONTENT; use the
        # numeric literal to stay compatible across versions.
        raise HTTPException(status_code=422, detail="Payload must be a JSON object")

    try:
        validated = MedicalRecordStrict.model_validate(data).model_dump(exclude_none=True)
    except Exception:
        # Fall back to the lenient schema — we still reject if even that fails.
        try:
            validated = MedicalRecord.model_validate(data).model_dump(exclude_none=True)
        except Exception:
            _fail(correlation_id, "record failed schema validation", 422)

    try:
        lineage = await run_in_threadpool(
            capture_lineage,
            extra=data.get("_lineage") if isinstance(data.get("_lineage"), dict) else None,
        )
        await run_in_threadpool(save_record, validated, lineage)
    except ValueError as exc:
        _fail(correlation_id, str(exc), status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

    await run_in_threadpool(
        record_audit_event,
        "record_confirmed",
        mrn_hash=hash_identifier(validated.get("patient", {}).get("mrn")),
        correlation_id=correlation_id,
        payload={"git_sha": lineage.get("git_sha")},
    )
    return {"status": "saved", "correlation_id": correlation_id}


@app.get("/artifacts/{artifact_path:path}", dependencies=[Depends(require_api_key)])
async def get_artifact(artifact_path: str):
    """Authenticated artifact serving with path canonicalisation."""
    upload_root = Path(ensure_upload_root())
    resolved = resolve_artifact_path(artifact_path, upload_root)
    return FileResponse(resolved)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness probe: the process is up. Never exercises dependencies."""
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    """Readiness probe: exercise critical dependencies (DB + vector store).

    Returns 200 only when the process can actually serve requests. Use this
    in k8s ``readinessProbe`` / Docker ``HEALTHCHECK`` instead of ``/health``.
    """
    checks: dict[str, Any] = {}
    overall_ok = True

    # 1. SQLite — cheap round-trip through the connection pool.
    try:
        from backend.database import _get_connection

        def _ping_db() -> None:
            _get_connection().execute("SELECT 1").fetchone()

        await run_in_threadpool(_ping_db)
        checks["database"] = "ok"
    except Exception as exc:
        overall_ok = False
        checks["database"] = f"error: {exc.__class__.__name__}"

    # 2. Vector store factory — surface Qdrant misconfiguration here rather
    #    than at first /analyze call. Unconfigured stores are considered OK
    #    because retrieval is optional.
    try:
        store = await run_in_threadpool(create_vector_store)
        checks["vector_store"] = "ok" if store is not None else "not_configured"
    except Exception as exc:
        overall_ok = False
        checks["vector_store"] = f"error: {exc.__class__.__name__}"

    status_code = 200 if overall_ok else 503
    body = {"status": "ready" if overall_ok else "degraded", "checks": checks}
    if not overall_ok:
        raise HTTPException(status_code=status_code, detail=body)
    return body


@app.get("/review/pending", dependencies=[Depends(require_api_key)])
async def list_review_queue(limit: int = 50) -> dict[str, Any]:
    """List pending human-review tasks created by the extraction workflow."""
    from backend.database import list_pending_reviews

    tasks = await run_in_threadpool(list_pending_reviews, limit)
    return {"tasks": tasks, "count": len(tasks)}


@app.post("/review/{task_id}/approve", dependencies=[Depends(require_api_key)])
async def approve_review(task_id: int, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    from backend.database import resolve_review_task

    reviewer = (payload or {}).get("reviewer") if isinstance(payload, dict) else None
    notes = (payload or {}).get("notes") if isinstance(payload, dict) else None
    updated = await run_in_threadpool(
        resolve_review_task, task_id, approve=True, reviewer=reviewer, notes=notes
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Review task not found or already resolved")
    return {"status": "approved", "task_id": task_id}


@app.post("/review/{task_id}/reject", dependencies=[Depends(require_api_key)])
async def reject_review(task_id: int, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    from backend.database import resolve_review_task

    reviewer = (payload or {}).get("reviewer") if isinstance(payload, dict) else None
    notes = (payload or {}).get("notes") if isinstance(payload, dict) else None
    updated = await run_in_threadpool(
        resolve_review_task, task_id, approve=False, reviewer=reviewer, notes=notes
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Review task not found or already resolved")
    return {"status": "rejected", "task_id": task_id}


# ---------------------------------------------------------------------------
# Internal helpers (kept as module-level callables for monkeypatching in tests)
# ---------------------------------------------------------------------------


def _call_document_pipeline(file_path, provider, model, api_key, **kwargs):
    return process_document_pipeline(file_path, provider, model, api_key, **kwargs)


def _call_insurance_check(medical_data, policy_text, provider, model, api_key, **kwargs):
    return check_insurance_coverage(medical_data, policy_text, provider, model, api_key, **kwargs)


def _load_history(mrn: str | None):
    from backend.database import get_patient_history

    return get_patient_history(mrn) if mrn else None


def _index_for_retrieval(file_path: str, current_data: dict, ocr_payload: dict):
    store = create_vector_store()
    if store is None:
        logger.warning("retrieval_disabled", reason="no_vector_store")
        return {"indexed": False, "reason": "No vector store is configured"}

    patient_hash = hash_identifier(current_data.get("patient", {}).get("mrn"))
    metadata = {
        "patient_id_hash": patient_hash,
        "encounter_date": current_data.get("encounter", {}).get("date"),
        "source_type": "medical_record",
        "ocr_backend": ocr_payload.get("backend"),
    }
    try:
        chunks = build_chunks_from_ocr_payload(file_path, ocr_payload, metadata)
        if not chunks:
            chunks = build_chunks_from_text(
                file_path,
                json.dumps(current_data),
                metadata,
                section_type="structured_json",
            )
        return store.upsert_chunks(chunks)
    except Exception as exc:
        logger.warning("indexing_failed", reason=str(exc))
        return {"indexed": False, "reason": "Indexing failed"}


def _retrieve_patient_context(current_data: dict, ocr_payload: dict) -> list[dict]:
    store = create_vector_store()
    if store is None:
        return []

    patient_id_hash = hash_identifier(current_data.get("patient", {}).get("mrn"))
    if not patient_id_hash:
        return []

    diagnosis_list = current_data.get("clinical", {}).get("diagnosis_list", [])
    medications = current_data.get("clinical", {}).get("medications", [])
    medication_names = [item.get("name") for item in medications if isinstance(item, dict) and item.get("name")]
    query = ", ".join(diagnosis_list + medication_names)
    if not query:
        query = ocr_payload.get("raw_text") or ocr_payload.get("markdown") or json.dumps(current_data)

    try:
        results = store.search(
            query,
            limit=5,
            filters={
                "patient_id_hash": patient_id_hash,
                "source_type": "medical_record",
            },
        )
    except Exception:
        return []

    return [
        {**hit, "text": sanitize_retrieved_text(str(hit.get("text", "")))}
        if isinstance(hit, dict)
        else hit
        for hit in results
    ]


def _retrieve_policy_context(medical_data: dict, policy_text: str, policy_document_id: str) -> list[dict]:
    store = create_vector_store()
    if store is None:
        return []

    metadata = {
        "source_type": "insurance_policy",
        "policy_document_id": policy_document_id,
    }
    try:
        chunks = build_chunks_from_text(policy_document_id, policy_text, metadata, section_type="policy_text")
        if chunks:
            store.upsert_chunks(chunks)
        diagnoses = medical_data.get("clinical", {}).get("diagnosis_list", [])
        query = ", ".join(diagnoses) or json.dumps(medical_data)
        results = store.search(
            query,
            limit=5,
            filters={
                "source_type": "insurance_policy",
                "policy_document_id": policy_document_id,
            },
        )
    except Exception:
        return []

    return [
        {**hit, "text": sanitize_retrieved_text(str(hit.get("text", "")))}
        if isinstance(hit, dict)
        else hit
        for hit in results
    ]


# Run uvicorn explicitly with a loopback bind by default; callers override via env.
if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("MEDISCAN_HOST", "127.0.0.1")
    port = int(os.environ.get("MEDISCAN_PORT", "8000"))
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
