"""Direct extraction pipeline: OCR -> structuring LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import ollama
from pdf2image import convert_from_path

from backend.ai_wrapper import AIProviderError, get_ai_response, parse_model_json
from backend.artifacts import create_document_workspace, render_document_pages
from backend.logging_config import get_logger
from backend.ocr import OCRBackendError, materialize_annotations, run_ocr
from backend.pii_scrub import scrub_text
from backend.retrieval.chunking import sanitize_retrieved_text
from backend.security import MAX_PDF_PAGES, firewall_clause, generate_boundary_nonce, wrap_untrusted

_logger = get_logger(__name__)


_STRUCTURING_SCHEMA_DOC = """{
  "patient": {"full_name": "string", "dob": "YYYY-MM-DD", "mrn": "string"},
  "encounter": {"date": "YYYY-MM-DD", "provider": "string", "facility": "string"},
  "clinical": {
    "diagnosis_list": ["string"],
    "medications": [{"name": "string", "dosage": "string", "frequency": "string"}],
    "vitals": {"bp": "string", "hr": "string", "temp": "string", "weight": "string"}
  }
}"""


def _build_structuring_system_prompt(nonce: str) -> str:
    return (
        "You are a medical data entry specialist. Convert the document content "
        "delimited below into a JSON object that matches this schema exactly:\n"
        f"{_STRUCTURING_SCHEMA_DOC}\n\n"
        "Return ONLY the JSON object — no markdown, no commentary.\n"
        f"{firewall_clause(nonce)}"
    )


def process_document_pipeline(
    file_path: str,
    provider: str = "Ollama",
    model: str = "glm-4.7-flash",
    api_key: str | None = None,
    *,
    ocr_backend: str = "glm",
    ocr_model: str | None = None,
    ocr_prompt_mode: str = "text",
    use_gpu: bool = True,
    paddle_service_url: str | None = None,
    return_details: bool = False,
    structuring_provider: str | None = None,
    structuring_model: str | None = None,
    structuring_api_key: str | None = None,
):
    """Orchestrate OCR + structuring.

    Either the legacy positional triple ``(provider, model, api_key)`` or the
    explicit ``structuring_*`` kwargs can be used. The structuring_* kwargs
    take precedence when both are supplied.
    """
    resolved_structuring_provider = structuring_provider or provider
    resolved_structuring_model = structuring_model or model
    resolved_structuring_api_key = structuring_api_key if structuring_api_key is not None else api_key

    _logger.info("ocr_scanning", backend=ocr_backend, model=ocr_model)
    ocr_payload = run_document_ocr(
        file_path,
        ocr_backend=ocr_backend,
        ocr_model=ocr_model,
        ocr_prompt_mode=ocr_prompt_mode,
        use_gpu=use_gpu,
        paddle_service_url=paddle_service_url,
    )
    if "error" in ocr_payload:
        return ocr_payload

    raw_text = ocr_payload.get("markdown") or ocr_payload.get("raw_text") or ""
    _logger.info("ocr_complete", raw_text_length=len(raw_text))

    _logger.info(
        "structuring_start",
        provider=resolved_structuring_provider,
        model=resolved_structuring_model,
    )
    try:
        user_text, nonce = _build_structuring_user_input(ocr_payload, provider=resolved_structuring_provider)
        raw_response = get_ai_response(
            resolved_structuring_provider,
            resolved_structuring_model,
            resolved_structuring_api_key,
            _build_structuring_system_prompt(nonce),
            user_text,
            force_json=True,
        )

        structured_data = parse_model_json(raw_response)
        if return_details:
            return {"structured_data": structured_data, "ocr": ocr_payload}
        return structured_data
    except AIProviderError as exc:
        return {"error": f"Structuring failed: {exc.detail}"}
    except Exception as exc:
        _logger.warning("structuring_failed", reason=str(exc))
        return {"error": "Structuring failed"}


def run_document_ocr(
    file_path: str,
    ocr_backend: str = "glm",
    ocr_model: str | None = None,
    ocr_prompt_mode: str = "text",
    use_gpu: bool = True,
    paddle_service_url: str | None = None,
):
    # Reject pathological PDFs before pdf2image can pin a worker.
    # The granular graph runs the same check inside _split_pages_node.
    if Path(file_path).suffix.lower() == ".pdf":
        try:
            from pdf2image import pdfinfo_from_path

            info = pdfinfo_from_path(file_path)
            page_count = int(info.get("Pages", 0) or 0)
            if page_count > MAX_PDF_PAGES:
                _logger.warning(
                    "pdf_rejected_too_many_pages",
                    pages=page_count,
                    limit=MAX_PDF_PAGES,
                )
                return {
                    "error": (
                        f"PDF has {page_count} pages, which exceeds the "
                        f"{MAX_PDF_PAGES}-page cap (MEDISCAN_MAX_PDF_PAGES)."
                    )
                }
        except Exception as exc:  # pragma: no cover - best-effort guard
            # If pdfinfo isn't available we fall through to pdf2image's own
            # error path; the per-page render still gets a chance to fail
            # gracefully below.
            _logger.info("pdfinfo_unavailable", reason=str(exc))

    try:
        workspace = create_document_workspace(file_path)
        page_image_paths = render_document_pages(file_path, workspace, converter=convert_from_path)
    except Exception as exc:
        _logger.warning("pdf_conversion_failed", reason=str(exc))
        return {"error": f"PDF Conversion failed. Is Poppler installed? Error: {exc}"}

    if not page_image_paths:
        return {"error": "Could not process file format."}

    normalized_ocr_backend = (ocr_backend or "glm").strip().lower()
    try:
        ocr_result = run_ocr(
            document_path=file_path,
            page_image_paths=page_image_paths,
            backend=ocr_backend,
            model=ocr_model,
            ocr_mode=ocr_prompt_mode,
            use_gpu=use_gpu,
            paddle_service_url=paddle_service_url,
            artifact_root=workspace.raw_ocr_dir,
            ollama_client=ollama,
        )
        ocr_result = materialize_annotations(file_path, ocr_result, page_image_paths)
        return ocr_result.model_dump()
    except OCRBackendError as exc:
        if normalized_ocr_backend in {"glm", "glm-ocr", "ollama", "ollama-glm"}:
            return {"error": f"GLM-OCR failed: {exc}"}
        return {"error": f"OCR failed: {exc}"}
    except Exception as exc:
        _logger.warning("ocr_failed", reason=str(exc), backend=normalized_ocr_backend)
        if normalized_ocr_backend in {"glm", "glm-ocr", "ollama", "ollama-glm"}:
            return {"error": f"GLM-OCR failed: {exc}"}
        return {"error": f"OCR failed: {exc}"}


def _build_structuring_user_input(ocr_payload: dict[str, Any], *, provider: str | None = None) -> tuple[str, str]:
    """Compose the user turn with prompt-injection-resistant delimiters.

    Returns ``(user_text, nonce)`` — the nonce MUST be passed to the matching
    :func:`_build_structuring_system_prompt` so the model can verify delimiters.
    """
    nonce = generate_boundary_nonce()
    sections: list[str] = []

    raw_text = ocr_payload.get("raw_text") or ""
    markdown = ocr_payload.get("markdown") or ""
    structured_doc = ocr_payload.get("structured_doc")

    # Prefer markdown over raw_text so we only send one representation when
    # they duplicate each other.
    primary = markdown or raw_text
    if primary:
        # Defense-in-depth: strip injection phrases embedded in OCR text even
        # though the untrusted boundary nonce already provides primary
        # isolation. Same scrub we apply to retrieved RAG chunks.
        primary = sanitize_retrieved_text(primary)
        primary = scrub_text(primary, provider=provider)
        wrapped, _ = wrap_untrusted(primary, nonce)
        sections.append(f"OCR_CONTENT:\n{wrapped}")

    # Only include structured_doc (PaddleOCR layout JSON) when present and
    # distinct, because it can balloon the prompt and confuse the model.
    if structured_doc:
        payload_text = json.dumps(structured_doc, ensure_ascii=False)
        if len(payload_text) <= 30_000:
            wrapped_struct, _ = wrap_untrusted(payload_text, nonce)
            sections.append(f"OCR_STRUCTURED_DOC:\n{wrapped_struct}")

    return "\n\n".join(sections), nonce
