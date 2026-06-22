"""Granular extraction LangGraph.

The compiled graph is cached at module scope and reused across requests — the
previous implementation rebuilt the ``StateGraph`` on every call. Other
improvements:

* The confidence gate now also consumes OCR-reported per-page confidence
  (when available) rather than relying solely on heuristic deductions.
* ``_deep_copy_dict`` is replaced with :func:`copy.deepcopy` to avoid the
  shallow-list bug that could leak mutations back into shared state.
* ``_retrieve_context_node`` sanitises retrieved chunk text to defuse stored
  prompt-injection attacks before the reasoning LLM sees them.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Literal, TypedDict

from backend.ai_wrapper import clean_json_output, get_ai_response, parse_model_json
from backend.database import get_patient_history
from backend.errors import AIProviderError, MediscanError
from backend.extract import run_document_ocr
from backend.logging_config import get_logger
from backend.logic import analyze_medical_logic
from backend.models import MedicalRecord
from backend.retrieval import build_chunks_from_ocr_payload, create_vector_store, hash_identifier
from backend.retrieval.chunking import sanitize_retrieved_text
from backend.security import firewall_clause, generate_boundary_nonce, wrap_untrusted

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional during dev
    END = None
    START = None
    StateGraph = None


_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Typed state
# ---------------------------------------------------------------------------


class ExtractionGraphState(TypedDict, total=False):
    file_path: str
    structuring_provider: str
    structuring_model: str
    structuring_api_key: str | None
    reasoning_provider: str
    reasoning_model: str
    reasoning_api_key: str | None
    ocr_backend: str
    ocr_model: str | None
    ocr_prompt_mode: str
    use_gpu: bool
    paddle_service_url: str | None

    document_type: str | None
    page_count: int
    page_image_paths: list[str]
    ocr: dict[str, Any]
    candidate_fields: dict[str, Any]
    validation_errors: list[str]
    normalized_fields: dict[str, Any]

    past_data: dict[str, Any] | None
    retrieved_context: list[dict[str, Any]]

    structured_data: dict[str, Any]
    confidence_score: float
    requires_human_review: bool

    analysis: dict[str, Any]
    vector_index_status: dict[str, Any] | None
    persisted: bool
    error: str | None


# ---------------------------------------------------------------------------
# Graph builder (compiled lazily at import time)
# ---------------------------------------------------------------------------


_compiled_graph = None


def build_extraction_graph():
    """Build and compile the granular extraction LangGraph."""
    if StateGraph is None or START is None or END is None:
        raise RuntimeError("LangGraph is not installed. Install langgraph to use this workflow.")

    graph = StateGraph(ExtractionGraphState)

    graph.add_node("ingest_document", _ingest_document_node)
    graph.add_node("classify_document_type", _classify_document_type_node)
    graph.add_node("split_pages", _split_pages_node)
    graph.add_node("ocr_per_page", _ocr_per_page_node)
    graph.add_node("extract_candidate_fields", _extract_candidate_fields_node)
    graph.add_node("validate_against_schema", _validate_against_schema_node)
    graph.add_node("normalize_codes", _normalize_codes_node)
    graph.add_node("retrieve_context", _retrieve_context_node)
    graph.add_node("merge_document_record", _merge_document_record_node)
    graph.add_node("confidence_gate", _confidence_gate_node)
    graph.add_node("human_review", _human_review_node)
    graph.add_node("persist_record", _persist_record_node)

    graph.add_edge(START, "ingest_document")
    graph.add_edge("ingest_document", "classify_document_type")
    graph.add_edge("classify_document_type", "split_pages")
    graph.add_edge("split_pages", "ocr_per_page")
    graph.add_edge("ocr_per_page", "extract_candidate_fields")
    graph.add_edge("extract_candidate_fields", "validate_against_schema")
    graph.add_edge("validate_against_schema", "normalize_codes")
    graph.add_edge("normalize_codes", "retrieve_context")
    graph.add_edge("retrieve_context", "merge_document_record")
    graph.add_edge("merge_document_record", "confidence_gate")

    graph.add_conditional_edges(
        "confidence_gate",
        _route_after_confidence_gate,
        {
            "human_review": "human_review",
            "persist_record": "persist_record",
        },
    )
    graph.add_edge("human_review", "persist_record")
    graph.add_edge("persist_record", END)

    return graph.compile()


def _get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_extraction_graph()
    return _compiled_graph


def run_extraction_graph(
    *,
    file_path: str,
    structuring_provider: str,
    structuring_model: str,
    structuring_api_key: str | None,
    reasoning_provider: str,
    reasoning_model: str,
    reasoning_api_key: str | None,
    ocr_backend: str,
    ocr_model: str | None,
    ocr_prompt_mode: str,
    use_gpu: bool,
    paddle_service_url: str | None,
) -> ExtractionGraphState:
    return _get_compiled_graph().invoke(
        {
            "file_path": file_path,
            "structuring_provider": structuring_provider,
            "structuring_model": structuring_model,
            "structuring_api_key": structuring_api_key,
            "reasoning_provider": reasoning_provider,
            "reasoning_model": reasoning_model,
            "reasoning_api_key": reasoning_api_key,
            "ocr_backend": ocr_backend,
            "ocr_model": ocr_model,
            "ocr_prompt_mode": ocr_prompt_mode,
            "use_gpu": use_gpu,
            "paddle_service_url": paddle_service_url,
        }
    )


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------


def _route_after_confidence_gate(state: ExtractionGraphState) -> Literal["human_review", "persist_record"]:
    if state.get("requires_human_review"):
        return "human_review"
    return "persist_record"


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------


def _ingest_document_node(state: ExtractionGraphState) -> ExtractionGraphState:
    file_path = state.get("file_path", "")
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}
    if not path.is_file():
        return {"error": f"Path is not a file: {file_path}"}
    return {"error": None}


def _classify_document_type_node(state: ExtractionGraphState) -> ExtractionGraphState:
    if state.get("error"):
        return {"document_type": None}

    path = Path(state.get("file_path", ""))
    filename_lower = path.stem.lower()

    known_types = {
        "insurance": "insurance_policy",
        "policy": "insurance_policy",
        "lab": "lab_report",
        "labs": "lab_report",
        "blood": "lab_report",
        "ecg": "ecg_report",
        "ekg": "ecg_report",
        "discharge": "discharge_summary",
        "summary": "discharge_summary",
        "prescription": "prescription",
        "rx": "prescription",
        "radiology": "radiology_report",
        "xray": "radiology_report",
        "mri": "radiology_report",
        "ct": "radiology_report",
    }
    document_type = "medical_record"
    for keyword, classified_type in known_types.items():
        if keyword in filename_lower:
            document_type = classified_type
            break
    return {"document_type": document_type}


def _split_pages_node(state: ExtractionGraphState) -> ExtractionGraphState:
    if state.get("error"):
        return {"page_count": 0, "page_image_paths": []}

    file_path = state.get("file_path", "")
    suffix = Path(file_path).suffix.lower()

    page_count = 0
    if suffix == ".pdf":
        try:
            from pdf2image import pdfinfo_from_path

            info = pdfinfo_from_path(file_path)
            page_count = info.get("Pages", 0)
        except Exception:
            pass
    elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"):
        page_count = 1

    return {"page_count": page_count, "page_image_paths": []}


def _ocr_per_page_node(state: ExtractionGraphState) -> ExtractionGraphState:
    if state.get("error"):
        return {"ocr": {}}

    try:
        ocr_payload = run_document_ocr(
            state["file_path"],
            ocr_backend=state.get("ocr_backend", "glm"),
            ocr_model=state.get("ocr_model"),
            ocr_prompt_mode=state.get("ocr_prompt_mode", "text"),
            use_gpu=state.get("use_gpu", True),
            paddle_service_url=state.get("paddle_service_url"),
        )
    except MediscanError as exc:
        _logger.warning("ocr_failed", reason=str(exc))
        return {"error": str(exc), "ocr": {}}

    per_page = ocr_payload.get("per_page_results", [])
    page_images = ocr_payload.get("page_images", [])
    return {
        "ocr": ocr_payload,
        "page_count": len(per_page),
        "page_image_paths": page_images,
    }


_SCHEMA_DOC = """{
  "patient": {"full_name": "string", "dob": "YYYY-MM-DD", "mrn": "string"},
  "encounter": {"date": "YYYY-MM-DD", "provider": "string", "facility": "string"},
  "clinical": {
    "diagnosis_list": ["string"],
    "medications": [{"name": "string", "dosage": "string", "frequency": "string"}],
    "vitals": {"bp": "string", "hr": "string", "temp": "string", "weight": "string"}
  }
}"""


def _extract_candidate_fields_node(state: ExtractionGraphState) -> ExtractionGraphState:
    if state.get("error"):
        return {"candidate_fields": {}}

    ocr_payload = state.get("ocr") or {}
    raw_text = ocr_payload.get("markdown") or ocr_payload.get("raw_text") or ""
    if not raw_text:
        return {"error": "OCR produced no text", "candidate_fields": {}}

    provider = state.get("structuring_provider", "Ollama")
    model = state.get("structuring_model", "glm-4.7-flash")
    api_key = state.get("structuring_api_key")

    nonce = generate_boundary_nonce()
    primary, _ = wrap_untrusted(ocr_payload.get("markdown") or raw_text, nonce)
    sections = [f"OCR_CONTENT:\n{primary}"]
    structured_doc = ocr_payload.get("structured_doc")
    if structured_doc:
        struct_json = json.dumps(structured_doc, ensure_ascii=False)
        if len(struct_json) <= 30_000:
            wrapped_struct, _ = wrap_untrusted(struct_json, nonce)
            sections.append(f"OCR_STRUCTURED_DOC:\n{wrapped_struct}")
    user_text = "\n\n".join(sections)

    system_prompt = (
        "You are a medical data entry specialist. Convert the delimited OCR "
        "content into JSON matching this schema exactly:\n"
        f"{_SCHEMA_DOC}\n\nReturn ONLY the JSON object.\n" + firewall_clause(nonce)
    )

    try:
        response = get_ai_response(provider, model, api_key, system_prompt, user_text, force_json=True)
        # ``clean_json_output`` remains the module-level back-compat shim so
        # older tests/monkeypatches that replace it still take effect.
        candidate_fields = parse_model_json(clean_json_output(response))
        return {"candidate_fields": candidate_fields}
    except AIProviderError as exc:
        return {"error": f"Structuring failed: {exc}", "candidate_fields": {}}
    except (ValueError, json.JSONDecodeError) as exc:
        return {"error": f"Structuring failed to parse JSON: {exc}", "candidate_fields": {}}


def _validate_against_schema_node(state: ExtractionGraphState) -> ExtractionGraphState:
    errors: list[str] = []
    candidate_fields = state.get("candidate_fields") or {}
    if not candidate_fields:
        errors.append("No candidate fields extracted")
    else:
        try:
            MedicalRecord.model_validate(candidate_fields)
        except Exception as exc:
            errors.append(str(exc))
    return {"validation_errors": errors}


def _normalize_codes_node(state: ExtractionGraphState) -> ExtractionGraphState:
    candidate_fields = state.get("candidate_fields") or {}
    if not candidate_fields:
        return {"normalized_fields": candidate_fields}

    normalized = copy.deepcopy(candidate_fields)
    clinical = normalized.get("clinical") or {}
    diagnosis_list = clinical.get("diagnosis_list") or []
    normalized_diagnoses = []
    for diagnosis in diagnosis_list:
        if isinstance(diagnosis, str):
            diagnosis = diagnosis.strip().rstrip(".,;")
            diagnosis = re.sub(r"\s+", " ", diagnosis)
            normalized_diagnoses.append(diagnosis)
    if diagnosis_list:
        clinical["diagnosis_list"] = normalized_diagnoses

    medications = clinical.get("medications") or []
    normalized_meds = []
    for med in medications:
        if isinstance(med, dict):
            name = med.get("name", "")
            if isinstance(name, str) and name.isupper():
                med = {**med, "name": name.title()}
            normalized_meds.append(med)
    if medications:
        clinical["medications"] = normalized_meds

    if clinical:
        normalized["clinical"] = clinical

    return {"normalized_fields": normalized}


def _retrieve_context_node(state: ExtractionGraphState) -> ExtractionGraphState:
    normalized_fields = state.get("normalized_fields") or state.get("candidate_fields") or {}
    mrn = normalized_fields.get("patient", {}).get("mrn")
    past_data = get_patient_history(mrn) if mrn else None

    store = create_vector_store()
    if store is None or not mrn:
        return {"past_data": past_data, "retrieved_context": []}

    patient_hash = hash_identifier(mrn)
    if not patient_hash:
        # No MRN_HMAC_PEPPER configured → retrieval disabled (intentional).
        return {"past_data": past_data, "retrieved_context": []}

    diagnoses = normalized_fields.get("clinical", {}).get("diagnosis_list", [])
    meds = normalized_fields.get("clinical", {}).get("medications", [])
    med_names = [m.get("name") for m in meds if isinstance(m, dict) and m.get("name")]
    query = ", ".join(diagnoses + med_names)
    if not query:
        ocr_payload = state.get("ocr") or {}
        query = ocr_payload.get("raw_text") or ocr_payload.get("markdown") or json.dumps(normalized_fields)

    source_type = state.get("document_type") or "medical_record"
    try:
        hits = store.search(
            query,
            limit=5,
            filters={
                "patient_id_hash": patient_hash,
                "source_type": source_type,
            },
        )
    except Exception as exc:
        _logger.warning("retrieval_failed", reason=str(exc))
        hits = []

    sanitized_hits: list[dict[str, Any]] = []
    for hit in hits:
        if isinstance(hit, dict) and "text" in hit:
            hit = {**hit, "text": sanitize_retrieved_text(str(hit["text"]))}
        sanitized_hits.append(hit)
    return {"past_data": past_data, "retrieved_context": sanitized_hits}


def _merge_document_record_node(state: ExtractionGraphState) -> ExtractionGraphState:
    normalized_fields = state.get("normalized_fields") or state.get("candidate_fields") or {}
    analysis = analyze_medical_logic(
        normalized_fields,
        state.get("past_data"),
        state.get("reasoning_provider", "Ollama"),
        state.get("reasoning_model", "glm-4.7-flash"),
        state.get("reasoning_api_key"),
        retrieved_context=state.get("retrieved_context"),
    )
    return {"structured_data": normalized_fields, "analysis": analysis}


def _confidence_gate_node(state: ExtractionGraphState) -> ExtractionGraphState:
    errors = state.get("validation_errors") or []
    ocr_payload = state.get("ocr") or {}
    structured_data = state.get("structured_data") or {}

    score = 1.0
    score -= min(len(errors) * 0.3, 0.6)
    if not (ocr_payload.get("raw_text") or ocr_payload.get("markdown")):
        score -= 0.2

    # Factor in OCR-reported confidence when present (e.g. PaddleOCR-VL).
    ocr_confidence = ocr_payload.get("confidence")
    if isinstance(ocr_confidence, (int, float)):
        # Low OCR confidence (<0.6) subtracts up to 0.25; high leaves unchanged.
        score -= max(0.0, (0.6 - float(ocr_confidence)) / 0.6 * 0.25)

    for section in ("patient", "encounter", "clinical"):
        if not structured_data.get(section):
            score -= 0.1

    score = max(0.0, round(score, 2))
    requires_human_review = score < 0.6 or bool(errors)

    return {"confidence_score": score, "requires_human_review": requires_human_review}


def _human_review_node(state: ExtractionGraphState) -> ExtractionGraphState:
    """Emit an audit event for the pending review.

    A production deployment should use LangGraph's ``interrupt()`` + a
    durable checkpointer (PostgresSaver) to pause the graph here and resume
    only after a human approves. For now we log the event so it is at least
    visible in observability.
    """
    try:
        from backend.database import enqueue_review_task, record_audit_event

        structured = state.get("structured_data") or {}
        mrn = (structured.get("patient") or {}).get("mrn")
        mrn_hash = hash_identifier(mrn)
        errors = state.get("validation_errors") or []
        doc_type = state.get("document_type")
        confidence = state.get("confidence_score")
        task_id: int | None = None
        try:
            task_id = enqueue_review_task(
                mrn_hash=mrn_hash,
                correlation_id=state.get("correlation_id"),
                confidence_score=confidence,
                validation_errors=errors,
                document_type=doc_type,
                structured_data=structured,
            )
        except Exception:  # pragma: no cover - enqueue is best-effort
            pass
        record_audit_event(
            "human_review_required",
            mrn_hash=mrn_hash,
            payload={
                "confidence_score": confidence,
                "validation_errors": errors,
                "document_type": doc_type,
                "review_task_id": task_id,
            },
        )
    except Exception:  # pragma: no cover - audit must never fail the graph
        pass
    return {}


def _deep_copy_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Back-compat shim used by older tests. Equivalent to :func:`copy.deepcopy`."""
    return copy.deepcopy(d)


def _persist_record_node(state: ExtractionGraphState) -> ExtractionGraphState:
    if state.get("requires_human_review"):
        return {
            "vector_index_status": {"indexed": False, "reason": "Pending human review"},
            "persisted": False,
        }

    file_path = state.get("file_path", "")
    structured_data = state.get("structured_data") or {}
    ocr_payload = state.get("ocr") or {}

    source_type = state.get("document_type") or "medical_record"
    store = create_vector_store()
    vector_index_status: dict[str, Any] | None = None
    if store is not None:
        patient_hash = hash_identifier(structured_data.get("patient", {}).get("mrn"))
        metadata = {
            "patient_id_hash": patient_hash,
            "encounter_date": structured_data.get("encounter", {}).get("date"),
            "source_type": source_type,
            "ocr_backend": ocr_payload.get("backend"),
        }
        try:
            chunks = build_chunks_from_ocr_payload(file_path, ocr_payload, metadata)
            if not chunks:
                from backend.retrieval import build_chunks_from_text

                chunks = build_chunks_from_text(
                    file_path,
                    json.dumps(structured_data),
                    metadata,
                    section_type="structured_json",
                )
            vector_index_status = store.upsert_chunks(chunks)
        except Exception as exc:
            _logger.warning("indexing_failed", reason=str(exc))
            vector_index_status = {"indexed": False, "reason": "Indexing failed"}
    else:
        vector_index_status = {"indexed": False, "reason": "No vector store is configured"}

    return {"vector_index_status": vector_index_status, "persisted": True}
