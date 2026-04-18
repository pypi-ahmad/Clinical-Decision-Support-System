"""First-generation agentic extraction workflow.

This graph is the coarser legacy sibling of
``backend.workflows.extraction_graph``. It remains available for regression
purposes but the granular graph should be preferred. The compiled graph is
cached in module scope.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from backend.database import get_patient_history
from backend.extract import process_document_pipeline
from backend.logging_config import get_logger
from backend.logic import analyze_medical_logic
from backend.models import MedicalRecord
from backend.retrieval import build_chunks_from_ocr_payload, create_vector_store, hash_identifier
from backend.retrieval.chunking import sanitize_retrieved_text

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional during dev
    END = None
    START = None
    StateGraph = None


_logger = get_logger(__name__)
_compiled_graph = None


class ExtractionState(TypedDict, total=False):
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
    structured_data: dict[str, Any]
    ocr: dict[str, Any]
    past_data: dict[str, Any] | None
    analysis: dict[str, Any]
    retrieved_context: list[dict[str, Any]]
    validation_errors: list[str]
    requires_human_review: bool
    vector_index_status: dict[str, Any] | None
    error: str | None


def build_agentic_extraction_graph():
    if StateGraph is None or START is None or END is None:
        raise RuntimeError("LangGraph is not installed")

    graph = StateGraph(ExtractionState)
    graph.add_node("extract_document", _extract_document_node)
    graph.add_node("validate_structured_data", _validate_structured_data_node)
    graph.add_node("human_review_gate", _human_review_gate_node)
    graph.add_node("load_history", _load_history_node)
    graph.add_node("retrieve_context", _retrieve_context_node)
    graph.add_node("analyze_document", _analyze_document_node)
    graph.add_node("index_document", _index_document_node)

    graph.add_edge(START, "extract_document")
    graph.add_edge("extract_document", "validate_structured_data")
    graph.add_edge("validate_structured_data", "human_review_gate")
    graph.add_edge("human_review_gate", "load_history")
    graph.add_edge("load_history", "retrieve_context")
    graph.add_edge("retrieve_context", "analyze_document")
    graph.add_edge("analyze_document", "index_document")
    graph.add_edge("index_document", END)
    return graph.compile()


def _get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_agentic_extraction_graph()
    return _compiled_graph


def run_agentic_extraction_workflow(
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
) -> ExtractionState:
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


def _extract_document_node(state: ExtractionState) -> ExtractionState:
    result = process_document_pipeline(
        state["file_path"],
        state.get("structuring_provider", "Ollama"),
        state.get("structuring_model", "glm-4.7-flash"),
        state.get("structuring_api_key"),
        ocr_backend=state.get("ocr_backend", "ollama"),
        ocr_model=state.get("ocr_model"),
        ocr_prompt_mode=state.get("ocr_prompt_mode", "text"),
        use_gpu=state.get("use_gpu", True),
        paddle_service_url=state.get("paddle_service_url"),
        return_details=True,
        structuring_provider=state.get("structuring_provider"),
        structuring_model=state.get("structuring_model"),
        structuring_api_key=state.get("structuring_api_key"),
    )
    if "error" in result:
        return {"error": result["error"]}

    return {
        "structured_data": result["structured_data"],
        "ocr": result.get("ocr", {}),
        "error": None,
    }


def _validate_structured_data_node(state: ExtractionState) -> ExtractionState:
    errors: list[str] = []
    structured_data = state.get("structured_data") or {}
    try:
        MedicalRecord.model_validate(structured_data)
    except Exception as exc:
        errors.append(str(exc))
    return {"validation_errors": errors}


def _human_review_gate_node(state: ExtractionState) -> ExtractionState:
    return {"requires_human_review": bool(state.get("validation_errors"))}


def _load_history_node(state: ExtractionState) -> ExtractionState:
    structured_data = state.get("structured_data") or {}
    mrn = structured_data.get("patient", {}).get("mrn")
    history = get_patient_history(mrn) if mrn else None
    return {"past_data": history}


def _retrieve_context_node(state: ExtractionState) -> ExtractionState:
    structured_data = state.get("structured_data") or {}
    ocr_payload = state.get("ocr") or {}
    store = create_vector_store()
    if store is None:
        return {"retrieved_context": []}

    patient_hash = hash_identifier(structured_data.get("patient", {}).get("mrn"))
    if not patient_hash:
        return {"retrieved_context": []}

    diagnosis_text = ", ".join(structured_data.get("clinical", {}).get("diagnosis_list", []))
    if not diagnosis_text:
        diagnosis_text = ocr_payload.get("raw_text") or ocr_payload.get("markdown") or json.dumps(structured_data)

    try:
        hits = store.search(
            diagnosis_text,
            limit=5,
            filters={
                "patient_id_hash": patient_hash,
                "source_type": "medical_record",
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
    return {"retrieved_context": sanitized_hits}


def _analyze_document_node(state: ExtractionState) -> ExtractionState:
    if state.get("error"):
        return {"analysis": {"summary": "Analysis skipped", "alerts": [], "trends": []}}

    analysis = analyze_medical_logic(
        state.get("structured_data") or {},
        state.get("past_data"),
        state.get("reasoning_provider", "Ollama"),
        state.get("reasoning_model", "glm-4.7-flash"),
        state.get("reasoning_api_key"),
        retrieved_context=state.get("retrieved_context"),
    )
    return {"analysis": analysis}


def _index_document_node(state: ExtractionState) -> ExtractionState:
    store = create_vector_store()
    if store is None:
        return {"vector_index_status": {"indexed": False, "reason": "No vector store is configured"}}

    ocr_payload = state.get("ocr", {})
    structured_data = state.get("structured_data") or {}
    metadata = {
        "patient_id_hash": hash_identifier(structured_data.get("patient", {}).get("mrn")),
        "encounter_date": structured_data.get("encounter", {}).get("date"),
        "source_type": "medical_record",
        "ocr_backend": ocr_payload.get("backend"),
    }
    try:
        chunks = build_chunks_from_ocr_payload(state["file_path"], ocr_payload, metadata)
        if not chunks:
            chunks = build_chunks_from_ocr_payload(
                state["file_path"],
                {"raw_text": json.dumps(structured_data), "per_page_results": []},
                metadata,
            )
        status = store.upsert_chunks(chunks)
    except Exception as exc:
        _logger.warning("indexing_failed", reason=str(exc))
        status = {"indexed": False, "reason": "Indexing failed"}
    return {"vector_index_status": status}
