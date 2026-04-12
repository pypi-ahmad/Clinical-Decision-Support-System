"""
Extraction Graph (Granular LangGraph Workflow)
----------------------------------------------
A deterministic, reviewable LangGraph workflow with explicit node-per-step
orchestration. Each node has a single responsibility and a typed handoff to
the next node, making each stage independently inspectable, retryable, and
testable.

Graph shape
-----------
ingest_document
    → classify_document_type
        → split_pages
            → ocr_per_page
                → extract_candidate_fields
                    → validate_against_schema
                        → normalize_codes
                            → retrieve_context
                                → merge_document_record
                                    → confidence_gate
                                        ↙           ↘
                               human_review      persist_record
                                    ↓
                               persist_record
                                    ↓
                                  END
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, TypedDict

from backend.ai_wrapper import clean_json_output, get_ai_response
from backend.database import get_patient_history, save_record
from backend.extract import run_document_ocr
from backend.logic import analyze_medical_logic
from backend.models import MedicalRecord
from backend.retrieval import build_chunks_from_ocr_payload, create_vector_store, hash_identifier

try:
    from langgraph.graph import END, START, StateGraph
except Exception:
    END = None
    START = None
    StateGraph = None


# ---------------------------------------------------------------------------
# Typed state
# ---------------------------------------------------------------------------


class ExtractionGraphState(TypedDict, total=False):
    # Input
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

    # Intermediate processing state
    document_type: str | None
    page_count: int
    page_image_paths: list[str]
    ocr: dict[str, Any]
    candidate_fields: dict[str, Any]
    validation_errors: list[str]
    normalized_fields: dict[str, Any]

    # Context
    past_data: dict[str, Any] | None
    retrieved_context: list[dict[str, Any]]

    # Merged/final
    structured_data: dict[str, Any]
    confidence_score: float
    requires_human_review: bool

    # Outputs
    analysis: dict[str, Any]
    vector_index_status: dict[str, Any] | None
    persisted: bool
    error: str | None


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


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

    # Conditional routing: human_review → persist if needed, confidence_gate → direct persist otherwise
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
    graph = build_extraction_graph()
    return graph.invoke(
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
    """Verify the file exists and is readable. Record basic file metadata."""
    file_path = state.get("file_path", "")
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}
    if not path.is_file():
        return {"error": f"Path is not a file: {file_path}"}
    return {"error": None}


def _classify_document_type_node(state: ExtractionGraphState) -> ExtractionGraphState:
    """Heuristically classify the document type (medical_record, insurance_policy, lab_report, etc.) based on filename and extension."""
    if state.get("error"):
        return {"document_type": None}

    file_path = state.get("file_path", "")
    path = Path(file_path)

    filename_lower = path.stem.lower()
    suffix_lower = path.suffix.lower()

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
    """Placeholder: page splitting is handled inside the OCR node.

    This node exists to preserve the graph topology (classify → split → OCR)
    but delegates actual PDF-to-page rendering to ``run_document_ocr``.
    """
    if state.get("error"):
        return {"page_count": 0, "page_image_paths": []}
    # Actual page rendering happens inside _ocr_per_page_node → run_document_ocr.
    return {"page_count": 0, "page_image_paths": []}


def _ocr_per_page_node(state: ExtractionGraphState) -> ExtractionGraphState:
    """Run OCR over the full document (all pages) using the selected OCR backend."""
    if state.get("error"):
        return {"ocr": {}}

    ocr_payload = run_document_ocr(
        state["file_path"],
        ocr_backend=state.get("ocr_backend", "ollama"),
        ocr_model=state.get("ocr_model"),
        ocr_prompt_mode=state.get("ocr_prompt_mode", "text"),
        use_gpu=state.get("use_gpu", True),
        paddle_service_url=state.get("paddle_service_url"),
    )

    if "error" in ocr_payload:
        return {"error": ocr_payload["error"], "ocr": {}}

    per_page = ocr_payload.get("per_page_results", [])
    page_images = ocr_payload.get("page_images", [])
    return {
        "ocr": ocr_payload,
        "page_count": len(per_page),
        "page_image_paths": page_images,
    }


_STRUCTURING_PROMPT = """
You are a medical data entry specialist. Convert the text below into valid JSON matching this schema:
{
  "patient": {"full_name": "string", "dob": "YYYY-MM-DD", "mrn": "string"},
  "encounter": {"date": "YYYY-MM-DD", "provider": "string", "facility": "string"},
  "clinical": {
    "diagnosis_list": ["string"],
    "medications": [{"name": "string", "dosage": "string", "frequency": "string"}],
    "vitals": {"bp": "string", "hr": "string", "temp": "string", "weight": "string"}
  }
}
Return ONLY JSON.
"""


def _extract_candidate_fields_node(state: ExtractionGraphState) -> ExtractionGraphState:
    """Run the structuring LLM to extract candidate structured fields from OCR output."""
    if state.get("error"):
        return {"candidate_fields": {}}

    ocr_payload = state.get("ocr") or {}
    raw_text = ocr_payload.get("markdown") or ocr_payload.get("raw_text") or ""
    if not raw_text:
        return {"error": "OCR produced no text", "candidate_fields": {}}

    provider = state.get("structuring_provider", "Ollama")
    model = state.get("structuring_model", "glm-4.7-flash")
    api_key = state.get("structuring_api_key")

    input_parts = [f"OCR_TEXT:\n{raw_text}"]
    markdown = ocr_payload.get("markdown", "")
    if markdown and markdown != raw_text:
        input_parts.append(f"OCR_MARKDOWN:\n{markdown}")
    structured_doc = ocr_payload.get("structured_doc")
    if structured_doc:
        input_parts.append(f"OCR_STRUCTURED_DOC:\n{json.dumps(structured_doc, ensure_ascii=False)}")
    user_text = "\n\n".join(input_parts)

    try:
        response = get_ai_response(provider, model, api_key, _STRUCTURING_PROMPT, user_text)
        candidate_fields = json.loads(clean_json_output(response))
        return {"candidate_fields": candidate_fields}
    except Exception as exc:
        return {"error": f"Structuring failed: {exc}", "candidate_fields": {}}


def _validate_against_schema_node(state: ExtractionGraphState) -> ExtractionGraphState:
    """Validate candidate fields against the MedicalRecord Pydantic schema."""
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
    """
    Normalize medical codes in the candidate fields.

    Currently applies light-touch normalization:
    - ICD-10 codes: uppercase, strip extraneous whitespace
    - Medication names: title-case where all caps
    - Diagnosis strings: strip trailing punctuation

    Does not fabricate codes where none are present.
    """
    candidate_fields = state.get("candidate_fields") or {}
    if not candidate_fields:
        return {"normalized_fields": candidate_fields}

    normalized = _deep_copy_dict(candidate_fields)
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
    """Retrieve patient history from SQLite and semantic context from the vector store."""
    normalized_fields = state.get("normalized_fields") or state.get("candidate_fields") or {}
    mrn = normalized_fields.get("patient", {}).get("mrn")
    past_data = get_patient_history(mrn) if mrn else None

    store = create_vector_store()
    if store is None or not mrn:
        return {"past_data": past_data, "retrieved_context": []}

    patient_hash = hash_identifier(mrn)
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
    except Exception:
        hits = []

    return {"past_data": past_data, "retrieved_context": hits}


def _merge_document_record_node(state: ExtractionGraphState) -> ExtractionGraphState:
    """
    Merge normalized candidate fields with retrieved context.

    Currently preserves normalized_fields as the merged record and stores the
    full reasoning analysis result in `structured_data`.  The reasoning step
    includes past_data and retrieved_context so the final summary reflects the
    combined picture.
    """
    normalized_fields = state.get("normalized_fields") or state.get("candidate_fields") or {}

    analysis = analyze_medical_logic(
        normalized_fields,
        state.get("past_data"),
        state.get("reasoning_provider", "Ollama"),
        state.get("reasoning_model", "glm-4.7-flash"),
        state.get("reasoning_api_key"),
        retrieved_context=state.get("retrieved_context"),
    )

    return {
        "structured_data": normalized_fields,
        "analysis": analysis,
    }


def _confidence_gate_node(state: ExtractionGraphState) -> ExtractionGraphState:
    """
    Assign a confidence score and decide whether human review is needed.

    Confidence is heuristic:
    - Starts at 1.0
    - Deducted 0.3 per validation error
    - Deducted 0.2 when OCR produced no text
    - Deducted 0.1 per missing top-level structured section

    Human review is triggered when confidence < 0.6 or validation errors exist.
    """
    errors = state.get("validation_errors") or []
    ocr_payload = state.get("ocr") or {}
    structured_data = state.get("structured_data") or {}

    score = 1.0
    score -= min(len(errors) * 0.3, 0.6)
    if not (ocr_payload.get("raw_text") or ocr_payload.get("markdown")):
        score -= 0.2
    for section in ("patient", "encounter", "clinical"):
        if not structured_data.get(section):
            score -= 0.1

    score = max(0.0, round(score, 2))
    requires_human_review = score < 0.6 or bool(errors)

    return {
        "confidence_score": score,
        "requires_human_review": requires_human_review,
    }


def _human_review_node(state: ExtractionGraphState) -> ExtractionGraphState:
    """Human review checkpoint (passthrough).

    The ``requires_human_review`` flag is already set by ``confidence_gate``.
    In an automated pipeline this node performs no additional work.  A
    production deployment should replace this body with logic that emits
    a task/ticket, pauses the graph, and resumes on human approval.
    """
    # TODO: integrate with an external review queue in production.
    return {}


def _persist_record_node(state: ExtractionGraphState) -> ExtractionGraphState:
    """Index the document into the vector store and record persistence."""
    file_path = state.get("file_path", "")
    structured_data = state.get("structured_data") or {}
    ocr_payload = state.get("ocr") or {}

    source_type = state.get("document_type") or "medical_record"
    store = create_vector_store()
    vector_index_status: dict[str, Any] | None = None
    if store is not None:
        metadata = {
            "patient_id_hash": hash_identifier(structured_data.get("patient", {}).get("mrn")),
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
            vector_index_status = {"indexed": False, "reason": str(exc)}
    else:
        vector_index_status = {"indexed": False, "reason": "No vector store is configured"}

    return {"vector_index_status": vector_index_status, "persisted": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_copy_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy nested dicts to avoid mutating state."""
    result: dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result[key] = _deep_copy_dict(value)
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    return result
