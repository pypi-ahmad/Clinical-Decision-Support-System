"""
Tests for backend/workflows/extraction_graph.py

Covers each graph node in isolation as well as the conditional routing.
LangGraph is imported only where available; if absent, tests are skipped.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import backend.workflows.extraction_graph as eg


def test_classify_document_type_medical_record(tmp_path):
    """Default classification is medical_record."""
    file_path = str(tmp_path / "patient_notes.pdf")
    state = eg._classify_document_type_node({"file_path": file_path})
    assert state["document_type"] == "medical_record"


def test_classify_document_type_insurance_policy(tmp_path):
    file_path = str(tmp_path / "insurance_policy_2024.pdf")
    state = eg._classify_document_type_node({"file_path": file_path})
    assert state["document_type"] == "insurance_policy"


def test_classify_document_type_lab_report(tmp_path):
    file_path = str(tmp_path / "lab_results_q1.pdf")
    state = eg._classify_document_type_node({"file_path": file_path})
    assert state["document_type"] == "lab_report"


def test_classify_document_type_radiology(tmp_path):
    file_path = str(tmp_path / "xray_chest_2024.png")
    state = eg._classify_document_type_node({"file_path": file_path})
    assert state["document_type"] == "radiology_report"


def test_classify_document_type_skipped_on_error(tmp_path):
    file_path = str(tmp_path / "anything.pdf")
    state = eg._classify_document_type_node({"file_path": file_path, "error": "prior failure"})
    assert state["document_type"] is None


def test_ingest_document_node_missing_file(tmp_path):
    state = eg._ingest_document_node({"file_path": str(tmp_path / "nonexistent.pdf")})
    assert state["error"] is not None
    assert "File not found" in state["error"]


def test_ingest_document_node_existing_file(tmp_path):
    real_file = tmp_path / "doc.pdf"
    real_file.write_bytes(b"%PDF")
    state = eg._ingest_document_node({"file_path": str(real_file)})
    assert state["error"] is None


def test_ingest_document_node_directory_fails(tmp_path):
    state = eg._ingest_document_node({"file_path": str(tmp_path)})
    assert state["error"] is not None
    assert "not a file" in state["error"].lower()


def test_split_pages_node_returns_placeholder_counts():
    state = eg._split_pages_node({"file_path": "dummy.pdf"})
    assert "page_count" in state
    assert state["page_image_paths"] == []


def test_split_pages_node_skipped_on_error():
    state = eg._split_pages_node({"file_path": "x.pdf", "error": "bad"})
    assert state["page_count"] == 0
    assert state["page_image_paths"] == []


def test_validate_against_schema_node_valid_data():
    candidate = {
        "patient": {"full_name": "John Doe", "dob": "1980-01-01", "mrn": "MRN001"},
        "encounter": {"date": "2024-01-01", "provider": "Dr Smith", "facility": "Hospital"},
        "clinical": {
            "diagnosis_list": ["hypertension"],
            "medications": [{"name": "Lisinopril", "dosage": "10mg", "frequency": "once"}],
            "vitals": {"bp": "120/80", "hr": "70", "temp": "98.6", "weight": "70kg"},
        },
    }
    state = eg._validate_against_schema_node({"candidate_fields": candidate})
    assert state["validation_errors"] == []


def test_validate_against_schema_node_empty_candidate():
    state = eg._validate_against_schema_node({"candidate_fields": {}})
    assert len(state["validation_errors"]) > 0


def test_normalize_codes_node_strips_diagnosis_punctuation():
    candidate = {
        "clinical": {
            "diagnosis_list": ["Hypertension.", "Diabetes Mellitus,"],
            "medications": [],
        }
    }
    state = eg._normalize_codes_node({"candidate_fields": candidate})
    normalized = state["normalized_fields"]["clinical"]["diagnosis_list"]
    assert "Hypertension." not in normalized
    assert "Hypertension" in normalized
    assert "Diabetes Mellitus" in normalized


def test_normalize_codes_node_title_cases_all_caps_medication():
    candidate = {
        "clinical": {
            "diagnosis_list": [],
            "medications": [{"name": "LISINOPRIL", "dosage": "10mg", "frequency": "once"}],
        }
    }
    state = eg._normalize_codes_node({"candidate_fields": candidate})
    med_name = state["normalized_fields"]["clinical"]["medications"][0]["name"]
    assert med_name == "Lisinopril"


def test_normalize_codes_node_preserves_mixed_case_medication():
    candidate = {
        "clinical": {
            "diagnosis_list": [],
            "medications": [{"name": "Metformin", "dosage": "500mg", "frequency": "twice daily"}],
        }
    }
    state = eg._normalize_codes_node({"candidate_fields": candidate})
    med_name = state["normalized_fields"]["clinical"]["medications"][0]["name"]
    assert med_name == "Metformin"


def test_normalize_codes_node_empty_fields():
    state = eg._normalize_codes_node({"candidate_fields": {}})
    assert state["normalized_fields"] == {}


def test_confidence_gate_full_confidence():
    state = eg._confidence_gate_node(
        {
            "validation_errors": [],
            "ocr": {"raw_text": "some text"},
            "structured_data": {
                "patient": {"mrn": "X"},
                "encounter": {"date": "2024-01-01"},
                "clinical": {"diagnosis_list": ["HTN"]},
            },
        }
    )
    assert state["confidence_score"] == pytest.approx(1.0)
    assert state["requires_human_review"] is False


def test_confidence_gate_low_confidence_on_errors():
    state = eg._confidence_gate_node(
        {
            "validation_errors": ["missing mrn", "missing dob", "missing provider"],
            "ocr": {"raw_text": "some text"},
            "structured_data": {"patient": {}, "encounter": {}, "clinical": {}},
        }
    )
    assert state["confidence_score"] < 0.6
    assert state["requires_human_review"] is True


def test_confidence_gate_low_when_no_ocr_text():
    state = eg._confidence_gate_node(
        {
            "validation_errors": [],
            "ocr": {},
            "structured_data": {"patient": {"mrn": "X"}, "encounter": {}, "clinical": {}},
        }
    )
    # 0.2 deducted for no OCR text, 0.2 deducted for missing encounter/clinical = 0.6 borderline
    assert state["confidence_score"] <= 0.8


def test_route_after_confidence_gate_human_review():
    route = eg._route_after_confidence_gate({"requires_human_review": True})
    assert route == "human_review"


def test_route_after_confidence_gate_direct_persist():
    route = eg._route_after_confidence_gate({"requires_human_review": False})
    assert route == "persist_record"


def test_human_review_node_is_passthrough():
    state = eg._human_review_node({"requires_human_review": True, "structured_data": {}})
    assert state == {}


def test_persist_record_node_no_store(monkeypatch):
    """When no vector store is available, persist_record still completes gracefully."""
    monkeypatch.setattr(eg, "create_vector_store", lambda: None)
    state = eg._persist_record_node(
        {
            "file_path": "dummy.pdf",
            "structured_data": {"patient": {"mrn": "X"}},
            "ocr": {},
        }
    )
    assert state["persisted"] is True
    assert state["vector_index_status"]["indexed"] is False


def test_extract_candidate_fields_node_error_passthrough():
    state = eg._extract_candidate_fields_node({"error": "prior err", "ocr": {}})
    assert state["candidate_fields"] == {}


def test_extract_candidate_fields_node_no_text():
    state = eg._extract_candidate_fields_node({"ocr": {}, "structuring_provider": "Ollama"})
    assert state.get("error") is not None
    assert state["candidate_fields"] == {}


def test_extract_candidate_fields_node_calls_structuring_llm(monkeypatch):
    monkeypatch.setattr(
        eg,
        "get_ai_response",
        lambda provider, model, key, sys, user: '{"patient": {}, "encounter": {}, "clinical": {}}',
    )
    monkeypatch.setattr(eg, "clean_json_output", lambda s: s)
    state = eg._extract_candidate_fields_node(
        {
            "ocr": {"raw_text": "patient notes text"},
            "structuring_provider": "Ollama",
            "structuring_model": "glm-4.7-flash",
            "structuring_api_key": None,
        }
    )
    assert "patient" in state["candidate_fields"]
    assert "error" not in state


def test_ocr_per_page_node_error_passthrough():
    state = eg._ocr_per_page_node({"error": "already failed", "file_path": "x.pdf"})
    assert state["ocr"] == {}


def test_ocr_per_page_node_calls_run_document_ocr(monkeypatch):
    monkeypatch.setattr(
        eg,
        "run_document_ocr",
        lambda file_path, **kwargs: {
            "raw_text": "ocr output",
            "per_page_results": [{"page_number": 1, "raw_text": "ocr output"}],
            "page_images": ["page1.png"],
        },
    )
    state = eg._ocr_per_page_node(
        {
            "file_path": "dummy.pdf",
            "ocr_backend": "ollama",
            "ocr_model": None,
            "ocr_prompt_mode": "text",
            "use_gpu": True,
            "paddle_service_url": None,
        }
    )
    assert state["ocr"]["raw_text"] == "ocr output"
    assert state["page_count"] == 1
    assert state["page_image_paths"] == ["page1.png"]


def test_ocr_per_page_node_propagates_ocr_error(monkeypatch):
    monkeypatch.setattr(
        eg,
        "run_document_ocr",
        lambda file_path, **kwargs: {"error": "poppler not found"},
    )
    state = eg._ocr_per_page_node(
        {
            "file_path": "dummy.pdf",
            "ocr_backend": "ollama",
        }
    )
    assert state["error"] == "poppler not found"
    assert state["ocr"] == {}


def test_retrieve_context_node_no_mrn():
    state = eg._retrieve_context_node({"candidate_fields": {}, "normalized_fields": {}})
    assert state["past_data"] is None
    assert state["retrieved_context"] == []


def test_retrieve_context_node_with_store(monkeypatch):
    captured = {}

    class DummyStore:
        def search(self, query, limit=5, filters=None):
            captured["query"] = query
            captured["filters"] = filters
            return [{"text": "prior visit", "score": 0.9}]

    monkeypatch.setattr(eg, "create_vector_store", lambda: DummyStore())
    monkeypatch.setattr(eg, "get_patient_history", lambda mrn: None)

    state = eg._retrieve_context_node(
        {
            "normalized_fields": {"patient": {"mrn": "MRN-1"}, "clinical": {"diagnosis_list": ["HTN"], "medications": []}},
        }
    )
    assert len(state["retrieved_context"]) == 1
    assert "HTN" in captured["query"]


def test_retrieve_context_uses_document_type(monkeypatch):
    """Verify that document_type from classify_document_type flows into retrieval filters."""
    captured = {}

    class DummyStore:
        def search(self, query, limit=5, filters=None):
            captured["filters"] = filters
            return []

    monkeypatch.setattr(eg, "create_vector_store", lambda: DummyStore())
    monkeypatch.setattr(eg, "get_patient_history", lambda mrn: None)

    eg._retrieve_context_node(
        {
            "document_type": "lab_report",
            "normalized_fields": {"patient": {"mrn": "MRN-2"}, "clinical": {"diagnosis_list": ["Anemia"]}},
        }
    )
    assert captured["filters"]["source_type"] == "lab_report"


def test_persist_record_uses_document_type(monkeypatch):
    """Verify that document_type from classify_document_type flows into persist chunk metadata."""
    captured = {}

    class DummyStore:
        def upsert_chunks(self, chunks):
            captured["source_types"] = [c.source_type for c in chunks]
            return {"indexed": True, "chunks": len(chunks)}

    monkeypatch.setattr(eg, "create_vector_store", lambda: DummyStore())

    state = eg._persist_record_node(
        {
            "file_path": "dummy.pdf",
            "document_type": "discharge_summary",
            "structured_data": {"patient": {"mrn": "X"}},
            "ocr": {"raw_text": "Patient discharged in stable condition.", "per_page_results": []},
        }
    )
    assert state["persisted"] is True
    assert all(st == "discharge_summary" for st in captured["source_types"])


def test_merge_document_record_node_calls_reasoning(monkeypatch):
    monkeypatch.setattr(
        eg,
        "analyze_medical_logic",
        lambda current, past, provider, model, key, **kwargs: {"summary": "stable", "alerts": [], "trends": []},
    )
    state = eg._merge_document_record_node(
        {
            "normalized_fields": {"patient": {"mrn": "X"}},
            "past_data": None,
            "reasoning_provider": "Ollama",
            "reasoning_model": "glm-4.7-flash",
            "reasoning_api_key": None,
            "retrieved_context": [],
        }
    )
    assert state["analysis"]["summary"] == "stable"
    assert state["structured_data"] == {"patient": {"mrn": "X"}}


def test_deep_copy_dict_is_independent():
    original = {"a": {"b": 1}, "c": [1, 2, 3]}
    copy = eg._deep_copy_dict(original)
    copy["a"]["b"] = 99
    assert original["a"]["b"] == 1


def test_build_extraction_graph_requires_langgraph():
    """build_extraction_graph should raise RuntimeError when LangGraph is unavailable."""
    original_state_graph = eg.StateGraph
    original_end = eg.END
    original_start = eg.START
    eg.StateGraph = None
    eg.END = None
    eg.START = None
    try:
        with pytest.raises(RuntimeError, match="LangGraph"):
            eg.build_extraction_graph()
    finally:
        eg.StateGraph = original_state_graph
        eg.END = original_end
        eg.START = original_start


# ---------------------------------------------------------------------------
# Compiled-graph end-to-end invocation
# ---------------------------------------------------------------------------

_needs_langgraph = pytest.mark.skipif(eg.StateGraph is None, reason="LangGraph not installed")


@_needs_langgraph
def test_compiled_extraction_graph_runs_end_to_end(tmp_path, monkeypatch):
    """Build and invoke the compiled graph, proving node ordering and state propagation."""
    # Create a real file so ingest_document passes
    sample = tmp_path / "patient_notes.pdf"
    sample.write_bytes(b"%PDF-1.4 fake")

    # Mock only external IO: OCR, LLM, vector store, patient history
    monkeypatch.setattr(
        eg,
        "run_document_ocr",
        lambda file_path, **kw: {
            "raw_text": "John Doe BP 130/85",
            "markdown": "John Doe BP 130/85",
            "per_page_results": [{"page_number": 1, "raw_text": "John Doe BP 130/85"}],
            "page_images": [str(sample)],
        },
    )
    monkeypatch.setattr(
        eg,
        "get_ai_response",
        lambda prov, model, key, sys, user: '{"patient": {"full_name": "John Doe", "mrn": "MRN-1"}, "encounter": {"date": "2024-06-01"}, "clinical": {"diagnosis_list": ["HTN"], "medications": [], "vitals": {"bp": "130/85"}}}',
    )
    monkeypatch.setattr(eg, "clean_json_output", lambda s: s)
    monkeypatch.setattr(eg, "create_vector_store", lambda: None)
    monkeypatch.setattr(eg, "get_patient_history", lambda mrn: None)
    monkeypatch.setattr(
        eg,
        "analyze_medical_logic",
        lambda cur, past, prov, model, key, **kw: {"summary": "stable", "alerts": [], "trends": []},
    )

    result = eg.run_extraction_graph(
        file_path=str(sample),
        structuring_provider="Ollama",
        structuring_model="test-model",
        structuring_api_key=None,
        reasoning_provider="Ollama",
        reasoning_model="test-model",
        reasoning_api_key=None,
        ocr_backend="ollama",
        ocr_model=None,
        ocr_prompt_mode="text",
        use_gpu=False,
        paddle_service_url=None,
    )

    # Verify state propagated through all 12 nodes
    assert result.get("error") is None
    assert result["document_type"] == "medical_record"
    assert result["structured_data"]["patient"]["full_name"] == "John Doe"
    assert result["analysis"]["summary"] == "stable"
    assert result["confidence_score"] == pytest.approx(1.0)
    assert result["requires_human_review"] is False
    assert result["persisted"] is True
    assert result["vector_index_status"]["indexed"] is False


@_needs_langgraph
def test_compiled_extraction_graph_routes_to_human_review(tmp_path, monkeypatch):
    """When OCR yields no text, confidence drops and graph routes through human_review."""
    sample = tmp_path / "blank.pdf"
    sample.write_bytes(b"%PDF-1.4 blank")

    monkeypatch.setattr(
        eg,
        "run_document_ocr",
        lambda file_path, **kw: {
            "raw_text": "",
            "markdown": "",
            "per_page_results": [],
            "page_images": [],
        },
    )
    monkeypatch.setattr(eg, "create_vector_store", lambda: None)
    monkeypatch.setattr(eg, "get_patient_history", lambda mrn: None)
    monkeypatch.setattr(
        eg,
        "analyze_medical_logic",
        lambda cur, past, prov, model, key, **kw: {"summary": "N/A", "alerts": [], "trends": []},
    )

    result = eg.run_extraction_graph(
        file_path=str(sample),
        structuring_provider="Ollama",
        structuring_model="test-model",
        structuring_api_key=None,
        reasoning_provider="Ollama",
        reasoning_model="test-model",
        reasoning_api_key=None,
        ocr_backend="ollama",
        ocr_model=None,
        ocr_prompt_mode="text",
        use_gpu=False,
        paddle_service_url=None,
    )

    assert result["requires_human_review"] is True
    assert result["confidence_score"] < 0.6
    assert result["persisted"] is True
