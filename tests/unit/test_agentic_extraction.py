"""
Tests for backend/workflows/agentic_extraction.py

Covers each agentic graph node in isolation plus a compiled-graph invocation.
"""

from __future__ import annotations

import pytest

import backend.workflows.agentic_extraction as ae

# ---------------------------------------------------------------------------
# Node-level unit tests
# ---------------------------------------------------------------------------


def test_extract_document_node_success(monkeypatch):
    monkeypatch.setattr(
        ae,
        "process_document_pipeline",
        lambda file_path, prov, model, key, **kw: {
            "structured_data": {"patient": {"mrn": "MRN-1"}},
            "ocr": {"raw_text": "some text"},
        },
    )
    state = ae._extract_document_node(
        {
            "file_path": "dummy.pdf",
            "structuring_provider": "Ollama",
            "structuring_model": "test",
            "structuring_api_key": None,
        }
    )
    assert state["structured_data"]["patient"]["mrn"] == "MRN-1"
    assert state["error"] is None


def test_extract_document_node_error(monkeypatch):
    monkeypatch.setattr(
        ae,
        "process_document_pipeline",
        lambda file_path, prov, model, key, **kw: {"error": "OCR failed"},
    )
    state = ae._extract_document_node(
        {
            "file_path": "dummy.pdf",
            "structuring_provider": "Ollama",
            "structuring_model": "test",
            "structuring_api_key": None,
        }
    )
    assert state["error"] == "OCR failed"


def test_validate_structured_data_node_valid():
    state = ae._validate_structured_data_node(
        {
            "structured_data": {
                "patient": {"full_name": "A", "mrn": "1"},
                "encounter": {},
                "clinical": {},
            }
        }
    )
    assert state["validation_errors"] == []


def test_validate_structured_data_node_invalid():
    # The hardened schema is intentionally lenient on extra keys (OCR is
    # noisy) but rejects fundamentally wrong types — e.g. a string where a
    # nested object is expected.
    state = ae._validate_structured_data_node({"structured_data": {"patient": "not-an-object"}})
    assert len(state["validation_errors"]) > 0


def test_human_review_gate_with_errors():
    state = ae._human_review_gate_node({"validation_errors": ["missing field"]})
    assert state["requires_human_review"] is True


def test_human_review_gate_no_errors():
    state = ae._human_review_gate_node({"validation_errors": []})
    assert state["requires_human_review"] is False


def test_load_history_node_with_mrn(monkeypatch):
    monkeypatch.setattr(ae, "get_patient_history", lambda mrn: {"patient": {"mrn": mrn}})
    state = ae._load_history_node({"structured_data": {"patient": {"mrn": "MRN-1"}}})
    assert state["past_data"] is not None
    assert state["past_data"]["patient"]["mrn"] == "MRN-1"


def test_load_history_node_no_mrn(monkeypatch):
    monkeypatch.setattr(ae, "get_patient_history", lambda mrn: None)
    state = ae._load_history_node({"structured_data": {}})
    assert state["past_data"] is None


def test_retrieve_context_node_no_store(monkeypatch):
    monkeypatch.setattr(ae, "create_vector_store", lambda: None)
    state = ae._retrieve_context_node({"structured_data": {"patient": {"mrn": "X"}}, "ocr": {}})
    assert state["retrieved_context"] == []


def test_retrieve_context_node_with_store(monkeypatch):
    captured = {}

    class DummyStore:
        def search(self, query, limit=5, filters=None):
            captured["filters"] = filters
            return [{"text": "previous note", "score": 0.8}]

    monkeypatch.setattr(ae, "create_vector_store", lambda: DummyStore())

    state = ae._retrieve_context_node(
        {
            "structured_data": {
                "patient": {"mrn": "MRN-5"},
                "clinical": {"diagnosis_list": ["Diabetes"]},
            },
            "ocr": {},
        }
    )
    assert len(state["retrieved_context"]) == 1
    assert captured["filters"]["source_type"] == "medical_record"


def test_analyze_document_node_success(monkeypatch):
    monkeypatch.setattr(
        ae,
        "analyze_medical_logic",
        lambda cur, past, prov, model, key, **kw: {"summary": "ok", "alerts": [], "trends": []},
    )
    state = ae._analyze_document_node(
        {
            "structured_data": {"patient": {}},
            "past_data": None,
            "reasoning_provider": "Ollama",
            "reasoning_model": "test",
            "reasoning_api_key": None,
            "retrieved_context": [],
        }
    )
    assert state["analysis"]["summary"] == "ok"


def test_analyze_document_node_skips_on_error():
    state = ae._analyze_document_node({"error": "upstream failure"})
    assert state["analysis"]["summary"] == "Analysis skipped"


def test_index_document_node_no_store(monkeypatch):
    monkeypatch.setattr(ae, "create_vector_store", lambda: None)
    state = ae._index_document_node({"file_path": "x.pdf", "structured_data": {}, "ocr": {}})
    assert state["vector_index_status"]["indexed"] is False


def test_index_document_node_with_store(monkeypatch):
    class DummyStore:
        def upsert_chunks(self, chunks):
            return {"indexed": True, "chunks": len(chunks)}

    monkeypatch.setattr(ae, "create_vector_store", lambda: DummyStore())

    state = ae._index_document_node(
        {
            "file_path": "x.pdf",
            "structured_data": {"patient": {"mrn": "X"}},
            "ocr": {"raw_text": "some note", "per_page_results": []},
        }
    )
    assert state["vector_index_status"]["indexed"] is True


# ---------------------------------------------------------------------------
# Compiled-graph end-to-end invocation
# ---------------------------------------------------------------------------

_needs_langgraph = pytest.mark.skipif(ae.StateGraph is None, reason="LangGraph not installed")


@_needs_langgraph
def test_compiled_agentic_graph_runs_end_to_end(tmp_path, monkeypatch):
    """Build and invoke the agentic graph, proving node ordering and state propagation."""
    sample = tmp_path / "record.pdf"
    sample.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(
        ae,
        "process_document_pipeline",
        lambda file_path, prov, model, key, **kw: {
            "structured_data": {
                "patient": {"full_name": "Jane Doe", "mrn": "MRN-99"},
                "encounter": {"date": "2024-07-01"},
                "clinical": {"diagnosis_list": ["Asthma"], "medications": [], "vitals": {}},
            },
            "ocr": {"raw_text": "Jane Doe Asthma", "per_page_results": []},
        },
    )
    monkeypatch.setattr(ae, "get_patient_history", lambda mrn: None)
    monkeypatch.setattr(ae, "create_vector_store", lambda: None)
    monkeypatch.setattr(
        ae,
        "analyze_medical_logic",
        lambda cur, past, prov, model, key, **kw: {"summary": "new patient", "alerts": [], "trends": []},
    )

    result = ae.run_agentic_extraction_workflow(
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

    assert result.get("error") is None
    assert result["structured_data"]["patient"]["full_name"] == "Jane Doe"
    assert result["analysis"]["summary"] == "new patient"
    assert result["requires_human_review"] is False
    assert result["vector_index_status"]["indexed"] is False
