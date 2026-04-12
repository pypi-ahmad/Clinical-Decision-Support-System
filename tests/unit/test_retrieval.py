import backend.main as main
from backend.retrieval import hash_identifier
from backend.retrieval.chunking import build_chunks_from_ocr_payload


def test_build_chunks_from_ocr_payload_preserves_page_metadata():
    """Targets backend.retrieval.chunking build_chunks_from_ocr_payload."""
    payload = {
        "backend": "glm",
        "per_page_results": [
            {"page_number": 1, "markdown": "page one findings", "image_path": "page1.png"},
            {"page_number": 2, "raw_text": "page two findings", "image_path": "page2.png"},
        ],
    }
    metadata = {"patient_id_hash": "abc123", "source_type": "medical_record", "ocr_backend": "glm"}

    chunks = build_chunks_from_ocr_payload("doc-1", payload, metadata)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2
    assert chunks[0].source_ref == "page1.png"
    assert chunks[0].patient_id_hash == "abc123"
    assert chunks[0].ocr_backend == "glm"


def test_retrieve_patient_context_uses_exact_patient_hash_filter(monkeypatch):
    """Targets backend.main _retrieve_patient_context exact retrieval filters."""
    captured = {}

    class DummyStore:
        def search(self, query, limit=5, filters=None):
            captured["query"] = query
            captured["filters"] = filters
            captured["limit"] = limit
            return [{"text": "prior hypertension note", "score": 0.9}]

    monkeypatch.setattr(main, "create_vector_store", lambda: DummyStore())

    current_data = {
        "patient": {"mrn": "MRN-42"},
        "clinical": {
            "diagnosis_list": ["hypertension"],
            "medications": [{"name": "lisinopril"}],
        },
    }

    result = main._retrieve_patient_context(current_data, {"raw_text": "ignored fallback"})

    assert result[0]["text"] == "prior hypertension note"
    assert captured["filters"] == {
        "patient_id_hash": hash_identifier("MRN-42"),
        "source_type": "medical_record",
    }
    assert "hypertension" in captured["query"]
    assert "lisinopril" in captured["query"]