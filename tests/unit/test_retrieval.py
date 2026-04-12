import backend.main as main
from backend.retrieval import hash_identifier, create_vector_store, get_enabled_store_type
from backend.retrieval.chunking import build_chunks_from_ocr_payload, build_chunks_from_text, split_text_to_chunks
from backend.retrieval.vector_store import RetrievalChunk


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


# ---------------------------------------------------------------------------
# hash_identifier
# ---------------------------------------------------------------------------


def test_hash_identifier_returns_sha256_hex():
    h = hash_identifier("MRN-001")
    assert isinstance(h, str)
    assert len(h) == 64


def test_hash_identifier_same_input_same_output():
    assert hash_identifier("MRN-001") == hash_identifier("MRN-001")


def test_hash_identifier_different_inputs_different_hashes():
    assert hash_identifier("MRN-001") != hash_identifier("MRN-002")


def test_hash_identifier_returns_none_for_empty():
    assert hash_identifier(None) is None
    assert hash_identifier("") is None
    assert hash_identifier("   ") is None


# ---------------------------------------------------------------------------
# split_text_to_chunks
# ---------------------------------------------------------------------------


def test_split_text_to_chunks_respects_chunk_size():
    text = "A" * 3000
    chunks = split_text_to_chunks(text, chunk_size=1200, chunk_overlap=0)
    assert len(chunks) >= 2
    assert all(len(c) <= 1200 for c in chunks)


def test_split_text_to_chunks_empty_returns_empty():
    assert split_text_to_chunks("") == []
    assert split_text_to_chunks("   ") == []


def test_split_text_to_chunks_short_text_single_chunk():
    text = "short text"
    chunks = split_text_to_chunks(text, chunk_size=1200)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_text_to_chunks_overlap_produces_more_chunks():
    text = "x" * 2500
    no_overlap = split_text_to_chunks(text, chunk_size=1200, chunk_overlap=0)
    with_overlap = split_text_to_chunks(text, chunk_size=1200, chunk_overlap=150)
    assert len(with_overlap) >= len(no_overlap)


# ---------------------------------------------------------------------------
# build_chunks_from_text
# ---------------------------------------------------------------------------


def test_build_chunks_from_text_creates_chunks_with_metadata():
    chunks = build_chunks_from_text(
        "doc-1",
        "Clinical notes text for testing purposes.",
        {"patient_id_hash": "deadbeef", "source_type": "medical_record", "encounter_date": "2024-01-01"},
        page_number=3,
        source_ref="page3.png",
        section_type="ocr_page",
    )
    assert len(chunks) >= 1
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].patient_id_hash == "deadbeef"
    assert chunks[0].source_type == "medical_record"
    assert chunks[0].page_number == 3
    assert chunks[0].source_ref == "page3.png"
    assert chunks[0].section_type == "ocr_page"


def test_build_chunks_from_text_empty_returns_empty():
    chunks = build_chunks_from_text("doc-1", "", {})
    assert chunks == []


def test_build_chunks_from_text_no_metadata_uses_defaults():
    chunks = build_chunks_from_text("doc-1", "some text", None)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source_type == "document"
    assert chunk.patient_id_hash is None


# ---------------------------------------------------------------------------
# build_chunks_from_ocr_payload fallback paths
# ---------------------------------------------------------------------------


def test_build_chunks_from_ocr_payload_falls_back_to_document_text():
    """When per_page_results is empty, falls back to raw_text/markdown."""
    payload = {
        "raw_text": "entire document text",
        "per_page_results": [],
    }
    chunks = build_chunks_from_ocr_payload("doc-1", payload, {})
    assert len(chunks) >= 1
    assert "entire document text" in chunks[0].text


def test_build_chunks_from_ocr_payload_prefers_per_page():
    payload = {
        "raw_text": "entire document text",
        "per_page_results": [
            {"page_number": 1, "markdown": "page one"},
        ],
    }
    chunks = build_chunks_from_ocr_payload("doc-1", payload, {})
    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].text == "page one"


def test_build_chunks_from_ocr_payload_skips_empty_pages():
    payload = {
        "per_page_results": [
            {"page_number": 1, "markdown": ""},
            {"page_number": 2, "raw_text": "   "},
            {"page_number": 3, "markdown": "valid content"},
        ],
    }
    chunks = build_chunks_from_ocr_payload("doc-1", payload, {})
    assert len(chunks) == 1
    assert chunks[0].page_number == 3


# ---------------------------------------------------------------------------
# RetrievalChunk.to_payload
# ---------------------------------------------------------------------------


def test_retrieval_chunk_to_payload_includes_all_fields():
    chunk = RetrievalChunk(
        chunk_id="id-x",
        document_id="doc-1",
        text="patient text",
        page_number=2,
        section_type="ocr_page",
        patient_id_hash="abc",
        encounter_date="2024-01-01",
        source_type="medical_record",
        source_ref="page2.png",
        ocr_backend="glm",
        metadata={"extra": "value"},
        chunk_index=1,
    )
    payload = chunk.to_payload()
    assert payload["chunk_id"] == "id-x"
    assert payload["text"] == "patient text"
    assert payload["page_number"] == 2
    assert payload["patient_id_hash"] == "abc"
    assert payload["ocr_backend"] == "glm"
    assert payload["extra"] == "value"


# ---------------------------------------------------------------------------
# create_vector_store with no configuration
# ---------------------------------------------------------------------------


def test_create_vector_store_returns_none_when_nothing_configured(monkeypatch):
    """Without QDRANT_URL or PGVECTOR_URL, create_vector_store returns None."""
    import backend.retrieval as retrieval
    monkeypatch.setattr(retrieval, "get_enabled_store_type", lambda: None)
    store = retrieval.create_vector_store()
    assert store is None


# ---------------------------------------------------------------------------
# _retrieve_policy_context (via main module)
# ---------------------------------------------------------------------------


def test_retrieve_policy_context_no_store_returns_empty(monkeypatch):
    monkeypatch.setattr(main, "create_vector_store", lambda: None)
    result = main._retrieve_policy_context(
        {"clinical": {"diagnosis_list": ["HTN"]}},
        "policy text",
        "policy-doc-1",
    )
    assert result == []


def test_retrieve_policy_context_indexes_and_queries(monkeypatch):
    upserted = {}
    searched = {}

    class DummyStore:
        def upsert_chunks(self, chunks):
            upserted["n"] = len(chunks)
            return {"indexed": True, "chunks": len(chunks)}

        def search(self, query, limit=5, filters=None):
            searched["query"] = query
            searched["filters"] = filters
            return [{"text": "policy clause", "score": 0.88}]

    monkeypatch.setattr(main, "create_vector_store", lambda: DummyStore())

    result = main._retrieve_policy_context(
        {"clinical": {"diagnosis_list": ["Hypertension"]}},
        "This policy covers hypertension treatment.",
        "policy-doc-ABC",
    )

    assert upserted["n"] >= 1
    assert result[0]["text"] == "policy clause"
    assert searched["filters"]["source_type"] == "insurance_policy"
    assert searched["filters"]["policy_document_id"] == "policy-doc-ABC"


# ---------------------------------------------------------------------------
# _index_for_retrieval (via main module)
# ---------------------------------------------------------------------------


def test_index_for_retrieval_no_store_returns_not_indexed(monkeypatch):
    monkeypatch.setattr(main, "create_vector_store", lambda: None)
    result = main._index_for_retrieval("doc.pdf", {"patient": {"mrn": "X"}}, {})
    assert result["indexed"] is False


def test_index_for_retrieval_upserts_to_store(monkeypatch):
    inserted = {}

    class DummyStore:
        def upsert_chunks(self, chunks):
            inserted["n"] = len(chunks)
            return {"indexed": True, "chunks": len(chunks)}

    monkeypatch.setattr(main, "create_vector_store", lambda: DummyStore())

    result = main._index_for_retrieval(
        "doc.pdf",
        {"patient": {"mrn": "X"}, "encounter": {"date": "2024-01-01"}},
        {"raw_text": "clinical note text", "per_page_results": []},
    )
    assert result["indexed"] is True
    assert inserted["n"] >= 1


# ---------------------------------------------------------------------------
# QdrantRetrievalStore with mocked client
# ---------------------------------------------------------------------------


def test_qdrant_store_upsert_mocked_client(monkeypatch):
    """Exercise QdrantRetrievalStore.upsert_chunks with a mocked QdrantClient."""
    import backend.retrieval.qdrant_store as qs

    if qs.QdrantClient is None:
        pytest.skip("qdrant-client not installed")

    # Mock the embeddings function to return fixed vectors
    monkeypatch.setattr(qs, "embed_texts", lambda texts, model=None: [[0.1, 0.2, 0.3]] * len(texts))

    # Create a store but mock out the actual client
    monkeypatch.setenv("QDRANT_URL", "http://fake:6333")

    upserted = {}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def collection_exists(self, name):
            return True

        def upsert(self, collection_name, points):
            upserted["collection"] = collection_name
            upserted["count"] = len(points)

    monkeypatch.setattr(qs, "QdrantClient", FakeClient)

    store = qs.QdrantRetrievalStore(collection_name="test_collection")
    chunks = [
        RetrievalChunk(
            chunk_id="c1",
            document_id="doc-1",
            text="patient has hypertension",
            section_type="text",
            source_type="medical_record",
            chunk_index=0,
        ),
    ]
    result = store.upsert_chunks(chunks)

    assert result["indexed"] is True
    assert result["chunks"] == 1
    assert upserted["collection"] == "test_collection"
    assert upserted["count"] == 1


def test_qdrant_store_search_mocked_client(monkeypatch):
    """Exercise QdrantRetrievalStore.search with a mocked QdrantClient."""
    import backend.retrieval.qdrant_store as qs

    if qs.QdrantClient is None:
        pytest.skip("qdrant-client not installed")

    monkeypatch.setattr(qs, "embed_texts", lambda texts, model=None: [[0.1, 0.2, 0.3]])
    monkeypatch.setenv("QDRANT_URL", "http://fake:6333")

    class FakeHit:
        def __init__(self):
            self.score = 0.95
            self.payload = {"text": "prior visit note", "page_number": 1, "source_ref": "p1.png"}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def search(self, collection_name, query_vector, query_filter, limit):
            return [FakeHit()]

    monkeypatch.setattr(qs, "QdrantClient", FakeClient)

    store = qs.QdrantRetrievalStore(collection_name="test_collection")
    hits = store.search("hypertension", limit=3, filters={"patient_id_hash": "abc123"})

    assert len(hits) == 1
    assert hits[0]["score"] == 0.95
    assert hits[0]["text"] == "prior visit note"
