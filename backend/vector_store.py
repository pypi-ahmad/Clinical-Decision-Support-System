from __future__ import annotations

from typing import Any

from backend.retrieval import build_chunks_from_text, create_vector_store, get_enabled_store_type


def is_vector_indexing_enabled() -> bool:
    return get_enabled_store_type() is not None


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> list[str]:
    return [chunk.text for chunk in build_chunks_from_text("preview", text, {})]


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    store = create_vector_store()
    if store is None:
        return []
    from backend.retrieval.embeddings import embed_texts as retrieval_embed_texts

    return retrieval_embed_texts(texts, model=model)


def index_document_chunks(
    document_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
    collection_name: str | None = None,
) -> dict[str, Any]:
    store = create_vector_store(collection_name=collection_name)
    if store is None:
        return {"indexed": False, "reason": "No vector store is configured"}
    chunks = build_chunks_from_text(document_id, text, metadata)
    return store.upsert_chunks(chunks)


def search_document_chunks(
    query: str,
    limit: int = 5,
    filters: dict[str, Any] | None = None,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    store = create_vector_store(collection_name=collection_name)
    if store is None:
        return []
    return store.search(query, limit=limit, filters=filters)