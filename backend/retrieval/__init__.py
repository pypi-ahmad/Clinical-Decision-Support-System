from __future__ import annotations

import hashlib
import os

from backend.retrieval.chunking import build_chunks_from_ocr_payload, build_chunks_from_text
from backend.retrieval.pgvector_store import PgvectorRetrievalStore
from backend.retrieval.qdrant_store import QdrantRetrievalStore


def hash_identifier(identifier: str | None) -> str | None:
    if not identifier:
        return None
    normalized = identifier.strip()
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def get_enabled_store_type() -> str | None:
    preferred = os.getenv("VECTOR_STORE", "").strip().lower()
    if preferred == "pgvector" and PgvectorRetrievalStore.is_configured():
        return "pgvector"
    if preferred == "qdrant" and QdrantRetrievalStore.is_configured():
        return "qdrant"
    if QdrantRetrievalStore.is_configured():
        return "qdrant"
    if PgvectorRetrievalStore.is_configured():
        return "pgvector"
    return None


def create_vector_store(collection_name: str | None = None):
    store_type = get_enabled_store_type()
    if store_type == "qdrant":
        return QdrantRetrievalStore(collection_name=collection_name)
    if store_type == "pgvector":
        return PgvectorRetrievalStore(collection_name=collection_name)
    return None


__all__ = [
    "build_chunks_from_ocr_payload",
    "build_chunks_from_text",
    "create_vector_store",
    "get_enabled_store_type",
    "hash_identifier",
]