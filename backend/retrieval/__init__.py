"""Retrieval subsystem public surface.

The ``hash_identifier`` helper is HMAC-SHA256 with a server-held pepper
(``MRN_HMAC_PEPPER``). Without the pepper it returns ``None`` and retrieval
silently no-ops — this is intentional: short MRN strings are re-identifiable
via plain-SHA256 rainbow tables, so we refuse to ever produce an unsalted
digest. There is deliberately no weak-hash fallback.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from functools import lru_cache
from threading import Lock

from backend.retrieval.chunking import build_chunks_from_ocr_payload, build_chunks_from_text
from backend.retrieval.pgvector_store import PgvectorRetrievalStore
from backend.retrieval.qdrant_store import QdrantRetrievalStore

_store_lock = Lock()
_cached_store = None


def hash_identifier(identifier: str | None) -> str | None:
    """Return an HMAC-SHA256 digest of ``identifier`` using ``MRN_HMAC_PEPPER``.

    Returns ``None`` when either the identifier is empty or when no pepper is
    configured. Callers treat ``None`` as "retrieval / audit linkage disabled".
    The helper NEVER falls back to plain SHA-256: an unsalted digest of a short
    MRN is trivially re-identifiable via rainbow table.
    """
    if not identifier:
        return None
    normalized = identifier.strip()
    if not normalized:
        return None
    pepper = os.environ.get("MRN_HMAC_PEPPER")
    if not pepper:
        return None
    return hmac.new(
        pepper.encode("utf-8"),
        normalized.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


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


@lru_cache(maxsize=4)
def _cached_vector_store(collection_name: str | None):
    store_type = get_enabled_store_type()
    if store_type == "qdrant":
        return QdrantRetrievalStore(collection_name=collection_name)
    if store_type == "pgvector":
        return PgvectorRetrievalStore(collection_name=collection_name)
    return None


def create_vector_store(collection_name: str | None = None):
    """Return a process-wide singleton vector store (or ``None``).

    The returned client holds its own TCP connections to Qdrant and is thread-
    safe per the qdrant-client docs, so reusing it across requests is the
    right call. ``reset_vector_store`` is provided for tests.
    """
    return _cached_vector_store(collection_name)


def reset_vector_store() -> None:
    """Clear the cached store (primarily for tests and reconfiguration)."""
    _cached_vector_store.cache_clear()


__all__ = [
    "build_chunks_from_ocr_payload",
    "build_chunks_from_text",
    "create_vector_store",
    "get_enabled_store_type",
    "hash_identifier",
    "reset_vector_store",
]
