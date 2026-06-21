"""Qdrant-backed retrieval store.

Notes for reviewers:

* ``search`` is removed in Qdrant >=1.18; we use ``query_points``.
* A single ``QdrantClient`` is constructed at store-init time and reused for
  the lifetime of the process (see ``backend.retrieval.create_vector_store``).
* When ``QDRANT_URL`` is non-local the client is forced to require an API key
  — a misconfigured Qdrant that allows anonymous access would otherwise leak
  the entire PHI vector store.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from backend.logging_config import get_logger
from backend.retrieval.embeddings import embed_texts
from backend.retrieval.vector_store import RetrievalChunk, VectorStoreBackend

_logger = get_logger(__name__)


try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )
except Exception:  # pragma: no cover - optional runtime dependency
    QdrantClient = None
    Distance = None
    FieldCondition = None
    Filter = None
    MatchValue = None
    PointStruct = None
    VectorParams = None


DEFAULT_QDRANT_COLLECTION = "medical_documents"


def _is_local_host(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", ""}


class QdrantRetrievalStore(VectorStoreBackend):
    def __init__(self, collection_name: str | None = None):
        if QdrantClient is None:
            raise RuntimeError("qdrant-client is not installed")
        self.collection_name = collection_name or DEFAULT_QDRANT_COLLECTION
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")
        if not _is_local_host(url) and not api_key:
            raise RuntimeError(
                "Remote Qdrant URL configured without QDRANT_API_KEY; refusing to "
                "connect anonymously to an off-host vector store."
            )
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=int(os.getenv("QDRANT_TIMEOUT", 10)),
        )
        self._collection_ready = False

    @staticmethod
    def is_configured() -> bool:
        enabled_flag = os.getenv("QDRANT_ENABLED", "").strip().lower()
        return enabled_flag in {"1", "true", "yes"} or bool(os.getenv("QDRANT_URL"))

    def upsert_chunks(self, chunks: list[RetrievalChunk]) -> dict[str, Any]:
        if not chunks:
            return {"indexed": False, "reason": "No chunks available for indexing"}

        embeddings = embed_texts([chunk.text for chunk in chunks])
        if not embeddings:
            return {"indexed": False, "reason": "Embedding generation returned no vectors"}

        self._ensure_collection(len(embeddings[0]))
        points = [
            PointStruct(id=chunk.chunk_id, vector=embedding, payload=chunk.to_payload())
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)
        _logger.info("qdrant_upsert", chunks=len(points), collection=self.collection_name)
        return {
            "indexed": True,
            "chunks": len(points),
            "collection": self.collection_name,
            "store": "qdrant",
        }

    def search(self, query: str, limit: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        embeddings = embed_texts([query])
        if not embeddings:
            return []

        query_filter = None
        if filters:
            must = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filters.items()
                if value is not None
            ]
            if must:
                query_filter = Filter(must=must)

        # ``query_points`` is the modern API. The older ``search`` is being
        # removed in qdrant-server 1.18. We prefer ``query_points`` and fall
        # back to ``search`` only if running against an older client binding.
        response = self._query_points(
            collection_name=self.collection_name,
            query_vector=embeddings[0],
            query_filter=query_filter,
            limit=limit,
        )
        return [
            {
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "page_number": hit.payload.get("page_number"),
                "source_ref": hit.payload.get("source_ref"),
                "payload": hit.payload,
            }
            for hit in response
        ]

    # -- Helpers -------------------------------------------------------------
    def _query_points(self, *, collection_name, query_vector, query_filter, limit):
        if hasattr(self.client, "query_points"):
            result = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            # query_points returns an object with ``.points``; normalise.
            return getattr(result, "points", result)
        # Legacy fallback. qdrant-client <1.10 still exposes ``search``.
        return self.client.search(  # type: ignore[attr-defined]
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
        )

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_ready:
            return
        if self.client.collection_exists(self.collection_name):
            self._collection_ready = True
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        self._collection_ready = True
