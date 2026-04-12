from __future__ import annotations

import os
from typing import Any

from backend.retrieval.embeddings import embed_texts
from backend.retrieval.vector_store import RetrievalChunk, VectorStoreBackend

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
except Exception:
    QdrantClient = None
    Distance = None
    FieldCondition = None
    Filter = None
    MatchValue = None
    PointStruct = None
    VectorParams = None


DEFAULT_QDRANT_COLLECTION = "medical_documents"


class QdrantRetrievalStore(VectorStoreBackend):
    def __init__(self, collection_name: str | None = None):
        if QdrantClient is None:
            raise RuntimeError("qdrant-client is not installed")
        self.collection_name = collection_name or DEFAULT_QDRANT_COLLECTION
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=10,
        )

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
        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            points.append(PointStruct(id=chunk.chunk_id, vector=embedding, payload=chunk.to_payload()))

        self.client.upsert(collection_name=self.collection_name, points=points)
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

        hits = self.client.search(
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
            for hit in hits
        ]

    def _ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )