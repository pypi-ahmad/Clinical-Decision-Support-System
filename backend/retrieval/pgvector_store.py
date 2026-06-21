from __future__ import annotations

import os
from typing import Any

from backend.retrieval.vector_store import RetrievalChunk, VectorStoreBackend


class PgvectorRetrievalStore(VectorStoreBackend):
    _SCAFFOLD = True  # Not yet wired; prevents auto-selection by the factory

    def __init__(self, collection_name: str | None = None):
        self.collection_name = collection_name or os.getenv("PGVECTOR_TABLE", "medical_documents")
        self.connection_url = os.getenv("PGVECTOR_URL")
        if not self.connection_url:
            raise RuntimeError("PGVECTOR_URL is not configured")

    @staticmethod
    def is_configured() -> bool:
        # pgvector support is scaffolded; always report unconfigured until wired.
        return False

    def upsert_chunks(self, chunks: list[RetrievalChunk]) -> dict[str, Any]:
        raise RuntimeError(
            "pgvector support is scaffolded but not wired in this repo yet. Configure Qdrant for active retrieval today."
        )

    def search(self, query: str, limit: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        raise RuntimeError(
            "pgvector support is scaffolded but not wired in this repo yet. Configure Qdrant for active retrieval today."
        )
