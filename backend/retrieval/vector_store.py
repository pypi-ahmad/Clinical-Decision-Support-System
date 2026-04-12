from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class RetrievalChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page_number: int | None = None
    section_type: str = "text"
    patient_id_hash: str | None = None
    encounter_date: str | None = None
    source_type: str = "document"
    source_ref: str | None = None
    ocr_backend: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_index: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "page_number": self.page_number,
            "section_type": self.section_type,
            "patient_id_hash": self.patient_id_hash,
            "encounter_date": self.encounter_date,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "ocr_backend": self.ocr_backend,
            "chunk_index": self.chunk_index,
        }


class VectorStoreBackend(ABC):
    @abstractmethod
    def upsert_chunks(self, chunks: list[RetrievalChunk]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, limit: int = 5, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError