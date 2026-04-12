from __future__ import annotations

import uuid
from typing import Any

from backend.retrieval.vector_store import RetrievalChunk


def split_text_to_chunks(text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end])
        if end >= len(normalized):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def build_chunks_from_text(
    document_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
    *,
    page_number: int | None = None,
    source_ref: str | None = None,
    section_type: str = "text",
) -> list[RetrievalChunk]:
    payload = metadata.copy() if metadata else {}
    chunks: list[RetrievalChunk] = []
    for chunk_index, chunk_text in enumerate(split_text_to_chunks(text)):
        chunks.append(
            RetrievalChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                text=chunk_text,
                page_number=page_number,
                section_type=section_type,
                patient_id_hash=payload.get("patient_id_hash"),
                encounter_date=payload.get("encounter_date"),
                source_type=payload.get("source_type", "document"),
                source_ref=source_ref,
                ocr_backend=payload.get("ocr_backend"),
                metadata=payload,
                chunk_index=chunk_index,
            )
        )
    return chunks


def build_chunks_from_ocr_payload(
    document_id: str,
    ocr_payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> list[RetrievalChunk]:
    chunks: list[RetrievalChunk] = []
    per_page_results = ocr_payload.get("per_page_results") or []
    if per_page_results:
        for page_result in per_page_results:
            page_number = page_result.get("page_number")
            source_text = (page_result.get("markdown") or page_result.get("raw_text") or "").strip()
            if not source_text:
                continue
            chunks.extend(
                build_chunks_from_text(
                    document_id,
                    source_text,
                    metadata,
                    page_number=page_number,
                    source_ref=page_result.get("image_path"),
                    section_type="ocr_page",
                )
            )
        if chunks:
            return chunks

    candidate_text = (ocr_payload.get("markdown") or ocr_payload.get("raw_text") or "").strip()
    if candidate_text:
        return build_chunks_from_text(document_id, candidate_text, metadata, section_type="ocr_document")
    return []