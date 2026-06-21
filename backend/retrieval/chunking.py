"""Text chunking utilities.

The chunker favours **paragraph and sentence boundaries** over a naive
character window so chunks stay semantically coherent. Sizes are configurable
via ``MEDISCAN_CHUNK_SIZE`` / ``MEDISCAN_CHUNK_OVERLAP``.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Any

from backend.retrieval.vector_store import RetrievalChunk

DEFAULT_CHUNK_SIZE = int(os.environ.get("MEDISCAN_CHUNK_SIZE", 1200))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get("MEDISCAN_CHUNK_OVERLAP", 150))

# Progressive separators: paragraph > newline > sentence > space > char.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Prompt-injection neutraliser. Strip common override phrases before we send
# retrieved chunks back into an LLM prompt as "context".
_INJECTION_PATTERNS = [
    re.compile(r"\bignore (?:all )?(?:previous|prior|above) (?:instructions|messages|prompts)\b", re.IGNORECASE),
    re.compile(r"\b(?:system|assistant|user)\s*:", re.IGNORECASE),
    re.compile(r"<<<UNTRUSTED_DOCUMENT_[a-f0-9]+_(?:BEGIN|END)>>>", re.IGNORECASE),
]


def sanitize_retrieved_text(text: str) -> str:
    """Replace likely prompt-injection tokens with a safe marker."""
    cleaned = text
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("[REDACTED_PROMPT_FRAGMENT]", cleaned)
    return cleaned


def _split_by_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        # Last-resort fixed-size slicing
        return [text[i : i + DEFAULT_CHUNK_SIZE] for i in range(0, len(text), DEFAULT_CHUNK_SIZE)]
    parts = text.split(separator)
    return [part + (separator if separator and idx < len(parts) - 1 else "") for idx, part in enumerate(parts)]


def _assemble(parts: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if len(current) + len(part) <= chunk_size:
            current += part
            continue
        if current:
            chunks.append(current)
        # Start next chunk with an overlap tail from the previous chunk.
        tail = current[-chunk_overlap:] if chunk_overlap and current else ""
        current = tail + part
        # Guard against pathological single parts larger than chunk_size.
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[chunk_size - chunk_overlap :]
    if current.strip():
        chunks.append(current)
    return chunks


def split_text_to_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []

    for separator in _SEPARATORS:
        parts = _split_by_separator(normalized, separator)
        if all(len(part) <= chunk_size for part in parts):
            return _assemble(parts, chunk_size, chunk_overlap)

    # Final fallback: sliding window
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
