"""Ollama embedding adapter with a module-level client cache."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import ollama

DEFAULT_EMBED_MODEL = "nomic-embed-text"


@lru_cache(maxsize=1)
def _default_client() -> Any:
    """Return the module-level Ollama client.

    Kept in a function so tests can monkeypatch ``ollama`` without mutating
    a module-level global. ``lru_cache`` de-duplicates repeated construction.
    """
    return ollama


def embed_texts(
    texts: list[str],
    model: str | None = None,
    *,
    client: Any | None = None,
) -> list[list[float]]:
    """Embed a batch of texts via Ollama.

    Uses the batch ``embed`` endpoint when available (Ollama >=0.4) and falls
    back to the legacy ``embeddings`` call otherwise.
    """
    if not texts:
        return []

    embed_model = model or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    active = client or _default_client()

    if hasattr(active, "embed"):
        response = active.embed(model=embed_model, input=texts)
        # The SDK returns either a dict with 'embeddings' or an object with
        # ``.embeddings`` — handle both.
        if isinstance(response, dict):
            return list(response.get("embeddings", []))
        return list(getattr(response, "embeddings", []))

    embeddings: list[list[float]] = []
    for text in texts:
        response = active.embeddings(model=embed_model, prompt=text)
        if isinstance(response, dict):
            embeddings.append(response["embedding"])
        else:
            embeddings.append(response.embedding)
    return embeddings
