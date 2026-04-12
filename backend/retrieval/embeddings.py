from __future__ import annotations

import os

import ollama


DEFAULT_EMBED_MODEL = "nomic-embed-text"


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    if not texts:
        return []

    embed_model = model or os.getenv("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    if hasattr(ollama, "embed"):
        response = ollama.embed(model=embed_model, input=texts)
        return response.get("embeddings", [])

    embeddings: list[list[float]] = []
    for text in texts:
        response = ollama.embeddings(model=embed_model, prompt=text)
        embeddings.append(response["embedding"])
    return embeddings