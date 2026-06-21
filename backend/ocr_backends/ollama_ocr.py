"""Ollama-backed OCR implementation.

Pages are processed concurrently (bounded by a semaphore) to hide the
per-request latency of the Ollama chat endpoint.  The concurrency cap defaults
to 2 because local Ollama servers get overwhelmed quickly; tune via the
``MEDISCAN_OLLAMA_CONCURRENCY`` env var.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import ollama

from backend.ocr_backends.base import BaseOCRBackend, OCRBackendConfig, OCRPageResult, aggregate_page_results


GLM_PROMPTS = {
    "text": "Text Recognition:",
    "ocr": "Text Recognition:",
    "table": "Table Recognition:",
    "figure": "Figure Recognition:",
    "chart": "Figure Recognition:",
    "formula": "Formula Recognition:",
}


def _concurrency() -> int:
    try:
        return max(1, int(os.environ.get("MEDISCAN_OLLAMA_CONCURRENCY", 2)))
    except ValueError:
        return 2


class OllamaOCRBackend(BaseOCRBackend):
    def __init__(self, ollama_client: Any | None = None):
        self.ollama_client = ollama_client or ollama

    def run(
        self,
        document_path: str,
        page_image_paths: list[str],
        config: OCRBackendConfig,
        artifact_root: str | None = None,
    ):
        prompt = _build_prompt(config)
        page_results: list[OCRPageResult] = []

        # Prefer async + gather when an AsyncClient is available; otherwise run
        # sequentially (e.g. for tests that inject a synchronous fake).
        if _can_use_async(self.ollama_client):
            page_results = asyncio.run(
                _run_pages_async(
                    config.resolved_model,
                    prompt,
                    page_image_paths,
                    _concurrency(),
                )
            )
        else:
            for page_number, image_path in enumerate(page_image_paths, start=1):
                response = self.ollama_client.chat(
                    model=config.resolved_model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_path],
                        }
                    ],
                    options={"temperature": 0},
                )
                page_text = response["message"]["content"]
                page_results.append(
                    OCRPageResult(
                        page_number=page_number,
                        image_path=image_path,
                        raw_text=page_text,
                        markdown=page_text,
                    )
                )

        return aggregate_page_results(
            backend=config.normalized_backend,
            model=config.resolved_model,
            ocr_mode=config.ocr_mode,
            page_results=page_results,
        )


def _can_use_async(client: Any) -> bool:
    """Only use AsyncClient when we know we're talking to the real SDK."""
    return client is ollama and hasattr(ollama, "AsyncClient")


async def _run_pages_async(
    model: str,
    prompt: str,
    page_image_paths: list[str],
    concurrency: int,
) -> list[OCRPageResult]:
    client = ollama.AsyncClient()
    semaphore = asyncio.Semaphore(concurrency)

    async def _page(idx: int, image_path: str) -> OCRPageResult:
        async with semaphore:
            response = await client.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_path],
                    }
                ],
                options={"temperature": 0},
            )
            page_text = response["message"]["content"]
            return OCRPageResult(
                page_number=idx,
                image_path=image_path,
                raw_text=page_text,
                markdown=page_text,
            )

    tasks = [
        _page(idx, path)
        for idx, path in enumerate(page_image_paths, start=1)
    ]
    return list(await asyncio.gather(*tasks))


def _build_prompt(config: OCRBackendConfig) -> str:
    prompt_key = (config.ocr_mode or "text").strip().lower()
    return GLM_PROMPTS.get(prompt_key, GLM_PROMPTS["text"])
