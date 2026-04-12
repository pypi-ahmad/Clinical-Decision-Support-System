from __future__ import annotations

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

DEEPSEEK_PROMPTS = {
    "text": "Transcribe this medical document text exactly.",
    "ocr": "Transcribe this medical document text exactly.",
    "table": "Extract the table contents from this medical document exactly.",
    "figure": "Describe the figure or chart contents from this medical document exactly.",
    "chart": "Describe the figure or chart contents from this medical document exactly.",
    "formula": "Extract the formula or structured notation from this document exactly.",
}


class OllamaOCRBackend(BaseOCRBackend):
    def __init__(self, ollama_client=None):
        self.ollama_client = ollama_client or ollama

    def run(
        self,
        document_path: str,
        page_image_paths: list[str],
        config: OCRBackendConfig,
        artifact_root: str | None = None,
    ):
        page_results: list[OCRPageResult] = []
        for page_number, image_path in enumerate(page_image_paths, start=1):
            response = self.ollama_client.chat(
                model=config.resolved_model,
                messages=[
                    {
                        "role": "user",
                        "content": _build_prompt(config),
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
            backend="glm" if config.normalized_backend == "glm" else "ollama",
            model=config.resolved_model,
            ocr_mode=config.ocr_mode,
            page_results=page_results,
        )


def _build_prompt(config: OCRBackendConfig) -> str:
    prompt_key = (config.ocr_mode or "text").strip().lower()
    if config.normalized_backend == "glm":
        return GLM_PROMPTS.get(prompt_key, GLM_PROMPTS["text"])
    return DEEPSEEK_PROMPTS.get(prompt_key, DEEPSEEK_PROMPTS["text"])