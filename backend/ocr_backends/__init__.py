from backend.ocr_backends.base import OCRBackendConfig, OCRBoundingBox, OCRPageResult, OCRResult
from backend.ocr_backends.ollama_ocr import OllamaOCRBackend
from backend.ocr_backends.paddleocr_vl import PaddleOCRVLBackend


def get_ocr_backend(config: OCRBackendConfig):
    if config.normalized_backend in {"ollama", "glm"}:
        return OllamaOCRBackend()
    if config.normalized_backend == "paddle":
        return PaddleOCRVLBackend()
    raise ValueError(f"Unsupported OCR backend: {config.backend}")


__all__ = [
    "OCRBackendConfig",
    "OCRBoundingBox",
    "OCRPageResult",
    "OCRResult",
    "OllamaOCRBackend",
    "PaddleOCRVLBackend",
    "get_ocr_backend",
]