from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field


DEFAULT_OCR_MODELS = {
    "glm": "glm-ocr",
    "paddle": "PaddlePaddle/PaddleOCR-VL-1.5",
}


class OCRBoundingBox(BaseModel):
    page_number: int
    polygon: list[list[float]]
    label: str | None = None
    text: str | None = None
    confidence: float | None = None
    source_ref: str | None = None
    backend_metadata: dict[str, Any] = Field(default_factory=dict)


class OCRPageResult(BaseModel):
    page_number: int
    image_path: str | None = None
    raw_text: str = ""
    markdown: str = ""
    structured_doc: dict[str, Any] | None = None
    bounding_boxes: list[OCRBoundingBox] = Field(default_factory=list)
    confidence: float | None = None
    annotations_metadata: dict[str, Any] = Field(default_factory=dict)


class OCRResult(BaseModel):
    backend: str
    model: str
    ocr_mode: str
    raw_text: str = ""
    markdown: str = ""
    structured_doc: dict[str, Any] | None = None
    per_page_results: list[OCRPageResult] = Field(default_factory=list)
    bounding_boxes: list[OCRBoundingBox] = Field(default_factory=list)
    confidence: float | None = None
    page_images: list[str] = Field(default_factory=list)
    annotations_metadata: dict[str, Any] = Field(default_factory=dict)
    artifact_manifest: dict[str, Any] = Field(default_factory=dict)


class OCRBackendConfig(BaseModel):
    backend: str = "ollama"
    model: str | None = None
    ocr_mode: str = "text"
    use_gpu: bool = True
    service_url: str | None = None
    service_backend: str = "vllm-server"
    request_timeout_seconds: int = 180
    healthcheck_timeout_seconds: int = 10
    api_key: str | None = None

    @property
    def normalized_backend(self) -> str:
        normalized = (self.backend or "glm").strip().lower()
        if normalized in {"glm", "glm-ocr", "ollama-glm", "ollama"}:
            return "glm"
        if normalized in {"paddle", "paddleocr", "paddleocr-vl", "paddleocr-vl-1.5"}:
            return "paddle"
        return "glm"

    @property
    def resolved_model(self) -> str:
        return self.model or DEFAULT_OCR_MODELS[self.normalized_backend]


class BaseOCRBackend(ABC):
    @abstractmethod
    def run(
        self,
        document_path: str,
        page_image_paths: list[str],
        config: OCRBackendConfig,
        artifact_root: str | None = None,
    ) -> OCRResult:
        raise NotImplementedError


def aggregate_page_results(
    backend: str,
    model: str,
    ocr_mode: str,
    page_results: list[OCRPageResult],
    structured_doc: dict[str, Any] | None = None,
) -> OCRResult:
    raw_text_parts = [page.raw_text.strip() for page in page_results if page.raw_text.strip()]
    markdown_parts = [page.markdown.strip() for page in page_results if page.markdown.strip()]
    confidence_values = [page.confidence for page in page_results if page.confidence is not None]
    bounding_boxes = [box for page in page_results for box in page.bounding_boxes]

    page_delimited_text = []
    for page in page_results:
        page_text = page.markdown or page.raw_text
        if page_text.strip():
            page_delimited_text.append(f"PAGE {page.page_number}\n{page_text.strip()}")

    return OCRResult(
        backend=backend,
        model=model,
        ocr_mode=ocr_mode,
        raw_text="\n\n".join(page_delimited_text) if page_delimited_text else "\n\n".join(raw_text_parts),
        markdown="\n\n".join(markdown_parts),
        structured_doc=structured_doc,
        per_page_results=page_results,
        bounding_boxes=bounding_boxes,
        confidence=mean(confidence_values) if confidence_values else None,
        page_images=[page.image_path for page in page_results if page.image_path],
        annotations_metadata={
            "page_count": len(page_results),
            "has_bounding_boxes": bool(bounding_boxes),
            "has_native_confidence": bool(confidence_values),
        },
    )


def collect_bounding_boxes(payload: Any, page_number: int = 1) -> list[OCRBoundingBox]:
    boxes: list[OCRBoundingBox] = []
    _walk_payload_for_boxes(payload, boxes, page_number)
    deduped: list[OCRBoundingBox] = []
    seen: set[tuple[Any, ...]] = set()
    for box in boxes:
        fingerprint = (
            box.page_number,
            tuple(tuple(round(value, 3) for value in point) for point in box.polygon),
            box.label,
            box.text,
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(box)
    return deduped


def _walk_payload_for_boxes(payload: Any, boxes: list[OCRBoundingBox], page_number: int) -> None:
    if isinstance(payload, dict):
        current_page_number = _extract_page_number(payload, page_number)
        polygon = None
        for key in ("polygon", "bbox", "box", "points", "quad", "coordinate"):
            if key in payload:
                polygon = _normalize_polygon(payload[key])
                if polygon:
                    break

        if polygon:
            boxes.append(
                OCRBoundingBox(
                    page_number=current_page_number,
                    polygon=polygon,
                    label=_coerce_optional_text(payload.get("label") or payload.get("type") or payload.get("category")),
                    text=_coerce_optional_text(
                        payload.get("text")
                        or payload.get("content")
                        or payload.get("markdown")
                        or payload.get("label_text")
                    ),
                    confidence=_coerce_optional_float(payload.get("score") or payload.get("confidence")),
                    backend_metadata={"payload_keys": sorted(payload.keys())},
                )
            )

        for value in payload.values():
            _walk_payload_for_boxes(value, boxes, current_page_number)
        return

    if isinstance(payload, list):
        for item in payload:
            _walk_payload_for_boxes(item, boxes, page_number)


def _extract_page_number(payload: dict[str, Any], fallback: int) -> int:
    for key in ("page_number", "page_index", "page_id", "page_no", "page"):
        value = payload.get(key)
        if isinstance(value, int):
            if key in {"page_index", "page_id"}:
                return value + 1 if value >= 0 else fallback
            return value if value > 0 else fallback
        if isinstance(value, str) and value.isdigit():
            numeric = int(value)
            if key in {"page_index", "page_id"}:
                return numeric + 1
            return numeric if numeric > 0 else fallback
    return fallback


def _normalize_polygon(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or not value:
        return None

    if all(isinstance(item, (int, float)) for item in value) and len(value) == 4:
        left, top, right, bottom = [float(item) for item in value]
        return [[left, top], [right, top], [right, bottom], [left, bottom]]

    polygon: list[list[float]] = []
    for point in value:
        if isinstance(point, dict):
            x = point.get("x")
            y = point.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                polygon.append([float(x), float(y)])
        elif isinstance(point, list) and len(point) >= 2:
            x, y = point[0], point[1]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                polygon.append([float(x), float(y)])

    return polygon or None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None