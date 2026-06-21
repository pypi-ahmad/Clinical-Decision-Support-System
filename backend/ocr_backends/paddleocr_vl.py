"""PaddleOCR-VL backend with a process-wide pipeline cache.

The PaddleOCRVL pipeline loads several hundred megabytes of weights and GPU
memory on first use. Rebuilding it per request would make latency prohibitive,
so we cache the pipeline per (device, config) key.
"""

from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.ocr_backends.base import (
    BaseOCRBackend,
    OCRBackendConfig,
    OCRPageResult,
    aggregate_page_results,
    collect_bounding_boxes,
)
from backend.ocr_backends.service_client import (
    PaddleOCRVLServiceClient,
    PaddleOCRVLServiceError,
    PaddleOCRVLServiceSettings,
)

try:
    from paddleocr import PaddleOCRVL
except Exception:  # pragma: no cover - optional runtime dependency
    PaddleOCRVL = None


@lru_cache(maxsize=2)
def _cached_local_pipeline(device: str, use_doc_orientation: bool, use_doc_unwarp: bool, use_layout: bool):
    if PaddleOCRVL is None:
        raise PaddleOCRVLServiceError(
            "PaddleOCR-VL is not installed. Install paddleocr[doc-parser] and a compatible PaddlePaddle runtime."
        )
    return PaddleOCRVL(
        device=device,
        use_doc_orientation_classify=use_doc_orientation,
        use_doc_unwarping=use_doc_unwarp,
        use_layout_detection=use_layout,
    )


class PaddleOCRVLBackend(BaseOCRBackend):
    def run(
        self,
        document_path: str,
        page_image_paths: list[str],
        config: OCRBackendConfig,
        artifact_root: str | None = None,
    ):
        page_objects, pipeline = _run_paddle_pipeline(document_path, config)
        page_results, page_payloads, page_markdowns = _serialize_page_results(
            page_objects,
            page_image_paths,
            artifact_root,
        )
        merged_payloads, merged_markdowns = _serialize_merged_results(pipeline, page_objects, artifact_root)

        structured_doc = {"pages": page_payloads}
        if merged_payloads:
            structured_doc["merged"] = merged_payloads

        result = aggregate_page_results(
            backend="paddle",
            model=config.resolved_model,
            ocr_mode=config.ocr_mode,
            page_results=page_results,
            structured_doc=structured_doc,
        )
        if merged_markdowns:
            result.markdown = "\n\n".join(chunk for chunk in merged_markdowns if chunk)
            result.raw_text = result.markdown or result.raw_text
        elif page_markdowns:
            result.markdown = "\n\n".join(page_markdowns)
            result.raw_text = result.markdown or result.raw_text
        return result


def _build_local_pipeline(config: OCRBackendConfig):
    """Return a cached local PaddleOCRVL instance (build-once-per-process)."""
    device = "gpu" if config.use_gpu else "cpu"
    return _cached_local_pipeline(device, True, True, True)


def _run_paddle_pipeline(document_path: str, config: OCRBackendConfig):
    if config.service_url:
        client = PaddleOCRVLServiceClient(
            PaddleOCRVLServiceSettings(
                model_name=config.resolved_model,
                service_url=config.service_url,
                service_backend=config.service_backend,
                api_key=config.api_key,
                use_gpu=config.use_gpu,
                healthcheck_timeout_seconds=config.healthcheck_timeout_seconds,
                request_timeout_seconds=config.request_timeout_seconds,
            )
        )
        return client.predict(document_path)

    if PaddleOCRVL is None:
        raise PaddleOCRVLServiceError(
            "PaddleOCR-VL is not installed. Install paddleocr[doc-parser] and a compatible PaddlePaddle runtime."
        )

    pipeline = _build_local_pipeline(config)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: list(pipeline.predict(document_path)))
        try:
            output = future.result(timeout=config.request_timeout_seconds)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise PaddleOCRVLServiceError(
                f"PaddleOCR-VL local Python inference timed out after {config.request_timeout_seconds} seconds"
            ) from exc
    return output, pipeline


def _serialize_page_results(page_objects, page_image_paths: list[str], artifact_root: str | None):
    page_results: list[OCRPageResult] = []
    payloads: list[dict[str, Any]] = []
    markdown_chunks: list[str] = []

    with tempfile.TemporaryDirectory(prefix="paddleocr_vl_") as temp_dir:
        output_root = Path(artifact_root or temp_dir) / "raw_paddle_pages"
        output_root.mkdir(parents=True, exist_ok=True)
        for page_number, page_object in enumerate(page_objects, start=1):
            page_dir = output_root / f"page_{page_number:04d}"
            page_dir.mkdir(parents=True, exist_ok=True)
            page_payload, page_markdown = _save_result_payloads(page_object, page_dir)
            payloads.append(page_payload or {})
            if page_markdown:
                markdown_chunks.append(page_markdown)
            boxes = collect_bounding_boxes(page_payload or {}, page_number=page_number)
            confidence_values = [box.confidence for box in boxes if box.confidence is not None]
            page_results.append(
                OCRPageResult(
                    page_number=page_number,
                    image_path=page_image_paths[page_number - 1] if page_number - 1 < len(page_image_paths) else None,
                    raw_text=page_markdown or json.dumps(page_payload or {}, ensure_ascii=False),
                    markdown=page_markdown or "",
                    structured_doc=page_payload,
                    bounding_boxes=boxes,
                    confidence=sum(confidence_values) / len(confidence_values) if confidence_values else None,
                    annotations_metadata={"page_dir": str(page_dir)},
                )
            )

    return page_results, payloads, markdown_chunks


def _serialize_merged_results(pipeline, page_objects, artifact_root: str | None):
    if not hasattr(pipeline, "restructure_pages") or len(page_objects) <= 1:
        return [], []

    try:
        merged_objects = list(pipeline.restructure_pages(page_objects, merge_tables=True, relevel_titles=True))
    except Exception:
        return [], []

    merged_payloads: list[dict[str, Any]] = []
    merged_markdowns: list[str] = []
    with tempfile.TemporaryDirectory(prefix="paddleocr_vl_merged_") as temp_dir:
        output_root = Path(artifact_root or temp_dir) / "merged_paddle"
        output_root.mkdir(parents=True, exist_ok=True)
        for merged_index, merged_object in enumerate(merged_objects, start=1):
            merged_dir = output_root / f"merged_{merged_index:04d}"
            merged_dir.mkdir(parents=True, exist_ok=True)
            merged_payload, merged_markdown = _save_result_payloads(merged_object, merged_dir)
            merged_payloads.append(merged_payload or {})
            if merged_markdown:
                merged_markdowns.append(merged_markdown)
    return merged_payloads, merged_markdowns


def _save_result_payloads(result_object, output_dir: Path):
    if hasattr(result_object, "save_to_json"):
        result_object.save_to_json(save_path=str(output_dir))
    if hasattr(result_object, "save_to_markdown"):
        result_object.save_to_markdown(save_path=str(output_dir))

    json_payload = None
    markdown_text = None
    json_files = sorted(output_dir.rglob("*.json"))
    markdown_files = sorted(output_dir.rglob("*.md"))

    if json_files:
        json_payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    if markdown_files:
        markdown_text = markdown_files[0].read_text(encoding="utf-8")
    return json_payload, markdown_text
