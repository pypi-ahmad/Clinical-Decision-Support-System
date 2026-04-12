from __future__ import annotations

from backend.artifacts import annotate_document, build_artifact_manifest, create_document_workspace
from backend.ocr_backends import OCRBackendConfig, get_ocr_backend
from backend.ocr_backends.ollama_ocr import OllamaOCRBackend


class OCRBackendError(RuntimeError):
    pass


def run_ocr(
    document_path: str,
    page_image_paths: list[str],
    backend: str,
    model: str | None,
    ocr_mode: str = "text",
    use_gpu: bool = True,
    paddle_service_url: str | None = None,
    ocr_api_key: str | None = None,
    service_backend: str = "vllm-server",
    request_timeout_seconds: int = 180,
    healthcheck_timeout_seconds: int = 10,
    artifact_root: str | None = None,
    ollama_client=None,
):
    config = OCRBackendConfig(
        backend=backend,
        model=model,
        ocr_mode=ocr_mode,
        use_gpu=use_gpu,
        service_url=paddle_service_url,
        service_backend=service_backend,
        request_timeout_seconds=request_timeout_seconds,
        healthcheck_timeout_seconds=healthcheck_timeout_seconds,
        api_key=ocr_api_key,
    )
    try:
        backend_impl = get_ocr_backend(config)
        if config.normalized_backend in {"ollama", "glm"} and ollama_client is not None:
            backend_impl = OllamaOCRBackend(ollama_client=ollama_client)
        return backend_impl.run(document_path, page_image_paths, config, artifact_root=artifact_root)
    except Exception as exc:
        raise OCRBackendError(str(exc)) from exc


def materialize_annotations(document_path: str, ocr_result, page_image_paths: list[str]):
    workspace = create_document_workspace(document_path)
    workspace.page_image_paths = list(page_image_paths)
    workspace = annotate_document(page_image_paths, [box.model_dump() for box in ocr_result.bounding_boxes], workspace)
    ocr_result.artifact_manifest = build_artifact_manifest(document_path, workspace)
    return ocr_result