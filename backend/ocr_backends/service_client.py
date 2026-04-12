from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any

import requests

try:
    from paddleocr import PaddleOCRVL
except Exception:
    PaddleOCRVL = None


class PaddleOCRVLServiceError(RuntimeError):
    pass


@dataclass
class PaddleOCRVLServiceSettings:
    model_name: str
    service_url: str
    service_backend: str = "vllm-server"
    api_key: str | None = None
    use_gpu: bool = True
    healthcheck_timeout_seconds: int = 10
    request_timeout_seconds: int = 180
    use_doc_orientation_classify: bool = True
    use_doc_unwarping: bool = True
    use_layout_detection: bool = True


class PaddleOCRVLServiceClient:
    def __init__(self, settings: PaddleOCRVLServiceSettings):
        self.settings = settings

    def healthcheck(self) -> dict[str, Any]:
        candidate_urls = _candidate_healthcheck_urls(self.settings.service_url)
        last_error = "No candidate service URL responded"
        for url in candidate_urls:
            try:
                response = requests.get(url, timeout=self.settings.healthcheck_timeout_seconds)
                if response.status_code < 500:
                    return {"healthy": True, "url": url, "status_code": response.status_code}
                last_error = f"Healthcheck failed with status {response.status_code} at {url}"
            except requests.RequestException as exc:
                last_error = str(exc)

        raise PaddleOCRVLServiceError(f"PaddleOCR-VL service healthcheck failed: {last_error}")

    def build_pipeline(self):
        if PaddleOCRVL is None:
            raise PaddleOCRVLServiceError(
                "PaddleOCR-VL is not installed. Install paddleocr[doc-parser] to use service mode."
            )

        self.healthcheck()
        pipeline_kwargs: dict[str, Any] = {
            "device": "gpu" if self.settings.use_gpu else "cpu",
            "vl_rec_backend": self.settings.service_backend,
            "vl_rec_server_url": self.settings.service_url,
            "vl_rec_api_model_name": self.settings.model_name,
            "use_doc_orientation_classify": self.settings.use_doc_orientation_classify,
            "use_doc_unwarping": self.settings.use_doc_unwarping,
            "use_layout_detection": self.settings.use_layout_detection,
        }
        if self.settings.api_key:
            pipeline_kwargs["vl_rec_api_key"] = self.settings.api_key
        return PaddleOCRVL(**pipeline_kwargs)

    def predict(self, input_path: str):
        pipeline = self.build_pipeline()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: list(pipeline.predict(input_path)))
            try:
                output = future.result(timeout=self.settings.request_timeout_seconds)
            except FuturesTimeoutError as exc:
                future.cancel()
                raise PaddleOCRVLServiceError(
                    f"PaddleOCR-VL service timed out after {self.settings.request_timeout_seconds} seconds"
                ) from exc
        return output, pipeline


def _candidate_healthcheck_urls(service_url: str) -> list[str]:
    base = service_url.rstrip("/")
    candidates = [base]
    if base.endswith("/v1"):
        candidates.append(f"{base}/models")
        candidates.append(f"{base[:-3]}/health")
        candidates.append(f"{base[:-3]}/docs")
    else:
        candidates.append(f"{base}/health")
        candidates.append(f"{base}/v1/models")
        candidates.append(f"{base}/docs")
    seen: set[str] = set()
    deduped: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped