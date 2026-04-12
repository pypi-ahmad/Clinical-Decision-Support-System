from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    from paddleocr import PaddleOCRVL
except Exception:
    PaddleOCRVL = None


WINDOWS_FRIENDLY_DEPLOYMENTS = {
    "local-python": {
        "summary": "Runs PaddleOCR-VL directly in Python. Best when PaddleOCR and CUDA are available in the local environment.",
        "host_os": "Windows native Python",
        "gpu": True,
    },
    "docker-vllm": {
        "summary": "Run the PaddleOCR-VL service in Docker or WSL2 with vLLM, then connect to it from the repo.",
        "host_os": "Windows + Docker Desktop or WSL2",
        "gpu": True,
    },
    "llama-cpp-service": {
        "summary": "Use the published PaddleOCR-VL GGUF files behind a llama.cpp server for a lightweight local service path.",
        "host_os": "Windows native or WSL2",
        "gpu": True,
    },
}


@dataclass
class PaddleOCRVLServiceConfig:
    model_name: str = "PaddlePaddle/PaddleOCR-VL-1.5"
    use_gpu: bool = True
    use_local_service: bool = False
    service_url: str | None = None
    service_backend: str = "vllm-server"
    api_key: str | None = None
    use_doc_orientation_classify: bool = True
    use_doc_unwarping: bool = True
    use_layout_detection: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_paddleocr_vl_client(config: PaddleOCRVLServiceConfig):
    if PaddleOCRVL is None:
        raise RuntimeError(
            "PaddleOCR-VL is not installed. Install paddleocr[doc-parser] and a compatible PaddlePaddle runtime."
        )

    device = "gpu" if config.use_gpu else "cpu"
    client_kwargs: dict[str, Any] = {
        "device": device,
        "use_doc_orientation_classify": config.use_doc_orientation_classify,
        "use_doc_unwarping": config.use_doc_unwarping,
        "use_layout_detection": config.use_layout_detection,
    }

    if config.use_local_service and config.service_url:
        client_kwargs.update(
            {
                "vl_rec_backend": config.service_backend,
                "vl_rec_server_url": config.service_url,
                "vl_rec_api_model_name": config.model_name,
            }
        )
        if config.api_key:
            client_kwargs["vl_rec_api_key"] = config.api_key

    return PaddleOCRVL(**client_kwargs)


def run_paddleocr_vl_prediction(input_path: str, config: PaddleOCRVLServiceConfig) -> list[Any]:
    client = create_paddleocr_vl_client(config)
    return list(client.predict(input_path))