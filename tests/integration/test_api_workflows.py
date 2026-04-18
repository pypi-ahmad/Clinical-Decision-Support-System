import io

import pytest
from fastapi.testclient import TestClient

import backend.main as main


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def upload_root(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDISCAN_UPLOAD_ROOT", str(tmp_path))
    yield tmp_path


def _valid_pdf_bytes() -> bytes:
    return b"%PDF-1.4\n%fake\n"


def _valid_png_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\0" * 32


def _valid_jpeg_bytes() -> bytes:
    return b"\xff\xd8\xff\xe0" + b"\0" * 32


# ---------------------------------------------------------------------------
# Direct pipeline tests
# ---------------------------------------------------------------------------


def test_analyze_endpoint_workflow_success(monkeypatch, upload_root):
    monkeypatch.setattr(
        main,
        "process_document_pipeline",
        lambda *args, **kwargs: {"patient": {"mrn": "MRN9"}, "clinical": {}},
    )
    monkeypatch.setattr(main, "_load_history", lambda mrn: {"patient": {"mrn": mrn}})
    monkeypatch.setattr(main, "create_vector_store", lambda: None)
    monkeypatch.setattr(main, "analyze_medical_logic", lambda *args, **kwargs: {"summary": "done", "alerts": []})

    client = TestClient(main.app)
    files = {"file": ("scan.pdf", io.BytesIO(_valid_pdf_bytes()), "application/pdf")}
    data = {"provider": "Ollama", "model": "glm-4.7-flash"}

    response = client.post("/analyze", files=files, data=data)
    body = response.json()

    assert response.status_code == 200
    assert body["extracted"]["patient"]["mrn"] == "MRN9"
    assert body["history_available"] is True
    assert body["analysis"]["summary"] == "done"


def test_analyze_endpoint_returns_500_on_pipeline_error(monkeypatch, upload_root):
    monkeypatch.setattr(main, "process_document_pipeline", lambda *args, **kwargs: {"error": "pipeline failed"})

    client = TestClient(main.app)
    files = {"file": ("scan.pdf", io.BytesIO(_valid_pdf_bytes()), "application/pdf")}
    response = client.post("/analyze", files=files, data={"provider": "Ollama", "model": "m"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["error"] == "Request failed"
    assert "correlation_id" in detail


# ---------------------------------------------------------------------------
# Extraction graph mode tests
# ---------------------------------------------------------------------------


def test_analyze_extraction_graph_mode_success(monkeypatch, upload_root):
    monkeypatch.setattr(
        main,
        "run_extraction_graph",
        lambda **kwargs: {
            "structured_data": {"patient": {"mrn": "GRAPH-1"}, "clinical": {}},
            "ocr": {"raw_text": "graph ocr", "bounding_boxes": [], "artifact_manifest": {}},
            "past_data": None,
            "analysis": {"summary": "graph done", "alerts": [], "trends": []},
            "requires_human_review": True,
            "vector_index_status": {"indexed": True},
            "error": None,
        },
    )

    client = TestClient(main.app)
    files = {"file": ("scan.pdf", io.BytesIO(_valid_pdf_bytes()), "application/pdf")}
    data = {"extraction_graph_mode": "true", "agentic_mode": "false"}

    response = client.post("/analyze", files=files, data=data)
    body = response.json()

    assert response.status_code == 200
    assert body["extracted"]["patient"]["mrn"] == "GRAPH-1"
    assert body["analysis"]["summary"] == "graph done"
    assert body["requires_human_review"] is True
    assert body["vector_index_status"]["indexed"] is True


def test_analyze_extraction_graph_mode_error(monkeypatch, upload_root):
    monkeypatch.setattr(
        main,
        "run_extraction_graph",
        lambda **kwargs: {"error": "graph node failed", "structured_data": {}},
    )

    client = TestClient(main.app)
    files = {"file": ("scan.pdf", io.BytesIO(_valid_pdf_bytes()), "application/pdf")}
    data = {"extraction_graph_mode": "true"}

    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 500
    assert response.json()["detail"]["error"] == "Request failed"


# ---------------------------------------------------------------------------
# Agentic workflow mode tests
# ---------------------------------------------------------------------------


def test_analyze_agentic_mode_success(monkeypatch, upload_root):
    monkeypatch.setattr(
        main,
        "run_agentic_extraction_workflow",
        lambda **kwargs: {
            "structured_data": {"patient": {"mrn": "AGENT-1"}, "clinical": {}},
            "ocr": {"raw_text": "agent ocr", "bounding_boxes": [], "artifact_manifest": {}},
            "past_data": {"patient": {"mrn": "AGENT-1"}},
            "analysis": {"summary": "agent done", "alerts": [], "trends": []},
            "requires_human_review": False,
            "vector_index_status": {"indexed": False, "reason": "no store"},
            "error": None,
        },
    )

    client = TestClient(main.app)
    files = {"file": ("report.jpg", io.BytesIO(_valid_jpeg_bytes()), "image/jpeg")}
    data = {"agentic_mode": "true", "extraction_graph_mode": "false"}

    response = client.post("/analyze", files=files, data=data)
    body = response.json()

    assert response.status_code == 200
    assert body["extracted"]["patient"]["mrn"] == "AGENT-1"
    assert body["analysis"]["summary"] == "agent done"
    assert body["history_available"] is True
    assert body["requires_human_review"] is False


def test_analyze_agentic_mode_error(monkeypatch, upload_root):
    monkeypatch.setattr(
        main,
        "run_agentic_extraction_workflow",
        lambda **kwargs: {"error": "agentic failure"},
    )

    client = TestClient(main.app)
    files = {"file": ("scan.pdf", io.BytesIO(_valid_pdf_bytes()), "application/pdf")}
    data = {"agentic_mode": "true"}

    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# OCR backend selection tests
# ---------------------------------------------------------------------------


def test_analyze_passes_ocr_backend_and_model_to_pipeline(monkeypatch, upload_root):
    captured = {}

    def fake_pipeline(file_path, provider, model, api_key, **kwargs):
        captured.update(kwargs)
        return {
            "structured_data": {"patient": {"mrn": "OCR-1"}, "clinical": {}},
            "ocr": {"raw_text": "text", "backend": "glm", "bounding_boxes": [], "artifact_manifest": {}},
        }

    monkeypatch.setattr(main, "process_document_pipeline", fake_pipeline)
    monkeypatch.setattr(main, "_load_history", lambda mrn: None)
    monkeypatch.setattr(main, "create_vector_store", lambda: None)
    monkeypatch.setattr(main, "analyze_medical_logic", lambda *a, **kw: {"summary": "ok", "alerts": [], "trends": []})

    client = TestClient(main.app)
    files = {"file": ("doc.png", io.BytesIO(_valid_png_bytes()), "image/png")}
    data = {
        "ocr_backend": "glm",
        "ocr_model": "glm-ocr",
        "ocr_mode": "table",
        "use_gpu": "true",
    }

    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 200

    assert captured.get("ocr_backend") == "glm"
    assert captured.get("ocr_model") == "glm-ocr"
    assert captured.get("ocr_prompt_mode") == "table"


# ---------------------------------------------------------------------------
# Insurance endpoint tests
# ---------------------------------------------------------------------------


def test_check_insurance_endpoint_passes_decoded_text_and_json(monkeypatch, upload_root):
    captured = {"medical": None, "policy": None}

    def fake_check(medical_data, policy_text, *args, **kwargs):
        captured["medical"] = medical_data
        captured["policy"] = policy_text
        return {"eligible": True, "confidence": "High", "reasoning": "ok", "missing_info": []}

    monkeypatch.setattr(main, "check_insurance_coverage", fake_check)
    monkeypatch.setattr(main, "create_vector_store", lambda: None)

    client = TestClient(main.app)
    files = {"policy_file": ("policy.txt", io.BytesIO(b"policy body"), "text/plain")}
    data = {"medical_json": '{"patient": {"mrn": "M2"}}'}
    response = client.post("/check_insurance", files=files, data=data)

    assert response.status_code == 200
    assert response.json()["eligible"] is True
    assert captured["medical"]["patient"]["mrn"] == "M2"
    assert captured["policy"] == "policy body"


def test_check_insurance_with_policy_ocr(monkeypatch, upload_root):
    monkeypatch.setattr(
        main,
        "run_document_ocr",
        lambda path, **kw: {"markdown": "OCR extracted policy text", "raw_text": "OCR extracted policy text"},
    )

    captured = {}

    def fake_check(medical_data, policy_text, *args, **kwargs):
        captured["policy"] = policy_text
        return {"eligible": False, "reasoning": "not covered", "missing_info": ["authorization"]}

    monkeypatch.setattr(main, "check_insurance_coverage", fake_check)
    monkeypatch.setattr(main, "create_vector_store", lambda: None)

    client = TestClient(main.app)
    files = {"policy_file": ("policy.pdf", io.BytesIO(b"%PDF-1.4 binary"), "application/pdf")}
    data = {
        "medical_json": '{"clinical": {"diagnosis_list": ["HTN"]}}',
        "policy_ocr": "true",
        "ocr_backend": "glm",
        "ocr_model": "glm-ocr",
    }

    response = client.post("/check_insurance", files=files, data=data)
    assert response.status_code == 200
    assert captured["policy"] == "OCR extracted policy text"


def test_check_insurance_with_reasoning_provider_split(monkeypatch, upload_root):
    captured = {}

    def fake_check(medical_data, policy_text, provider, model, api_key, **kwargs):
        captured["provider"] = provider
        captured["model"] = model
        return {"eligible": True, "reasoning": "covered", "missing_info": []}

    monkeypatch.setattr(main, "check_insurance_coverage", fake_check)
    monkeypatch.setattr(main, "create_vector_store", lambda: None)
    monkeypatch.setenv("MEDISCAN_ALLOW_USER_API_KEYS", "1")

    client = TestClient(main.app)
    files = {"policy_file": ("policy.txt", io.BytesIO(b"text"), "text/plain")}
    data = {
        "medical_json": '{"clinical": {}}',
        "reasoning_provider": "OpenAI",
        "reasoning_model": "gpt-4o",
        "reasoning_api_key": "sk-test",
    }

    response = client.post("/check_insurance", files=files, data=data)
    assert response.status_code == 200
    assert captured["provider"] == "OpenAI"
    assert captured["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# Confirm endpoint tests
# ---------------------------------------------------------------------------


def test_confirm_endpoint_calls_save_record(monkeypatch):
    captured = {"data": None, "lineage": None}

    def fake_save(data, lineage=None):
        captured["data"] = data
        captured["lineage"] = lineage

    monkeypatch.setattr(main, "save_record", fake_save)

    client = TestClient(main.app)
    payload = {
        "patient": {"mrn": "M3"},
        "encounter": {"date": "2026-02-20"},
        "clinical": {"diagnosis_list": [], "medications": [], "vitals": {}},
    }
    response = client.post("/confirm", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "saved"
    assert "correlation_id" in body
    assert captured["data"]["patient"]["mrn"] == "M3"


# ---------------------------------------------------------------------------
# Paddle service URL: SSRF hardening
# ---------------------------------------------------------------------------

def test_analyze_ignores_client_paddle_service_url(monkeypatch, upload_root):
    """A client-supplied ``paddle_service_url`` form field must NOT reach the
    pipeline. The handler resolves the Paddle URL exclusively from
    ``PADDLE_SERVICE_URL`` on the server.
    """
    monkeypatch.delenv("PADDLE_SERVICE_URL", raising=False)
    seen = {}

    def fake_pipeline(file_path, provider, model, api_key, **kwargs):
        seen["paddle_service_url"] = kwargs.get("paddle_service_url")
        return {"patient": {"mrn": "SSRF"}, "clinical": {}}

    monkeypatch.setattr(main, "process_document_pipeline", fake_pipeline)
    monkeypatch.setattr(main, "_load_history", lambda mrn: None)
    monkeypatch.setattr(main, "create_vector_store", lambda: None)
    monkeypatch.setattr(
        main, "analyze_medical_logic",
        lambda *a, **kw: {"summary": "ok", "alerts": [], "trends": []},
    )

    client = TestClient(main.app)
    files = {"file": ("scan.pdf", io.BytesIO(_valid_pdf_bytes()), "application/pdf")}
    data = {
        "provider": "Ollama",
        "model": "m",
        "ocr_backend": "paddle",
        # Attacker-supplied URL that previously reached the Paddle client.
        "paddle_service_url": "http://169.254.169.254/latest/meta-data/",
    }

    response = client.post("/analyze", files=files, data=data)

    assert response.status_code == 200
    # Client-supplied URL must be ignored; env var is unset so None flows through.
    assert seen["paddle_service_url"] is None


def test_analyze_uses_env_paddle_service_url(monkeypatch, upload_root):
    """When ``PADDLE_SERVICE_URL`` is set, the handler forwards it."""
    monkeypatch.setenv("PADDLE_SERVICE_URL", "http://127.0.0.1:8118/v1")
    seen = {}

    def fake_pipeline(file_path, provider, model, api_key, **kwargs):
        seen["paddle_service_url"] = kwargs.get("paddle_service_url")
        return {"patient": {"mrn": "OK"}, "clinical": {}}

    monkeypatch.setattr(main, "process_document_pipeline", fake_pipeline)
    monkeypatch.setattr(main, "_load_history", lambda mrn: None)
    monkeypatch.setattr(main, "create_vector_store", lambda: None)
    monkeypatch.setattr(
        main, "analyze_medical_logic",
        lambda *a, **kw: {"summary": "ok", "alerts": [], "trends": []},
    )

    client = TestClient(main.app)
    files = {"file": ("scan.pdf", io.BytesIO(_valid_pdf_bytes()), "application/pdf")}
    data = {"provider": "Ollama", "model": "m", "ocr_backend": "paddle"}

    response = client.post("/analyze", files=files, data=data)

    assert response.status_code == 200
    assert seen["paddle_service_url"] == "http://127.0.0.1:8118/v1"


# ---------------------------------------------------------------------------
# Response-shape tests: bbox, annotation, and capability fields
# ---------------------------------------------------------------------------


def test_analyze_response_contains_bbox_and_annotation_fields(monkeypatch, upload_root):
    def fake_pipeline(file_path, provider, model, api_key, **kw):
        return {
            "structured_data": {"patient": {"mrn": "BBOX-1"}, "clinical": {}},
            "ocr": {
                "raw_text": "text",
                "bounding_boxes": [{"page_number": 1, "polygon": [[0, 0], [1, 1]], "label": "field"}],
                "artifact_manifest": {
                    "original_file_url": "/artifacts/orig.png",
                    "annotated_pdf_url": "/artifacts/annotated.pdf",
                    "annotated_pdf_path": "backend/uploads/annotated.pdf",
                    "annotated_image_paths": ["backend/uploads/ann1.png"],
                    "annotated_image_urls": ["/artifacts/ann1.png"],
                    "page_image_urls": ["/artifacts/page1.png"],
                },
            },
        }

    monkeypatch.setattr(main, "process_document_pipeline", fake_pipeline)
    monkeypatch.setattr(main, "_load_history", lambda mrn: None)
    monkeypatch.setattr(main, "create_vector_store", lambda: None)
    monkeypatch.setattr(main, "analyze_medical_logic", lambda *a, **kw: {"summary": "ok", "alerts": [], "trends": []})

    client = TestClient(main.app)
    files = {"file": ("scan.png", io.BytesIO(_valid_png_bytes()), "image/png")}
    response = client.post("/analyze", files=files, data={"ocr_backend": "paddle"})
    body = response.json()

    assert response.status_code == 200
    assert len(body["bounding_boxes"]) == 1
    assert body["bounding_boxes"][0]["label"] == "field"
    assert body["annotated_pdf_url"] == "/artifacts/annotated.pdf"
    assert body["annotated_image_urls"] == ["/artifacts/ann1.png"]
    assert body["page_image_urls"] == ["/artifacts/page1.png"]
    assert body["ocr_supports_bboxes"] is True
    assert body["retrieval_enabled"] is False


def test_analyze_response_ocr_supports_bboxes_false_for_ollama(monkeypatch, upload_root):
    def fake_pipeline(file_path, provider, model, api_key, **kw):
        return {
            "structured_data": {"patient": {"mrn": "OL-1"}, "clinical": {}},
            "ocr": {"raw_text": "text", "bounding_boxes": [], "artifact_manifest": {}},
        }

    monkeypatch.setattr(main, "process_document_pipeline", fake_pipeline)
    monkeypatch.setattr(main, "_load_history", lambda mrn: None)
    monkeypatch.setattr(main, "create_vector_store", lambda: None)
    monkeypatch.setattr(main, "analyze_medical_logic", lambda *a, **kw: {"summary": "ok", "alerts": [], "trends": []})

    client = TestClient(main.app)
    files = {"file": ("scan.png", io.BytesIO(_valid_png_bytes()), "image/png")}
    response = client.post("/analyze", files=files, data={"ocr_backend": "ollama"})
    body = response.json()

    assert response.status_code == 200
    assert body["ocr_supports_bboxes"] is False
    assert body["bounding_boxes"] == []


def test_analyze_response_retrieval_enabled_when_indexed(monkeypatch, upload_root):
    monkeypatch.setattr(
        main,
        "run_extraction_graph",
        lambda **kwargs: {
            "structured_data": {"patient": {"mrn": "RET-1"}, "clinical": {}},
            "ocr": {"raw_text": "text", "bounding_boxes": [], "artifact_manifest": {}},
            "past_data": None,
            "analysis": {"summary": "done", "alerts": [], "trends": []},
            "requires_human_review": False,
            "vector_index_status": {"indexed": True, "chunks": 3, "store": "qdrant"},
            "error": None,
        },
    )

    client = TestClient(main.app)
    files = {"file": ("doc.pdf", io.BytesIO(_valid_pdf_bytes()), "application/pdf")}
    data = {"extraction_graph_mode": "true"}

    response = client.post("/analyze", files=files, data=data)
    body = response.json()

    assert response.status_code == 200
    assert body["retrieval_enabled"] is True
    assert body["vector_index_status"]["indexed"] is True


# ---------------------------------------------------------------------------
# Upload validation tests (new)
# ---------------------------------------------------------------------------


def test_analyze_rejects_disallowed_extension(upload_root):
    client = TestClient(main.app)
    files = {"file": ("malicious.exe", io.BytesIO(b"MZ\x90\x00" + b"\x00" * 32), "application/octet-stream")}
    response = client.post("/analyze", files=files, data={})
    assert response.status_code == 415


def test_analyze_rejects_mime_mismatch(upload_root):
    """A file declared as PDF but missing the %PDF magic bytes is rejected."""
    client = TestClient(main.app)
    files = {"file": ("fake.pdf", io.BytesIO(b"<html>not a pdf</html>"), "application/pdf")}
    response = client.post("/analyze", files=files, data={})
    assert response.status_code == 415


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------


def _auth_client(monkeypatch, api_key: str) -> TestClient:
    """Build a TestClient with a real API key required (ALLOW_ANONYMOUS off)."""
    monkeypatch.setenv("MEDISCAN_API_KEY", api_key)
    monkeypatch.delenv("MEDISCAN_ALLOW_ANONYMOUS", raising=False)
    return TestClient(main.app)


def test_auth_missing_header_returns_401(monkeypatch):
    client = _auth_client(monkeypatch, "s3cret-test-key")
    response = client.post("/confirm", json={"patient": {"mrn": "X"}})
    assert response.status_code == 401
    assert "API key" in response.json()["detail"]


def test_auth_wrong_header_returns_401(monkeypatch):
    client = _auth_client(monkeypatch, "s3cret-test-key")
    response = client.post(
        "/confirm",
        json={"patient": {"mrn": "X"}},
        headers={"X-API-Key": "not-the-right-key"},
    )
    assert response.status_code == 401


def test_auth_valid_header_accepted(monkeypatch):
    monkeypatch.setattr(main, "save_record", lambda data, lineage=None: None)
    client = _auth_client(monkeypatch, "s3cret-test-key")
    response = client.post(
        "/confirm",
        json={
            "patient": {"mrn": "OK"},
            "encounter": {"date": "2026-02-20"},
            "clinical": {"diagnosis_list": [], "medications": [], "vitals": {}},
        },
        headers={"X-API-Key": "s3cret-test-key"},
    )
    assert response.status_code == 200


def test_auth_unset_and_anonymous_off_returns_503(monkeypatch):
    monkeypatch.delenv("MEDISCAN_API_KEY", raising=False)
    monkeypatch.delenv("MEDISCAN_ALLOW_ANONYMOUS", raising=False)
    client = TestClient(main.app)
    response = client.post("/confirm", json={"patient": {"mrn": "X"}})
    assert response.status_code == 503
    assert "misconfigured" in response.json()["detail"].lower()


def test_health_is_public(monkeypatch):
    """Liveness probe MUST remain unauthenticated so orchestrators can reach it."""
    client = _auth_client(monkeypatch, "s3cret-test-key")
    response = client.get("/health")
    assert response.status_code == 200


def test_artifacts_endpoint_requires_auth(monkeypatch):
    client = _auth_client(monkeypatch, "s3cret-test-key")
    response = client.get("/artifacts/some-file.pdf")
    assert response.status_code == 401


def test_review_pending_requires_auth(monkeypatch):
    client = _auth_client(monkeypatch, "s3cret-test-key")
    response = client.get("/review/pending")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Artifact access control + safe path resolution
# ---------------------------------------------------------------------------


def test_artifacts_serves_file_with_valid_key(monkeypatch, upload_root):
    target = upload_root / "report.pdf"
    target.write_bytes(b"%PDF-1.4\nreal-bytes\n")
    client = _auth_client(monkeypatch, "s3cret-test-key")
    response = client.get(
        "/artifacts/report.pdf", headers={"X-API-Key": "s3cret-test-key"}
    )
    assert response.status_code == 200
    assert response.content == b"%PDF-1.4\nreal-bytes\n"


def test_artifacts_rejects_path_traversal(monkeypatch, upload_root, tmp_path):
    # Create a secret outside the upload root that an attacker might target.
    secret = tmp_path.parent / "outside-secret.txt"
    secret.write_text("TOP SECRET")
    client = _auth_client(monkeypatch, "s3cret-test-key")
    response = client.get(
        "/artifacts/../outside-secret.txt",
        headers={"X-API-Key": "s3cret-test-key"},
    )
    assert response.status_code == 404


def test_artifacts_rejects_absolute_path_escape(monkeypatch, upload_root):
    client = _auth_client(monkeypatch, "s3cret-test-key")
    # Encoded absolute-like escapes must not leak files from the host FS.
    response = client.get(
        "/artifacts/..%2F..%2F..%2Fetc%2Fpasswd",
        headers={"X-API-Key": "s3cret-test-key"},
    )
    assert response.status_code == 404


def test_artifacts_missing_file_returns_404(monkeypatch, upload_root):
    client = _auth_client(monkeypatch, "s3cret-test-key")
    response = client.get(
        "/artifacts/does-not-exist.pdf",
        headers={"X-API-Key": "s3cret-test-key"},
    )
    assert response.status_code == 404


def test_artifacts_has_no_static_mount():
    """Regression: raw artifacts must never be served via an unauth static mount."""
    for route in main.app.router.routes:
        # StaticFiles mount would appear as a Mount with path '/artifacts'.
        if getattr(route, "path", None) == "/artifacts":
            assert False, "Unauthenticated static /artifacts mount detected."
        name = type(route).__name__
        if name == "Mount" and getattr(route, "path", "").startswith("/artifacts"):
            assert False, "Unauthenticated static /artifacts mount detected."
