import io

from fastapi.testclient import TestClient

import backend.main as main


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _patch_file_io(monkeypatch):
    """Patch file I/O so uploads don't touch disk."""
    monkeypatch.setattr(main.shutil, "copyfileobj", lambda src, dst: None)

    class DummyBuffer:
        def write(self, *_):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: DummyBuffer())


# ---------------------------------------------------------------------------
# Direct pipeline tests
# ---------------------------------------------------------------------------


def test_analyze_endpoint_workflow_success(monkeypatch):
    """Targets backend.main /analyze workflow in backend/main.py."""
    _patch_file_io(monkeypatch)
    monkeypatch.setattr(
        main,
        "process_document_pipeline",
        lambda file_path, provider, model, api_key: {"patient": {"mrn": "MRN9"}, "clinical": {}},
    )
    monkeypatch.setattr(main, "get_patient_history", lambda mrn: {"patient": {"mrn": mrn}})
    monkeypatch.setattr(main, "analyze_medical_logic", lambda *args, **kwargs: {"summary": "done", "alerts": []})

    client = TestClient(main.app)
    files = {"file": ("scan.pdf", io.BytesIO(b"fake"), "application/pdf")}
    data = {"provider": "Ollama", "model": "glm-4.7-flash", "api_key": ""}

    response = client.post("/analyze", files=files, data=data)
    body = response.json()

    assert response.status_code == 200
    assert body["extracted"]["patient"]["mrn"] == "MRN9"
    assert body["history_available"] is True
    assert body["analysis"]["summary"] == "done"


def test_analyze_endpoint_returns_500_on_pipeline_error(monkeypatch):
    """Targets backend.main /analyze failure branch in backend/main.py."""
    _patch_file_io(monkeypatch)
    monkeypatch.setattr(main, "process_document_pipeline", lambda *args, **kwargs: {"error": "pipeline failed"})

    client = TestClient(main.app)
    files = {"file": ("scan.pdf", io.BytesIO(b"fake"), "application/pdf")}
    response = client.post("/analyze", files=files, data={"provider": "Ollama", "model": "m", "api_key": ""})

    assert response.status_code == 500
    assert response.json()["detail"] == {"error": "pipeline failed"}


# ---------------------------------------------------------------------------
# Extraction graph mode tests
# ---------------------------------------------------------------------------


def test_analyze_extraction_graph_mode_success(monkeypatch):
    """Targets backend.main /analyze with extraction_graph_mode=true."""
    _patch_file_io(monkeypatch)
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
    files = {"file": ("scan.pdf", io.BytesIO(b"fake"), "application/pdf")}
    data = {
        "provider": "Ollama",
        "model": "glm-4.7-flash",
        "api_key": "",
        "extraction_graph_mode": "true",
        "agentic_mode": "false",
    }

    response = client.post("/analyze", files=files, data=data)
    body = response.json()

    assert response.status_code == 200
    assert body["extracted"]["patient"]["mrn"] == "GRAPH-1"
    assert body["analysis"]["summary"] == "graph done"
    assert body["requires_human_review"] is True
    assert body["vector_index_status"]["indexed"] is True


def test_analyze_extraction_graph_mode_error(monkeypatch):
    """Targets backend.main /analyze extraction_graph_mode error path."""
    _patch_file_io(monkeypatch)
    monkeypatch.setattr(
        main,
        "run_extraction_graph",
        lambda **kwargs: {"error": "graph node failed", "structured_data": {}},
    )

    client = TestClient(main.app)
    files = {"file": ("scan.pdf", io.BytesIO(b"fake"), "application/pdf")}
    data = {"extraction_graph_mode": "true"}

    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 500
    assert "graph node failed" in response.json()["detail"]["error"]


# ---------------------------------------------------------------------------
# Agentic workflow mode tests
# ---------------------------------------------------------------------------


def test_analyze_agentic_mode_success(monkeypatch):
    """Targets backend.main /analyze with agentic_mode=true."""
    _patch_file_io(monkeypatch)
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
    files = {"file": ("report.jpg", io.BytesIO(b"img"), "image/jpeg")}
    data = {"agentic_mode": "true", "extraction_graph_mode": "false"}

    response = client.post("/analyze", files=files, data=data)
    body = response.json()

    assert response.status_code == 200
    assert body["extracted"]["patient"]["mrn"] == "AGENT-1"
    assert body["analysis"]["summary"] == "agent done"
    assert body["history_available"] is True
    assert body["requires_human_review"] is False


def test_analyze_agentic_mode_error(monkeypatch):
    """Targets backend.main /analyze agentic_mode error path."""
    _patch_file_io(monkeypatch)
    monkeypatch.setattr(
        main,
        "run_agentic_extraction_workflow",
        lambda **kwargs: {"error": "agentic failure"},
    )

    client = TestClient(main.app)
    files = {"file": ("scan.pdf", io.BytesIO(b"fake"), "application/pdf")}
    data = {"agentic_mode": "true"}

    response = client.post("/analyze", files=files, data=data)
    assert response.status_code == 500


# ---------------------------------------------------------------------------
# OCR backend selection tests
# ---------------------------------------------------------------------------


def test_analyze_passes_ocr_backend_and_model_to_pipeline(monkeypatch):
    """Targets OCR backend selection flowing through to pipeline."""
    _patch_file_io(monkeypatch)
    captured = {}

    def fake_pipeline(file_path, provider, model, api_key, **kwargs):
        captured.update(kwargs)
        return {
            "structured_data": {"patient": {"mrn": "OCR-1"}, "clinical": {}},
            "ocr": {"raw_text": "text", "backend": "glm", "bounding_boxes": [], "artifact_manifest": {}},
        }

    monkeypatch.setattr(main, "process_document_pipeline", fake_pipeline)
    monkeypatch.setattr(main, "get_patient_history", lambda mrn: None)
    monkeypatch.setattr(main, "create_vector_store", lambda: None)
    monkeypatch.setattr(main, "analyze_medical_logic", lambda *a, **kw: {"summary": "ok", "alerts": [], "trends": []})

    client = TestClient(main.app)
    files = {"file": ("doc.png", io.BytesIO(b"img"), "image/png")}
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


def test_check_insurance_endpoint_passes_decoded_text_and_json(monkeypatch):
    """Targets backend.main /check_insurance workflow in backend/main.py."""
    captured = {"medical": None, "policy": None}

    def fake_check(medical_data, policy_text):
        captured["medical"] = medical_data
        captured["policy"] = policy_text
        return {"eligible": True, "confidence": "High", "reasoning": "ok", "missing_info": []}

    monkeypatch.setattr(main, "check_insurance_coverage", fake_check)

    client = TestClient(main.app)
    files = {"policy_file": ("policy.txt", io.BytesIO(b"policy body"), "text/plain")}
    data = {"medical_json": '{"patient": {"mrn": "M2"}}'}
    response = client.post("/check_insurance", files=files, data=data)

    assert response.status_code == 200
    assert response.json()["eligible"] is True
    assert captured["medical"]["patient"]["mrn"] == "M2"
    assert captured["policy"] == "policy body"


def test_check_insurance_with_policy_ocr(monkeypatch):
    """Targets backend.main /check_insurance with policy_ocr=true."""
    _patch_file_io(monkeypatch)
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
    files = {"policy_file": ("policy.pdf", io.BytesIO(b"%PDF-binary"), "application/pdf")}
    data = {
        "medical_json": '{"clinical": {"diagnosis_list": ["HTN"]}}',
        "policy_ocr": "true",
        "ocr_backend": "glm",
        "ocr_model": "glm-ocr",
    }

    response = client.post("/check_insurance", files=files, data=data)
    assert response.status_code == 200
    assert captured["policy"] == "OCR extracted policy text"


def test_check_insurance_with_reasoning_provider_split(monkeypatch):
    """Targets separate reasoning provider/model in /check_insurance."""
    captured = {}

    def fake_check(medical_data, policy_text, provider, model, api_key, **kwargs):
        captured["provider"] = provider
        captured["model"] = model
        return {"eligible": True, "reasoning": "covered", "missing_info": []}

    monkeypatch.setattr(main, "check_insurance_coverage", fake_check)
    monkeypatch.setattr(main, "create_vector_store", lambda: None)

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
    """Targets backend.main /confirm workflow in backend/main.py."""
    captured = {"data": None}

    def fake_save(data):
        captured["data"] = data

    monkeypatch.setattr(main, "save_record", fake_save)

    client = TestClient(main.app)
    payload = {"patient": {"mrn": "M3"}, "encounter": {"date": "2026-02-20"}}
    response = client.post("/confirm", json=payload)

    assert response.status_code == 200
    assert response.json() == {"status": "saved"}
    assert captured["data"] == payload
