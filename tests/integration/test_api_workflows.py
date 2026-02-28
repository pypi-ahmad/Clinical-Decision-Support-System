import io

from fastapi.testclient import TestClient

import backend.main as main


def test_analyze_endpoint_workflow_success(monkeypatch):
    """Targets backend.main /analyze workflow in backend/main.py."""
    monkeypatch.setattr(main.shutil, "copyfileobj", lambda src, dst: None)

    class DummyBuffer:
        def write(self, *_):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: DummyBuffer())
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
    monkeypatch.setattr(main.shutil, "copyfileobj", lambda src, dst: None)

    class DummyBuffer:
        def write(self, *_):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: DummyBuffer())
    monkeypatch.setattr(main, "process_document_pipeline", lambda *args, **kwargs: {"error": "pipeline failed"})

    client = TestClient(main.app)
    files = {"file": ("scan.pdf", io.BytesIO(b"fake"), "application/pdf")}
    response = client.post("/analyze", files=files, data={"provider": "Ollama", "model": "m", "api_key": ""})

    assert response.status_code == 500
    assert response.json()["detail"] == {"error": "pipeline failed"}


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
