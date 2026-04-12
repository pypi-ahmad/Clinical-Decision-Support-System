import asyncio
import io

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

import backend.main as main


def test_confirm_record_calls_save_record(monkeypatch):
    """Targets backend.main.confirm_record in backend/main.py."""
    captured = {"data": None}

    def fake_save_record(data):
        captured["data"] = data

    monkeypatch.setattr(main, "save_record", fake_save_record)
    payload = {"patient": {"mrn": "X"}}
    result = main.confirm_record(payload)

    assert result == {"status": "saved"}
    assert captured["data"] == payload


def test_check_insurance_function_decodes_utf8_and_calls_logic(monkeypatch):
    """Targets backend.main.check_insurance in backend/main.py."""
    captured = {"medical": None, "policy": None}

    def fake_check(medical_data, policy_text):
        captured["medical"] = medical_data
        captured["policy"] = policy_text
        return {"eligible": True, "reasoning": "ok", "missing_info": []}

    monkeypatch.setattr(main, "check_insurance_coverage", fake_check)
    upload = UploadFile(filename="policy.txt", file=io.BytesIO(b"plain policy text"))

    result = asyncio.run(main.check_insurance(upload, '{"patient": {"mrn": "1"}}'))
    assert result["eligible"] is True
    assert captured["medical"]["patient"]["mrn"] == "1"
    assert captured["policy"] == "plain policy text"


def test_check_insurance_function_binary_decode_fallback(monkeypatch):
    """Targets backend.main.check_insurance binary decode fallback in backend/main.py."""
    captured = {"policy": None}

    def fake_check(medical_data, policy_text):
        captured["policy"] = policy_text
        return {"eligible": False, "reasoning": "r", "missing_info": []}

    monkeypatch.setattr(main, "check_insurance_coverage", fake_check)
    upload = UploadFile(filename="policy.pdf", file=io.BytesIO(b"\xff\xfe\xfd"))

    asyncio.run(main.check_insurance(upload, '{"a": 1}'))
    assert captured["policy"] == "Binary PDF content - (Simulated OCR would go here)"


def test_analyze_medical_doc_function_success(monkeypatch):
    """Targets backend.main.analyze_medical_doc in backend/main.py."""
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
        lambda file_path, provider, model, api_key: {"patient": {"mrn": "A1"}},
    )
    monkeypatch.setattr(main, "get_patient_history", lambda mrn: {"past": True})
    monkeypatch.setattr(main, "analyze_medical_logic", lambda *args, **kwargs: {"summary": "ok", "alerts": []})

    upload = UploadFile(filename="record.pdf", file=io.BytesIO(b"pdf"))
    result = asyncio.run(main.analyze_medical_doc(upload, "Ollama", "m", None))

    assert result["history_available"] is True
    assert result["analysis"]["summary"] == "ok"


def test_analyze_medical_doc_function_raises_http_exception_on_error(monkeypatch):
    """Targets backend.main.analyze_medical_doc error raise in backend/main.py."""
    monkeypatch.setattr(main.shutil, "copyfileobj", lambda src, dst: None)

    class DummyBuffer:
        def write(self, *_):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: DummyBuffer())
    monkeypatch.setattr(main, "process_document_pipeline", lambda *args, **kwargs: {"error": "bad"})

    upload = UploadFile(filename="record.pdf", file=io.BytesIO(b"pdf"))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.analyze_medical_doc(upload, "Ollama", "m", None))

    assert exc.value.status_code == 500
    assert exc.value.detail == {"error": "bad"}


def test_check_insurance_invalid_json_raises_400():
    """Targets JSONDecodeError guard in backend.main check_insurance."""
    upload = UploadFile(filename="policy.txt", file=io.BytesIO(b"some policy"))

    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.check_insurance(upload, "not valid json"))

    assert exc.value.status_code == 400
    assert "Invalid medical_json" in exc.value.detail


def test_confirm_record_non_dict_raises_422():
    """Targets isinstance guard in backend.main confirm_record."""
    with pytest.raises(HTTPException) as exc:
        main.confirm_record("not a dict")

    assert exc.value.status_code == 422
    assert "JSON object" in exc.value.detail


def test_analyze_medical_doc_uses_split_structuring_and_reasoning_config(monkeypatch):
    """Targets separated structuring vs reasoning configuration in backend.main.analyze_medical_doc."""
    monkeypatch.setattr(main.shutil, "copyfileobj", lambda src, dst: None)

    class DummyBuffer:
        def write(self, *_):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    pipeline_call = {}
    reasoning_call = {}

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: DummyBuffer())
    monkeypatch.setattr(
        main,
        "process_document_pipeline",
        lambda file_path, provider, model, api_key, **kwargs: pipeline_call.update(
            {"provider": provider, "model": model, "api_key": api_key, "kwargs": kwargs}
        )
        or {"structured_data": {"patient": {"mrn": "P1"}, "clinical": {"diagnosis_list": []}}, "ocr": {"raw_text": "ocr"}},
    )
    monkeypatch.setattr(main, "get_patient_history", lambda mrn: None)
    monkeypatch.setattr(main, "create_vector_store", lambda: None)
    monkeypatch.setattr(
        main,
        "analyze_medical_logic",
        lambda current_data, past_data, provider, model, api_key, **kwargs: reasoning_call.update(
            {"provider": provider, "model": model, "api_key": api_key, "kwargs": kwargs}
        )
        or {"summary": "ok", "alerts": [], "trends": []},
    )

    upload = UploadFile(filename="record.pdf", file=io.BytesIO(b"pdf"))
    result = asyncio.run(
        main.analyze_medical_doc(
            upload,
            provider="Ollama",
            model="legacy-model",
            api_key="legacy-key",
            structuring_provider="OpenAI",
            structuring_model="gpt-4o",
            structuring_api_key="struct-key",
            reasoning_provider="Anthropic",
            reasoning_model="claude-3-5-sonnet-20240620",
            reasoning_api_key="reason-key",
        )
    )

    assert result["analysis"]["summary"] == "ok"
    assert pipeline_call["provider"] == "OpenAI"
    assert pipeline_call["model"] == "gpt-4o"
    assert pipeline_call["api_key"] == "struct-key"
    assert reasoning_call["provider"] == "Anthropic"
    assert reasoning_call["model"] == "claude-3-5-sonnet-20240620"
    assert reasoning_call["api_key"] == "reason-key"
