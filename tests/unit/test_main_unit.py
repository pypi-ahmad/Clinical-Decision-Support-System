"""Unit tests for individual handlers in ``backend.main``.

These tests exercise the handler functions directly (not through the HTTP
layer) so they can bypass ``Depends`` auth. The integration suite
(``tests/integration/test_api_workflows.py``) covers the full request path.
"""

import asyncio
import io

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

import backend.main as main


def _upload(name: str, content: bytes, content_type: str = "application/pdf") -> UploadFile:
    from starlette.datastructures import Headers

    headers = Headers({"content-type": content_type})
    return UploadFile(filename=name, file=io.BytesIO(content), headers=headers)


def test_confirm_record_calls_save_record(monkeypatch):
    captured = {"data": None, "lineage": None}

    def fake_save_record(data, lineage=None):
        captured["data"] = data
        captured["lineage"] = lineage

    monkeypatch.setattr(main, "save_record", fake_save_record)

    payload = {
        "patient": {"mrn": "X"},
        "encounter": {"date": "2026-02-20"},
        "clinical": {"diagnosis_list": [], "medications": [], "vitals": {}},
    }
    result = asyncio.run(main.confirm_record(payload))

    assert result["status"] == "saved"
    assert "correlation_id" in result
    assert captured["data"]["patient"]["mrn"] == "X"


def test_check_insurance_function_decodes_utf8_and_calls_logic(monkeypatch):
    captured = {"medical": None, "policy": None}

    def fake_check(medical_data, policy_text, *args, **kwargs):
        captured["medical"] = medical_data
        captured["policy"] = policy_text
        return {"eligible": True, "reasoning": "ok", "missing_info": []}

    monkeypatch.setattr(main, "check_insurance_coverage", fake_check)
    monkeypatch.setattr(main, "create_vector_store", lambda: None)

    upload = _upload("policy.txt", b"plain policy text", "text/plain")

    result = asyncio.run(main.check_insurance(upload, '{"patient": {"mrn": "1"}}'))
    assert result["eligible"] is True
    assert captured["medical"]["patient"]["mrn"] == "1"
    assert captured["policy"] == "plain policy text"


def test_check_insurance_function_binary_pdf_requires_ocr(monkeypatch):
    """A non-text policy upload must go through OCR — the old stub fallback was a bug."""
    captured = {"policy": None}

    def fake_check(medical_data, policy_text, *args, **kwargs):
        captured["policy"] = policy_text
        return {"eligible": False, "reasoning": "r", "missing_info": []}

    def fake_ocr(path, **kwargs):
        return {"markdown": "extracted policy clauses", "raw_text": "extracted policy clauses"}

    monkeypatch.setattr(main, "check_insurance_coverage", fake_check)
    monkeypatch.setattr(main, "run_document_ocr", fake_ocr)
    monkeypatch.setattr(main, "create_vector_store", lambda: None)

    upload = _upload("policy.pdf", b"%PDF-1.4 dummy", "application/pdf")

    asyncio.run(main.check_insurance(upload, '{"patient": {"mrn": "1"}}'))
    assert captured["policy"] == "extracted policy clauses"


def test_analyze_medical_doc_function_success(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDISCAN_UPLOAD_ROOT", str(tmp_path))

    def fake_pipeline(file_path, provider, model, api_key, **kwargs):
        return {"patient": {"mrn": "A1"}, "clinical": {"diagnosis_list": []}}

    monkeypatch.setattr(main, "process_document_pipeline", fake_pipeline)
    monkeypatch.setattr(main, "_load_history", lambda mrn: {"past": True})
    monkeypatch.setattr(main, "create_vector_store", lambda: None)
    monkeypatch.setattr(
        main,
        "analyze_medical_logic",
        lambda *args, **kwargs: {"summary": "ok", "alerts": [], "trends": []},
    )

    upload = _upload("record.pdf", b"%PDF-1.4 dummy")
    result = asyncio.run(main.analyze_medical_doc(upload))

    assert result["history_available"] is True
    assert result["analysis"]["summary"] == "ok"
    assert "correlation_id" in result


def test_analyze_medical_doc_function_raises_http_exception_on_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDISCAN_UPLOAD_ROOT", str(tmp_path))
    monkeypatch.setattr(main, "process_document_pipeline", lambda *args, **kwargs: {"error": "bad"})

    upload = _upload("record.pdf", b"%PDF-1.4 dummy")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.analyze_medical_doc(upload))

    assert exc.value.status_code == 500
    detail = exc.value.detail
    # Detail is now a sanitised envelope, not the raw upstream error.
    assert isinstance(detail, dict)
    assert detail["error"] == "Request failed"
    assert "correlation_id" in detail


def test_check_insurance_invalid_json_raises_400(monkeypatch):
    monkeypatch.setattr(main, "create_vector_store", lambda: None)
    upload = _upload("policy.txt", b"some policy", "text/plain")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.check_insurance(upload, "not valid json"))

    assert exc.value.status_code == 400


def test_confirm_record_non_dict_raises_422(monkeypatch):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.confirm_record("not a dict"))

    assert exc.value.status_code == 422


def test_confirm_record_offloads_blocking_audit_and_lineage(monkeypatch):
    """Blocking audit/lineage calls must run in the threadpool so the event
    loop keeps servicing other coroutines concurrently.
    """
    import time

    block_duration = 0.25

    def slow_save_record(data, lineage=None):
        time.sleep(block_duration)

    def slow_audit(event_type, **kwargs):
        time.sleep(block_duration)

    def slow_lineage(**kwargs):
        time.sleep(block_duration)
        return {"git_sha": "abc"}

    monkeypatch.setattr(main, "save_record", slow_save_record)
    monkeypatch.setattr(main, "record_audit_event", slow_audit)
    monkeypatch.setattr(main, "capture_lineage", slow_lineage)

    payload = {
        "patient": {"mrn": "X"},
        "encounter": {"date": "2026-02-20"},
        "clinical": {"diagnosis_list": [], "medications": [], "vitals": {}},
    }

    async def _run():
        heartbeats = 0

        async def ticker():
            nonlocal heartbeats
            # Each iteration yields control for 10ms. If the handler blocks
            # the loop (e.g. a bare `capture_lineage()` call), these ticks
            # stop firing while the handler is stuck in a sleep.
            while True:
                await asyncio.sleep(0.01)
                heartbeats += 1

        tick_task = asyncio.create_task(ticker())
        try:
            await main.confirm_record(payload)
        finally:
            tick_task.cancel()
            try:
                await tick_task
            except asyncio.CancelledError:
                pass
        return heartbeats

    heartbeats = asyncio.run(_run())
    # Three ~250ms blocking calls = ~0.75s wall time. With proper threadpool
    # offloading the ticker fires ~75 times; with blocking calls on the loop
    # the ticker would manage at most a handful of ticks.
    assert heartbeats >= 25, (
        f"Event loop stalled: only {heartbeats} 10ms ticks fired during "
        "confirm_record (expected >=25 if blocking work is offloaded)."
    )


def test_analyze_medical_doc_uses_split_structuring_and_reasoning_config(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDISCAN_UPLOAD_ROOT", str(tmp_path))
    pipeline_call = {}
    reasoning_call = {}

    def fake_pipeline(file_path, provider, model, api_key, **kwargs):
        pipeline_call.update({"provider": provider, "model": model, "api_key": api_key, "kwargs": kwargs})
        return {
            "structured_data": {"patient": {"mrn": "P1"}, "clinical": {"diagnosis_list": []}},
            "ocr": {"raw_text": "ocr"},
        }

    monkeypatch.setattr(main, "process_document_pipeline", fake_pipeline)
    monkeypatch.setattr(main, "_load_history", lambda mrn: None)
    monkeypatch.setattr(main, "create_vector_store", lambda: None)
    monkeypatch.setattr(
        main,
        "analyze_medical_logic",
        lambda current_data, past_data, provider, model, api_key, **kwargs: reasoning_call.update(
            {"provider": provider, "model": model, "api_key": api_key, "kwargs": kwargs}
        )
        or {"summary": "ok", "alerts": [], "trends": []},
    )
    # Enable user-supplied API keys for this specific test so they flow
    # through without being overridden by the server-side vault.
    monkeypatch.setenv("MEDISCAN_ALLOW_USER_API_KEYS", "1")

    upload = _upload("record.pdf", b"%PDF-1.4 dummy")
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
