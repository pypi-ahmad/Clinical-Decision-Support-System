"""Tests for the observability primitives.

Focus (one-task-only):
  * Request-context middleware attaches a correlation ID on response.
  * Inbound ``X-Request-ID`` is honoured when safe, rejected when unsafe.
  * ``record_llm_call`` is emitted for every ``get_ai_response`` dispatch
    (both success and failure paths).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import ai_wrapper, observability
from backend.ai_wrapper import AIProviderError


def _app_with_middleware() -> FastAPI:
    app = FastAPI()
    observability.install_request_context_middleware(app)

    @app.get("/ping")
    async def _ping() -> dict:
        return {"ok": True}

    return app


def test_request_id_generated_when_missing():
    client = TestClient(_app_with_middleware())
    response = client.get("/ping")
    assert response.status_code == 200
    rid = response.headers.get(observability.REQUEST_ID_HEADER)
    assert rid and len(rid) >= 16  # UUID4 hex is 32 chars


def test_request_id_echoed_when_safe():
    client = TestClient(_app_with_middleware())
    response = client.get("/ping", headers={observability.REQUEST_ID_HEADER: "req-abc_123"})
    assert response.status_code == 200
    assert response.headers.get(observability.REQUEST_ID_HEADER) == "req-abc_123"


def test_unsafe_inbound_request_id_is_replaced():
    client = TestClient(_app_with_middleware())
    # Contains characters outside the alnum / [-_] allow-list.
    response = client.get("/ping", headers={observability.REQUEST_ID_HEADER: "bad id!$"})
    assert response.status_code == 200
    echoed = response.headers.get(observability.REQUEST_ID_HEADER)
    assert echoed and echoed != "bad id!$"


def test_record_llm_call_emitted_on_success(monkeypatch):
    captured: list[tuple] = []

    def _fake_record(provider, model, duration, *, status="ok"):
        captured.append((provider, model, status, duration))

    monkeypatch.setattr(observability, "record_llm_call", _fake_record)
    # Also patch the name the wrapper imports lazily.
    monkeypatch.setattr("backend.observability.record_llm_call", _fake_record, raising=False)

    # Stub out the provider call so no network is used.
    monkeypatch.setattr(ai_wrapper, "_call_ollama", lambda *a, **kw: '{"ok": true}')

    out = ai_wrapper.get_ai_response("Ollama", "llama3", None, "sys", "usr")
    assert out == '{"ok": true}'
    assert len(captured) == 1
    provider, model, status, duration = captured[0]
    assert provider == "ollama"
    assert model == "llama3"
    assert status == "ok"
    assert duration >= 0.0


def test_record_llm_call_emitted_on_failure(monkeypatch):
    captured: list[tuple] = []

    def _fake_record(provider, model, duration, *, status="ok"):
        captured.append((provider, model, status))

    monkeypatch.setattr("backend.observability.record_llm_call", _fake_record, raising=False)

    def _boom(*_a, **_kw):
        raise RuntimeError("downstream blew up")

    monkeypatch.setattr(ai_wrapper, "_call_openai", _boom)

    try:
        ai_wrapper.get_ai_response("OpenAI", "gpt-x", "k", "sys", "usr")
    except AIProviderError:
        pass
    else:  # pragma: no cover - failure path must raise
        raise AssertionError("expected AIProviderError")

    assert len(captured) == 1
    assert captured[0] == ("openai", "gpt-x", "error")
