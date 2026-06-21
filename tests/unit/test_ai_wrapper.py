import types

import pytest

import backend.ai_wrapper as ai_wrapper
from backend.ai_wrapper import AIProviderError


def test_clean_json_output_strips_fences_and_extracts_json_object():
    raw = '```json\n{"a": 1, "b": 2}\n```'
    assert ai_wrapper.clean_json_output(raw) == '{"a": 1, "b": 2}'


def test_parse_model_json_handles_prose_then_object():
    raw = 'Sure, here is the JSON: {"eligible": true, "confidence": "High"}'
    parsed = ai_wrapper.parse_model_json(raw)
    assert parsed == {"eligible": True, "confidence": "High"}


def test_parse_model_json_handles_string_with_braces_inside():
    raw = '{"note": "contains {{ braces }} fine", "ok": true}'
    assert ai_wrapper.parse_model_json(raw) == {"note": "contains {{ braces }} fine", "ok": True}


def test_get_ai_response_ollama_branch_returns_message(monkeypatch):
    class DummyOllama:
        @staticmethod
        def chat(**kwargs):
            assert kwargs["model"] == "m"
            assert kwargs["messages"][0]["role"] == "system"
            assert kwargs["messages"][1]["role"] == "user"
            assert kwargs.get("format") == "json"
            return {"message": {"content": "ollama-ok"}}

    monkeypatch.setattr(ai_wrapper, "ollama", DummyOllama)
    result = ai_wrapper.get_ai_response("Ollama", "m", None, "sys", "usr")
    assert result == "ollama-ok"


def test_get_ai_response_openai_branch_returns_choice_content(monkeypatch):
    class DummyCompletions:
        @staticmethod
        def create(**kwargs):
            assert kwargs["model"] == "gpt-model"
            assert kwargs.get("response_format") == {"type": "json_object"}
            assert kwargs.get("temperature") == 0
            message = types.SimpleNamespace(content="openai-ok")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

    class DummyChat:
        completions = DummyCompletions()

    class DummyClient:
        def __init__(self, api_key=None, timeout=None):
            assert api_key == "k"
            self.chat = DummyChat()

    monkeypatch.setattr(ai_wrapper, "OpenAI", DummyClient)
    result = ai_wrapper.get_ai_response("OpenAI", "gpt-model", "k", "sys", "usr")
    assert result == "openai-ok"


def test_get_ai_response_gemini_branch_returns_text(monkeypatch):
    """The Gemini path now uses the new google-genai Client API."""
    calls = {}

    class DummyModels:
        def generate_content(self, *, model, contents, config):
            calls["model"] = model
            calls["contents"] = contents
            calls["config"] = config
            return types.SimpleNamespace(text="gemini-ok")

    class DummyClient:
        def __init__(self, api_key=None):
            calls["api_key"] = api_key
            self.models = DummyModels()

    class DummyTypes:
        @staticmethod
        def GenerateContentConfig(**kwargs):
            return kwargs

    class DummyGenAI:
        Client = DummyClient

    monkeypatch.setattr(ai_wrapper, "genai", DummyGenAI)
    monkeypatch.setattr(ai_wrapper, "genai_types", DummyTypes)

    result = ai_wrapper.get_ai_response("Gemini", "gem-model", "kg", "sys", "usr")
    assert result == "gemini-ok"
    assert calls["api_key"] == "kg"
    assert calls["model"] == "gem-model"
    assert calls["contents"] == "usr"
    assert calls["config"]["system_instruction"] == "sys"


def test_get_ai_response_gemini_forwards_force_json_and_schema(monkeypatch):
    """force_json + response_schema must map onto google-genai config fields."""
    captured = {}

    class DummyModels:
        def generate_content(self, *, model, contents, config):
            captured["config"] = config
            return types.SimpleNamespace(text='{"ok": true}')

    class DummyClient:
        def __init__(self, api_key=None):
            self.models = DummyModels()

    class DummyTypes:
        @staticmethod
        def GenerateContentConfig(**kwargs):
            return kwargs

    class DummyGenAI:
        Client = DummyClient

    monkeypatch.setattr(ai_wrapper, "genai", DummyGenAI)
    monkeypatch.setattr(ai_wrapper, "genai_types", DummyTypes)

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    result = ai_wrapper.get_ai_response(
        "Gemini",
        "gem-model",
        "kg",
        "sys",
        "usr",
        force_json=True,
        response_schema=schema,
    )

    assert result == '{"ok": true}'
    cfg = captured["config"]
    assert cfg["system_instruction"] == "sys"
    assert cfg["temperature"] == 0
    assert cfg["response_mime_type"] == "application/json"
    assert cfg["response_schema"] is schema


def test_get_ai_response_gemini_raises_when_sdk_missing(monkeypatch):
    """When google-genai is not importable we surface AIProviderError, not AttributeError."""
    monkeypatch.setattr(ai_wrapper, "genai", None)
    monkeypatch.setattr(ai_wrapper, "genai_types", None)
    with pytest.raises(AIProviderError) as excinfo:
        ai_wrapper.get_ai_response("Gemini", "gem-model", "kg", "sys", "usr")
    assert "google-genai" in excinfo.value.detail


def test_get_ai_response_anthropic_branch_returns_text(monkeypatch):
    class DummyMessages:
        @staticmethod
        def create(**kwargs):
            assert kwargs["model"] == "claude-model"
            assert kwargs["max_tokens"] == 4096
            assert kwargs["system"].startswith("sys")
            assert kwargs["messages"][0]["content"] == "usr"
            assert kwargs.get("temperature") == 0
            block = types.SimpleNamespace(type="text", text="anthropic-ok")
            return types.SimpleNamespace(content=[block])

    class DummyClient:
        def __init__(self, api_key=None, timeout=None):
            assert api_key == "ka"
            self.messages = DummyMessages()

    class DummyAnthropicModule:
        Anthropic = DummyClient

    monkeypatch.setattr(ai_wrapper, "anthropic", DummyAnthropicModule)
    result = ai_wrapper.get_ai_response("Anthropic", "claude-model", "ka", "sys", "usr")
    assert result == "anthropic-ok"


def test_get_ai_response_error_path_raises_ai_provider_error(monkeypatch):
    class DummyOllama:
        @staticmethod
        def chat(**_):
            raise RuntimeError("boom")

    monkeypatch.setattr(ai_wrapper, "ollama", DummyOllama)
    with pytest.raises(AIProviderError, match="boom"):
        ai_wrapper.get_ai_response("Ollama", "m", None, "sys", "usr")


def test_get_ai_response_unsupported_provider_raises():
    with pytest.raises(AIProviderError, match="Unsupported provider"):
        ai_wrapper.get_ai_response("UnknownProvider", "m", None, "sys", "usr")


def test_ai_provider_error_redacts_openai_key():
    err = AIProviderError("OpenAI", "Authorization: Bearer sk-test-1234567890abcdef failed")
    assert "sk-test" not in err.detail
    assert "<REDACTED_KEY>" in err.detail


# ---------------------------------------------------------------------------
# parse_model_json robustness
# ---------------------------------------------------------------------------


def test_parse_model_json_prefers_longest_object_over_decoy():
    """Earlier {...} decoy (e.g. an example) must not hide the real payload."""
    raw = (
        'Here is an example: {"example": true}.\n'
        'And here is the real answer: {"eligible": false, "confidence": "Low", "reasons": ["x"]}'
    )
    parsed = ai_wrapper.parse_model_json(raw)
    assert parsed == {"eligible": False, "confidence": "Low", "reasons": ["x"]}


def test_parse_model_json_extracts_fenced_block_after_prose():
    """Fenced block may appear after explanatory prose (not only as a prefix)."""
    raw = 'Thinking... here it is:\n```json\n{"a": 1, "b": [1, 2]}\n```\nDone.'
    assert ai_wrapper.parse_model_json(raw) == {"a": 1, "b": [1, 2]}


def test_parse_model_json_plain_fence_without_json_tag():
    raw = '```\n{"ok": true}\n```'
    assert ai_wrapper.parse_model_json(raw) == {"ok": True}


def test_parse_model_json_accepts_array_with_leading_object():
    """Some providers wrap output as [{...}]. Unwrap the first object."""
    raw = '[{"mrn": "A", "dx": "X"}]'
    assert ai_wrapper.parse_model_json(raw) == {"mrn": "A", "dx": "X"}


def test_parse_model_json_rejects_empty_input():
    with pytest.raises(ValueError):
        ai_wrapper.parse_model_json("")
    with pytest.raises(ValueError):
        ai_wrapper.parse_model_json("   \n\t  ")


def test_parse_model_json_rejects_non_string_input():
    with pytest.raises(ValueError):
        ai_wrapper.parse_model_json(None)  # type: ignore[arg-type]


def test_parse_model_json_rejects_pure_prose():
    with pytest.raises(ValueError):
        ai_wrapper.parse_model_json("I cannot answer that question.")


def test_parse_model_json_rejects_top_level_scalar_or_list():
    """Callers expect a dict; a bare int/string/array with no dict must fail."""
    with pytest.raises(ValueError):
        ai_wrapper.parse_model_json("42")
    with pytest.raises(ValueError):
        ai_wrapper.parse_model_json('"just a string"')
    with pytest.raises(ValueError):
        ai_wrapper.parse_model_json("[1, 2, 3]")


def test_parse_model_json_object_with_brace_literal_in_string():
    """Braces inside a string literal must not confuse the balanced scanner."""
    raw = 'Preamble. {"code": "if (x) { return 1; }", "ok": true}'
    assert ai_wrapper.parse_model_json(raw) == {
        "code": "if (x) { return 1; }",
        "ok": True,
    }


def test_clean_json_output_returns_text_on_unrecoverable_input():
    """Back-compat: clean_json_output must fall back to raw text, not raise."""
    raw = "no json here at all"
    assert ai_wrapper.clean_json_output(raw) == raw


# ---------------------------------------------------------------------------
# Provider-client caching (long-lived httpx pools)
# ---------------------------------------------------------------------------


def test_openai_client_is_cached_per_key_and_timeout(monkeypatch):
    calls = []

    class DummyOpenAI:
        def __init__(self, api_key=None, timeout=None):
            calls.append((api_key, timeout))

    monkeypatch.setattr(ai_wrapper, "OpenAI", DummyOpenAI)
    ai_wrapper.reset_provider_clients()

    a1 = ai_wrapper._get_openai_client("k1", 30)
    a2 = ai_wrapper._get_openai_client("k1", 30)
    b = ai_wrapper._get_openai_client("k2", 30)
    c = ai_wrapper._get_openai_client("k1", 60)

    assert a1 is a2  # same key+timeout => same object
    assert a1 is not b  # different key => new object
    assert a1 is not c  # different timeout => new object
    assert len(calls) == 3  # constructor hit exactly 3 times


def test_anthropic_client_is_cached_per_key_and_timeout(monkeypatch):
    calls = []

    class DummyAnthropic:
        def __init__(self, api_key=None, timeout=None):
            calls.append((api_key, timeout))

    class DummyAnthropicModule:
        Anthropic = DummyAnthropic

    monkeypatch.setattr(ai_wrapper, "anthropic", DummyAnthropicModule)
    ai_wrapper.reset_provider_clients()

    a1 = ai_wrapper._get_anthropic_client("k1", 30)
    a2 = ai_wrapper._get_anthropic_client("k1", 30)
    assert a1 is a2
    assert len(calls) == 1


def test_gemini_client_is_cached_per_key(monkeypatch):
    calls = []

    class DummyGemini:
        def __init__(self, api_key=None, **kwargs):
            calls.append(api_key)

    class DummyGenAI:
        Client = DummyGemini

    monkeypatch.setattr(ai_wrapper, "genai", DummyGenAI)
    ai_wrapper.reset_provider_clients()

    a1 = ai_wrapper._get_gemini_client("k1")
    a2 = ai_wrapper._get_gemini_client("k1")
    b = ai_wrapper._get_gemini_client("k2")

    assert a1 is a2
    assert a1 is not b
    assert len(calls) == 2


def test_reset_provider_clients_forces_fresh_construction(monkeypatch):
    calls = []

    class DummyOpenAI:
        def __init__(self, api_key=None, timeout=None):
            calls.append((api_key, timeout))

    monkeypatch.setattr(ai_wrapper, "OpenAI", DummyOpenAI)
    ai_wrapper.reset_provider_clients()

    ai_wrapper._get_openai_client("k", 30)
    ai_wrapper._get_openai_client("k", 30)
    ai_wrapper.reset_provider_clients()
    ai_wrapper._get_openai_client("k", 30)

    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Provider resilience: timeouts + retries
# ---------------------------------------------------------------------------


def test_ollama_call_uses_client_with_timeout(monkeypatch):
    """Ollama path must wrap the call in a Client(timeout=...) when available."""
    seen = {"timeout": None, "kwargs": None}

    class FakeClient:
        def __init__(self, timeout=None):
            seen["timeout"] = timeout

        def chat(self, **kwargs):
            seen["kwargs"] = kwargs
            return {"message": {"content": "ok"}}

    class FakeOllama:
        Client = FakeClient

        @staticmethod
        def chat(**kwargs):  # must NOT be invoked when Client exists
            raise AssertionError("module-level ollama.chat was used despite Client")

    monkeypatch.setattr(ai_wrapper, "ollama", FakeOllama)
    monkeypatch.setenv("MEDISCAN_LLM_TIMEOUT", "45")

    result = ai_wrapper.get_ai_response("Ollama", "m", None, "sys", "usr")
    assert result == "ok"
    assert seen["timeout"] == 45
    assert seen["kwargs"]["model"] == "m"
    assert seen["kwargs"]["format"] == "json"


def test_ollama_call_falls_back_to_module_chat_when_client_missing(monkeypatch):
    """Legacy SDK builds without Client must still work (via ollama.chat)."""
    seen = {"kwargs": None}

    class FakeOllama:
        # No ``Client`` attr on purpose.
        @staticmethod
        def chat(**kwargs):
            seen["kwargs"] = kwargs
            return {"message": {"content": "fallback-ok"}}

    monkeypatch.setattr(ai_wrapper, "ollama", FakeOllama)
    result = ai_wrapper.get_ai_response("Ollama", "m", None, "sys", "usr")
    assert result == "fallback-ok"
    assert seen["kwargs"]["model"] == "m"


def test_gemini_client_receives_http_timeout(monkeypatch):
    """Gemini client construction must forward timeout via HttpOptions."""
    seen = {"timeout": None}

    class FakeHttpOptions:
        def __init__(self, timeout=None):
            seen["timeout"] = timeout

    class FakeClient:
        def __init__(self, api_key=None, http_options=None):
            seen["http_options_obj"] = http_options

    class FakeGenAI:
        Client = FakeClient

    class FakeGenAITypes:
        HttpOptions = FakeHttpOptions

    monkeypatch.setattr(ai_wrapper, "genai", FakeGenAI)
    monkeypatch.setattr(ai_wrapper, "genai_types", FakeGenAITypes)
    ai_wrapper.reset_provider_clients()

    ai_wrapper._get_gemini_client("k", 30)
    # 30s => 30_000ms forwarded to the SDK's HttpOptions.
    assert seen["timeout"] == 30_000


def test_is_retryable_skips_authentication_and_validation_errors():
    """Auth / validation errors are final \u2014 the predicate must reject them."""

    class AuthErr(Exception):
        status_code = 401

    class PermErr(Exception):
        status_code = 403

    class BadReq(Exception):
        status_code = 400

    class NotFound(Exception):
        status_code = 404

    for exc_cls in (AuthErr, PermErr, BadReq, NotFound):
        assert ai_wrapper._is_retryable(exc_cls("x")) is False

    # AIProviderError is already final (wrapped by get_ai_response on giveup).
    assert ai_wrapper._is_retryable(AIProviderError("OpenAI", "auth failed")) is False


def test_is_retryable_accepts_transient_network_and_5xx():
    """Transient failures (5xx, 429, timeouts, connection errors) must retry."""

    class Transient503(Exception):
        status_code = 503

    class RateLimited(Exception):
        status_code = 429

    class APITimeoutError(Exception):
        pass

    for exc in (
        Transient503("down"),
        RateLimited("slow down"),
        APITimeoutError("deadline"),
        ConnectionError("dns"),
        TimeoutError("timed out"),
    ):
        assert ai_wrapper._is_retryable(exc) is True
