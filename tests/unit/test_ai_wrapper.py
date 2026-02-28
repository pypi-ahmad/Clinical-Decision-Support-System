import types

import backend.ai_wrapper as ai_wrapper


def test_clean_json_output_strips_fences_and_extracts_json_object():
    """Targets backend.ai_wrapper.clean_json_output in backend/ai_wrapper.py."""
    raw = "```json\n{\"a\": 1, \"b\": 2}\n```"
    assert ai_wrapper.clean_json_output(raw) == '{"a": 1, "b": 2}'


def test_get_ai_response_ollama_branch_returns_message(monkeypatch):
    """Targets backend.ai_wrapper.get_ai_response Ollama branch in backend/ai_wrapper.py."""

    class DummyOllama:
        @staticmethod
        def chat(model, messages):
            assert model == "m"
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            return {"message": {"content": "ollama-ok"}}

    monkeypatch.setattr(ai_wrapper, "ollama", DummyOllama)
    result = ai_wrapper.get_ai_response("Ollama", "m", None, "sys", "usr")
    assert result == "ollama-ok"


def test_get_ai_response_openai_branch_returns_choice_content(monkeypatch):
    """Targets backend.ai_wrapper.get_ai_response OpenAI branch in backend/ai_wrapper.py."""

    class DummyCompletions:
        @staticmethod
        def create(model, messages, response_format):
            assert model == "gpt-model"
            assert response_format == {"type": "json_object"}
            message = types.SimpleNamespace(content="openai-ok")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

    class DummyChat:
        completions = DummyCompletions()

    class DummyClient:
        def __init__(self, api_key):
            assert api_key == "k"
            self.chat = DummyChat()

    monkeypatch.setattr(ai_wrapper, "OpenAI", DummyClient)
    result = ai_wrapper.get_ai_response("OpenAI", "gpt-model", "k", "sys", "usr")
    assert result == "openai-ok"


def test_get_ai_response_gemini_branch_returns_text(monkeypatch):
    """Targets backend.ai_wrapper.get_ai_response Gemini branch in backend/ai_wrapper.py."""
    calls = {"configured": None, "prompt": None}

    class DummyModel:
        def __init__(self, model):
            assert model == "gem-model"

        def generate_content(self, prompt):
            calls["prompt"] = prompt
            return types.SimpleNamespace(text="gemini-ok")

    class DummyGenAI:
        @staticmethod
        def configure(api_key):
            calls["configured"] = api_key

        GenerativeModel = DummyModel

    monkeypatch.setattr(ai_wrapper, "genai", DummyGenAI)
    result = ai_wrapper.get_ai_response("Gemini", "gem-model", "kg", "sys", "usr")
    assert result == "gemini-ok"
    assert calls["configured"] == "kg"
    assert "USER INPUT: usr" in calls["prompt"]


def test_get_ai_response_anthropic_branch_returns_text(monkeypatch):
    """Targets backend.ai_wrapper.get_ai_response Anthropic branch in backend/ai_wrapper.py."""

    class DummyMessages:
        @staticmethod
        def create(model, max_tokens, system, messages):
            assert model == "claude-model"
            assert max_tokens == 4096
            assert system == "sys"
            assert messages[0]["content"] == "usr"
            content_item = types.SimpleNamespace(text="anthropic-ok")
            return types.SimpleNamespace(content=[content_item])

    class DummyClient:
        def __init__(self, api_key):
            assert api_key == "ka"
            self.messages = DummyMessages()

    class DummyAnthropicModule:
        Anthropic = DummyClient

    monkeypatch.setattr(ai_wrapper, "anthropic", DummyAnthropicModule)
    result = ai_wrapper.get_ai_response("Anthropic", "claude-model", "ka", "sys", "usr")
    assert result == "anthropic-ok"


def test_get_ai_response_error_path_returns_error_string(monkeypatch):
    """Targets backend.ai_wrapper.get_ai_response exception path in backend/ai_wrapper.py."""

    class DummyOllama:
        @staticmethod
        def chat(model, messages):
            raise RuntimeError("boom")

    monkeypatch.setattr(ai_wrapper, "ollama", DummyOllama)
    result = ai_wrapper.get_ai_response("Ollama", "m", None, "sys", "usr")
    assert result.startswith("Error with Ollama:")


def test_get_ai_response_unsupported_provider_returns_none():
    """Targets backend.ai_wrapper.get_ai_response unsupported-provider behavior in backend/ai_wrapper.py."""
    assert ai_wrapper.get_ai_response("UnknownProvider", "m", None, "sys", "usr") == "Error with UnknownProvider: Unsupported provider"
