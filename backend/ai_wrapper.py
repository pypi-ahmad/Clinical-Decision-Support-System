"""Multi-provider LLM adapter.

Design goals for this rewrite:

* **Correctness** — Gemini now uses the new ``google-genai`` Client API.
  The old ``genai.configure`` / ``GenerativeModel`` calls are removed.
* **Structured output first** — OpenAI uses ``response_format=json_object``
  (and can be upgraded to ``json_schema`` via the ``response_schema`` arg).
  Ollama uses ``format="json"``. Gemini uses ``response_mime_type``. Anthropic
  is instructed via prompt + validated downstream.
* **Reliability** — provider calls are wrapped with :mod:`tenacity`
  exponential-backoff retries so transient 429/503 don't fail the request.
* **Robust parsing** — :func:`parse_model_json` tries a direct ``json.loads``
  then walks a balanced-brace scanner, so stray prose before/after the JSON
  object no longer corrupts the payload (the previous greedy ``{.*}`` regex
  often spliced unrelated objects).
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

import anthropic
import ollama
from openai import OpenAI

try:
    from google import genai  # new unified SDK (google-genai)
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - allow import when SDK is absent
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential_jitter,
    )
except Exception:  # pragma: no cover - tenacity is optional at import time
    retry = None  # type: ignore[assignment]

from backend.logging_config import get_logger

_logger = get_logger(__name__)


class AIProviderError(RuntimeError):
    """Raised when an AI provider call fails or the provider is unsupported."""

    def __init__(self, provider: str, detail: str):
        self.provider = provider
        # Hide secrets from ``detail`` — the structlog processor also scrubs
        # these, but the error is shown to callers as well.
        self.detail = _scrub_error(detail)
        super().__init__(f"Error with {provider}: {self.detail}")


_SECRET_SUBSTRINGS = ("sk-ant-", "sk-")


def _scrub_error(text: str) -> str:
    for needle in _SECRET_SUBSTRINGS:
        if needle in text:
            # Replace the token prefix followed by at least 3 chars.
            text = re.sub(needle + r"[A-Za-z0-9_\-]{3,}", "<REDACTED_KEY>", text)
    return text


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


def _is_retryable(exc: BaseException) -> bool:
    """Only retry on transient network/timeout/5xx/rate-limit errors.

    Avoid retrying authentication (401/403), invalid-request (400/404), or
    ``AIProviderError`` (already final) — those will not succeed on retry and
    just waste the user's budget.
    """
    if isinstance(exc, AIProviderError):
        return False
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    # Provider-specific transient error detection by class name to avoid
    # hard-importing SDK error types (which differ across versions).
    transient_names = {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
        "ResourceExhausted",
        "DeadlineExceeded",
        "Unavailable",
    }
    if type(exc).__name__ in transient_names:
        return True
    # Status-code driven: anything with a 5xx or 429 status_code attribute.
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    return False


def _retry(fn):
    if retry is None:
        return fn
    from tenacity import retry_if_exception

    return retry(
        reraise=True,
        stop=stop_after_attempt(int(os.environ.get("MEDISCAN_LLM_RETRIES", 3))),
        wait=wait_exponential_jitter(initial=1, max=10),
        retry=retry_if_exception(_is_retryable),
    )(fn)


@_retry
def _call_ollama(
    model: str,
    system_prompt: str,
    user_text: str,
    *,
    force_json: bool,
    timeout: int,
) -> str:
    options: dict[str, Any] = {"temperature": 0}
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "options": options,
    }
    if force_json:
        kwargs["format"] = "json"
    # Bound the call with a real HTTP timeout so a stalled local Ollama server
    # cannot hang a FastAPI worker. Fall back to the module-level ``chat`` when
    # the SDK object does not expose ``Client`` (older versions / test doubles).
    client_cls = getattr(ollama, "Client", None)
    if client_cls is not None:
        client = client_cls(timeout=timeout)
        response = client.chat(**kwargs)
    else:
        response = ollama.chat(**kwargs)
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Long-lived provider clients
#
# Each SDK client owns an ``httpx.Client`` (and its connection pool). Building
# a new one per request is wasteful and, under load, measurably slower than
# reusing a process-wide instance. Cache by (api_key, timeout) — these are the
# only construction-time knobs the callers vary, so two distinct keys produce
# two distinct pools as expected. ``lru_cache`` is thread-safe for hits, and
# all three SDK clients are documented as safe for concurrent use.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _get_openai_client(api_key: str | None, timeout: int) -> OpenAI:
    return OpenAI(api_key=api_key, timeout=timeout)


@lru_cache(maxsize=8)
def _get_anthropic_client(api_key: str | None, timeout: int) -> "anthropic.Anthropic":
    return anthropic.Anthropic(api_key=api_key, timeout=timeout)


@lru_cache(maxsize=8)
def _get_gemini_client(api_key: str | None, timeout: int = 120):
    if genai is None:
        raise AIProviderError("Gemini", "google-genai SDK is not installed")
    # Forward the timeout to the SDK's underlying httpx pool. ``HttpOptions``
    # takes the value in milliseconds. Older SDK builds without HttpOptions
    # fall back to their own default (60s) — still finite, still bounded.
    http_options_cls = getattr(genai_types, "HttpOptions", None) if genai_types else None
    if http_options_cls is not None:
        return genai.Client(
            api_key=api_key,
            http_options=http_options_cls(timeout=int(timeout) * 1000),
        )
    return genai.Client(api_key=api_key)


def reset_provider_clients() -> None:
    """Drop all cached provider clients. Primarily for tests / key rotation."""
    _get_openai_client.cache_clear()
    _get_anthropic_client.cache_clear()
    _get_gemini_client.cache_clear()


@_retry
def _call_openai(
    model: str,
    api_key: str | None,
    system_prompt: str,
    user_text: str,
    *,
    force_json: bool,
    timeout: int,
) -> str:
    client = _get_openai_client(api_key, timeout)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0,
    }
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


@_retry
def _call_gemini(
    model: str,
    api_key: str | None,
    system_prompt: str,
    user_text: str,
    *,
    force_json: bool,
    timeout: int,
    response_schema: dict | None = None,
) -> str:
    if genai is None or genai_types is None:
        raise AIProviderError("Gemini", "google-genai SDK is not installed")
    client = _get_gemini_client(api_key, timeout)
    config_kwargs: dict[str, Any] = {
        "system_instruction": system_prompt,
        "temperature": 0,
    }
    if force_json:
        config_kwargs["response_mime_type"] = "application/json"
        if response_schema is not None:
            # google-genai accepts either a pydantic model or a JSON-schema dict.
            config_kwargs["response_schema"] = response_schema
    config = genai_types.GenerateContentConfig(**config_kwargs)
    response = client.models.generate_content(
        model=model,
        contents=user_text,
        config=config,
    )
    return response.text or ""


@_retry
def _call_anthropic(
    model: str,
    api_key: str | None,
    system_prompt: str,
    user_text: str,
    *,
    force_json: bool,
    timeout: int,
) -> str:
    client = _get_anthropic_client(api_key, timeout)
    # Anthropic does not have a structured-output response_format, so we
    # strengthen the system prompt when JSON is required.
    augmented_system = system_prompt
    if force_json:
        augmented_system = (
            system_prompt
            + "\n\nOUTPUT CONSTRAINT: Reply with a single JSON object only. "
              "Do not include markdown fences, commentary or explanations."
        )
    response = client.messages.create(
        model=model,
        max_tokens=int(os.environ.get("MEDISCAN_ANTHROPIC_MAX_TOKENS", 4096)),
        system=augmented_system,
        temperature=0,
        messages=[{"role": "user", "content": user_text}],
    )
    if not response.content:
        return ""
    # Concatenate all text blocks (Anthropic may return multiple).
    return "".join(block.text for block in response.content if getattr(block, "type", "text") == "text")


def get_ai_response(
    provider: str,
    model: str,
    api_key: str | None,
    system_prompt: str,
    user_text: str,
    *,
    force_json: bool = True,
    timeout: int | None = None,
    response_schema: dict | None = None,
) -> str:
    """Universal wrapper for AI text-to-text generation.

    ``force_json`` enables native JSON modes for providers that support them.
    ``response_schema`` is currently forwarded to Gemini (``response_schema``)
    only; the balanced-brace fallback in :func:`parse_model_json` keeps the
    other providers robust until their structured-output modes are wired in.
    ``timeout`` is per-provider (seconds).
    """
    provider_normalized = (provider or "Ollama").strip().lower()
    effective_timeout = int(timeout or os.environ.get("MEDISCAN_LLM_TIMEOUT", 120))

    # Import lazily to avoid a circular import at module load time and to keep
    # observability strictly optional.
    try:
        from backend.observability import record_llm_call
    except Exception:  # pragma: no cover - observability is optional
        record_llm_call = None  # type: ignore[assignment]

    import time as _time

    _start = _time.perf_counter()
    _status = "ok"
    try:
        if provider_normalized == "ollama":
            return _call_ollama(
                model,
                system_prompt,
                user_text,
                force_json=force_json,
                timeout=effective_timeout,
            )
        if provider_normalized == "openai":
            return _call_openai(
                model,
                api_key,
                system_prompt,
                user_text,
                force_json=force_json,
                timeout=effective_timeout,
            )
        if provider_normalized == "gemini":
            return _call_gemini(
                model,
                api_key,
                system_prompt,
                user_text,
                force_json=force_json,
                timeout=effective_timeout,
                response_schema=response_schema,
            )
        if provider_normalized == "anthropic":
            return _call_anthropic(
                model,
                api_key,
                system_prompt,
                user_text,
                force_json=force_json,
                timeout=effective_timeout,
            )
        raise AIProviderError(provider, "Unsupported provider")
    except AIProviderError:
        _status = "error"
        raise
    except Exception as exc:
        _status = "error"
        _logger.warning("llm_call_failed", provider=provider_normalized, model=model)
        raise AIProviderError(provider, str(exc)) from exc
    finally:
        if record_llm_call is not None:
            try:
                record_llm_call(
                    provider_normalized,
                    model or "",
                    _time.perf_counter() - _start,
                    status=_status,
                )
            except Exception:  # pragma: no cover - never break the call path
                pass


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _balanced_spans(text: str, open_ch: str, close_ch: str) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) spans of top-level balanced ``open/close`` pairs.

    String-aware (respects escapes and quoted delimiters). Returns every
    top-level balanced region in order of appearance, so callers can try the
    most promising candidate (e.g. the longest) rather than just the first.
    """
    spans: list[tuple[int, int]] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch:
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append((start, i + 1))
                    start = None
    return spans


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def parse_model_json(text: str) -> dict:
    """Parse ``text`` into a dict, tolerating markdown fences and surrounding prose.

    Resolution order (each step fails loud only if all remaining steps fail):
      1. Direct ``json.loads`` after trimming whitespace.
      2. Content of any ```` ```json ... ``` ```` fence anywhere in the text.
      3. The **longest** balanced ``{...}`` substring.
      4. The **longest** balanced ``[...]`` substring whose element is a dict.

    Raises ``ValueError`` on empty input or when nothing parses to a dict.
    """
    if text is None or not isinstance(text, str):
        raise ValueError("Empty model response")
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty model response")

    # --- 1. direct parse ------------------------------------------------------
    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # --- 2. fenced blocks anywhere in the text --------------------------------
    for match in _FENCE_RE.finditer(stripped):
        body = match.group(1).strip()
        if not body:
            continue
        try:
            result = json.loads(body)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result

    # --- 3. longest balanced object -------------------------------------------
    obj_spans = sorted(
        _balanced_spans(stripped, "{", "}"),
        key=lambda s: s[1] - s[0],
        reverse=True,
    )
    for start, end in obj_spans:
        try:
            result = json.loads(stripped[start:end])
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result

    # --- 4. longest balanced array whose first element is a dict --------------
    arr_spans = sorted(
        _balanced_spans(stripped, "[", "]"),
        key=lambda s: s[1] - s[0],
        reverse=True,
    )
    for start, end in arr_spans:
        try:
            result = json.loads(stripped[start:end])
        except json.JSONDecodeError:
            continue
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return result[0]

    raise ValueError("Model response did not contain a parseable JSON object")


def clean_json_output(text: str) -> str:
    """Back-compat shim: return a JSON string from messy model output."""
    try:
        return json.dumps(parse_model_json(text))
    except (ValueError, json.JSONDecodeError):
        return text
