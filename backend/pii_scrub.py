"""Lightweight PII scrub hook used before sending OCR text to non-local LLMs.

Opt in by setting ``MEDISCAN_PII_SCRUB=1`` (and optionally installing Presidio
for higher-quality analysis). Falls back to a deterministic regex-based
scrubber so the hook is safe to call unconditionally.

Design notes:
- We never scrub text for local (Ollama) providers — the HIPAA threat model
  there is "data never leaves the box".
- Scrub is reversible *only* via an opaque placeholder map we do NOT persist,
  so downstream code must treat scrubbed output as lossy.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from backend.logging_config import get_logger

_logger = get_logger(__name__)

# Conservative patterns — intentionally narrow to avoid corrupting medical text.
_REGEX_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
)


def is_enabled() -> bool:
    return os.environ.get("MEDISCAN_PII_SCRUB", "0") in {"1", "true", "True", "yes"}


@lru_cache(maxsize=1)
def _presidio():  # pragma: no cover - optional dependency
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        return AnalyzerEngine(), AnonymizerEngine()
    except Exception:
        return None


def scrub_text(text: str, *, provider: str | None = None) -> str:
    """Return ``text`` with obvious PII replaced by ``<TYPE>`` tokens.

    When ``provider`` is "Ollama" (case-insensitive) the text is returned
    unchanged — local inference does not leave the trust boundary.
    """
    if not text or not is_enabled():
        return text
    if provider and provider.strip().lower() == "ollama":
        return text

    presidio = _presidio()
    if presidio is not None:  # pragma: no cover - exercised only when installed
        analyzer, anonymizer = presidio
        try:
            results = analyzer.analyze(text=text, language="en")
            return anonymizer.anonymize(text=text, analyzer_results=results).text
        except Exception as exc:
            _logger.warning("presidio_scrub_failed", reason=str(exc))

    out = text
    for label, pattern in _REGEX_PATTERNS:
        out = pattern.sub(f"<{label}>", out)
    return out
