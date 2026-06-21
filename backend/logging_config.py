"""Structured logging configuration with a PHI-aware redactor.

``configure_logging`` is safe to call multiple times; it is also idempotent
when invoked from tests. The redactor strips obvious PHI / secrets from log
events before they reach stdout, which is the primary HIPAA-driven concern
(HHS §164.312(b) audit controls require logs but explicitly not PHI).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Mapping
from typing import Any

try:
    import structlog
except Exception:  # pragma: no cover - optional dependency
    structlog = None  # type: ignore[assignment]


# Known-sensitive keys in event dicts
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "token",
        "structuring_api_key",
        "reasoning_api_key",
        "mrn",
        "dob",
        "full_name",
        "patient_id",
    }
)

# Pattern-based scrubbers. MRN redaction is intentionally NOT done here via a
# naive regex — a 6-12 alphanumeric pattern would false-positive on model
# names, correlation IDs, and git SHAs. MRNs are protected by (1) the
# ``_SENSITIVE_KEYS`` deny-list above (which catches any structured log field
# named ``mrn``) and (2) HMAC-peppered hashing at persistence time
# (see :func:`backend.security.hash_identifier`). Callers must never log
# raw MRN values in free-text messages.
_BEARER_PATTERN = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.IGNORECASE)
_OPENAI_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9]{10,}")
_ANTHROPIC_KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}")


def _scrub_value(value: Any) -> Any:
    if isinstance(value, str):
        scrubbed = _BEARER_PATTERN.sub(r"\1<REDACTED>", value)
        scrubbed = _OPENAI_KEY_PATTERN.sub("<REDACTED>", scrubbed)
        scrubbed = _ANTHROPIC_KEY_PATTERN.sub("<REDACTED>", scrubbed)
        return scrubbed
    if isinstance(value, Mapping):
        return {k: _scrub_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_scrub_value(v) for v in value)
    return value


def phi_redactor(logger, method_name, event_dict):
    """structlog processor that redacts sensitive keys / patterns."""
    for key in list(event_dict.keys()):
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = "<REDACTED>"
        else:
            event_dict[key] = _scrub_value(event_dict[key])
    return event_dict


_configured = False


def configure_logging(level: str | None = None) -> None:
    """Configure stdlib + structlog. Idempotent."""
    global _configured
    if _configured:
        return

    effective = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, effective, logging.INFO),
    )

    if structlog is not None:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                phi_redactor,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, effective, logging.INFO)),
            cache_logger_on_first_use=True,
        )
    _configured = True


class _StdlibStructShim:
    """Adapter that makes ``logging.Logger`` tolerant of structlog kwargs."""

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def _format(self, msg: str, kwargs: dict) -> str:
        if not kwargs:
            return msg
        redacted = phi_redactor(None, None, dict(kwargs))
        extras = " ".join(f"{k}={redacted[k]!r}" for k in redacted)
        return f"{msg} {extras}"

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(self._format(msg, kwargs), *args)

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(self._format(msg, kwargs), *args)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(self._format(msg, kwargs), *args)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(self._format(msg, kwargs), *args)

    def exception(self, msg: str, *args, **kwargs):
        self._logger.exception(self._format(msg, kwargs), *args)


def get_logger(name: str | None = None):
    if structlog is not None:
        return structlog.get_logger(name)
    return _StdlibStructShim(logging.getLogger(name or __name__))
