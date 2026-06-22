"""MediScan typed exception hierarchy.

Every error raised by application code inherits from :class:`MediscanError`,
which carries an HTTP ``status_code`` and a ``correlation_id`` so that route
handlers can always return structured JSON instead of ad-hoc ``{"error": ..}``
dicts.
"""

from __future__ import annotations

import uuid
from typing import Any


class MediscanError(Exception):
    """Base exception for the entire MediScan application."""

    status_code: int = 500
    correlation_id: str | None = None

    def __init__(
        self,
        message: str = "",
        *,
        status_code: int | None = None,
        correlation_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        if status_code is not None:
            self.status_code = status_code
        self.correlation_id = correlation_id or uuid.uuid4().hex[:12]
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "error": self.__class__.__name__,
            "message": str(self),
            "correlation_id": self.correlation_id,
            "status_code": self.status_code,
        }
        if self.details:
            base["details"] = self.details
        return base


# ---------------------------------------------------------------------------
# Domain-specific exceptions
# ---------------------------------------------------------------------------


class AIProviderError(MediscanError):
    """One of the upstream LLM providers (OpenAI, Anthropic, Gemini) failed."""

    status_code = 502


class OCRBackendError(MediscanError):
    """OCR extraction failed (PaddleOCR, GLM-OCR, or the aggregator)."""

    status_code = 502


class RetrievalError(MediscanError):
    """Qdrant vector-store operation failed."""

    status_code = 502


class ValidationError(MediscanError):
    """Input validation or precondition check failed."""

    status_code = 422


class MigrationError(MediscanError):
    """Database schema migration error."""

    status_code = 500


class NotFoundError(MediscanError):
    """Resource not found."""

    status_code = 404


class AuthenticationError(MediscanError):
    """API key authentication failure."""

    status_code = 401


class RateLimitError(MediscanError):
    """Rate limit exceeded."""

    status_code = 429
