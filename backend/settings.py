"""Typed configuration via pydantic-settings.

All runtime configuration lives here. Environment variables are the source of
truth; this module validates them and provides a single ``Settings`` instance
for the rest of the application to import.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from typing import Union

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment and optional ``.env`` files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_prefix="MEDISCAN_",
    )

    # ---------------------------------------------------------------------
    # Server
    # ---------------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000

    # ---------------------------------------------------------------------
    # Security
    # ---------------------------------------------------------------------
    api_key: SecretStr | None = None
    allow_anonymous: bool = False
    allow_user_api_keys: bool = False
    allowed_origins: Union[str, list[str]] = Field(
        default_factory=lambda: ["http://localhost:8501", "http://127.0.0.1:8501"]
    )
    # Maps from MEDISCAN_RATE_LIMIT (legacy boolean toggle)
    enable_rate_limit: bool = Field(
        default=True,
        validation_alias="MEDISCAN_RATE_LIMIT",
    )
    # Maps from MEDISCAN_DEFAULT_RATE (legacy string)
    default_rate_limit: str = Field(
        default="60/minute",
        validation_alias="MEDISCAN_DEFAULT_RATE",
    )

    # Placeholder detection for the default docker-compose secret.
    @field_validator("api_key", mode="before")
    @classmethod
    def _reject_placeholder(cls, v: str | SecretStr | None) -> SecretStr | None:
        if v is None:
            return None
        val = v.get_secret_value() if isinstance(v, SecretStr) else str(v)
        if val.strip().lower() in {"changeme", "change-me"}:
            raise ValueError(
                "MEDISCAN_API_KEY is set to the placeholder 'changeme'. Generate a "
                "real secret (e.g. `python -c \"import secrets; print(secrets.token_urlsafe(32))\"`) "
                "and set it via the environment before starting the server."
            )
        return SecretStr(val)

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _parse_allowed_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    # ---------------------------------------------------------------------
    # Upload / storage
    # ---------------------------------------------------------------------
    upload_root: Path = Path("./backend/uploads")
    max_upload_mb: int = 50
    max_pdf_pages: int = 200

    @field_validator("upload_root", mode="before")
    @classmethod
    def _expand_upload_root(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    # ---------------------------------------------------------------------
    # OCR
    # ---------------------------------------------------------------------
    ocr_default_backend: str = "glm"
    ocr_default_model: str = "glm-ocr"
    ocr_default_mode: str = "text"
    use_gpu: bool = True
    paddle_service_url: str | None = None

    # ---------------------------------------------------------------------
    # LLM providers (server-side keys always take precedence)
    # ---------------------------------------------------------------------
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None

    # ---------------------------------------------------------------------
    # Observability
    # ---------------------------------------------------------------------
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "console"
    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str | None = None

    # ---------------------------------------------------------------------
    # Docs
    # ---------------------------------------------------------------------
    enable_docs: bool = False

    # ---------------------------------------------------------------------
    # Internal / advanced
    # ---------------------------------------------------------------------
    secrets_dir: Path | None = None

    def resolve_secret(self, name: str) -> str | None:
        """Resolve a secret from the secrets directory or environment."""
        # 1. secrets_dir/<name>
        if self.secrets_dir:
            secret_path = self.secrets_dir / name
            if secret_path.is_file():
                return secret_path.read_text().strip()
        # 2. environment variable
        env_val = os.environ.get(name)
        if env_val:
            return env_val
        # 3. fall back to the corresponding field on this instance
        return getattr(self, name.lower(), None)


# Singleton instance (populated at import time).
settings = Settings()

# Convenience alias for legacy aliases used by existing code during the migration.
API_KEY = settings.api_key.get_secret_value() if settings.api_key else None
ALLOW_ANONYMOUS = settings.allow_anonymous
ALLOWED_ORIGINS = settings.allowed_origins
UPLOAD_ROOT = settings.upload_root
MAX_UPLOAD_MB = settings.max_upload_mb
MAX_PDF_PAGES = settings.max_pdf_pages
PADDLE_SERVICE_URL = settings.paddle_service_url