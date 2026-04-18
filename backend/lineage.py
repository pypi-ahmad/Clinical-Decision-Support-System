"""Data-lineage helpers.

Every persisted record and every audit event should carry enough provenance
information to be re-derived or revoked. The :func:`capture_lineage` helper
collects the ambient context — git commit, backend versions, model names —
into a compact dict that callers serialise into the record.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from typing import Any


def _safe_call(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    return out or None


@lru_cache(maxsize=1)
def git_sha() -> str | None:
    """Return the short git SHA if available, else ``None``.

    Cached: a running server's git SHA does not change mid-process.
    """
    env = os.environ.get("MEDISCAN_GIT_SHA")
    if env:
        return env.strip()[:12] or None
    # ``git rev-parse`` may fail on a deploy artifact without a git dir.
    return (_safe_call(["git", "rev-parse", "--short=12", "HEAD"]) or "").strip() or None


@lru_cache(maxsize=1)
def package_version(pkg: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(pkg)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def capture_lineage(
    *,
    ocr_backend: str | None = None,
    ocr_model: str | None = None,
    structuring_provider: str | None = None,
    structuring_model: str | None = None,
    reasoning_provider: str | None = None,
    reasoning_model: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the lineage dict embedded into records + audit events."""
    lineage: dict[str, Any] = {
        "git_sha": git_sha(),
        "app_version": package_version("mediscan-cdss"),
        "fastapi_version": package_version("fastapi"),
        "langgraph_version": package_version("langgraph"),
        "qdrant_client_version": package_version("qdrant-client"),
        "ocr_backend": ocr_backend,
        "ocr_model": ocr_model,
        "structuring_provider": structuring_provider,
        "structuring_model": structuring_model,
        "reasoning_provider": reasoning_provider,
        "reasoning_model": reasoning_model,
    }
    if extra:
        lineage.update({k: v for k, v in extra.items() if v is not None})
    return {k: v for k, v in lineage.items() if v is not None}
