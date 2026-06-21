"""Global pytest configuration.

Sets environment variables required by the hardened code paths so tests run
with predictable behaviour:

* ``MEDISCAN_ALLOW_ANONYMOUS=1`` disables the auth dependency when tests hit
  FastAPI routes directly (``TestClient``).
* ``MRN_HMAC_PEPPER`` enables deterministic ``hash_identifier`` output.
* ``MEDISCAN_LLM_RETRIES=1`` stops the Tenacity retry loop from slowing the
  negative-path LLM tests.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MEDISCAN_ALLOW_ANONYMOUS", "1")
os.environ.setdefault("MRN_HMAC_PEPPER", "test-pepper-not-for-production")
os.environ.setdefault("MEDISCAN_LLM_RETRIES", "1")
os.environ.setdefault("MEDISCAN_ENABLE_DOCS", "0")
os.environ.setdefault("MEDISCAN_ALLOWED_ORIGINS", "http://testserver")


@pytest.fixture(autouse=True)
def _reset_retrieval_cache():
    """Clear the vector-store singleton between tests so monkeypatches bite."""
    from backend import retrieval

    retrieval.reset_vector_store()
    yield
    retrieval.reset_vector_store()


@pytest.fixture(autouse=True)
def _reset_paddle_pipeline_cache():
    """Clear the Paddle pipeline LRU so each test gets a fresh construction."""
    try:
        from backend.ocr_backends.paddleocr_vl import _cached_local_pipeline

        _cached_local_pipeline.cache_clear()
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def _reset_database_connection(tmp_path, monkeypatch):
    """Point the DB at a per-test file and reset the cached connection."""
    import backend.database as database

    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "records.db"))
    monkeypatch.setattr(database, "_conn", None)
    yield
    if database._conn is not None:
        try:
            database._conn.close()
        except Exception:
            pass
        monkeypatch.setattr(database, "_conn", None)


@pytest.fixture(autouse=True)
def _reset_provider_clients():
    """Drop cached LLM provider clients between tests so monkeypatches bite."""
    import backend.ai_wrapper as ai_wrapper

    ai_wrapper.reset_provider_clients()
    yield
    ai_wrapper.reset_provider_clients()
