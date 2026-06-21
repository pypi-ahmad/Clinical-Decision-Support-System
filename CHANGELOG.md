# Changelog

All notable changes to this project will be documented in this file.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and aims to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **uv toolchain end-to-end.** `pyproject.toml` is now the single source of truth for project metadata, runtime deps, and `[dependency-groups]` (PEP 735) for dev/test. PEP 621 license (`MIT`), classifiers, `requires-python = ">=3.12.10,<3.13"`, and `uv_build` build backend. `uv.lock` (3 136 lines, fully resolved) is the only lockfile.
- **Python 3.12.10 pin.** `.python-version` pins the toolchain version; `uv` creates the venv from that pin.
- **Makefile canonical task runner.** `make install`, `sync`, `install-dev`, `lock`, `upgrade`, `lint`, `format`, `typecheck`, `test`, `test-fast`, `test-cov`, `test-live`, `test-eval`, `test-property`, `test-contract`, `audit`, `pre-commit`, `run-api`, `run-frontend`, `release-check`, `clean`, `help`.
- **Pre-commit refresh.** Ruff + ruff-format + housekeeping + a local `uv lock --check` hook that prevents pyproject/lock drift.
- **CI rewritten for uv.** `.github/workflows/ci.yml` uses `astral-sh/setup-uv@v5` with `uv.lock` cache, Python 3.12.10 install, `uv sync --frozen`, `uv run --frozen pytest`, a `test-live` job (push to `main` only), and a `pip-audit --strict` security job gated on the rest.
- **Production Dockerfile.** Multi-stage build (`ghcr.io/astral-sh/uv:python3.12-bookworm-slim` builder, `python:3.12-slim` runtime, `uv sync --frozen --no-dev`), non-root `mediscan` user, healthcheck on `/health`, gunicorn-style uvicorn entrypoint.
- **Pyright installed and configured.** Dev dep + `[tool.pyright]` block (`typeCheckingMode = "basic"`, Python 3.12, `include = ["backend/"]`). The strict type-clean sweep is delivered by Phase B.
- **Live test path confinement fix.** `tests/integration/test_live_pipeline.py` now sets `MEDISCAN_UPLOAD_ROOT` via `monkeypatch` in every live test that touches the upload path, so the test image under `tmp_path` is no longer rejected by the upload-root confinement check.
- **Stale error-string assertion fixed.** `tests/unit/test_extract.py::test_process_document_pipeline_ocr_failure` now asserts the post-P0 `"GLM-OCR failed"` string instead of the legacy `"Ollama OCR failed"`.

### Changed

- **Lint selector narrowed to safe core for Phase A.** `pyproject.toml` ruff config selects only `E`, `F`, `W`, `I`, `UP` so the existing codebase passes the gate. The full `B` / `SIM` / `RUF` / `N` / `S` / `PIE` / `TID` / `FA` selector is delivered by Phase B together with the legacy code sweep.
- **pytest config consolidated.** `pyproject.toml [tool.pytest.ini_options]` is now the only source; the legacy `pytest.ini` is deleted.
- **`make typecheck` runs mypy + Pyright.** Both checks fire so the dev workflow exercises them; CI runs them as soft checks in Phase A.

### Removed

- `requirements.txt` and `requirements.lock.txt` (uv now owns dep resolution end-to-end).
- `pytest.ini` (config consolidated into `pyproject.toml`).
- **DeepSeek-OCR engine.** The codebase now uses GLM-OCR as the only Ollama-backed OCR engine. The `ollama` value is preserved as a transparent alias for `glm`. Removed `DEEPSEEK_PROMPTS` (`backend/ocr_backends/ollama_ocr.py`), the `deepseek-ocr` default in `DEFAULT_OCR_MODELS` (`backend/ocr_backends/base.py:11`), the `DeepSeek entry from the Streamlit sidebar (`frontend/app.py:142`), the broken Qdrant live test that referenced removed symbols (`tests/integration/test_live_pipeline.py:251`), the deprecated model names in `tests/unit/test_main_unit.py`, and the DeepSeek mentions in the docs.

### Fixed

- `TEST_REPORT.md` no longer claims 34 tests / 97% coverage; that was the state of the suite before the eval harness and security hardening landed.
- **PDF page-cap enforcement.** `backend/extract.py:run_document_ocr` now reads `MEDISCAN_MAX_PDF_PAGES` (default 200) and rejects oversized PDFs via `pdfinfo_from_path` before pdf2image renders. The constant was previously defined but never read.
- **Qdrant live test.** `tests/integration/test_live_pipeline.py:test_live_qdrant_index_and_retrieve` now uses the real production surface (`QdrantRetrievalStore`, `build_chunks_from_text`, `upsert_chunks`, HMAC `hash_identifier`) instead of removed symbols (`QdrantVectorStore`, `OllamaEmbeddings`, plain SHA256).

### Docs

- `docs/HANDBOOK.md` &mdash; zero-to-hero walkthrough of every major subsystem (definition &rarr; why &rarr; how &rarr; code &rarr; outputs).
- `docs/ARCHITECTURE.md` &mdash; module dependency graph, request lifecycle, and SQLite schema.
- `docs/SECURITY.md` &mdash; threat model, defense-in-depth layers, and operator hardening checklist.
- `docs/DEVELOPMENT.md` &mdash; lint, type-check, tests, CI, and "add a new backend / node / provider" recipes.
- `README.md` &mdash; added "What is MediScan OCR?" elevator, "Why use it?" benefit table, and a documentation navigation block. Existing API reference and env-var tables preserved.
- `README.quickstart.md` &mdash; now points at `docs/HANDBOOK.md` for the full reference and at the per-subsystem docs for deep dives.

## [2026-06-13]

### Added

- OSS companion documentation initialized (license, contributing, security, conduct, changelog).
