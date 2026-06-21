# Changelog

All notable changes to this project will be documented in this file.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and aims to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `docs/HANDBOOK.md` &mdash; zero-to-hero walkthrough of every major subsystem (definition &rarr; why &rarr; how &rarr; code &rarr; outputs).
- `docs/ARCHITECTURE.md` &mdash; module dependency graph, request lifecycle, and SQLite schema.
- `docs/SECURITY.md` &mdash; threat model, defense-in-depth layers, and operator hardening checklist.
- `docs/DEVELOPMENT.md` &mdash; lint, type-check, tests, CI, and "add a new backend / node / provider" recipes.

### Changed

- `README.md` &mdash; added "What is MediScan OCR?" elevator, "Why use it?" benefit table, and a documentation navigation block. Existing API reference and env-var tables preserved.
- `README.quickstart.md` &mdash; now points at `docs/HANDBOOK.md` for the full reference and at the per-subsystem docs for deep dives.
- `TEST_REPORT.md` &mdash; refreshed to reflect the post-eval-harness test inventory (241 pass / 8 skip on default markers, ~249 total).

### Removed

- **DeepSeek-OCR engine.** The codebase now uses GLM-OCR as the only Ollama-backed OCR engine. The `ollama` value is preserved as a transparent alias for `glm`. Removed `DEEPSEEK_PROMPTS` (`backend/ocr_backends/ollama_ocr.py`), the `deepseek-ocr` default in `DEFAULT_OCR_MODELS` (`backend/ocr_backends/base.py:11`), the DeepSeek entry from the Streamlit sidebar (`frontend/app.py:142`), the broken Qdrant live test that referenced removed symbols (`tests/integration/test_live_pipeline.py:251`), the deprecated model names in `tests/unit/test_main_unit.py`, and the DeepSeek mentions in the docs.

### Fixed

- `TEST_REPORT.md` no longer claims 34 tests / 97% coverage; that was the state of the suite before the eval harness and security hardening landed.
- **PDF page-cap enforcement.** `backend/extract.py:run_document_ocr` now reads `MEDISCAN_MAX_PDF_PAGES` (default 200) and rejects oversized PDFs via `pdfinfo_from_path` before pdf2image renders. The constant was previously defined but never read.
- **Qdrant live test.** `tests/integration/test_live_pipeline.py:test_live_qdrant_index_and_retrieve` now uses the real production surface (`QdrantRetrievalStore`, `build_chunks_from_text`, `upsert_chunks`, HMAC `hash_identifier`) instead of removed symbols (`QdrantVectorStore`, `OllamaEmbeddings`, plain SHA256).

## [2026-06-13]

### Added

- OSS companion documentation initialized (license, contributing, security, conduct, changelog).
