# Test Report

> **Last refreshed.** 2026-06-22. Generated from the current `tests/`
> tree and the live pytest run that ships with the README badge
> (`241 passed, 8 skipped` on `pytest -m "not live and not eval"`).

## 1. Codebase snapshot

| Module | File | Public surface |
|---|---|---|
| FastAPI entry | `backend/main.py` | Routes `/analyze`, `/check_insurance`, `/confirm`, `/artifacts/*`, `/review/*`, `/health`, `/ready`, `/metrics` |
| Auth &amp; upload | `backend/security.py` | `require_api_key`, `validate_upload_or_raise`, `write_upload_with_limit`, `validate_outbound_url`, prompt firewall |
| Extraction | `backend/extract.py` | `process_document_pipeline`, `run_document_ocr` |
| Workflows | `backend/workflows/extraction_graph.py`, `backend/workflows/agentic_extraction.py` | 12-node granular + first-gen agentic LangGraph |
| LLM adapter | `backend/ai_wrapper.py` | `get_ai_response`, `parse_model_json` (multi-provider + balanced-brace JSON parser) |
| Reasoning | `backend/logic.py` | `analyze_medical_logic`, `check_insurance_coverage` |
| Persistence | `backend/database.py` | `save_record`, `get_patient_history`, `record_audit_event`, `enqueue_review_task`, `list_pending_reviews`, `resolve_review_task` |
| Retrieval | `backend/retrieval/*` | HMAC hash, chunker, embeddings, Qdrant store, ABC |
| Observability | `backend/observability.py` | Prometheus metrics, request-id middleware, OTel hook |
| Lineage | `backend/lineage.py` | Git SHA + library versions + OCR/LLM identifiers |
| Logging | `backend/logging_config.py` | Structlog + PHI redactor |
| Models | `backend/models.py` | `MedicalRecord` (forgiving) + `MedicalRecordStrict` (persistence boundary) |
| Frontend | `frontend/app.py` | Streamlit operator console |

## 2. Test inventory

| Path | Count | Purpose |
|---|---|---|
| `tests/unit/test_ai_wrapper.py` | 7 | Provider dispatch, retries, JSON parser, scrubber, error scrubbing |
| `tests/unit/test_agentic_extraction.py` | ~12 | First-gen LangGraph node behavior |
| `tests/unit/test_database.py` | 5 | Save, history, audit, review queue, WAL |
| `tests/unit/test_extract.py` | 5 | Direct pipeline: OCR + structuring happy and error paths |
| `tests/unit/test_extraction_graph.py` | ~22 | 12-node graph: routing, confidence gate, human review, retrieval fallback |
| `tests/unit/test_logic.py` | 4 | Reasoning + insurance prompts, stable failure schema |
| `tests/unit/test_main_unit.py` | 7 | Auth, upload, /confirm payload validation, route signatures |
| `tests/unit/test_models.py` | 2 | Field truncation, date coercion, extra-key handling |
| `tests/unit/test_observability.py` | ~5 | Prometheus metric registration, request-id middleware, LLM call recorder |
| `tests/unit/test_ocr_backends.py` | ~25 | Backend dispatch, GLM prompt modes, Paddle local &amp; service mocks, bbox aggregation |
| `tests/unit/test_retrieval.py` | ~22 | HMAC hash, chunker, embeddings, Qdrant store (mocked), factory selection |
| `tests/integration/test_api_workflows.py` | ~12 | FastAPI `TestClient` end-to-end happy and error paths |
| `tests/integration/test_live_pipeline.py` | 8 (all auto-skip without services) | Real Ollama + Qdrant + multi-page PDF |
| `tests/eval/test_metrics.py` | ~10 | CER / WER / set-F1 / exact-match correctness |
| `tests/eval/test_extraction_quality.py` | 0 active (skips when `tests/eval/gold/` is empty) | Quality harness against gold fixtures |
| **Total** | **~249** (241 pass · 8 skip on default markers) | |

## 3. Test tiers

| Tier | Markers | Skip rule | What it proves |
|---|---|---|---|
| Mocked unit + integration | *(none)* | Never | Code paths and contracts are correct |
| Live | `live` | Skips when Ollama / Qdrant / PaddleOCR unreachable | Real model output and retrieval recall |
| Eval | `eval` | Skips when `tests/eval/gold/*.json` is empty | CER, set-F1, exact-match against curated fixtures |

> A green mocked suite proves the wiring is correct. Run live tests
> against real services to prove end-to-end behavior.

## 4. Last full run

```
$ python -m pytest -m "not live and not eval"
============================= 241 passed, 8 skipped in 38.96s ==============================
```

Backend coverage: **~81%** (varies with the surface of the most recent
change). Modules that are deliberately excluded or only loosely
covered:

- `backend/pii_scrub.py` &mdash; the Presidio branch is best-effort
  and only runs when Presidio is installed; covered via regex fallback
- `backend/observability.py` &mdash; Prometheus exporter wiring is
  `pragma: no cover` because the global `prometheus_client` registry
  has process-wide state
- `backend/workflows/extraction_graph.py` `_human_review_node` and
  `_persist_record_node` &mdash; error paths are covered; the success
  path is integration-tested

## 5. CI behavior

`.github/workflows/ci.yml` runs:

1. **lint** &mdash; `ruff check .` and `ruff format --check .`
2. **type-check** &mdash; `mypy` on `backend/security.py`,
   `backend/database.py`, `backend/retrieval/*`
3. **test** &mdash; `pytest -q -m "not live and not eval"` with
   `MEDISCAN_ALLOW_ANONYMOUS=1`
4. **audit** &mdash; `pip-audit --strict -r requirements.txt || true`
   (non-blocking, surfaces advisories for triage)

The latest CI run was green on the `main` branch.

## 6. Refreshing this report

After any change to `tests/`, regenerate the totals with:

```bash
python -m pytest -m "not live and not eval" --collect-only -q
python -m pytest -m "not live and not eval" --cov=backend --cov-report=term
```

Update the counts in &sect;2 and the badge totals in the top-level
`README.md` to match.
