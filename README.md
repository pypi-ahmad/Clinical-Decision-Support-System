# MediScan OCR

Implementation-accurate documentation for the current repository.

For a shorter startup path, see `README.quickstart.md`.

## Overview

MediScan OCR is a FastAPI + Streamlit application for medical document intake, structuring, review, reasoning, insurance checks, and persistence.

The current repository implements:

- Multi-page PDF and image ingestion.
- Pluggable OCR backends.
- Separate OCR, structuring, and reasoning model configuration.
- Optional LangGraph-based extraction orchestration.
- SQLite persistence for confirmed records.
- Qdrant-first semantic retrieval with exact metadata filters.
- Annotated artifact generation and frontend previews/downloads.

## Current Architecture

- Frontend: `frontend/app.py`
- API orchestration: `backend/main.py`
- OCR/extraction orchestration: `backend/extract.py`
- OCR backend package: `backend/ocr_backends/`
- Artifact rendering and URLs: `backend/artifacts.py`
- Reasoning and insurance logic: `backend/logic.py`
- Provider adapter: `backend/ai_wrapper.py`
- Retrieval package: `backend/retrieval/`
- Granular extraction graph: `backend/workflows/extraction_graph.py`
- First-gen agentic workflow: `backend/workflows/agentic_extraction.py`
- Persistence: `backend/database.py`

```mermaid
flowchart LR
    UI[Streamlit\nfrontend/app.py] -->|/analyze| API[FastAPI\nbackend/main.py]
    UI -->|/check_insurance| API
    UI -->|/confirm| API

    API --> EXT[Direct pipeline\nbackend/extract.py]
    API --> EG[Extraction graph\nbackend/workflows/extraction_graph.py]
    API --> WF[Agentic workflow\nbackend/workflows/agentic_extraction.py]
    API --> LOG[Reasoning\nbackend/logic.py]
    API --> RET[Retrieval\nbackend/retrieval]
    API --> DB[SQLite\nbackend/database.py]

    EXT --> OCR[OCR Backends\nollama | glm | paddle]
    EXT --> AIW[AI Wrapper\nbackend/ai_wrapper.py]
    EG --> OCR
    EG --> AIW
    LOG --> AIW
    RET --> QDR[(Qdrant)]
```

## Analyze Flow

`POST /analyze` now works as follows:

1. Save the upload under `backend/uploads/` with a UUID prefix.
2. Render all pages for PDFs into `*_artifacts/pages/`.
3. Run the selected OCR backend.
4. Build a typed OCR payload with:
   - `per_page_results`
   - `raw_text`
   - `markdown`
   - `structured_doc` when the backend provides one
   - `bounding_boxes`
   - `artifact_manifest`
5. Send OCR output to the selected structuring model.
6. Load latest exact relational history from SQLite by MRN.
7. Optionally retrieve semantic context from Qdrant using exact filters on `patient_id_hash` and `source_type`.
8. Run the selected reasoning model.
9. Index the current document into retrieval storage when a vector store is configured.

The response includes:

- `extracted`
- `analysis`
- `history_available`
- `file_path`
- `file_url`
- `ocr`
- `bounding_boxes`
- `annotated_pdf_path`
- `annotated_pdf_url`
- `annotated_image_paths`
- `annotated_image_urls`
- `page_image_urls`
- `requires_human_review`
- `vector_index_status`

## Insurance Flow

`POST /check_insurance` supports:

- `medical_json`
- separate reasoning provider/model/api key
- OCR settings for policy OCR
- `policy_ocr=true` to run OCR over the uploaded policy before reasoning

The endpoint still accepts plain UTF-8 policy text directly. When retrieval is enabled, the policy text is chunked, indexed, and queried with exact filters on `source_type=insurance_policy` and `policy_document_id`.

## OCR Backends

The OCR layer lives under `backend/ocr_backends/`.

| Backend | File | Implemented behavior |
|---|---|---|
| Ollama DeepSeek OCR | `backend/ocr_backends/ollama_ocr.py` | Multi-page OCR using Ollama chat, text-focused prompts, no native boxes |
| Ollama GLM OCR | `backend/ocr_backends/ollama_ocr.py` | Multi-page OCR using GLM prompt modes such as `text`, `table`, and `figure` |
| PaddleOCR-VL local Python | `backend/ocr_backends/paddleocr_vl.py` | Uses local PaddleOCR-VL runtime, captures markdown/JSON, extracts boxes when present |
| PaddleOCR-VL local service | `backend/ocr_backends/service_client.py` | Healthcheck + timeout wrapper around Paddle service mode |

Common OCR output is normalized through `OCRResult`, `OCRPageResult`, and `OCRBoundingBox` in `backend/ocr_backends/base.py`.

## Retrieval

The retrieval package is now split into small modules:

- `backend/retrieval/chunking.py`
- `backend/retrieval/embeddings.py`
- `backend/retrieval/vector_store.py`
- `backend/retrieval/qdrant_store.py`
- `backend/retrieval/pgvector_store.py`

Current behavior:

- Qdrant is the active store.
- pgvector is scaffolded but intentionally not wired for live writes/search yet.
- Documents are indexed by chunks with metadata such as:
  - `patient_id_hash`
  - `encounter_date`
  - `source_type`
  - `page_number`
  - `source_ref`
  - `ocr_backend`

Exact relational filters are preserved. Vector search is additive context, not a replacement for the SQLite MRN history lookup.

## Agentic Extraction

The `backend/workflows/` package provides two LangGraph workflow options:

### Granular extraction graph (`backend/workflows/extraction_graph.py`)

The default workflow selected from the UI via "Granular extraction graph". Implements the full
step-by-step graph shape with one node per processing stage:

| Node | Responsibility |
|---|---|
| `ingest_document` | Verify file exists and is readable |
| `classify_document_type` | Heuristically classify doc type from filename |
| `split_pages` | Pre-stage before OCR (page split happens inside OCR node) |
| `ocr_per_page` | Run selected OCR backend across all pages |
| `extract_candidate_fields` | Structuring LLM call to extract JSON fields |
| `validate_against_schema` | Validate against `MedicalRecord` Pydantic schema |
| `normalize_codes` | Normalize ICD codes, diagnosis punctuation, medication casing |
| `retrieve_context` | Load SQLite history + Qdrant vector context |
| `merge_document_record` | Reasoning LLM call with combined context |
| `confidence_gate` | Compute confidence score; route to human_review if low |
| `human_review` | Flag document for manual review (passthrough in automated path) |
| `persist_record` | Index document to vector store |

The graph uses a typed `ExtractionGraphState` TypedDict so every intermediate value
is visible for inspection, replay, or debugging.

### First-gen workflow (`backend/workflows/agentic_extraction.py`)

An earlier coarser-grained workflow. Uses larger composite nodes. Still available via "Agentic workflow" UI option.

Both workflows support the same split OCR / structuring / reasoning configuration.

### API parameter

| `POST /analyze` parameter | Value | Behavior |
|---|---|---|
| `extraction_graph_mode=true` | true | Runs granular extraction graph |
| `agentic_mode=true` | true | Runs first-gen workflow |
| both false | – | Runs direct single-call pipeline |

## Frontend Behavior

The Streamlit app now exposes:

- **OCR backend and mode**: select DeepSeek-OCR, GLM-OCR, PaddleOCR-VL (local Python or service).
- **Extraction workflow**: radio selector between Direct pipeline, Granular extraction graph, and Agentic workflow.
- **Structuring provider/model**: separate provider + model for the structuring LLM.
- **Reasoning provider/model**: separate provider + model for clinical reasoning.

The document review area supports:

- Original artifact preview (PDF inline or image)
- Annotated artifact preview (bounding boxes drawn over page images)
- Page-by-page original vs annotated overlay comparison
- Artifact downloads (original and annotated PDF)
- Bounding box table (page number, polygon, label, confidence)

## Setup

### Base requirements

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Base install includes FastAPI, Streamlit, Ollama client, LangGraph, Qdrant client, Pillow, and test dependencies.

### PDF prerequisites

`pdf2image` requires Poppler to be installed and available on the host.

### Optional Qdrant setup

Set one of the following:

- `QDRANT_ENABLED=true`
- `QDRANT_URL=http://localhost:6333`

Optional:

- `QDRANT_API_KEY`
- `VECTOR_STORE=qdrant`
- `OLLAMA_EMBED_MODEL=nomic-embed-text`

### Optional PaddleOCR-VL setup

PaddleOCR-VL is intentionally optional because it has heavier runtime requirements.

For local Python mode, install PaddleOCR-VL support:

```bash
python -m pip install "paddleocr[doc-parser]"
```

For Windows GPU mode, install a compatible PaddlePaddle GPU wheel for your Python and CUDA combination. The code will use:

- local Python mode when `ocr_backend=paddle` and no service URL is provided
- local service mode when `ocr_backend=paddle` and `paddle_service_url` is set

### Optional Paddle service mode

Run a compatible local service and point the UI or API at it with `paddle_service_url`, for example:

```text
http://127.0.0.1:8118/v1
```

## Running the App

Start the backend:

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

Start the frontend in a second terminal:

```bash
streamlit run frontend/app.py
```

## Testing

Run the full test suite:

```bash
python -m pytest
```

Focused validation examples:

```bash
python -m pytest tests/unit/test_extract.py
python -m pytest tests/unit/test_logic.py tests/unit/test_main_unit.py tests/integration/test_api_workflows.py
```

## Current Limitations

- Qdrant is the only live retrieval backend. `backend/retrieval/pgvector_store.py` is a scaffold, not a production path yet.
- Ollama OCR backends do not emit native bounding boxes, so overlays are empty for Ollama/GLM paths.
- SQLite history returns the latest prior record by MRN, not a full longitudinal timeline.
- PaddleOCR-VL runtime setup requires compatible Paddle packages and, for GPU, a matching Paddle GPU wheel.
- The API has no authentication or authorization and CORS is fully open.
- Binary/non-UTF8 policy documents are not OCR-processed in the `/check_insurance` endpoint without enabling `policy_ocr=true`.
- No explicit retry/backoff mechanism for provider/API failures.
- LangGraph `human_review` node is a passthrough in the automated path; production deployments should emit a task/ticket there.

## Future Improvements (code-implied)

The following are direct extensions of current constraints:

- Wire pgvector as a live retrieval backend (schema, adapter, and migration).
- Add a real human-review pause point in `human_review` node (emit ticket/task, await callback).
- Add provider/model selection support to `/check_insurance` in a dedicated UI panel.
- Introduce request-body Pydantic models on API endpoints for stronger schema validation.
- Tighten CORS and add authentication for non-local deployments.
- Add ICD-10 code normalization via a lookup table in `normalize_codes_node`.
- Add longitudinal history view (full timeline per MRN) rather than latest-record-only.

## Project Structure

```text
mediscan-ocr/
├─ backend/
│  ├─ ai_wrapper.py
│  ├─ artifacts.py
│  ├─ database.py
│  ├─ extract.py
│  ├─ logic.py
│  ├─ main.py
│  ├─ models.py
│  ├─ ocr.py
│  ├─ ocr_backends/
│  │  ├─ base.py
│  │  ├─ ollama_ocr.py
│  │  ├─ paddleocr_vl.py
│  │  └─ service_client.py
│  ├─ retrieval/
│  │  ├─ chunking.py
│  │  ├─ embeddings.py
│  │  ├─ pgvector_store.py
│  │  ├─ qdrant_store.py
│  │  └─ vector_store.py
│  ├─ workflows/
│  │  ├─ agentic_extraction.py
│  │  └─ extraction_graph.py
│  └─ uploads/
├─ frontend/
│  └─ app.py
├─ tests/
│  ├─ integration/
│  │  └─ test_api_workflows.py
│  └─ unit/
│     ├─ test_ai_wrapper.py
│     ├─ test_database.py
│     ├─ test_extract.py
│     ├─ test_extraction_graph.py
│     ├─ test_logic.py
│     ├─ test_main_unit.py
│     ├─ test_models.py
│     ├─ test_ocr_backends.py
│     └─ test_retrieval.py
├─ pytest.ini
├─ requirements.txt
└─ README.md
```
