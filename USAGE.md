# Usage Guide

## About

MediScan OCR is a clinical decision-support system that automates medical document intake, structured data extraction, clinical reasoning, and insurance eligibility verification. It combines pluggable OCR backends, large language model orchestration, and semantic retrieval to transform unstructured medical records into actionable, machine-readable clinical data.

### Core Capabilities

| Capability | Description |
|---|---|
| **Document Ingestion** | Multi-page PDF and image upload with automatic page rendering and artifact management. |
| **Pluggable OCR** | Interchangeable OCR backends — GLM-OCR (default), DeepSeek-OCR, and PaddleOCR-VL — with configurable prompt modes. |
| **Structured Extraction** | LLM-driven extraction of patient demographics, vitals, diagnoses (with ICD codes), medications, and clinical notes into a normalized JSON schema. |
| **Clinical Reasoning** | Automated trend detection, alert generation, and narrative summarization powered by a separately configurable reasoning model. |
| **Insurance Verification** | Policy document ingestion (plain text or OCR), semantic comparison against extracted diagnoses, and eligibility determination with explainable reasoning. |
| **Semantic Retrieval** | Qdrant-backed vector search with exact metadata filters, enabling cross-document context retrieval by patient and encounter. |
| **Workflow Orchestration** | Three extraction modes — direct pipeline, granular 12-node LangGraph, and agentic LangGraph — selectable per request. |
| **Artifact Generation** | Bounding box overlays, annotated PDFs/images, and downloadable artifacts for audit and review. |

### System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Streamlit Frontend (frontend/app.py)                                │
│  OCR config ─ Workflow selector ─ Model config ─ Document preview    │
└──────────────┬───────────────────┬──────────────────┬────────────────┘
               │ /analyze          │ /check_insurance  │ /confirm
               ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FastAPI Backend (backend/main.py)                                   │
│  ┌────────────┐  ┌──────────────────────┐  ┌──────────────────────┐ │
│  │  Direct     │  │  Extraction Graph    │  │  Agentic Workflow    │ │
│  │  Pipeline   │  │  (12-node LangGraph) │  │  (LangGraph)         │ │
│  └─────┬──────┘  └──────────┬───────────┘  └──────────┬───────────┘ │
│        └─────────────┬──────┴──────────────────┬──────┘             │
│                      ▼                         ▼                    │
│  ┌──────────────────────────┐  ┌───────────────────────────┐        │
│  │  OCR Backends            │  │  Retrieval                │        │
│  │  GLM-OCR (default)       │  │  Qdrant (active)          │        │
│  │  DeepSeek / PaddleOCR-VL │  │  pgvector (scaffold)      │        │
│  └──────────────────────────┘  └───────────────────────────┘        │
│                                        │                            │
│  ┌─────────────────────────────────────┴──────────────────────────┐ │
│  │  SQLite (persistence)  ─  AI Wrapper (provider adapter)       │ │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

| Dependency | Required | Purpose |
|---|---|---|
| Python 3.11+ | Yes | Runtime |
| [Poppler](https://poppler.freedesktop.org/) | Yes | PDF page rendering via `pdf2image` |
| [Ollama](https://ollama.com/) | Yes (for Ollama backends) | Local LLM inference for OCR, structuring, and reasoning |
| [Qdrant](https://qdrant.tech/) | Optional | Semantic vector retrieval |
| PaddleOCR-VL | Optional | Alternative OCR backend with native bounding box support |

## Installation

```bash
# Clone and enter the repository
git clone https://github.com/pypi-ahmad/Clinical-Decision-Support-System.git
cd Clinical-Decision-Support-System

# Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Optional: PaddleOCR-VL

```bash
python -m pip install "paddleocr[doc-parser]"
```

### Optional: Qdrant Retrieval

```powershell
$env:QDRANT_URL = "http://localhost:6333"
$env:QDRANT_ENABLED = "true"
```

## Starting the Application

### Set required secrets

The backend rejects every data route unless an API key is configured (or you explicitly opt into anonymous local-dev mode). Retrieval and audit linkage also require an HMAC pepper.

```powershell
# Windows PowerShell
$env:MEDISCAN_API_KEY = "change-me-to-a-long-random-string"
$env:MRN_HMAC_PEPPER  = "long-random-server-only-secret"

# Optional (local development only — SKIPS AUTH, do not use in production):
# $env:MEDISCAN_ALLOW_ANONYMOUS = "1"
```

```bash
# Linux / macOS
export MEDISCAN_API_KEY="change-me-to-a-long-random-string"
export MRN_HMAC_PEPPER="long-random-server-only-secret"
```

The Streamlit frontend automatically reads `MEDISCAN_API_KEY` from its own environment and attaches it as the `X-API-Key` header on every backend call. If you start Streamlit in a separate shell, set the same variable there.

**Terminal 1 — Backend:**

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**

```bash
streamlit run frontend/app.py
```

The frontend is accessible at `http://localhost:8501`. The API serves at `http://localhost:8000`.

## Workflow

### 1. Configure the Sidebar

All configuration is performed in the Streamlit sidebar before document upload.

#### OCR Backend

Select the engine used for optical character recognition.

| Option | Engine | Bounding Boxes | Default | Notes |
|---|---|---|---|---|
| **GLM-OCR (Ollama)** | Ollama + GLM-OCR prompts | No | **Yes** | Supports text, table, figure, formula, chart prompt modes |
| DeepSeek-OCR (Ollama) | Ollama + DeepSeek prompts | No | — | Text-focused, fast |
| PaddleOCR-VL-1.5 (Local Python) | Local PaddleOCR-VL runtime | Yes | — | Requires `paddleocr[doc-parser]` install |
| PaddleOCR-VL-1.5 (Local Service) | HTTP service client | Yes | — | The service URL is **server-configured only** via the `PADDLE_SERVICE_URL` environment variable — clients cannot override it |

#### OCR Mode

The sidebar exposes the full prompt-mode list supported by the GLM-OCR backend: `text`, `ocr`, `table`, `formula`, `chart`, `spotting`, `seal`. `text` is the default and the recommended choice for narrative clinical records. The remaining modes are forwarded verbatim to the vision model as the prompt template selector. DeepSeek-OCR supports `text` mode only.

#### Extraction Workflow

Choose how document processing is orchestrated.

| Mode | Description |
|---|---|
| **Direct pipeline** | Single-pass: OCR → structuring LLM → reasoning LLM. Fastest path. |
| **Granular extraction graph** | 12-node LangGraph: ingest → classify → split → OCR → extract fields → validate schema → normalize codes → retrieve context → merge record → confidence gate → human review → persist. Full observability per node. |
| **Agentic workflow** | First-generation LangGraph with coarser composite nodes. |

#### Model Configuration

Structuring and reasoning models are configured independently. Each accepts a provider (Ollama, OpenAI, Anthropic, Gemini), a model name, and an optional API key. Ollama requires no API key.

### 2. Upload and Analyze

1. Upload a PDF or image file (`.pdf`, `.jpg`, `.png`) via the sidebar uploader.
2. Click **Analyze Document**.
3. The backend processes the file through the selected workflow and returns structured data, clinical analysis, and artifacts.

### 3. Review Extracted Data

The **Extraction & Validation** tab provides a two-column layout:

**Left column — Document Preview** with four sub-tabs:

| Tab | Content |
|---|---|
| Original | Rendered source document (inline PDF viewer or per-page images) |
| Annotated | Bounding box overlays drawn on rendered page images |
| Overlay | Side-by-side original vs. annotated comparison with page selector |
| Bounding Boxes | Tabular view of all detected regions (page, polygon, label, confidence) |

**Right column — Data Editor:**

- Editable JSON table of extracted fields (patient info, vitals, diagnoses, medications, clinical notes).
- Inline correction of OCR errors before persistence.
- **Confirm & Save** writes the record to SQLite and optionally indexes it into the vector store.

If the extraction graph's confidence gate flags the document, a review warning banner is displayed.

### 4. Clinical Insights

The **AI Insights Panel** tab surfaces:

- **Clinical Alerts**: Traffic-light severity indicators for abnormal values or risk factors.
- **Vitals Trends**: Metric cards comparing current values against historical records (when prior encounters exist for the same MRN).
- **AI Summary**: Narrative clinical summary generated by the reasoning model.

### 5. Deep Analysis

The **Deep Analysis** tab provides structured breakdowns of:

- Medications (tabular view)
- Diagnosis list with associated codes

### 6. Insurance Eligibility

The **Insurance Eligibility** tab supports:

1. Upload a policy document (`.txt` or `.pdf`).
2. Click **Check Eligibility**.
3. The system compares extracted diagnoses against policy terms.
4. For PDF policies, OCR is automatically applied before reasoning.
5. When Qdrant is enabled, the policy is chunked, indexed, and queried with exact filters for precise clause retrieval.

The response includes:

- **Eligibility verdict** (Likely Eligible / Risk of Rejection)
- **Reasoning** (natural language explanation of the determination)
- **Missing information** (documents or data points needed for a definitive determination)

## API Reference

All data routes require `X-API-Key: <MEDISCAN_API_KEY>`. Document-upload routes accept `multipart/form-data`; `/confirm` and the review-queue endpoints accept `application/json`. Every response echoes an `X-Request-ID` header (generated when the client does not send one) so log lines and downstream errors can be correlated.

### Route map

| Route | Method | Auth | Content-Type | Description |
|---|---|---|---|---|
| `/analyze` | POST | `X-API-Key` | multipart | Run the selected extraction workflow on an uploaded document |
| `/check_insurance` | POST | `X-API-Key` | multipart | Compare extracted diagnoses against a policy document |
| `/confirm` | POST | `X-API-Key` | JSON | Persist a confirmed `MedicalRecord` to SQLite |
| `/artifacts/{path}` | GET | `X-API-Key` | — | Stream a generated artifact (rendered page, annotated PDF, manifest) |
| `/review/pending` | GET | `X-API-Key` | — | List pending human-review tasks (SQLite-backed) |
| `/review/{task_id}/approve` | POST | `X-API-Key` | JSON (optional) | Approve a queued review task |
| `/review/{task_id}/reject` | POST | `X-API-Key` | JSON (optional) | Reject a queued review task |
| `/health` | GET | — | — | Liveness probe |
| `/ready` | GET | — | — | Readiness probe — checks Ollama reachability, compiled graph warmup, and vector-store config |
| `/metrics` | GET | `X-API-Key` | — | Prometheus scrape endpoint — **only mounted when `MEDISCAN_PROMETHEUS=1`** |

### `POST /analyze`

Process a medical document through the selected extraction workflow.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Medical document (PDF or image) |
| `provider` | string | `"Ollama"` | Fallback provider when `structuring_*` / `reasoning_*` are unset |
| `model` | string | `"glm-4.7-flash"` | Fallback model name |
| `api_key` | string | `null` | Fallback API key. **Accepted only when `MEDISCAN_ALLOW_USER_API_KEYS=1`** — otherwise the server resolves keys from its own `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` env vars |
| `structuring_provider` | string | `null` | Override provider for the structuring LLM |
| `structuring_model` | string | `null` | Override model for the structuring LLM |
| `structuring_api_key` | string | `null` | Same opt-in rule as `api_key` |
| `reasoning_provider` | string | `null` | Override provider for the reasoning LLM |
| `reasoning_model` | string | `null` | Override model for the reasoning LLM |
| `reasoning_api_key` | string | `null` | Same opt-in rule as `api_key` |
| `ocr_backend` | string | `"glm"` | OCR engine: `glm` (default), `ollama`, or `paddle` |
| `ocr_model` | string | `null` | OCR model identifier |
| `ocr_mode` | string | `"text"` | Prompt mode for Ollama-based OCR |
| `use_gpu` | boolean | `true` | Enable GPU acceleration for PaddleOCR-VL |
| `agentic_mode` | boolean | `false` | Use the agentic LangGraph workflow |
| `extraction_graph_mode` | boolean | `false` | Use the granular 12-node extraction graph |

> **Removed:** `paddle_service_url` is no longer a client parameter. The backend reads it exclusively from `PADDLE_SERVICE_URL` to close an SSRF vector.

**Response fields:** `extracted`, `analysis`, `history_available`, `file_path`, `file_url`, `ocr`, `bounding_boxes`, `annotated_pdf_path`, `annotated_pdf_url`, `annotated_image_paths`, `annotated_image_urls`, `page_image_urls`, `requires_human_review`, `vector_index_status`, `retrieval_enabled`, `ocr_supports_bboxes`, `correlation_id`, `lineage` (commit SHA + key dependency versions).

### `POST /check_insurance`

Compare extracted medical data against an insurance policy document.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `policy_file` | file | required | Insurance policy document (TXT or PDF) |
| `medical_json` | string | required | JSON-serialized extracted medical data (validated against `MedicalRecord`) |
| `provider` | string | `"Ollama"` | Fallback provider |
| `model` | string | `"glm-4.7-flash"` | Fallback model |
| `api_key` | string | `null` | Fallback key; gated by `MEDISCAN_ALLOW_USER_API_KEYS` |
| `reasoning_provider` | string | `null` | Override provider for reasoning |
| `reasoning_model` | string | `null` | Override model for reasoning |
| `reasoning_api_key` | string | `null` | Same opt-in rule as `api_key` |
| `policy_ocr` | boolean | `false` | Run OCR on the policy document before reasoning (auto-enabled for non-`.txt` uploads) |
| `ocr_backend` | string | `"glm"` | OCR backend for policy OCR |
| `ocr_model` | string | `null` | OCR model for policy OCR |
| `ocr_mode` | string | `"text"` | OCR mode for policy OCR |
| `use_gpu` | boolean | `true` | GPU flag for policy OCR |

**Response fields:** `eligible`, `reasoning`, `missing_info`.

### `POST /confirm`

Persist a confirmed medical record to SQLite.

**Request body:** JSON object matching the `MedicalRecord` schema (patient info, vitals, clinical data).

**Response:** `{ "status": "saved" }`

### Review-queue routes

- `GET /review/pending?limit=50` returns `{"tasks": [...], "count": n}` from the SQLite review queue.
- `POST /review/{task_id}/approve` and `POST /review/{task_id}/reject` mutate the task. Both accept an optional JSON body `{"reviewer": "...", "notes": "..."}`. A 404 is returned when the task does not exist or is already resolved.

### Artifact downloads

`GET /artifacts/{path}` streams files under the backend upload root. The path is validated against directory-traversal. All requests require the API key.

## Running Tests

```bash
# Full suite with coverage (last known: 241 passed, 8 skipped)
python -m pytest

# Unit tests only
python -m pytest tests/unit/

# Integration tests only (live-stack tests auto-skip when services are unreachable)
python -m pytest tests/integration/

# Quality evaluation harness — OCR CER, field-set F1, MRN exact-match
# Skips when tests/eval/gold/ is empty; add JSON fixtures to enable scoring.
python -m pytest -m eval

# Specific module
python -m pytest tests/unit/test_extraction_graph.py -v
```

## Environment Variables

Grouped by concern. Defaults reflect what the code supplies; `—` means no default.

### Auth, uploads, and hardening

| Variable | Default | Description |
|---|---|---|
| `MEDISCAN_API_KEY` | — | Shared secret required on every data route as `X-API-Key` |
| `MEDISCAN_ALLOW_ANONYMOUS` | `0` | Local-dev opt-in: skips auth **only** when `MEDISCAN_API_KEY` is unset |
| `MEDISCAN_ALLOW_USER_API_KEYS` | `0` | Set to `1` to accept client-supplied provider keys on `/analyze` and `/check_insurance` |
| `MRN_HMAC_PEPPER` | — | Server-held secret for HMAC-SHA256 hashing of MRNs; required for retrieval / audit linkage (never falls back to plain SHA-256) |
| `MEDISCAN_MAX_UPLOAD_BYTES` | `52428800` | Hard cap per upload (50 MiB) |
| `MEDISCAN_MAX_PDF_PAGES` | `200` | Reject PDFs with more pages |
| `MEDISCAN_MAX_PIXELS` | `60000000` | Pillow decompression-bomb guard |
| `MEDISCAN_MAX_RECORD_BYTES` | `262144` | Max serialized record before `/confirm` persistence |
| `MEDISCAN_RATE_LIMIT` | `1` | Set to `0` to disable slowapi |
| `MEDISCAN_DEFAULT_RATE` | `60/minute` | slowapi default bucket per remote address |
| `MEDISCAN_ALLOWED_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Comma-separated CORS allow-list |
| `MEDISCAN_ENABLE_DOCS` | `0` | Set to `1` to expose `/docs`, `/redoc`, `/openapi.json` |
| `MEDISCAN_PII_SCRUB` | `0` | Opt-in PHI scrubber (Presidio-compatible) |

### Storage and retrieval

| Variable | Default | Description |
|---|---|---|
| `MEDISCAN_DB_PATH` | `backend/records.db` | SQLite file location |
| `MEDISCAN_UPLOAD_ROOT` | `backend/uploads` | Artifact / upload directory |
| `MEDISCAN_RENDER_DPI` | `150` | PDF → PNG render DPI |
| `VECTOR_STORE` | *(auto)* | Force `qdrant` or `pgvector`; auto-picks the first configured backend |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_API_KEY` | — | Required whenever `QDRANT_URL` is off-host |
| `QDRANT_ENABLED` | *(auto)* | Force-enable Qdrant (`1` / `true`) |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model for document indexing |

### LLM client

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | — | Server-side provider credentials |
| `MEDISCAN_LLM_TIMEOUT` | `120` | Per-provider request timeout in seconds |
| `MEDISCAN_LLM_RETRIES` | `3` | Tenacity `stop_after_attempt`. Transient-only: 400/401/403/404 are never retried |
| `MEDISCAN_ANTHROPIC_MAX_TOKENS` | `4096` | Anthropic `max_tokens` ceiling |

### Observability

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Structlog / stdlib logging level |
| `MEDISCAN_PROMETHEUS` | `0` | Mount `/metrics` (requires `X-API-Key`) |
| `MEDISCAN_OTEL` | `0` | Enable OpenTelemetry FastAPI auto-instrumentation |
| `MEDISCAN_GIT_SHA` | *(git)* | Overrides the commit SHA reported in the `lineage` response field |

### Service URLs

| Variable | Default | Description |
|---|---|---|
| `PADDLE_SERVICE_URL` | — | PaddleOCR-VL service-mode endpoint. Server-side only; never accepted from clients. |

## Verifying Live Prerequisites

Before relying on a feature, confirm the underlying service is reachable. The table below shows how to validate each optional backend.

### Ollama

```bash
# Verify the server is running and your models are available
ollama list
# Expected: lists pulled models (e.g., glm-ocr, glm-4.7-flash, deepseek-ocr)

# Pull GLM-OCR if not already available (default OCR backend)
ollama pull glm-ocr
```

If `ollama list` fails with a connection error, start the server with `ollama serve`.

### PaddleOCR-VL

```python
# Verify the Python package is importable
python -c "from paddleocr import PaddleOCRVL; print('PaddleOCR-VL OK')"
```

If this fails, install with `pip install "paddleocr[doc-parser]"`. For GPU mode, ensure a compatible PaddlePaddle GPU wheel is installed.

For service mode, verify the endpoint is reachable:

```bash
curl http://127.0.0.1:8118/v1/models
# Expected: 200 OK with available models
```

### Qdrant Retrieval

```bash
# Verify Qdrant is reachable (default port 6333)
curl http://localhost:6333/collections
# Expected: 200 OK with a collections list
```

Then set the environment variables:

```powershell
$env:QDRANT_URL = "http://localhost:6333"
$env:QDRANT_ENABLED = "true"
```

> **Without these variables set and Qdrant healthy, the retrieval path silently no-ops.** The API response includes `retrieval_enabled: false` when retrieval is inactive. The frontend displays a warning banner on the Retrieval Index section.

### Bounding Box Artifacts

The Annotated, Overlay, and Bounding Boxes tabs in the UI only populate when using a PaddleOCR-VL backend (local or service). Ollama-based backends (GLM, DeepSeek) produce text-only OCR output — those tabs will be empty. The API response includes `ocr_supports_bboxes: false` for these backends.

## Known Limitations

- Ollama OCR backends do not emit native bounding boxes; the Annotated, Overlay, and Bounding Boxes tabs are **empty** for GLM and DeepSeek paths. Use PaddleOCR-VL for bbox artifacts.
- Qdrant retrieval is best-effort. If `QDRANT_URL` is unreachable or `MRN_HMAC_PEPPER` is unset, the retrieval path silently no-ops and `retrieval_enabled` is `false` in the response.
- pgvector is scaffolded but not wired for live reads or writes. `PgvectorRetrievalStore.is_configured()` returns `False` by default.
- SQLite history returns the latest prior record per MRN, not a full longitudinal timeline.
- PaddleOCR-VL requires compatible Paddle packages and, for GPU mode, a matching PaddlePaddle GPU wheel.
- The review queue (`/review/pending`, `/review/{id}/approve`, `/review/{id}/reject`) is SQLite-backed. There is no external ticketing integration (Jira / ServiceNow / PagerDuty) — that wiring is left to deployment.
- The quality-evaluation harness (`pytest -m eval`) measures OCR CER, field-set F1, and MRN exact-match only. Retrieval relevance and end-to-end reasoning quality are not yet scored, and `tests/eval/gold/` is empty by design until real fixtures are contributed.
- The server-side log stream is JSON-structured and PHI-redacted via structlog, but log **shipping / retention / rotation** is a deployment concern and is not handled by the application.
