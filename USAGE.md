# Usage Guide

## About

MediScan OCR is a clinical decision-support system that automates medical document intake, structured data extraction, clinical reasoning, and insurance eligibility verification. It combines pluggable OCR backends, large language model orchestration, and semantic retrieval to transform unstructured medical records into actionable, machine-readable clinical data.

### Core Capabilities

| Capability | Description |
|---|---|
| **Document Ingestion** | Multi-page PDF and image upload with automatic page rendering and artifact management. |
| **Pluggable OCR** | Interchangeable OCR backends (GLM-4V, DeepSeek, PaddleOCR-VL) with configurable prompt modes. |
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
│  │  Ollama (GLM / DeepSeek) │  │  Qdrant (active)          │        │
│  │  PaddleOCR-VL            │  │  pgvector (scaffold)      │        │
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

| Option | Engine | Bounding Boxes | Notes |
|---|---|---|---|
| DeepSeek-OCR (Ollama) | Ollama + DeepSeek prompts | No | Text-focused, fast |
| GLM-OCR (Ollama) | Ollama + GLM-4V prompts | No | Supports `text`, `table`, `figure`, `formula` modes |
| PaddleOCR-VL (Local Python) | Local PaddleOCR-VL runtime | Yes | Requires `paddleocr[doc-parser]` install |
| PaddleOCR-VL (Local Service) | HTTP service client | Yes | Requires a running Paddle service endpoint |

#### OCR Mode

Applies to GLM-OCR. Controls the prompt template sent to the vision model.

| Mode | Use Case |
|---|---|
| `text` | General document text extraction |
| `table` | Tabular data extraction |
| `formula` | Mathematical or lab formula recognition |
| `chart` | Chart and figure interpretation |

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

All endpoints accept `multipart/form-data`.

### `POST /analyze`

Process a medical document through the selected extraction workflow.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Medical document (PDF or image) |
| `provider` | string | `"Ollama"` | AI provider for default/fallback model selection |
| `model` | string | `"glm-4.7-flash"` | Default model name |
| `api_key` | string | `null` | API key (not required for Ollama) |
| `structuring_provider` | string | `null` | Override provider for the structuring LLM |
| `structuring_model` | string | `null` | Override model for the structuring LLM |
| `structuring_api_key` | string | `null` | API key for the structuring provider |
| `reasoning_provider` | string | `null` | Override provider for the reasoning LLM |
| `reasoning_model` | string | `null` | Override model for the reasoning LLM |
| `reasoning_api_key` | string | `null` | API key for the reasoning provider |
| `ocr_backend` | string | `"ollama"` | OCR engine: `ollama`, `glm`, or `paddle` |
| `ocr_model` | string | `null` | OCR model identifier |
| `ocr_mode` | string | `"text"` | Prompt mode for GLM OCR |
| `use_gpu` | boolean | `true` | Enable GPU acceleration for PaddleOCR-VL |
| `paddle_service_url` | string | `null` | Service URL for PaddleOCR-VL service mode |
| `agentic_mode` | boolean | `false` | Use the agentic LangGraph workflow |
| `extraction_graph_mode` | boolean | `false` | Use the granular 12-node extraction graph |

**Response fields:** `extracted`, `analysis`, `history_available`, `file_path`, `file_url`, `ocr`, `bounding_boxes`, `annotated_pdf_path`, `annotated_pdf_url`, `annotated_image_paths`, `annotated_image_urls`, `page_image_urls`, `requires_human_review`, `vector_index_status`.

### `POST /check_insurance`

Compare extracted medical data against an insurance policy document.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `policy_file` | file | required | Insurance policy document (TXT or PDF) |
| `medical_json` | string | required | JSON-serialized extracted medical data |
| `provider` | string | `"Ollama"` | AI provider |
| `model` | string | `"llama3"` | Model name |
| `api_key` | string | `null` | API key |
| `reasoning_provider` | string | `null` | Override provider for reasoning |
| `reasoning_model` | string | `null` | Override model for reasoning |
| `reasoning_api_key` | string | `null` | API key for reasoning provider |
| `policy_ocr` | boolean | `false` | Run OCR on the policy document before reasoning |
| `ocr_backend` | string | `"ollama"` | OCR backend for policy OCR |
| `ocr_model` | string | `null` | OCR model for policy OCR |
| `ocr_mode` | string | `"text"` | OCR mode for policy OCR |
| `use_gpu` | boolean | `true` | GPU flag for policy OCR |
| `paddle_service_url` | string | `null` | Service URL for Paddle policy OCR |

**Response fields:** `eligible`, `reasoning`, `missing_info`.

### `POST /confirm`

Persist a confirmed medical record to SQLite.

**Request body:** JSON object matching the `MedicalRecord` schema (patient info, vitals, clinical data).

**Response:** `{ "status": "saved" }`

## Running Tests

```bash
# Full suite with coverage
python -m pytest

# Unit tests only
python -m pytest tests/unit/

# Integration tests only
python -m pytest tests/integration/

# Specific module
python -m pytest tests/unit/test_extraction_graph.py -v
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | — | Qdrant server URL (e.g., `http://localhost:6333`) |
| `QDRANT_ENABLED` | `false` | Enable Qdrant retrieval |
| `QDRANT_API_KEY` | — | Qdrant authentication key |
| `VECTOR_STORE` | `qdrant` | Active vector store backend |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model for vector indexing |

## Known Limitations

- Ollama OCR backends do not emit native bounding boxes; overlay tabs are empty for GLM and DeepSeek paths.
- SQLite history returns the latest prior record per MRN, not a full longitudinal timeline.
- PaddleOCR-VL requires compatible Paddle packages and, for GPU mode, a matching PaddlePaddle GPU wheel.
- The API has no authentication or authorization; CORS is fully open. Not suitable for production deployment without additional hardening.
- The `human_review` node in the extraction graph is a passthrough in the automated path; production deployments should wire it to an external task/ticket system.
- pgvector is scaffolded but not wired for live reads or writes.
