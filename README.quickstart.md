# MediScan OCR &mdash; Quickstart

Short path for running the current implementation locally.

> **Looking for the full reference?** Read [docs/HANDBOOK.md](docs/HANDBOOK.md)
> for a zero-to-hero walkthrough, or the focused docs:
> [ARCHITECTURE](docs/ARCHITECTURE.md) &middot;
> [SECURITY](docs/SECURITY.md) &middot;
> [DEVELOPMENT](docs/DEVELOPMENT.md).

## What you get

- FastAPI backend with `/analyze`, `/check_insurance`, and `/confirm`
- Streamlit frontend with OCR backend selection, review, bbox overlays, annotated downloads, and insurance checks
- Three extraction workflow modes: Direct pipeline, Granular extraction graph, Agentic workflow
- Separate OCR / Structuring / Reasoning model configuration
- SQLite persistence with audit log and review queue
- Optional Qdrant semantic retrieval wired into analysis and insurance reasoning
- Annotated PDF/image output with bounding box visualization

## 1. Create and activate a virtual environment (uv)

```bash
uv python install 3.12.10
uv venv --python 3.12.10
source .venv/bin/activate
```

Windows PowerShell:

```powershell
uv python install 3.12.10
uv venv --python 3.12.10
.\.venv\Scripts\Activate.ps1
```

## 2. Install dependencies

For day-to-day development and tests:

```bash
uv sync --frozen
```

For runtime-only dependencies:

```bash
uv sync --frozen --no-dev
```

## 3. Ensure local prerequisites

- Install Poppler so `pdf2image` can render PDFs.
- Start Ollama if you want Ollama OCR or Ollama structuring/reasoning.
- Start Qdrant if you want semantic retrieval.

Example retrieval env:

```powershell
$env:QDRANT_URL = "http://localhost:6333"
$env:QDRANT_ENABLED = "true"
```

## 4. Run the backend

```bash
uv run --frozen uvicorn backend.main:app --reload --port 8000
```

## 5. Run the frontend

```bash
uv run --frozen streamlit run frontend/app.py
```

## 6. Use the app

1. Choose an OCR backend (GLM-OCR via Ollama, PaddleOCR-VL local, PaddleOCR-VL service). GLM-OCR is selected by default.
2. Choose structuring and reasoning models separately.
3. Select a workflow mode: **Direct pipeline**, **Granular extraction graph** (12-node LangGraph), or **Agentic workflow**.
4. Upload a medical PDF or image.
5. Review extracted JSON, bounding boxes, annotated output, and overlays across 4 tabs.
   > **Note:** The Annotated, Overlay, and Bounding Boxes tabs only populate when using a PaddleOCR-VL backend. The Ollama-based GLM-OCR backend produces text-only output &mdash; those tabs will be empty.
6. Save the record if it looks correct.
7. Optionally upload a policy and run insurance checking with semantic retrieval.

## Test command

```bash
uv run --frozen pytest
```

## Notes

- Qdrant retrieval is **disabled by default**. Set `QDRANT_URL` and `QDRANT_ENABLED=true` for semantic search. Without these, the retrieval path silently no-ops.
- pgvector is scaffolded but not live yet.
- PaddleOCR-VL requires extra local setup (see [USAGE.md](USAGE.md)).

See `README.md` for the full architecture and API details, and
[docs/HANDBOOK.md](docs/HANDBOOK.md) for a complete walkthrough.
