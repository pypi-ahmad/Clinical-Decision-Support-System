# MediScan OCR — Quickstart

Short path for running the current implementation locally.

## What you get

- FastAPI backend with `/analyze`, `/check_insurance`, and `/confirm`
- Streamlit frontend with OCR backend selection, review, bbox overlays, annotated downloads, and insurance checks
- Three extraction workflow modes: Direct pipeline, Granular extraction graph, Agentic workflow
- Separate OCR / Structuring / Reasoning model configuration
- SQLite persistence
- Optional Qdrant semantic retrieval wired into analysis and insurance reasoning
- Annotated PDF/image output with bounding box visualization

## 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. Install base dependencies

For a reproducible, hash-verified install (recommended for CI/prod):

```bash
python -m pip install --upgrade pip
python -m pip install --require-hashes --no-deps -r requirements.lock.txt
```

For day-to-day dev against the ranges in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Regenerate the lockfile when you intentionally change `requirements.txt`:

```bash
uv pip compile --python-version 3.11 --index-strategy unsafe-best-match \
  --generate-hashes --output-file requirements.lock.txt requirements.txt
```

If you want PaddleOCR-VL local Python mode, also install:

```bash
python -m pip install "paddleocr[doc-parser]"
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
python -m uvicorn backend.main:app --reload --port 8000
```

## 5. Run the frontend

```bash
streamlit run frontend/app.py
```

## 6. Use the app

1. Choose an OCR backend (GLM-OCR, DeepSeek-OCR, PaddleOCR-VL local, PaddleOCR-VL service). GLM-OCR is selected by default.
2. Choose structuring and reasoning models separately.
3. Select a workflow mode: **Direct pipeline**, **Granular extraction graph** (12-node LangGraph), or **Agentic workflow**.
4. Upload a medical PDF or image.
5. Review extracted JSON, bounding boxes, annotated output, and overlays across 4 tabs.
   > **Note:** The Annotated, Overlay, and Bounding Boxes tabs only populate when using a PaddleOCR-VL backend. Ollama-based backends (GLM, DeepSeek) produce text-only output — those tabs will be empty.
6. Save the record if it looks correct.
7. Optionally upload a policy and run insurance checking with semantic retrieval.

## Test command

```bash
python -m pytest
```

## Notes

- Qdrant retrieval is **disabled by default**. Set `QDRANT_URL` and `QDRANT_ENABLED=true` for semantic search. Without these, the retrieval path silently no-ops.
- pgvector is scaffolded but not live yet.
- PaddleOCR-VL requires extra local setup (see [USAGE.md](USAGE.md)).

See `README.md` for the full architecture and API details.
