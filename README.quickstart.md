# MediScan OCR — Quickstart

Concise guide for running the current implementation locally.

## What it is

MediScan OCR is a local web app with:
- FastAPI backend (`/analyze`, `/check_insurance`, `/confirm`)
- Streamlit frontend for upload, review, insights, and insurance check
- SQLite persistence for confirmed records and history lookup

## Stack (implemented)

- Backend: FastAPI + Uvicorn
- Frontend: Streamlit
- OCR: Ollama `deepseek-ocr` in extraction pipeline
- LLM adapter: Ollama, OpenAI, Gemini, Anthropic
- Database: SQLite (`backend/medical_records.db`)

## Local setup (Windows)

### 1) Create and activate venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3) Run backend

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

### 4) Run frontend (new terminal)

```bash
streamlit run frontend/app.py
```

## How to use

1. In the sidebar, select provider/model and upload a medical PDF/image.
2. Click **Analyze Document**.
3. Review/edit extracted JSON in **Extraction & Validation**.
4. Click **Confirm & Save to Database** to persist the record.
5. Optionally upload a policy in **Insurance Eligibility** and run eligibility check.

## Test command

```bash
python -m pytest
```

## Current constraints

- PDF extraction processes first page only.
- OCR engine is fixed to Ollama `deepseek-ocr`.
- Insurance binary files use fallback placeholder text (no OCR path in that endpoint).
- Frontend preview expects local filesystem access to backend-provided file path.

## Full technical documentation

See `README.md` for architecture, full flow, module-level breakdowns, and limitations.
