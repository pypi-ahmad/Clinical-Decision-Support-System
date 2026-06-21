"""
Live integration tests — require a running Ollama server.

These tests exercise real OCR + structuring + reasoning against Ollama.
They are skipped automatically when Ollama is unreachable.

Run explicitly::

    python -m pytest tests/integration/test_live_pipeline.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Skip the entire module when Ollama is not reachable
# ---------------------------------------------------------------------------

_ollama_available = False
_ollama_models: set[str] = set()
try:
    import ollama as _ollama_mod

    _model_list = _ollama_mod.list()
    _ollama_available = True
    # Collect available model names (strip :tag suffixes for flexible matching)
    for m in getattr(_model_list, "models", []):
        name = getattr(m, "model", "") or ""
        _ollama_models.add(name)
        _ollama_models.add(name.split(":")[0])
except Exception:
    pass

pytestmark = pytest.mark.skipif(not _ollama_available, reason="Ollama server not reachable")


def _require_ollama_model(model_name: str):
    """Skip the test if the requested Ollama model is not pulled."""
    base = model_name.split(":")[0]
    if base not in _ollama_models and model_name not in _ollama_models:
        pytest.skip(f"Ollama model '{model_name}' not pulled")


def _skip_on_model_error(payload: dict, context: str = ""):
    """Skip the test if the payload contains a model-not-found or similar error."""
    err = payload.get("error", "")
    if err and ("not found" in err or "404" in err):
        pytest.skip(f"Model unavailable{': ' + context if context else ''}: {err}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_TEXT = (
    "Patient: Jane Smith  DOB: 1985-03-15  MRN: MRN-LIVE-1\n"
    "Date: 2024-08-01  Provider: Dr. Adams  Facility: Metro Hospital\n"
    "Diagnosis: Hypertension, Type 2 Diabetes\n"
    "Medications: Lisinopril 10 mg once daily, Metformin 500 mg twice daily\n"
    "Vitals: BP 138/88  HR 72  Temp 98.6  Weight 82 kg\n"
)


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    """Create a tiny PNG with the sample medical text burned in.

    We write a real (but minimal) PNG so that Ollama's vision model can
    attempt OCR.  The text itself is embedded as metadata/alt text; the
    image content is a 1×1 white pixel which the model will read as blank
    but still exercise the full pipeline path.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (800, 400), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 20), _SAMPLE_TEXT, fill="black", font=font)
    path = tmp_path / "live_test_record.png"
    img.save(str(path))
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_live_ocr_produces_raw_text(sample_image: Path):
    """Run real Ollama OCR on an image and verify we get text back."""
    _require_ollama_model("glm-ocr")
    from backend.extract import run_document_ocr

    payload = run_document_ocr(
        str(sample_image),
        ocr_backend="ollama",
        ocr_model="glm-ocr",
        ocr_prompt_mode="text",
        use_gpu=True,
    )

    _skip_on_model_error(payload, "glm-ocr")
    assert "error" not in payload, f"OCR failed: {payload.get('error')}"
    raw = payload.get("raw_text") or payload.get("markdown") or ""
    assert len(raw) > 0, "OCR returned empty text"


def test_live_structuring_returns_valid_json(sample_image: Path):
    """Run real OCR + structuring and verify the result parses as JSON."""
    _require_ollama_model("glm-ocr")
    _require_ollama_model("glm-4.7-flash")
    from backend.extract import process_document_pipeline

    result = process_document_pipeline(
        str(sample_image),
        provider="Ollama",
        model="glm-4.7-flash",
        api_key=None,
        ocr_backend="ollama",
        ocr_model="glm-ocr",
        return_details=True,
    )

    _skip_on_model_error(result, "glm-ocr / glm-4.7-flash")
    assert "error" not in result, f"Pipeline failed: {result.get('error')}"
    structured = result.get("structured_data") or result
    assert isinstance(structured, dict)
    assert "patient" in structured or "clinical" in structured, (
        f"Structuring did not produce expected schema keys: {list(structured.keys())}"
    )


def test_live_reasoning_produces_analysis(sample_image: Path):
    """Run real OCR + structuring + reasoning and verify analysis output."""
    _require_ollama_model("glm-ocr")
    _require_ollama_model("glm-4.7-flash")
    from backend.extract import process_document_pipeline
    from backend.logic import analyze_medical_logic

    result = process_document_pipeline(
        str(sample_image),
        provider="Ollama",
        model="glm-4.7-flash",
        api_key=None,
        ocr_backend="ollama",
        ocr_model="glm-ocr",
        return_details=True,
    )

    _skip_on_model_error(result, "glm-ocr / glm-4.7-flash")
    assert "error" not in result, f"Pipeline failed: {result.get('error')}"
    structured = result.get("structured_data") or result

    analysis = analyze_medical_logic(
        structured,
        None,
        provider="Ollama",
        model="glm-4.7-flash",
        api_key=None,
    )

    assert isinstance(analysis, dict)
    assert "summary" in analysis


# ---------------------------------------------------------------------------
# PaddleOCR-VL stubs — skip when not installed
# ---------------------------------------------------------------------------

_paddle_available = False
try:
    from paddleocr import PaddleOCRVL  # noqa: F401

    _paddle_available = True
except Exception:
    pass


@pytest.mark.skipif(not _paddle_available, reason="PaddleOCR not installed")
def test_live_paddle_local_ocr(sample_image: Path):
    """Smoke-test PaddleOCR-VL local Python mode (requires paddleocr package)."""
    from backend.extract import run_document_ocr

    payload = run_document_ocr(
        str(sample_image),
        ocr_backend="paddle",
        ocr_model="PaddlePaddle/PaddleOCR-VL-1.5",
        ocr_prompt_mode="text",
        use_gpu=True,
    )

    err = payload.get("error", "")
    if err and ("dependency" in err.lower() or "pipeline creation" in err.lower()):
        pytest.skip(f"PaddleOCR runtime dependencies not satisfied: {err}")
    assert "error" not in payload, f"PaddleOCR failed: {payload.get('error')}"
    raw = payload.get("raw_text") or payload.get("markdown") or ""
    assert len(raw) > 0, "PaddleOCR returned empty text"
    # PaddleOCR should produce bounding boxes
    bboxes = payload.get("bounding_boxes", [])
    assert len(bboxes) > 0, "PaddleOCR should produce bounding boxes but returned none"


@pytest.mark.skipif(not _paddle_available, reason="PaddleOCR not installed")
def test_live_paddle_service_ocr(sample_image: Path):
    """Smoke-test PaddleOCR-VL service mode (requires running service)."""
    service_url = os.getenv("PADDLE_SERVICE_URL", "http://127.0.0.1:8118/v1")
    from backend.extract import run_document_ocr

    try:
        payload = run_document_ocr(
            str(sample_image),
            ocr_backend="paddle",
            ocr_model="PaddlePaddle/PaddleOCR-VL-1.5",
            ocr_prompt_mode="text",
            use_gpu=True,
            paddle_service_url=service_url,
        )
    except Exception as exc:
        pytest.skip(f"PaddleOCR service not reachable at {service_url}: {exc}")

    err = payload.get("error", "")
    if err and ("healthcheck" in err.lower() or "connection" in err.lower()):
        pytest.skip(f"PaddleOCR service not reachable: {err}")
    assert "error" not in payload, f"PaddleOCR service failed: {payload.get('error')}"


# ---------------------------------------------------------------------------
# Qdrant retrieval live test — requires QDRANT_URL set and healthy
# ---------------------------------------------------------------------------

_qdrant_available = False
_qdrant_url = os.getenv("QDRANT_URL", "")
if _qdrant_url:
    try:
        import requests as _req

        _resp = _req.get(f"{_qdrant_url.rstrip('/')}/collections", timeout=3)
        _qdrant_available = _resp.status_code < 500
    except Exception:
        pass


@pytest.mark.skipif(not _qdrant_available, reason="Qdrant not reachable at QDRANT_URL")
@pytest.mark.skipif(not _ollama_available, reason="Ollama server not reachable")
def test_live_qdrant_index_and_retrieve():
    """Index a synthetic document into Qdrant and retrieve it by patient hash."""
    from backend.retrieval import hash_identifier
    from backend.retrieval.chunking import build_chunks_from_text
    from backend.retrieval.qdrant_store import QdrantRetrievalStore
    from backend.retrieval.vector_store import RetrievalChunk

    collection = "test_live_mediscan"
    store = QdrantRetrievalStore(collection_name=collection)

    text = "Patient MRN-LIVE-QDRANT diagnosed with pneumonia. Prescribed amoxicillin 500 mg."

    # Use the production HMAC path so the filter matches what the rest of
    # the system writes; this test is gated on MRN_HMAC_PEPPER being set
    # (see tests/conftest.py), but guard explicitly in case it was unset.
    patient_hash = hash_identifier("MRN-LIVE-QDRANT")
    if not patient_hash:
        pytest.skip("MRN_HMAC_PEPPER must be set for the Qdrant live test")

    metadata = {
        "patient_id_hash": patient_hash,
        "source_type": "medical_record",
    }
    chunks = build_chunks_from_text("live_test_doc", text, metadata, section_type="ocr_document")
    status = store.upsert_chunks(chunks)
    assert status.get("indexed") is True, f"upsert failed: {status}"

    results = store.search(
        "pneumonia treatment",
        limit=3,
        filters={"patient_id_hash": patient_hash, "source_type": "medical_record"},
    )

    assert len(results) > 0, "Qdrant returned no results for indexed document"
    assert any("pneumonia" in r.get("text", "") for r in results)

    # Cleanup
    try:
        from qdrant_client import QdrantClient

        QdrantClient(url=_qdrant_url).delete_collection(collection)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Multi-page PDF live test — requires Ollama + Poppler
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ollama_available, reason="Ollama server not reachable")
def test_live_multipage_pdf(tmp_path: Path):
    """Process a synthetically constructed 2-page PDF through the full pipeline."""
    _require_ollama_model("glm-ocr")
    _require_ollama_model("glm-4.7-flash")
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        pytest.skip("reportlab not installed — needed for multi-page PDF generation")

    pdf_path = tmp_path / "multipage.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(72, 700, "Patient: Bob Test  MRN: MRN-MP-1  DOB: 1990-01-01")
    c.drawString(72, 680, "Diagnosis: Asthma  Medications: Albuterol inhaler")
    c.showPage()
    c.drawString(72, 700, "Vitals: BP 120/80  HR 68  Temp 98.2")
    c.drawString(72, 680, "Provider: Dr. Multi  Facility: Test Hospital")
    c.showPage()
    c.save()

    from backend.extract import process_document_pipeline

    result = process_document_pipeline(
        str(pdf_path),
        provider="Ollama",
        model="glm-4.7-flash",
        api_key=None,
        ocr_backend="ollama",
        ocr_model="glm-ocr",
        return_details=True,
    )

    assert "error" not in result, f"Multi-page pipeline failed: {result.get('error')}"
    structured = result.get("structured_data") or result
    assert isinstance(structured, dict)
