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
try:
    import ollama as _ollama_mod

    _ollama_mod.list()
    _ollama_available = True
except Exception:
    pass

pytestmark = pytest.mark.skipif(not _ollama_available, reason="Ollama server not reachable")


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
    from backend.extract import run_document_ocr

    payload = run_document_ocr(
        str(sample_image),
        ocr_backend="ollama",
        ocr_model="deepseek-ocr",
        ocr_prompt_mode="text",
        use_gpu=True,
    )

    assert "error" not in payload, f"OCR failed: {payload.get('error')}"
    raw = payload.get("raw_text") or payload.get("markdown") or ""
    assert len(raw) > 0, "OCR returned empty text"


def test_live_structuring_returns_valid_json(sample_image: Path):
    """Run real OCR + structuring and verify the result parses as JSON."""
    from backend.extract import process_document_pipeline

    result = process_document_pipeline(
        str(sample_image),
        provider="Ollama",
        model="glm-4.7-flash",
        api_key=None,
        ocr_backend="ollama",
        ocr_model="deepseek-ocr",
        return_details=True,
    )

    assert "error" not in result, f"Pipeline failed: {result.get('error')}"
    structured = result.get("structured_data") or result
    assert isinstance(structured, dict)
    assert "patient" in structured or "clinical" in structured, (
        f"Structuring did not produce expected schema keys: {list(structured.keys())}"
    )


def test_live_reasoning_produces_analysis(sample_image: Path):
    """Run real OCR + structuring + reasoning and verify analysis output."""
    from backend.extract import process_document_pipeline
    from backend.logic import analyze_medical_logic

    result = process_document_pipeline(
        str(sample_image),
        provider="Ollama",
        model="glm-4.7-flash",
        api_key=None,
        ocr_backend="ollama",
        ocr_model="deepseek-ocr",
        return_details=True,
    )

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
    import paddleocr  # noqa: F401

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

    assert "error" not in payload, f"PaddleOCR service failed: {payload.get('error')}"
