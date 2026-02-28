import backend.extract as extract


def test_process_document_pipeline_image_success(monkeypatch, tmp_path):
    """Targets backend.extract.process_document_pipeline image branch in backend/extract.py."""
    image_file = tmp_path / "doc.jpg"
    image_file.write_bytes(b"dummy")

    monkeypatch.setattr(
        extract,
        "ollama",
        type("DummyOllama", (), {"chat": staticmethod(lambda **kwargs: {"message": {"content": "ocr-text"}})}),
    )
    monkeypatch.setattr(
        extract,
        "get_ai_response",
        lambda provider, model, api_key, system_prompt, user_text: '{"patient": {}, "encounter": {}, "clinical": {"diagnosis_list": [], "medications": [], "vitals": {}}}',
    )

    result = extract.process_document_pipeline(str(image_file), "Ollama", "m", None)
    assert "error" not in result
    assert "clinical" in result


def test_process_document_pipeline_pdf_conversion_failure(monkeypatch, tmp_path):
    """Targets backend.extract.process_document_pipeline PDF conversion error path in backend/extract.py."""
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(b"%PDF")

    def raise_conversion(_):
        raise RuntimeError("poppler missing")

    monkeypatch.setattr(extract, "convert_from_path", raise_conversion)
    result = extract.process_document_pipeline(str(pdf_file))
    assert "error" in result
    assert "PDF Conversion failed" in result["error"]


def test_process_document_pipeline_ocr_failure(monkeypatch, tmp_path):
    """Targets backend.extract.process_document_pipeline OCR error path in backend/extract.py."""
    image_file = tmp_path / "doc.jpg"
    image_file.write_bytes(b"dummy")

    def raise_ocr(**kwargs):
        raise RuntimeError("ocr unavailable")

    monkeypatch.setattr(extract, "ollama", type("DummyOllama", (), {"chat": staticmethod(raise_ocr)}))

    result = extract.process_document_pipeline(str(image_file))
    assert "error" in result
    assert "Ollama OCR failed" in result["error"]


def test_process_document_pipeline_pdf_success(monkeypatch, tmp_path):
    """Targets backend.extract.process_document_pipeline PDF success path in backend/extract.py."""
    from unittest.mock import MagicMock

    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(b"%PDF")

    mock_image = MagicMock()
    monkeypatch.setattr(extract, "convert_from_path", lambda path, **kwargs: [mock_image])
    monkeypatch.setattr(
        extract,
        "ollama",
        type("DummyOllama", (), {"chat": staticmethod(lambda **kwargs: {"message": {"content": "ocr-text"}})}),
    )
    monkeypatch.setattr(
        extract,
        "get_ai_response",
        lambda provider, model, api_key, system_prompt, user_text: '{"patient": {}, "encounter": {}, "clinical": {"diagnosis_list": [], "medications": [], "vitals": {}}}',
    )

    result = extract.process_document_pipeline(str(pdf_file), "Ollama", "m", None)
    assert "error" not in result
    assert "clinical" in result
    mock_image.save.assert_called_once()


def test_process_document_pipeline_structuring_failure_returns_raw_text(monkeypatch, tmp_path):
    """Targets backend.extract.process_document_pipeline structuring error path in backend/extract.py."""
    image_file = tmp_path / "doc.png"
    image_file.write_bytes(b"dummy")

    monkeypatch.setattr(
        extract,
        "ollama",
        type("DummyOllama", (), {"chat": staticmethod(lambda **kwargs: {"message": {"content": "raw-ocr"}})}),
    )

    def raise_structuring(provider, model, api_key, system_prompt, user_text):
        raise RuntimeError("bad model response")

    monkeypatch.setattr(extract, "get_ai_response", raise_structuring)
    result = extract.process_document_pipeline(str(image_file))

    assert "error" in result
    assert "Structuring failed" in result["error"]
    assert "raw_text" not in result
