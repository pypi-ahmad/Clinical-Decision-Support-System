import backend.extract as extract


def test_process_document_pipeline_image_success(monkeypatch, tmp_path):
    image_file = tmp_path / "doc.jpg"
    image_file.write_bytes(b"dummy")
    monkeypatch.setenv("MEDISCAN_UPLOAD_ROOT", str(tmp_path))

    monkeypatch.setattr(
        extract,
        "ollama",
        type("DummyOllama", (), {"chat": staticmethod(lambda **kwargs: {"message": {"content": "ocr-text"}})}),
    )
    monkeypatch.setattr(
        extract,
        "get_ai_response",
        lambda *args, **kwargs: '{"patient": {}, "encounter": {}, "clinical": {"diagnosis_list": [], "medications": [], "vitals": {}}}',
    )

    result = extract.process_document_pipeline(str(image_file), "Ollama", "m", None)
    assert "error" not in result
    assert "clinical" in result


def test_process_document_pipeline_pdf_conversion_failure(monkeypatch, tmp_path):
    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(b"%PDF")
    monkeypatch.setenv("MEDISCAN_UPLOAD_ROOT", str(tmp_path))

    def raise_conversion(*_args, **_kwargs):
        raise RuntimeError("poppler missing")

    monkeypatch.setattr(extract, "convert_from_path", raise_conversion)
    result = extract.process_document_pipeline(str(pdf_file))
    assert "error" in result
    assert "PDF Conversion failed" in result["error"]


def test_process_document_pipeline_ocr_failure(monkeypatch, tmp_path):
    image_file = tmp_path / "doc.jpg"
    image_file.write_bytes(b"dummy")
    monkeypatch.setenv("MEDISCAN_UPLOAD_ROOT", str(tmp_path))

    def raise_ocr(**kwargs):
        raise RuntimeError("ocr unavailable")

    monkeypatch.setattr(extract, "ollama", type("DummyOllama", (), {"chat": staticmethod(raise_ocr)}))

    result = extract.process_document_pipeline(str(image_file))
    assert "error" in result
    assert "Ollama OCR failed" in result["error"]


def test_process_document_pipeline_pdf_success(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    pdf_file = tmp_path / "doc.pdf"
    pdf_file.write_bytes(b"%PDF")
    monkeypatch.setenv("MEDISCAN_UPLOAD_ROOT", str(tmp_path))

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
        lambda *args, **kwargs: '{"patient": {}, "encounter": {}, "clinical": {"diagnosis_list": [], "medications": [], "vitals": {}}}',
    )

    result = extract.process_document_pipeline(str(pdf_file), "Ollama", "m", None)
    assert "error" not in result
    assert "clinical" in result
    mock_image.save.assert_called_once()


def test_process_document_pipeline_structuring_failure_returns_raw_text(monkeypatch, tmp_path):
    image_file = tmp_path / "doc.png"
    image_file.write_bytes(b"dummy")
    monkeypatch.setenv("MEDISCAN_UPLOAD_ROOT", str(tmp_path))

    monkeypatch.setattr(
        extract,
        "ollama",
        type("DummyOllama", (), {"chat": staticmethod(lambda **kwargs: {"message": {"content": "raw-ocr"}})}),
    )

    def raise_structuring(*args, **kwargs):
        raise RuntimeError("bad model response")

    monkeypatch.setattr(extract, "get_ai_response", raise_structuring)
    result = extract.process_document_pipeline(str(image_file))

    assert "error" in result
    assert "Structuring failed" in result["error"]
    assert "raw_text" not in result


def test_run_document_ocr_returns_multi_page_payload(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    pdf_file = tmp_path / "multipage.pdf"
    pdf_file.write_bytes(b"%PDF")
    monkeypatch.setenv("MEDISCAN_UPLOAD_ROOT", str(tmp_path))

    mock_page_one = MagicMock()
    mock_page_two = MagicMock()
    monkeypatch.setattr(extract, "convert_from_path", lambda path, **kwargs: [mock_page_one, mock_page_two])

    responses = iter([
        {"message": {"content": "page one text"}},
        {"message": {"content": "page two text"}},
    ])
    monkeypatch.setattr(
        extract,
        "ollama",
        type("DummyOllama", (), {"chat": staticmethod(lambda **kwargs: next(responses))}),
    )

    result = extract.run_document_ocr(str(pdf_file), ocr_backend="glm", ocr_model="glm-ocr")

    assert "error" not in result
    assert len(result["per_page_results"]) == 2
    assert result["annotations_metadata"]["page_count"] == 2
    assert "PAGE 1" in result["raw_text"]
    assert "PAGE 2" in result["raw_text"]
