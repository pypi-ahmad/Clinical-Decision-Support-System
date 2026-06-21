"""
Tests for OCR backend implementations.

Covers:
- OllamaOCRBackend: GLM prompt modes, multi-page aggregation
- OCRBackendConfig normalization
- collect_bounding_boxes polygon parsing
- annotate_document bounding box rendering
- build_artifact_manifest URL mapping
- PaddleOCRVLServiceClient error/success handling
- PaddleOCR-VL serialization helpers
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.ocr_backends.base import (
    OCRBackendConfig,
    OCRBoundingBox,
    OCRPageResult,
    _normalize_polygon,
    aggregate_page_results,
    collect_bounding_boxes,
)
from backend.ocr_backends.ollama_ocr import GLM_PROMPTS, OllamaOCRBackend, _build_prompt
from backend.ocr_backends.service_client import PaddleOCRVLServiceClient, PaddleOCRVLServiceError, PaddleOCRVLServiceSettings


# ---------------------------------------------------------------------------
# OCRBackendConfig normalization
# ---------------------------------------------------------------------------


def test_backend_config_normalizes_glm_variant():
    config = OCRBackendConfig(backend="glm-ocr", model=None, ocr_mode="text")
    assert config.normalized_backend == "glm"


def test_backend_config_normalizes_paddle_variants():
    for variant in ("paddle", "paddleocr", "paddleocr-vl", "paddleocr-vl-1.5"):
        config = OCRBackendConfig(backend=variant)
        assert config.normalized_backend == "paddle", f"Expected 'paddle' for {variant!r}"


def test_backend_config_normalizes_ollama_to_glm():
    """The historical ``ollama`` value is preserved as an alias for ``glm``."""
    config = OCRBackendConfig(backend="ollama")
    assert config.normalized_backend == "glm"


def test_backend_config_resolved_model_uses_default_for_glm():
    config = OCRBackendConfig(backend="glm", model=None)
    assert config.resolved_model == "glm-ocr"


def test_backend_config_resolved_model_uses_default_for_ollama_alias():
    config = OCRBackendConfig(backend="ollama", model=None)
    assert config.resolved_model == "glm-ocr"


def test_backend_config_resolved_model_respects_override():
    config = OCRBackendConfig(backend="ollama", model="custom-ocr")
    assert config.resolved_model == "custom-ocr"


# ---------------------------------------------------------------------------
# OllamaOCRBackend – GLM prompts
# ---------------------------------------------------------------------------


def test_glm_prompt_text_mode():
    config = OCRBackendConfig(backend="glm", ocr_mode="text")
    assert _build_prompt(config) == GLM_PROMPTS["text"]


def test_glm_prompt_table_mode():
    config = OCRBackendConfig(backend="glm", ocr_mode="table")
    assert _build_prompt(config) == GLM_PROMPTS["table"]


def test_glm_prompt_figure_mode():
    config = OCRBackendConfig(backend="glm", ocr_mode="figure")
    assert _build_prompt(config) == GLM_PROMPTS["figure"]


def test_glm_prompt_formula_mode():
    config = OCRBackendConfig(backend="glm", ocr_mode="formula")
    assert _build_prompt(config) == GLM_PROMPTS["formula"]


def test_glm_prompt_unknown_mode_falls_back_to_text():
    config = OCRBackendConfig(backend="glm", ocr_mode="unknown_xyz")
    assert _build_prompt(config) == GLM_PROMPTS["text"]


def test_glm_prompt_is_used_for_ollama_alias():
    """Even with the historical ``ollama`` alias the GLM prompt set is used."""
    config = OCRBackendConfig(backend="ollama", ocr_mode="text")
    assert _build_prompt(config) == GLM_PROMPTS["text"]


# ---------------------------------------------------------------------------
# OllamaOCRBackend – multi-page
# ---------------------------------------------------------------------------


def test_ollama_ocr_backend_runs_multi_page(tmp_path):
    """OllamaOCRBackend should call ollama.chat once per page image."""
    page_image_paths = [
        str(tmp_path / "page_0001.png"),
        str(tmp_path / "page_0002.png"),
    ]
    for path in page_image_paths:
        Path(path).write_bytes(b"fake_png")

    call_counter = {"n": 0}
    responses = ["page one content", "page two content"]

    def fake_chat(**kwargs):
        idx = call_counter["n"]
        call_counter["n"] += 1
        return {"message": {"content": responses[idx]}}

    fake_ollama = type("FakeOllama", (), {"chat": staticmethod(fake_chat)})()
    backend = OllamaOCRBackend(ollama_client=fake_ollama)
    config = OCRBackendConfig(backend="glm", model="glm-ocr", ocr_mode="text")

    result = backend.run("doc.pdf", page_image_paths, config)

    assert call_counter["n"] == 2
    assert len(result.per_page_results) == 2
    assert result.per_page_results[0].raw_text == "page one content"
    assert result.per_page_results[1].raw_text == "page two content"
    assert "PAGE 1" in result.raw_text
    assert "PAGE 2" in result.raw_text


def test_ollama_ocr_backend_glm_backend_label(tmp_path):
    """When backend=glm the result backend label should be 'glm'."""
    page_path = tmp_path / "page.png"
    page_path.write_bytes(b"x")

    fake_ollama = type("FakeOllama", (), {"chat": staticmethod(lambda **kw: {"message": {"content": "text"}})})()
    backend = OllamaOCRBackend(ollama_client=fake_ollama)
    config = OCRBackendConfig(backend="glm", model="glm-ocr", ocr_mode="text")

    result = backend.run("doc.pdf", [str(page_path)], config)

    assert result.backend == "glm"
    assert result.model == "glm-ocr"


# ---------------------------------------------------------------------------
# Bounding box normalization helpers
# ---------------------------------------------------------------------------


def test_normalize_polygon_xyxy_to_quad():
    polygon = _normalize_polygon([10.0, 20.0, 50.0, 60.0])
    assert polygon == [[10.0, 20.0], [50.0, 20.0], [50.0, 60.0], [10.0, 60.0]]


def test_normalize_polygon_list_of_points():
    polygon = _normalize_polygon([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert polygon == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]


def test_normalize_polygon_dict_points():
    polygon = _normalize_polygon([{"x": 5, "y": 5}, {"x": 15, "y": 5}, {"x": 15, "y": 15}, {"x": 5, "y": 15}])
    assert len(polygon) == 4
    assert polygon[0] == [5.0, 5.0]


def test_normalize_polygon_returns_none_for_empty():
    assert _normalize_polygon([]) is None
    assert _normalize_polygon(None) is None


def test_collect_bounding_boxes_from_paddle_like_payload():
    payload = {
        "blocks": [
            {
                "type": "text",
                "bbox": [100.0, 200.0, 300.0, 250.0],
                "text": "Patient Name",
                "score": 0.97,
                "page_number": 1,
            },
            {
                "type": "table",
                "bbox": [50.0, 300.0, 400.0, 500.0],
                "markdown": "| col1 | col2 |",
                "score": 0.82,
                "page_number": 1,
            },
        ]
    }
    boxes = collect_bounding_boxes(payload, page_number=1)
    assert len(boxes) == 2
    assert boxes[0].text == "Patient Name"
    assert boxes[0].confidence == pytest.approx(0.97)
    assert len(boxes[0].polygon) == 4


def test_collect_bounding_boxes_deduplicates():
    entry = {
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "text": "dup",
        "page_number": 1,
    }
    payload = [entry, entry]
    boxes = collect_bounding_boxes(payload, page_number=1)
    assert len(boxes) == 1


def test_collect_bounding_boxes_no_payload():
    boxes = collect_bounding_boxes({}, page_number=1)
    assert boxes == []


# ---------------------------------------------------------------------------
# aggregate_page_results
# ---------------------------------------------------------------------------


def test_aggregate_page_results_combines_text():
    pages = [
        OCRPageResult(page_number=1, raw_text="first page", markdown="first page"),
        OCRPageResult(page_number=2, raw_text="second page", markdown="second page"),
    ]
    result = aggregate_page_results("glm", "glm-ocr", "text", pages)
    assert "PAGE 1" in result.raw_text
    assert "PAGE 2" in result.raw_text
    assert result.backend == "glm"
    assert result.model == "glm-ocr"
    assert len(result.per_page_results) == 2


def test_aggregate_page_results_averages_confidence():
    pages = [
        OCRPageResult(page_number=1, raw_text="a", confidence=0.8),
        OCRPageResult(page_number=2, raw_text="b", confidence=0.6),
    ]
    result = aggregate_page_results("glm", "glm-ocr", "text", pages)
    assert result.confidence == pytest.approx(0.7)


def test_aggregate_page_results_no_confidence_if_none():
    pages = [
        OCRPageResult(page_number=1, raw_text="a"),
    ]
    result = aggregate_page_results("glm", "glm-ocr", "text", pages)
    assert result.confidence is None


def test_aggregate_page_results_collects_bounding_boxes():
    box1 = OCRBoundingBox(page_number=1, polygon=[[0, 0], [10, 0], [10, 10], [0, 10]])
    box2 = OCRBoundingBox(page_number=2, polygon=[[0, 0], [20, 0], [20, 20], [0, 20]])
    pages = [
        OCRPageResult(page_number=1, raw_text="a", bounding_boxes=[box1]),
        OCRPageResult(page_number=2, raw_text="b", bounding_boxes=[box2]),
    ]
    result = aggregate_page_results("glm", "glm-ocr", "text", pages)
    assert len(result.bounding_boxes) == 2


# ---------------------------------------------------------------------------
# Annotated document generation
# ---------------------------------------------------------------------------


def test_annotate_document_creates_annotated_images(tmp_path):
    from PIL import Image
    from backend.artifacts import annotate_document, create_document_workspace, DocumentWorkspace

    image_path = tmp_path / "page_0001.png"
    img = Image.new("RGB", (200, 300), color=(255, 255, 255))
    img.save(image_path, "PNG")

    workspace = DocumentWorkspace(
        source_file_path=str(image_path),
        artifact_root=str(tmp_path / "artifacts"),
        page_images_dir=str(tmp_path / "pages"),
        raw_ocr_dir=str(tmp_path / "ocr"),
        annotations_dir=str(tmp_path / "annotations"),
        page_image_paths=[str(image_path)],
    )
    (tmp_path / "annotations").mkdir()

    bounding_boxes = [
        {
            "page_number": 1,
            "polygon": [[10, 10], [100, 10], [100, 50], [10, 50]],
            "text": "Patient Name",
            "label": None,
        }
    ]

    result = annotate_document([str(image_path)], bounding_boxes, workspace)

    assert len(result.annotated_image_paths) == 1
    assert Path(result.annotated_image_paths[0]).exists()
    assert result.annotated_pdf_path is not None
    assert Path(result.annotated_pdf_path).exists()


def test_annotate_document_no_boxes_skips_image_work(tmp_path):
    """When no bounding boxes are present ``annotate_document`` is a no-op.

    This is a deliberate optimisation: Ollama/GLM never produce
    bboxes, so re-opening each page through Pillow would be pure overhead.
    """
    from PIL import Image
    from backend.artifacts import annotate_document, DocumentWorkspace

    image_path = tmp_path / "page_0001.png"
    img = Image.new("RGB", (100, 100), color=(200, 200, 200))
    img.save(image_path, "PNG")

    workspace = DocumentWorkspace(
        source_file_path=str(image_path),
        artifact_root=str(tmp_path),
        page_images_dir=str(tmp_path),
        raw_ocr_dir=str(tmp_path),
        annotations_dir=str(tmp_path / "ann"),
        page_image_paths=[str(image_path)],
    )
    (tmp_path / "ann").mkdir()

    result = annotate_document([str(image_path)], [], workspace)
    assert result.annotated_image_paths == []
    assert result.annotated_pdf_path is None


def test_annotate_document_missing_image_skipped(tmp_path):
    from backend.artifacts import annotate_document, DocumentWorkspace

    workspace = DocumentWorkspace(
        source_file_path=str(tmp_path / "missing.png"),
        artifact_root=str(tmp_path),
        page_images_dir=str(tmp_path),
        raw_ocr_dir=str(tmp_path),
        annotations_dir=str(tmp_path / "ann2"),
        page_image_paths=[],
    )
    (tmp_path / "ann2").mkdir()

    result = annotate_document([str(tmp_path / "does_not_exist.png")], [], workspace)
    assert result.annotated_image_paths == []
    assert result.annotated_pdf_path is None


# ---------------------------------------------------------------------------
# build_artifact_manifest URL mapping
# ---------------------------------------------------------------------------


def test_build_artifact_manifest_maps_paths_to_urls(tmp_path, monkeypatch):
    from backend.artifacts import UPLOAD_ROOT, build_artifact_manifest, DocumentWorkspace

    fake_upload_root = tmp_path / "uploads"
    fake_upload_root.mkdir()
    monkeypatch.setattr("backend.artifacts.UPLOAD_ROOT", fake_upload_root)

    page_image = fake_upload_root / "doc_artifacts" / "pages" / "page_0001.png"
    page_image.parent.mkdir(parents=True)
    page_image.write_bytes(b"img")

    ann_image = fake_upload_root / "doc_artifacts" / "annotations" / "annotated_page_0001.png"
    ann_image.parent.mkdir(parents=True)
    ann_image.write_bytes(b"ann")

    workspace = DocumentWorkspace(
        source_file_path=str(fake_upload_root / "doc.pdf"),
        artifact_root=str(fake_upload_root / "doc_artifacts"),
        page_images_dir=str(fake_upload_root / "doc_artifacts" / "pages"),
        raw_ocr_dir=str(fake_upload_root / "doc_artifacts" / "ocr"),
        annotations_dir=str(fake_upload_root / "doc_artifacts" / "annotations"),
        page_image_paths=[str(page_image)],
        annotated_image_paths=[str(ann_image)],
    )

    manifest = build_artifact_manifest(str(fake_upload_root / "doc.pdf"), workspace)

    assert manifest["page_image_urls"][0].startswith("/artifacts/")
    assert manifest["annotated_image_urls"][0].startswith("/artifacts/")


# ---------------------------------------------------------------------------
# PaddleOCRVLServiceClient error handling stubs
# ---------------------------------------------------------------------------


def test_service_client_healthcheck_raises_on_all_failures():
    settings = PaddleOCRVLServiceSettings(
        model_name="PaddlePaddle/PaddleOCR-VL-1.5",
        service_url="http://127.0.0.1:19999",
        healthcheck_timeout_seconds=1,
        request_timeout_seconds=5,
    )
    client = PaddleOCRVLServiceClient(settings)
    with pytest.raises(PaddleOCRVLServiceError, match="healthcheck"):
        client.healthcheck()


def test_service_client_build_pipeline_raises_when_paddle_not_installed():
    settings = PaddleOCRVLServiceSettings(
        model_name="PaddlePaddle/PaddleOCR-VL-1.5",
        service_url="http://127.0.0.1:19999",
    )
    import backend.ocr_backends.service_client as sc_module

    original = sc_module.PaddleOCRVL
    sc_module.PaddleOCRVL = None
    try:
        client = PaddleOCRVLServiceClient(settings)
        with pytest.raises(PaddleOCRVLServiceError, match="not installed"):
            client.build_pipeline()
    finally:
        sc_module.PaddleOCRVL = original


# ---------------------------------------------------------------------------
# PaddleOCRVLServiceClient success-path tests
# ---------------------------------------------------------------------------


def test_service_client_healthcheck_success():
    """Healthcheck succeeds when the first candidate URL returns 200."""
    settings = PaddleOCRVLServiceSettings(
        model_name="PaddlePaddle/PaddleOCR-VL-1.5",
        service_url="http://127.0.0.1:8118/v1",
        healthcheck_timeout_seconds=2,
    )
    client = PaddleOCRVLServiceClient(settings)

    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("backend.ocr_backends.service_client.requests.get", return_value=mock_response) as mock_get:
        result = client.healthcheck()

    assert result["healthy"] is True
    assert result["status_code"] == 200
    mock_get.assert_called_once()


def test_service_client_predict_success():
    """predict() delegates to build_pipeline → pipeline.predict and returns (output, pipeline)."""
    settings = PaddleOCRVLServiceSettings(
        model_name="PaddlePaddle/PaddleOCR-VL-1.5",
        service_url="http://127.0.0.1:8118/v1",
        request_timeout_seconds=30,
    )
    client = PaddleOCRVLServiceClient(settings)

    fake_page = MagicMock()
    fake_pipeline = MagicMock()
    fake_pipeline.predict.return_value = iter([fake_page])

    with patch.object(client, "build_pipeline", return_value=fake_pipeline):
        output, pipeline = client.predict("/tmp/doc.pdf")

    assert output == [fake_page]
    assert pipeline is fake_pipeline
    fake_pipeline.predict.assert_called_once_with("/tmp/doc.pdf")


# ---------------------------------------------------------------------------
# PaddleOCR-VL serialization helpers
# ---------------------------------------------------------------------------


def test_serialize_page_results_with_json_and_markdown(tmp_path):
    """_serialize_page_results produces page results from objects that emit JSON + markdown."""
    from backend.ocr_backends.paddleocr_vl import _serialize_page_results

    page_data = {"text": "hello", "regions": [{"polygon": [[0, 0], [100, 0], [100, 50], [0, 50]], "label": "text"}]}

    def fake_save_json(save_path):
        p = Path(save_path) / "result.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(page_data), encoding="utf-8")

    def fake_save_md(save_path):
        p = Path(save_path) / "result.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# Result\nhello", encoding="utf-8")

    page_obj = MagicMock()
    page_obj.save_to_json = fake_save_json
    page_obj.save_to_markdown = fake_save_md

    artifact_root = str(tmp_path / "artifacts")
    page_results, payloads, markdowns = _serialize_page_results(
        [page_obj],
        [str(tmp_path / "page1.png")],
        artifact_root,
    )

    assert len(page_results) == 1
    assert page_results[0].page_number == 1
    assert payloads[0] == page_data
    assert "hello" in markdowns[0]


def test_serialize_page_results_empty_object(tmp_path):
    """_serialize_page_results handles objects that have no save methods gracefully."""
    from backend.ocr_backends.paddleocr_vl import _serialize_page_results

    page_obj = MagicMock(spec=[])  # no save_to_json / save_to_markdown

    page_results, payloads, markdowns = _serialize_page_results(
        [page_obj],
        [],
        str(tmp_path / "artifacts"),
    )

    assert len(page_results) == 1
    assert payloads[0] == {}
    assert markdowns == []


def test_save_result_payloads_reads_written_files(tmp_path):
    """_save_result_payloads calls save_to_json/save_to_markdown and reads back the files."""
    from backend.ocr_backends.paddleocr_vl import _save_result_payloads

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    payload_data = {"key": "value"}

    def fake_save_json(save_path):
        p = Path(save_path) / "out.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload_data), encoding="utf-8")

    def fake_save_md(save_path):
        p = Path(save_path) / "out.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("markdown content", encoding="utf-8")

    obj = MagicMock()
    obj.save_to_json = fake_save_json
    obj.save_to_markdown = fake_save_md

    json_payload, md_text = _save_result_payloads(obj, output_dir)

    assert json_payload == payload_data
    assert md_text == "markdown content"


def test_serialize_merged_results_with_restructure(tmp_path):
    """_serialize_merged_results calls pipeline.restructure_pages and serializes output."""
    from backend.ocr_backends.paddleocr_vl import _serialize_merged_results

    merged_data = {"merged": True}

    def fake_save_json(save_path):
        p = Path(save_path) / "merged.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(merged_data), encoding="utf-8")

    def fake_save_md(save_path):
        p = Path(save_path) / "merged.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("merged markdown", encoding="utf-8")

    merged_obj = MagicMock()
    merged_obj.save_to_json = fake_save_json
    merged_obj.save_to_markdown = fake_save_md

    pipeline = MagicMock()
    pipeline.restructure_pages.return_value = [merged_obj]

    page_objects = [MagicMock(), MagicMock()]  # >1 page to trigger merge

    payloads, markdowns = _serialize_merged_results(pipeline, page_objects, str(tmp_path / "art"))

    assert payloads == [merged_data]
    assert markdowns == ["merged markdown"]
    pipeline.restructure_pages.assert_called_once()


def test_serialize_merged_results_skips_single_page():
    """_serialize_merged_results returns empty lists when there is only one page."""
    from backend.ocr_backends.paddleocr_vl import _serialize_merged_results

    pipeline = MagicMock()
    pipeline.restructure_pages = MagicMock()

    payloads, markdowns = _serialize_merged_results(pipeline, [MagicMock()], None)

    assert payloads == []
    assert markdowns == []
    pipeline.restructure_pages.assert_not_called()


def test_build_local_pipeline_calls_paddleocrvl():
    """_build_local_pipeline constructs a PaddleOCRVL with expected kwargs."""
    from backend.ocr_backends.paddleocr_vl import _build_local_pipeline

    config = OCRBackendConfig(backend="paddle", model="test", use_gpu=False)
    fake_cls = MagicMock()

    with patch("backend.ocr_backends.paddleocr_vl.PaddleOCRVL", fake_cls):
        result = _build_local_pipeline(config)

    fake_cls.assert_called_once_with(
        device="cpu",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_layout_detection=True,
    )
    assert result is fake_cls.return_value
