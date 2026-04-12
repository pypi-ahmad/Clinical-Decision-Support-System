from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from pdf2image import convert_from_path
from PIL import Image, ImageDraw


UPLOAD_ROOT = Path(__file__).resolve().parent / "uploads"


@dataclass
class DocumentWorkspace:
    source_file_path: str
    artifact_root: str
    page_images_dir: str
    raw_ocr_dir: str
    annotations_dir: str
    page_image_paths: list[str] = field(default_factory=list)
    annotated_image_paths: list[str] = field(default_factory=list)
    annotated_pdf_path: str | None = None


def ensure_upload_root() -> str:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    return str(UPLOAD_ROOT)


def create_document_workspace(file_path: str) -> DocumentWorkspace:
    source_path = Path(file_path)
    artifact_root = source_path.with_suffix("")
    artifact_root = artifact_root.parent / f"{artifact_root.name}_artifacts"
    page_images_dir = artifact_root / "pages"
    raw_ocr_dir = artifact_root / "ocr"
    annotations_dir = artifact_root / "annotations"

    page_images_dir.mkdir(parents=True, exist_ok=True)
    raw_ocr_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    return DocumentWorkspace(
        source_file_path=str(source_path),
        artifact_root=str(artifact_root),
        page_images_dir=str(page_images_dir),
        raw_ocr_dir=str(raw_ocr_dir),
        annotations_dir=str(annotations_dir),
    )


def render_document_pages(file_path: str, workspace: DocumentWorkspace, converter=convert_from_path) -> list[str]:
    source_path = Path(file_path)
    if source_path.suffix.lower() == ".pdf":
        images = converter(file_path)
        page_paths: list[str] = []
        for page_index, image in enumerate(images, start=1):
            page_path = Path(workspace.page_images_dir) / f"page_{page_index:04d}.png"
            image.save(page_path, "PNG")
            page_paths.append(str(page_path))
        workspace.page_image_paths = page_paths
        return page_paths

    workspace.page_image_paths = [file_path]
    return workspace.page_image_paths


def group_boxes_by_page(bounding_boxes: Iterable[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for box in bounding_boxes:
        page_number = int(box.get("page_number", 1))
        grouped.setdefault(page_number, []).append(box)
    return grouped


def annotate_document(
    page_image_paths: list[str],
    bounding_boxes: list[dict],
    workspace: DocumentWorkspace,
) -> DocumentWorkspace:
    boxes_by_page = group_boxes_by_page(bounding_boxes)
    annotated_paths: list[str] = []

    for page_index, image_path in enumerate(page_image_paths, start=1):
        page_boxes = boxes_by_page.get(page_index, [])
        annotated_path = Path(workspace.annotations_dir) / f"annotated_page_{page_index:04d}.png"
        if _annotate_image(image_path, page_boxes, annotated_path):
            annotated_paths.append(str(annotated_path))

    workspace.annotated_image_paths = annotated_paths
    if annotated_paths:
        annotated_pdf_path = Path(workspace.annotations_dir) / "annotated_document.pdf"
        _write_pdf_from_images(annotated_paths, annotated_pdf_path)
        workspace.annotated_pdf_path = str(annotated_pdf_path)

    return workspace


def build_artifact_url(path: str | None) -> str | None:
    if not path:
        return None

    resolved_path = Path(path).resolve()
    try:
        relative_path = resolved_path.relative_to(UPLOAD_ROOT.resolve())
    except ValueError:
        return None
    return f"/artifacts/{relative_path.as_posix()}"


def build_artifact_manifest(source_file_path: str, workspace: DocumentWorkspace) -> dict[str, object]:
    return {
        "original_file_path": source_file_path,
        "original_file_url": build_artifact_url(source_file_path),
        "page_image_paths": list(workspace.page_image_paths),
        "page_image_urls": [build_artifact_url(path) for path in workspace.page_image_paths],
        "annotated_image_paths": list(workspace.annotated_image_paths),
        "annotated_image_urls": [build_artifact_url(path) for path in workspace.annotated_image_paths],
        "annotated_pdf_path": workspace.annotated_pdf_path,
        "annotated_pdf_url": build_artifact_url(workspace.annotated_pdf_path),
    }


def _annotate_image(image_path: str, page_boxes: list[dict], destination: Path) -> bool:
    try:
        image = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, OSError):
        return False

    draw = ImageDraw.Draw(image)
    for box in page_boxes:
        polygon = box.get("polygon") or []
        flattened = [tuple(point) for point in polygon if isinstance(point, list) and len(point) >= 2]
        if len(flattened) >= 2:
            draw.line(flattened + [flattened[0]], fill=(255, 0, 0), width=3)
            label = box.get("text") or box.get("label")
            if label:
                draw.text(flattened[0], str(label)[:80], fill=(255, 0, 0))

    image.save(destination, format="PNG")
    return True


def _write_pdf_from_images(image_paths: list[str], destination: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        return
    first_image, *rest = images
    first_image.save(destination, save_all=True, append_images=rest, format="PDF")