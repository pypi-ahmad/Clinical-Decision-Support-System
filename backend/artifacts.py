"""Document workspace / annotation helpers.

The module owns a single absolute ``UPLOAD_ROOT`` inside the backend package
so path handling never depends on the current working directory. All public
functions return paths rooted under ``UPLOAD_ROOT``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from pdf2image import convert_from_path
from PIL import Image, ImageDraw


# Cap Pillow's decompression heuristic as a defence-in-depth against crafted
# images with millions of pixels. ``None`` disables Pillow's internal guard.
Image.MAX_IMAGE_PIXELS = int(os.environ.get("MEDISCAN_MAX_PIXELS", 60_000_000))


_DEFAULT_ROOT = Path(__file__).resolve().parent / "uploads"


def _current_upload_root() -> Path:
    """Resolve the active upload root from the environment at call time.

    Reading this lazily lets tests override ``MEDISCAN_UPLOAD_ROOT`` after
    the module has been imported (which is what pytest ``monkeypatch.setenv``
    does).
    """
    return Path(os.environ.get("MEDISCAN_UPLOAD_ROOT", str(_DEFAULT_ROOT)))


class _UploadRootProxy:
    """Path-like proxy that re-resolves on every attribute access."""

    def __fspath__(self) -> str:
        return str(_current_upload_root())

    def __str__(self) -> str:
        return str(_current_upload_root())

    def __truediv__(self, other):
        return _current_upload_root() / other

    def __rtruediv__(self, other):
        return other / _current_upload_root()

    def resolve(self):
        return _current_upload_root().resolve()

    def mkdir(self, *args, **kwargs):
        return _current_upload_root().mkdir(*args, **kwargs)

    def relative_to(self, *args, **kwargs):
        return _current_upload_root().relative_to(*args, **kwargs)


UPLOAD_ROOT = _UploadRootProxy()


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
    root = _current_upload_root()
    root.mkdir(parents=True, exist_ok=True)
    return str(root)


def _assert_within_root(path: Path) -> Path:
    resolved = path.resolve()
    root = _current_upload_root().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Path {resolved} escapes UPLOAD_ROOT={root}"
        ) from exc
    return resolved


def create_document_workspace(file_path: str) -> DocumentWorkspace:
    """Return a fresh workspace rooted under :data:`UPLOAD_ROOT`.

    The callers pass absolute paths from ``main.py`` — we defensively verify
    the source lives under the upload root before creating sibling folders.
    """
    source_path = Path(file_path)
    _assert_within_root(source_path if source_path.is_absolute() else (UPLOAD_ROOT / source_path))

    artifact_root = source_path.with_suffix("")
    artifact_root = artifact_root.parent / f"{artifact_root.name}_artifacts"
    page_images_dir = artifact_root / "pages"
    raw_ocr_dir = artifact_root / "ocr"
    annotations_dir = artifact_root / "annotations"

    for directory in (page_images_dir, raw_ocr_dir, annotations_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return DocumentWorkspace(
        source_file_path=str(source_path),
        artifact_root=str(artifact_root),
        page_images_dir=str(page_images_dir),
        raw_ocr_dir=str(raw_ocr_dir),
        annotations_dir=str(annotations_dir),
    )


def render_document_pages(
    file_path: str,
    workspace: DocumentWorkspace,
    converter=convert_from_path,
    *,
    dpi: int | None = None,
) -> list[str]:
    """Render PDF pages to PNG (or return the single image path untouched).

    DPI defaults to 150 (tunable via ``MEDISCAN_RENDER_DPI``) instead of the
    upstream default of 200 — text-only medical scans read fine at 150 and
    halve OCR latency for long documents.
    """
    source_path = Path(file_path)
    if source_path.suffix.lower() == ".pdf":
        effective_dpi = int(dpi or os.environ.get("MEDISCAN_RENDER_DPI", 150))
        images = converter(file_path, dpi=effective_dpi) if "dpi" in converter.__code__.co_varnames else converter(file_path)
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
    # Fast-path: no boxes → no Pillow work, no annotated PDF. Saves ~hundreds
    # of ms for Ollama/GLM which never emit bboxes.
    if not bounding_boxes:
        return workspace

    boxes_by_page = group_boxes_by_page(bounding_boxes)
    annotated_paths: list[str] = []

    for page_index, image_path in enumerate(page_image_paths, start=1):
        page_boxes = boxes_by_page.get(page_index, [])
        if not page_boxes:
            continue
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
    # Prefer the module-level ``UPLOAD_ROOT`` binding so tests that
    # monkeypatch it directly continue to work; otherwise re-resolve from
    # the environment.
    candidate_roots: list[Path] = []
    module_root = UPLOAD_ROOT
    if isinstance(module_root, (str, os.PathLike)) and not isinstance(module_root, _UploadRootProxy):
        candidate_roots.append(Path(module_root).resolve())
    candidate_roots.append(_current_upload_root().resolve())

    for root in candidate_roots:
        try:
            relative_path = resolved_path.relative_to(root)
        except ValueError:
            continue
        return f"/artifacts/{relative_path.as_posix()}"
    return None


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
