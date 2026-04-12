import json

import ollama
from pdf2image import convert_from_path

from backend.ai_wrapper import get_ai_response, clean_json_output
from backend.artifacts import create_document_workspace, render_document_pages
from backend.ocr import OCRBackendError, materialize_annotations, run_ocr

# System prompt defining the target JSON schema for the Structuring phase
SYSTEM_PROMPT = """
You are a medical data entry specialist. Convert the text below into valid JSON matching this schema:
{
  "patient": {"full_name": "string", "dob": "YYYY-MM-DD", "mrn": "string"},
  "encounter": {"date": "YYYY-MM-DD", "provider": "string", "facility": "string"},
  "clinical": {
    "diagnosis_list": ["string"],
    "medications": [{"name": "string", "dosage": "string", "frequency": "string"}],
    "vitals": {"bp": "string", "hr": "string", "temp": "string", "weight": "string"}
  }
}
Return ONLY JSON.
"""

def process_document_pipeline(
    file_path: str,
    provider="Ollama",
    model="glm-4.7-flash",
    api_key=None,
    ocr_backend="ollama",
    ocr_model=None,
    ocr_prompt_mode="text",
    use_gpu=True,
    paddle_service_url=None,
    return_details=False,
    structuring_provider=None,
    structuring_model=None,
    structuring_api_key=None,
):
    """
    Orchestrates the full extraction pipeline:
    1. Pre-processing: Converts PDF to Image if necessary.
    2. OCR (The Eye): Uses the selected OCR backend (Ollama DeepSeek, Ollama GLM, or PaddleOCR-VL) to read the document.
    3. Structuring (The Clerk): Uses the selected AI Provider/Model to format text into JSON.

    Args:
        file_path (str): Absolute path to the uploaded file.
        provider (str): AI provider for the structuring phase.
        model (str): AI model name for the structuring phase.
        api_key (str): Optional API key for cloud providers.
        ocr_backend (str): OCR backend to use: ollama, glm, or paddle.
        ocr_model (str | None): Optional OCR model override.
        ocr_prompt_mode (str): OCR task prompt mode.
        use_gpu (bool): Whether GPU/CUDA should be preferred for OCR backends that support it.
        paddle_service_url (str | None): Optional local PaddleOCR-VL service URL.
        return_details (bool): When true, returns OCR artifacts alongside the structured data.

    Returns:
        dict: The structured JSON data or an error dictionary.
    """
    resolved_structuring_provider = structuring_provider or provider
    resolved_structuring_model = structuring_model or model
    resolved_structuring_api_key = structuring_api_key if structuring_api_key is not None else api_key

    print(f"👀 OCR Scanning: {file_path}")
    ocr_payload = run_document_ocr(
        file_path,
        ocr_backend=ocr_backend,
        ocr_model=ocr_model,
        ocr_prompt_mode=ocr_prompt_mode,
        use_gpu=use_gpu,
        paddle_service_url=paddle_service_url,
    )
    if "error" in ocr_payload:
        return ocr_payload

    raw_text = ocr_payload.get("markdown") or ocr_payload.get("raw_text") or ""
    print(f"✅ OCR Success. Raw Text Length: {len(raw_text)}")

    print(f"📝 Structuring Data with {resolved_structuring_provider} ({resolved_structuring_model})...")
    try:
        user_text = _build_structuring_input(ocr_payload)
        raw_response = get_ai_response(
            resolved_structuring_provider,
            resolved_structuring_model,
            resolved_structuring_api_key,
            SYSTEM_PROMPT,
            user_text,
        )

        json_str = clean_json_output(raw_response)
        structured_data = json.loads(json_str)
        if return_details:
            return {
                "structured_data": structured_data,
                "ocr": ocr_payload,
            }
        return structured_data
    except Exception as e:
        return {"error": f"Structuring failed: {str(e)}"}


def run_document_ocr(
    file_path: str,
    ocr_backend="ollama",
    ocr_model=None,
    ocr_prompt_mode="text",
    use_gpu=True,
    paddle_service_url=None,
):
    try:
        workspace = create_document_workspace(file_path)
        page_image_paths = render_document_pages(file_path, workspace, converter=convert_from_path)
    except Exception as e:
        return {"error": f"PDF Conversion failed. Is Poppler installed? Error: {str(e)}"}

    if not page_image_paths:
        return {"error": "Could not process file format."}

    normalized_ocr_backend = (ocr_backend or "ollama").strip().lower()
    try:
        print(f"👀 OCR Scanning with {ocr_backend} ({ocr_model or 'default'})...")
        ocr_result = run_ocr(
            document_path=file_path,
            page_image_paths=page_image_paths,
            backend=ocr_backend,
            model=ocr_model,
            ocr_mode=ocr_prompt_mode,
            use_gpu=use_gpu,
            paddle_service_url=paddle_service_url,
            artifact_root=workspace.raw_ocr_dir,
            ollama_client=ollama,
        )
        ocr_result = materialize_annotations(file_path, ocr_result, page_image_paths)
        return ocr_result.model_dump()
    except OCRBackendError as e:
        if normalized_ocr_backend in {"ollama", "glm", "glm-ocr", "ollama-glm"}:
            return {"error": f"Ollama OCR failed: {str(e)}"}
        return {"error": f"OCR failed: {str(e)}"}
    except Exception as e:
        if normalized_ocr_backend in {"ollama", "glm", "glm-ocr", "ollama-glm"}:
            return {"error": f"Ollama OCR failed: {str(e)}"}
        return {"error": f"OCR failed: {str(e)}"}


def _build_structuring_input(ocr_payload: dict) -> str:
    sections = []
    raw_text = ocr_payload.get("raw_text") or ""
    markdown = ocr_payload.get("markdown") or ""
    structured_doc = ocr_payload.get("structured_doc")
    if raw_text:
        sections.append(f"OCR_TEXT:\n{raw_text}")
    if markdown and markdown != raw_text:
        sections.append(f"OCR_MARKDOWN:\n{markdown}")
    if structured_doc:
        sections.append(f"OCR_STRUCTURED_DOC:\n{json.dumps(structured_doc, ensure_ascii=False)}")
    return "\n\n".join(section for section in sections if section)
