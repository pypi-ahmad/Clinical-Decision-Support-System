"""
Extraction Module (The Eye & The Clerk)
---------------------------------------
This module handles the first two phases of the pipeline:
1. "The Eye": Optical Character Recognition (OCR) using DeepSeek-OCR to extract raw text from images or PDFs.
2. "The Clerk": Structuring the raw OCR text into a clean, standardized JSON format using an LLM.

It utilizes the `pdf2image` library for PDF conversion and calls the `ai_wrapper` for the structuring phase.
"""

import ollama
import json
import os
from pdf2image import convert_from_path
from backend.ai_wrapper import get_ai_response, clean_json_output

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

def process_document_pipeline(file_path: str, provider="Ollama", model="glm-4.7-flash", api_key=None):
    """
    Orchestrates the full extraction pipeline:
    1. Pre-processing: Converts PDF to Image if necessary.
    2. OCR (The Eye): Uses DeepSeek-OCR (Local) to read text from the image.
    3. Structuring (The Clerk): Uses the selected AI Provider/Model to format text into JSON.

    Args:
        file_path (str): Absolute path to the uploaded file.
        provider (str): AI provider for the structuring phase.
        model (str): AI model name for the structuring phase.
        api_key (str): Optional API key for cloud providers.

    Returns:
        dict: The structured JSON data or an error dictionary.
    """
    print(f"👀 OCR Scanning: {file_path}")
    
    image_data = None
    temp_img_path = None
    
    # --- PHASE 1: PRE-PROCESSING (HANDLE PDF vs IMAGE) ---
    if file_path.lower().endswith('.pdf'):
        try:
            # Convert first page of PDF to image using pdf2image (requires Poppler)
            images = convert_from_path(file_path, first_page=1, last_page=1)
            if images:
                # Save temp image to send to OCR
                temp_img_path = file_path + "_temp.jpg"
                images[0].save(temp_img_path, 'JPEG')
                image_data = temp_img_path
        except Exception as e:
            return {"error": f"PDF Conversion failed. Is Poppler installed? Error: {str(e)}"}
    else:
        # It's already an image (jpg/png)
        image_data = file_path

    if not image_data:
        return {"error": "Could not process file format."}

    # --- PHASE 2: OCR (THE EYE) ---
    # We use Ollama's DeepSeek-OCR locally because it is specialized and free for vision tasks.
    try:
        try:
            print(f"👀 OCR Scanning with DeepSeek-OCR...")
            # Note: Ollama python client handles file paths automatically if they point to valid images
            ocr_response = ollama.chat(
                model='deepseek-ocr',
                messages=[{
                    'role': 'user', 
                    'content': 'Transcribe this medical document text exactly.', 
                    'images': [image_data]
                }]
            )
            raw_text = ocr_response['message']['content']
            print(f"✅ OCR Success. Raw Text Length: {len(raw_text)}")
            
        except Exception as e:
            return {"error": f"Ollama OCR failed: {str(e)}"}
        
        # --- PHASE 3: STRUCTURING (THE CLERK) ---
        # We use the 'ai_wrapper' to allow the user to choose their preferred model (Clerk).
        print(f"📝 Structuring Data with {provider} ({model})...")
        try:
            raw_response = get_ai_response(provider, model, api_key, SYSTEM_PROMPT, f"OCR TEXT:\n{raw_text}")
            
            # Clean and Parse JSON
            json_str = clean_json_output(raw_response)
            return json.loads(json_str)
        except Exception as e:
            return {"error": f"Structuring failed: {str(e)}"}
    finally:
        if temp_img_path and os.path.exists(temp_img_path):
            os.remove(temp_img_path)
