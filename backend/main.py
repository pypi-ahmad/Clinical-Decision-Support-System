"""
Main API Module (The Interface)
-------------------------------
This is the entry point for the FastAPI backend. It exposes endpoints for:
1. Uploading and analyzing medical documents (/analyze).
2. Checking insurance eligibility (/check_insurance).
3. Saving confirmed records to the database (/confirm).

It serves as the bridge between the Frontend (Streamlit) and the Backend logic modules.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import json
from backend.database import init_db, save_record, get_patient_history
from backend.extract import process_document_pipeline
from backend.logic import analyze_medical_logic, check_insurance_coverage

app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing) to allow the Streamlit frontend
# (running on a different port) to communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the upload directory exists
os.makedirs("backend/uploads", exist_ok=True)
# Initialize the database on startup
init_db()

@app.post("/analyze")
async def analyze_medical_doc(
    file: UploadFile = File(...),
    provider: str = Form("Ollama"),
    model: str = Form("glm-4.7-flash"),
    api_key: str = Form(None)
):
    """
    Endpoint to process a medical document.
    
    Pipeline:
    1. Saves the uploaded file.
    2. Runs Extraction (OCR + Structuring).
    3. Fetches patient history from DB (if MRN exists).
    4. Runs Logic Analysis (Trends + Alerts).
    
    Args:
        file (UploadFile): The medical document (PDF/Image).
        provider (str): User-selected AI provider (e.g., "Ollama", "OpenAI").
        model (str): Specific model to use.
        api_key (str): API key for the selected provider.
        
    Returns:
        JSON with extracted data, analysis results, history status, and file path.
    """
    # 1. Save File Locally
    file_path = f"backend/uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Extract Data (OCR + Structuring)
    # The Structuring phase will use the selected provider/model
    current_data = process_document_pipeline(file_path, provider, model, api_key)
    if "error" in current_data:
        raise HTTPException(status_code=500, detail=current_data)

    # 3. Fetch History (The Memory)
    mrn = current_data.get("patient", {}).get("mrn")
    past_data = get_patient_history(mrn) if mrn else None
    
    # 4. Reason (The Brain)
    # Analysis will also use the selected provider/model
    analysis = analyze_medical_logic(current_data, past_data, provider, model, api_key)
    
    # 5. Return Results
    return {
        "extracted": current_data,
        "analysis": analysis,
        "history_available": bool(past_data),
        "file_path": file_path # Return path so frontend can display PDF
    }

@app.post("/check_insurance")
async def check_insurance(
    policy_file: UploadFile = File(...), 
    medical_json: str = Form(...)
):
    """
    Endpoint to check insurance eligibility.
    
    Args:
        policy_file (UploadFile): The insurance policy document.
        medical_json (str): The structured medical data (as a JSON string).
        
    Returns:
        JSON with eligibility status and reasoning.
    """
    # 1. Read Policy Text
    # For a real app, you'd run OCR on this too. Here we assume text-readable content.
    content = await policy_file.read()
    try:
        policy_text = content.decode('utf-8')
    except:
        policy_text = "Binary PDF content - (Simulated OCR would go here)"
    
    # 2. Run Analysis
    medical_data = json.loads(medical_json)
    result = check_insurance_coverage(medical_data, policy_text)
    
    return result

@app.post("/confirm")
def confirm_record(data: dict):
    """
    Endpoint to finalize and save a record to the database.
    Called after the user validates the data in the frontend.
    """
    save_record(data)
    return {"status": "saved"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
