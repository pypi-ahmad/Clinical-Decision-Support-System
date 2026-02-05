"""
Logic Module (The Brain & The Adjuster)
---------------------------------------
This module handles the advanced reasoning tasks:
1. "The Brain": Clinical Decision Support (CDS) logic that compares current patient data against 
   historical records to identify trends (e.g., BP rising) and inconsistencies.
2. "The Adjuster": Insurance eligibility checks that compare the diagnosed conditions against 
   a provided policy document.
"""

import json
from backend.ai_wrapper import get_ai_response, clean_json_output

# --- Existing Reasoning Prompt (Clinical) ---
# Instructs the AI to perform Trend Analysis and Consistency Checks.
REASONING_PROMPT = """
You are a Clinical Decision Support System. Compare the Current Visit vs Past History.
Task 1: TRENDS. Compare Vitals (BP, Weight, HR). State if they are Increasing, Decreasing, or Stable.
Task 2: CONSISTENCY. Check if prescribed medications match the diagnoses.

Output JSON:
{
  "alerts": ["High Priority Alert", "Medium Priority Warning"],
  "trends": [{"metric": "BP", "status": "Worsening", "details": "120/80 -> 140/90"}],
  "summary": "Brief clinical summary of changes."
}
"""

# --- Insurance Prompt (Administrative) ---
# Instructs the AI to act as a Claims Adjuster.
INSURANCE_PROMPT = """
You are an Insurance Claims Adjuster. 
Review the MEDICAL_DATA (Diagnosis & Treatments) and the INSURANCE_POLICY summary.

Determine if the patient's condition is likely covered.
1. Match Diagnosis against Policy Inclusions/Exclusions.
2. Check for waiting periods or pre-existing condition clauses.

Output JSON:
{
  "eligible": true/false,
  "confidence": "High/Medium/Low",
  "reasoning": "Explanation of why it is covered or rejected.",
  "missing_info": ["List of documents or details needed to confirm"]
}
"""

def analyze_medical_logic(current_data, past_data, provider="Ollama", model="lfm2.5-thinking", api_key=None):
    """
    Performs Clinical Decision Support analysis.

    Args:
        current_data (dict): The data extracted from the current document.
        past_data (dict): The data from the patient's previous visit (if any).
        provider (str): AI Provider to use for reasoning.
        model (str): AI Model name.
        api_key (str): API Key for the provider.

    Returns:
        dict: A JSON object containing 'alerts', 'trends', and a 'summary'.
    """
    # Prepare the context for the AI
    if not past_data:
        context = f"CURRENT_DATA: {json.dumps(current_data)}\nPAST_DATA: None (New Patient)"
    else:
        context = f"CURRENT_DATA: {json.dumps(current_data)}\nPAST_DATA: {json.dumps(past_data)}"

    print(f"🧠 Analyzing Logic with {provider}...")
    
    try:
        # Call the AI Wrapper
        response = get_ai_response(provider, model, api_key, REASONING_PROMPT, context)
        # Clean and return JSON
        return json.loads(clean_json_output(response))
    except:
        return {"summary": "Analysis failed", "alerts": []}

def check_insurance_coverage(medical_data, policy_text, provider="Ollama", model="glm-4.7-flash", api_key=None):
    """
    Checks if the patient's medical data matches the insurance policy criteria.

    Args:
        medical_data (dict): The extracted clinical data (diagnosis/treatments).
        policy_text (str): The raw text content of the insurance policy.
        provider (str): AI Provider to use.
        model (str): AI Model name.
        api_key (str): API Key.

    Returns:
        dict: JSON object with 'eligible' (bool), 'reasoning', and 'missing_info'.
    """
    print(f"🛡️ Checking Insurance with {provider}...")
    
    # Construct the prompt with data and policy text (truncated to avoid token limits if necessary)
    prompt = f"""
    MEDICAL_DATA: {json.dumps(medical_data)}
    INSURANCE_POLICY_TEXT: {policy_text[:4000]} (Truncated for context window)
    
    {INSURANCE_PROMPT}
    """
    
    try:
        response = get_ai_response(provider, model, api_key, INSURANCE_PROMPT, prompt)
        return json.loads(clean_json_output(response))
    except Exception as e:
        return {"eligible": False, "reasoning": f"Error: {str(e)}", "missing_info": []}

