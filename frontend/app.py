"""
Frontend Application (Streamlit)
--------------------------------
This is the user interface for MediScan AI. It renders a web application allowing users to:
1. Upload medical documents.
2. Configure AI settings (Provider, Model, Keys).
3. View extracted data side-by-side with the original document.
4. See AI insights, alerts, and trends.
5. Check insurance eligibility.

It communicates with the FastAPI backend via HTTP requests.
"""

import streamlit as st
import requests
import json
import pandas as pd
from streamlit_pdf_viewer import pdf_viewer

API_URL = "http://localhost:8000"

st.set_page_config(page_title="MediScan AI", layout="wide", page_icon="🏥")

# --- Session State Management ---
# Initialize session variables to persist data across re-runs
if 'extracted_data' not in st.session_state:
    st.session_state['extracted_data'] = None
if 'analysis' not in st.session_state:
    st.session_state['analysis'] = None
if 'pdf_path' not in st.session_state:
    st.session_state['pdf_path'] = None

# --- Header ---
st.title("🏥 MediScan: AI Medical Record Digitizer")

# --- Sidebar (Configuration & Upload) ---
with st.sidebar:
    st.header("⚙️ AI Configuration")
    
    # 1. Select Provider (Universal Adapter)
    provider = st.selectbox("Select Model Provider", ["Ollama", "OpenAI", "Anthropic", "Gemini"])
    
    # 2. Dynamic Model Options based on Provider
    model_options = []
    api_key = None
    
    if provider == "Ollama":
        model_options = ["glm-4.7-flash", "lfm2.5-thinking", "llama3"]
    elif provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password")
        model_options = ["gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"]
    elif provider == "Anthropic":
        api_key = st.text_input("Anthropic API Key", type="password")
        model_options = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"]
    elif provider == "Gemini":
        api_key = st.text_input("Gemini API Key", type="password")
        model_options = ["gemini-1.5-pro", "gemini-1.5-flash"]
        
    selected_model = st.selectbox("Select Model", model_options)

    st.header("Upload Medical Record")
    uploaded_file = st.file_uploader("Upload PDF/Image", type=["pdf", "jpg", "png"])
    
    if uploaded_file and st.button("🚀 Analyze Document"):
        with st.spinner(f"Running Analysis with {provider} ({selected_model})..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {
                "provider": provider, 
                "model": selected_model, 
                "api_key": api_key if api_key else ""
            }
            try:
                # Call Backend API
                response = requests.post(f"{API_URL}/analyze", files=files, data=data, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    # Update Session State
                    st.session_state['extracted_data'] = data['extracted']
                    st.session_state['analysis'] = data['analysis']
                    st.session_state['pdf_path'] = data.get('file_path') 
                    st.success("Analysis Complete!")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# --- Main Interface Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Extraction & Validation", 
    "🚨 AI Insights Panel", 
    "📊 Deep Analysis", 
    "🛡️ Insurance Eligibility"
])

# === TAB 1: EXTRACTION & VALIDATION ===
with tab1:
    if st.session_state['extracted_data']:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📄 Original Document")
            # Display PDF/Image Preview
            if st.session_state['pdf_path']:
                if st.session_state['pdf_path'].endswith('.pdf'):
                    # Read the file from backend path (assuming local dev environment for simplicity)
                    try:
                        with open(st.session_state['pdf_path'], "rb") as f:
                            pdf_bytes = f.read()
                        pdf_viewer(input=pdf_bytes, width=700, height=800)
                    except Exception:
                        st.warning("Could not load PDF preview.")
                else:
                    st.image(st.session_state['pdf_path'])
        
        with col2:
            st.subheader("✏️ Data Editor (Fix OCR Errors)")
            st.info("Hovering over fields simulates bounding box focus (Prototype)")
            
            # Interactive JSON Editor allows users to correct AI mistakes
            edited_data = st.data_editor(
                st.session_state['extracted_data'], 
                height=800, 
                use_container_width=True
            )
            
            if st.button("💾 Confirm & Save to Database"):
                try:
                    save_response = requests.post(f"{API_URL}/confirm", json=edited_data, timeout=60)
                    if save_response.status_code == 200:
                        st.toast("Record Saved successfully!", icon="✅")
                    else:
                        st.error(f"Save failed: {save_response.text}")
                except Exception as e:
                    st.error(f"Save Error: {e}")

    else:
        st.info("Please upload a document in the sidebar to begin.")

# === TAB 2: AI INSIGHTS PANEL ===
with tab2:
    if st.session_state['analysis']:
        analysis = st.session_state['analysis']
        
        # 1. Alerts Section (Traffic Light System)
        st.subheader("⚠️ Clinical Alerts")
        if analysis.get('alerts'):
            for alert in analysis['alerts']:
                if "High" in alert or "Critical" in alert:
                    st.error(f"🔴 {alert}")
                else:
                    st.warning(f"🟡 {alert}")
        else:
            st.success("✅ No critical alerts detected.")

        st.divider()

        # 2. Vitals Trends (Comparison with Past History)
        st.subheader("📈 Vitals Trends")
        trends = analysis.get('trends', [])
        if trends:
            cols = st.columns(len(trends))
            for idx, trend in enumerate(trends):
                with cols[idx]:
                    st.metric(
                        label=trend['metric'], 
                        value=trend['status'], 
                        delta=trend.get('details', '')
                    )
        else:
            st.info("No historical data available for trends.")

        st.divider()
        st.subheader("📋 AI Summary")
        st.write(analysis.get('summary', 'No summary generated.'))
        
    else:
        st.write("No analysis data yet.")

# === TAB 3: DEEP ANALYSIS ===
with tab3:
    if st.session_state['extracted_data']:
        data = st.session_state['extracted_data']
        st.header("🔬 Detailed Breakdown")
        clinical_data = data.get('clinical', {})
        medications = clinical_data.get('medications', [])
        diagnosis_list = clinical_data.get('diagnosis_list', [])
        
        # Clinical Data Table
        st.subheader("Medications")
        if medications:
            st.table(pd.DataFrame(medications))
        else:
            st.write("No medications found.")
            
        st.subheader("Diagnosis")
        for diag in diagnosis_list:
            st.markdown(f"- **{diag}**")
            
    else:
        st.write("Waiting for data...")

# === TAB 4: INSURANCE CHECK ===
with tab4:
    st.header("🛡️ Insurance Coverage Check")
    st.write("Upload an insurance policy to check if the extracted diagnosis is covered.")
    
    policy_file = st.file_uploader("Upload Policy Document (TXT/PDF)", key="policy")
    
    if policy_file and st.session_state['extracted_data']:
        if st.button("Check Eligibility"):
            with st.spinner("Comparing Policy vs Diagnosis..."):
                files = {"policy_file": policy_file}
                # Send the extracted medical data as a JSON string field
                payload = {"medical_json": json.dumps(st.session_state['extracted_data'])}
                
                try:
                    res = requests.post(f"{API_URL}/check_insurance", files=files, data=payload, timeout=60)
                    if res.status_code == 200:
                        result = res.json()
                        
                        if result.get('eligible'):
                            st.success("✅ Likely Eligible")
                        else:
                            st.error("❌ Risk of Rejection")
                            
                        st.subheader("Reasoning")
                        st.write(result.get('reasoning', 'No reasoning provided.'))
                        
                        if result.get('missing_info'):
                            st.warning("⚠️ Missing Documents:")
                            for item in result.get('missing_info', []):
                                st.write(f"- {item}")
                    else:
                        st.error("Check failed.")
                except Exception as e:
                    st.error(f"Error: {e}")
    elif not st.session_state['extracted_data']:
        st.warning("Please analyze a medical report first (Tab 1).")
