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
PROVIDER_MODELS = {
    "Ollama": ["glm-4.7-flash", "lfm2.5-thinking", "llama3"],
    "OpenAI": ["gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"],
    "Anthropic": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"],
    "Gemini": ["gemini-1.5-pro", "gemini-1.5-flash"],
}

st.set_page_config(page_title="MediScan AI", layout="wide", page_icon="🏥")

# --- Session State Management ---
# Initialize session variables to persist data across re-runs
if 'extracted_data' not in st.session_state:
    st.session_state['extracted_data'] = None
if 'analysis' not in st.session_state:
    st.session_state['analysis'] = None
if 'pdf_path' not in st.session_state:
    st.session_state['pdf_path'] = None
if 'file_url' not in st.session_state:
    st.session_state['file_url'] = None
if 'ocr_artifacts' not in st.session_state:
    st.session_state['ocr_artifacts'] = None
if 'annotated_pdf_path' not in st.session_state:
    st.session_state['annotated_pdf_path'] = None
if 'annotated_pdf_url' not in st.session_state:
    st.session_state['annotated_pdf_url'] = None
if 'annotated_image_paths' not in st.session_state:
    st.session_state['annotated_image_paths'] = []
if 'annotated_image_urls' not in st.session_state:
    st.session_state['annotated_image_urls'] = []
if 'page_image_urls' not in st.session_state:
    st.session_state['page_image_urls'] = []
if 'bounding_boxes' not in st.session_state:
    st.session_state['bounding_boxes'] = []
if 'requires_human_review' not in st.session_state:
    st.session_state['requires_human_review'] = False
if 'vector_index_status' not in st.session_state:
    st.session_state['vector_index_status'] = None
if 'retrieval_enabled' not in st.session_state:
    st.session_state['retrieval_enabled'] = False
if 'ocr_supports_bboxes' not in st.session_state:
    st.session_state['ocr_supports_bboxes'] = False


def render_model_config(title: str, key_prefix: str, default_provider: str = "Ollama"):
    st.subheader(title)
    provider = st.selectbox(
        f"{title} Provider",
        list(PROVIDER_MODELS.keys()),
        index=list(PROVIDER_MODELS.keys()).index(default_provider),
        key=f"{key_prefix}_provider",
    )
    model_options = PROVIDER_MODELS[provider]
    api_key = None
    if provider != "Ollama":
        api_key = st.text_input(f"{title} API Key", type="password", key=f"{key_prefix}_api_key")
    model = st.selectbox(f"{title} Model", model_options, key=f"{key_prefix}_model")
    return provider, model, api_key


def artifact_url(path: str | None) -> str | None:
    if not path:
        return None
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return f"{API_URL}{path}"


def fetch_artifact_bytes(path: str | None) -> bytes | None:
    url = artifact_url(path)
    if not url:
        return None
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.content
    except Exception:
        return None


def render_pdf_preview(path: str | None, height: int = 800):
    content = fetch_artifact_bytes(path)
    if content:
        pdf_viewer(input=content, width=700, height=height)
    else:
        st.warning("Could not load PDF preview.")


def render_download_button(label: str, path: str | None, file_name: str, mime: str):
    content = fetch_artifact_bytes(path)
    if content:
        st.download_button(label, data=content, file_name=file_name, mime=mime, use_container_width=True)

# --- Header ---
st.title("🏥 MediScan: AI Medical Record Digitizer")

# --- Sidebar (Configuration & Upload) ---
with st.sidebar:
    st.header("⚙️ AI Configuration")

    ocr_backend_option = st.selectbox(
        "OCR Backend",
        [
            "DeepSeek-OCR (Ollama)",
            "GLM-OCR (Ollama)",
            "PaddleOCR-VL-1.5 (Local Python)",
            "PaddleOCR-VL-1.5 (Local Service)",
        ],
    )
    ocr_prompt_mode = st.selectbox(
        "OCR Mode",
        ["text", "ocr", "table", "formula", "chart", "spotting", "seal"],
        index=0,
    )
    use_gpu = st.checkbox("Use local GPU/CUDA", value=True)
    workflow_mode = st.radio(
        "Extraction workflow",
        ["Direct pipeline", "Granular extraction graph", "Agentic workflow"],
        index=1,
        help="Direct: fast single-call pipeline. Granular: step-by-step LangGraph with classify → split → OCR → validate → normalize → retrieve → merge → confidence gate. Agentic: first-gen LangGraph workflow.",
    )
    agentic_mode = workflow_mode == "Agentic workflow"
    extraction_graph_mode = workflow_mode == "Granular extraction graph"

    ocr_backend = "ollama"
    ocr_model = "deepseek-ocr"
    paddle_service_url = None
    if ocr_backend_option == "GLM-OCR (Ollama)":
        ocr_backend = "glm"
        ocr_model = "glm-ocr"
    elif ocr_backend_option == "PaddleOCR-VL-1.5 (Local Python)":
        ocr_backend = "paddle"
        ocr_model = "PaddlePaddle/PaddleOCR-VL-1.5"
    elif ocr_backend_option == "PaddleOCR-VL-1.5 (Local Service)":
        ocr_backend = "paddle"
        ocr_model = "PaddlePaddle/PaddleOCR-VL-1.5"
        paddle_service_url = st.text_input("PaddleOCR-VL Service URL", value="http://127.0.0.1:8118/v1")

    structuring_provider, structuring_model, structuring_api_key = render_model_config(
        "Structuring",
        "structuring",
    )
    reasoning_provider, reasoning_model, reasoning_api_key = render_model_config(
        "Reasoning",
        "reasoning",
    )

    st.header("Upload Medical Record")
    uploaded_file = st.file_uploader("Upload PDF/Image", type=["pdf", "jpg", "png"])
    
    if uploaded_file and st.button("🚀 Analyze Document"):
        with st.spinner(f"Running Analysis with {structuring_provider} ({structuring_model}) and {reasoning_provider} ({reasoning_model})..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {
                "provider": structuring_provider,
                "model": structuring_model,
                "api_key": structuring_api_key if structuring_api_key else "",
                "structuring_provider": structuring_provider,
                "structuring_model": structuring_model,
                "structuring_api_key": structuring_api_key if structuring_api_key else "",
                "reasoning_provider": reasoning_provider,
                "reasoning_model": reasoning_model,
                "reasoning_api_key": reasoning_api_key if reasoning_api_key else "",
                "ocr_backend": ocr_backend,
                "ocr_model": ocr_model,
                "ocr_mode": ocr_prompt_mode,
                "use_gpu": str(use_gpu).lower(),
                "paddle_service_url": paddle_service_url if paddle_service_url else "",
                "agentic_mode": str(agentic_mode).lower(),
                "extraction_graph_mode": str(extraction_graph_mode).lower(),
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
                    st.session_state['file_url'] = data.get('file_url')
                    st.session_state['ocr_artifacts'] = data.get('ocr')
                    st.session_state['annotated_pdf_path'] = data.get('annotated_pdf_path')
                    st.session_state['annotated_pdf_url'] = data.get('annotated_pdf_url')
                    st.session_state['annotated_image_paths'] = data.get('annotated_image_paths', [])
                    st.session_state['annotated_image_urls'] = data.get('annotated_image_urls', [])
                    st.session_state['page_image_urls'] = data.get('page_image_urls', [])
                    st.session_state['bounding_boxes'] = data.get('bounding_boxes', [])
                    st.session_state['requires_human_review'] = data.get('requires_human_review', False)
                    st.session_state['vector_index_status'] = data.get('vector_index_status')
                    st.session_state['retrieval_enabled'] = data.get('retrieval_enabled', False)
                    st.session_state['ocr_supports_bboxes'] = data.get('ocr_supports_bboxes', False)
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
        if st.session_state['requires_human_review']:
            st.warning("This document passed through the agentic validation gate and should receive human review before operational use.")

        col1, col2 = st.columns([1, 1])
        
        with col1:
            doc_tabs = st.tabs(["📄 Original", "🖍️ Annotated", "🔎 Overlay", "📦 Bounding Boxes"])
            with doc_tabs[0]:
                st.subheader("Original Document")
                if st.session_state['file_url']:
                    if str(st.session_state['file_url']).endswith('.pdf'):
                        render_pdf_preview(st.session_state['file_url'])
                    else:
                        page_urls = st.session_state.get('page_image_urls') or [st.session_state['file_url']]
                        for page_url in page_urls:
                            st.image(artifact_url(page_url), use_container_width=True)
                    render_download_button("Download Original", st.session_state['file_url'], "original_document", "application/pdf")

            with doc_tabs[1]:
                st.subheader("Annotated Output")
                if not st.session_state.get('ocr_supports_bboxes'):
                    st.info("Current OCR backend does not produce bounding boxes. Switch to PaddleOCR-VL for visual annotations.")
                if st.session_state['annotated_pdf_url']:
                    render_pdf_preview(st.session_state['annotated_pdf_url'])
                    render_download_button(
                        "Download Annotated PDF",
                        st.session_state['annotated_pdf_url'],
                        "annotated_document.pdf",
                        "application/pdf",
                    )
                elif st.session_state['annotated_image_urls']:
                    for annotated_url in st.session_state['annotated_image_urls']:
                        st.image(artifact_url(annotated_url), use_container_width=True)
                else:
                    st.info("No annotated output is available for this OCR backend yet.")

            with doc_tabs[2]:
                st.subheader("Original vs Annotated")
                if not st.session_state.get('ocr_supports_bboxes'):
                    st.info("Overlay comparison requires an OCR backend that produces bounding boxes (PaddleOCR-VL).")
                original_urls = st.session_state.get('page_image_urls') or []
                annotated_urls = st.session_state.get('annotated_image_urls') or []
                if original_urls and annotated_urls:
                    page_count = min(len(original_urls), len(annotated_urls))
                    selected_page = st.selectbox(
                        "Page",
                        list(range(1, page_count + 1)),
                        format_func=lambda page: f"Page {page}",
                    )
                    overlay_left, overlay_right = st.columns(2)
                    with overlay_left:
                        st.caption("Original")
                        st.image(artifact_url(original_urls[selected_page - 1]), use_container_width=True)
                    with overlay_right:
                        st.caption("Annotated")
                        st.image(artifact_url(annotated_urls[selected_page - 1]), use_container_width=True)
                else:
                    st.info("Overlay preview is available when both rendered pages and annotated pages exist.")

            with doc_tabs[3]:
                st.subheader("Bounding Boxes")
                bounding_boxes = st.session_state['bounding_boxes'] or []
                if bounding_boxes:
                    st.dataframe(pd.DataFrame(bounding_boxes), use_container_width=True)
                else:
                    st.info("No bounding boxes were returned for this OCR backend.")
        
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

            vector_index_status = st.session_state.get('vector_index_status')
            if vector_index_status:
                st.divider()
                st.subheader("🧠 Retrieval Index")
                if not st.session_state.get('retrieval_enabled'):
                    st.warning("Semantic retrieval is disabled. Configure Qdrant (QDRANT_ENABLED=true) to enable cross-document search.")
                st.json(vector_index_status)

            if st.session_state.get('ocr_artifacts'):
                st.divider()
                st.subheader("OCR Metadata")
                st.json(
                    {
                        "backend": st.session_state['ocr_artifacts'].get('backend'),
                        "model": st.session_state['ocr_artifacts'].get('model'),
                        "ocr_mode": st.session_state['ocr_artifacts'].get('ocr_mode'),
                        "pages": len(st.session_state['ocr_artifacts'].get('per_page_results', [])),
                        "annotations": st.session_state['ocr_artifacts'].get('annotations_metadata', {}),
                    }
                )

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
                files = {"policy_file": (policy_file.name, policy_file.getvalue(), policy_file.type)}
                # Send the extracted medical data as a JSON string field
                payload = {
                    "medical_json": json.dumps(st.session_state['extracted_data']),
                    "provider": reasoning_provider,
                    "model": reasoning_model,
                    "api_key": reasoning_api_key if reasoning_api_key else "",
                    "reasoning_provider": reasoning_provider,
                    "reasoning_model": reasoning_model,
                    "reasoning_api_key": reasoning_api_key if reasoning_api_key else "",
                    "ocr_backend": ocr_backend,
                    "ocr_model": ocr_model,
                    "ocr_mode": ocr_prompt_mode,
                    "use_gpu": str(use_gpu).lower(),
                    "paddle_service_url": paddle_service_url if paddle_service_url else "",
                    "policy_ocr": str(policy_file.type != "text/plain").lower(),
                }
                
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
