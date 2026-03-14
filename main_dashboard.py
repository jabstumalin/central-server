"""
Federated Learning - Central Server Dashboard
Entry point: streamlit run main_dashboard.py
"""
import os
import streamlit as st
import requests

from dashboard.config import HOSPITAL_1_URL, HOSPITAL_2_URL, MODELS_DIR
from dashboard.session import init_session_state
from dashboard.helpers import check_node_status, METRICS_FILE
from dashboard.tab_aggregation import render as render_aggregation
from dashboard.tab_metrics import render as render_metrics

os.makedirs(MODELS_DIR, exist_ok=True)
init_session_state()

# --- Page Config ---
st.set_page_config(
    page_title="Federated Learning Central Server",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.25rem; }
    .sub-header  { font-size: 1rem; color: #666; margin-bottom: 1.5rem; }
    .model-card  { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px;
                   padding: 1rem 1.2rem; margin-top: 0.75rem; }
    .status-ok   { color: #198754; font-weight: 600; }
    .status-err  { color: #dc3545; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Federated Learning Central Server</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Orchestrates model retrieval, aggregation (FedAvg), and performance tracking</p>', unsafe_allow_html=True)
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("Node Configuration")
    h1_url = st.text_input("Hospital 1 API URL", value=HOSPITAL_1_URL)
    h2_url = st.text_input("Hospital 2 API URL", value=HOSPITAL_2_URL)
    st.divider()

    st.markdown("### Models on Disk")
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    if model_files:
        for mf in sorted(model_files):
            size_kb = os.path.getsize(os.path.join(MODELS_DIR, mf)) / 1024
            st.text(f"• {mf}  ({size_kb:.1f} KB)")
    else:
        st.caption("No models found")
    st.divider()

    st.markdown("### Danger Zone")
    if st.button("Reset Central Server", type="secondary", use_container_width=True):
        try:
            r = requests.post("http://localhost:8000/reset", timeout=5)
            if r.status_code == 200:
                data = r.json()
                st.session_state.h1_downloaded = False
                st.session_state.h2_downloaded = False
                st.session_state.aggregation_done = False
                st.session_state.h1_model_info = None
                st.session_state.h2_model_info = None
                st.session_state.agg_model_info = None
                if "aggregation_time" in st.session_state:
                    del st.session_state.aggregation_time
                # Clear persisted metrics
                st.session_state.metrics = {
                    "Main Model v1":              {"Accuracy": 0.7243, "Precision": 0.7434, "Recall": 0.6847, "F1 Score": 0.7128},
                    "Hospital-1 v2":              {"Accuracy": None, "Precision": None, "Recall": None, "F1 Score": None},
                    "Hospital-2 v2":              {"Accuracy": None, "Precision": None, "Recall": None, "F1 Score": None},
                    "Main Model v2 (Aggregated)": {"Accuracy": None, "Precision": None, "Recall": None, "F1 Score": None},
                }
                st.session_state.metrics_source = {
                    "Main Model v1":              "baseline",
                    "Hospital-1 v2":              "pending",
                    "Hospital-2 v2":              "pending",
                    "Main Model v2 (Aggregated)": "pending",
                }
                st.session_state.metrics_last_updated = None
                if os.path.exists(METRICS_FILE):
                    os.remove(METRICS_FILE)
                deleted = data.get("deleted_files", [])
                if deleted:
                    st.success(f"Reset complete. Deleted: {', '.join(deleted)}")
                else:
                    st.info("Reset complete. No model files were found.")
                st.rerun()
            else:
                st.error(f"Reset failed — HTTP {r.status_code}")
        except Exception as e:
            st.error(f"Could not reach API: {e}")

    st.divider()
    st.caption("Federated Learning System v1.0")

# --- Node status ---
h1_online = check_node_status(h1_url)
h2_online = check_node_status(h2_url)

# --- Tabs ---
tab1, tab2 = st.tabs(["Federated Aggregation", "Performance Metrics"])

with tab1:
    render_aggregation(h1_url, h2_url, h1_online, h2_online)

with tab2:
    render_metrics()
