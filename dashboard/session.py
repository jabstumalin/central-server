import streamlit as st
import os
from .config import MODELS_DIR
from .helpers import load_metrics


def init_session_state():
    if "h1_downloaded" not in st.session_state:
        st.session_state.h1_downloaded = os.path.exists(os.path.join(MODELS_DIR, "hospital_1_v2.pkl"))
    if "h2_downloaded" not in st.session_state:
        st.session_state.h2_downloaded = os.path.exists(os.path.join(MODELS_DIR, "hospital_2_v2.pkl"))
    if "aggregation_done" not in st.session_state:
        st.session_state.aggregation_done = os.path.exists(os.path.join(MODELS_DIR, "main_model_v2.pkl"))
    if "h1_model_info" not in st.session_state:
        st.session_state.h1_model_info = None
    if "h2_model_info" not in st.session_state:
        st.session_state.h2_model_info = None
    if "agg_model_info" not in st.session_state:
        st.session_state.agg_model_info = None

    if "metrics" not in st.session_state:
        # Try loading persisted metrics from disk first
        saved = load_metrics()
        if saved:
            st.session_state.metrics = saved["metrics"]
            st.session_state.metrics_source = saved["sources"]
            st.session_state.metrics_last_updated = saved.get("last_updated")
        else:
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

    if "metrics_source" not in st.session_state:
        st.session_state.metrics_source = {
            "Main Model v1":              "baseline",
            "Hospital-1 v2":              "pending",
            "Hospital-2 v2":              "pending",
            "Main Model v2 (Aggregated)": "pending",
        }
    if "metrics_last_updated" not in st.session_state:
        st.session_state.metrics_last_updated = None
