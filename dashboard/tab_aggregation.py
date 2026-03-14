import streamlit as st
import requests
import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.linear_model import SGDClassifier

from .config import MODELS_DIR
from .helpers import inspect_model, fetch_node_metrics, save_metrics


def _find_latest_node_model(node_prefix: str) -> str | None:
    """Return the newest model file matching a node prefix in MODELS_DIR."""
    if not os.path.isdir(MODELS_DIR):
        return None

    matches = [
        name for name in os.listdir(MODELS_DIR)
        if name.lower().endswith(".pkl") and name.lower().startswith(node_prefix.lower())
    ]
    if not matches:
        return None

    matches.sort(
        key=lambda name: os.path.getmtime(os.path.join(MODELS_DIR, name)),
        reverse=True,
    )
    return matches[0]


def render(h1_url, h2_url, h1_online, h2_online):
    st.header("Node Management & Model Retrieval")
    st.markdown("Retrieve locally trained models from each hospital node, inspect them, then run Federated Averaging.")

    col1, col2 = st.columns(2)

    # ---- Hospital 1 ----
    with col1:
        st.subheader("Hospital 1 Node")
        if h1_online:
            st.success("Online")
        else:
            st.error("Offline")

        if st.button("Retrieve Model — Hospital 1", disabled=not h1_online, use_container_width=True, type="primary"):
            with st.spinner("Downloading from Hospital 1..."):
                try:
                    r1 = requests.get(f"{h1_url}/model/download", timeout=10)
                    if r1.status_code == 200:
                        h1_path = os.path.join(MODELS_DIR, "hospital_1_v2.pkl")
                        with open(h1_path, "wb") as f:
                            f.write(r1.content)
                        info, _ = inspect_model(h1_path)
                        st.session_state.h1_model_info = info
                        st.session_state.h1_downloaded = True
                        live_m = fetch_node_metrics(h1_url)
                        if live_m:
                            st.session_state.metrics["Hospital-1 v2"] = live_m
                            st.session_state.metrics_source["Hospital-1 v2"] = "live"
                            st.session_state.metrics_last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            save_metrics(st.session_state.metrics, st.session_state.metrics_source, st.session_state.metrics_last_updated)
                        st.toast("Hospital 1 model retrieved!", icon="✅")
                        st.rerun()
                    else:
                        st.error(f"Failed — HTTP {r1.status_code}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

        if st.session_state.h1_downloaded:
            h1_path = os.path.join(MODELS_DIR, "hospital_1_v2.pkl")
            if st.session_state.h1_model_info is None and os.path.exists(h1_path):
                info, _ = inspect_model(h1_path)
                st.session_state.h1_model_info = info
            if st.session_state.h1_model_info:
                with st.expander("Hospital 1 — Model Details", expanded=True):
                    for k, v in st.session_state.h1_model_info.items():
                        st.markdown(f"**{k}:** `{v}`")

    # ---- Hospital 2 ----
    with col2:
        st.subheader("Hospital 2 Node")
        if h2_online:
            st.success("Online")
        else:
            st.error("Offline")

        if st.button("Retrieve Model — Hospital 2", disabled=not h2_online, use_container_width=True, type="primary"):
            with st.spinner("Downloading from Hospital 2..."):
                try:
                    r2 = requests.get(f"{h2_url}/model/download", timeout=10)
                    if r2.status_code == 200:
                        h2_path = os.path.join(MODELS_DIR, "hospital_2_v2.pkl")
                        with open(h2_path, "wb") as f:
                            f.write(r2.content)
                        info, _ = inspect_model(h2_path)
                        st.session_state.h2_model_info = info
                        st.session_state.h2_downloaded = True
                        live_m = fetch_node_metrics(h2_url)
                        if live_m:
                            st.session_state.metrics["Hospital-2 v2"] = live_m
                            st.session_state.metrics_source["Hospital-2 v2"] = "live"
                            st.session_state.metrics_last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            save_metrics(st.session_state.metrics, st.session_state.metrics_source, st.session_state.metrics_last_updated)
                        st.toast("Hospital 2 model retrieved!", icon="✅")
                        st.rerun()
                    else:
                        st.error(f"Failed — HTTP {r2.status_code}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

        if st.session_state.h2_downloaded:
            h2_path = os.path.join(MODELS_DIR, "hospital_2_v2.pkl")
            if st.session_state.h2_model_info is None and os.path.exists(h2_path):
                info, _ = inspect_model(h2_path)
                st.session_state.h2_model_info = info
            if st.session_state.h2_model_info:
                with st.expander("Hospital 2 — Model Details", expanded=True):
                    for k, v in st.session_state.h2_model_info.items():
                        st.markdown(f"**{k}:** `{v}`")

    st.divider()

    # ---- Status Table ----
    st.subheader("Local Model Storage Status")
    node_rows = [
        {
            "Node": "Hospital 1",
            "Network Status": "Online" if h1_online else "Offline",
            "Expected Prefix": "hospital_1",
            "Fallback File": "hospital_1_v2.pkl",
        },
        {
            "Node": "Hospital 2",
            "Network Status": "Online" if h2_online else "Offline",
            "Expected Prefix": "hospital_2",
            "Fallback File": "hospital_2_v2.pkl",
        },
    ]

    table_rows = []
    for row in node_rows:
        detected_file = _find_latest_node_model(row["Expected Prefix"])
        model_file = detected_file or row["Fallback File"]
        stored_locally = os.path.exists(os.path.join(MODELS_DIR, model_file))
        table_rows.append({
            "Node": row["Node"],
            "Network Status": row["Network Status"],
            "Model File": model_file,
            "Stored Locally": "Yes" if stored_locally else "No",
        })

    status_df = pd.DataFrame(table_rows)
    st.dataframe(status_df, use_container_width=True, hide_index=True)

    st.divider()

    # ---- FedAvg Aggregation ----
    st.subheader("Federated Averaging (FedAvg)")
    models_ready = st.session_state.h1_downloaded and st.session_state.h2_downloaded

    if not models_ready:
        st.info("Retrieve models from both Hospital 1 and Hospital 2 before aggregating.")

    if st.button("Aggregate Models and Update Global Model (main_model_v2)",
                 type="primary", disabled=not models_ready, use_container_width=True):
        with st.status("Performing Federated Averaging...", expanded=True) as agg_status:
            try:
                h1_path = os.path.join(MODELS_DIR, "hospital_1_v2.pkl")
                h2_path = os.path.join(MODELS_DIR, "hospital_2_v2.pkl")

                st.write("→ Loading hospital models...")
                m1 = joblib.load(h1_path)
                m2 = joblib.load(h2_path)
                st.write(f"  Hospital 1: {m1.coef_.shape[1]} features, classes {list(m1.classes_)}")
                st.write(f"  Hospital 2: {m2.coef_.shape[1]} features, classes {list(m2.classes_)}")

                st.write("→ Performing Weighted Federated Averaging (FedAvg)...")
                n1 = 12842
                n2 = 12842
                total_n = n1 + n2

                averaged_coef = ((m1.coef_ * n1) + (m2.coef_ * n2)) / total_n
                averaged_intercept = ((m1.intercept_ * n1) + (m2.intercept_ * n2)) / total_n

                main_v2 = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
                main_v2.coef_ = averaged_coef
                main_v2.intercept_ = averaged_intercept
                main_v2.classes_   = m1.classes_
                main_v2.t_ = getattr(m1, "t_", 1.0) + getattr(m2, "t_", 1.0)
                if hasattr(m1, "n_iter_") and hasattr(m2, "n_iter_"):
                    main_v2.n_iter_ = max(m1.n_iter_, m2.n_iter_)

                v2_path = os.path.join(MODELS_DIR, "main_model_v2.pkl")
                joblib.dump(main_v2, v2_path)
                st.write(f"✓ Global model saved: {v2_path}")

                agg_info, _ = inspect_model(v2_path)
                st.session_state.agg_model_info = agg_info
                st.session_state.aggregation_done = True
                st.session_state.aggregation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                _mk = ["Accuracy", "Precision", "Recall", "F1 Score"]
                h1_m = st.session_state.metrics.get("Hospital-1 v2", {})
                h2_m = st.session_state.metrics.get("Hospital-2 v2", {})
                if all(h1_m.get(k) is not None for k in _mk) and \
                   all(h2_m.get(k) is not None for k in _mk):
                    agg_m = {k: ((h1_m[k] * n1) + (h2_m[k] * n2)) / total_n for k in _mk}
                    st.session_state.metrics["Main Model v2 (Aggregated)"] = agg_m
                    st.session_state.metrics_source["Main Model v2 (Aggregated)"] = "computed"
                    st.session_state.metrics_last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    save_metrics(st.session_state.metrics, st.session_state.metrics_source, st.session_state.metrics_last_updated)

                agg_status.update(label="Aggregation Complete!", state="complete", expanded=False)
                st.success("Federated Averaging completed. Global model (main_model_v2.pkl) is ready. "
                           "View updated results in the Performance Metrics tab.")

            except Exception as e:
                agg_status.update(label="Aggregation failed", state="error")
                st.error(f"Error: {str(e)}")

    if st.session_state.aggregation_done:
        v2_path = os.path.join(MODELS_DIR, "main_model_v2.pkl")
        if st.session_state.agg_model_info is None and os.path.exists(v2_path):
            info, _ = inspect_model(v2_path)
            st.session_state.agg_model_info = info
        if st.session_state.agg_model_info:
            with st.expander("Aggregated Global Model (main_model_v2) — Details", expanded=False):
                for k, v in st.session_state.agg_model_info.items():
                    st.markdown(f"**{k}:** `{v}`")
                if "aggregation_time" in st.session_state:
                    st.markdown(f"**Aggregated at:** `{st.session_state.aggregation_time}`")
