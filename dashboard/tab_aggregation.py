import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from .helpers import fetch_node_metrics, save_metrics


def _get_backend_models(central_api_url: str) -> list[str]:
    try:
        response = requests.get(f"{central_api_url}/models/list", timeout=5)
        if response.status_code == 200:
            payload = response.json()
            return payload.get("models", [])
    except Exception:
        pass
    return []


def render(h1_url, h2_url, h1_online, h2_online, central_api_url):
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
                    r1 = requests.post(
                        f"{central_api_url}/nodes/retrieve",
                        json={"node_url": h1_url, "target_filename": "hospital_1_v2.pkl"},
                        timeout=30,
                    )
                    if r1.status_code == 200:
                        st.session_state.h1_model_info = r1.json().get("model_info")
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
                        st.error(f"Failed — HTTP {r1.status_code}: {r1.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

        if st.session_state.h1_downloaded:
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
                    r2 = requests.post(
                        f"{central_api_url}/nodes/retrieve",
                        json={"node_url": h2_url, "target_filename": "hospital_2_v2.pkl"},
                        timeout=30,
                    )
                    if r2.status_code == 200:
                        st.session_state.h2_model_info = r2.json().get("model_info")
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
                        st.error(f"Failed — HTTP {r2.status_code}: {r2.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

        if st.session_state.h2_downloaded:
            if st.session_state.h2_model_info:
                with st.expander("Hospital 2 — Model Details", expanded=True):
                    for k, v in st.session_state.h2_model_info.items():
                        st.markdown(f"**{k}:** `{v}`")

    st.divider()

    # ---- Status Table ----
    st.subheader("Central Server Model Storage Status")
    backend_models = set(_get_backend_models(central_api_url))
    node_rows = [
        {
            "Node": "Hospital 1",
            "Network Status": "Online" if h1_online else "Offline",
            "Model File": "hospital_1_v2.pkl",
        },
        {
            "Node": "Hospital 2",
            "Network Status": "Online" if h2_online else "Offline",
            "Model File": "hospital_2_v2.pkl",
        },
    ]

    table_rows = []
    for row in node_rows:
        table_rows.append({
            "Node": row["Node"],
            "Network Status": row["Network Status"],
            "Model File": row["Model File"],
            "Stored on Central Server": "Yes" if row["Model File"] in backend_models else "No",
        })

    status_df = pd.DataFrame(table_rows)
    st.dataframe(status_df, use_container_width=True, hide_index=True)

    st.divider()

    # ---- FedAvg Aggregation ----
    st.subheader("Federated Averaging (FedAvg)")
    models_ready = {"hospital_1_v2.pkl", "hospital_2_v2.pkl"}.issubset(backend_models)

    if not models_ready:
        st.info("Retrieve models from both Hospital 1 and Hospital 2 before aggregating.")

    if st.button("Aggregate Models and Update Global Model (main_model_v2)",
                 type="primary", disabled=not models_ready, use_container_width=True):
        with st.status("Performing Federated Averaging...", expanded=True) as agg_status:
            try:
                n1 = 12842
                n2 = 12842
                total_n = n1 + n2
                response = requests.post(
                    f"{central_api_url}/aggregate",
                    json={"n1": n1, "n2": n2},
                    timeout=45,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

                st.session_state.agg_model_info = response.json().get("model_info")

                # Verify the aggregated model is visible on the backend disk listing.
                models_after = set(_get_backend_models(central_api_url))
                if "main_model_v2.pkl" not in models_after:
                    raise RuntimeError(
                        "Aggregation API returned success, but main_model_v2.pkl is not visible on central storage. "
                        "Check MODEL_PATH and volume mount settings."
                    )

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
                st.rerun()

            except Exception as e:
                agg_status.update(label="Aggregation failed", state="error")
                st.error(f"Error: {str(e)}")

    if st.session_state.aggregation_done:
        if st.session_state.agg_model_info:
            with st.expander("Aggregated Global Model (main_model_v2) — Details", expanded=False):
                for k, v in st.session_state.agg_model_info.items():
                    st.markdown(f"**{k}:** `{v}`")
                if "aggregation_time" in st.session_state:
                    st.markdown(f"**Aggregated at:** `{st.session_state.aggregation_time}`")
