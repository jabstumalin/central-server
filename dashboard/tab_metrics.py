import streamlit as st
import pandas as pd
import requests


def render(central_api_url: str):
    st.header("Model Evaluation Results")
    st.markdown("Evaluation metrics for all models involved in the federated learning process.")

    if st.session_state.aggregation_done:
        st.success("Main Model v2 is available — aggregation has been performed.")
    else:
        st.info("Main Model v2 metrics are based on default values. Run aggregation in the Federated Aggregation tab to update.")

    metrics = st.session_state.metrics
    df_results = pd.DataFrame([
        {"Model": model, **vals} for model, vals in metrics.items()
    ])

    # ---- Summary metric cards ----
    st.subheader("Performance Summary")
    best_row = df_results.loc[df_results["Accuracy"].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Accuracy",  f"{best_row['Accuracy']:.4f}",  best_row["Model"])
    c2.metric("Best Precision", f"{best_row['Precision']:.4f}", best_row["Model"])
    c3.metric("Best Recall",    f"{best_row['Recall']:.4f}",    best_row["Model"])
    c4.metric("Best F1 Score",  f"{best_row['F1 Score']:.4f}",  best_row["Model"])

    st.divider()

    # ---- Detailed metrics table ----
    st.subheader("Detailed Metrics Table")

    # Query central server for actual model files on disk
    try:
        resp = requests.get(f"{central_api_url}/models/list", timeout=5)
        if resp.status_code == 200:
            backend_models = set(resp.json().get("models", []))
        else:
            backend_models = set()
    except Exception:
        backend_models = set()

    def row_availability(model_name):
        mapping = {
            "Main Model v1":              "main_model_v1.pkl",
            "Hospital-1 v2":              "hospital_1_v2.pkl",
            "Hospital-2 v2":              "hospital_2_v2.pkl",
            "Main Model v2 (Aggregated)": "main_model_v2.pkl",
        }
        fname = mapping.get(model_name)
        if fname and fname in backend_models:
            return "Available"
        return "Not on disk"

    df_results["Model File Status"] = df_results["Model"].apply(row_availability)

    styled = df_results.style.format({
        "Accuracy":  "{:.4f}",
        "Precision": "{:.4f}",
        "Recall":    "{:.4f}",
        "F1 Score":  "{:.4f}",
    }).set_properties(subset=["Accuracy", "Precision", "Recall", "F1 Score"],
                      **{"text-align": "center"}) \
     .set_table_styles([
        {"selector": "th", "props": [("text-align", "center"),
                                     ("font-weight", "bold"),
                                     ("background-color", "#f0f2f6")]},
        {"selector": "td", "props": [("padding", "8px")]},
    ])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # ---- Retrieved model inspection ----
    st.subheader("Retrieved Model Inspection")

    inspect_col1, inspect_col2, inspect_col3 = st.columns(3)

    with inspect_col1:
        st.markdown("**Hospital-1 v2**")
        if st.session_state.h1_downloaded and st.session_state.h1_model_info:
            info = st.session_state.h1_model_info
            st.caption(f"File Size: {info.get('File Size', 'N/A')}")
            st.caption(f"Features:  {info.get('Features', 'N/A')}")
            st.caption(f"Classes:   {info.get('Classes', 'N/A')}")
        else:
            st.caption("Not yet retrieved")

    with inspect_col2:
        st.markdown("**Hospital-2 v2**")
        if st.session_state.h2_downloaded and st.session_state.h2_model_info:
            info = st.session_state.h2_model_info
            st.caption(f"File Size: {info.get('File Size', 'N/A')}")
            st.caption(f"Features:  {info.get('Features', 'N/A')}")
            st.caption(f"Classes:   {info.get('Classes', 'N/A')}")
        else:
            st.caption("Not yet retrieved")

    with inspect_col3:
        st.markdown("**Main Model v2 (Aggregated)**")
        if st.session_state.aggregation_done and st.session_state.agg_model_info:
            info = st.session_state.agg_model_info
            st.caption(f"File Size: {info.get('File Size', 'N/A')}")
            st.caption(f"Features:  {info.get('Features', 'N/A')}")
            st.caption(f"Classes:   {info.get('Classes', 'N/A')}")
            if "aggregation_time" in st.session_state:
                st.caption(f"Aggregated: {st.session_state.aggregation_time}")
        else:
            st.caption("Not yet aggregated")

    st.divider()

    # ---- Charts ----
    st.subheader("Performance Visualizations")

    chart_data = df_results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Accuracy and F1 Score**")
        st.bar_chart(chart_data[["Accuracy", "F1 Score"]], height=350)
    with col2:
        st.markdown("**Precision and Recall**")
        st.bar_chart(chart_data[["Precision", "Recall"]], height=350)

    st.divider()
