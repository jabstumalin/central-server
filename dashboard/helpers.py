import streamlit as st
import requests
import joblib
import json
import os

METRICS_FILE = os.path.join("models", "metrics.json")


@st.cache_data(ttl=5)
def check_node_status(url):
    for path in ("/health", "/"):
        try:
            response = requests.get(f"{url}{path}", timeout=2)
            if response.status_code == 200:
                return True
        except Exception:
            pass
    return False


def inspect_model(path):
    try:
        model = joblib.load(path)
        file_size = os.path.getsize(path)
        info = {
            "File": os.path.basename(path),
            "File Size": f"{file_size:,} bytes ({file_size / 1024:.1f} KB)",
            "Model Type": type(model).__name__,
        }
        if hasattr(model, "coef_"):
            info["Features"] = model.coef_.shape[1]
            info["Classes"] = list(model.classes_)
            info["Coefficients Shape"] = str(model.coef_.shape)
            info["Intercept"] = [round(float(x), 6) for x in model.intercept_]
        return info, model
    except Exception as e:
        return {"Error": str(e)}, None


def fetch_node_metrics(url):
    try:
        r = requests.get(f"{url}/metrics", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return {
                "Accuracy":  float(data.get("accuracy",  data.get("Accuracy",  0))),
                "Precision": float(data.get("precision", data.get("Precision", 0))),
                "Recall":    float(data.get("recall",    data.get("Recall",    0))),
                "F1 Score":  float(
                    data.get("f1_score", data.get("f1", data.get("F1 Score", data.get("f1score", 0))))
                ),
            }
    except Exception:
        pass
    return None


def save_metrics(metrics: dict, sources: dict, last_updated: str):
    """Persist metrics to models/metrics.json."""
    os.makedirs("models", exist_ok=True)
    payload = {
        "metrics": metrics,
        "sources": sources,
        "last_updated": last_updated,
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def load_metrics() -> dict | None:
    """Load persisted metrics from models/metrics.json. Returns None if not found."""
    if not os.path.exists(METRICS_FILE):
        return None
    try:
        with open(METRICS_FILE) as f:
            return json.load(f)
    except Exception:
        return None
