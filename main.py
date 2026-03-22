"""
FastAPI Main Server
Central server for handling API requests
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Any
import uvicorn
import os
import io
import glob
import zipfile
import requests
import joblib
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from config import settings

app = FastAPI(
    title=settings.API_TITLE,
    description="Main server for handling global requests",
    version=settings.API_VERSION
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RetrieveModelRequest(BaseModel):
    node_url: str
    target_filename: str


class AggregateRequest(BaseModel):
    n1: int = 12842
    n2: int = 12842


def _to_jsonable(obj: Any) -> Any:
    """Convert nested structures (including numpy types) to JSON-serializable primitives."""
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - fallback when numpy unavailable
        np = None  # type: ignore

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if np is not None:
        if isinstance(obj, np.generic):  # numpy scalar, e.g. np.int64
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    # Fallback: string representation
    return str(obj)


def inspect_model(path: str) -> dict:
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
    return info


# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": settings.API_TITLE,
        "status": "running",
        "version": settings.API_VERSION
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/global/package")
async def get_global_package():
    """
    Endpoint for hospital nodes to retrieve both global model and scaler as a zip package.
    This is the recommended endpoint for nodes to download all necessary files at once.
    """
    # Determine which model version to use (prefer v2, fall back to v1)
    model_v2_path = os.path.join(settings.MODEL_PATH, "main_model_v2.pkl")
    model_v1_path = os.path.join(settings.MODEL_PATH, "main_model_v1.pkl")
    scaler_path = os.path.join(settings.MODEL_PATH, "global_scaler.pkl")
    
    if os.path.exists(model_v2_path):
        model_path = model_v2_path
        model_filename = "main_model_v2.pkl"
    elif os.path.exists(model_v1_path):
        model_path = model_v1_path
        model_filename = "main_model_v1.pkl"
    else:
        raise HTTPException(
            status_code=404,
            detail="Global model not found. Please ensure the model is trained and available."
        )
    
    # Check if scaler exists
    if not os.path.exists(scaler_path):
        raise HTTPException(
            status_code=404,
            detail="Global scaler not found. Both model and scaler are required."
        )
    
    # Create a zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add model to zip
        zip_file.write(model_path, arcname=model_filename)
        # Add scaler to zip
        zip_file.write(scaler_path, arcname="global_scaler.pkl")
    
    zip_buffer.seek(0)
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=global_model_package.zip"}
    )


@app.get("/global/model")
async def get_global_model():
    """
    Endpoint for hospital nodes to retrieve the global model.
    Returns the latest version of the main model.
    """
    # Check for v2 first, then fall back to v1
    model_v2_path = os.path.join(settings.MODEL_PATH, "main_model_v2.pkl")
    model_v1_path = os.path.join(settings.MODEL_PATH, "main_model_v1.pkl")
    
    if os.path.exists(model_v2_path):
        return FileResponse(
            path=model_v2_path,
            media_type="application/octet-stream",
            filename="main_model_v2.pkl"
        )
    elif os.path.exists(model_v1_path):
        return FileResponse(
            path=model_v1_path,
            media_type="application/octet-stream",
            filename="main_model_v1.pkl"
        )
    else:
        raise HTTPException(
            status_code=404,
            detail="Global model not found. Please ensure the model is trained and available."
        )


@app.get("/global/scaler")
async def get_global_scaler():
    """
    Endpoint for hospital nodes to retrieve the global scaler.
    """
    scaler_path = os.path.join(settings.MODEL_PATH, "global_scaler.pkl")
    
    if os.path.exists(scaler_path):
        return FileResponse(
            path=scaler_path,
            media_type="application/octet-stream",
            filename="global_scaler.pkl"
        )
    else:
        raise HTTPException(
            status_code=404,
            detail="Global scaler not found."
        )


@app.post("/reset")
def reset_node():
    """Reset the central server by deleting all locally stored model files, excluding baseline models."""
    baseline_files = {"main_model_v1.pkl", "global_scaler.pkl"}
    deleted_files = []
    models_dir = settings.MODEL_PATH

    if not os.path.exists(models_dir):
        return {
            "status": "success",
            "message": "Model directory does not exist. Nothing to reset.",
            "deleted_files": [],
            "directory": models_dir,
        }

    for file in glob.glob(os.path.join(models_dir, "*.pkl")):
        if os.path.basename(file) in baseline_files:
            continue
        try:
            os.remove(file)
            deleted_files.append(os.path.basename(file))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed deleting {os.path.basename(file)}: {exc}") from exc

    return {
        "status": "success",
        "message": "Node reset successfully.",
        "deleted_files": deleted_files,
        "directory": models_dir,
    }


@app.get("/models/list")
async def list_models():
    """
    List all available models in the models directory.
    """
    models_dir = settings.MODEL_PATH
    
    if not os.path.exists(models_dir):
        return {"models": []}
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    return {
        "models": model_files,
        "count": len(model_files),
        "directory": models_dir
    }


@app.post("/nodes/retrieve")
async def retrieve_node_model(payload: RetrieveModelRequest):
    """Retrieve a hospital model from a node endpoint and save it on the central server."""
    os.makedirs(settings.MODEL_PATH, exist_ok=True)

    try:
        response = requests.get(f"{payload.node_url}/model/download", timeout=20)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not connect to node: {exc}") from exc

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Node returned HTTP {response.status_code}")

    target_path = os.path.join(settings.MODEL_PATH, payload.target_filename)
    with open(target_path, "wb") as f:
        f.write(response.content)

    try:
        info = inspect_model(target_path)
        info = _to_jsonable(info)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Downloaded file is not a valid model: {exc}") from exc

    return {
        "status": "success",
        "saved_to": target_path,
        "model_info": info,
    }


@app.post("/aggregate")
async def aggregate_models(payload: AggregateRequest):
    """Aggregate hospital models using weighted FedAvg and save main_model_v2.pkl."""
    h1_path = os.path.join(settings.MODEL_PATH, "hospital_1_v2.pkl")
    h2_path = os.path.join(settings.MODEL_PATH, "hospital_2_v2.pkl")

    if not os.path.exists(h1_path) or not os.path.exists(h2_path):
        raise HTTPException(
            status_code=400,
            detail="Both hospital_1_v2.pkl and hospital_2_v2.pkl must exist before aggregation.",
        )

    try:
        m1 = joblib.load(h1_path)
        m2 = joblib.load(h2_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model files: {exc}") from exc

    n1 = payload.n1
    n2 = payload.n2
    total_n = n1 + n2
    if total_n <= 0:
        raise HTTPException(status_code=400, detail="n1 + n2 must be greater than zero.")

    try:
        averaged_coef = ((m1.coef_ * n1) + (m2.coef_ * n2)) / total_n
        averaged_intercept = ((m1.intercept_ * n1) + (m2.intercept_ * n2)) / total_n

        main_v2 = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
        main_v2.coef_ = averaged_coef
        main_v2.intercept_ = averaged_intercept
        main_v2.classes_ = m1.classes_
        main_v2.t_ = getattr(m1, "t_", 1.0) + getattr(m2, "t_", 1.0)
        if hasattr(m1, "n_iter_") and hasattr(m2, "n_iter_"):
            main_v2.n_iter_ = max(m1.n_iter_, m2.n_iter_)

        v2_path = os.path.join(settings.MODEL_PATH, "main_model_v2.pkl")
        joblib.dump(main_v2, v2_path)
        info = inspect_model(v2_path)
        info = _to_jsonable(info)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Aggregation failed: {exc}") from exc

    return {
        "status": "success",
        "saved_to": v2_path,
        "model_info": info,
        "weights": {"n1": n1, "n2": n2, "total": total_n},
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
