"""
FastAPI Main Server
Central server for handling API requests
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import os
import io
import zipfile
from pathlib import Path
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
    # Determine which model version to use
    model_v1_path = os.path.join(settings.MODEL_PATH, "main_model_v1.pkl")
    scaler_path = os.path.join(settings.MODEL_PATH, "global_scaler.pkl")
    
    if os.path.exists(model_v1_path):
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


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
