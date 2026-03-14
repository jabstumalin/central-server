# Central Server

A FastAPI-based central server with Streamlit dashboard for monitoring and management.

## Features

- **FastAPI Backend**: High-performance REST API
- **Streamlit Dashboard**: Interactive web interface for monitoring and control
- **ML Integration**: Ready for scikit-learn model deployment
- **Health Monitoring**: Built-in health check endpoints
- **CORS Enabled**: Configured for cross-origin requests

## Installation

1. Create and activate virtual environment:
```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

3. (Optional) Create environment file for custom configuration:
```powershell
Copy-Item .env.example .env
# Then edit .env with your custom settings
```

Note: The `.env` file is optional. The application will use default values from [config.py](config.py) if no `.env` file exists.

## Running the Server

### Start FastAPI Server
```powershell
python main.py
```
The API will be available at: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### Start Streamlit Dashboard
```powershell
streamlit run main_dashboard.py
```
The dashboard will be available at: http://localhost:8501

### Run Both (in separate terminals)
```powershell
# Terminal 1 - FastAPI
python main.py

# Terminal 2 - Streamlit
streamlit run main_dashboard.py
```

## Project Structure

```
central-server/
├── main.py                   # FastAPI application
├── main_dashboard.py         # Streamlit federated learning dashboard
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── download_global_model.py  # Utility for hospital nodes to download global model
├── hospital_node_example.py  # Example hospital node server implementation
├── .env.example              # Environment variables template
├── .env                      # Your local environment variables (create this)
└── models/                   # ML models directory (create as needed)
```

## API Endpoints

### System Endpoints
- `GET /` - Root endpoint with server info
- `GET /health` - Health check endpoint

### Model Distribution Endpoints
- `GET /global/package` - **Recommended:** Download both global model and scaler as a zip package
- `GET /global/model` - Download only the latest global model (main_model_v2.pkl or main_model_v1.pkl)
- `GET /global/scaler` - Download only the global scaler (global_scaler.pkl)
- `GET /models/list` - List all available models in the models directory

### Usage Examples

#### Quick Start: Use the Download Script
The easiest way for hospital nodes to get the global model:
```powershell
python download_global_model.py
```
This interactive script handles the download and extraction automatically.

#### Option 1: Download Complete Package (Recommended)
Hospital nodes can retrieve both model and scaler in one request:
```python
import requests
import zipfile
import io

# Download the complete package
response = requests.get("http://127.0.0.1:8000/global/package")
zip_file = zipfile.ZipFile(io.BytesIO(response.content))

# Extract all files
zip_file.extractall("./")

# Files extracted: main_model_v2.pkl and global_scaler.pkl
```

#### Option 2: Download Files Separately
```python
import requests

# Download global model
response = requests.get("http://127.0.0.1:8000/global/model")
with open("main_model.pkl", "wb") as f:
    f.write(response.content)

# Download global scaler
response = requests.get("http://127.0.0.1:8000/global/scaler")
with open("global_scaler.pkl", "wb") as f:
    f.write(response.content)
```

## Development

- FastAPI supports hot reload when running with `python main.py`
- Streamlit automatically reloads on file changes
- API documentation is auto-generated at `/docs`

## Federated Learning: Hospital Node Setup

### How Hospital Nodes Should Expose Their Models

Each hospital node must implement a `/model/download` endpoint that returns their trained model file. The Central Server will retrieve models from this endpoint during the federated aggregation process.

**Required Endpoint:** `GET /model/download`

**Example Hospital Node Implementation:**

See [hospital_node_example.py](hospital_node_example.py) for a complete working example.

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Hospital Node API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/model/download")
async def download_model():
    """
    Central Server retrieves the model by calling this endpoint.
    FileResponse automatically converts the pickle file to bytes
    and safely transmits it over HTTP.
    """
    model_file = "hospital_1_v2.pkl"
    
    if not os.path.exists(model_file):
        raise HTTPException(status_code=404, detail="Model not found")
    
    return FileResponse(
        path=model_file,
        media_type="application/octet-stream",
        filename=model_file
    )
```

**Running a Hospital Node:**
```powershell
# Hospital 1 on port 8002
uvicorn hospital_node_example:app --port 8002

# Hospital 2 on port 8003  
uvicorn hospital_node_example:app --port 8003
```

### How the Central Server Retrieves Models

The Central Server dashboard makes simple HTTP GET requests to retrieve hospital models:

```python
import requests

# Retrieve Hospital 1's trained model
response = requests.get("http://127.0.0.1:8002/model/download")

if response.status_code == 200:
    # Save the model to Central Server's drive
    with open("hospital_1_v2.pkl", "wb") as f:
        f.write(response.content)
    print("Successfully retrieved Hospital 1's model!")
```

**Key Points:**
- FastAPI's `FileResponse` automatically handles binary file transmission
- No need to manually encode/decode - it's handled automatically
- The Central Server receives the exact same pickle file that was saved on the hospital's system
- Models are stored in the `models/` directory on the Central Server

### Federated Learning Workflow

1. **Initial Distribution**: Hospital nodes download global model and scaler from Central Server
   - Use: `GET /global/package` or `GET /global/model` + `GET /global/scaler`

2. **Local Training**: Each hospital trains the model on their private local data
   - Data never leaves the hospital premises
   - Only model weights are shared

3. **Model Exposure**: Hospital nodes expose their trained models via `/model/download`
   - Hospital 1: `http://127.0.0.1:8002/model/download`
   - Hospital 2: `http://127.0.0.1:8003/model/download`

4. **Aggregation**: Central Server retrieves and aggregates all models using FedAvg
   - Use the Streamlit dashboard at `http://localhost:8501`
   - Click "Retrieve Model" buttons for each hospital
   - Click "Perform FedAvg Aggregation" to combine models

5. **Update Global Model**: Aggregated model becomes new global model
   - Saved as `main_model_v2.pkl`
   - Ready for next iteration

## Next Steps

1. Add your ML models to the `models/` directory
2. Implement prediction logic in `main.py`
3. Customize the Streamlit dashboard in `main_dashboard.py`
4. Configure environment variables in `.env`
5. Add authentication if needed
6. Deploy to production server

## Deploy On Railway

This repository contains two runnable apps:
- `FastAPI API` (`main.py`)
- `Streamlit Dashboard` (`main_dashboard.py`)

Railway runs one start command per service, so create **two Railway services** from the same repo.

### Service 1: FastAPI API

1. Create a new service from this repo.
2. Set the start command to:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

3. Optional environment variables:
- `API_TITLE`
- `API_VERSION`
- `MODEL_PATH` (default: `models/`)
- `LOG_LEVEL`

### Service 2: Streamlit Dashboard

1. Create another service from the same repo.
2. Set the start command to:

```bash
streamlit run main_dashboard.py --server.address 0.0.0.0 --server.port $PORT
```

3. Set environment variables:
- `CENTRAL_API_URL=https://<your-fastapi-service>.up.railway.app`
- `HOSPITAL_1_URL=https://<hospital-1-service-url>` (optional)
- `HOSPITAL_2_URL=https://<hospital-2-service-url>` (optional)

### Important Notes

- The API now supports Railway's `PORT` automatically.
- The dashboard reset action uses `CENTRAL_API_URL`, so it works across Railway services.
- Keep `models/` writable at runtime if you plan to persist downloaded/aggregated model files.
