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
streamlit run streamlit_app.py
```
The dashboard will be available at: http://localhost:8501

### Run Both (in separate terminals)
```powershell
# Terminal 1 - FastAPI
python main.py

# Terminal 2 - Streamlit
streamlit run streamlit_app.py
```

## Project Structure

```
central-server/
├── main.py              # FastAPI application
├── streamlit_app.py     # Streamlit dashboard
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
├── .env                 # Your local environment variables (create this)
└── models/              # ML models directory (create as needed)
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

## Next Steps

1. Add your ML models to the `models/` directory
2. Implement prediction logic in `main.py`
3. Customize the Streamlit dashboard in `streamlit_app.py`
4. Configure environment variables in `.env`
5. Add authentication if needed
6. Deploy to production server
