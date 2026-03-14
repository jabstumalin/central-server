import os

HOSPITAL_1_URL = os.getenv("HOSPITAL_1_URL", "http://localhost:8001")
HOSPITAL_2_URL = os.getenv("HOSPITAL_2_URL", "http://localhost:8002")
CENTRAL_API_URL = os.getenv("CENTRAL_API_URL", "http://localhost:8000")
MODELS_DIR = os.getenv("MODELS_DIR", "models")
