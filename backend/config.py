# FastAPI Backend Configuration
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in backend directory
backend_dir = Path(__file__).parent
env_path = backend_dir / ".env"
load_dotenv(dotenv_path=env_path)

# HuggingFace Model
MODEL_NAME = "tzaware/codet5p-spider-finetuned"
MODEL_MAX_LENGTH = 512
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
USE_HF_INFERENCE_API = True
CUSTOM_MODEL_API_URL = os.getenv("CUSTOM_MODEL_API_URL", "")

# FAISS Vector Store
FAISS_INDEX_PATH = "vector_service/output/faiss.index"
METADATA_PATH = "vector_service/output/metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval Settings
TOP_K_RETRIEVAL = 5

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
DEBUG = True

# CORS Settings
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5000",
]
