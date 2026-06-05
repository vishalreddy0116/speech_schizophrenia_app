import os

# Project Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
STATIC_DIR = os.path.join(BASE_DIR, "frontend")

# Create directories if they do not exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Database Configurations
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "schizophrenia_estimation"
SQLITE_PATH = os.path.join(BASE_DIR, "sqlite_fallback.db")

# Model Settings
WAVLM_MODEL_NAME = "microsoft/wavlm-base"
VQVAE_NUM_EMBEDDINGS = 512
VQVAE_EMBEDDING_DIM = 128
VQVAE_INPUT_DIM = 4

# Server Settings
HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", 8000))
