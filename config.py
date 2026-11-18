import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# NASA JPL API Endpoints
JPL_SBDB_QUERY_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
JPL_SENTRY_URL = "https://ssd-api.jpl.nasa.gov/sentry.api"
JPL_CAD_URL = "https://ssd-api.jpl.nasa.gov/cad.api"
JPL_SBDB_API_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"

# File paths
RAW_SBDB_FILE = RAW_DATA_DIR / "sbdb_neas.json"
RAW_SENTRY_FILE = RAW_DATA_DIR / "sentry_objects.json"
RAW_CAD_FILE = RAW_DATA_DIR / "close_approaches.json"

PROCESSED_DATASET_FILE = PROCESSED_DATA_DIR / "asteroids_processed.csv"
DATASET_METADATA_FILE = PROCESSED_DATA_DIR / "dataset_metadata.json"
EDA_STATS_FILE = PROCESSED_DATA_DIR / "eda_stats.json"

BEST_MODEL_FILE = SAVED_MODELS_DIR / "best_model.joblib"
MODEL_COMPARISON_FILE = SAVED_MODELS_DIR / "model_comparison.json"
FEATURE_IMPORTANCE_FILE = SAVED_MODELS_DIR / "feature_importance.json"
EVALUATION_METRICS_FILE = SAVED_MODELS_DIR / "evaluation_metrics.json"

# Training configuration
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.15

# Default App Configuration
APP_HOST = "127.0.0.1"
APP_PORT = 8000
