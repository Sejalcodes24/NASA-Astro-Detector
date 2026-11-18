import sys
import math
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app
from config import PROCESSED_DATASET_FILE, BEST_MODEL_FILE
from models.predictor import predictor
from orbit.propagator import solve_kepler, propagate_asteroid_and_earth, EARTH_ELEMENTS

client = TestClient(app)

def test_processed_dataset_integrity():
    """Verify processed dataset exists and matches expected scientific schema."""
    assert PROCESSED_DATASET_FILE.exists(), "Processed dataset must exist"
    import pandas as pd
    df = pd.read_csv(PROCESSED_DATASET_FILE)
    assert len(df) > 5000, f"Expected > 5000 records, got {len(df)}"
    
    # Required columns
    expected_cols = [
        "spkid", "full_name", "pdes", "pha", "H", "diameter_estimated_km",
        "a", "e", "i", "moid", "is_hazardous", "relative_velocity_kms",
        "is_sentry_monitored", "has_cad_approach"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"
        
    # Check target integrity
    assert set(df["is_hazardous"].unique()).issubset({0, 1})
    pha_count = (df["is_hazardous"] == 1).sum()
    assert pha_count > 1000, "Dataset must include a substantial PHA population"

def test_model_predictor_inference():
    """Verify ML predictor loads, evaluates probabilities, and provides explainability."""
    assert predictor.model is not None, "Model must be loaded"
    
    # 1. Hazardous test case (Critical MOID and large size)
    hazard_input = {
        "H": 18.0,
        "diameter": 0.8,
        "a": 1.1,
        "e": 0.35,
        "i": 5.0,
        "moid": 0.015,
        "orbit_class": "APO"
    }
    res_haz = predictor.predict(hazard_input)
    assert res_haz["model_prediction"]["is_hazardous"] is True
    assert res_haz["model_prediction"]["hazard_probability"] > 0.80
    assert res_haz["authoritative_status"]["nasa_pha_criteria_met"] is True
    assert len(res_haz["explanations"]) > 0
    assert "uncertainty_level" in res_haz["model_prediction"]
    
    # 2. Non-hazardous test case (Safe MOID)
    nominal_input = {
        "H": 25.0,
        "diameter": 0.04,
        "a": 1.6,
        "e": 0.15,
        "i": 18.0,
        "moid": 0.28,
        "orbit_class": "AMO"
    }
    res_nom = predictor.predict(nominal_input)
    assert res_nom["model_prediction"]["is_hazardous"] is False
    assert res_nom["model_prediction"]["hazard_probability"] < 0.35
    assert res_nom["authoritative_status"]["nasa_pha_criteria_met"] is False

def test_orbit_propagator():
    """Verify Kepler equation solver and 3D trajectory calculation."""
    # Test Kepler solver
    M = math.radians(45.0)
    e = 0.2
    E = solve_kepler(M, e)
    # Check M = E - e*sin(E)
    assert abs(E - e * math.sin(E) - M) < 1e-6
    
    # Test 3D trajectory propagation
    elements = {
        "a": 1.2, "e": 0.3, "i": 8.0, "om": 45.0, "w": 30.0, "ma": 10.0, "epoch_mjd": 59000.0
    }
    traj = propagate_asteroid_and_earth(elements, days_window=60, step_days=5)
    assert len(traj["asteroid_orbit_path"]) > 50
    assert len(traj["earth_orbit_path"]) > 50
    assert len(traj["time_series"]) > 5
    assert traj["closest_approach_in_window"] is not None
    assert traj["closest_approach_in_window"]["min_distance_au"] > 0

def test_api_status_endpoint():
    """Verify /api/status returns operational status and telemetry."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert data["data_status"]["total_records"] > 5000

def test_api_metrics_summary():
    """Verify /api/metrics/summary returns high-level counts and critical alerts."""
    response = client.get("/api/metrics/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_monitored"] > 5000
    assert data["pha_count"] > 1000
    assert len(data["alert_objects"]) > 0

def test_api_asteroids_list():
    """Verify /api/asteroids returns filterable and searchable catalog."""
    # Test basic pagination
    response = client.get("/api/asteroids?page=1&limit=10")
    assert response.status_code == 200
    data = response.json()
    assert len(data["records"]) == 10
    assert data["total"] > 5000
    
    # Test PHA filter
    res_pha = client.get("/api/asteroids?is_pha=true&limit=5")
    assert res_pha.status_code == 200
    for r in res_pha.json()["records"]:
        assert r["is_hazardous"] is True
        
    # Test search
    res_search = client.get("/api/asteroids?search=Apophis")
    assert res_search.status_code == 200
    records = res_search.json()["records"]
    assert any("Apophis" in r["full_name"] for r in records)

def test_api_asteroid_detail():
    """Verify /api/asteroids/{id} returns complete physical, Sentry, CAD, and ML fields."""
    response = client.get("/api/asteroids/99942")
    assert response.status_code == 200
    data = response.json()
    assert "asteroid_info" in data
    assert "ml_analysis" in data
    assert "Apophis" in data["asteroid_info"]["full_name"]
    assert data["ml_analysis"]["model_prediction"]["hazard_probability"] is not None

def test_api_predict():
    """Verify /api/predict POST endpoint validates inputs and returns predictions."""
    payload = {
        "H": 19.5,
        "diameter": 0.35,
        "a": 1.05,
        "e": 0.25,
        "i": 7.0,
        "moid": 0.025,
        "orbit_class": "APO"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "model_prediction" in data
    assert "explanations" in data
    assert "plain_language_summary" in data

def test_api_orbit_propagation():
    """Verify /api/orbit/propagate POST endpoint returns 3D trajectories."""
    payload = {
        "identifier": "99942",
        "days_window": 90,
        "step_days": 5
    }
    response = client.post("/api/orbit/propagate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "asteroid_orbit_path" in data
    assert "earth_orbit_path" in data
    assert "time_series" in data

def test_api_eda_and_models():
    """Verify /api/eda/stats and /api/models/comparison endpoints."""
    res_eda = client.get("/api/eda/stats")
    assert res_eda.status_code == 200
    eda_data = res_eda.json()
    assert "histograms" in eda_data
    assert "class_balance" in eda_data
    
    res_models = client.get("/api/models/comparison")
    assert res_models.status_code == 200
    models_data = res_models.json()
    assert "champion_model" in models_data
    assert "models" in models_data
    assert "Gradient_Boosting" in models_data["models"]
