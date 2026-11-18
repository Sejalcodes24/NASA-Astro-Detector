import os          #http://127.0.0.1:8000
import sys
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query, HTTPException, Request, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    STATIC_DIR, TEMPLATES_DIR, PROCESSED_DATASET_FILE,
    DATASET_METADATA_FILE, EDA_STATS_FILE, MODEL_COMPARISON_FILE,
    FEATURE_IMPORTANCE_FILE, EVALUATION_METRICS_FILE, BEST_MODEL_FILE,
    RAW_SBDB_FILE, RAW_SENTRY_FILE, RAW_CAD_FILE,
    APP_HOST, APP_PORT
)
from data.data_pipeline import run_data_pipeline
from models.train import train_and_evaluate_models
from models.predictor import predictor
from orbit.propagator import propagate_asteroid_and_earth

# Initialize FastAPI App
app = FastAPI(
    title="Asteroid Trajectory & Earth Risk Analysis API",
    description="Scientific near-Earth asteroid monitoring, orbital dynamics visualization, and machine learning hazard classification system.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files & Templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Cached DataFrame in memory for high-speed search and filtering
asteroids_df: Optional[pd.DataFrame] = None
data_status = {
    "is_live": False,
    "last_sync": None,
    "source": "Local NASA/JPL SSD Snapshot",
    "total_records": 0,
    "pha_count": 0,
    "sentry_count": 0,
    "cad_count": 0
}

def load_data_into_memory():
    """Load processed dataset and update status."""
    global asteroids_df, data_status
    if PROCESSED_DATASET_FILE.exists():
        asteroids_df = pd.read_csv(PROCESSED_DATASET_FILE)
        # Fill NaNs for JSON safety
        total = len(asteroids_df)
        pha = int((asteroids_df["is_hazardous"] == 1).sum()) if "is_hazardous" in asteroids_df.columns else 0
        sentry = int((asteroids_df["is_sentry_monitored"] == 1).sum()) if "is_sentry_monitored" in asteroids_df.columns else 0
        cad = int((asteroids_df["has_cad_approach"] == 1).sum()) if "has_cad_approach" in asteroids_df.columns else 0
        
        last_mtime = datetime.fromtimestamp(PROCESSED_DATASET_FILE.stat().st_mtime, timezone.utc).isoformat()
        data_status.update({
            "is_live": True,
            "last_sync": last_mtime,
            "total_records": total,
            "pha_count": pha,
            "sentry_count": sentry,
            "cad_count": cad
        })
        print(f"[APP] Loaded {total} asteroids into memory. (PHAs: {pha}, Sentry: {sentry}, CAD: {cad})")
    else:
        print("[APP] Processed dataset not found! Running data pipeline...")
        run_data_pipeline(force_download=False)
        load_data_into_memory()

# Load data on module startup
load_data_into_memory()

# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------
class AsteroidPredictionRequest(BaseModel):
    H: float = Field(..., ge=5.0, le=35.0, description="Absolute magnitude (mag)")
    diameter: Optional[float] = Field(None, ge=0.001, le=100.0, description="Estimated or measured diameter (km)")
    a: float = Field(..., ge=0.1, le=10.0, description="Semi-major axis (AU)")
    e: float = Field(..., ge=0.0, lt=1.0, description="Orbital eccentricity")
    i: float = Field(..., ge=0.0, le=180.0, description="Inclination (deg)")
    moid: float = Field(..., ge=0.0, le=5.0, description="Earth Minimum Orbit Intersection Distance (AU)")
    q: Optional[float] = Field(None, ge=0.0, le=10.0, description="Perihelion distance (AU)")
    ad: Optional[float] = Field(None, ge=0.0, le=20.0, description="Aphelion distance (AU)")
    per_y: Optional[float] = Field(None, ge=0.05, le=100.0, description="Orbital period (years)")
    orbit_class: Optional[str] = Field("APO", description="Orbit class (AMO, APO, ATE, IEO)")

class OrbitPropagationRequest(BaseModel):
    identifier: Optional[str] = Field(None, description="Asteroid SPK ID or designation")
    a: Optional[float] = Field(None, ge=0.2, le=10.0)
    e: Optional[float] = Field(None, ge=0.0, lt=1.0)
    i: Optional[float] = Field(None, ge=0.0, le=180.0)
    om: Optional[float] = Field(0.0, ge=0.0, le=360.0)
    w: Optional[float] = Field(0.0, ge=0.0, le=360.0)
    ma: Optional[float] = Field(0.0, ge=0.0, le=360.0)
    epoch_mjd: Optional[float] = Field(59000.0)
    days_window: Optional[int] = Field(180, ge=10, le=730)
    step_days: Optional[int] = Field(2, ge=1, le=15)

# ---------------------------------------------------------
# Web Routes
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Render main dashboard view."""
    return templates.TemplateResponse("index.html", {"request": request, "status": data_status})

# ---------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------
@app.get("/api/status")
async def get_system_status():
    """Return backend status, live vs cached data telemetry, and source info."""
    return {
        "status": "operational",
        "data_status": data_status,
        "models_status": {
            "model_loaded": predictor.model is not None,
            "model_file_exists": BEST_MODEL_FILE.exists()
        },
        "sources": [
            {"name": "NASA/JPL SBDB", "status": "active"},
            {"name": "NASA/JPL Sentry", "status": "active"},
            {"name": "NASA/JPL CAD", "status": "active"}
        ],
        "server_time": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/refresh-data")
async def refresh_dataset(background_tasks: BackgroundTasks):
    """Trigger background re-fetch and model update."""
    def run_refresh():
        try:
            print("[SYNC] Refreshing live NASA data...")
            run_data_pipeline(force_download=True)
            print("[SYNC] Retraining candidate models...")
            train_and_evaluate_models()
            predictor.load_artifacts()
            load_data_into_memory()
            print("[SYNC] Refresh complete!")
        except Exception as e:
            print(f"[SYNC ERROR] {e}")

    background_tasks.add_task(run_refresh)
    return {"message": "Data refresh initiated in background. Refreshing NASA JPL feeds."}

@app.get("/api/metrics/summary")
async def get_summary_metrics():
    """Return high-level operational metrics and critical risk alerts."""
    global asteroids_df
    if asteroids_df is None or len(asteroids_df) == 0:
        load_data_into_memory()
        
    df = asteroids_df
    total = len(df)
    pha_count = int((df["is_hazardous"] == 1).sum())
    sentry_count = int((df["is_sentry_monitored"] == 1).sum())
    cad_count = int((df["has_cad_approach"] == 1).sum())
    
    avg_velocity = round(float(df["relative_velocity_kms"].mean()), 2)
    min_moid = round(float(df["moid"].min()), 5)
    max_diameter = round(float(df["diameter_estimated_km"].max()), 2)
    
    # Critical alerts: Objects on Sentry or with immediate close approach (< 0.05 AU)
    alert_mask = (df["is_sentry_monitored"] == 1) | ((df["has_cad_approach"] == 1) & (df["cad_dist_au"] <= 0.05))
    alerts_subset = df[alert_mask].sort_values(by=["sentry_ps_max", "moid"], ascending=[False, True]).head(10)
    
    alerts_list = []
    for _, row in alerts_subset.iterrows():
        is_sentry = bool(row["is_sentry_monitored"] == 1)
        alerts_list.append({
            "spkid": int(row["spkid"]),
            "full_name": str(row["full_name"]),
            "pdes": str(row["pdes"]),
            "is_hazardous": bool(row["is_hazardous"]),
            "moid": round(float(row["moid"]), 4),
            "diameter_km": round(float(row["diameter_estimated_km"]), 3),
            "is_sentry": is_sentry,
            "sentry_ps_max": float(row["sentry_ps_max"]) if is_sentry and pd.notna(row["sentry_ps_max"]) else None,
            "sentry_ts_max": int(row["sentry_ts_max"]) if is_sentry and pd.notna(row["sentry_ts_max"]) else 0,
            "sentry_ip": float(row["sentry_ip"]) if is_sentry and pd.notna(row["sentry_ip"]) else 0.0,
            "sentry_range": str(row["sentry_range"]) if is_sentry and pd.notna(row["sentry_range"]) else "",
            "has_cad": bool(row["has_cad_approach"] == 1),
            "cad_date": str(row["cad_next_date"]) if pd.notna(row["cad_next_date"]) else "",
            "cad_dist_au": round(float(row["cad_dist_au"]), 4) if pd.notna(row["cad_dist_au"]) else None,
            "cad_dist_ld": round(float(row["cad_dist_ld"]), 2) if pd.notna(row["cad_dist_ld"]) else None
        })
        
    return {
        "total_monitored": total,
        "pha_count": pha_count,
        "pha_percentage": round((pha_count / total) * 100, 2),
        "sentry_monitored_count": sentry_count,
        "upcoming_close_approaches": cad_count,
        "avg_encounter_velocity_kms": avg_velocity,
        "minimum_moid_au": min_moid,
        "maximum_diameter_km": max_diameter,
        "alert_objects": alerts_list
    }

@app.get("/api/asteroids")
async def list_asteroids(
    search: Optional[str] = Query(None, description="Search by designation, name, or SPK ID"),
    orbit_class: Optional[str] = Query(None, description="Filter by class: AMO, APO, ATE, IEO"),
    is_pha: Optional[bool] = Query(None, description="Filter by PHA status"),
    sentry_only: Optional[bool] = Query(False, description="Filter Sentry monitored"),
    cad_only: Optional[bool] = Query(False, description="Filter Close Approaches"),
    max_moid: Optional[float] = Query(None, description="Max Earth MOID in AU"),
    min_diameter: Optional[float] = Query(None, description="Min diameter in km"),
    page: int = Query(1, ge=1),
    limit: int = Query(25, ge=5, le=100),
    sort_by: str = Query("moid", description="Field to sort by: moid, H, diameter_estimated_km, relative_velocity_kms, full_name"),
    order: str = Query("asc", description="Sort order: asc or desc")
):
    """Search, filter, and paginate through the asteroid catalog."""
    global asteroids_df
    if asteroids_df is None:
        load_data_into_memory()
        
    df = asteroids_df.copy()
    
    # 1. Search text
    if search:
        s = search.strip().lower()
        df = df[
            df["full_name"].str.lower().str.contains(s, na=False) |
            df["pdes"].str.lower().str.contains(s, na=False) |
            df["spkid"].astype(str).str.contains(s, na=False)
        ]
        
    # 2. Filters
    if orbit_class and orbit_class != "ALL":
        df = df[df["class"] == orbit_class.upper()]
    if is_pha is not None:
        df = df[df["is_hazardous"] == (1 if is_pha else 0)]
    if sentry_only:
        df = df[df["is_sentry_monitored"] == 1]
    if cad_only:
        df = df[df["has_cad_approach"] == 1]
    if max_moid is not None:
        df = df[df["moid"] <= max_moid]
    if min_diameter is not None:
        df = df[df["diameter_estimated_km"] >= min_diameter]
        
    total_matches = len(df)
    
    # 3. Sort
    ascending = (order.lower() == "asc")
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)
    else:
        df = df.sort_values(by="moid", ascending=True)
        
    # 4. Pagination
    start_idx = (page - 1) * limit
    page_df = df.iloc[start_idx : start_idx + limit]
    
    # Format records
    records = []
    for _, r in page_df.iterrows():
        records.append({
            "spkid": int(r["spkid"]),
            "full_name": str(r["full_name"]),
            "pdes": str(r["pdes"]),
            "class": str(r["class"]),
            "H": round(float(r["H"]), 2),
            "diameter_km": round(float(r["diameter_estimated_km"]), 3),
            "diameter_is_measured": bool(r["diameter_is_measured"]),
            "a": round(float(r["a"]), 4),
            "e": round(float(r["e"]), 4),
            "i": round(float(r["i"]), 2),
            "moid": round(float(r["moid"]), 4),
            "moid_ld": round(float(r["moid"]) * 389.1703, 1),
            "is_hazardous": bool(r["is_hazardous"]),
            "relative_velocity_kms": round(float(r["relative_velocity_kms"]), 1),
            "is_sentry": bool(r["is_sentry_monitored"]),
            "sentry_ps_max": float(r["sentry_ps_max"]) if pd.notna(r["sentry_ps_max"]) else None,
            "has_cad": bool(r["has_cad_approach"]),
            "cad_date": str(r["cad_next_date"]) if pd.notna(r["cad_next_date"]) else "",
            "cad_dist_ld": round(float(r["cad_dist_ld"]), 2) if pd.notna(r["cad_dist_ld"]) else None
        })
        
    return {
        "page": page,
        "limit": limit,
        "total": total_matches,
        "total_pages": math.ceil(total_matches / limit) if limit > 0 else 1,
        "records": records
    }

@app.get("/api/asteroids/{identifier}")
async def get_asteroid_detail(identifier: str):
    """Retrieve full physical, orbital, Sentry, CAD, and ML inference details for an asteroid."""
    global asteroids_df
    if asteroids_df is None:
        load_data_into_memory()
        
    df = asteroids_df
    id_clean = identifier.strip().lower()
    
    # Lookup by spkid, pdes, or full_name
    match = df[
        (df["spkid"].astype(str) == id_clean) |
        (df["pdes"].str.lower() == id_clean) |
        (df["full_name"].str.lower() == id_clean)
    ]
    
    if match.empty:
        # Partial match fallback
        match = df[df["full_name"].str.lower().str.contains(id_clean, na=False)]
        
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Asteroid '{identifier}' not found in database.")
        
    row = match.iloc[0].to_dict()
    
    # Run model prediction on this asteroid's actual parameters
    pred_result = predictor.predict(row)
    
    # Clean dictionary for JSON response
    cleaned_row = {}
    for k, v in row.items():
        if isinstance(v, (int, np.integer)):
            cleaned_row[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            cleaned_row[k] = None if (math.isnan(v) or math.isinf(v)) else round(float(v), 5)
        else:
            cleaned_row[k] = str(v)
            
    return {
        "asteroid_info": cleaned_row,
        "ml_analysis": pred_result,
        "orbital_kepler_elements": {
            "a": cleaned_row.get("a"),
            "e": cleaned_row.get("e"),
            "i": cleaned_row.get("i"),
            "om": cleaned_row.get("om"),
            "w": cleaned_row.get("w"),
            "ma": cleaned_row.get("ma"),
            "epoch_mjd": cleaned_row.get("epoch_mjd")
        }
    }

@app.post("/api/predict")
async def predict_custom_asteroid(req: AsteroidPredictionRequest):
    """Run model prediction and explainability on custom asteroid parameters."""
    try:
        input_dict = req.model_dump()
        result = predictor.predict(input_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/api/orbit/propagate")
async def get_orbital_propagation(req: OrbitPropagationRequest):
    """Propagate 3D orbital trajectory for Earth and Asteroid over a selectable time window."""
    global asteroids_df
    if asteroids_df is None:
        load_data_into_memory()
        
    elements = {}
    
    # If identifier provided, lookup from dataset
    if req.identifier:
        id_clean = req.identifier.strip().lower()
        match = asteroids_df[
            (asteroids_df["spkid"].astype(str) == id_clean) |
            (asteroids_df["pdes"].str.lower() == id_clean) |
            (asteroids_df["full_name"].str.lower() == id_clean)
        ]
        if not match.empty:
            r = match.iloc[0]
            elements = {
                "a": float(r["a"]),
                "e": float(r["e"]),
                "i": float(r["i"]) if pd.notna(r["i"]) else 0.0,
                "om": float(r["om"]) if pd.notna(r["om"]) else 0.0,
                "w": float(r["w"]) if pd.notna(r["w"]) else 0.0,
                "ma": float(r["ma"]) if pd.notna(r["ma"]) else 0.0,
                "epoch_mjd": float(r["epoch_mjd"]) if pd.notna(r["epoch_mjd"]) else 59000.0,
                "name": str(r["full_name"])
            }
            
    # Override with explicit parameters if supplied
    if req.a is not None:
        elements["a"] = req.a
    if req.e is not None:
        elements["e"] = req.e
    if req.i is not None:
        elements["i"] = req.i
    if req.om is not None:
        elements["om"] = req.om
    if req.w is not None:
        elements["w"] = req.w
    if req.ma is not None:
        elements["ma"] = req.ma
    if req.epoch_mjd is not None:
        elements["epoch_mjd"] = req.epoch_mjd
        
    if "a" not in elements or "e" not in elements:
        # Fallback to default (Apophis-like NEA)
        elements = {
            "a": 0.9224, "e": 0.1912, "i": 3.33, "om": 204.04, "w": 126.40, "ma": 134.7, "epoch_mjd": 59000.0,
            "name": "99942 Apophis (Default NEA Orbit)"
        }
        
    trajectory_data = propagate_asteroid_and_earth(
        elements, 
        days_window=req.days_window or 180, 
        step_days=req.step_days or 2
    )
    trajectory_data["asteroid_name"] = elements.get("name", "Target Asteroid")
    return trajectory_data

@app.get("/api/eda/stats")
async def get_eda_statistics():
    """Return precomputed EDA stats and distributions."""
    if not EDA_STATS_FILE.exists():
        run_data_pipeline(force_download=False)
    with open(EDA_STATS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/models/comparison")
async def get_model_benchmarks():
    """Return benchmark comparison table, metrics, ROC and PR curves."""
    if not MODEL_COMPARISON_FILE.exists():
        train_and_evaluate_models()
    with open(MODEL_COMPARISON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/models/feature-importance")
async def get_feature_importances():
    """Return global feature importances and domain descriptions."""
    if not FEATURE_IMPORTANCE_FILE.exists():
        train_and_evaluate_models()
    with open(FEATURE_IMPORTANCE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/metadata")
async def get_metadata():
    """Return data source documentation and scientific limitations."""
    if not DATASET_METADATA_FILE.exists():
        run_data_pipeline(force_download=False)
    with open(DATASET_METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=APP_HOST, port=APP_PORT, reload=False)
