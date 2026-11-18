import os
import sys
import json
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    RAW_SBDB_FILE, RAW_SENTRY_FILE, RAW_CAD_FILE,
    PROCESSED_DATASET_FILE, DATASET_METADATA_FILE, EDA_STATS_FILE,
    JPL_SBDB_QUERY_URL, JPL_SENTRY_URL, JPL_CAD_URL
)
from data.fetch_raw import collect_raw_data

def clean_designation(des_str):
    """Normalize designation strings for matching across catalogs."""
    if not des_str:
        return ""
    s = str(des_str).strip()
    # Strip parentheses and excessive whitespace
    s = s.replace("(", "").replace(")", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def calculate_photometric_diameter(h_mag, albedo=0.14):
    """
    Standard NASA/IAU photometric diameter estimation:
    D = (1329 / sqrt(albedo)) * 10^(-0.2 * H) [km]
    Default geometric albedo for NEAs is 0.14 when unknown.
    """
    if pd.isna(h_mag):
        return np.nan
    alb = albedo if (pd.notna(albedo) and albedo > 0) else 0.14
    return (1329.0 / math.sqrt(alb)) * (10.0 ** (-0.2 * float(h_mag)))

def calculate_kinetic_energy_mt(diameter_km, velocity_kms=20.0, density_kg_m3=2500.0):
    """
    Approximate impact kinetic energy in Megatons of TNT.
    1 MT TNT = 4.184e15 Joules.
    """
    if pd.isna(diameter_km) or diameter_km <= 0:
        return 0.0
    radius_m = (diameter_km * 1000.0) / 2.0
    volume_m3 = (4.0 / 3.0) * math.pi * (radius_m ** 3)
    mass_kg = volume_m3 * density_kg_m3
    vel_ms = (velocity_kms if pd.notna(velocity_kms) and velocity_kms > 0 else 20.0) * 1000.0
    energy_joules = 0.5 * mass_kg * (vel_ms ** 2)
    return energy_joules / 4.184e15

def calculate_opik_encounter_velocity(a, e, i_deg):
    """
    Öpik encounter velocity relative to Earth (in km/s).
    v_rel / v_earth = sqrt(3 - 1/a - 2*sqrt(a*(1-e^2))*cos(i))
    v_earth ~ 29.78 km/s.
    """
    try:
        if a <= 0:
            return 20.0
        i_rad = math.radians(i_deg)
        term = 3.0 - (1.0 / a) - 2.0 * math.sqrt(max(0.0, a * (1.0 - e**2))) * math.cos(i_rad)
        if term <= 0:
            return 12.0
        return math.sqrt(term) * 29.78
    except Exception:
        return 20.0

def run_data_pipeline(force_download=False):
    """Run end-to-end data processing and feature engineering pipeline."""
    print("=" * 60)
    print("STARTING DATA PIPELINE: NASA JPL NEAS & RISK DATA")
    print("=" * 60)
    
    # 1. Collect / Load Raw Data
    sbdb_data, sentry_data, cad_data = collect_raw_data(force_download=force_download)
    
    # 2. Convert SBDB rows to DataFrame
    fields = sbdb_data.get("fields", [])
    data_rows = sbdb_data.get("data", [])
    df = pd.DataFrame(data_rows, columns=fields)
    print(f"[PIPELINE] Initial SBDB rows: {len(df)}")
    
    # Schema check and duplicates removal
    df = df.drop_duplicates(subset=["spkid"])
    df = df.drop_duplicates(subset=["pdes"])
    print(f"[PIPELINE] Rows after deduplication: {len(df)}")
    
    # Strip string fields
    for col in ["full_name", "pdes", "pha", "neo", "class"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    # Target definition: Authoritative PHA status from NASA JPL
    df["is_hazardous"] = (df["pha"] == "Y").astype(int)
    
    # Convert numerical columns
    num_cols = ["H", "diameter", "albedo", "rot_per", "a", "e", "i", "q", "ad", "per_y", "moid", "moid_jup", "t_jup", "epoch_mjd", "om", "w", "ma"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # Remove records missing crucial orbital parameters (a, e, i, moid, H)
    initial_count = len(df)
    df = df.dropna(subset=["a", "e", "i", "moid", "H"]).copy()
    print(f"[PIPELINE] Dropped {initial_count - len(df)} records with invalid essential orbital elements.")
    
    # Analyze and treat missing values
    missing_summary = {}
    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        missing_summary[col] = {
            "missing_count": n_miss,
            "missing_pct": round((n_miss / len(df)) * 100, 2)
        }
    print("[PIPELINE] Missing value summary:", {k: f"{v['missing_pct']}%" for k, v in missing_summary.items() if v['missing_count'] > 0})
    
    # Physical diameter imputation via photometric equation
    df["diameter_is_measured"] = (~df["diameter"].isna()).astype(int)
    estimated_diameters = [
        calculate_photometric_diameter(h, alb) 
        for h, alb in zip(df["H"], df["albedo"])
    ]
    df["diameter_estimated_km"] = np.where(df["diameter"].isna(), estimated_diameters, df["diameter"])
    
    # Approximate missing per_y using Kepler's 3rd Law: T = a^1.5
    df["per_y"] = np.where(df["per_y"].isna(), df["a"] ** 1.5, df["per_y"])
    
    # 3. Feature Engineering
    print("[PIPELINE] Performing feature engineering...")
    df["semi_minor_axis"] = df["a"] * np.sqrt(np.maximum(0.0, 1.0 - df["e"] ** 2))
    df["orbit_energy"] = -1.0 / (2.0 * df["a"])
    df["perihelion_ratio"] = df["q"] / df["a"]
    df["aphelion_ratio"] = df["ad"] / df["a"]
    df["eccentricity_sq"] = df["e"] ** 2
    df["inclination_rad"] = np.radians(df["i"])
    
    # Opik encounter velocity
    df["relative_velocity_kms"] = [
        calculate_opik_encounter_velocity(a, e, i) 
        for a, e, i in zip(df["a"], df["e"], df["i"])
    ]
    
    # Collision probability proxy factor
    df["collision_prob_factor"] = 1.0 / (
        (df["moid"] + 0.0001) * 
        np.sin(np.maximum(df["inclination_rad"], np.radians(0.5))) * 
        np.sqrt(np.maximum(0.001, df["a"] * (1.0 - df["e"]**2)))
    )
    # Clip extreme values for stability
    df["collision_prob_factor"] = np.clip(df["collision_prob_factor"], 0, 10000.0)
    
    # Impact energy
    df["kinetic_energy_mt"] = [
        calculate_kinetic_energy_mt(d, v) 
        for d, v in zip(df["diameter_estimated_km"], df["relative_velocity_kms"])
    ]
    
    # 4. Integrate NASA/JPL Sentry Impact Risk Data
    print("[PIPELINE] Integrating NASA JPL Sentry impact risk data...")
    sentry_lookup = {}
    if sentry_data and "data" in sentry_data:
        for s in sentry_data["data"]:
            des = clean_designation(s.get("des", ""))
            fullname = clean_designation(s.get("fullname", ""))
            entry = {
                "sentry_id": s.get("id"),
                "sentry_ps_max": float(s.get("ps_max")) if s.get("ps_max") is not None else -99.0,
                "sentry_ps_cum": float(s.get("ps_cum")) if s.get("ps_cum") is not None else -99.0,
                "sentry_ts_max": int(s.get("ts_max")) if s.get("ts_max") is not None else 0,
                "sentry_ip": float(s.get("ip")) if s.get("ip") is not None else 0.0,
                "sentry_n_imp": int(s.get("n_imp", 0)),
                "sentry_v_inf": float(s.get("v_inf")) if s.get("v_inf") is not None else 0.0,
                "sentry_range": str(s.get("range", "")),
                "sentry_last_obs": str(s.get("last_obs", ""))
            }
            if des:
                sentry_lookup[des.lower()] = entry
            if fullname:
                sentry_lookup[fullname.lower()] = entry
                
    df["is_sentry_monitored"] = 0
    df["sentry_ps_max"] = np.nan
    df["sentry_ps_cum"] = np.nan
    df["sentry_ts_max"] = 0
    df["sentry_ip"] = 0.0
    df["sentry_n_imp"] = 0
    df["sentry_v_inf"] = np.nan
    df["sentry_range"] = ""
    
    for idx, row in df.iterrows():
        pdes_clean = clean_designation(row["pdes"]).lower()
        full_clean = clean_designation(row["full_name"]).lower()
        match = sentry_lookup.get(pdes_clean) or sentry_lookup.get(full_clean)
        if match:
            df.at[idx, "is_sentry_monitored"] = 1
            df.at[idx, "sentry_ps_max"] = match["sentry_ps_max"]
            df.at[idx, "sentry_ps_cum"] = match["sentry_ps_cum"]
            df.at[idx, "sentry_ts_max"] = match["sentry_ts_max"]
            df.at[idx, "sentry_ip"] = match["sentry_ip"]
            df.at[idx, "sentry_n_imp"] = match["sentry_n_imp"]
            df.at[idx, "sentry_v_inf"] = match["sentry_v_inf"]
            df.at[idx, "sentry_range"] = match["sentry_range"]
            
    sentry_matches = int(df["is_sentry_monitored"].sum())
    print(f"[PIPELINE] Matched {sentry_matches} asteroids in dataset with authoritative JPL Sentry monitoring.")
    
    # 5. Integrate NASA/JPL CAD (Close Approach Data)
    print("[PIPELINE] Integrating NASA JPL Close Approach Data...")
    cad_lookup = {}
    if cad_data and "data" in cad_data:
        cad_fields = cad_data.get("fields", [])
        des_idx = cad_fields.index("des") if "des" in cad_fields else 0
        cd_idx = cad_fields.index("cd") if "cd" in cad_fields else 3
        dist_idx = cad_fields.index("dist") if "dist" in cad_fields else 4
        v_rel_idx = cad_fields.index("v_rel") if "v_rel" in cad_fields else 7
        
        for c_row in cad_data["data"]:
            des = clean_designation(c_row[des_idx]).lower()
            if des not in cad_lookup:
                try:
                    dist_au = float(c_row[dist_idx])
                    dist_ld = dist_au * 389.1703
                    v_rel = float(c_row[v_rel_idx])
                    cad_lookup[des] = {
                        "cad_next_date": c_row[cd_idx],
                        "cad_dist_au": dist_au,
                        "cad_dist_ld": round(dist_ld, 2),
                        "cad_v_rel": v_rel
                    }
                except Exception:
                    continue
                    
    df["has_cad_approach"] = 0
    df["cad_next_date"] = ""
    df["cad_dist_au"] = np.nan
    df["cad_dist_ld"] = np.nan
    df["cad_v_rel"] = np.nan
    
    for idx, row in df.iterrows():
        pdes_clean = clean_designation(row["pdes"]).lower()
        full_clean = clean_designation(row["full_name"]).lower()
        c_match = cad_lookup.get(pdes_clean) or cad_lookup.get(full_clean)
        if c_match:
            df.at[idx, "has_cad_approach"] = 1
            df.at[idx, "cad_next_date"] = c_match["cad_next_date"]
            df.at[idx, "cad_dist_au"] = c_match["cad_dist_au"]
            df.at[idx, "cad_dist_ld"] = c_match["cad_dist_ld"]
            df.at[idx, "cad_v_rel"] = c_match["cad_v_rel"]
            
    cad_matches = int(df["has_cad_approach"].sum())
    print(f"[PIPELINE] Matched {cad_matches} asteroids with upcoming/recent JPL close approaches.")
    
    # 6. Save Processed Dataset
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATASET_FILE, index=False)
    print(f"[SUCCESS] Processed dataset saved to {PROCESSED_DATASET_FILE} ({len(df)} rows, {len(df.columns)} columns)")
    
    # 7. Generate Comprehensive EDA & Stats
    generate_eda_and_metadata(df, missing_summary, sentry_matches, cad_matches)
    
    print("=" * 60)
    print("DATA PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return df

def generate_eda_and_metadata(df, missing_summary, sentry_matches, cad_matches):
    """Generate structured EDA statistics and dataset metadata for API & UI."""
    total_count = len(df)
    pha_count = int(df["is_hazardous"].sum())
    non_pha_count = total_count - pha_count
    
    # Class distribution
    orbit_classes = df["class"].value_counts().to_dict()
    
    # Numeric summaries
    stats_cols = [
        "H", "diameter_estimated_km", "a", "e", "i", "q", "ad", "per_y", 
        "moid", "relative_velocity_kms", "collision_prob_factor", "kinetic_energy_mt"
    ]
    numeric_stats = {}
    for col in stats_cols:
        if col in df.columns:
            s = df[col].dropna()
            numeric_stats[col] = {
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "q25": round(float(s.quantile(0.25)), 4),
                "median": round(float(s.median()), 4),
                "q75": round(float(s.quantile(0.75)), 4),
                "max": round(float(s.max()), 4)
            }
            
    # Correlations with target
    correlations = {}
    for col in stats_cols:
        if col in df.columns:
            r = df[col].corr(df["is_hazardous"])
            correlations[col] = round(float(r) if pd.notna(r) else 0.0, 4)
            
    # Histograms data for dashboard EDA charts
    histograms = {}
    for col, bins in [("H", 20), ("moid", 20), ("diameter_estimated_km", 20), ("relative_velocity_kms", 20), ("e", 20), ("i", 20)]:
        series = df[col].dropna()
        if col == "diameter_estimated_km":
            # Cap at 98th percentile for clean histogram visualization
            series = series[series <= series.quantile(0.98)]
        counts, bin_edges = np.histogram(series, bins=bins)
        bin_labels = [f"{round(bin_edges[i], 2)} - {round(bin_edges[i+1], 2)}" for i in range(len(counts))]
        histograms[col] = {
            "counts": [int(c) for c in counts],
            "bins": [round(float(b), 3) for b in bin_edges],
            "bin_labels": bin_labels
        }
        
    # Scatter sample for fast frontend EDA rendering (e.g. MOID vs Diameter with target coloring)
    sample_size = min(1200, len(df))
    scatter_sample = df.sample(n=sample_size, random_state=42)[[
        "full_name", "pdes", "H", "diameter_estimated_km", "moid", "a", "e", "i", "is_hazardous", "is_sentry_monitored", "class"
    ]].to_dict(orient="records")
    for r in scatter_sample:
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                r[k] = None
                
    eda_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_records": total_count,
        "class_balance": {
            "hazardous_pha": pha_count,
            "non_hazardous": non_pha_count,
            "pha_percentage": round((pha_count / total_count) * 100, 2)
        },
        "orbit_classes": orbit_classes,
        "sentry_monitoring_count": sentry_matches,
        "cad_approaches_count": cad_matches,
        "missing_summary": missing_summary,
        "numeric_stats": numeric_stats,
        "correlations_with_target": correlations,
        "histograms": histograms,
        "scatter_sample": scatter_sample
    }
    
    with open(EDA_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(eda_data, f, indent=2)
    print(f"[SUCCESS] EDA statistics saved to {EDA_STATS_FILE}")
    
    metadata = {
        "pipeline_version": "1.0.0",
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {
                "name": "NASA/JPL Small-Body Database (SBDB) Query API",
                "url": JPL_SBDB_QUERY_URL,
                "description": "Orbital solutions, physical properties, and authoritative PHA classification for all near-Earth asteroids.",
                "units": {
                    "a": "au (astronomical units)",
                    "e": "dimensionless (0 to 1 for elliptic)",
                    "i": "degrees with respect to ecliptic",
                    "q": "au (perihelion distance)",
                    "ad": "au (aphelion distance)",
                    "per_y": "Julian years (orbital period)",
                    "moid": "au (Earth minimum orbit intersection distance)",
                    "H": "mag (absolute magnitude)",
                    "diameter": "km (effective diameter)"
                }
            },
            {
                "name": "NASA/JPL Sentry: Earth Impact Monitoring",
                "url": JPL_SENTRY_URL,
                "description": "Automated collision monitoring system continuously scanning asteroid orbits for potential Earth impacts over next 100+ years."
            },
            {
                "name": "NASA/JPL Close Approach Data (CAD) API",
                "url": JPL_CAD_URL,
                "description": "High-precision geometric close approaches of NEAs to Earth, including approach date, miss distance (AU / Lunar Distances), and relative velocity (km/s)."
            }
        ],
        "authoritative_target_definition": (
            "NASA/JPL defines a Potentially Hazardous Asteroid (PHA) based on two strict physical/orbital criteria: "
            "1) Earth Minimum Orbit Intersection Distance (MOID) <= 0.05 AU (approx 19.5 Lunar Distances, ~7.48 million km); and "
            "2) Absolute magnitude H <= 22.0, corresponding roughly to a diameter of >= 140 meters."
        ),
        "dataset_limitations": (
            "1. Physical diameters are directly measured for only ~15% of asteroids (radar/WISE); the remainder are photometrically estimated from H assuming mean albedo p_v=0.14. "
            "2. Asteroid orbits evolve under gravitational perturbations from planets and Yarkovsky thermal radiation drift. "
            "3. Machine learning models predict PHA category membership based on orbital-physical geometry, but definitive impact risk requires numerical N-body orbit propagation and covariance propagation as performed by NASA Sentry."
        )
    }
    with open(DATASET_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SUCCESS] Dataset metadata saved to {DATASET_METADATA_FILE}")

if __name__ == "__main__":
    run_data_pipeline(force_download=False)
