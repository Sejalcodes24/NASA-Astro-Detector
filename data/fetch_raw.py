import os
import sys
import json
import time
import math
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    RAW_SBDB_FILE, RAW_SENTRY_FILE, RAW_CAD_FILE,
    PROCESSED_DATASET_FILE, DATASET_METADATA_FILE, EDA_STATS_FILE,
    JPL_SBDB_QUERY_URL, JPL_SENTRY_URL, JPL_CAD_URL
)

HEADERS = {'User-Agent': 'NASA-Asteroid-Risk-App/1.0 (Educational & Scientific Analysis)'}

def fetch_json(url, timeout=25):
    """Fetch JSON from a URL with error handling."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))

def collect_raw_data(force_download=False):
    """
    Collect real asteroid data from NASA JPL SSD APIs:
    1. JPL SBDB Query (NEAs: PHAs and non-PHAs)
    2. JPL Sentry (Impact risk monitoring)
    3. JPL CAD (Close Approach Data)
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # 1. SBDB NEAs
    sbdb_data = None
    if not force_download and RAW_SBDB_FILE.exists():
        print(f"[INFO] Loading cached SBDB data from {RAW_SBDB_FILE}")
        with open(RAW_SBDB_FILE, "r", encoding="utf-8") as f:
            sbdb_data = json.load(f)
    else:
        print("[INFO] Fetching real asteroid catalog from NASA JPL SBDB...")
        fields = "spkid,full_name,pdes,pha,neo,H,diameter,albedo,rot_per,a,e,i,q,ad,per_y,moid,moid_jup,t_jup,class,epoch_mjd,om,w,ma"
        
        # We fetch PHAs (all or up to 2600) and Non-PHA NEAs (up to 4000) for balanced, comprehensive sample
        url_pha = f"{JPL_SBDB_QUERY_URL}?sb-group=pha&fields={fields}&limit=2600"
        url_neo = f"{JPL_SBDB_QUERY_URL}?sb-group=neo&fields={fields}&limit=4500"
        
        try:
            print("  Fetching PHAs...")
            res_pha = fetch_json(url_pha)
            print(f"  Got {len(res_pha.get('data', []))} PHAs.")
            
            print("  Fetching general NEAs...")
            res_neo = fetch_json(url_neo)
            print(f"  Got {len(res_neo.get('data', []))} general NEAs.")
            
            # Combine without duplicates (by spkid index 0)
            fields_list = res_pha.get("fields", [])
            seen_ids = set()
            combined_rows = []
            
            for row in res_pha.get("data", []):
                spkid = row[0]
                if spkid not in seen_ids:
                    seen_ids.add(spkid)
                    combined_rows.append(row)
                    
            for row in res_neo.get("data", []):
                spkid = row[0]
                if spkid not in seen_ids:
                    seen_ids.add(spkid)
                    combined_rows.append(row)
            
            sbdb_data = {
                "source": "NASA/JPL Solar System Dynamics (SSD) Small-Body Database Query API",
                "url": JPL_SBDB_QUERY_URL,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "fields": fields_list,
                "count": len(combined_rows),
                "data": combined_rows
            }
            with open(RAW_SBDB_FILE, "w", encoding="utf-8") as f:
                json.dump(sbdb_data, f, indent=2)
            print(f"[SUCCESS] Saved {len(combined_rows)} asteroids to {RAW_SBDB_FILE}")
        except Exception as e:
            print(f"[WARN] Failed to fetch SBDB online: {e}")
            if RAW_SBDB_FILE.exists():
                with open(RAW_SBDB_FILE, "r", encoding="utf-8") as f:
                    sbdb_data = json.load(f)
            else:
                raise RuntimeError(f"No SBDB data available: {e}")

    # 2. JPL Sentry Objects
    sentry_data = None
    if not force_download and RAW_SENTRY_FILE.exists():
        print(f"[INFO] Loading cached Sentry data from {RAW_SENTRY_FILE}")
        with open(RAW_SENTRY_FILE, "r", encoding="utf-8") as f:
            sentry_data = json.load(f)
    else:
        print("[INFO] Fetching Sentry impact monitoring data from NASA JPL...")
        try:
            res_sentry = fetch_json(JPL_SENTRY_URL)
            sentry_data = {
                "source": "NASA/JPL Sentry Earth Impact Monitoring System",
                "url": JPL_SENTRY_URL,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "count": res_sentry.get("count", len(res_sentry.get("data", []))),
                "data": res_sentry.get("data", [])
            }
            with open(RAW_SENTRY_FILE, "w", encoding="utf-8") as f:
                json.dump(sentry_data, f, indent=2)
            print(f"[SUCCESS] Saved {len(sentry_data['data'])} Sentry risk objects to {RAW_SENTRY_FILE}")
        except Exception as e:
            print(f"[WARN] Failed to fetch Sentry online: {e}")
            if RAW_SENTRY_FILE.exists():
                with open(RAW_SENTRY_FILE, "r", encoding="utf-8") as f:
                    sentry_data = json.load(f)

    # 3. JPL CAD (Close Approach Data)
    cad_data = None
    if not force_download and RAW_CAD_FILE.exists():
        print(f"[INFO] Loading cached CAD data from {RAW_CAD_FILE}")
        with open(RAW_CAD_FILE, "r", encoding="utf-8") as f:
            cad_data = json.load(f)
    else:
        print("[INFO] Fetching Close Approach Data from NASA JPL CAD API...")
        try:
            url_cad = f"{JPL_CAD_URL}?dist-max=0.08&date-min=2024-01-01&date-max=2030-01-01&limit=600"
            res_cad = fetch_json(url_cad)
            cad_data = {
                "source": "NASA/JPL SSD Close-Approach Data API",
                "url": url_cad,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "fields": res_cad.get("fields", []),
                "count": res_cad.get("count", 0),
                "data": res_cad.get("data", [])
            }
            with open(RAW_CAD_FILE, "w", encoding="utf-8") as f:
                json.dump(cad_data, f, indent=2)
            print(f"[SUCCESS] Saved {len(cad_data['data'])} close approaches to {RAW_CAD_FILE}")
        except Exception as e:
            print(f"[WARN] Failed to fetch CAD online: {e}")
            if RAW_CAD_FILE.exists():
                with open(RAW_CAD_FILE, "r", encoding="utf-8") as f:
                    cad_data = json.load(f)

    return sbdb_data, sentry_data, cad_data

if __name__ == "__main__":
    sbdb, sentry, cad = collect_raw_data(force_download=True)
    print(f"Data collection complete! SBDB: {len(sbdb.get('data', []))}, Sentry: {len(sentry.get('data', [])) if sentry else 0}, CAD: {len(cad.get('data', [])) if cad else 0}")
