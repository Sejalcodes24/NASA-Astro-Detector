import os
import sys
import math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    BEST_MODEL_FILE, FEATURE_IMPORTANCE_FILE, EVALUATION_METRICS_FILE,
    PROCESSED_DATASET_FILE
)
from data.data_pipeline import (
    calculate_photometric_diameter,
    calculate_kinetic_energy_mt,
    calculate_opik_encounter_velocity
)
from models.train import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES

class AsteroidRiskPredictor:
    """Predictor and Explainability engine for Asteroid Earth Risk."""
    
    def __init__(self):
        self.model = None
        self.feature_meta = None
        self.baseline_stats = None
        self.load_artifacts()
        
    def load_artifacts(self):
        """Load fitted pipeline and reference statistics."""
        if not BEST_MODEL_FILE.exists():
            raise FileNotFoundError(f"Model file not found at {BEST_MODEL_FILE}. Run train.py first!")
        self.model = joblib.load(BEST_MODEL_FILE)
        
        # Load baseline statistics from processed dataset for local explanations
        if PROCESSED_DATASET_FILE.exists():
            df = pd.read_csv(PROCESSED_DATASET_FILE)
            self.baseline_stats = {
                col: {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()) if df[col].std() > 0 else 1.0,
                    "median": float(df[col].median()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                }
                for col in NUMERIC_FEATURES if col in df.columns
            }
        else:
            self.baseline_stats = {}
            
    def engineer_input_features(self, input_data: dict) -> pd.DataFrame:
        """
        Validate, normalize and feature-engineer raw user or catalog input.
        Supported keys:
          H (mag), diameter (km, optional), a (AU), e, i (deg), moid (AU),
          q (AU, optional), ad (AU, optional), per_y (years, optional),
          class (str: AMO, APO, ATE, IEO)
        """
        data = dict(input_data)
        
        # Required core inputs with defaults if missing
        h_mag = float(data.get("H") if data.get("H") is not None else 20.0)
        a = float(data.get("a") if data.get("a") is not None else 1.4)
        e = max(0.0001, min(0.9999, float(data.get("e") if data.get("e") is not None else 0.3)))
        i_deg = max(0.0, float(data.get("i") if data.get("i") is not None else 10.0))
        moid = max(0.0, float(data.get("moid") if data.get("moid") is not None else 0.04))
        
        raw_class = data.get("class") or data.get("orbit_class") or "APO"
        orbit_class = str(raw_class).upper()
        if orbit_class not in ["AMO", "APO", "ATE", "IEO"]:
            # Deduce class based on orbital elements
            q_val = a * (1.0 - e)
            ad_val = a * (1.0 + e)
            if a > 1.0 and 1.017 < q_val < 1.3:
                orbit_class = "AMO"
            elif a >= 1.0 and q_val <= 1.017:
                orbit_class = "APO"
            elif a < 1.0 and ad_val > 0.983:
                orbit_class = "ATE"
            elif a < 1.0 and ad_val < 0.983:
                orbit_class = "IEO"
            else:
                orbit_class = "APO"
                
        # Perihelion & Aphelion
        q = float(data.get("q") if data.get("q") is not None else (a * (1.0 - e)))
        ad = float(data.get("ad") if data.get("ad") is not None else (a * (1.0 + e)))
        per_y = float(data.get("per_y") if data.get("per_y") is not None else (a ** 1.5))
        
        # Diameter
        if "diameter" in data and data["diameter"] is not None and not math.isnan(float(data["diameter"])) and float(data["diameter"]) > 0:
            diameter_km = float(data["diameter"])
        else:
            diameter_km = calculate_photometric_diameter(h_mag)
            
        # Engineered features
        semi_minor_axis = a * math.sqrt(max(0.0, 1.0 - e**2))
        orbit_energy = -1.0 / (2.0 * a)
        perihelion_ratio = q / a
        aphelion_ratio = ad / a
        eccentricity_sq = e ** 2
        inclination_rad = math.radians(i_deg)
        relative_velocity_kms = calculate_opik_encounter_velocity(a, e, i_deg)
        
        collision_prob_factor = 1.0 / (
            (moid + 0.0001) * 
            math.sin(max(inclination_rad, math.radians(0.5))) * 
            math.sqrt(max(0.001, a * (1.0 - e**2)))
        )
        collision_prob_factor = min(10000.0, collision_prob_factor)
        
        kinetic_energy_mt = calculate_kinetic_energy_mt(diameter_km, relative_velocity_kms)
        
        row = {
            "H": h_mag,
            "diameter_estimated_km": diameter_km,
            "a": a,
            "e": e,
            "i": i_deg,
            "q": q,
            "ad": ad,
            "per_y": per_y,
            "moid": moid,
            "semi_minor_axis": semi_minor_axis,
            "orbit_energy": orbit_energy,
            "perihelion_ratio": perihelion_ratio,
            "aphelion_ratio": aphelion_ratio,
            "eccentricity_sq": eccentricity_sq,
            "inclination_rad": inclination_rad,
            "relative_velocity_kms": relative_velocity_kms,
            "collision_prob_factor": collision_prob_factor,
            "kinetic_energy_mt": kinetic_energy_mt,
            "class": orbit_class
        }
        return pd.DataFrame([row])

    def predict(self, input_dict: dict) -> dict:
        """Run model inference with explainability and uncertainty quantification."""
        df_row = self.engineer_input_features(input_dict)
        
        # Predict probability
        proba = self.model.predict_proba(df_row)[0]
        hazard_prob = float(proba[1])
        non_hazard_prob = float(proba[0])
        pred_class = int(self.model.predict(df_row)[0])
        
        # Authoritative rule check (NASA criteria: MOID <= 0.05 AU and H <= 22.0)
        moid_val = float(df_row["moid"].iloc[0])
        h_val = float(df_row["H"].iloc[0])
        authoritative_pha = (moid_val <= 0.05) and (h_val <= 22.0)
        
        # Risk classification level
        if hazard_prob >= 0.70:
            risk_level = "CRITICAL / HIGH RISK"
            risk_badge = "critical"
            risk_color = "#ef4444"
        elif hazard_prob >= 0.35:
            risk_level = "ELEVATED CONCERN"
            risk_badge = "elevated"
            risk_color = "#f59e0b"
        else:
            risk_level = "NOMINAL / LOW RISK"
            risk_badge = "nominal"
            risk_color = "#10b981"
            
        # Model Confidence & Uncertainty
        confidence_score = abs(hazard_prob - 0.5) * 2.0  # 0 to 1
        if confidence_score > 0.85:
            uncertainty_text = "High model confidence. Clear separation along orbital intersection and absolute magnitude boundaries."
            uncertainty_level = "Low Uncertainty"
        elif confidence_score > 0.50:
            uncertainty_text = "Moderate model confidence. Object resides in transitional parameter region near risk thresholds."
            uncertainty_level = "Moderate Uncertainty"
        else:
            uncertainty_text = "High prediction uncertainty. Near-boundary orbital geometry warrants observational confirmation."
            uncertainty_level = "High Uncertainty"
            
        # Local Explainability (feature contributions)
        explanations = self._compute_local_explanation(df_row.iloc[0].to_dict(), hazard_prob)
        
        # Human-readable summary
        plain_language = self._generate_plain_language_summary(
            hazard_prob, pred_class, authoritative_pha, moid_val, h_val,
            df_row["diameter_estimated_km"].iloc[0], df_row["relative_velocity_kms"].iloc[0]
        )
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_prediction": {
                "is_hazardous": bool(pred_class),
                "hazard_probability": round(hazard_prob, 4),
                "hazard_percentage": round(hazard_prob * 100, 2),
                "non_hazard_probability": round(non_hazard_prob, 4),
                "risk_level": risk_level,
                "risk_badge": risk_badge,
                "risk_color": risk_color,
                "confidence_score": round(confidence_score, 4),
                "uncertainty_level": uncertainty_level,
                "uncertainty_text": uncertainty_text
            },
            "authoritative_status": {
                "nasa_pha_criteria_met": bool(authoritative_pha),
                "moid_threshold_met": bool(moid_val <= 0.05),
                "magnitude_threshold_met": bool(h_val <= 22.0),
                "notice": (
                    "Authoritative NASA/JPL Potentially Hazardous Asteroid (PHA) designation requires "
                    "both Earth MOID <= 0.05 AU (~19.5 Lunar Distances) AND Absolute Magnitude H <= 22.0 (diameter >= ~140m)."
                )
            },
            "engineered_features": {
                k: round(float(v), 4) if isinstance(v, (int, float, np.number)) else str(v)
                for k, v in df_row.iloc[0].items()
            },
            "explanations": explanations,
            "plain_language_summary": plain_language
        }
        
    def _compute_local_explanation(self, row_dict: dict, hazard_prob: float) -> list:
        """
        Compute local directional feature contributions:
        Indicates which parameters push probability toward hazardous (+) vs nominal (-).
        """
        factors = []
        
        # 1. MOID contribution
        moid = row_dict["moid"]
        if moid <= 0.05:
            delta = (0.05 - moid) / 0.05
            factors.append({
                "feature": "moid",
                "label": "Earth MOID",
                "value": f"{moid:.4f} AU",
                "impact": "INCREASES_RISK",
                "weight": min(1.0, 0.45 + 0.5 * delta),
                "description": f"Earth MOID of {moid:.4f} AU is within the critical 0.05 AU collision danger envelope ({moid * 389.17:.1f} Lunar Distances)."
            })
        else:
            factors.append({
                "feature": "moid",
                "label": "Earth MOID",
                "value": f"{moid:.4f} AU",
                "impact": "DECREASES_RISK",
                "weight": min(1.0, 0.3 + 0.1 * (moid - 0.05)),
                "description": f"Earth MOID of {moid:.4f} AU safely exceeds the 0.05 AU collision threshold ({moid * 389.17:.1f} Lunar Distances away)."
            })
            
        # 2. Absolute magnitude H & Diameter
        h = row_dict["H"]
        diam = row_dict["diameter_estimated_km"]
        if h <= 22.0:
            factors.append({
                "feature": "H",
                "label": "Absolute Magnitude & Size",
                "value": f"H={h:.2f} (D~{diam*1000:.0f} m)",
                "impact": "INCREASES_RISK",
                "weight": min(1.0, 0.35 + 0.05 * (22.0 - h)),
                "description": f"Brightness H={h:.2f} indicates a large body (diameter ~{diam*1000:.0f} m), capable of severe regional or global catastrophe."
            })
        else:
            factors.append({
                "feature": "H",
                "label": "Absolute Magnitude & Size",
                "value": f"H={h:.2f} (D~{diam*1000:.0f} m)",
                "impact": "DECREASES_RISK",
                "weight": min(1.0, 0.3 + 0.05 * (h - 22.0)),
                "description": f"Brightness H={h:.2f} corresponds to a sub-140m object (D~{diam*1000:.0f} m), below NASA catastrophic PHA threshold."
            })
            
        # 3. Collision Probability Proxy
        coll_proxy = row_dict.get("collision_prob_factor", 1.0)
        if coll_proxy > 100.0:
            factors.append({
                "feature": "collision_prob_factor",
                "label": "Orbital Geometry Alignment",
                "value": f"{coll_proxy:.1f}",
                "impact": "INCREASES_RISK",
                "weight": 0.25,
                "description": "High alignment of orbit size and low inclination increases orbital intersection frequency with Earth's path."
            })
            
        # 4. Kinetic Energy
        ke = row_dict.get("kinetic_energy_mt", 0.0)
        if ke > 100.0:
            factors.append({
                "feature": "kinetic_energy_mt",
                "label": "Impact Kinetic Energy",
                "value": f"{ke:,.1f} MT",
                "impact": "INCREASES_RISK",
                "weight": 0.20,
                "description": f"Potential impact kinetic energy of {ke:,.1f} Megatons of TNT exceeds major nuclear arsenals."
            })
            
        # Sort factors by weight descending
        factors.sort(key=lambda x: x["weight"], reverse=True)
        return factors
        
    def _generate_plain_language_summary(self, prob, pred_class, auth_pha, moid, h, diam, v_rel):
        """Build clear, transparent natural language risk summary."""
        lines = []
        if prob >= 0.70:
            lines.append(
                f"The predictive model classifies this asteroid as POTENTIALLY HAZARDOUS with a model risk probability of {prob * 100:.1f}%."
            )
        elif prob >= 0.35:
            lines.append(
                f"The predictive model flags this asteroid under ELEVATED CONCERN with a model risk probability of {prob * 100:.1f}%."
            )
        else:
            lines.append(
                f"The predictive model classifies this asteroid as NOMINAL / LOW RISK with a hazard probability of {prob * 100:.1f}%."
            )
            
        if auth_pha:
            lines.append(
                f"This matches NASA/JPL's authoritative PHA standard: its Earth MOID ({moid:.4f} AU) is <= 0.05 AU and absolute magnitude H ({h:.2f}) is <= 22.0 (~{diam*1000:.0f} meters across)."
            )
        else:
            reasons = []
            if moid > 0.05:
                reasons.append(f"its Earth MOID ({moid:.4f} AU) is outside the 0.05 AU hazard limit")
            if h > 22.0:
                reasons.append(f"its size (H={h:.2f}, diameter ~{diam*1000:.0f} m) is smaller than the 140m threshold")
            lines.append(f"Under authoritative NASA/JPL taxonomy, this does NOT meet both PHA criteria because {', and '.join(reasons)}.")
            
        lines.append(
            f"Encounter velocity relative to Earth is calculated at {v_rel:.1f} km/s. Note: Planetary risk predictions are model approximations; definitive impact determinations require NASA Sentry covariance matrix integration."
        )
        return " ".join(lines)

# Singleton instance
predictor = AsteroidRiskPredictor()
