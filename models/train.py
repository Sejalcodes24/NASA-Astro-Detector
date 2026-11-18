import os
import sys
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    PROCESSED_DATASET_FILE, SAVED_MODELS_DIR,
    BEST_MODEL_FILE, MODEL_COMPARISON_FILE, FEATURE_IMPORTANCE_FILE,
    EVALUATION_METRICS_FILE, RANDOM_STATE, TEST_SIZE
)

NUMERIC_FEATURES = [
    "H", "diameter_estimated_km", "a", "e", "i", "q", "ad", "per_y", 
    "moid", "semi_minor_axis", "orbit_energy", "perihelion_ratio", 
    "aphelion_ratio", "eccentricity_sq", "inclination_rad", 
    "relative_velocity_kms", "collision_prob_factor", "kinetic_energy_mt"
]

CATEGORICAL_FEATURES = ["class"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_hazardous"

def train_and_evaluate_models():
    """Reproducible training and rigorous evaluation across baseline and advanced models."""
    print("=" * 65)
    print("ASTEROID RISK CLASSIFICATION: MODEL TRAINING & BENCHMARKING")
    print(f"Random State: {RANDOM_STATE} | Test Split: {TEST_SIZE * 100}%")
    print("=" * 65)
    
    # 1. Load Processed Data
    if not PROCESSED_DATASET_FILE.exists():
        raise FileNotFoundError(f"Processed dataset not found at {PROCESSED_DATASET_FILE}. Run data_pipeline.py first!")
        
    df = pd.read_csv(PROCESSED_DATASET_FILE)
    print(f"[DATA] Total samples loaded: {len(df)}")
    
    # Verify features and target exist
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    
    # Handle any remaining nan in numeric features with median (on X)
    for col in NUMERIC_FEATURES:
        if X[col].isna().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].fillna("Unknown").astype(str)
        
    class_counts = y.value_counts().to_dict()
    print(f"[DATA] Class distribution: Non-Hazardous (0): {class_counts.get(0, 0)}, Hazardous PHA (1): {class_counts.get(1, 0)}")
    pha_prevalence = y.mean() * 100
    print(f"[DATA] PHA Prevalence: {pha_prevalence:.2f}%")
    
    # 2. Stratified Train-Test Split (strictly keeping test set unseen)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[SPLIT] Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # 3. Build Preprocessing Pipeline
    # Preprocessor fits ONLY on training data!
    numeric_transformer = Pipeline(steps=[
        ("scaler", RobustScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    
    # 4. Define Candidate Models
    models = {
        "Logistic_Regression": {
            "name": "Logistic Regression (Balanced Baseline)",
            "estimator": LogisticRegression(
                class_weight="balanced", 
                max_iter=1000, 
                random_state=RANDOM_STATE,
                C=1.0
            )
        },
        "Random_Forest": {
            "name": "Random Forest Classifier",
            "estimator": RandomForestClassifier(
                n_estimators=150, 
                max_depth=14, 
                min_samples_split=4, 
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        },
        "Gradient_Boosting": {
            "name": "Gradient Boosting (GBM)",
            "estimator": GradientBoostingClassifier(
                n_estimators=150, 
                learning_rate=0.08, 
                max_depth=5, 
                subsample=0.85,
                random_state=RANDOM_STATE
            )
        },
        "Deep_Learning_MLP": {
            "name": "Deep Neural Network (MLP Tabular)",
            "estimator": MLPClassifier(
                hidden_layer_sizes=(64, 32), 
                activation="relu", 
                alpha=0.001, 
                early_stopping=True, 
                max_iter=300, 
                random_state=RANDOM_STATE
            )
        }
    }
    
    results = {}
    fitted_pipelines = {}
    best_score = -1.0
    best_model_key = None
    
    # 5. Train and Evaluate Each Model
    for key, item in models.items():
        name = item["name"]
        clf = item["estimator"]
        print(f"\n[TRAINING] {name}...")
        
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        # Fit on training set only
        pipeline.fit(X_train, y_train)
        fitted_pipelines[key] = pipeline
        
        # Predictions on unseen test set
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = pipeline.decision_function(X_test)
            # normalize to [0, 1]
            y_prob = 1.0 / (1.0 + np.exp(-y_prob))
            
        # Comprehensive Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # ROC and PR curves
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        pr_curve_p, pr_curve_r, _ = precision_recall_curve(y_test, y_prob)
        
        # Downsample curve points for fast, lightweight JSON transfer
        step_roc = max(1, len(fpr) // 40)
        step_pr = max(1, len(pr_curve_p) // 40)
        
        # Calibration curve (reliability diagram)
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=8, strategy="uniform")
        
        # Composite score prioritizing PR-AUC and Recall (Planetary Defense requires high recall)
        composite_score = 0.5 * pr_auc + 0.3 * roc_auc + 0.2 * f1
        
        print(f"  Accuracy: {acc:.4f} | Bal Acc: {bal_acc:.4f}")
        print(f"  Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}")
        print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        
        results[key] = {
            "key": key,
            "name": name,
            "metrics": {
                "accuracy": round(float(acc), 4),
                "balanced_accuracy": round(float(bal_acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "roc_auc": round(float(roc_auc), 4),
                "pr_auc": round(float(pr_auc), 4),
                "brier_score": round(float(brier), 4),
                "composite_score": round(float(composite_score), 4)
            },
            "confusion_matrix": {
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
            },
            "roc_curve": {
                "fpr": [round(float(x), 4) for x in fpr[::step_roc]],
                "tpr": [round(float(x), 4) for x in tpr[::step_roc]]
            },
            "pr_curve": {
                "precision": [round(float(x), 4) for x in pr_curve_p[::step_pr]],
                "recall": [round(float(x), 4) for x in pr_curve_r[::step_pr]]
            },
            "calibration_curve": {
                "mean_predicted_value": [round(float(x), 4) for x in prob_pred],
                "fraction_of_positives": [round(float(x), 4) for x in prob_true]
            }
        }
        
        if composite_score > best_score:
            best_score = composite_score
            best_model_key = key
            
    print(f"\n[CHAMPION MODEL] Selected: {models[best_model_key]['name']} (Composite: {best_score:.4f})")
    
    # 6. Feature Importance & Explainability (computed on champion model)
    champion_pipeline = fitted_pipelines[best_model_key]
    
    # Get feature names after column transformation
    ohe = champion_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERIC_FEATURES + cat_feature_names
    
    importance_dict = {}
    clf = champion_pipeline.named_steps["classifier"]
    
    if hasattr(clf, "feature_importances_"):
        raw_importances = clf.feature_importances_
        for fname, imp in zip(all_feature_names, raw_importances):
            importance_dict[fname] = round(float(imp), 4)
    elif hasattr(clf, "coef_"):
        raw_importances = np.abs(clf.coef_[0])
        for fname, imp in zip(all_feature_names, raw_importances):
            importance_dict[fname] = round(float(imp), 4)
    else:
        # Permutation importance fallback for neural network
        perm_imp = permutation_importance(champion_pipeline, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE)
        for fname, imp in zip(ALL_FEATURES, perm_imp.importances_mean):
            importance_dict[fname] = round(float(imp), 4)
            
    sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    print("\n[EXPLAINABILITY] Top 8 Features by Global Importance:")
    for fname, val in sorted_importances[:8]:
        print(f"  {fname:25}: {val:.4f}")
        
    # 7. Save Best Fitted Pipeline and Evaluation Data
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save best model
    joblib.dump(champion_pipeline, BEST_MODEL_FILE)
    print(f"\n[SUCCESS] Champion pipeline saved to {BEST_MODEL_FILE}")
    
    # Save feature importances
    feature_imp_payload = {
        "model_key": best_model_key,
        "model_name": models[best_model_key]["name"],
        "importances": [{"feature": k, "importance": v} for k, v in sorted_importances],
        "feature_descriptions": {
            "moid": "Minimum Orbit Intersection Distance to Earth (AU) - strongest orbital collision criterion",
            "H": "Absolute magnitude (mag) - inversely proportional to physical asteroid size",
            "diameter_estimated_km": "Estimated physical asteroid diameter (km)",
            "collision_prob_factor": "Geometrical collision proxy factor based on inclination, MOID, and orbit size",
            "kinetic_energy_mt": "Estimated potential impact kinetic energy in Megatons of TNT",
            "relative_velocity_kms": "Öpik-calculated encounter velocity relative to Earth (km/s)",
            "q": "Perihelion distance (AU) - closest orbital approach to Sun",
            "a": "Semi-major axis (AU) - mean distance to Sun",
            "e": "Orbital eccentricity - how stretched the ellipse is",
            "i": "Orbital inclination (degrees) - angle relative to Earth's orbital plane"
        }
    }
    with open(FEATURE_IMPORTANCE_FILE, "w", encoding="utf-8") as f:
        json.dump(feature_imp_payload, f, indent=2)
        
    # Save Model Comparison
    comparison_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "champion_model": best_model_key,
        "models": results
    }
    with open(MODEL_COMPARISON_FILE, "w", encoding="utf-8") as f:
        json.dump(comparison_payload, f, indent=2)
        
    # Save Evaluation Metrics summary
    eval_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "champion_model_name": models[best_model_key]["name"],
        "metrics": results[best_model_key]["metrics"],
        "confusion_matrix": results[best_model_key]["confusion_matrix"]
    }
    with open(EVALUATION_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
        
    print(f"[SUCCESS] All model metrics, curves, and artifacts saved to {SAVED_MODELS_DIR}")
    print("=" * 65)
    return results

if __name__ == "__main__":
    train_and_evaluate_models()
