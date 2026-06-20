#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════╗
# ║  shap_explainer.py                                           ║
# ║  UrbanCool AI — Model 3: SHAP Driver Attribution             ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Computes SHAP values for the LST model and provides per-location
# top driver attribution.

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

# Feature list matching the trained model
FEATURES = [
    "NDVI_L", "NDBI_L", "MNDWI_L", "EVI_L", "SAVI_L", "albedo",
    "air_temp_C", "air_temp_C_max", "air_temp_C_min",
    "humidity_pct", "wind_speed", "rainfall_mm", "solar_rad_W_m2",
    "Elevation_m", "Slope_deg", "TPI_500m",
    "pop_density", "ntl_radiance", "children_ratio", "elderly_ratio",
    "road_density", "building_density", "impervious_ratio", "dist_road_m",
]

# Human-readable feature labels
FEATURE_LABELS = {
    "NDVI_L": "Vegetation Cover (NDVI)",
    "NDBI_L": "Built-up Density (NDBI)",
    "MNDWI_L": "Water Presence (MNDWI)",
    "EVI_L": "Enhanced Vegetation (EVI)",
    "SAVI_L": "Soil-Adjusted Vegetation (SAVI)",
    "albedo": "Surface Reflectivity (Albedo)",
    "air_temp_C": "Air Temperature",
    "air_temp_C_max": "Max Air Temperature",
    "air_temp_C_min": "Min Air Temperature",
    "humidity_pct": "Relative Humidity",
    "wind_speed": "Wind Speed",
    "rainfall_mm": "Rainfall",
    "solar_rad_W_m2": "Solar Radiation",
    "Elevation_m": "Elevation",
    "Slope_deg": "Terrain Slope",
    "TPI_500m": "Topographic Position",
    "pop_density": "Population Density",
    "ntl_radiance": "Nighttime Lights",
    "children_ratio": "Children Population Ratio",
    "elderly_ratio": "Elderly Population Ratio",
    "road_density": "Road Density",
    "building_density": "Building Density",
    "impervious_ratio": "Impervious Surface Ratio",
    "dist_road_m": "Distance to Road",
    "dist_park_m": "Distance to Park",
    "dist_water_m": "Distance to Water",
}


def load_model_and_features(
    model_path: Path | None = None,
) -> tuple:
    """Load the trained LST model and determine feature names."""
    if model_path is None:
        # Try multiple model paths
        candidates = [
            MODEL_DIR / "grid_lst_model.joblib",
            MODEL_DIR / "lst_xgboost.pkl",
            MODEL_DIR / "lst_xgboost.joblib",
        ]
        model_path = next((p for p in candidates if p.exists()), None)
        if model_path is None:
            raise FileNotFoundError("No trained LST model found in models/")

    if model_path.suffix == ".joblib":
        pipeline = joblib.load(model_path)
    else:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)

    # Load feature list
    features_json = MODEL_DIR / "grid_lst_features.json"
    if features_json.exists():
        meta = json.loads(features_json.read_text())
        features = meta.get("features", FEATURES)
    else:
        features = FEATURES

    return pipeline, features


def create_shap_explainer(pipeline, X_background: pd.DataFrame | np.ndarray):
    """
    Create a SHAP explainer for the model pipeline.
    Falls back to permutation-based explanation if TreeExplainer fails.
    """
    try:
        import shap
    except ImportError:
        print("  ⚠️ SHAP not installed. Using permutation importance fallback.")
        return None, "permutation_fallback"

    # Extract the core model from the pipeline
    model = pipeline.named_steps.get("model", pipeline)
    imputer = pipeline.named_steps.get("imputer", None)

    if imputer is not None:
        X_bg = imputer.transform(X_background)
    else:
        X_bg = np.asarray(X_background)

    # Try TreeExplainer first (fastest for tree models)
    try:
        explainer = shap.TreeExplainer(model)
        print("  Using SHAP TreeExplainer")
        return explainer, "tree"
    except Exception:
        pass

    # Fallback to KernelExplainer with a background sample
    try:
        bg_sample = shap.sample(X_bg, min(100, len(X_bg)))
        explainer = shap.KernelExplainer(model.predict, bg_sample)
        print("  Using SHAP KernelExplainer (slower)")
        return explainer, "kernel"
    except Exception as e:
        print(f"  ⚠️ SHAP explainer creation failed: {e}")
        return None, "failed"


def get_top_drivers(
    explainer,
    explainer_type: str,
    row_features: np.ndarray,
    feature_names: list[str],
    n: int = 5,
    pipeline=None,
) -> list[dict]:
    """
    Get top-N SHAP drivers for a single grid cell.

    Parameters
    ----------
    explainer : SHAP explainer or None
    explainer_type : 'tree', 'kernel', or 'permutation_fallback'
    row_features : 1D array of feature values
    feature_names : list of feature names
    n : number of top drivers to return
    pipeline : model pipeline (needed for permutation fallback)

    Returns
    -------
    list of dicts: [{'feature': str, 'label': str, 'effect_C': float}, ...]
    """
    if explainer is None or explainer_type == "failed":
        # Return model feature importances as a fallback
        if pipeline is not None:
            model = pipeline.named_steps.get("model", pipeline)
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                order = np.argsort(np.abs(importances))[::-1][:n]
                return [
                    {
                        "feature": feature_names[i],
                        "label": FEATURE_LABELS.get(feature_names[i], feature_names[i]),
                        "effect_C": round(float(importances[i]), 3),
                    }
                    for i in order
                ]
        return []

    row = row_features.reshape(1, -1) if row_features.ndim == 1 else row_features

    if explainer_type == "tree":
        sv = explainer.shap_values(row)
        if isinstance(sv, list):
            sv = sv[0]
        sv = sv.flatten()
    else:
        sv = explainer.shap_values(row).flatten()

    # Sort by absolute SHAP value
    order = np.argsort(np.abs(sv))[::-1][:n]
    return [
        {
            "feature": feature_names[i],
            "label": FEATURE_LABELS.get(feature_names[i], feature_names[i]),
            "effect_C": round(float(sv[i]), 3),
        }
        for i in order
    ]


def compute_global_shap(
    explainer,
    explainer_type: str,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    max_samples: int = 1000,
) -> pd.DataFrame:
    """
    Compute global SHAP values for a sample of the dataset.

    Returns
    -------
    DataFrame with mean absolute SHAP values per feature
    """
    if explainer is None:
        return pd.DataFrame()

    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample = np.asarray(X)[idx]
    else:
        X_sample = np.asarray(X)

    print(f"  Computing SHAP values for {len(X_sample)} samples...")
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        sv = sv[0]

    mean_abs = np.abs(sv).mean(axis=0)
    result = pd.DataFrame({
        "feature": feature_names,
        "label": [FEATURE_LABELS.get(f, f) for f in feature_names],
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)

    return result


def build_and_save_explainer(
    input_path: Path | None = None,
    model_path: Path | None = None,
    output_path: Path | None = None,
    max_background: int = 500,
) -> tuple:
    """
    Full SHAP explainer pipeline: load model, create explainer, save.
    """
    pipeline, features = load_model_and_features(model_path)

    if input_path is None:
        input_path = PROC_DIR / "pune_with_osm_features.parquet"
    if output_path is None:
        output_path = MODEL_DIR / "shap_explainer.pkl"

    df = pd.read_parquet(input_path)
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].copy()

    # Impute for SHAP
    imputer = pipeline.named_steps.get("imputer", None)
    if imputer is not None:
        X_imp = pd.DataFrame(imputer.transform(X), columns=available_features, index=X.index)
    else:
        X_imp = X.fillna(X.median())

    # Background sample
    bg_idx = np.random.choice(len(X_imp), min(max_background, len(X_imp)), replace=False)
    X_bg = X_imp.iloc[bg_idx]

    explainer, etype = create_shap_explainer(pipeline, X_bg)

    if explainer is not None:
        # Compute global SHAP
        global_shap = compute_global_shap(explainer, etype, X_imp, available_features)
        print("\n  Top 10 global SHAP drivers:")
        print(global_shap.head(10).to_string(index=False))

        # Save explainer
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump({
                "explainer": explainer,
                "type": etype,
                "features": available_features,
                "global_shap": global_shap.to_dict("records"),
            }, f)
        print(f"\n  ✅ Saved SHAP explainer: {output_path}")
    else:
        print("  ⚠️ SHAP explainer not available; saved feature importance fallback")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model = pipeline.named_steps.get("model", pipeline)
        imp = getattr(model, "feature_importances_", np.zeros(len(available_features)))
        with open(output_path, "wb") as f:
            pickle.dump({
                "explainer": None,
                "type": "feature_importance",
                "features": available_features,
                "importances": imp.tolist(),
            }, f)

    return explainer, etype, available_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SHAP explainer for LST model")
    parser.add_argument("--input", type=Path, default=PROC_DIR / "pune_with_osm_features.parquet")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "shap_explainer.pkl")
    args = parser.parse_args()

    print("=" * 60)
    print("Model 3: SHAP Driver Attribution")
    print("=" * 60)
    build_and_save_explainer(args.input, args.model, args.output)


if __name__ == "__main__":
    main()
