#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════╗
# ║  intervention_simulator.py                                    ║
# ║  UrbanCool AI — Model 4: Spectral In-Painting Simulation      ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Simulates cooling interventions by modifying spectral and urban
# features, then re-predicting LST through the trained model.

from __future__ import annotations

import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "models"

# ────────────────────────────────────────────────────────────
# Intervention definitions
# ────────────────────────────────────────────────────────────

INTERVENTIONS = {
    "trees": {
        "NDVI_L": +0.15, "NDBI_L": -0.05, "albedo": +0.03,
        "impervious_ratio": -0.08, "building_density": -0.02,
        "description": "Urban tree planting / pocket park",
        "cost_inr_per_m2": 550,
        "lifespan_years": 40,
        "delta_T_range": (1.5, 3.0),
    },
    "cool_roofs": {
        "albedo": +0.25, "NDBI_L": -0.02,
        "description": "Cool reflective roof coating",
        "cost_inr_per_m2": 1050,
        "lifespan_years": 12,
        "delta_T_range": (2.0, 4.5),
    },
    "reflective_pavement": {
        "albedo": +0.15, "impervious_ratio": -0.02,
        "description": "Reflective / permeable pavement",
        "cost_inr_per_m2": 800,
        "lifespan_years": 25,
        "delta_T_range": (1.0, 2.5),
    },
    "blue_green": {
        "MNDWI_L": +0.12, "dist_water_m": -100.0, "humidity_pct": +3.0,
        "description": "Blue-green infrastructure corridor",
        "cost_inr_per_m2": 2200,
        "lifespan_years": 35,
        "delta_T_range": (1.5, 4.0),
    },
    "combined": {
        "NDVI_L": +0.15, "albedo": +0.30, "MNDWI_L": +0.08,
        "impervious_ratio": -0.10, "dist_water_m": -80.0,
        "NDBI_L": -0.05, "building_density": -0.03,
        "description": "Combined: trees + cool roofs + blue-green",
        "cost_inr_per_m2": 3500,
        "lifespan_years": 30,
        "delta_T_range": (3.0, 6.0),
    },
}

INTERVENTION_NAMES = {0: "none", 1: "trees", 2: "cool_roofs",
                      3: "reflective_pavement", 4: "blue_green"}
INTERVENTION_IDS = {v: k for k, v in INTERVENTION_NAMES.items()}

# Feature names matching the trained model
FEATURE_NAMES = [
    "NDVI_L", "NDBI_L", "MNDWI_L", "EVI_L", "SAVI_L", "albedo",
    "air_temp_C", "air_temp_C_max", "air_temp_C_min",
    "humidity_pct", "wind_speed", "rainfall_mm", "solar_rad_W_m2",
    "Elevation_m", "Slope_deg", "TPI_500m",
    "pop_density", "ntl_radiance", "children_ratio", "elderly_ratio",
    "road_density", "building_density", "impervious_ratio", "dist_road_m",
]


def load_lst_model(model_path: Path | None = None):
    """Load the trained LST model pipeline."""
    if model_path is None:
        candidates = [
            MODEL_DIR / "grid_lst_model.joblib",
            MODEL_DIR / "lst_xgboost.pkl",
            MODEL_DIR / "lst_xgboost.joblib",
        ]
        model_path = next((p for p in candidates if p.exists()), None)
        if model_path is None:
            raise FileNotFoundError("No trained LST model found in models/")

    if model_path.suffix == ".joblib":
        return joblib.load(model_path)
    else:
        with open(model_path, "rb") as f:
            return pickle.load(f)


def simulate_intervention(
    row: pd.Series | dict | np.ndarray,
    intervention: str,
    model,
    feature_list: list[str] | None = None,
) -> float:
    """
    Simulate an intervention on a single grid cell by modifying features
    and re-predicting LST.

    Parameters
    ----------
    row : feature values (Series with feature names, dict, or array)
    intervention : intervention name from INTERVENTIONS
    model : trained model pipeline
    feature_list : feature names (required if row is an array)

    Returns
    -------
    Predicted LST after intervention (°C)
    """
    if feature_list is None:
        feature_list = FEATURE_NAMES

    if isinstance(row, np.ndarray):
        modified = pd.Series(row.copy(), index=feature_list)
    elif isinstance(row, dict):
        modified = pd.Series({f: row.get(f, 0) for f in feature_list})
    else:
        modified = row[feature_list].copy()

    if intervention in INTERVENTIONS:
        deltas = INTERVENTIONS[intervention]
        for feat, delta in deltas.items():
            if isinstance(delta, (int, float)) and feat in modified.index:
                # Clip to reasonable ranges
                if feat in ("NDVI_L", "NDBI_L", "MNDWI_L", "EVI_L", "SAVI_L"):
                    modified[feat] = np.clip(modified[feat] + delta, -1.0, 1.0)
                elif feat == "albedo":
                    modified[feat] = np.clip(modified[feat] + delta, 0.0, 1.0)
                elif feat in ("impervious_ratio", "building_density"):
                    modified[feat] = np.clip(modified[feat] + delta, 0.0, 1.0)
                elif feat == "humidity_pct":
                    modified[feat] = np.clip(modified[feat] + delta, 0.0, 100.0)
                elif feat == "dist_water_m":
                    modified[feat] = max(0, modified[feat] + delta)
                else:
                    modified[feat] = modified[feat] + delta

    return float(model.predict(modified.values.reshape(1, -1))[0])


def simulate_delta_T(
    model,
    features: np.ndarray | pd.Series,
    intervention_id: int,
    feature_list: list[str] | None = None,
) -> float:
    """
    Compute temperature reduction from a given intervention.

    Returns
    -------
    delta_T : positive value = cooling (°C)
    """
    if feature_list is None:
        feature_list = FEATURE_NAMES

    if isinstance(features, np.ndarray):
        row = pd.Series(features, index=feature_list[:len(features)])
    else:
        row = features

    baseline = float(model.predict(row.values.reshape(1, -1))[0])

    if intervention_id == 0:
        return 0.0

    iv_name = INTERVENTION_NAMES.get(intervention_id, "trees")
    modified_lst = simulate_intervention(row, iv_name, model, feature_list)

    return max(0.0, baseline - modified_lst)


def full_scenario_compare(
    row: pd.Series | dict,
    model,
    feature_list: list[str] | None = None,
) -> dict:
    """
    Compare all intervention scenarios for a single grid cell.

    Returns
    -------
    dict with baseline LST and per-intervention results:
    {
        'baseline_lst': float,
        'scenarios': {
            'trees': {'lst': float, 'reduction_C': float, 'cost_inr_m2': int, ...},
            ...
        }
    }
    """
    if feature_list is None:
        feature_list = FEATURE_NAMES

    if isinstance(row, dict):
        row_series = pd.Series({f: row.get(f, 0) for f in feature_list})
    else:
        row_series = row

    baseline = float(model.predict(row_series[feature_list].values.reshape(1, -1))[0])

    scenarios = {}
    for name, config in INTERVENTIONS.items():
        iv_lst = simulate_intervention(row_series, name, model, feature_list)
        reduction = baseline - iv_lst
        scenarios[name] = {
            "lst_C": round(iv_lst, 2),
            "reduction_C": round(max(0, reduction), 2),
            "description": config.get("description", name),
            "cost_inr_per_m2": config.get("cost_inr_per_m2", 0),
            "lifespan_years": config.get("lifespan_years", 0),
        }

    return {
        "baseline_lst": round(baseline, 2),
        "scenarios": scenarios,
        "best_intervention": max(scenarios, key=lambda k: scenarios[k]["reduction_C"]),
    }


def batch_simulate(
    df: pd.DataFrame,
    model,
    feature_list: list[str] | None = None,
    intervention: str = "combined",
) -> pd.DataFrame:
    """
    Simulate an intervention across all rows in a DataFrame.

    Returns
    -------
    DataFrame with added columns: pred_lst_baseline, pred_lst_{intervention}, delta_T
    """
    if feature_list is None:
        feature_list = FEATURE_NAMES

    available = [f for f in feature_list if f in df.columns]
    X = df[available].copy()

    # Baseline prediction
    df = df.copy()
    df["pred_lst_baseline"] = model.predict(X)

    # Modified prediction
    X_mod = X.copy()
    if intervention in INTERVENTIONS:
        deltas = INTERVENTIONS[intervention]
        for feat, delta in deltas.items():
            if isinstance(delta, (int, float)) and feat in X_mod.columns:
                X_mod[feat] = X_mod[feat] + delta
                if feat in ("NDVI_L", "NDBI_L", "MNDWI_L"):
                    X_mod[feat] = X_mod[feat].clip(-1, 1)
                elif feat == "albedo":
                    X_mod[feat] = X_mod[feat].clip(0, 1)
                elif feat in ("impervious_ratio", "building_density"):
                    X_mod[feat] = X_mod[feat].clip(0, 1)

    df[f"pred_lst_{intervention}"] = model.predict(X_mod)
    df["delta_T"] = df["pred_lst_baseline"] - df[f"pred_lst_{intervention}"]

    return df
