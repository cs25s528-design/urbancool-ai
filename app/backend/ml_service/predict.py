"""Local model inference service for UrbanCool AI backend."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_DIR / "models" / "grid_lst_model.joblib"
FEATURES_PATH = PROJECT_DIR / "models" / "grid_lst_features.json"
FEATURE_TABLE = PROJECT_DIR / "data" / "processed" / "pune_with_osm_features.parquet"
WARD_TABLE = PROJECT_DIR / "data" / "processed" / "pune_with_osm_wards.parquet"
METRICS_PATH = PROJECT_DIR / "data" / "processed" / "grid_lst_metrics.json"

INTERVENTION_DELTAS = {
    "none": {},
    "trees": {"NDVI_L": 0.15, "NDBI_L": -0.05, "albedo": 0.03, "impervious_ratio": -0.08},
    "cool_roofs": {"albedo": 0.25, "NDBI_L": -0.02},
    "reflective_pavement": {"albedo": 0.15, "impervious_ratio": -0.02},
    "blue_green": {"NDVI_L": 0.10, "MNDWI_L": 0.12, "albedo": 0.06, "impervious_ratio": -0.05},
    "combined": {"NDVI_L": 0.15, "MNDWI_L": 0.08, "albedo": 0.30, "impervious_ratio": -0.12},
}


@lru_cache(maxsize=1)
def load_feature_table() -> pd.DataFrame:
    return pd.read_parquet(FEATURE_TABLE)


@lru_cache(maxsize=1)
def load_ward_table() -> pd.DataFrame:
    return pd.read_parquet(WARD_TABLE) if WARD_TABLE.exists() else load_feature_table()


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_features() -> list[str]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature list not found: {FEATURES_PATH}")
    data = json.loads(FEATURES_PATH.read_text())
    return data["features"]


@lru_cache(maxsize=1)
def load_metrics() -> dict[str, Any]:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


def nearest_row(lat: float, lon: float, include_wards: bool = False) -> pd.Series:
    df = load_ward_table() if include_wards else load_feature_table()
    d2 = (df["lat"].astype(float) - lat) ** 2 + (df["lon"].astype(float) - lon) ** 2
    return df.loc[d2.idxmin()].copy()


def predict_from_row(row: pd.Series) -> float:
    features = load_features()
    X = pd.DataFrame([{f: row.get(f, np.nan) for f in features}])
    return float(load_model().predict(X)[0])


def current_heat(lat: float, lon: float) -> dict[str, Any]:
    row = nearest_row(lat, lon, include_wards=True)
    pred = predict_from_row(row)
    return {
        "query": {"lat": lat, "lon": lon},
        "nearest_grid": {"lat": float(row["lat"]), "lon": float(row["lon"]), "grid_id": str(row.get("grid_id", ""))},
        "observed_lst_C": float(row.get("lst_celsius", np.nan)),
        "pred_lst_C": pred,
        "air_temp_C": float(row.get("air_temp_C", np.nan)),
        "humidity_pct": float(row.get("humidity_pct", np.nan)),
        "wind_speed": float(row.get("wind_speed", np.nan)),
        "risk_level": classify_heat(pred),
        "ward_id": None if pd.isna(row.get("ward_id", np.nan)) else str(row.get("ward_id")),
        "ward_name": None if pd.isna(row.get("ward_name", np.nan)) else str(row.get("ward_name")),
        "ward_join_method": None if pd.isna(row.get("ward_join_method", np.nan)) else str(row.get("ward_join_method")),
    }


def classify_heat(lst_c: float) -> str:
    if not np.isfinite(lst_c):
        return "unknown"
    if lst_c >= 42:
        return "very_high"
    if lst_c >= 38:
        return "high"
    if lst_c >= 34:
        return "moderate"
    return "low"


def top_drivers(lat: float, lon: float, n: int = 5) -> dict[str, Any]:
    row = nearest_row(lat, lon)
    features = load_features()
    model = load_model().named_steps.get("model", load_model())
    if hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_, dtype=float)
    else:
        scores = np.abs(pd.Series({f: row.get(f, 0.0) for f in features}).fillna(0).to_numpy(float))
        scores = scores / max(scores.sum(), 1.0)
    order = np.argsort(scores)[::-1][:n]
    return {
        "lat": lat,
        "lon": lon,
        "drivers": [{"feature": features[i], "importance": float(scores[i])} for i in order],
    }


def simulate_intervention(lat: float, lon: float, intervention: str) -> dict[str, Any]:
    row = nearest_row(lat, lon)
    base = predict_from_row(row)
    modified = row.copy()
    for feature, delta in INTERVENTION_DELTAS.get(intervention, {}).items():
        if feature in modified:
            modified[feature] = float(modified[feature]) + delta
    after = predict_from_row(modified)
    return {
        "intervention": intervention,
        "base_lst_C": base,
        "scenario_lst_C": after,
        "delta_C": base - after,
    }


def compare_scenarios(lat: float, lon: float, interventions: list[str] | None = None) -> dict[str, Any]:
    if interventions is None:
        interventions = [k for k in INTERVENTION_DELTAS if k != "none"]
    scenarios = {name: simulate_intervention(lat, lon, name) for name in interventions}
    best = max(scenarios.values(), key=lambda x: x["delta_C"]) if scenarios else None
    return {"lat": lat, "lon": lon, "scenarios": scenarios, "best": best}


def recommend(lat: float, lon: float) -> dict[str, Any]:
    comparison = compare_scenarios(lat, lon)
    row = nearest_row(lat, lon, include_wards=True)
    return {
        "recommendation": comparison["best"],
        "risk_level": classify_heat(predict_from_row(row)),
        "ward_id": None if pd.isna(row.get("ward_id", np.nan)) else str(row.get("ward_id")),
        "ward_name": None if pd.isna(row.get("ward_name", np.nan)) else str(row.get("ward_name")),
    }

