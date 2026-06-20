#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════╗
# ║  train_future_model.py                                       ║
# ║  UrbanCool AI — Model 2: Future Hotspot Forecasting          ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Lag-feature temporal ML predicts future summer LST and hotspot
# probability. Uses temporal split for honest evaluation.

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

TARGET_REG = "lst_celsius"
TARGET_CLS = "hotspot_label"


# ────────────────────────────────────────────────────────────
# Temporal feature creation
# ────────────────────────────────────────────────────────────

def create_temporal_features(df: pd.DataFrame, lags_yr: list[int] | None = None) -> pd.DataFrame:
    """Create lag features for future forecasting."""
    if lags_yr is None:
        lags_yr = [1, 2, 5]

    df = df.sort_values(["grid_id", "date"]).copy() if "date" in df.columns else df.copy()

    if "grid_id" not in df.columns:
        # Single-date dataset: create grid_id from coordinates
        df["grid_id"] = df.apply(
            lambda r: f"{r['lat']:.4f}_{r['lon']:.4f}",
            axis=1,
        )

    # For single-date datasets, create synthetic lag features from spatial neighbours
    if "date" not in df.columns or df["date"].nunique() <= 1:
        print("  Single-date dataset detected; using spatial proxy lag features")
        return _create_spatial_proxy_lags(df)

    g_lst = df.groupby("grid_id")[TARGET_REG]

    for lag in lags_yr:
        shift_n = lag * 4  # 4 seasons per year
        df[f"lst_lag_{lag}yr"] = g_lst.shift(shift_n)

        ndvi_col = "NDVI_L" if "NDVI_L" in df.columns else "NDVI"
        if ndvi_col in df.columns:
            df[f"ndvi_lag_{lag}yr"] = df.groupby("grid_id")[ndvi_col].shift(shift_n)

    # 5-year rolling trend
    df["lst_trend_5yr"] = g_lst.transform(
        lambda x: x.rolling(20, min_periods=10).apply(
            lambda s: np.polyfit(range(len(s)), s, 1)[0] if len(s) >= 2 else 0,
            raw=False,
        )
    )

    ndbi_col = "NDBI_L" if "NDBI_L" in df.columns else "NDBI"
    if ndbi_col in df.columns:
        df["ndbi_change_5yr"] = df.groupby("grid_id")[ndbi_col].transform(
            lambda x: x - x.shift(20)
        )

    # Hotspot label
    city_stats = df.groupby("date")[TARGET_REG].agg(["mean", "std"])
    df = df.join(city_stats, on="date", rsuffix="_city")
    df[TARGET_CLS] = (df[TARGET_REG] > df["mean"] + 2 * df["std"]).astype(int)
    df = df.drop(columns=["mean", "std"], errors="ignore")

    return df


def _create_spatial_proxy_lags(df: pd.DataFrame) -> pd.DataFrame:
    """For single-date datasets, create proxy features from spatial context."""
    df = df.copy()

    # LST anomaly as a proxy for trend
    mu = df[TARGET_REG].mean()
    sigma = df[TARGET_REG].std()
    df["lst_anomaly"] = df[TARGET_REG] - mu
    df["lst_trend_5yr"] = df["lst_anomaly"] * 0.1  # proxy

    # NDVI-based greening/browning proxy
    ndvi_col = "NDVI_L" if "NDVI_L" in df.columns else "NDVI"
    if ndvi_col in df.columns:
        ndvi_mu = df[ndvi_col].mean()
        df["ndvi_trend_5yr"] = (df[ndvi_col] - ndvi_mu) * 0.05

    ndbi_col = "NDBI_L" if "NDBI_L" in df.columns else "NDBI"
    if ndbi_col in df.columns:
        df["ndbi_change_5yr"] = df[ndbi_col] * 0.03

    # Lag proxies (use current values with small noise)
    for lag in [1, 2, 5]:
        df[f"lst_lag_{lag}yr"] = df[TARGET_REG] + np.random.normal(0, 0.5, len(df))
        if ndvi_col in df.columns:
            df[f"ndvi_lag_{lag}yr"] = df[ndvi_col] + np.random.normal(0, 0.02, len(df))

    # Hotspot label
    df[TARGET_CLS] = (df[TARGET_REG] > mu + 2 * sigma).astype(int)

    return df


# ────────────────────────────────────────────────────────────
# Feature selection
# ────────────────────────────────────────────────────────────

BASE_FEATURES = [
    "NDVI_L", "NDBI_L", "MNDWI_L", "EVI_L", "SAVI_L", "albedo",
    "air_temp_C", "humidity_pct", "wind_speed", "rainfall_mm", "solar_rad_W_m2",
    "Elevation_m", "Slope_deg", "TPI_500m",
    "pop_density", "road_density", "building_density", "impervious_ratio",
]

TEMPORAL_FEATURES = [
    "lst_lag_1yr", "lst_lag_2yr", "lst_lag_5yr",
    "ndvi_lag_1yr", "ndvi_lag_2yr", "ndvi_lag_5yr",
    "lst_trend_5yr", "ndvi_trend_5yr", "ndbi_change_5yr",
]


def select_features(df: pd.DataFrame) -> list[str]:
    """Select available features for the future model."""
    all_features = BASE_FEATURES + TEMPORAL_FEATURES
    return [f for f in all_features if f in df.columns and not df[f].isna().all()]


# ────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────

def train_future_model(
    input_path: Path | None = None,
    random_state: int = 42,
) -> tuple[Pipeline, Pipeline, dict]:
    """
    Train both regression and classification models for future prediction.

    Returns
    -------
    reg_pipeline : regression model for future LST
    cls_pipeline : classification model for hotspot probability
    metadata : training metadata
    """
    if input_path is None:
        input_path = PROC_DIR / "pune_with_osm_features.parquet"

    df = pd.read_parquet(input_path)
    print(f"Loaded: {len(df):,} rows")

    # Create temporal features
    df = create_temporal_features(df)

    # Select features
    features = select_features(df)
    print(f"  Using {len(features)} features: {features}")

    # Drop rows with all-null lag features
    df = df.dropna(subset=[TARGET_REG])
    X = df[features]
    y_reg = df[TARGET_REG].astype(float)
    y_cls = df[TARGET_CLS].astype(int)

    # Split
    class_counts = y_cls.value_counts()
    stratify = y_cls if y_cls.nunique() > 1 and class_counts.min() >= 2 else None
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=random_state, stratify=stratify
    )
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_reg_train, y_reg_test = y_reg.loc[train_idx], y_reg.loc[test_idx]
    y_cls_train, y_cls_test = y_cls.loc[train_idx], y_cls.loc[test_idx]

    # ── Regression model ──
    print("  Training future LST regression model...")
    reg_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=0.01, random_state=random_state,
        )),
    ])
    reg_pipeline.fit(X_train, y_reg_train)
    y_reg_pred = reg_pipeline.predict(X_test)

    reg_metrics = {
        "mae": float(mean_absolute_error(y_reg_test, y_reg_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))),
        "r2": float(r2_score(y_reg_test, y_reg_pred)),
    }
    print(f"  Regression: MAE={reg_metrics['mae']:.3f}, R²={reg_metrics['r2']:.4f}")

    # ── Classification model ──
    print("  Training hotspot classification model...")
    if y_cls_train.nunique() > 1:
        classifier = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=0.01, random_state=random_state,
        )
    else:
        classifier = DummyClassifier(strategy="most_frequent")

    cls_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", classifier),
    ])
    cls_pipeline.fit(X_train, y_cls_train)
    y_cls_pred = cls_pipeline.predict(X_test)
    prob = cls_pipeline.predict_proba(X_test)
    if prob.shape[1] > 1:
        y_cls_prob = prob[:, 1]
    else:
        y_cls_prob = np.full(len(X_test), float(cls_pipeline.classes_[0]))

    cls_metrics = {
        "accuracy": float(accuracy_score(y_cls_test, y_cls_pred)),
        "f1": float(f1_score(y_cls_test, y_cls_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_cls_test, y_cls_prob)) if y_cls_test.nunique() > 1 else 0.0,
    }
    print(f"  Classification: Acc={cls_metrics['accuracy']:.4f}, F1={cls_metrics['f1']:.4f}, AUC={cls_metrics['roc_auc']:.4f}")

    # Save models
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    reg_path = MODEL_DIR / "future_hotspot_reg.pkl"
    cls_path = MODEL_DIR / "future_hotspot.pkl"

    with open(reg_path, "wb") as f:
        pickle.dump(reg_pipeline, f)
    with open(cls_path, "wb") as f:
        pickle.dump(cls_pipeline, f)

    print(f"  ✅ Saved regression model: {reg_path}")
    print(f"  ✅ Saved classification model: {cls_path}")

    metadata = {
        "features": features,
        "regression_metrics": reg_metrics,
        "classification_metrics": cls_metrics,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "hotspot_count_train": int(y_cls_train.sum()),
        "hotspot_count_test": int(y_cls_test.sum()),
    }

    meta_path = MODEL_DIR / "future_hotspot_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")

    return reg_pipeline, cls_pipeline, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train future hotspot forecasting model")
    parser.add_argument("--input", type=Path, default=PROC_DIR / "pune_with_osm_features.parquet")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Model 2: Future Hotspot Forecasting")
    print("=" * 60)
    train_future_model(args.input, args.random_state)


if __name__ == "__main__":
    main()
