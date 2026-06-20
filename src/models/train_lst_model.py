#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════╗
# ║  train_lst_model.py                                          ║
# ║  UrbanCool AI — Model 1: XGBoost / HistGBR LST Regression    ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Trains the primary Land Surface Temperature regression model.
# Uses the existing processed Parquet from the data pipeline.
# Falls back to scikit-learn HistGradientBoostingRegressor if
# XGBoost is not installed.

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"

TARGET = "lst_celsius"

# Features matching the existing trained model's feature set
FEATURES = [
    # Spectral
    "NDVI_L", "NDBI_L", "MNDWI_L", "EVI_L", "SAVI_L", "albedo",
    # Weather
    "air_temp_C", "air_temp_C_max", "air_temp_C_min",
    "humidity_pct", "wind_speed", "rainfall_mm", "solar_rad_W_m2",
    # Terrain / population
    "Elevation_m", "Slope_deg", "TPI_500m",
    "pop_density", "ntl_radiance", "children_ratio", "elderly_ratio",
    # OSM
    "road_density", "building_density", "impervious_ratio", "dist_road_m",
]

# Features to include if available (park/water may have been fixed)
OPTIONAL_FEATURES = ["dist_park_m", "dist_water_m"]


def load_data(input_path: Path) -> pd.DataFrame:
    """Load and validate the processed Parquet file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_parquet(input_path)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    if df[TARGET].isna().any():
        df = df.dropna(subset=[TARGET])
        print(f"  Dropped rows with null {TARGET}. Remaining: {len(df):,}")

    return df


def select_features(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Select available features and return the feature matrix."""
    available = [f for f in FEATURES if f in df.columns]

    # Add optional features if they have data
    for f in OPTIONAL_FEATURES:
        if f in df.columns and not df[f].isna().all():
            available.append(f)
            print(f"  Including optional feature: {f}")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"  ⚠️ Missing features: {missing}")

    X = df[available].copy()
    return available, X


def make_model(random_state: int = 42) -> tuple[str, object]:
    """Create the best available tree-based regressor."""
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=1000,
            max_depth=7,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
        # Try CUDA first
        try:
            model.set_params(device="cuda")
            return "xgboost_cuda", model
        except Exception:
            model.set_params(device="cpu")
            return "xgboost_cpu", model
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(
            n_estimators=800,
            max_depth=7,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        return "lightgbm", model
    except ImportError:
        pass

    model = HistGradientBoostingRegressor(
        max_iter=600,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=random_state,
    )
    return "sklearn_hist_gradient_boosting", model


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_lst_model(
    input_path: Path | None = None,
    model_output: Path | None = None,
    random_state: int = 42,
) -> tuple[Pipeline, dict]:
    """
    Full training pipeline for the LST regression model.

    Returns
    -------
    pipeline : trained sklearn Pipeline
    metadata : dict with features, metrics, model info
    """
    if input_path is None:
        input_path = PROC_DIR / "pune_with_osm_features.parquet"
    if model_output is None:
        model_output = MODEL_DIR / "lst_xgboost.pkl"

    df = load_data(input_path)
    feature_names, X = select_features(df)
    y = df[TARGET].astype(float)

    # Split: 80/10/10
    train_idx, holdout_idx = train_test_split(
        df.index.to_numpy(), test_size=0.2, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        holdout_idx, test_size=0.5, random_state=random_state
    )

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    print(f"  Split: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")

    backend, regressor = make_model(random_state)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", regressor),
    ])

    print(f"  Training with {backend}...")

    # Fit with early stopping if XGBoost
    try:
        if "xgboost" in backend:
            imputer = SimpleImputer(strategy="median")
            X_train_imp = imputer.fit_transform(X_train)
            X_val_imp = imputer.transform(X_val)
            regressor.fit(
                X_train_imp, y_train,
                eval_set=[(X_val_imp, y_val)],
                verbose=100,
            )
            pipeline = Pipeline([
                ("imputer", imputer),
                ("model", regressor),
            ])
        else:
            pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"  ⚠️ {backend} failed ({e}), falling back to sklearn")
        _, regressor = "sklearn_hist_gradient_boosting", HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.04, max_leaf_nodes=31,
            l2_regularization=0.01, random_state=random_state,
        )
        backend = "sklearn_hist_gradient_boosting"
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", regressor),
        ])
        pipeline.fit(X_train, y_train)

    # Evaluate
    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)
    val_metrics = compute_metrics(y_val, y_val_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    print(f"  Validation: MAE={val_metrics['mae']:.3f}, RMSE={val_metrics['rmse']:.3f}, R²={val_metrics['r2']:.4f}")
    print(f"  Test:       MAE={test_metrics['mae']:.3f}, RMSE={test_metrics['rmse']:.3f}, R²={test_metrics['r2']:.4f}")

    # Save model
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output)
    print(f"  ✅ Saved model: {model_output}")

    # Also save as .pkl for compatibility
    pkl_path = model_output.with_suffix(".pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(pipeline, f)

    metadata = {
        "target": TARGET,
        "features": feature_names,
        "backend": backend,
        "input": str(input_path),
        "row_count": len(df),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
        },
    }

    # Save metadata
    meta_path = model_output.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")

    # Save predictions
    predictions = pd.DataFrame({
        "lon": pd.concat([df.loc[val_idx, "lon"], df.loc[test_idx, "lon"]]),
        "lat": pd.concat([df.loc[val_idx, "lat"], df.loc[test_idx, "lat"]]),
        TARGET: pd.concat([y_val, y_test]),
        "pred_lst_celsius": np.concatenate([y_val_pred, y_test_pred]),
        "split": ["val"] * len(val_idx) + ["test"] * len(test_idx),
    })
    predictions["residual"] = predictions[TARGET] - predictions["pred_lst_celsius"]
    pred_path = PROC_DIR / "grid_lst_test_predictions.parquet"
    predictions.to_parquet(pred_path, index=False)

    return pipeline, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LST regression model")
    parser.add_argument("--input", type=Path, default=PROC_DIR / "pune_with_osm_features.parquet")
    parser.add_argument("--model-out", type=Path, default=MODEL_DIR / "lst_xgboost.pkl")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Model 1: LST Regression Training")
    print("=" * 60)
    train_lst_model(args.input, args.model_out, args.random_state)


if __name__ == "__main__":
    main()
