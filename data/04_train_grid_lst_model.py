from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT = PROJECT_DIR / "data" / "processed" / "pune_with_osm_features.parquet"
MODEL_DIR = PROJECT_DIR / "models"
PROC_DIR = PROJECT_DIR / "data" / "processed"

MODEL_OUT = MODEL_DIR / "grid_lst_model.joblib"
FEATURES_OUT = MODEL_DIR / "grid_lst_features.json"
METRICS_OUT = PROC_DIR / "grid_lst_metrics.json"
PREDICTIONS_OUT = PROC_DIR / "grid_lst_eval_predictions.parquet"

TARGET = "lst_celsius"
EXPECTED_ROWS = 38244
RANDOM_STATE = 42
TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 0.10

CORE_FEATURES = [
    # Spectral
    "NDVI_L",
    "NDBI_L",
    "MNDWI_L",
    "EVI_L",
    "SAVI_L",
    "albedo",
    # Weather
    "air_temp_C",
    "air_temp_C_max",
    "air_temp_C_min",
    "humidity_pct",
    "wind_speed",
    "rainfall_mm",
    "solar_rad_W_m2",
    # Terrain / population
    "Elevation_m",
    "Slope_deg",
    "TPI_500m",
    "pop_density",
    "ntl_radiance",
    "children_ratio",
    "elderly_ratio",
    # OSM
    "road_density",
    "building_density",
    "impervious_ratio",
    "dist_road_m",
]

EXCLUDED_COLUMNS = {
    "dist_park_m",
    "dist_water_m",
    "ward_id",
    "ward_name",
    "ward_join_method",
    "ward_distance_m",
    "grid_id",
    ".geo",
    "system:index",
    TARGET,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train grid-level Pune LST regression model."
    )
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--model-out", type=Path, default=MODEL_OUT)
    parser.add_argument("--features-out", type=Path, default=FEATURES_OUT)
    parser.add_argument("--metrics-out", type=Path, default=METRICS_OUT)
    parser.add_argument("--predictions-out", type=Path, default=PREDICTIONS_OUT)
    parser.add_argument("--train-size", type=float, default=TRAIN_SIZE)
    parser.add_argument("--val-size", type=float, default=VAL_SIZE)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument(
        "--skip-permutation-importance",
        action="store_true",
        help="Skip permutation importance for faster smoke tests.",
    )
    return parser.parse_args()


def validate_split_sizes(train_size: float, val_size: float, test_size: float) -> None:
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(
            "Train/validation/test split sizes must sum to 1.0. "
            f"Got {train_size:.4f} + {val_size:.4f} + {test_size:.4f} = {total:.4f}."
        )
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("Train/validation/test split sizes must all be positive.")


def train_val_test_split_indices(
    index: np.ndarray,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_split_sizes(train_size, val_size, test_size)
    train_idx, holdout_idx = train_test_split(
        index,
        test_size=val_size + test_size,
        random_state=random_state,
    )
    relative_test_size = test_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=relative_test_size,
        random_state=random_state,
    )
    return train_idx, val_idx, test_idx


def load_dataset(path: Path, expected_rows: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input parquet not found: {path}")

    df = pd.read_parquet(path)
    if len(df) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows:,} rows in {path.name}, found {len(df):,}."
        )
    if TARGET not in df.columns:
        raise AssertionError(f"Missing target column: {TARGET}")
    if df[TARGET].isna().any():
        raise AssertionError(f"Target {TARGET} contains null values.")
    return df


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    unavailable = [col for col in CORE_FEATURES if col not in df.columns]
    candidate_features = [col for col in CORE_FEATURES if col in df.columns]

    forbidden_selected = sorted(set(candidate_features) & EXCLUDED_COLUMNS)
    if forbidden_selected:
        raise AssertionError(f"Excluded columns selected as features: {forbidden_selected}")

    non_numeric = [
        col for col in candidate_features
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise AssertionError(f"Non-numeric training features selected: {non_numeric}")

    all_null = [col for col in candidate_features if df[col].isna().all()]
    features = [col for col in candidate_features if col not in all_null]
    if not features:
        raise AssertionError("No usable numeric features selected.")

    explicitly_excluded_present = sorted(
        col for col in EXCLUDED_COLUMNS if col in df.columns and col != TARGET
    )
    for required_exclusion in ("dist_park_m", "dist_water_m"):
        if required_exclusion in df.columns and required_exclusion in features:
            raise AssertionError(f"{required_exclusion} must be excluded from first model.")

    X = df[features].copy()
    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        raise AssertionError("Training feature matrix must be numeric only.")
    if X.isna().all().any():
        bad = X.columns[X.isna().all()].tolist()
        raise AssertionError(f"Fully-null features remained after filtering: {bad}")

    metadata = {
        "target": TARGET,
        "features": features,
        "unavailable_core_features": unavailable,
        "dropped_all_null_features": all_null,
        "excluded_columns_present": explicitly_excluded_present,
    }
    return X, metadata


def make_model(random_state: int) -> tuple[str, object]:
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            random_state=random_state,
            n_jobs=-1,
        )
        return "xgboost_cuda_requested", model
    except Exception:
        model = HistGradientBoostingRegressor(
            max_iter=600,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=random_state,
        )
        return "sklearn_hist_gradient_boosting", model


def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> tuple[str, Pipeline]:
    backend, regressor = make_model(random_state)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", regressor),
        ]
    )

    try:
        pipeline.fit(X_train, y_train)
        return backend, pipeline
    except Exception as exc:
        if backend != "xgboost_cuda_requested":
            raise

        from xgboost import XGBRegressor

        print(f"CUDA XGBoost failed; falling back to CPU XGBoost. Reason: {exc}")
        pipeline.set_params(
            model=XGBRegressor(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                tree_method="hist",
                device="cpu",
                random_state=random_state,
                n_jobs=-1,
            )
        )
        pipeline.fit(X_train, y_train)
        return "xgboost_cpu_fallback", pipeline


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_prediction_frame(
    df: pd.DataFrame,
    indices: np.ndarray,
    y_true: pd.Series,
    y_pred: np.ndarray,
    split_name: str,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "split": split_name,
            "source_index": indices,
            "lon": df.loc[indices, "lon"].to_numpy(),
            "lat": df.loc[indices, "lat"].to_numpy(),
            TARGET: y_true.to_numpy(),
            "pred_lst_celsius": y_pred,
        }
    )
    out["residual"] = out[TARGET] - out["pred_lst_celsius"]
    return out


def top_importances(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: list[str],
    skip_permutation: bool,
    random_state: int,
) -> list[dict]:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_, dtype=float)
    elif skip_permutation:
        return []
    else:
        result = permutation_importance(
            pipeline,
            X_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=3,
            random_state=random_state,
            n_jobs=-1,
        )
        scores = result.importances_mean

    order = np.argsort(scores)[::-1]
    return [
        {"feature": features[i], "importance": float(scores[i])}
        for i in order[:20]
    ]


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input, args.expected_rows)
    X, feature_metadata = build_feature_frame(df)
    y = df[TARGET].astype(float)

    train_idx, val_idx, test_idx = train_val_test_split_indices(
        df.index.to_numpy(),
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    X_train, X_val, X_test = X.loc[train_idx], X.loc[val_idx], X.loc[test_idx]
    y_train, y_val, y_test = y.loc[train_idx], y.loc[val_idx], y.loc[test_idx]

    backend, pipeline = fit_model(X_train, y_train, args.random_state)
    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)

    val_metrics = compute_metrics(y_val, y_val_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)
    importances = top_importances(
        pipeline,
        X_val,
        y_val,
        feature_metadata["features"],
        args.skip_permutation_importance,
        args.random_state,
    )

    predictions = pd.concat(
        [
            make_prediction_frame(df, val_idx, y_val, y_val_pred, "val"),
            make_prediction_frame(df, test_idx, y_test, y_test_pred, "test"),
        ],
        ignore_index=True,
    )
    expected_prediction_rows = len(X_val) + len(X_test)
    if len(predictions) != expected_prediction_rows:
        raise AssertionError(
            "Prediction row count does not match validation + test row count."
        )

    artifact = {
        **feature_metadata,
        "input": str(args.input),
        "model_backend": backend,
        "row_count": int(len(df)),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "train_size": float(args.train_size),
        "val_size": float(args.val_size),
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
        },
        "top_importances": importances,
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, args.model_out)
    args.features_out.write_text(json.dumps(feature_metadata, indent=2) + "\n")
    args.metrics_out.write_text(json.dumps(artifact, indent=2) + "\n")
    predictions.to_parquet(args.predictions_out, index=False)

    print("Grid LST model training complete")
    print(f"Input rows: {len(df):,}")
    print(f"Split rows: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")
    print(f"Features used: {len(feature_metadata['features'])}")
    print(f"Model backend: {backend}")
    print(
        "Validation metrics: "
        f"MAE={val_metrics['mae']:.3f}, "
        f"RMSE={val_metrics['rmse']:.3f}, "
        f"R2={val_metrics['r2']:.3f}"
    )
    print(
        "Test metrics: "
        f"MAE={test_metrics['mae']:.3f}, "
        f"RMSE={test_metrics['rmse']:.3f}, "
        f"R2={test_metrics['r2']:.3f}"
    )
    print(f"Saved model: {args.model_out}")
    print(f"Saved feature list: {args.features_out}")
    print(f"Saved metrics: {args.metrics_out}")
    print(f"Saved validation/test predictions: {args.predictions_out}")
    if importances:
        print("Top features:")
        for item in importances[:10]:
            print(f"  {item['feature']}: {item['importance']:.6f}")


if __name__ == "__main__":
    main()
