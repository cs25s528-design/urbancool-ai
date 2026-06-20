#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════╗
# ║  feature_engineering.py                                      ║
# ║  UrbanCool AI — Feature Engineering & Ward Aggregation        ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Index computation, multi-source feature join, ward-level aggregation,
# and temporal lag feature creation for the future forecasting model.

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_DIR / "data" / "processed"
ADMIN_DIR = PROJECT_DIR / "data" / "raw" / "admin"

CRS_UTM = "EPSG:32643"
CRS_WGS84 = "EPSG:4326"


# ────────────────────────────────────────────────────────────
# Spectral index computation
# ────────────────────────────────────────────────────────────

def compute_spectral_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all spectral vegetation/built-up/water indices."""
    df = df.copy()
    eps = 1e-9

    # ── Landsat indices ──
    if "L_NIR" in df.columns and "L_Red" in df.columns:
        df["NDVI"] = (df["L_NIR"] - df["L_Red"]) / (df["L_NIR"] + df["L_Red"] + eps)

    if "L_SWIR1" in df.columns and "L_NIR" in df.columns:
        df["NDBI"] = (df["L_SWIR1"] - df["L_NIR"]) / (df["L_SWIR1"] + df["L_NIR"] + eps)

    if "L_Green" in df.columns and "L_SWIR1" in df.columns:
        df["MNDWI"] = (df["L_Green"] - df["L_SWIR1"]) / (df["L_Green"] + df["L_SWIR1"] + eps)

    if all(c in df.columns for c in ["L_NIR", "L_Red", "L_Blue"]):
        nir, red, blue = df["L_NIR"], df["L_Red"], df["L_Blue"]
        df["EVI"] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
        df["SAVI"] = 1.5 * (nir - red) / (nir + red + 0.5 + eps)

    if "L_SWIR1" in df.columns and "L_Red" in df.columns:
        df["NBI"] = (df["L_Red"] * df["L_SWIR1"]) / (df["L_NIR"] + eps) if "L_NIR" in df.columns else np.nan

    # ── Albedo (Liang 2001) ──
    albedo_cols = ["L_Blue", "L_Red", "L_NIR", "L_SWIR1", "L_SWIR2"]
    if all(c in df.columns for c in albedo_cols):
        df["albedo_computed"] = (
            0.356 * df["L_Blue"]
            + 0.130 * df["L_Red"]
            + 0.373 * df["L_NIR"]
            + 0.085 * df["L_SWIR1"]
            + 0.072 * df["L_SWIR2"]
            - 0.0018
        ).clip(0, 1)

    # ── Sentinel-2 indices ──
    if "S2_NIR" in df.columns and "S2_Red" in df.columns:
        df["NDVI_S2"] = (df["S2_NIR"] - df["S2_Red"]) / (df["S2_NIR"] + df["S2_Red"] + eps)

    if "S2_SWIR1" in df.columns and "S2_NIR" in df.columns:
        df["NDBI_S2"] = (df["S2_SWIR1"] - df["S2_NIR"]) / (df["S2_SWIR1"] + df["S2_NIR"] + eps)

    return df


# ────────────────────────────────────────────────────────────
# Ward boundary join
# ────────────────────────────────────────────────────────────

def join_ward_boundaries(
    df: pd.DataFrame,
    ward_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Spatial join grid points to ward polygons.
    Falls back to nearest-ward assignment for points outside polygons.
    """
    if ward_path is None:
        ward_path = ADMIN_DIR / "pune_wards.geojson"
    ward_path = Path(ward_path)

    if not ward_path.exists():
        print(f"  ⚠️ Ward boundary file not found: {ward_path}")
        df["ward_id"] = None
        df["ward_name"] = None
        df["ward_join_method"] = "none"
        return df

    wards = gpd.read_file(ward_path).to_crs(CRS_UTM)

    # Ensure ward ID columns
    if "ward_id" not in wards.columns:
        wards["ward_id"] = wards.index.astype(str)
    if "ward_name" not in wards.columns:
        name_col = next(
            (c for c in wards.columns if "name" in c.lower()),
            wards.columns[0] if len(wards.columns) > 1 else "ward_id",
        )
        wards["ward_name"] = wards[name_col].astype(str)

    from shapely.geometry import Point
    geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
    grid_gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=CRS_WGS84).to_crs(CRS_UTM)

    # Within-polygon join
    joined = gpd.sjoin(grid_gdf, wards[["ward_id", "ward_name", "geometry"]], how="left", predicate="within")
    within_mask = joined["ward_id"].notna()

    df = df.copy()
    df["ward_id"] = joined["ward_id"].values
    df["ward_name"] = joined["ward_name"].values
    df["ward_join_method"] = np.where(within_mask, "within", "nearest")

    # Nearest fallback for unmatched points
    if (~within_mask).any():
        unmatched = grid_gdf.loc[~within_mask]
        nearest_join = gpd.sjoin_nearest(
            unmatched[["geometry"]],
            wards[["ward_id", "ward_name", "geometry"]],
            how="left",
            distance_col="ward_distance_m",
        )
        # Deduplicate
        nearest_join = nearest_join[~nearest_join.index.duplicated(keep="first")]

        for col in ["ward_id", "ward_name"]:
            df.loc[~within_mask, col] = nearest_join[col].values

        df["ward_distance_m"] = np.nan
        df.loc[~within_mask, "ward_distance_m"] = nearest_join["ward_distance_m"].values

    n_within = within_mask.sum()
    n_nearest = (~within_mask).sum()
    print(f"  Ward join: {n_within:,} within, {n_nearest:,} nearest-fallback")

    return df


# ────────────────────────────────────────────────────────────
# Ward-level aggregation
# ────────────────────────────────────────────────────────────

def aggregate_ward_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate grid-level features to ward level.
    Only uses rows with ward_join_method == 'within' for strict matching.
    """
    strict = df[df.get("ward_join_method", pd.Series("within")) == "within"].copy()

    if len(strict) == 0:
        print("  ⚠️ No strict ward matches; using all rows with ward_id")
        strict = df[df["ward_id"].notna()].copy()

    numeric_cols = strict.select_dtypes(include=[np.number]).columns
    skip_cols = {"lon", "lat", "ward_distance_m"}
    agg_cols = [c for c in numeric_cols if c not in skip_cols and c != "lst_celsius"]

    agg_dict = {"lst_celsius": ["mean", "max", "std", "count"]}
    for col in agg_cols:
        agg_dict[col] = "mean"

    ward_agg = strict.groupby(["ward_id", "ward_name"]).agg(agg_dict)
    ward_agg.columns = [
        f"{col}_{stat}" if stat != "mean" else col
        for col, stat in ward_agg.columns
    ]
    ward_agg = ward_agg.rename(columns={
        "lst_celsius_mean": "lst_celsius",
        "lst_celsius_max": "lst_max",
        "lst_celsius_std": "lst_std",
        "lst_celsius_count": "grid_count",
    })

    return ward_agg.reset_index()


# ────────────────────────────────────────────────────────────
# Temporal lag features for future forecasting
# ────────────────────────────────────────────────────────────

def create_temporal_features(
    df: pd.DataFrame,
    lags_yr: list[int] | None = None,
) -> pd.DataFrame:
    """
    Create temporal lag features for future hotspot forecasting.
    Assumes data is sorted by grid_id and date, with 4 seasons/year.
    """
    if lags_yr is None:
        lags_yr = [1, 2, 5]

    df = df.sort_values(["grid_id", "date"]).copy()
    g_lst = df.groupby("grid_id")["lst_celsius"]

    for lag in lags_yr:
        shift_n = lag * 4  # 4 seasons per year
        df[f"lst_lag_{lag}yr"] = g_lst.shift(shift_n)

        if "NDVI" in df.columns:
            df[f"ndvi_lag_{lag}yr"] = df.groupby("grid_id")["NDVI"].shift(shift_n)
        elif "NDVI_L" in df.columns:
            df[f"ndvi_lag_{lag}yr"] = df.groupby("grid_id")["NDVI_L"].shift(shift_n)

    # 5-year rolling trend (slope = °C/season)
    df["lst_trend_5yr"] = g_lst.transform(
        lambda x: x.rolling(20, min_periods=10).apply(
            lambda s: np.polyfit(range(len(s)), s, 1)[0] if len(s) >= 2 else 0,
            raw=False,
        )
    )

    ndvi_col = "NDVI" if "NDVI" in df.columns else "NDVI_L"
    if ndvi_col in df.columns:
        df["ndvi_trend_5yr"] = df.groupby("grid_id")[ndvi_col].transform(
            lambda x: x.rolling(20, min_periods=10).apply(
                lambda s: np.polyfit(range(len(s)), s, 1)[0] if len(s) >= 2 else 0,
                raw=False,
            )
        )

    ndbi_col = "NDBI" if "NDBI" in df.columns else "NDBI_L"
    if ndbi_col in df.columns:
        df["ndbi_change_5yr"] = df.groupby("grid_id")[ndbi_col].transform(
            lambda x: x - x.shift(20)
        )

    # Binary hotspot label: LST > city_mean + 2 * city_std
    if "date" in df.columns:
        city_stats = df.groupby("date")["lst_celsius"].agg(["mean", "std"])
        df = df.join(city_stats, on="date", rsuffix="_city")
        df["hotspot_label"] = (
            df["lst_celsius"] > df["mean"] + 2 * df["std"]
        ).astype(int)
        df = df.drop(columns=["mean", "std"], errors="ignore")
    else:
        # Single-date dataset: use global stats
        mu = df["lst_celsius"].mean()
        sigma = df["lst_celsius"].std()
        df["hotspot_label"] = (df["lst_celsius"] > mu + 2 * sigma).astype(int)

    return df


# ────────────────────────────────────────────────────────────
# Full pipeline
# ────────────────────────────────────────────────────────────

def run_feature_engineering(
    input_path: str | Path,
    output_path: str | Path | None = None,
    ward_path: str | Path | None = None,
    ward_agg_output: str | Path | None = None,
) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = pd.read_parquet(input_path)
    print(f"Loaded: {len(df):,} rows")

    # Compute spectral indices if raw bands present
    df = compute_spectral_indices(df)

    # Ward join
    df = join_ward_boundaries(df, ward_path)

    # Hotspot label (single-date dataset)
    mu = df["lst_celsius"].mean()
    sigma = df["lst_celsius"].std()
    df["hotspot_label"] = (df["lst_celsius"] > mu + 2 * sigma).astype(int)
    df["lst_anomaly"] = df["lst_celsius"] - mu

    print(f"  Hotspot threshold: {mu + 2 * sigma:.2f}°C")
    print(f"  Hotspot count: {df['hotspot_label'].sum():,} / {len(df):,}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"  ✅ Saved features: {output_path}")

    # Ward-level aggregation
    if ward_agg_output and df["ward_id"].notna().any():
        ward_df = aggregate_ward_features(df)
        ward_agg_output = Path(ward_agg_output)
        ward_agg_output.parent.mkdir(parents=True, exist_ok=True)
        ward_df.to_parquet(ward_agg_output, index=False)
        print(f"  ✅ Saved ward aggregates: {ward_agg_output} ({len(ward_df)} wards)")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature engineering pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROC_DIR / "pune_with_osm_features.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROC_DIR / "gee_features.parquet",
    )
    parser.add_argument(
        "--ward-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--ward-agg-output",
        type=Path,
        default=PROC_DIR / "ward_aggregates.parquet",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Feature Engineering Pipeline")
    print("=" * 60)
    run_feature_engineering(args.input, args.output, args.ward_path, args.ward_agg_output)


if __name__ == "__main__":
    main()
