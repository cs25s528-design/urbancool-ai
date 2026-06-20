#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════╗
# ║  gee_to_parquet.py                                           ║
# ║  UrbanCool AI — Ingest GEE CSV Exports into Parquet          ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Reads raw CSV exports from Google Earth Engine and converts them
# to the standardised ML feature Parquet format used downstream.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROC_DIR = PROJECT_DIR / "data" / "processed"

# GEE column → ML schema mapping
COLUMN_ALIASES = {
    "longitude": "lon",
    "latitude": "lat",
    "LST_Corrected_C": "lst_celsius",
    "Temp_C_mean": "air_temp_C",
    "Temp_C_max": "air_temp_C_max",
    "Temp_C_min": "air_temp_C_min",
    "RH_pct": "humidity_pct",
    "Wind_m_s": "wind_speed",
    "Precip_mm": "rainfall_mm",
    "WorldPop_2020": "pop_density",
    "BuiltUp_m2_2020": "BuiltUp_m2",
    "LULC_ESA_2021": "LULC_ESA",
    "solar_rad": "solar_rad_W_m2",
    "avg_rad": "ntl_radiance",
}

# Landsat broadband albedo weights (Liang 2001)
LANDSAT_ALBEDO_BANDS = {
    "L_Blue": 0.356,
    "L_Red": 0.130,
    "L_NIR": 0.373,
    "L_SWIR1": 0.085,
    "L_SWIR2": 0.072,
}
ALBEDO_OFFSET = -0.0018

# Sentinel-2 fallback albedo weights
S2_ALBEDO_BANDS = {
    "S2_Blue": 0.356,
    "S2_Red": 0.130,
    "S2_NIR": 0.373,
    "S2_SWIR1": 0.085,
    "S2_SWIR2": 0.072,
}


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column aliases — only rename when target name is absent."""
    rename_map = {
        src: dst
        for src, dst in COLUMN_ALIASES.items()
        if src in df.columns and dst not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  Renamed columns: {rename_map}")
    return df


def derive_spectral_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Derive NDVI, NDBI, MNDWI, EVI, SAVI from Landsat bands if missing."""
    eps = 1e-9

    # Landsat indices
    if "L_NIR" in df.columns and "L_Red" in df.columns:
        if "NDVI_L" not in df.columns:
            df["NDVI_L"] = (df["L_NIR"] - df["L_Red"]) / (df["L_NIR"] + df["L_Red"] + eps)
        if "NDWI_L" not in df.columns and "L_Green" in df.columns:
            df["NDWI_L"] = (df["L_Green"] - df["L_NIR"]) / (df["L_Green"] + df["L_NIR"] + eps)

    if "L_SWIR1" in df.columns and "L_NIR" in df.columns:
        if "NDBI_L" not in df.columns:
            df["NDBI_L"] = (df["L_SWIR1"] - df["L_NIR"]) / (df["L_SWIR1"] + df["L_NIR"] + eps)
        if "MNDWI_L" not in df.columns and "L_Green" in df.columns:
            df["MNDWI_L"] = (df["L_Green"] - df["L_SWIR1"]) / (df["L_Green"] + df["L_SWIR1"] + eps)

    if "L_NIR" in df.columns and "L_Red" in df.columns and "L_Blue" in df.columns:
        if "EVI_L" not in df.columns:
            nir, red, blue = df["L_NIR"], df["L_Red"], df["L_Blue"]
            df["EVI_L"] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
        if "SAVI_L" not in df.columns:
            df["SAVI_L"] = 1.5 * (df["L_NIR"] - df["L_Red"]) / (df["L_NIR"] + df["L_Red"] + 0.5 + eps)

    # Sentinel-2 indices
    if "S2_NIR" in df.columns and "S2_Red" in df.columns:
        if "NDVI_S2" not in df.columns:
            df["NDVI_S2"] = (df["S2_NIR"] - df["S2_Red"]) / (df["S2_NIR"] + df["S2_Red"] + eps)

    if "S2_SWIR1" in df.columns and "S2_NIR" in df.columns:
        if "NDBI_S2" not in df.columns:
            df["NDBI_S2"] = (df["S2_SWIR1"] - df["S2_NIR"]) / (df["S2_SWIR1"] + df["S2_NIR"] + eps)

    return df


def compute_albedo_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Three-tier albedo fallback: Landsat → Sentinel-2 → per-LULC median."""
    if "albedo" not in df.columns:
        df["albedo"] = np.nan

    # Tier 1: Landsat reflectance
    landsat_cols = list(LANDSAT_ALBEDO_BANDS.keys())
    if all(c in df.columns for c in landsat_cols):
        mask = df["albedo"].isna() & df[landsat_cols].notna().all(axis=1)
        if mask.any():
            est = sum(df.loc[mask, col] * w for col, w in LANDSAT_ALBEDO_BANDS.items())
            df.loc[mask, "albedo"] = (est + ALBEDO_OFFSET).clip(0, 1)
            print(f"  Tier 1 (Landsat albedo): filled {mask.sum():,}")

    # Tier 2: Sentinel-2 reflectance
    s2_cols = list(S2_ALBEDO_BANDS.keys())
    if all(c in df.columns for c in s2_cols):
        mask = df["albedo"].isna() & df[s2_cols].notna().all(axis=1)
        if mask.any():
            est = sum(df.loc[mask, col] * w for col, w in S2_ALBEDO_BANDS.items())
            df.loc[mask, "albedo"] = (est + ALBEDO_OFFSET).clip(0, 1)
            print(f"  Tier 2 (Sentinel-2 albedo): filled {mask.sum():,}")

    # Tier 3: per-LULC median
    mask = df["albedo"].isna()
    if mask.any() and "LULC_ESA" in df.columns:
        medians = df.groupby("LULC_ESA")["albedo"].median()
        df.loc[mask, "albedo"] = df.loc[mask, "LULC_ESA"].map(medians)

    # Tier 3b: global median
    mask = df["albedo"].isna()
    if mask.any():
        gmed = df["albedo"].median()
        if pd.notna(gmed):
            df.loc[mask, "albedo"] = gmed

    return df


def standardise_types(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise dtype for known categorical/id columns."""
    if "LULC_ESA" in df.columns:
        df["LULC_ESA"] = pd.to_numeric(df["LULC_ESA"], errors="coerce").round().astype("Int64")
    for col in ("year", "grid_id", "season"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def ingest_gee_csv(csv_path: str | Path, output_path: str | Path | None = None) -> pd.DataFrame:
    """
    Main ingestion pipeline: CSV → rename → indices → albedo → Parquet.

    Parameters
    ----------
    csv_path : path to the GEE CSV export
    output_path : optional Parquet output path; if None, uses default

    Returns
    -------
    pd.DataFrame with standardised schema
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"GEE CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path.name}: {len(df):,} rows, {df.shape[1]} columns")

    df = rename_columns(df)
    df = standardise_types(df)
    df = derive_spectral_indices(df)
    df = compute_albedo_fallback(df)

    # Drop rows missing the LST target
    if "lst_celsius" in df.columns:
        before = len(df)
        df = df.dropna(subset=["lst_celsius"])
        dropped = before - len(df)
        if dropped:
            print(f"  Dropped {dropped:,} rows missing lst_celsius")

    if output_path is None:
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PROC_DIR / "gee_features.parquet"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  ✅ Saved: {output_path}  ({len(df):,} rows, {df.shape[1]} columns)")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest GEE CSV exports into Parquet")
    parser.add_argument(
        "--csv",
        type=Path,
        default=RAW_DIR / "Pune_ML_Dataset_CSV_2023.csv",
        help="Path to GEE CSV export",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path (default: data/processed/gee_features.parquet)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GEE CSV → Parquet Ingestion Pipeline")
    print("=" * 60)
    ingest_gee_csv(args.csv, args.output)


if __name__ == "__main__":
    main()
