#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════╗
# ║  hvi_calculator.py                                           ║
# ║  UrbanCool AI — Model 5: Heat Vulnerability Index             ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Computes the Heat Vulnerability Index (HVI) per ward using:
# HVI = 0.35*H + 0.25*P + 0.15*A + 0.15*G + 0.10*I
#
# H = normalised heat severity (LST)
# P = normalised population density
# A = elderly + children ratio
# G = green access deficit (1 - NDVI, normalised)
# I = low-income proxy (inverted nighttime-light rank)

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_DIR / "data" / "processed"

# HVI component weights
WEIGHTS = {
    "heat": 0.35,
    "population": 0.25,
    "age": 0.15,
    "green_deficit": 0.15,
    "income_proxy": 0.10,
}


def compute_hvi(
    df: pd.DataFrame,
    lst_col: str = "lst_celsius",
    pop_col: str = "pop_density",
    elderly_col: str = "elderly_ratio",
    children_col: str = "children_ratio",
    ndvi_col: str = "NDVI_L",
    ntl_col: str = "ntl_radiance",
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    Compute Heat Vulnerability Index for each row (grid or ward level).

    Parameters
    ----------
    df : DataFrame with feature columns
    lst_col : column name for LST values
    pop_col : column name for population density
    elderly_col : column name for elderly ratio
    children_col : column name for children ratio
    ndvi_col : column name for NDVI
    ntl_col : column name for nighttime light radiance
    weights : optional custom weights (default: planning.tex formula)

    Returns
    -------
    DataFrame with added HVI columns
    """
    if weights is None:
        weights = WEIGHTS

    df = df.copy()
    scaler = MinMaxScaler()

    # H: Heat severity (normalised LST)
    if lst_col in df.columns:
        valid = df[lst_col].notna()
        df.loc[valid, "H_norm"] = scaler.fit_transform(
            df.loc[valid, [lst_col]]
        ).flatten()
    else:
        df["H_norm"] = 0.5

    # P: Population density (log-scaled, normalised)
    if pop_col in df.columns:
        valid = df[pop_col].notna()
        log_pop = np.log1p(df.loc[valid, pop_col].clip(lower=0))
        df.loc[valid, "P_norm"] = scaler.fit_transform(
            log_pop.values.reshape(-1, 1)
        ).flatten()
    else:
        df["P_norm"] = 0.5

    # A: Age vulnerability (elderly + children ratio)
    has_elderly = elderly_col in df.columns
    has_children = children_col in df.columns
    if has_elderly and has_children:
        df["A_raw"] = (
            df[elderly_col].fillna(0) + df[children_col].fillna(0)
        )
        valid = df["A_raw"].notna()
        df.loc[valid, "A_norm"] = scaler.fit_transform(
            df.loc[valid, ["A_raw"]]
        ).flatten()
        df = df.drop(columns=["A_raw"])
    elif has_elderly:
        valid = df[elderly_col].notna()
        df.loc[valid, "A_norm"] = scaler.fit_transform(
            df.loc[valid, [elderly_col]]
        ).flatten()
    elif has_children:
        valid = df[children_col].notna()
        df.loc[valid, "A_norm"] = scaler.fit_transform(
            df.loc[valid, [children_col]]
        ).flatten()
    else:
        df["A_norm"] = 0.5

    # G: Green access deficit (1 - NDVI, normalised)
    if ndvi_col in df.columns:
        green_deficit = 1 - df[ndvi_col].clip(-1, 1).fillna(0)
        valid = green_deficit.notna()
        df.loc[valid, "G_norm"] = scaler.fit_transform(
            green_deficit.loc[valid].values.reshape(-1, 1)
        ).flatten()
    else:
        df["G_norm"] = 0.5

    # I: Low-income proxy (inverted nighttime-light rank)
    if ntl_col in df.columns and df[ntl_col].notna().any():
        # Lower nighttime lights → higher vulnerability
        inverse_ntl = 1.0 / (df[ntl_col].clip(lower=0.001) + 0.001)
        valid = inverse_ntl.notna()
        df.loc[valid, "I_norm"] = scaler.fit_transform(
            inverse_ntl.loc[valid].values.reshape(-1, 1)
        ).flatten()
    else:
        df["I_norm"] = 0.5

    # Fill any remaining NaN in components
    for comp in ["H_norm", "P_norm", "A_norm", "G_norm", "I_norm"]:
        df[comp] = df[comp].fillna(0.5)

    # Compute HVI
    df["HVI"] = (
        weights["heat"] * df["H_norm"]
        + weights["population"] * df["P_norm"]
        + weights["age"] * df["A_norm"]
        + weights["green_deficit"] * df["G_norm"]
        + weights["income_proxy"] * df["I_norm"]
    ).clip(0, 1)

    # Classify
    df["hvi_class"] = pd.cut(
        df["HVI"],
        bins=[0, 0.25, 0.50, 0.75, 1.01],
        labels=["Low", "Medium", "High", "Very High"],
        include_lowest=True,
    )

    return df


def compute_ward_hvi(
    grid_df: pd.DataFrame,
    ward_col: str = "ward_name",
    ward_method_col: str = "ward_join_method",
    strict_only: bool = False,
) -> pd.DataFrame:
    """
    Aggregate grid-level features to ward level and compute HVI.

    Parameters
    ----------
    grid_df : grid-level DataFrame with ward assignments
    ward_col : column identifying wards
    ward_method_col : column indicating join method
    strict_only : if True, use only 'within' join matches

    Returns
    -------
    Ward-level DataFrame with HVI scores
    """
    df = grid_df.copy()

    if strict_only and ward_method_col in df.columns:
        df = df[df[ward_method_col] == "within"]
        print(f"  Strict ward matching: {len(df):,} rows")

    if ward_col not in df.columns or df[ward_col].isna().all():
        print("  ⚠️ No ward column available; computing grid-level HVI")
        return compute_hvi(df)

    # Aggregate to ward level
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    agg_cols = [c for c in numeric_cols if c not in ("lon", "lat")]

    ward_df = df.groupby(ward_col)[agg_cols].agg("mean").reset_index()

    # Add ward area and population
    if "pop_density" in df.columns:
        ward_pop = df.groupby(ward_col)["pop_density"].sum().reset_index()
        ward_pop.columns = [ward_col, "total_population"]
        ward_df = ward_df.merge(ward_pop, on=ward_col, how="left")

    # Count grid cells per ward
    ward_counts = df.groupby(ward_col).size().reset_index(name="grid_count")
    ward_df = ward_df.merge(ward_counts, on=ward_col, how="left")

    # Compute HVI
    ward_df = compute_hvi(ward_df)

    # Sort by HVI descending
    ward_df = ward_df.sort_values("HVI", ascending=False).reset_index(drop=True)

    return ward_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Heat Vulnerability Index")
    parser.add_argument(
        "--input", type=Path,
        default=PROC_DIR / "pune_with_osm_wards.parquet",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROC_DIR / "ward_hvi.parquet",
    )
    parser.add_argument("--strict", action="store_true", help="Use only strict ward matches")
    args = parser.parse_args()

    print("=" * 60)
    print("Model 5: Heat Vulnerability Index")
    print("=" * 60)

    df = pd.read_parquet(args.input)
    print(f"Loaded: {len(df):,} rows")

    # Grid-level HVI
    df = compute_hvi(df)
    print(f"\n  Grid-level HVI distribution:")
    print(f"    {df['hvi_class'].value_counts().to_dict()}")

    # Ward-level HVI
    if "ward_name" in df.columns and df["ward_name"].notna().any():
        ward_df = compute_ward_hvi(df, strict_only=args.strict)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ward_df.to_parquet(args.output, index=False)
        print(f"\n  ✅ Ward HVI saved: {args.output} ({len(ward_df)} wards)")
        print("\n  Top 10 vulnerable wards:")
        display_cols = ["ward_name", "HVI", "hvi_class"]
        if "lst_celsius" in ward_df.columns:
            display_cols.insert(2, "lst_celsius")
        print(ward_df[display_cols].head(10).to_string(index=False))
    else:
        print("  ⚠️ No ward information available; saving grid-level HVI only")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.output, index=False)

    print(f"\n  Mean HVI: {df['HVI'].mean():.3f}")
    print(f"  Max HVI: {df['HVI'].max():.3f}")


if __name__ == "__main__":
    main()
