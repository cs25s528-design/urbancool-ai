# ╔══════════════════════════════════════════════════════════════╗
# ║  01_schema_align_and_albedo.py                                 ║
# ║  UrbanCool AI — Pune Urban Heat Data Pipeline                  ║
# ║  Step 1 of 3 (Python, run after the GEE export)                ║
# ╚══════════════════════════════════════════════════════════════╝
#
# INPUT  : data/raw/Pune_ML_Dataset_CSV_<year>.csv         (required)
#          data/raw/Pune_ML_HotSeason_CSV_<year>.csv       (optional)
# OUTPUT : data/processed/pune_base_features.parquet       -> 02_add_osm_features.py
#          data/processed/pune_hotseason_features.parquet  (optional)
#
# WHAT THIS SCRIPT DOES
#   1. Loads the raw GEE CSV export(s)
#   2. Schema alignment:
#        - longitude/latitude -> lon/lat   (required by 02 & 03)
#        - LULC_ESA           -> nullable Int64 (categorical land-cover code)
#        - year/grid_id/season -> string dtype
#        - reports any expected columns that are missing / unexpected
#   3. Albedo fallback (3-tier):
#        Tier 1: native Landsat albedo (Liang 2001) where present
#        Tier 2: recompute from Sentinel-2 bands (S2-adapted Liang weights)
#        Tier 3: per-LULC-class median, then global median for stragglers
#   4. Null audit:
#        - prints a null-count report for every column
#        - drops rows missing the LST target (can't train/eval on these)
#        - imputes remaining numeric feature nulls (per-LULC median ->
#          global median)
#
# pip install pandas numpy pyarrow

import numpy as np
import pandas as pd
from pathlib import Path

# ────────────────────────────────────────────────────────────
# 0. CONFIG
# ────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR  = PROJECT_DIR / "data" / "raw"
PROC_DIR = PROJECT_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

YEAR = "2023"   # must match PARAMS.START_DATE year in the GEE script

ANNUAL_CSV = RAW_DIR / f"Pune_ML_Dataset_CSV_{YEAR}.csv"
HOT_CSV    = RAW_DIR / f"Pune_ML_HotSeason_CSV_{YEAR}.csv"   # optional

OUTPUT_ANNUAL = PROC_DIR / "pune_base_features.parquet"
OUTPUT_HOT    = PROC_DIR / "pune_hotseason_features.parquet"

TARGET_COLS = ["lst_celsius"]

# GEE exports have used both the planning names and the final model names over
# time. Rename only when the final column is not already present, so rerunning on
# an already-aligned export is idempotent.
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

# Columns expected from the GEE export (Model A + B feature set).
# Used only for a missing/extra-column sanity check.
EXPECTED_FEATURE_COLS = [
    # target
    "lst_celsius", "emissivity",
    # Landsat reflectance + indices + albedo
    "L_Blue", "L_Green", "L_Red", "L_NIR", "L_SWIR1", "L_SWIR2",
    "NDVI_L", "NDWI_L", "MNDWI_L", "NDBI_L", "EVI_L", "SAVI_L", "NBI_L",
    "albedo",
    # Sentinel-2 reflectance + indices
    "S2_Blue", "S2_Green", "S2_Red", "S2_NIR", "S2_SWIR1", "S2_SWIR2",
    "NDVI_S2", "NDWI_S2", "MNDWI_S2", "NDBI_S2", "EVI_S2", "SAVI_S2", "NBI_S2",
    # weather
    "air_temp_C", "air_temp_C_max", "air_temp_C_min",
    "rainfall_mm", "wind_speed", "humidity_pct",
    # population
    "pop_density", "GHSL_Pop_2020", "GPW_PopDensity",
    # built-up + terrain
    "BuiltUp_m2", "Elevation_m", "Slope_deg", "Aspect_deg", "TPI_500m",
    # LULC label
    "LULC_ESA",
    # Model B
    "solar_rad_W_m2", "ntl_radiance", "children_ratio", "elderly_ratio",
]

LANDSAT_ALBEDO_BANDS = {
    "L_Blue":  0.356,
    "L_Red":   0.130,
    "L_NIR":   0.373,
    "L_SWIR1": 0.085,
    "L_SWIR2": 0.072,
}

# Sentinel-2 bands + weights used for the Tier-2 albedo fallback. These are
# spectrally close enough for a fallback estimate, not a replacement for native
# Landsat albedo when Landsat bands are available.
S2_ALBEDO_BANDS = {
    "S2_Blue":  0.356,
    "S2_Red":   0.130,
    "S2_NIR":   0.373,
    "S2_SWIR1": 0.085,
    "S2_SWIR2": 0.072,
}
S2_ALBEDO_OFFSET = -0.0018


# ────────────────────────────────────────────────────────────
# 1. LOAD
# ────────────────────────────────────────────────────────────

def load_csv(path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    print(f"Loaded {path.name}: {len(df):,} rows, {len(df.columns)} cols")
    return df


# ────────────────────────────────────────────────────────────
# 2. SCHEMA ALIGNMENT
# ────────────────────────────────────────────────────────────

def align_schema(df, label):
    df = df.copy()

    rename_map = {
        src: dst
        for src, dst in COLUMN_ALIASES.items()
        if src in df.columns and dst not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  Renamed columns: {rename_map}")

    if "LULC_ESA" in df.columns:
        df["LULC_ESA"] = pd.to_numeric(df["LULC_ESA"], errors="coerce").round().astype("Int64")

    for col in ("year", "grid_id", "season"):
        if col in df.columns:
            df[col] = df[col].astype("string")

    required = ["lon", "lat", "lst_celsius"]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"{label}: missing required columns after schema alignment: "
            f"{missing_required}"
        )

    # Sanity check vs. expected feature columns
    missing = [c for c in EXPECTED_FEATURE_COLS if c not in df.columns]
    extra = [
        c for c in df.columns
        if c not in EXPECTED_FEATURE_COLS
        and c not in ("grid_id", "lon", "lat", "year", "season",
                       "longitude", "latitude", "system:index", ".geo")
    ]
    if missing:
        print(f"  ⚠️  {label}: missing expected columns: {missing}")
    if extra:
        print(f"  ℹ️  {label}: extra columns not in schema: {extra}")

    return df


# ────────────────────────────────────────────────────────────
# 3. ALBEDO FALLBACK (3-tier)
# ────────────────────────────────────────────────────────────

def fill_albedo(df, label):
    df = df.copy()
    if "albedo" not in df.columns:
        df["albedo"] = np.nan
        print(f"  {label}: no 'albedo' column — deriving from available bands.")

    n_total = len(df)
    n_null = df["albedo"].isna().sum()
    print(f"\n  Albedo nulls before fallback: {n_null:,} / {n_total:,} "
          f"({100 * n_null / max(n_total, 1):.2f}%)")
    if n_null == 0:
        return df

    # Tier 1 — Landsat reflectance fallback
    landsat_cols = list(LANDSAT_ALBEDO_BANDS.keys())
    if all(c in df.columns for c in landsat_cols):
        mask = df["albedo"].isna() & df[landsat_cols].notna().all(axis=1)
        if mask.any():
            est = pd.Series(0.0, index=df.index[mask])
            for col, weight in LANDSAT_ALBEDO_BANDS.items():
                est = est + df.loc[mask, col] * weight
            df.loc[mask, "albedo"] = (est + S2_ALBEDO_OFFSET).clip(0, 1)
            print(f"  Tier 1 (Landsat fallback): filled {mask.sum():,}")
    else:
        print("  Tier 1 skipped — Landsat albedo bands not all present.")

    # Tier 2 — Sentinel-2 reflectance fallback
    s2_cols = list(S2_ALBEDO_BANDS.keys())
    if all(c in df.columns for c in s2_cols):
        mask = df["albedo"].isna() & df[s2_cols].notna().all(axis=1)
        if mask.any():
            est = pd.Series(0.0, index=df.index[mask])
            for col, weight in S2_ALBEDO_BANDS.items():
                est = est + df.loc[mask, col] * weight
            df.loc[mask, "albedo"] = (est + S2_ALBEDO_OFFSET).clip(0, 1)
            print(f"  Tier 2 (Sentinel-2 fallback): filled {mask.sum():,}")
    else:
        print("  Tier 2 skipped — Sentinel-2 albedo bands not all present.")

    # Tier 3 — per-LULC-class median
    mask = df["albedo"].isna()
    if mask.any() and "LULC_ESA" in df.columns:
        before = mask.sum()
        medians = df.groupby("LULC_ESA")["albedo"].median()
        df.loc[mask, "albedo"] = df.loc[mask, "LULC_ESA"].map(medians)
        print(f"  Tier 3 (per-LULC median): filled {before - df['albedo'].isna().sum():,}")

    # Tier 3b — global median for any stragglers (e.g. classes with no
    # valid albedo pixels at all)
    mask = df["albedo"].isna()
    if mask.any():
        gmed = df["albedo"].median()
        if pd.notna(gmed):
            df.loc[mask, "albedo"] = gmed
            print(f"  Tier 3b (global median {gmed:.4f}): filled {mask.sum():,}")
        else:
            print(f"  ⚠️  {label}: albedo remains null; no valid fallback values found.")

    print(f"  Albedo nulls after fallback: {df['albedo'].isna().sum():,}")
    return df


# ────────────────────────────────────────────────────────────
# 4. NULL AUDIT + CLEAN
# ────────────────────────────────────────────────────────────

def null_audit_and_clean(df, label):
    df = df.copy()
    print(f"\n  Null report ({label}):")
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    if nulls.empty:
        print("    none")
    else:
        for col, n in nulls.items():
            print(f"    {col:<20s} {n:>7,} ({100 * n / len(df):5.2f}%)")

    # Hard drop: rows missing the LST target — can't train/eval on these
    target_present = [c for c in TARGET_COLS if c in df.columns]
    if target_present:
        n_before = len(df)
        df = df.dropna(subset=target_present)
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"  Dropped {n_dropped:,} rows missing target {target_present}")

    if "LULC_ESA" in df.columns and df["LULC_ESA"].isna().any():
        n_null = df["LULC_ESA"].isna().sum()
        mode = df["LULC_ESA"].mode(dropna=True)
        if not mode.empty:
            df["LULC_ESA"] = df["LULC_ESA"].fillna(mode.iloc[0]).astype("Int64")
            print(f"  Imputed LULC_ESA: {n_null:,} nulls -> modal class {mode.iloc[0]}")
        else:
            print(f"  ⚠️  LULC_ESA has {n_null:,} nulls and no modal class to impute")

    # Soft impute remaining numeric feature nulls
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skip = set(TARGET_COLS) | {"lon", "lat", "longitude", "latitude", "LULC_ESA"}
    for col in numeric_cols:
        if col in skip:
            continue
        n_null = df[col].isna().sum()
        if n_null == 0:
            continue
        if "LULC_ESA" in df.columns:
            medians = df.groupby("LULC_ESA")[col].median()
            df[col] = df[col].fillna(df["LULC_ESA"].map(medians))
        if df[col].isna().any():
            gmed = df[col].median()
            if pd.notna(gmed):
                df[col] = df[col].fillna(gmed)
        if df[col].isna().any():
            print(f"  ⚠️  {col}: {df[col].isna().sum():,} nulls remain after imputation")
        else:
            print(f"  Imputed {col}: {n_null:,} nulls -> per-LULC/global median")

    return df


# ────────────────────────────────────────────────────────────
# 5. PIPELINE
# ────────────────────────────────────────────────────────────

def process(csv_path, out_path, label):
    df = load_csv(csv_path)
    if df is None:
        return None
    print(f"\n── {label} ──")
    df = align_schema(df, label)
    df = fill_albedo(df, label)
    df = null_audit_and_clean(df, label)
    df.to_parquet(out_path, index=False)
    print(f"  ✅ Saved: {out_path}  ({len(df):,} rows, {len(df.columns)} cols)")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Schema alignment + albedo fallback + null audit")
    print("=" * 60)

    annual = process(ANNUAL_CSV, OUTPUT_ANNUAL, f"annual {YEAR}")
    if annual is None:
        raise FileNotFoundError(
            f"Expected {ANNUAL_CSV}. Download Pune_ML_Dataset_CSV_{YEAR}.csv "
            f"from the GEE Drive export folder into data/raw/."
        )

    hot = process(HOT_CSV, OUTPUT_HOT, f"hot-season {YEAR}")
    if hot is None:
        print(f"\nℹ️  {HOT_CSV} not found — skipping optional hot-season dataset.")

    print("\nNext step:")
    print("  python data/02_add_osm_features.py")
