from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = PROJECT_DIR / "data" / "processed"
RAW_DIR = PROJECT_DIR / "data" / "raw"

INPUT = PROC_DIR / "pune_with_osm_features.parquet"
OUTPUT = PROC_DIR / "pune_with_osm_wards.parquet"
DEFAULT_WARD_FILE = RAW_DIR / "admin" / "pune_wards.geojson"
WARD_DISTANCE_CRS = "EPSG:32643"

POSSIBLE_ID_COLS = [
    "ward_id",
    "WARD_ID",
    "ward_no",
    "Ward_No",
    "prabhag_no",
    "Prabhag_No",
    "id",
    "ID",
]
POSSIBLE_NAME_COLS = [
    "ward_name",
    "WARD_NAME",
    "name",
    "Name",
    "prabhag_name",
    "Prabhag_Name",
]


def find_ward_file():
    if DEFAULT_WARD_FILE.exists():
        return DEFAULT_WARD_FILE

    candidates = []
    for pattern in ("*ward*.geojson", "*Ward*.geojson", "*ward*.shp", "*Ward*.shp"):
        candidates.extend(RAW_DIR.glob(pattern))
        candidates.extend((RAW_DIR / "admin").glob(pattern) if (RAW_DIR / "admin").exists() else [])

    return sorted(candidates)[0] if candidates else None


def load_input():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run data/02_add_osm_features.py first."
        )

    df = pd.read_parquet(INPUT)
    missing = [col for col in ("lon", "lat") if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required coordinate columns in {INPUT}: {missing}")

    df = df.copy()
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    bad_coords = df[["lon", "lat"]].isna().any(axis=1).sum()
    if bad_coords:
        raise ValueError(f"{INPUT} has {bad_coords:,} rows with invalid lon/lat values.")

    return df


def pick_column(columns, candidates):
    exact = set(columns)
    by_lower = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in exact:
            return candidate
        match = by_lower.get(candidate.lower())
        if match is not None:
            return match
    return None


def normalize_wards(path):
    wards = gpd.read_file(path)
    if wards.empty or "geometry" not in wards.columns:
        raise ValueError(f"Ward file has no usable geometries: {path}")

    wards = wards[wards.geometry.notna() & ~wards.geometry.is_empty].copy()
    if wards.empty:
        raise ValueError(f"Ward file has only empty geometries: {path}")

    if wards.crs is None:
        wards = wards.set_crs("EPSG:4326")
    else:
        wards = wards.to_crs("EPSG:4326")

    ward_id_col = pick_column(wards.columns, POSSIBLE_ID_COLS)
    ward_name_col = pick_column(wards.columns, POSSIBLE_NAME_COLS)

    if ward_id_col is None:
        wards["ward_id"] = np.arange(1, len(wards) + 1)
        ward_id_col = "ward_id"

    if ward_name_col is None:
        wards["ward_name"] = wards[ward_id_col].astype("string")
        ward_name_col = "ward_name"

    wards_small = wards[[ward_id_col, ward_name_col, "geometry"]].rename(
        columns={ward_id_col: "ward_id", ward_name_col: "ward_name"}
    )
    wards_small["ward_id"] = wards_small["ward_id"].astype("string")
    wards_small["ward_name"] = wards_small["ward_name"].astype("string")
    return wards_small


def write_without_wards(df):
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["ward_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["ward_name"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["ward_join_method"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["ward_distance_m"] = np.nan
    out.to_parquet(OUTPUT, index=False)
    print(f"Ward file not found under {RAW_DIR}; saved nullable ward columns.")
    print("Saved:", OUTPUT)


def fill_nearest_wards(joined, grid, wards):
    missing_mask = joined["ward_id"].isna()
    missing_count = int(missing_mask.sum())
    if missing_count == 0:
        joined["ward_join_method"] = "within"
        joined["ward_distance_m"] = 0.0
        return joined

    print(f"Warning: {missing_count:,} rows did not match a ward polygon.")
    print("Filling unmatched rows with nearest ward; check ward_distance_m before ward-level analysis.")

    grid_m = grid.to_crs(WARD_DISTANCE_CRS)
    wards_m = wards.to_crs(WARD_DISTANCE_CRS)

    missing_rows = joined.loc[missing_mask, ["_row_id"]].copy()
    missing_points = grid_m.loc[missing_rows["_row_id"].to_numpy(), ["geometry"]].copy()
    missing_points["_row_id"] = missing_rows["_row_id"].to_numpy()

    nearest = gpd.sjoin_nearest(
        missing_points,
        wards_m[["ward_id", "ward_name", "geometry"]],
        how="left",
        distance_col="ward_distance_m",
    )
    nearest = nearest.sort_values("ward_distance_m").drop_duplicates("_row_id", keep="first")
    nearest = nearest.set_index("_row_id")

    fill_index = joined.loc[missing_mask, "_row_id"]
    joined.loc[missing_mask, "ward_id"] = fill_index.map(nearest["ward_id"]).astype("string")
    joined.loc[missing_mask, "ward_name"] = fill_index.map(nearest["ward_name"]).astype("string")
    joined["ward_join_method"] = "within"
    joined.loc[missing_mask, "ward_join_method"] = "nearest"
    joined.loc[missing_mask, "ward_distance_m"] = fill_index.map(nearest["ward_distance_m"]).to_numpy()
    joined.loc[~missing_mask, "ward_distance_m"] = 0.0

    remaining = int(joined["ward_id"].isna().sum())
    if remaining:
        print(f"Warning: {remaining:,} rows still have no ward after nearest fallback.")
    else:
        print("Nearest ward fallback filled all unmatched rows.")

    distances = joined.loc[joined["ward_join_method"] == "nearest", "ward_distance_m"]
    if not distances.empty:
        print(
            "Nearest fallback distance summary (m): "
            f"median={distances.median():.1f}, "
            f"p95={distances.quantile(0.95):.1f}, "
            f"max={distances.max():.1f}"
        )

    return joined


def main():
    df = load_input()
    ward_file = find_ward_file()
    if ward_file is None:
        write_without_wards(df)
        return

    grid = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    grid["_row_id"] = np.arange(len(grid))

    wards = normalize_wards(ward_file)
    print("Using ward file:", ward_file)
    print("Ward columns:", wards.columns.tolist())

    joined = gpd.sjoin(grid, wards, how="left", predicate="intersects")

    duplicate_rows = joined.duplicated("_row_id").sum()
    if duplicate_rows:
        print(f"Dropping {duplicate_rows:,} duplicate spatial-join rows.")
        joined = joined.drop_duplicates("_row_id", keep="first")

    joined = joined.sort_values("_row_id")
    joined["ward_distance_m"] = np.nan
    joined = fill_nearest_wards(joined, grid, wards)

    out = pd.DataFrame(
        joined.drop(columns=["geometry", "index_right", "_row_id"], errors="ignore")
    )
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT, index=False)

    print("Saved:", OUTPUT)
    print(out[["ward_id", "ward_name"]].head())


if __name__ == "__main__":
    main()
