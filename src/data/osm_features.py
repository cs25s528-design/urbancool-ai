#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════╗
# ║  osm_features.py                                             ║
# ║  UrbanCool AI — OSM Building, Road, Park, Water Features     ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Extracts urban morphology features from OpenStreetMap using OSMnx.
# Produces: road_density, building_density, impervious_ratio,
#           dist_road_m, dist_park_m, dist_water_m

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_DIR / "data" / "processed"
CRS_UTM = "EPSG:32643"  # WGS84 / UTM Zone 43N (Pune)
CRS_WGS84 = "EPSG:4326"


def create_grid_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Create GeoDataFrame from lon/lat columns."""
    geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CRS_WGS84)
    return gdf


def download_osm_features(bounds: tuple, timeout: int = 600) -> dict:
    """
    Download roads, buildings, parks, and water features from OSM.

    Parameters
    ----------
    bounds : (south, west, north, east) bounding box in WGS84
    timeout : Overpass API timeout in seconds

    Returns
    -------
    dict with keys: 'roads', 'buildings', 'parks', 'water'
    """
    try:
        import osmnx as ox
        ox.settings.timeout = timeout
        ox.settings.use_cache = True
    except ImportError:
        raise ImportError("osmnx is required: pip install osmnx")

    south, west, north, east = bounds
    bbox = (north, south, east, west)  # osmnx uses (N, S, E, W)

    result = {}

    # Roads
    try:
        G = ox.graph_from_bbox(bbox=bbox, network_type="all")
        result["roads"] = ox.graph_to_gdfs(G, nodes=False).to_crs(CRS_UTM)
        print(f"  Roads: {len(result['roads']):,} edges")
    except Exception as e:
        print(f"  ⚠️ Roads download failed: {e}")
        result["roads"] = None

    # Buildings
    try:
        bld = ox.features_from_bbox(bbox=bbox, tags={"building": True})
        bld = bld[bld.geometry.type.isin(["Polygon", "MultiPolygon"])]
        result["buildings"] = bld.to_crs(CRS_UTM)
        print(f"  Buildings: {len(result['buildings']):,} polygons")
    except Exception as e:
        print(f"  ⚠️ Buildings download failed: {e}")
        result["buildings"] = None

    # Parks / green spaces
    try:
        parks = ox.features_from_bbox(
            bbox=bbox,
            tags={"leisure": ["park", "garden", "nature_reserve", "playground"]},
        )
        parks = parks[parks.geometry.type.isin(["Polygon", "MultiPolygon", "Point"])]
        result["parks"] = parks.to_crs(CRS_UTM)
        print(f"  Parks: {len(result['parks']):,} features")
    except Exception as e:
        print(f"  ⚠️ Parks download failed: {e}")
        result["parks"] = None

    # Water bodies
    try:
        water = ox.features_from_bbox(
            bbox=bbox,
            tags={"natural": ["water"], "waterway": True},
        )
        result["water"] = water.to_crs(CRS_UTM)
        print(f"  Water: {len(result['water']):,} features")
    except Exception as e:
        print(f"  ⚠️ Water download failed: {e}")
        result["water"] = None

    return result


def compute_building_density(
    grid_utm: gpd.GeoDataFrame,
    buildings_utm: gpd.GeoDataFrame | None,
    buffer_m: float = 150.0,
) -> pd.Series:
    """Compute building footprint area fraction within buffer of each grid point."""
    if buildings_utm is None or len(buildings_utm) == 0:
        return pd.Series(0.0, index=grid_utm.index, name="building_density")

    buffered = grid_utm.geometry.buffer(buffer_m)
    buffer_area = np.pi * buffer_m**2

    densities = []
    sindex = buildings_utm.sindex

    for idx, buf in buffered.items():
        candidates = list(sindex.intersection(buf.bounds))
        if not candidates:
            densities.append(0.0)
            continue
        clipped = buildings_utm.iloc[candidates].intersection(buf)
        total_area = clipped.area.sum()
        densities.append(min(total_area / buffer_area, 1.0))

    return pd.Series(densities, index=grid_utm.index, name="building_density")


def compute_road_density(
    grid_utm: gpd.GeoDataFrame,
    roads_utm: gpd.GeoDataFrame | None,
    buffer_m: float = 150.0,
) -> pd.Series:
    """Compute road length (km) per km² within buffer of each grid point."""
    if roads_utm is None or len(roads_utm) == 0:
        return pd.Series(0.0, index=grid_utm.index, name="road_density")

    buffered = grid_utm.geometry.buffer(buffer_m)
    buffer_area_km2 = np.pi * (buffer_m / 1000) ** 2

    densities = []
    sindex = roads_utm.sindex

    for idx, buf in buffered.items():
        candidates = list(sindex.intersection(buf.bounds))
        if not candidates:
            densities.append(0.0)
            continue
        clipped = roads_utm.iloc[candidates].intersection(buf)
        total_length_km = clipped.length.sum() / 1000
        densities.append(total_length_km / buffer_area_km2)

    return pd.Series(densities, index=grid_utm.index, name="road_density")


def compute_nearest_distance(
    grid_utm: gpd.GeoDataFrame,
    features_utm: gpd.GeoDataFrame | None,
    col_name: str,
    max_dist_m: float = 50000.0,
) -> pd.Series:
    """Compute nearest distance from each grid point to a set of features."""
    if features_utm is None or len(features_utm) == 0:
        return pd.Series(np.nan, index=grid_utm.index, name=col_name)

    try:
        joined = gpd.sjoin_nearest(
            grid_utm[["geometry"]],
            features_utm[["geometry"]],
            how="left",
            max_distance=max_dist_m,
            distance_col="distance",
        )
        # sjoin_nearest may produce duplicates; keep the nearest
        distances = joined.groupby(joined.index)["distance"].min()
        return distances.reindex(grid_utm.index).rename(col_name)
    except Exception:
        # Fallback: brute-force nearest
        from shapely.ops import nearest_points

        union = features_utm.geometry.unary_union
        dists = grid_utm.geometry.apply(
            lambda pt: pt.distance(nearest_points(pt, union)[1])
        )
        return dists.clip(upper=max_dist_m).rename(col_name)


def extract_urban_features(
    df: pd.DataFrame,
    osm_data: dict | None = None,
    buffer_m: float = 150.0,
) -> pd.DataFrame:
    """
    Full urban feature extraction pipeline.

    Parameters
    ----------
    df : DataFrame with lon/lat columns
    osm_data : pre-downloaded OSM features (dict), or None to download
    buffer_m : buffer radius for density computations

    Returns
    -------
    DataFrame with added columns: road_density, building_density,
    impervious_ratio, dist_road_m, dist_park_m, dist_water_m
    """
    df = df.copy()
    gdf = create_grid_gdf(df)
    grid_utm = gdf.to_crs(CRS_UTM)

    if osm_data is None:
        bounds = (
            df["lat"].min() - 0.01,
            df["lon"].min() - 0.01,
            df["lat"].max() + 0.01,
            df["lon"].max() + 0.01,
        )
        print("Downloading OSM features...")
        osm_data = download_osm_features(bounds)

    print("Computing building density...")
    df["building_density"] = compute_building_density(
        grid_utm, osm_data.get("buildings"), buffer_m
    ).values

    print("Computing road density...")
    df["road_density"] = compute_road_density(
        grid_utm, osm_data.get("roads"), buffer_m
    ).values

    # Impervious ratio proxy
    df["impervious_ratio"] = (
        df["building_density"] + df["road_density"].clip(0, 1) * 0.15
    ).clip(0, 1)

    print("Computing distance to roads...")
    df["dist_road_m"] = compute_nearest_distance(
        grid_utm, osm_data.get("roads"), "dist_road_m"
    ).values

    print("Computing distance to parks...")
    df["dist_park_m"] = compute_nearest_distance(
        grid_utm, osm_data.get("parks"), "dist_park_m"
    ).values

    print("Computing distance to water...")
    df["dist_water_m"] = compute_nearest_distance(
        grid_utm, osm_data.get("water"), "dist_water_m"
    ).values

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OSM urban features")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROC_DIR / "pune_base_features.parquet",
        help="Input Parquet file with lon/lat",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROC_DIR / "pune_with_osm_features.parquet",
        help="Output Parquet path",
    )
    parser.add_argument("--buffer-m", type=float, default=150.0)
    args = parser.parse_args()

    print("=" * 60)
    print("OSM Urban Feature Extraction")
    print("=" * 60)

    df = pd.read_parquet(args.input)
    print(f"Loaded {args.input.name}: {len(df):,} rows")

    df = extract_urban_features(df, buffer_m=args.buffer_m)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"✅ Saved: {args.output}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
