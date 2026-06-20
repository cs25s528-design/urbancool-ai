from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

try:
    import osmnx as ox
except ImportError as exc:
    raise SystemExit(
        "osmnx is required for OSM feature generation. "
        "Install it with: pip install osmnx"
    ) from exc


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = PROJECT_DIR / "data" / "processed"
RAW_DIR = PROJECT_DIR / "data" / "raw"
CACHE_DIR = PROJECT_DIR / "cache" / "osmnx"

INPUT = PROC_DIR / "pune_base_features.parquet"
OUTPUT = PROC_DIR / "pune_with_osm_features.parquet"
WARD_FILE = RAW_DIR / "admin" / "pune_wards.geojson"

UTM_CRS = "EPSG:32643"
BUFFER_M = 250

def empty_gdf(crs="EPSG:4326"):
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)


def configure_osmnx():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(CACHE_DIR)
    ox.settings.timeout = 300


def require_columns(df, columns):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {INPUT}: {missing}")


def clean_gdf(gdf, crs="EPSG:4326", geom_types=None):
    if gdf is None or gdf.empty or "geometry" not in gdf.columns:
        return empty_gdf(crs)

    out = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if out.empty:
        return empty_gdf(crs)

    if out.crs is None:
        out = out.set_crs(crs)
    else:
        out = out.to_crs(crs)

    if geom_types is not None:
        out = out[out.geom_type.isin(geom_types)].copy()

    return out if not out.empty else empty_gdf(crs)


def load_points():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Run data/01_schema_align_and_albedo.py first."
        )

    df = pd.read_parquet(INPUT)
    require_columns(df, ["lon", "lat"])

    df = df.copy()
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

    bad_coords = df[["lon", "lat"]].isna().any(axis=1).sum()
    if bad_coords:
        raise ValueError(f"{INPUT} has {bad_coords:,} rows with invalid lon/lat values.")

    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )


def study_polygon_from_data(points):
    if WARD_FILE.exists():
        wards = clean_gdf(gpd.read_file(WARD_FILE), geom_types=["Polygon", "MultiPolygon"])
        if not wards.empty:
            boundary = geometry_union(wards)
            if boundary is not None:
                print(f"Using ward boundary for OSM download: {WARD_FILE}")
                return boundary
        print(f"Ward boundary file exists but has no usable polygons: {WARD_FILE}")

    minx, miny, maxx, maxy = points.total_bounds
    boundary = box(minx, miny, maxx, maxy).buffer(0.02)
    print("Using point bounding box for OSM download.")
    return boundary


def geometry_union(gdf):
    if gdf.empty:
        return None
    try:
        geom = gdf.geometry.union_all()
    except AttributeError:
        geom = unary_union(gdf.geometry)
    return None if geom is None or geom.is_empty else geom


def fetch_osm_features(polygon, tags, label):
    features_from_polygon = getattr(
        ox,
        "features_from_polygon",
        getattr(ox, "geometries_from_polygon", None),
    )
    if features_from_polygon is None:
        raise RuntimeError("This osmnx version has no polygon feature download API.")

    try:
        gdf = features_from_polygon(polygon, tags=tags)
        gdf = clean_gdf(gdf)
        print(f"Downloaded {label}: {len(gdf):,} features")
        return gdf
    except Exception as exc:
        print(f"{label} download failed: {exc}")
        return empty_gdf()


def fetch_roads(polygon):
    try:
        graph = ox.graph_from_polygon(polygon, network_type="drive")
        roads = ox.graph_to_gdfs(graph, nodes=False)
        roads = clean_gdf(roads, geom_types=["LineString", "MultiLineString"])
        print(f"Downloaded roads: {len(roads):,} edges")
        return roads
    except Exception as exc:
        print(f"Road download failed: {exc}")
        return empty_gdf()


def project_layers(*layers):
    projected = []
    for layer in layers:
        projected.append(layer.to_crs(UTM_CRS) if not layer.empty else layer)
    return projected


def nearest_distances(points, features, distance_col):
    """Compute point-to-nearest-feature distance with a spatial index."""
    if features.empty:
        return pd.Series(np.nan, index=points.index, dtype="float64")

    left = gpd.GeoDataFrame(
        {"_row_id": np.arange(len(points), dtype=np.int64)},
        geometry=points.geometry.values,
        crs=points.crs,
    )
    right = gpd.GeoDataFrame(
        geometry=features.geometry.reset_index(drop=True),
        crs=features.crs,
    )

    joined = gpd.sjoin_nearest(left, right, how="left", distance_col=distance_col)
    nearest = joined.groupby("_row_id", sort=False)[distance_col].min()

    out = pd.Series(np.nan, index=np.arange(len(points)), dtype="float64")
    out.loc[nearest.index] = nearest.to_numpy()
    out.index = points.index
    return out


def intersecting_rows(features, buffer_geom):
    if features.empty:
        return features
    try:
        idx = features.sindex.query(buffer_geom, predicate="intersects")
        return features.iloc[idx]
    except Exception:
        return features[features.intersects(buffer_geom)]


def road_density_for_buffer(roads, buffer_geom, buffer_area_km2):
    if roads.empty:
        return np.nan

    clipped = intersecting_rows(roads, buffer_geom)
    if clipped.empty:
        return 0.0

    length_km = clipped.geometry.intersection(buffer_geom).length.sum() / 1000
    return length_km / buffer_area_km2


def building_density_for_buffer(buildings, buffer_geom, buffer_area_m2):
    if buildings.empty:
        return np.nan

    clipped = intersecting_rows(buildings, buffer_geom)
    if clipped.empty:
        return 0.0

    area_m2 = clipped.geometry.intersection(buffer_geom).area.sum()
    return min(area_m2 / buffer_area_m2, 1.0)


def main():
    configure_osmnx()
    points = load_points()
    study_polygon = study_polygon_from_data(points)

    print("Downloading OSM data for Pune area...")
    roads = fetch_roads(study_polygon)
    buildings = fetch_osm_features(study_polygon, {"building": True}, "buildings")
    parks = fetch_osm_features(
        study_polygon,
        {"leisure": ["park", "garden"], "landuse": ["recreation_ground", "grass"]},
        "parks",
    )
    water = fetch_osm_features(study_polygon, {"natural": "water"}, "water")

    buildings = clean_gdf(buildings, geom_types=["Polygon", "MultiPolygon"])
    parks = clean_gdf(parks)
    water = clean_gdf(water)

    grid, roads, buildings, parks, water = project_layers(
        points, roads, buildings, parks, water
    )

    grid["buffer_geom"] = grid.geometry.buffer(BUFFER_M)
    buffer_area_m2 = np.pi * BUFFER_M * BUFFER_M
    buffer_area_km2 = buffer_area_m2 / 1e6

    print("Computing distances with spatial-index nearest joins...")
    grid["dist_road_m"] = nearest_distances(grid, roads, "dist_road_m")
    grid["dist_park_m"] = nearest_distances(grid, parks, "dist_park_m")
    grid["dist_water_m"] = nearest_distances(grid, water, "dist_water_m")

    print("Computing road and building density...")
    grid["road_density"] = grid["buffer_geom"].apply(
        lambda geom: road_density_for_buffer(roads, geom, buffer_area_km2)
    )
    grid["building_density"] = grid["buffer_geom"].apply(
        lambda geom: building_density_for_buffer(buildings, geom, buffer_area_m2)
    )

    grid["impervious_ratio"] = (
        grid["building_density"].fillna(0)
        + 0.03 * grid["road_density"].fillna(0)
    ).clip(0, 1)

    out = pd.DataFrame(
        grid.drop(columns=["geometry", "buffer_geom"], errors="ignore")
    )
    out.to_parquet(OUTPUT, index=False)

    print("Saved:", OUTPUT)
    preview_cols = [
        "road_density",
        "building_density",
        "impervious_ratio",
        "dist_water_m",
        "dist_park_m",
        "dist_road_m",
    ]
    print(out[preview_cols].head())


if __name__ == "__main__":
    main()
