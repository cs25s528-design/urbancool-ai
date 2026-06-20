#!/usr/bin/env python3
"""Export ward intervention plans as GeoJSON for Leaflet/MapLibre/QGIS."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

try:
    from .nsga2_optimizer import INTERVENTION_LABELS
except ImportError:  # Allows direct execution: python src/optimization/intervention_map_export.py
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.optimization.nsga2_optimizer import INTERVENTION_LABELS

PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_ADMIN = PROJECT_DIR / "data" / "raw" / "admin"
OUTPUT_DIR = PROJECT_DIR / "data" / "outputs"

COLORS = {
    0: "#e0e0e0",
    1: "#2e7d32",
    2: "#0288d1",
    3: "#fbc02d",
    4: "#00838f",
}


def export_intervention_map(plan_path: Path, ward_geojson: Path, output: Path) -> gpd.GeoDataFrame:
    if not plan_path.exists():
        raise FileNotFoundError(plan_path)
    if not ward_geojson.exists():
        raise FileNotFoundError(ward_geojson)
    plan = pd.read_csv(plan_path)
    wards = gpd.read_file(ward_geojson).to_crs("EPSG:4326")
    if "ward_id" not in wards.columns:
        wards["ward_id"] = range(1, len(wards) + 1)
    wards["ward_id"] = wards["ward_id"].astype(str)
    plan["ward_id"] = plan["ward_id"].astype(str)
    out = wards.merge(plan, on="ward_id", how="left")
    out["intervention_type"] = out["intervention_type"].fillna(0).astype(int)
    out["intervention_label"] = out["intervention_type"].map(INTERVENTION_LABELS)
    out["map_color"] = out["intervention_type"].map(COLORS)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(output, driver="GeoJSON")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=OUTPUT_DIR / "ga_best_plan.csv")
    parser.add_argument("--wards", type=Path, default=RAW_ADMIN / "pune_wards.geojson")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "pune_intervention_map.geojson")
    args = parser.parse_args()
    gdf = export_intervention_map(args.plan, args.wards, args.output)
    print(f"Saved intervention map: {args.output} ({len(gdf):,} polygons)")


if __name__ == "__main__":
    main()
