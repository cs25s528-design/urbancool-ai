from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, JSONResponse

from ..ml_service.predict import recommend

PROJECT_DIR = Path(__file__).resolve().parents[3]
MAP_PATH = PROJECT_DIR / "data" / "outputs" / "pune_intervention_map.geojson"

router = APIRouter(prefix="/v1/intervention", tags=["intervention"])


@router.get("/map/pune")
def pune_map():
    if MAP_PATH.exists():
        return FileResponse(MAP_PATH, media_type="application/geo+json")
    return JSONResponse({"type": "FeatureCollection", "features": [], "note": "intervention map not generated yet"})


@router.get("/recommend")
def recommendation(lat: float = Query(...), lon: float = Query(...)):
    return recommend(lat, lon)

