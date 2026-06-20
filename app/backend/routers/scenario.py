from __future__ import annotations

from pydantic import BaseModel
from fastapi import APIRouter

from ..ml_service.predict import compare_scenarios

router = APIRouter(prefix="/v1/scenario", tags=["scenario"])


class ScenarioRequest(BaseModel):
    lat: float
    lon: float
    intervention: str | None = None
    interventions: list[str] | None = None
    intensity: float = 1.0


@router.post("/simulate")
def simulate(req: ScenarioRequest):
    interventions = req.interventions
    if interventions is None and req.intervention:
        interventions = [req.intervention]
    result = compare_scenarios(req.lat, req.lon, interventions)
    result["intensity"] = req.intensity
    return result
