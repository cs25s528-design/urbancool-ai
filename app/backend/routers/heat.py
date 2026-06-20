from __future__ import annotations

from fastapi import APIRouter, Query

from ..ml_service.predict import current_heat, load_feature_table, load_metrics, top_drivers

router = APIRouter(prefix="/v1/heat", tags=["heat"])


@router.get("/current")
def current(lat: float = Query(...), lon: float = Query(...)):
    return current_heat(lat, lon)


@router.get("/history")
def history(lat: float = Query(...), lon: float = Query(...), years: int = 5):
    df = load_feature_table()
    row = current_heat(lat, lon)
    city = df["lst_celsius"].describe().to_dict()
    return {"query": {"lat": lat, "lon": lon, "years": years}, "nearest_current": row, "city_lst_summary": city}


@router.get("/future")
def future(lat: float = Query(...), lon: float = Query(...), horizon: int = 2030):
    now = current_heat(lat, lon)
    return {
        "query": {"lat": lat, "lon": lon, "horizon": horizon},
        "no_action_lst_C": now["pred_lst_C"],
        "hotspot_prob": 1.0 if now["risk_level"] in {"high", "very_high"} else 0.35,
        "risk_class": now["risk_level"],
        "model_metrics": load_metrics().get("metrics", {}),
    }


@router.get("/drivers")
def drivers(lat: float = Query(...), lon: float = Query(...), n: int = 5):
    return top_drivers(lat, lon, n=n)


@router.get("/samples")
def samples(limit: int = Query(900, ge=50, le=3000)):
    df = load_feature_table()
    cols = [c for c in ["lat", "lon", "lst_celsius", "air_temp_C", "NDVI_L", "NDBI_L", "albedo", "road_density"] if c in df.columns]
    sample = df[cols].dropna(subset=["lat", "lon", "lst_celsius"])
    if len(sample) > limit:
        sample = sample.sample(n=limit, random_state=42)
    records = sample.to_dict(orient="records")
    return {
        "count": len(records),
        "total_rows": int(len(df)),
        "lst_min": float(df["lst_celsius"].min()),
        "lst_max": float(df["lst_celsius"].max()),
        "lst_mean": float(df["lst_celsius"].mean()),
        "points": records,
    }
