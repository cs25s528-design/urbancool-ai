#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════╗
# ║  recommender.py                                              ║
# ║  UrbanCool AI — Model 6: Rule + Score Recommendation Engine  ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Composite intervention scoring and rule-based recommendation.
# Score = 0.35*ΔT + 0.20*HVI + 0.15*pop_benefit + 0.15*cost_inv + 0.15*feasibility

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_DIR = Path(__file__).resolve().parents[2]

# Intervention types and their characteristics
INTERVENTION_CATALOG = {
    "trees": {
        "label": "Tree Cover / Urban Forest",
        "cost_inr_m2": 550,
        "typical_delta_T": 2.2,
        "lifespan_yr": 40,
        "requires": {"open_land": True, "low_ndvi": True},
        "color": "#2e7d32",
    },
    "cool_roofs": {
        "label": "Cool Reflective Roofs",
        "cost_inr_m2": 1050,
        "typical_delta_T": 3.2,
        "lifespan_yr": 12,
        "requires": {"high_building": True},
        "color": "#0288d1",
    },
    "reflective_pavement": {
        "label": "Reflective / Permeable Pavement",
        "cost_inr_m2": 800,
        "typical_delta_T": 1.7,
        "lifespan_yr": 25,
        "requires": {"high_road": True},
        "color": "#fbc02d",
    },
    "blue_green": {
        "label": "Blue-Green Infrastructure",
        "cost_inr_m2": 2200,
        "typical_delta_T": 2.8,
        "lifespan_yr": 35,
        "requires": {"far_water": True},
        "color": "#00838f",
    },
    "combined": {
        "label": "Combined (Trees + Cool Roofs)",
        "cost_inr_m2": 3500,
        "typical_delta_T": 4.5,
        "lifespan_yr": 30,
        "requires": {"high_hvi": True},
        "color": "#7b1fa2",
    },
}


def check_feasibility(row: pd.Series | dict, intervention: str) -> float:
    """
    Compute feasibility score (0-1) for an intervention at a grid/ward location.

    Higher = more feasible/appropriate.
    """
    config = INTERVENTION_CATALOG.get(intervention, {})
    requires = config.get("requires", {})
    score = 0.5  # default neutral

    ndvi = _get(row, "NDVI_L", _get(row, "NDVI", 0.3))
    ndbi = _get(row, "NDBI_L", _get(row, "NDBI", 0.0))
    building = _get(row, "building_density", 0.3)
    road = _get(row, "road_density", 5.0)
    dist_water = _get(row, "dist_water_m", 500.0)
    hvi = _get(row, "HVI", 0.5)

    if requires.get("open_land") or requires.get("low_ndvi"):
        # Trees work best in low-NDVI areas with some open space
        score = max(0, 1.0 - ndvi) * (1 - building * 0.5)

    if requires.get("high_building"):
        # Cool roofs need buildings
        score = building

    if requires.get("high_road"):
        # Reflective pavement needs roads
        score = min(1.0, road / 20.0)

    if requires.get("far_water"):
        # Blue-green infra is most beneficial far from water
        score = min(1.0, dist_water / 2000.0)

    if requires.get("high_hvi"):
        # Combined intervention prioritised for high-vulnerability areas
        score = hvi

    return float(np.clip(score, 0, 1))


def compute_intervention_score(
    row: pd.Series | dict,
    intervention: str,
    delta_T: float | None = None,
    hvi: float | None = None,
) -> float:
    """
    Compute composite intervention score for a location-intervention pair.

    Score = 0.35*ΔT_norm + 0.20*HVI + 0.15*pop_benefit + 0.15*cost_inv + 0.15*feasibility
    """
    config = INTERVENTION_CATALOG.get(intervention, {})

    if delta_T is None:
        delta_T = config.get("typical_delta_T", 2.0)
    delta_T_norm = min(delta_T / 6.0, 1.0)  # normalise to ~6°C max

    if hvi is None:
        hvi = _get(row, "HVI", 0.5)

    pop = _get(row, "pop_density", 100)
    pop_benefit = min(1.0, np.log1p(pop) / np.log1p(5000))

    cost = config.get("cost_inr_m2", 1000)
    cost_inv = 1.0 - min(cost / 5000.0, 1.0)  # lower cost → higher score

    feasibility = check_feasibility(row, intervention)

    score = (
        0.35 * delta_T_norm
        + 0.20 * hvi
        + 0.15 * pop_benefit
        + 0.15 * cost_inv
        + 0.15 * feasibility
    )
    return float(np.clip(score, 0, 1))


def recommend_intervention(
    row: pd.Series | dict,
    delta_T_per_intervention: dict[str, float] | None = None,
    n_recommendations: int = 3,
) -> list[dict]:
    """
    Recommend the best interventions for a location.

    Parameters
    ----------
    row : feature values
    delta_T_per_intervention : optional pre-computed temperature reductions
    n_recommendations : number of recommendations to return

    Returns
    -------
    Sorted list of dicts: [{'intervention': str, 'score': float, ...}, ...]
    """
    results = []

    for name, config in INTERVENTION_CATALOG.items():
        delta_T = (
            delta_T_per_intervention.get(name, config["typical_delta_T"])
            if delta_T_per_intervention
            else config["typical_delta_T"]
        )

        score = compute_intervention_score(row, name, delta_T)
        feasibility = check_feasibility(row, name)

        results.append({
            "intervention": name,
            "label": config["label"],
            "score": round(score, 4),
            "delta_T_C": round(delta_T, 2),
            "cost_inr_m2": config["cost_inr_m2"],
            "lifespan_yr": config["lifespan_yr"],
            "feasibility": round(feasibility, 3),
            "color": config["color"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:n_recommendations]


def rule_based_recommendation(row: pd.Series | dict) -> dict:
    """
    Simple rule-based intervention recommendation.
    Returns the single best intervention based on conditions.
    """
    ndvi = _get(row, "NDVI_L", _get(row, "NDVI", 0.3))
    building = _get(row, "building_density", 0.3)
    road = _get(row, "road_density", 5.0)
    dist_water = _get(row, "dist_water_m", 500.0)
    hvi = _get(row, "HVI", 0.5)
    lst = _get(row, "lst_celsius", 38.0)

    # Decision rules from planning.tex
    if hvi > 0.75 and lst > 42:
        return {
            "intervention": "combined",
            "reason": "High HVI + extreme LST → combined priority intervention",
            **INTERVENTION_CATALOG["combined"],
        }
    elif ndvi < 0.2 and building < 0.4:
        return {
            "intervention": "trees",
            "reason": "Low NDVI + open land → tree cover / pocket park",
            **INTERVENTION_CATALOG["trees"],
        }
    elif building > 0.5:
        return {
            "intervention": "cool_roofs",
            "reason": "High building density → cool reflective roofs",
            **INTERVENTION_CATALOG["cool_roofs"],
        }
    elif road > 15:
        return {
            "intervention": "reflective_pavement",
            "reason": "High road density → reflective / permeable pavement",
            **INTERVENTION_CATALOG["reflective_pavement"],
        }
    elif dist_water > 1500:
        return {
            "intervention": "blue_green",
            "reason": "Far from water body → blue-green infrastructure",
            **INTERVENTION_CATALOG["blue_green"],
        }
    else:
        return {
            "intervention": "trees",
            "reason": "Default recommendation: urban tree planting",
            **INTERVENTION_CATALOG["trees"],
        }


def _get(row, key, default=0):
    """Safely get a value from a Series or dict."""
    if isinstance(row, dict):
        return row.get(key, default)
    elif hasattr(row, "get"):
        val = row.get(key, default)
        if pd.isna(val):
            return default
        return val
    return default
