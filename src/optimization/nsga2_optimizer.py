#!/usr/bin/env python3
"""NSGA-II optimization for ward-level cooling interventions.

This module is written to work with the current project state:
ward-level optimization is only valid after a clean matched ward table exists.
It therefore operates on generic ward feature arrays/dataframes and does not
assume a fixed ward count.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "data" / "outputs"

INTERVENTION_LABELS = {
    0: "No Action",
    1: "Tree Cover / Urban Forest",
    2: "Cool Roofs",
    3: "Reflective / Permeable Pavement",
    4: "Blue-Green Infrastructure",
}

COSTS_INR_M2 = {
    0: 0.0,
    1: 600.0,
    2: 1100.0,
    3: 800.0,
    4: 2200.0,
}

FEATURE_DELTAS = {
    0: {},
    1: {"NDVI_L": 0.15, "NDBI_L": -0.05, "albedo": 0.03, "impervious_ratio": -0.08},
    2: {"albedo": 0.25, "NDBI_L": -0.02},
    3: {"albedo": 0.15, "impervious_ratio": -0.02},
    4: {"NDVI_L": 0.10, "MNDWI_L": 0.12, "albedo": 0.06, "impervious_ratio": -0.05},
}


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.allclose(values, 0):
        return 0.0
    values = np.sort(np.clip(values, 0, None))
    n = values.size
    return float((2 * np.arange(1, n + 1) @ values) / (n * values.sum()) - (n + 1) / n)


def load_ward_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ward feature table not found: {path}")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "ward_join_method" in df.columns:
        df = df[df["ward_join_method"] == "within"].copy()
    if "ward_id" not in df.columns:
        raise ValueError("Ward optimization input must contain ward_id.")
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    keep = ["ward_id"] + (["ward_name"] if "ward_name" in df.columns else []) + numeric
    return df[keep].groupby([c for c in ["ward_id", "ward_name"] if c in keep], as_index=False).mean(numeric_only=True)


def simulate_temperature_delta(
    model,
    feature_names: list[str],
    base_features: np.ndarray,
    intervention: int,
) -> float:
    if intervention == 0:
        return 0.0
    modified = base_features.copy()
    for feature, delta in FEATURE_DELTAS.get(int(intervention), {}).items():
        if feature in feature_names:
            idx = feature_names.index(feature)
            modified[idx] = modified[idx] + delta
    base_pred = float(model.predict(pd.DataFrame([base_features], columns=feature_names))[0])
    mod_pred = float(model.predict(pd.DataFrame([modified], columns=feature_names))[0])
    return max(0.0, base_pred - mod_pred)


@dataclass
class OptimizationData:
    features: np.ndarray
    feature_names: list[str]
    areas_m2: np.ndarray
    populations: np.ndarray
    hvi: np.ndarray
    model: object


class PuneHeatProblem:
    """pymoo-compatible elementwise NSGA-II problem."""

    def __init__(self, data: OptimizationData, budget_inr: float = 500_000_000):
        try:
            from pymoo.core.problem import ElementwiseProblem
        except ImportError as exc:
            raise ImportError("pymoo is required for NSGA-II. Use ga_optimizer.py as a free fallback.") from exc

        self.data = data
        self.n_wards = len(data.features)
        self.budget_inr = budget_inr

        class _Problem(ElementwiseProblem):
            def __init__(inner_self):
                super().__init__(
                    n_var=self.n_wards * 2,
                    n_obj=4,
                    n_ieq_constr=1,
                    xl=np.hstack([np.zeros(self.n_wards), np.zeros(self.n_wards)]),
                    xu=np.hstack([4 * np.ones(self.n_wards), np.ones(self.n_wards)]),
                )

            def _evaluate(inner_self, x, out, *args, **kwargs):
                out["F"], out["G"] = self.evaluate(x)

        self.problem = _Problem()

    def evaluate(self, x: np.ndarray) -> tuple[list[float], list[float]]:
        n = self.n_wards
        types = np.round(x[:n]).astype(int).clip(0, 4)
        fracs = x[n:].clip(0, 1)
        deltas = np.array([
            simulate_temperature_delta(self.data.model, self.data.feature_names, self.data.features[i], types[i])
            for i in range(n)
        ])
        treated_area = self.data.areas_m2 * fracs
        costs = np.array([COSTS_INR_M2[int(t)] for t in types]) * treated_area
        total_cost = float(costs.sum())
        objectives = [
            -float(np.sum(deltas * treated_area)),
            -float(np.sum(deltas * self.data.populations * fracs)),
            total_cost,
            gini(fracs * (types > 0)),
        ]
        return objectives, [total_cost - self.budget_inr]


def run_nsga2(data: OptimizationData, budget_inr: float, generations: int, pop_size: int, seed: int):
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.optimize import minimize
    except ImportError as exc:
        raise ImportError("pymoo is not installed. Use src/optimization/ga_optimizer.py instead.") from exc

    wrapped = PuneHeatProblem(data, budget_inr)
    algorithm = NSGA2(
        pop_size=pop_size,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(prob=1.0 / len(data.features), eta=20),
        eliminate_duplicates=True,
    )
    return minimize(wrapped.problem, algorithm, termination=("n_gen", generations), seed=seed, verbose=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ward-features", type=Path, default=PROC_DIR / "ward_aggregates.parquet")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "grid_lst_model.joblib")
    parser.add_argument("--feature-list", type=Path, default=MODEL_DIR / "grid_lst_features.json")
    parser.add_argument("--budget", type=float, default=500_000_000)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "nsga2_pareto_solutions.npz")
    args = parser.parse_args()

    ward_df = load_ward_features(args.ward_features)
    feature_meta = json.loads(args.feature_list.read_text())
    feature_names = [f for f in feature_meta["features"] if f in ward_df.columns]
    if not feature_names:
        raise ValueError("No model features found in ward table.")

    areas = ward_df.get("area_m2", pd.Series(10_000.0, index=ward_df.index)).to_numpy(float)
    pops = ward_df.get("pop_density", pd.Series(1.0, index=ward_df.index)).to_numpy(float)
    hvi = ward_df.get("HVI", pd.Series(0.5, index=ward_df.index)).to_numpy(float)
    data = OptimizationData(ward_df[feature_names].to_numpy(float), feature_names, areas, pops, hvi, joblib.load(args.model))
    res = run_nsga2(data, args.budget, args.generations, args.pop_size, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, X=res.X, F=res.F, ward_ids=ward_df["ward_id"].astype(str).to_numpy())
    print(f"Saved Pareto solutions: {args.output}")


if __name__ == "__main__":
    main()

