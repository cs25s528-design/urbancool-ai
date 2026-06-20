#!/usr/bin/env python3
"""Fast single-objective genetic algorithm for intervention planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from .nsga2_optimizer import COSTS_INR_M2, INTERVENTION_LABELS, load_ward_features, simulate_temperature_delta
except ImportError:  # Allows direct execution: python src/optimization/ga_optimizer.py
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.optimization.nsga2_optimizer import COSTS_INR_M2, INTERVENTION_LABELS, load_ward_features, simulate_temperature_delta

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "data" / "outputs"


def repair_budget(types: np.ndarray, fracs: np.ndarray, areas: np.ndarray, budget: float) -> np.ndarray:
    costs = np.array([COSTS_INR_M2[int(t)] for t in types]) * areas * fracs
    total = costs.sum()
    if total > budget and total > 0:
        fracs = fracs * (budget / total)
    return fracs.clip(0, 1)


def evaluate_solution(types, fracs, features, feature_names, areas, populations, hvi, model, budget):
    fracs = repair_budget(types, fracs, areas, budget)
    deltas = np.array([
        simulate_temperature_delta(model, feature_names, features[i], types[i])
        for i in range(len(types))
    ])
    costs = np.array([COSTS_INR_M2[int(t)] for t in types]) * areas * fracs
    cooling = deltas * fracs
    pop_benefit = cooling * np.nan_to_num(populations, nan=1.0)
    equity = cooling * np.nan_to_num(hvi, nan=0.5)
    cost_penalty = costs.sum() / max(budget, 1.0)
    score = 0.40 * cooling.mean() + 0.25 * pop_benefit.mean() + 0.25 * equity.mean() + 0.10 * (1 - cost_penalty)
    return float(score), fracs, {
        "mean_delta_C": float(cooling.mean()),
        "population_benefit": float(pop_benefit.sum()),
        "equity_benefit": float(equity.sum()),
        "total_cost_inr": float(costs.sum()),
    }


def run_ga(
    features: np.ndarray,
    feature_names: list[str],
    areas: np.ndarray,
    populations: np.ndarray,
    hvi: np.ndarray,
    model,
    budget: float = 500_000_000,
    population_size: int = 120,
    generations: int = 150,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(features)
    types_pop = rng.integers(0, 5, size=(population_size, n))
    fracs_pop = rng.uniform(0, 1, size=(population_size, n))

    best = None
    for _ in range(generations):
        scored = []
        for i in range(population_size):
            score, fracs, info = evaluate_solution(types_pop[i], fracs_pop[i], features, feature_names, areas, populations, hvi, model, budget)
            scored.append((score, i, fracs, info))
            fracs_pop[i] = fracs
        scored.sort(reverse=True, key=lambda x: x[0])
        if best is None or scored[0][0] > best["score"]:
            idx = scored[0][1]
            best = {
                "score": scored[0][0],
                "types": types_pop[idx].copy(),
                "fractions": fracs_pop[idx].copy(),
                "metrics": scored[0][3],
            }

        elite_n = max(4, population_size // 10)
        elite_idx = [s[1] for s in scored[:elite_n]]
        new_types = [types_pop[i].copy() for i in elite_idx]
        new_fracs = [fracs_pop[i].copy() for i in elite_idx]
        while len(new_types) < population_size:
            p1, p2 = rng.choice(elite_idx, size=2, replace=True)
            cut = rng.integers(1, n)
            child_t = np.concatenate([types_pop[p1, :cut], types_pop[p2, cut:]])
            child_f = np.concatenate([fracs_pop[p1, :cut], fracs_pop[p2, cut:]])
            mut = rng.random(n) < (1.0 / max(n, 1))
            child_t[mut] = rng.integers(0, 5, size=mut.sum())
            child_f = np.clip(child_f + rng.normal(0, 0.08, size=n), 0, 1)
            new_types.append(child_t)
            new_fracs.append(child_f)
        types_pop = np.array(new_types)
        fracs_pop = np.array(new_fracs)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ward-features", type=Path, default=PROC_DIR / "ward_aggregates.parquet")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "grid_lst_model.joblib")
    parser.add_argument("--feature-list", type=Path, default=MODEL_DIR / "grid_lst_features.json")
    parser.add_argument("--budget", type=float, default=500_000_000)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--population-size", type=int, default=120)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "ga_best_plan.csv")
    args = parser.parse_args()

    ward_df = load_ward_features(args.ward_features)
    feature_meta = json.loads(args.feature_list.read_text())
    feature_names = [f for f in feature_meta["features"] if f in ward_df.columns]
    result = run_ga(
        ward_df[feature_names].to_numpy(float),
        feature_names,
        ward_df.get("area_m2", pd.Series(10_000.0, index=ward_df.index)).to_numpy(float),
        ward_df.get("pop_density", pd.Series(1.0, index=ward_df.index)).to_numpy(float),
        ward_df.get("HVI", pd.Series(0.5, index=ward_df.index)).to_numpy(float),
        joblib.load(args.model),
        args.budget,
        args.population_size,
        args.generations,
    )
    out = pd.DataFrame({
        "ward_id": ward_df["ward_id"].astype(str),
        "intervention_type": result["types"],
        "intervention_label": [INTERVENTION_LABELS[int(t)] for t in result["types"]],
        "intervention_frac": result["fractions"],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved GA best plan: {args.output}")
    print(result["metrics"])


if __name__ == "__main__":
    main()
