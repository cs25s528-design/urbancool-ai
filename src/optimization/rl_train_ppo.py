#!/usr/bin/env python3
"""Train PPO agent for PuneHeatEnv when stable-baselines3 is available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

try:
    from .nsga2_optimizer import load_ward_features
    from .rl_environment import PuneHeatEnv
except ImportError:  # Allows direct execution: python src/optimization/rl_train_ppo.py
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.optimization.nsga2_optimizer import load_ward_features
    from src.optimization.rl_environment import PuneHeatEnv

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ward-features", type=Path, default=PROC_DIR / "ward_aggregates.parquet")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "grid_lst_model.joblib")
    parser.add_argument("--feature-list", type=Path, default=MODEL_DIR / "grid_lst_features.json")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "ppo_pune_intervention")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:
        raise ImportError("stable-baselines3 is required to train PPO.") from exc

    ward_df = load_ward_features(args.ward_features)
    features = [f for f in json.loads(args.feature_list.read_text())["features"] if f in ward_df.columns]
    base_model = joblib.load(args.model)
    areas = ward_df.get("area_m2", pd.Series(10_000.0, index=ward_df.index)).to_numpy(float)
    pop = ward_df.get("pop_density", pd.Series(1.0, index=ward_df.index)).to_numpy(float)
    hvi = ward_df.get("HVI", pd.Series(0.5, index=ward_df.index)).to_numpy(float)

    def make_env():
        return PuneHeatEnv(ward_df[features].to_numpy(float), features, base_model, areas, pop, hvi)

    env = make_vec_env(make_env, n_envs=1)
    eval_env = make_vec_env(make_env, n_envs=1)
    callback = EvalCallback(eval_env, best_model_save_path=str(args.output.parent), log_path=str(args.output.parent), eval_freq=5000)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps, callback=callback)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Saved PPO model: {args.output}.zip")


if __name__ == "__main__":
    main()
