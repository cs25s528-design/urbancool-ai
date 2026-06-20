#!/usr/bin/env python3
"""Gymnasium environment for sequential cooling intervention planning."""

from __future__ import annotations

import numpy as np

from .nsga2_optimizer import COSTS_INR_M2, simulate_temperature_delta

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None


class PuneHeatEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        ward_features: np.ndarray,
        feature_names: list[str],
        lst_model,
        ward_areas_m2: np.ndarray,
        populations: np.ndarray,
        hvi_scores: np.ndarray,
        budget: float = 500_000_000,
        horizon: int = 5,
    ):
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required for PuneHeatEnv. Install gymnasium to use RL.")
        super().__init__()
        self.base_features = np.asarray(ward_features, dtype=float)
        self.feature_names = feature_names
        self.model = lst_model
        self.areas = np.asarray(ward_areas_m2, dtype=float)
        self.populations = np.asarray(populations, dtype=float)
        self.hvi = np.asarray(hvi_scores, dtype=float)
        self.budget0 = float(budget)
        self.horizon = int(horizon)
        self.n_wards, self.n_features = self.base_features.shape

        obs_dim = self.n_wards * self.n_features + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.n_wards, 5])
        self.reset()

    def _obs(self):
        return np.concatenate([
            self.state.reshape(-1),
            np.array([self.budget_remaining / max(self.budget0, 1), self.t / max(self.horizon, 1)]),
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.base_features.copy()
        self.budget_remaining = self.budget0
        self.t = 0
        self.applied = np.zeros(self.n_wards, dtype=int)
        return self._obs(), {}

    def step(self, action):
        ward, intervention = int(action[0]), int(action[1])
        cost = COSTS_INR_M2[intervention] * self.areas[ward]
        if cost > self.budget_remaining or self.applied[ward]:
            reward = -0.05
            info = {"valid": False, "reason": "budget_or_duplicate"}
        else:
            delta = simulate_temperature_delta(self.model, self.feature_names, self.state[ward], intervention)
            equity = max(0.0, self.hvi[ward] - 0.5)
            reward = delta * max(self.populations[ward], 1.0) * 1e-4 + equity * 0.5 - cost / max(self.budget0, 1)
            self.budget_remaining -= cost
            self.applied[ward] = intervention
            self._apply_intervention(ward, intervention)
            info = {"valid": True, "delta_T": delta, "cost": cost}
        self.t += 1
        done = self.t >= self.horizon or self.budget_remaining <= 0
        return self._obs(), float(reward), done, False, info

    def _apply_intervention(self, ward: int, intervention: int):
        try:
            from .nsga2_optimizer import FEATURE_DELTAS
        except ImportError:
            from src.optimization.nsga2_optimizer import FEATURE_DELTAS

        for feature, delta in FEATURE_DELTAS.get(intervention, {}).items():
            if feature in self.feature_names:
                self.state[ward, self.feature_names.index(feature)] += delta

    def render(self):
        print(f"t={self.t}, budget_remaining={self.budget_remaining:,.0f}")
