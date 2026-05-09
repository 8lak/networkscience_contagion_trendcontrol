"""Gymnasium environment for controlled contagion dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.dynamics import seed_trend, seed_trends, step as dynamics_step, step_multi_trend
from src.graph import normalize01, normalize_signed

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # Allows basic imports in minimal environments.
    gym = None
    spaces = None


@dataclass
class RewardWeights:
    fatigue: float = 0.5
    fatigue_inequality: float = 0.1
    energy: float = 0.1


if gym is None:
    _BaseEnv = object
else:
    _BaseEnv = gym.Env


class ContagionEnv(_BaseEnv):
    """Control a trend by choosing nodes to boost at each step."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        adjacency,
        sigma: np.ndarray,
        rho: np.ndarray,
        affinity: np.ndarray,
        budget: int = 10,
        boost_budget: int | None = None,
        suppress_budget: int | None = None,
        target_adoption: float = 0.3,
        max_steps: int = 100,
        num_seeds: int = 10,
        boost_strength: float = 0.8,
        suppress_strength: float = 0.5,
        eta: float = 0.08,
        dt: float = 0.1,
        adoption_threshold: float = 0.5,
        node_features: dict[str, np.ndarray] | None = None,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        if spaces is None:
            raise ModuleNotFoundError(
                "ContagionEnv requires gymnasium. Install dependencies before training."
            )

        self.adjacency = adjacency
        self.sigma = np.asarray(sigma, dtype=np.float32)
        self.rho = np.asarray(rho, dtype=np.float32)
        self.affinity = np.asarray(affinity, dtype=np.float32)
        self.n = len(self.sigma)
        self.budget = int(budget)
        self.boost_budget = int(boost_budget if boost_budget is not None else budget)
        self.suppress_budget = int(suppress_budget if suppress_budget is not None else budget)
        self.total_budget = max(self.boost_budget + self.suppress_budget, 1)
        self.target_adoption = float(target_adoption)
        self.max_steps = int(max_steps)
        self.num_seeds = int(num_seeds)
        self.boost_strength = float(boost_strength)
        self.suppress_strength = float(suppress_strength)
        self.eta = float(eta)
        self.dt = float(dt)
        self.adoption_threshold = float(adoption_threshold)
        self.reward_weights = reward_weights or RewardWeights()
        self.node_features = node_features or self._default_node_features()
        self.eigenvectors = np.asarray(self.node_features["eigenvectors"], dtype=np.float32)
        self.eigenvalues = np.asarray(self.node_features["eigenvalues"], dtype=np.float32)
        self.degree = np.asarray(self.node_features["degree"], dtype=np.float32)
        self.hub = np.asarray(self.node_features["hub"], dtype=np.float32)
        self.authority = np.asarray(self.node_features["authority"], dtype=np.float32)
        self.k = int(self.eigenvectors.shape[1])
        self.last_action_energy = 0.0
        self.last_selected = np.array([], dtype=np.int64)
        self.last_boosted = np.array([], dtype=np.int64)
        self.last_suppressed = np.array([], dtype=np.int64)

        self.action_space = spaces.Box(
            low=-np.ones(self.k + 3, dtype=np.float32),
            high=np.ones(self.k + 3, dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.ones(4 * self.k + 5, dtype=np.float32),
            high=np.ones(4 * self.k + 5, dtype=np.float32),
            dtype=np.float32,
        )
        self.rng = np.random.default_rng()
        self.v = np.zeros(self.n, dtype=np.float32)
        self.phi = np.zeros(self.n, dtype=np.float32)
        self.t = 0

    def _default_node_features(self) -> dict[str, np.ndarray]:
        eigenvectors = np.ones((self.n, 1), dtype=np.float32) / max(np.sqrt(self.n), 1.0)
        return {
            "eigenvectors": eigenvectors,
            "eigenvalues": np.zeros(1, dtype=np.float32),
            "degree": normalize01(np.asarray(self.adjacency.sum(axis=1)).reshape(-1)),
            "hub": np.ones(self.n, dtype=np.float32),
            "authority": np.ones(self.n, dtype=np.float32),
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.v = seed_trend(self.n, self.num_seeds, self.rng)
        self.phi = np.zeros(self.n, dtype=np.float32)
        self.t = 0
        self.last_action_energy = 0.0
        self.last_selected = np.array([], dtype=np.int64)
        self.last_boosted = np.array([], dtype=np.int64)
        self.last_suppressed = np.array([], dtype=np.int64)
        return self._get_observation(), self._get_info()

    def _get_observation(self) -> np.ndarray:
        sqrt_n = max(np.sqrt(self.n), 1.0)
        v_coeffs = normalize_signed((self.eigenvectors.T @ self.v) / sqrt_n)
        phi_coeffs = normalize_signed((self.eigenvectors.T @ self.phi) / sqrt_n)
        authority_v_coeffs = normalize_signed((self.eigenvectors.T @ (self.v * self.authority)) / sqrt_n)
        stats = np.array(
            [
                float(np.mean(self.v)),
                float(np.std(self.v)),
                float(np.mean(self.phi)),
                float(np.std(self.phi)),
                self.t / max(self.max_steps, 1),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                v_coeffs,
                phi_coeffs,
                self.eigenvalues,
                authority_v_coeffs,
                stats,
            ]
        ).astype(np.float32)

    def _get_info(self) -> dict[str, float]:
        return {
            "mean_adoption": float(np.mean(self.v)),
            "mean_fatigue": float(np.mean(self.phi)),
            "adopted_fraction": float(np.mean(self.v > self.adoption_threshold)),
            "action_energy": float(self.last_action_energy),
            "selected_count": float(len(self.last_selected)),
            "boosted_count": float(len(self.last_boosted)),
            "suppressed_count": float(len(self.last_suppressed)),
        }

    def _decode_action(self, action) -> tuple[np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self.k + 3:
            raise ValueError(f"Expected action shape ({self.k + 3},), got {action.shape}")

        mode_weights = action[: self.k]
        w_hub, w_auth, w_degree = action[self.k :]
        action_intensity = float(np.clip(np.mean(np.abs(action)), 0.0, 1.0))
        if action_intensity <= 1e-8:
            zeros = np.zeros(self.n, dtype=np.float32)
            return zeros, zeros

        spectral_pattern = normalize_signed(self.eigenvectors @ mode_weights)
        centrality_pref = normalize_signed(
            w_hub * self.hub + w_auth * self.authority + w_degree * self.degree
        )
        raw_score = normalize_signed(spectral_pattern * centrality_pref)

        neighbor_pressure = np.asarray(self.adjacency @ self.v).reshape(-1)
        frontier = normalize01((1.0 - self.v) * neighbor_pressure)
        low_fatigue = 1.0 - self.phi
        boost_eff = normalize01(low_fatigue * (frontier + 0.25 * self.v))
        suppress_eff = normalize01(self.v * (self.phi + 0.1))

        boost_scores = np.where(raw_score > 0.0, raw_score * boost_eff, 0.0)
        suppress_scores = np.where(raw_score < 0.0, raw_score * suppress_eff, 0.0)
        return (
            normalize_signed(boost_scores) * action_intensity,
            normalize_signed(suppress_scores) * action_intensity,
        )

    def _apply_action_scores(self, boost_scores: np.ndarray, suppress_scores: np.ndarray) -> None:
        if (
            self.boost_budget <= 0
            and self.suppress_budget <= 0
            or max(float(np.max(boost_scores)), float(np.max(np.abs(suppress_scores)))) <= 1e-8
        ):
            self.last_selected = np.array([], dtype=np.int64)
            self.last_boosted = np.array([], dtype=np.int64)
            self.last_suppressed = np.array([], dtype=np.int64)
            self.last_action_energy = 0.0
            return

        boost_candidates = np.flatnonzero(boost_scores > 1e-8)
        suppress_candidates = np.flatnonzero(suppress_scores < -1e-8)
        boosted = boost_candidates[np.argsort(-boost_scores[boost_candidates])[: self.boost_budget]]
        suppressed = suppress_candidates[
            np.argsort(suppress_scores[suppress_candidates])[: self.suppress_budget]
        ]

        boost_strengths = boost_scores[boosted].astype(np.float32)
        suppress_strengths = np.abs(suppress_scores[suppressed]).astype(np.float32)

        for node, strength in zip(boosted, boost_strengths):
            self.v[node] = np.clip(
                self.v[node] + (1.0 - self.v[node]) * self.boost_strength * strength,
                0.0,
                1.0,
            )

        for node, strength in zip(suppressed, suppress_strengths):
            self.v[node] = np.clip(
                self.v[node] * (1.0 - self.suppress_strength * strength),
                0.0,
                1.0,
            )

        self.last_boosted = boosted.astype(np.int64)
        self.last_suppressed = suppressed.astype(np.int64)
        self.last_selected = np.concatenate([self.last_boosted, self.last_suppressed])
        strengths = np.concatenate([boost_strengths, suppress_strengths])
        nodes_touched = len(strengths) / self.total_budget
        avg_intensity = float(np.mean(strengths)) if len(strengths) else 0.0
        self.last_action_energy = float(nodes_touched + avg_intensity)

    def step(self, action):
        boost_scores, suppress_scores = self._decode_action(action)
        self._apply_action_scores(boost_scores, suppress_scores)

        phi_global = float(np.mean(self.phi))
        self.v, self.phi = dynamics_step(
            self.v,
            self.phi,
            self.adjacency,
            self.sigma,
            self.affinity,
            self.rho,
            eta=self.eta,
            phi_global=phi_global,
            dt=self.dt,
        )
        self.t += 1

        adoption_error = abs(float(np.mean(self.v)) - self.target_adoption)
        reward = (
            -adoption_error
            - self.reward_weights.fatigue * float(np.mean(self.phi))
            - self.reward_weights.fatigue_inequality * float(np.std(self.phi))
            - self.reward_weights.energy * self.last_action_energy
        )
        terminated = self.t >= self.max_steps
        truncated = False
        return self._get_observation(), reward, terminated, truncated, self._get_info()


class MultiTrendContagionEnv(_BaseEnv):
    """Control one trend inside a competing multi-trend attention system."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        adjacency,
        sigma: np.ndarray,
        rho: np.ndarray,
        affinity: np.ndarray,
        target_trend: int = 3,
        target_share: float = 0.3,
        budget: int = 10,
        boost_budget: int | None = None,
        suppress_budget: int | None = None,
        max_steps: int = 100,
        num_seeds: int = 20,
        seed_strength: float = 0.5,
        action_scale: float = 0.1,
        beta: float = 0.3,
        eta: float = 0.1,
        dt: float = 0.5,
        decoder_mode: str = "current",
        node_features: dict[str, np.ndarray] | None = None,
        reward_weights: RewardWeights | None = None,
    ) -> None:
        if spaces is None:
            raise ModuleNotFoundError(
                "MultiTrendContagionEnv requires gymnasium. Install dependencies before training."
            )

        self.adjacency = adjacency
        self.sigma = np.asarray(sigma, dtype=np.float32)
        self.rho = np.asarray(rho, dtype=np.float32)
        self.affinity = np.asarray(affinity, dtype=np.float32)
        if self.affinity.ndim != 2:
            raise ValueError("MultiTrendContagionEnv affinity must have shape (n, T)")

        self.n, self.num_trends = self.affinity.shape
        self.target_trend = int(target_trend)
        if not 0 <= self.target_trend < self.num_trends:
            raise ValueError(f"target_trend must be in [0, {self.num_trends})")

        self.target_share = float(target_share)
        self.target_adoption = self.target_share
        self.budget = int(budget)
        self.boost_budget = int(boost_budget if boost_budget is not None else budget)
        self.suppress_budget = int(suppress_budget if suppress_budget is not None else budget)
        self.total_budget = max(self.boost_budget + self.suppress_budget, 1)
        self.max_steps = int(max_steps)
        self.num_seeds = int(num_seeds)
        self.seed_strength = float(seed_strength)
        self.action_scale = float(action_scale)
        self.beta = float(beta)
        self.eta = float(eta)
        self.dt = float(dt)
        self.decoder_mode = decoder_mode
        if self.decoder_mode not in {
            "current",
            "additive",
            "spectral_frontier",
            "frontier",
            "raw_frontier",
            "signed_frontier",
        }:
            raise ValueError(
                "decoder_mode must be one of: current, additive, spectral_frontier, "
                "frontier, raw_frontier, signed_frontier"
            )
        self.reward_weights = reward_weights or RewardWeights()

        self.node_features = node_features or self._default_node_features()
        self.eigenvectors = np.asarray(self.node_features["eigenvectors"], dtype=np.float32)
        self.degree = np.asarray(self.node_features["degree"], dtype=np.float32)
        self.hub = np.asarray(self.node_features["hub"], dtype=np.float32)
        self.authority = np.asarray(self.node_features["authority"], dtype=np.float32)
        self.k = int(self.eigenvectors.shape[1])

        self.action_space = spaces.Box(
            low=-np.ones(self.k + 3, dtype=np.float32),
            high=np.ones(self.k + 3, dtype=np.float32),
            dtype=np.float32,
        )
        obs_dim = 2 * self.k * self.num_trends + 4 * self.num_trends + 1
        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng()
        self.v = np.zeros((self.n, self.num_trends), dtype=np.float32)
        self.phi = np.zeros((self.n, self.num_trends), dtype=np.float32)
        self.t = 0
        self.last_action_energy = 0.0
        self.last_selected = np.array([], dtype=np.int64)
        self.last_boosted = np.array([], dtype=np.int64)
        self.last_suppressed = np.array([], dtype=np.int64)

    def _default_node_features(self) -> dict[str, np.ndarray]:
        eigenvectors = np.ones((self.n, 1), dtype=np.float32) / max(np.sqrt(self.n), 1.0)
        return {
            "eigenvectors": eigenvectors,
            "degree": normalize01(np.asarray(self.adjacency.sum(axis=1)).reshape(-1)),
            "hub": np.ones(self.n, dtype=np.float32),
            "authority": np.ones(self.n, dtype=np.float32),
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.v = seed_trends(
            self.n,
            self.num_trends,
            num_seeds_per_trend=self.num_seeds,
            rng=self.rng,
            seed_strength=self.seed_strength,
        )
        self.phi = np.zeros((self.n, self.num_trends), dtype=np.float32)
        self.t = 0
        self.last_action_energy = 0.0
        self.last_selected = np.array([], dtype=np.int64)
        self.last_boosted = np.array([], dtype=np.int64)
        self.last_suppressed = np.array([], dtype=np.int64)
        return self._get_observation(), self._get_info()

    def _get_observation(self) -> np.ndarray:
        sqrt_n = max(np.sqrt(self.n), 1.0)
        v_coeffs = normalize_signed((self.eigenvectors.T @ self.v) / sqrt_n).reshape(-1)
        phi_coeffs = normalize_signed((self.eigenvectors.T @ self.phi) / sqrt_n).reshape(-1)
        v_mean = self.v.mean(axis=0).astype(np.float32)
        v_std = self.v.std(axis=0).astype(np.float32)
        phi_mean = self.phi.mean(axis=0).astype(np.float32)
        target_indicator = np.zeros(self.num_trends, dtype=np.float32)
        target_indicator[self.target_trend] = 1.0
        time = np.array([self.t / max(self.max_steps, 1)], dtype=np.float32)
        return np.concatenate(
            [v_coeffs, phi_coeffs, v_mean, v_std, phi_mean, target_indicator, time]
        ).astype(np.float32)

    def _get_info(self) -> dict[str, float]:
        target_share = float(np.mean(self.v[:, self.target_trend]))
        target_fatigue = float(np.mean(self.phi[:, self.target_trend]))
        return {
            "mean_adoption": target_share,
            "mean_fatigue": target_fatigue,
            "target_share": target_share,
            "target_fatigue": target_fatigue,
            "target_deviation": abs(target_share - self.target_share),
            "action_energy": float(self.last_action_energy),
            "selected_count": float(len(self.last_selected)),
            "boosted_count": float(len(self.last_boosted)),
            "suppressed_count": float(len(self.last_suppressed)),
            "simplex_error": float(np.max(np.abs(self.v.sum(axis=1) - 1.0))),
        }

    def _decode_action(self, action) -> tuple[np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self.k + 3:
            raise ValueError(f"Expected action shape ({self.k + 3},), got {action.shape}")

        mode_weights = action[: self.k]
        w_hub, w_auth, w_degree = action[self.k :]
        action_intensity = float(np.clip(np.mean(np.abs(action)), 0.0, 1.0))
        if action_intensity <= 1e-8:
            zeros = np.zeros(self.n, dtype=np.float32)
            return zeros, zeros

        target_v = self.v[:, self.target_trend]
        target_phi = self.phi[:, self.target_trend]
        spectral_pattern = normalize_signed(self.eigenvectors @ mode_weights)
        centrality_pref = normalize_signed(
            w_hub * self.hub + w_auth * self.authority + w_degree * self.degree
        )

        neighbor_pressure = np.asarray(self.adjacency @ target_v).reshape(-1)
        frontier = normalize01((1.0 - target_v) * neighbor_pressure)
        low_fatigue = 1.0 - target_phi
        boost_eff = normalize01(low_fatigue * (frontier + 0.25 * target_v))
        suppress_eff = normalize01(target_v * (target_phi + 0.1))

        if self.decoder_mode == "current":
            raw_score = normalize_signed(spectral_pattern * centrality_pref)
        elif self.decoder_mode == "additive":
            raw_score = normalize_signed(spectral_pattern + centrality_pref + frontier)
        elif self.decoder_mode == "spectral_frontier":
            frontier_coeffs = self.eigenvectors.T @ frontier
            spectral_frontier = normalize_signed(self.eigenvectors @ frontier_coeffs)
            raw_score = normalize_signed(spectral_frontier * centrality_pref)
        elif self.decoder_mode == "frontier":
            raw_score = frontier.astype(np.float32)
        elif self.decoder_mode == "signed_frontier":
            direction = float(np.clip(np.mean(action), -1.0, 1.0))
            magnitude = abs(direction)
            if magnitude <= 1e-8:
                zeros = np.zeros(self.n, dtype=np.float32)
                return zeros, zeros
            if direction > 0.0:
                return (
                    normalize_signed(frontier) * magnitude,
                    np.zeros(self.n, dtype=np.float32),
                )
            suppress_target = normalize01(target_v * (0.5 + target_phi))
            return (
                np.zeros(self.n, dtype=np.float32),
                -normalize_signed(suppress_target) * magnitude,
            )
        else:
            boost_scores = frontier.astype(np.float32)
            return normalize_signed(boost_scores) * action_intensity, np.zeros(self.n, dtype=np.float32)

        boost_scores = np.where(raw_score > 0.0, raw_score * boost_eff, 0.0)
        suppress_scores = np.where(raw_score < 0.0, raw_score * suppress_eff, 0.0)
        return (
            normalize_signed(boost_scores) * action_intensity,
            normalize_signed(suppress_scores) * action_intensity,
        )

    def _apply_boost(self, node: int, strength: float) -> None:
        target_mass = float(self.v[node, self.target_trend])
        other_mass = max(1.0 - target_mass, 0.0)
        amount = min(self.action_scale * float(strength), other_mass)
        if amount <= 1e-8:
            return
        if other_mass > 1e-8:
            scale = max(1.0 - amount / other_mass, 0.0)
            self.v[node, :] *= scale
            self.v[node, self.target_trend] = target_mass + amount

    def _apply_suppress(self, node: int, strength: float) -> None:
        target_mass = float(self.v[node, self.target_trend])
        amount = min(self.action_scale * float(strength), target_mass)
        if amount <= 1e-8:
            return

        other_mass = max(1.0 - target_mass, 0.0)
        self.v[node, self.target_trend] = target_mass - amount
        if other_mass > 1e-8:
            for trend in range(self.num_trends):
                if trend != self.target_trend:
                    self.v[node, trend] += amount * float(self.v[node, trend]) / other_mass
        else:
            fallback = 0 if self.target_trend != 0 else 1
            self.v[node, fallback] += amount

    def _apply_action_scores(self, boost_scores: np.ndarray, suppress_scores: np.ndarray) -> None:
        if (
            self.boost_budget <= 0
            and self.suppress_budget <= 0
            or max(float(np.max(boost_scores)), float(np.max(np.abs(suppress_scores)))) <= 1e-8
        ):
            self.last_selected = np.array([], dtype=np.int64)
            self.last_boosted = np.array([], dtype=np.int64)
            self.last_suppressed = np.array([], dtype=np.int64)
            self.last_action_energy = 0.0
            return

        boost_candidates = np.flatnonzero(boost_scores > 1e-8)
        suppress_candidates = np.flatnonzero(suppress_scores < -1e-8)
        boosted = boost_candidates[np.argsort(-boost_scores[boost_candidates])[: self.boost_budget]]
        suppressed = suppress_candidates[
            np.argsort(suppress_scores[suppress_candidates])[: self.suppress_budget]
        ]

        boost_strengths = boost_scores[boosted].astype(np.float32)
        suppress_strengths = np.abs(suppress_scores[suppressed]).astype(np.float32)

        for node, strength in zip(boosted, boost_strengths):
            self._apply_boost(int(node), float(strength))
        for node, strength in zip(suppressed, suppress_strengths):
            self._apply_suppress(int(node), float(strength))

        row_sums = self.v.sum(axis=1, keepdims=True)
        self.v = (self.v / np.maximum(row_sums, 1e-8)).astype(np.float32)
        self.last_boosted = boosted.astype(np.int64)
        self.last_suppressed = suppressed.astype(np.int64)
        self.last_selected = np.concatenate([self.last_boosted, self.last_suppressed])
        strengths = np.concatenate([boost_strengths, suppress_strengths])
        nodes_touched = len(strengths) / self.total_budget
        avg_intensity = float(np.mean(strengths)) if len(strengths) else 0.0
        self.last_action_energy = float(nodes_touched + avg_intensity)

    def step(self, action):
        boost_scores, suppress_scores = self._decode_action(action)
        self._apply_action_scores(boost_scores, suppress_scores)

        self.v, self.phi = step_multi_trend(
            self.v,
            self.phi,
            self.adjacency,
            self.sigma,
            self.affinity,
            self.rho,
            beta=self.beta,
            eta=self.eta,
            dt=self.dt,
        )
        self.t += 1

        target_share = float(np.mean(self.v[:, self.target_trend]))
        target_fatigue = float(np.mean(self.phi[:, self.target_trend]))
        deviation = abs(target_share - self.target_share)
        reward = (
            -deviation
            - self.reward_weights.fatigue * target_fatigue
            - self.reward_weights.energy * self.last_action_energy
        )
        terminated = self.t >= self.max_steps
        truncated = False
        return self._get_observation(), reward, terminated, truncated, self._get_info()
