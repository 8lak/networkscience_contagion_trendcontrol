from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.env import MultiTrendContagionEnv, RewardWeights
from src.graph import (
    assign_archetypes,
    compute_affinity_matrix,
    generate_trend_embeddings,
    load_facebook_graph,
    normalize01,
    normalize_signed,
    precompute_node_features,
    sample_node_interests,
)


VARIANTS = (
    "raw_frontier",
    "spectral_smooth_frontier",
    "spectral_residual_frontier",
    "contested_frontier",
    "spectral_contested_frontier",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test spectral frontier heuristic variants.")
    parser.add_argument("--graph", default=ROOT / "data/facebook_combined.txt", type=Path)
    parser.add_argument("--target-trend", default=3, type=int)
    parser.add_argument("--target-share", default=0.5, type=float)
    parser.add_argument("--trend-strengths", nargs="*", default=[1.0, 1.25, 1.0, 0.75], type=float)
    parser.add_argument("--num-trends", default=4, type=int)
    parser.add_argument("--embedding-dim", default=10, type=int)
    parser.add_argument("--episodes", default=10, type=int)
    parser.add_argument("--max-steps", default=100, type=int)
    parser.add_argument("--spectral-modes", default=5, type=int)
    parser.add_argument("--boost-budget", default=1200, type=int)
    parser.add_argument("--suppress-budget", default=0, type=int)
    parser.add_argument("--action-scale", default=2.0, type=float)
    parser.add_argument("--energy-weight", default=0.02, type=float)
    parser.add_argument("--fatigue-weight", default=0.5, type=float)
    parser.add_argument("--fatigue-inequality-weight", default=0.1, type=float)
    parser.add_argument("--num-seeds", default=20, type=int)
    parser.add_argument("--seed-strength", default=0.5, type=float)
    parser.add_argument("--beta", default=0.3, type=float)
    parser.add_argument("--eta", default=0.1, type=float)
    parser.add_argument("--dt", default=0.5, type=float)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--tau", default=2.0, type=float)
    return parser.parse_args()


def build_inputs(args):
    graph, adjacency, nodes = load_facebook_graph(args.graph)
    n = len(nodes)
    sigma, rho, _ = assign_archetypes(n, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    trend_embeddings = generate_trend_embeddings(args.num_trends, args.embedding_dim, rng)
    node_interests = sample_node_interests(n, args.embedding_dim, rng)
    affinity = compute_affinity_matrix(node_interests, trend_embeddings)
    if args.trend_strengths is not None:
        strengths = np.asarray(args.trend_strengths, dtype=np.float32)
        if strengths.size != args.num_trends:
            raise ValueError(f"--trend-strengths must have {args.num_trends} values")
        affinity = np.clip(affinity * strengths[None, :], 0.0, 1.0)
    affinity[:, 0] = 0.5
    node_features = precompute_node_features(graph, adjacency, nodes, k=args.spectral_modes)
    return adjacency, sigma, rho, affinity, node_features


class SpectralVariantEnv(MultiTrendContagionEnv):
    def __init__(self, *args, variant: str, tau: float, **kwargs):
        super().__init__(*args, decoder_mode="raw_frontier", **kwargs)
        self.variant = variant
        self.tau = float(tau)
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")

    def _spectral_smooth(self, signal: np.ndarray) -> np.ndarray:
        coeffs = self.eigenvectors.T @ np.asarray(signal, dtype=np.float32)
        eigenvalues = np.asarray(self.node_features.get("eigenvalues"), dtype=np.float32)
        if eigenvalues.size != coeffs.size:
            eigenvalues = np.linspace(0.0, 1.0, coeffs.size, dtype=np.float32)
        weights = np.exp(-self.tau * eigenvalues).astype(np.float32)
        return normalize01(self.eigenvectors @ (weights * coeffs))

    def _decode_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action_intensity = float(np.clip(np.mean(np.abs(action)), 0.0, 1.0))
        if action_intensity <= 1e-8:
            zeros = np.zeros(self.n, dtype=np.float32)
            return zeros, zeros

        target_v = self.v[:, self.target_trend]
        target_phi = self.phi[:, self.target_trend]
        target_pressure = np.asarray(self.adjacency @ target_v).reshape(-1)
        frontier = normalize01((1.0 - target_v) * target_pressure)

        competitor_v = np.sum(self.v, axis=1) - target_v
        competitor_pressure = np.asarray(self.adjacency @ competitor_v).reshape(-1)
        contested = normalize01(frontier * target_pressure / (competitor_pressure + 1e-6))

        residual = np.maximum(self.target_share - target_v, 0.0)
        residual_frontier = normalize01(frontier * residual)
        low_fatigue = np.clip(1.0 - target_phi, 0.0, 1.0)

        target_affinity = self.affinity[:, self.target_trend]
        competitor_affinity = np.mean(
            np.delete(self.affinity, self.target_trend, axis=1),
            axis=1,
        )
        affinity_advantage = normalize01(target_affinity / (competitor_affinity + 1e-6))

        if self.variant == "raw_frontier":
            score = frontier
        elif self.variant == "spectral_smooth_frontier":
            score = frontier * self._spectral_smooth(frontier)
        elif self.variant == "spectral_residual_frontier":
            score = frontier * self._spectral_smooth(residual_frontier)
        elif self.variant == "contested_frontier":
            score = contested * low_fatigue * (0.5 + 0.5 * affinity_advantage)
        else:
            score = contested * self._spectral_smooth(contested) * low_fatigue
            score *= 0.5 + 0.5 * affinity_advantage

        boost_scores = normalize_signed(score) * action_intensity
        return boost_scores.astype(np.float32), np.zeros(self.n, dtype=np.float32)


def make_env(args, static_inputs, variant: str):
    adjacency, sigma, rho, affinity, node_features = static_inputs
    return SpectralVariantEnv(
        adjacency,
        sigma,
        rho,
        affinity,
        target_trend=args.target_trend,
        target_share=args.target_share,
        boost_budget=args.boost_budget,
        suppress_budget=args.suppress_budget,
        max_steps=args.max_steps,
        num_seeds=args.num_seeds,
        seed_strength=args.seed_strength,
        action_scale=args.action_scale,
        beta=args.beta,
        eta=args.eta,
        dt=args.dt,
        node_features=node_features,
        reward_weights=RewardWeights(
            fatigue=args.fatigue_weight,
            fatigue_inequality=args.fatigue_inequality_weight,
            energy=args.energy_weight,
        ),
        variant=variant,
        tau=args.tau,
    )


def action_all(env):
    return np.ones(env.action_space.shape, dtype=np.float32)


def action_none(env):
    return np.zeros(env.action_space.shape, dtype=np.float32)


def evaluate(env, policy, episodes: int):
    rewards = []
    deviations = []
    fatigue = []
    energy = []
    final_share = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        done = False
        total_reward = 0.0
        ep_deviation = []
        ep_fatigue = []
        ep_energy = []
        info = {}
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(env))
            done = terminated or truncated
            total_reward += float(reward)
            ep_deviation.append(float(info["target_deviation"]))
            ep_fatigue.append(float(info["target_fatigue"]))
            ep_energy.append(float(info["action_energy"]))
        rewards.append(total_reward)
        deviations.append(float(np.mean(ep_deviation)))
        fatigue.append(float(np.mean(ep_fatigue)))
        energy.append(float(np.mean(ep_energy)))
        final_share.append(float(info["target_share"]))
    return {
        "reward": float(np.mean(rewards)),
        "deviation": float(np.mean(deviations)),
        "fatigue": float(np.mean(fatigue)),
        "energy": float(np.mean(energy)),
        "final_share": float(np.mean(final_share)),
    }


def main():
    args = parse_args()
    static_inputs = build_inputs(args)
    none_env = make_env(args, static_inputs, "raw_frontier")
    none_metrics = evaluate(none_env, action_none, args.episodes)
    print(f"none: {none_metrics}")

    rows = []
    for variant in VARIANTS:
        env = make_env(args, static_inputs, variant)
        metrics = evaluate(env, action_all, args.episodes)
        rows.append((variant, metrics))
        print(f"{variant}: {metrics}")

    best = max(rows, key=lambda item: item[1]["reward"])
    print(f"best_by_reward: {best[0]} {best[1]}")


if __name__ == "__main__":
    main()
