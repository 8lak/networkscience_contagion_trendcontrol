from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3 import PPO

from src.env import ContagionEnv, MultiTrendContagionEnv, RewardWeights
from src.graph import (
    assign_archetypes,
    compute_affinity_matrix,
    generate_trend_embeddings,
    load_facebook_graph,
    precompute_node_features,
    sample_affinity,
    sample_node_interests,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on contagion control.")
    parser.add_argument("--graph", default=ROOT / "data/facebook_combined.txt", type=Path)
    parser.add_argument("--multi-trend", action="store_true")
    parser.add_argument("--num-trends", default=4, type=int)
    parser.add_argument("--embedding-dim", default=10, type=int)
    parser.add_argument("--target-trend", default=3, type=int)
    parser.add_argument("--target-share", default=0.3, type=float)
    parser.add_argument("--trend-strengths", nargs="*", default=None, type=float)
    parser.add_argument("--beta", default=0.3, type=float)
    parser.add_argument("--eta", default=0.1, type=float)
    parser.add_argument("--dt", default=0.5, type=float)
    parser.add_argument("--seed-strength", default=0.5, type=float)
    parser.add_argument("--action-scale", default=0.1, type=float)
    parser.add_argument(
        "--decoder-mode",
        default="current",
        choices=[
            "current",
            "additive",
            "spectral_frontier",
            "frontier",
            "raw_frontier",
            "signed_frontier",
        ],
    )
    parser.add_argument("--timesteps", default=100_000, type=int)
    parser.add_argument("--episodes", default=10, type=int)
    parser.add_argument("--num-seeds", default=20, type=int)
    parser.add_argument("--budget", default=10, type=int)
    parser.add_argument("--boost-budget", default=None, type=int)
    parser.add_argument("--suppress-budget", default=None, type=int)
    parser.add_argument("--target-adoption", default=0.3, type=float)
    parser.add_argument("--fatigue-weight", default=0.5, type=float)
    parser.add_argument("--fatigue-inequality-weight", default=0.1, type=float)
    parser.add_argument("--energy-weight", default=0.1, type=float)
    parser.add_argument("--intervention-threshold", default=0.1, type=float)
    parser.add_argument("--max-steps", default=100, type=int)
    parser.add_argument("--n-steps", default=2048, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--spectral-modes", default=20, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--model-path", default=ROOT / "models/ppo_contagion", type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    return parser.parse_args()


def make_env(args):
    graph, adjacency, nodes = load_facebook_graph(args.graph)
    n = len(nodes)
    sigma, rho, _ = assign_archetypes(n, seed=args.seed)
    node_features = precompute_node_features(graph, adjacency, nodes, k=args.spectral_modes)
    if args.multi_trend:
        rng = np.random.default_rng(args.seed)
        trend_embeddings = generate_trend_embeddings(args.num_trends, args.embedding_dim, rng)
        node_interests = sample_node_interests(n, args.embedding_dim, rng)
        affinity = compute_affinity_matrix(node_interests, trend_embeddings)
        if args.trend_strengths is not None:
            strengths = np.asarray(args.trend_strengths, dtype=np.float32)
            if strengths.size != args.num_trends:
                raise ValueError(
                    f"--trend-strengths must have {args.num_trends} values, got {strengths.size}"
                )
            affinity = np.clip(affinity * strengths[None, :], 0.0, 1.0)
        affinity[:, 0] = 0.5
        return MultiTrendContagionEnv(
            adjacency,
            sigma,
            rho,
            affinity,
            target_trend=args.target_trend,
            target_share=args.target_share,
            budget=args.budget,
            boost_budget=args.boost_budget,
            suppress_budget=args.suppress_budget,
            max_steps=args.max_steps,
            num_seeds=args.num_seeds,
            seed_strength=args.seed_strength,
            action_scale=args.action_scale,
            beta=args.beta,
            eta=args.eta,
            dt=args.dt,
            decoder_mode=args.decoder_mode,
            node_features=node_features,
            reward_weights=RewardWeights(
                fatigue=args.fatigue_weight,
                fatigue_inequality=args.fatigue_inequality_weight,
                energy=args.energy_weight,
            ),
        )

    affinity = sample_affinity(n, seed=args.seed)
    return ContagionEnv(
        adjacency,
        sigma,
        rho,
        affinity,
        budget=args.budget,
        boost_budget=args.boost_budget,
        suppress_budget=args.suppress_budget,
        target_adoption=args.target_adoption,
        max_steps=args.max_steps,
        num_seeds=args.num_seeds,
        node_features=node_features,
        reward_weights=RewardWeights(
            fatigue=args.fatigue_weight,
            fatigue_inequality=args.fatigue_inequality_weight,
            energy=args.energy_weight,
        ),
    )


def action_none(env):
    return np.zeros(env.action_space.shape, dtype=np.float32)


def action_random(env):
    return env.rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)


def action_greedy_degree(env):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 1.0
    action[-1] = 1.0
    return action


def action_raw_frontier(env):
    action = np.ones(env.action_space.shape, dtype=np.float32)
    if getattr(env, "decoder_mode", None) != "raw_frontier":
        return action_greedy_degree(env)
    return action


def action_target_frontier(env):
    action = np.ones(env.action_space.shape, dtype=np.float32)
    if getattr(env, "decoder_mode", None) != "signed_frontier":
        return action_raw_frontier(env)
    target_share = float(env.v[:, env.target_trend].mean())
    if target_share > env.target_share:
        action *= -1.0
    return action


def evaluate_policy(env, policy, episodes: int, intervention_threshold: float):
    rewards = []
    adoption_errors = []
    fatigue = []
    energy = []
    intervention_rates = []
    action_magnitudes = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        done = False
        total_reward = 0.0
        episode_adoption_errors = []
        episode_fatigue = []
        episode_energy = []
        episode_action_magnitudes = []
        intervention_steps = 0
        step_count = 0
        while not done:
            action = policy(obs, env)
            action_magnitude = float(np.mean(np.abs(action)))
            if action_magnitude > intervention_threshold:
                intervention_steps += 1
            episode_action_magnitudes.append(action_magnitude)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            episode_adoption_errors.append(abs(info["mean_adoption"] - env.target_adoption))
            episode_fatigue.append(info["mean_fatigue"])
            episode_energy.append(info["action_energy"])
            step_count += 1
        rewards.append(total_reward)
        adoption_errors.append(float(np.mean(episode_adoption_errors)))
        fatigue.append(float(np.mean(episode_fatigue)))
        energy.append(float(np.mean(episode_energy)))
        intervention_rates.append(intervention_steps / max(step_count, 1))
        action_magnitudes.append(float(np.mean(episode_action_magnitudes)))
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_adoption_deviation": float(np.mean(adoption_errors)),
        "mean_fatigue": float(np.mean(fatigue)),
        "mean_action_energy": float(np.mean(energy)),
        "mean_action_magnitude": float(np.mean(action_magnitudes)),
        "intervention_rate": float(np.mean(intervention_rates)),
    }


def main():
    args = parse_args()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.model_path = args.output_dir / "ppo_contagion"

    env = make_env(args)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=args.device,
    )
    model.learn(total_timesteps=args.timesteps)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    print(f"Saved model to {args.model_path}")

    policies = {
        "ppo": lambda obs, env: model.predict(obs, deterministic=True)[0],
        "random": lambda obs, env: action_random(env),
        "none": lambda obs, env: action_none(env),
        "greedy_degree": lambda obs, env: action_greedy_degree(env),
    }
    if args.multi_trend:
        policies["raw_frontier"] = lambda obs, env: action_raw_frontier(env)
        if args.decoder_mode == "signed_frontier":
            policies["target_frontier"] = lambda obs, env: action_target_frontier(env)
    for name, policy in policies.items():
        metrics = evaluate_policy(env, policy, args.episodes, args.intervention_threshold)
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    main()
