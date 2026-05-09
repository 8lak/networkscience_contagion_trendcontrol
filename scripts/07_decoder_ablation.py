from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from src.env import MultiTrendContagionEnv, RewardWeights
from src.graph import (
    assign_archetypes,
    compute_affinity_matrix,
    generate_trend_embeddings,
    load_facebook_graph,
    precompute_node_features,
    sample_node_interests,
)


DECODER_MODES = ("current", "additive", "spectral_frontier", "frontier", "raw_frontier")
POLICIES = ("ppo", "random", "none", "greedy_degree", "raw_frontier")


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-trend decoder ablations.")
    parser.add_argument("--graph", default=ROOT / "data/facebook_combined.txt", type=Path)
    parser.add_argument("--decoder-modes", nargs="+", default=list(DECODER_MODES), choices=DECODER_MODES)
    parser.add_argument("--target-trend", default=3, type=int)
    parser.add_argument("--target-share", default=0.3, type=float)
    parser.add_argument("--trend-strengths", nargs="*", default=[1.0, 1.25, 1.0, 0.75], type=float)
    parser.add_argument("--num-trends", default=4, type=int)
    parser.add_argument("--embedding-dim", default=10, type=int)
    parser.add_argument("--timesteps", default=150_000, type=int)
    parser.add_argument("--episodes", default=10, type=int)
    parser.add_argument("--max-steps", default=100, type=int)
    parser.add_argument("--spectral-modes", default=5, type=int)
    parser.add_argument("--budget", default=10, type=int)
    parser.add_argument("--boost-budget", default=400, type=int)
    parser.add_argument("--suppress-budget", default=400, type=int)
    parser.add_argument("--action-scale", default=0.5, type=float)
    parser.add_argument("--energy-weight", default=0.02, type=float)
    parser.add_argument("--fatigue-weight", default=0.5, type=float)
    parser.add_argument("--fatigue-inequality-weight", default=0.1, type=float)
    parser.add_argument("--intervention-threshold", default=0.1, type=float)
    parser.add_argument("--num-seeds", default=20, type=int)
    parser.add_argument("--seed-strength", default=0.5, type=float)
    parser.add_argument("--beta", default=0.3, type=float)
    parser.add_argument("--eta", default=0.1, type=float)
    parser.add_argument("--dt", default=0.5, type=float)
    parser.add_argument("--n-steps", default=2048, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--output-dir", default=ROOT / "results/multi_trend_decoder_ablation", type=Path)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def build_static_inputs(args):
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


def make_env(args, static_inputs, decoder_mode: str):
    adjacency, sigma, rho, affinity, node_features = static_inputs
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
        decoder_mode=decoder_mode,
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
    if getattr(env, "decoder_mode", None) == "raw_frontier":
        return np.ones(env.action_space.shape, dtype=np.float32)
    return action_greedy_degree(env)


def evaluate_policy(env, policy, episodes: int, intervention_threshold: float):
    rewards = []
    adoption_errors = []
    fatigue = []
    energy = []
    action_magnitudes = []
    intervention_rates = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        done = False
        total_reward = 0.0
        episode_errors = []
        episode_fatigue = []
        episode_energy = []
        episode_magnitudes = []
        intervention_steps = 0
        step_count = 0
        while not done:
            action = policy(obs, env)
            magnitude = float(np.mean(np.abs(action)))
            if magnitude > intervention_threshold:
                intervention_steps += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            episode_errors.append(abs(info["mean_adoption"] - env.target_adoption))
            episode_fatigue.append(info["mean_fatigue"])
            episode_energy.append(info["action_energy"])
            episode_magnitudes.append(magnitude)
            step_count += 1

        rewards.append(total_reward)
        adoption_errors.append(float(np.mean(episode_errors)))
        fatigue.append(float(np.mean(episode_fatigue)))
        energy.append(float(np.mean(episode_energy)))
        action_magnitudes.append(float(np.mean(episode_magnitudes)))
        intervention_rates.append(intervention_steps / max(step_count, 1))

    return {
        "reward": float(np.mean(rewards)),
        "deviation": float(np.mean(adoption_errors)),
        "fatigue": float(np.mean(fatigue)),
        "energy": float(np.mean(energy)),
        "action_magnitude": float(np.mean(action_magnitudes)),
        "intervention_rate": float(np.mean(intervention_rates)),
    }


def train_or_load_model(args, env, run_dir: Path):
    model_path = run_dir / "ppo_contagion.zip"
    if args.skip_existing and model_path.exists():
        print(f"Loading existing model from {model_path}")
        return PPO.load(run_dir / "ppo_contagion", env=env, device=args.device)

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
    model.save(run_dir / "ppo_contagion")
    return model


def flatten_row(decoder_mode: str, metrics_by_policy: dict[str, dict[str, float]]):
    row = {"decoder_mode": decoder_mode}
    for policy in POLICIES:
        for metric, value in metrics_by_policy[policy].items():
            row[f"{policy}_{metric}"] = value
    row["ppo_noop_reward_gap"] = row["ppo_reward"] - row["none_reward"]
    row["ppo_greedy_reward_gap"] = row["ppo_reward"] - row["greedy_degree_reward"]
    return row


def write_csv(path: Path, rows: list[dict[str, float]]):
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows, metric: str, output_path: Path):
    modes = [row["decoder_mode"] for row in rows]
    x = np.arange(len(modes))
    width = 0.15
    plt.figure(figsize=(10, 4))
    for idx, policy in enumerate(POLICIES):
        values = [row[f"{policy}_{metric}"] for row in rows]
        plt.bar(x + (idx - 2) * width, values, width=width, label=policy)
    plt.xticks(x, modes, rotation=20)
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def make_plots(output_dir: Path, rows: list[dict[str, float]]):
    plot_metric(rows, "reward", output_dir / "decoder_rewards.png")
    plot_metric(rows, "deviation", output_dir / "decoder_deviation.png")
    plot_metric(rows, "energy", output_dir / "decoder_energy.png")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    static_inputs = build_static_inputs(args)
    rows = []

    for decoder_mode in args.decoder_modes:
        run_dir = args.output_dir / decoder_mode
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== decoder_mode={decoder_mode} ===")
        env = make_env(args, static_inputs, decoder_mode)
        model = train_or_load_model(args, env, run_dir)

        policies = {
            "ppo": lambda obs, env, model=model: model.predict(obs, deterministic=True)[0],
            "random": lambda obs, env: action_random(env),
            "none": lambda obs, env: action_none(env),
            "greedy_degree": lambda obs, env: action_greedy_degree(env),
            "raw_frontier": lambda obs, env: action_raw_frontier(env),
        }
        metrics_by_policy = {}
        for policy_name, policy in policies.items():
            metrics = evaluate_policy(env, policy, args.episodes, args.intervention_threshold)
            metrics_by_policy[policy_name] = metrics
            print(f"{policy_name}: {metrics}")

        rows.append(flatten_row(decoder_mode, metrics_by_policy))
        write_csv(args.output_dir / "decoder_ablation.csv", rows)

    make_plots(args.output_dir, rows)
    print(f"\nSaved decoder ablation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
