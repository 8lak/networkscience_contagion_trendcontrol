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

from src.env import ContagionEnv, RewardWeights
from src.graph import assign_archetypes, load_facebook_graph, precompute_node_features, sample_affinity


POLICIES = ("ppo", "random", "none", "greedy_degree")


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep PPO energy penalty values.")
    parser.add_argument("--graph", default=ROOT / "data/facebook_combined.txt", type=Path)
    parser.add_argument("--weights", nargs="+", default=[0.0, 0.01, 0.02, 0.05, 0.1], type=float)
    parser.add_argument("--target-adoption", default=0.3, type=float)
    parser.add_argument("--timesteps", default=150_000, type=int)
    parser.add_argument("--episodes", default=10, type=int)
    parser.add_argument("--budget", default=10, type=int)
    parser.add_argument("--boost-budget", default=None, type=int)
    parser.add_argument("--suppress-budget", default=None, type=int)
    parser.add_argument("--max-steps", default=100, type=int)
    parser.add_argument("--n-steps", default=2048, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--spectral-modes", default=5, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--fatigue-weight", default=0.5, type=float)
    parser.add_argument("--fatigue-inequality-weight", default=0.1, type=float)
    parser.add_argument("--intervention-threshold", default=0.1, type=float)
    parser.add_argument("--output-dir", default=ROOT / "results/energy_sweep", type=Path)
    return parser.parse_args()


def make_env(args, node_features, energy_weight: float):
    _, adjacency, nodes = load_facebook_graph(args.graph)
    n = len(nodes)
    sigma, rho, _ = assign_archetypes(n, seed=args.seed)
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
        node_features=node_features,
        reward_weights=RewardWeights(
            fatigue=args.fatigue_weight,
            fatigue_inequality=args.fatigue_inequality_weight,
            energy=energy_weight,
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
        "reward": float(np.mean(rewards)),
        "deviation": float(np.mean(adoption_errors)),
        "fatigue": float(np.mean(fatigue)),
        "energy": float(np.mean(energy)),
        "action_magnitude": float(np.mean(action_magnitudes)),
        "intervention_rate": float(np.mean(intervention_rates)),
    }


def train_model(env, args):
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
    return model


def flatten_row(energy_weight: float, metrics_by_policy: dict[str, dict[str, float]]):
    row = {"energy_weight": energy_weight}
    for policy in POLICIES:
        for metric, value in metrics_by_policy[policy].items():
            row[f"{policy}_{metric}"] = value
    row["ppo_noop_reward_gap"] = row["ppo_reward"] - row["none_reward"]
    return row


def write_csv(path: Path, rows: list[dict[str, float]]):
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_line(rows, y_key: str, ylabel: str, output_path: Path, baseline_key: str | None = None):
    weights = [row["energy_weight"] for row in rows]
    values = [row[y_key] for row in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(weights, values, marker="o", label=y_key)
    if baseline_key is not None:
        baseline = [row[baseline_key] for row in rows]
        plt.plot(weights, baseline, marker="s", label=baseline_key)
        plt.legend()
    plt.xlabel("energy_weight")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_pareto(rows, output_path: Path):
    plt.figure(figsize=(7, 4))
    plt.plot(
        [row["ppo_energy"] for row in rows],
        [row["ppo_deviation"] for row in rows],
        marker="o",
    )
    for row in rows:
        plt.annotate(f'{row["energy_weight"]:.2g}', (row["ppo_energy"], row["ppo_deviation"]))
    plt.xlabel("PPO action energy")
    plt.ylabel("PPO adoption deviation")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def make_plots(output_dir: Path, rows: list[dict[str, float]]):
    plot_line(
        rows,
        "ppo_noop_reward_gap",
        "PPO reward - no-op reward",
        output_dir / "ppo_noop_reward_gap.png",
    )
    plot_line(rows, "ppo_energy", "PPO action energy", output_dir / "ppo_action_energy.png")
    plot_line(
        rows,
        "ppo_deviation",
        "Adoption deviation",
        output_dir / "adoption_deviation.png",
        baseline_key="none_deviation",
    )
    plot_pareto(rows, output_dir / "pareto_deviation_vs_energy.png")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    graph, adjacency, nodes = load_facebook_graph(args.graph)
    node_features = precompute_node_features(graph, adjacency, nodes, k=args.spectral_modes)

    rows = []
    for energy_weight in args.weights:
        run_dir = args.output_dir / f"energy_w{energy_weight:g}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== energy_weight={energy_weight:g} ===")

        env = make_env(args, node_features, energy_weight)
        model = train_model(env, args)
        model.save(run_dir / "ppo_contagion")

        policies = {
            "ppo": lambda obs, env, model=model: model.predict(obs, deterministic=True)[0],
            "random": lambda obs, env: action_random(env),
            "none": lambda obs, env: action_none(env),
            "greedy_degree": lambda obs, env: action_greedy_degree(env),
        }
        metrics_by_policy = {}
        for name, policy in policies.items():
            metrics = evaluate_policy(env, policy, args.episodes, args.intervention_threshold)
            metrics_by_policy[name] = metrics
            print(f"{name}: {metrics}")

        rows.append(flatten_row(energy_weight, metrics_by_policy))
        write_csv(args.output_dir / "energy_sweep.csv", rows)

    make_plots(args.output_dir, rows)
    print(f"\nSaved sweep outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
