from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired significance tests for signed-frontier PPO sweep."
    )
    parser.add_argument("--graph", default=ROOT / "data/facebook_combined.txt", type=Path)
    parser.add_argument("--model-dir", default=ROOT / "results/ppo_signed_frontier_target_sweep", type=Path)
    parser.add_argument("--output", default=ROOT / "results/ppo_signed_frontier_target_sweep/significance_tests.csv", type=Path)
    parser.add_argument("--targets", nargs="+", default=[0.2, 0.3, 0.5], type=float)
    parser.add_argument("--episodes", default=10, type=int)
    parser.add_argument("--target-trend", default=3, type=int)
    parser.add_argument("--trend-strengths", nargs="*", default=[1.0, 1.25, 1.0, 0.75], type=float)
    parser.add_argument("--num-trends", default=4, type=int)
    parser.add_argument("--embedding-dim", default=10, type=int)
    parser.add_argument("--max-steps", default=100, type=int)
    parser.add_argument("--spectral-modes", default=5, type=int)
    parser.add_argument("--boost-budget", default=1200, type=int)
    parser.add_argument("--suppress-budget", default=1200, type=int)
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
    return parser.parse_args()


def target_label(target: float) -> str:
    return f"t{target:.2f}".replace(".", "p")


def build_static_inputs(args):
    graph, adjacency, nodes = load_facebook_graph(args.graph)
    n = len(nodes)
    sigma, rho, _ = assign_archetypes(n, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    trend_embeddings = generate_trend_embeddings(args.num_trends, args.embedding_dim, rng)
    node_interests = sample_node_interests(n, args.embedding_dim, rng)
    affinity = compute_affinity_matrix(node_interests, trend_embeddings)
    strengths = np.asarray(args.trend_strengths, dtype=np.float32)
    if strengths.size != args.num_trends:
        raise ValueError(f"--trend-strengths must have {args.num_trends} values")
    affinity = np.clip(affinity * strengths[None, :], 0.0, 1.0)
    affinity[:, 0] = 0.5
    node_features = precompute_node_features(graph, adjacency, nodes, k=args.spectral_modes)
    return adjacency, sigma, rho, affinity, node_features


def make_env(args, static_inputs, target_share: float):
    adjacency, sigma, rho, affinity, node_features = static_inputs
    return MultiTrendContagionEnv(
        adjacency,
        sigma,
        rho,
        affinity,
        target_trend=args.target_trend,
        target_share=target_share,
        boost_budget=args.boost_budget,
        suppress_budget=args.suppress_budget,
        max_steps=args.max_steps,
        num_seeds=args.num_seeds,
        seed_strength=args.seed_strength,
        action_scale=args.action_scale,
        beta=args.beta,
        eta=args.eta,
        dt=args.dt,
        decoder_mode="signed_frontier",
        node_features=node_features,
        reward_weights=RewardWeights(
            fatigue=args.fatigue_weight,
            fatigue_inequality=args.fatigue_inequality_weight,
            energy=args.energy_weight,
        ),
    )


def action_none(env):
    return np.zeros(env.action_space.shape, dtype=np.float32)


def action_target_frontier(env):
    action = np.ones(env.action_space.shape, dtype=np.float32)
    if float(env.v[:, env.target_trend].mean()) > env.target_share:
        action *= -1.0
    return action


def action_random(env):
    return env.rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)


def evaluate(env, policy_fn, episodes: int):
    rows = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        done = False
        total_reward = 0.0
        deviations = []
        fatigue = []
        energy = []
        final_info = {}
        while not done:
            action = policy_fn(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            deviations.append(float(info["target_deviation"]))
            fatigue.append(float(info["target_fatigue"]))
            energy.append(float(info["action_energy"]))
            final_info = info
        rows.append(
            {
                "episode": episode,
                "reward": total_reward,
                "deviation": float(np.mean(deviations)),
                "fatigue": float(np.mean(fatigue)),
                "energy": float(np.mean(energy)),
                "final_share": float(final_info["target_share"]),
            }
        )
    return rows


def summarize(values: np.ndarray):
    return float(np.mean(values)), float(np.std(values, ddof=1))


def paired_row(target, metric, policy_a, policy_b, rows_a, rows_b):
    a = np.asarray([row[metric] for row in rows_a], dtype=np.float64)
    b = np.asarray([row[metric] for row in rows_b], dtype=np.float64)
    t_stat, p_value = stats.ttest_rel(a, b)
    mean_a, std_a = summarize(a)
    mean_b, std_b = summarize(b)
    diff = a - b
    diff_mean, diff_std = summarize(diff)
    cohen_dz = diff_mean / diff_std if diff_std > 1e-12 else np.inf
    return {
        "target_share": target,
        "metric": metric,
        "policy_a": policy_a,
        "policy_b": policy_b,
        "policy_a_mean": mean_a,
        "policy_a_std": std_a,
        "policy_b_mean": mean_b,
        "policy_b_std": std_b,
        "paired_diff_mean": diff_mean,
        "paired_diff_std": diff_std,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohen_dz": float(cohen_dz),
    }


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    static_inputs = build_static_inputs(args)
    output_rows = []
    detail_rows = []

    for target in args.targets:
        env = make_env(args, static_inputs, target)
        model_path = args.model_dir / target_label(target) / "ppo_contagion.zip"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = PPO.load(model_path, env=env, device="cpu")

        policies = {
            "ppo": lambda obs, env, model=model: model.predict(obs, deterministic=True)[0],
            "none": lambda obs, env: action_none(env),
            "target_frontier": lambda obs, env: action_target_frontier(env),
            "random": lambda obs, env: action_random(env),
        }
        results = {name: evaluate(env, policy, args.episodes) for name, policy in policies.items()}

        for name, rows in results.items():
            for row in rows:
                detail_rows.append({"target_share": target, "policy": name, **row})

        for metric in ("reward", "deviation", "fatigue", "energy", "final_share"):
            output_rows.append(
                paired_row(target, metric, "ppo", "none", results["ppo"], results["none"])
            )
            output_rows.append(
                paired_row(
                    target,
                    metric,
                    "ppo",
                    "target_frontier",
                    results["ppo"],
                    results["target_frontier"],
                )
            )

    write_csv(args.output, output_rows)
    write_csv(args.output.with_name("signed_frontier_episode_metrics.csv"), detail_rows)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_name('signed_frontier_episode_metrics.csv')}")
    for row in output_rows:
        if row["metric"] in {"reward", "deviation"} and row["policy_b"] in {"none", "target_frontier"}:
            print(row)


if __name__ == "__main__":
    main()
