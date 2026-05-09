from __future__ import annotations

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot signed-frontier PPO target-share trajectories against baselines."
    )
    parser.add_argument("--graph", default=ROOT / "data/facebook_combined.txt", type=Path)
    parser.add_argument("--model-dir", default=ROOT / "results/ppo_signed_frontier_target_sweep", type=Path)
    parser.add_argument("--output-dir", default=ROOT / "results/visualizations", type=Path)
    parser.add_argument("--output-name", default="signed_frontier_ppo_vs_baselines")
    parser.add_argument("--targets", nargs="+", default=[0.2, 0.3, 0.5], type=float)
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
    parser.add_argument("--rollout-seed", default=42, type=int)
    parser.add_argument("--dpi", default=160, type=int)
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


def action_none(obs, env):
    return np.zeros(env.action_space.shape, dtype=np.float32)


def action_always_boost(obs, env):
    return np.ones(env.action_space.shape, dtype=np.float32)


def action_target_frontier(obs, env):
    action = np.ones(env.action_space.shape, dtype=np.float32)
    if float(env.v[:, env.target_trend].mean()) > env.target_share:
        action *= -1.0
    return action


def run_rollout(env, policy_fn, seed: int):
    obs, _ = env.reset(seed=seed)
    shares = [float(env.v[:, env.target_trend].mean())]
    deviations = [abs(shares[-1] - env.target_share)]
    actions = [0.0]
    rewards = []
    done = False
    while not done:
        action = np.asarray(policy_fn(obs, env), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        shares.append(float(info["target_share"]))
        deviations.append(float(info["target_deviation"]))
        actions.append(float(np.mean(action)))
        rewards.append(float(reward))
        done = terminated or truncated
    return {
        "shares": np.asarray(shares, dtype=np.float32),
        "deviations": np.asarray(deviations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "reward": float(np.sum(rewards)),
    }


def plot_results(args, results):
    fig, axes = plt.subplots(2, len(args.targets), figsize=(5.2 * len(args.targets), 7.2), sharex=True)
    if len(args.targets) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    colors = {
        "No intervention": "#555555",
        "Always boost frontier": "#2f9e44",
        "Target-aware frontier": "#3478bf",
        "Signed-frontier PPO": "#d94841",
    }

    for col, target in enumerate(args.targets):
        target_results = results[target]
        share_ax = axes[0, col]
        dev_ax = axes[1, col]
        for name, rollout in target_results.items():
            share_ax.plot(rollout["shares"], label=name, linewidth=2.2, color=colors[name])
            dev_ax.plot(rollout["deviations"], label=name, linewidth=2.2, color=colors[name])

        share_ax.axhline(target, color="black", linestyle="--", linewidth=1.0)
        share_ax.set_title(f"Target share = {target:.2f}")
        share_ax.set_ylabel(f"Trend {args.target_trend} share")
        share_ax.set_ylim(0.0, max(0.55, target + 0.08))
        share_ax.grid(alpha=0.22)

        dev_ax.set_xlabel("Step")
        dev_ax.set_ylabel("Absolute deviation")
        dev_ax.set_ylim(bottom=0.0)
        dev_ax.grid(alpha=0.22)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Signed-frontier PPO vs frontier baselines", y=0.98, fontsize=15)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{args.output_name}.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    static_inputs = build_static_inputs(args)
    all_results = {}
    for target in args.targets:
        env = make_env(args, static_inputs, target)
        model_path = args.model_dir / target_label(target) / "ppo_contagion.zip"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        model = PPO.load(model_path, env=env, device="cpu")
        policies = {
            "No intervention": action_none,
            "Always boost frontier": action_always_boost,
            "Target-aware frontier": action_target_frontier,
            "Signed-frontier PPO": lambda obs, env, model=model: model.predict(
                obs,
                deterministic=True,
            )[0],
        }
        all_results[target] = {
            name: run_rollout(make_env(args, static_inputs, target), policy, args.rollout_seed)
            for name, policy in policies.items()
        }

    path = plot_results(args, all_results)
    print(f"Wrote {path}")
    for target, target_results in all_results.items():
        print(f"target={target:.2f}")
        for name, rollout in target_results.items():
            print(
                f"  {name}: final_share={rollout['shares'][-1]:.4f}, "
                f"mean_deviation={rollout['deviations'].mean():.4f}, "
                f"reward={rollout['reward']:.4f}"
            )


if __name__ == "__main__":
    main()
