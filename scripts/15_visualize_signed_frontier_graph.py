from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import networkx as nx
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
        description="Render graph heatmaps comparing signed-frontier PPO to frontier baselines."
    )
    parser.add_argument("--graph", default=ROOT / "data/facebook_combined.txt", type=Path)
    parser.add_argument("--model-dir", default=ROOT / "results/ppo_signed_frontier_target_sweep", type=Path)
    parser.add_argument("--output-dir", default=ROOT / "results/visualizations", type=Path)
    parser.add_argument("--output-name", default="signed_frontier_ppo_graph_target03")
    parser.add_argument("--target-share", default=0.3, type=float)
    parser.add_argument("--target-trend", default=3, type=int)
    parser.add_argument("--trend-strengths", nargs="*", default=[1.0, 1.25, 1.0, 0.75], type=float)
    parser.add_argument("--num-trends", default=4, type=int)
    parser.add_argument("--embedding-dim", default=10, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--snapshots", nargs="*", default=[0, 25, 50, 75, 100], type=int)
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
    parser.add_argument("--layout", choices=["cached", "spring", "spectral"], default="cached")
    parser.add_argument("--layout-iterations", default=40, type=int)
    parser.add_argument("--bins", default=150, type=int)
    parser.add_argument("--min-bin-count", default=1, type=int)
    parser.add_argument("--vmax", default=0.45, type=float)
    parser.add_argument("--dpi", default=170, type=int)
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
    indexed_graph = nx.from_scipy_sparse_array(adjacency)
    return indexed_graph, adjacency, sigma, rho, affinity, node_features


def make_env(args, static_inputs):
    _, adjacency, sigma, rho, affinity, node_features = static_inputs
    return MultiTrendContagionEnv(
        adjacency,
        sigma,
        rho,
        affinity,
        target_trend=args.target_trend,
        target_share=args.target_share,
        boost_budget=args.boost_budget,
        suppress_budget=args.suppress_budget,
        max_steps=args.steps,
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


def spectral_positions(eigenvectors: np.ndarray) -> np.ndarray:
    if eigenvectors.shape[1] >= 3:
        pos = eigenvectors[:, 1:3].copy()
    elif eigenvectors.shape[1] == 2:
        pos = eigenvectors[:, :2].copy()
    else:
        pos = np.column_stack([np.arange(eigenvectors.shape[0]), np.zeros(eigenvectors.shape[0])])
    return normalize_positions(pos)


def normalize_positions(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    pos -= pos.mean(axis=0, keepdims=True)
    scale = np.maximum(np.ptp(pos, axis=0, keepdims=True), 1e-8)
    return pos / scale


def get_positions(args, graph, node_features):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cached = [
        args.output_dir / f"layout_forceatlas2_n{graph.number_of_nodes()}_iter100_seed{args.seed}.npz",
        args.output_dir / f"layout_spring_n{graph.number_of_nodes()}_iter40_seed{args.seed}.npz",
    ]
    if args.layout == "cached":
        for path in cached:
            if path.exists():
                pos = np.load(path)["pos"]
                if np.all(np.isfinite(pos)):
                    return pos

    initial_array = spectral_positions(node_features["eigenvectors"])
    if args.layout == "spectral":
        return initial_array

    initial = {i: initial_array[i] for i in range(graph.number_of_nodes())}
    pos_dict = nx.spring_layout(
        graph,
        pos=initial,
        iterations=args.layout_iterations,
        seed=args.seed,
    )
    pos = normalize_positions(
        np.asarray([pos_dict[i] for i in range(graph.number_of_nodes())], dtype=np.float32)
    )
    cache = args.output_dir / f"layout_spring_n{graph.number_of_nodes()}_iter{args.layout_iterations}_seed{args.seed}.npz"
    np.savez_compressed(cache, pos=pos)
    return pos


def action_none(obs, env):
    return np.zeros(env.action_space.shape, dtype=np.float32)


def action_always_boost(obs, env):
    return np.ones(env.action_space.shape, dtype=np.float32)


def run_rollout(env, policy_fn, seed: int):
    obs, _ = env.reset(seed=seed)
    history = [env.v.copy()]
    shares = [float(env.v[:, env.target_trend].mean())]
    done = False
    while not done:
        action = np.asarray(policy_fn(obs, env), dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        history.append(env.v.copy())
        shares.append(float(info["target_share"]))
        done = terminated or truncated
    return np.stack(history), np.asarray(shares, dtype=np.float32)


def target_grid(pos, values, bins: int, min_count: int):
    x = pos[:, 0]
    y = pos[:, 1]
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    counts, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[extent[0], extent[1]], [extent[2], extent[3]]],
    )
    weighted, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=values)
    grid = weighted / np.maximum(counts, 1.0)
    grid[counts < min_count] = np.nan
    return grid.T, extent


def plot_heatmap(args, pos, rollouts):
    snapshots = [
        step
        for step in args.snapshots
        if all(0 <= step < len(history) for history, _ in rollouts.values())
    ]
    rows = [
        ("No intervention", "No intervention"),
        ("Always boost frontier", "Boost frontier"),
        ("Signed-frontier PPO", "Signed PPO"),
    ]

    fig, axes = plt.subplots(
        len(rows),
        len(snapshots),
        figsize=(3.15 * len(snapshots), 2.8 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )

    last_image = None
    for row_idx, (key, label) in enumerate(rows):
        history, shares = rollouts[key]
        for col_idx, step in enumerate(snapshots):
            ax = axes[row_idx, col_idx]
            grid, extent = target_grid(
                pos,
                history[step][:, args.target_trend],
                args.bins,
                args.min_bin_count,
            )
            last_image = ax.imshow(
                grid,
                origin="lower",
                extent=extent,
                cmap="Greens",
                vmin=0.0,
                vmax=args.vmax,
                interpolation="bilinear",
            )
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=12)
            ax.set_title(f"t={step}\nshare={shares[step]:.3f}", fontsize=10)
            ax.set_aspect("equal")
            ax.axis("off")

    if last_image is not None:
        fig.colorbar(
            last_image,
            ax=axes,
            location="right",
            shrink=0.82,
            label=f"Trend {args.target_trend} share",
        )
    fig.suptitle(
        f"Graph heatmap: signed-frontier PPO vs frontier boost, target={args.target_share:.2f}",
        fontsize=15,
    )
    path = args.output_dir / f"{args.output_name}.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = parse_args()
    static_inputs = build_static_inputs(args)
    graph, *_rest, node_features = static_inputs
    pos = get_positions(args, graph, node_features)

    model_path = args.model_dir / target_label(args.target_share) / "ppo_contagion.zip"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model_env = make_env(args, static_inputs)
    model = PPO.load(model_path, env=model_env, device="cpu")

    policies = {
        "No intervention": action_none,
        "Always boost frontier": action_always_boost,
        "Signed-frontier PPO": lambda obs, env: model.predict(obs, deterministic=True)[0],
    }
    rollouts = {
        name: run_rollout(make_env(args, static_inputs), policy, args.rollout_seed)
        for name, policy in policies.items()
    }
    path = plot_heatmap(args, pos, rollouts)
    print(f"Wrote {path}")
    for name, (_history, shares) in rollouts.items():
        print(f"  {name}: final_share={shares[-1]:.4f}, mean_share={shares.mean():.4f}")


if __name__ == "__main__":
    main()
