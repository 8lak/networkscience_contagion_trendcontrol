from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

from src.env import MultiTrendContagionEnv, RewardWeights
from src.graph import (
    assign_archetypes,
    compute_affinity_matrix,
    generate_trend_embeddings,
    load_facebook_graph,
    precompute_node_features,
    sample_node_interests,
)


TREND_COLORS = ["#b8b8b8", "#d94841", "#3478bf", "#2f9e44", "#8e5cc2", "#e39d25"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize multi-trend contagion rollouts on the graph."
    )
    parser.add_argument("--graph", default=ROOT / "data/facebook_combined.txt", type=Path)
    parser.add_argument("--output-dir", default=ROOT / "results/visualizations", type=Path)
    parser.add_argument("--output-name", default="multi_trend_noop_vs_raw_frontier")
    parser.add_argument("--target-trend", default=3, type=int)
    parser.add_argument("--target-share", default=0.3, type=float)
    parser.add_argument("--trend-strengths", nargs="*", default=[1.0, 1.25, 1.0, 0.75], type=float)
    parser.add_argument("--num-trends", default=4, type=int)
    parser.add_argument("--embedding-dim", default=10, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--frame-stride", default=3, type=int)
    parser.add_argument("--spectral-modes", default=5, type=int)
    parser.add_argument("--boost-budget", default=400, type=int)
    parser.add_argument("--suppress-budget", default=400, type=int)
    parser.add_argument("--action-scale", default=0.5, type=float)
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
    parser.add_argument("--layout", choices=["spectral", "spring"], default="spectral")
    parser.add_argument("--spring-iterations", default=50, type=int)
    parser.add_argument("--edge-sample", default=12000, type=int)
    parser.add_argument("--node-size", default=8.0, type=float)
    parser.add_argument("--fps", default=5, type=int)
    parser.add_argument("--dpi", default=120, type=int)
    parser.add_argument("--no-gif", action="store_true")
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
    indexed_graph = nx.from_scipy_sparse_array(adjacency)
    return indexed_graph, adjacency, sigma, rho, affinity, node_features


def make_env(args, static_inputs, decoder_mode="raw_frontier"):
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


def action_raw_frontier(env):
    return np.ones(env.action_space.shape, dtype=np.float32)


def run_rollout(env, policy_fn, seed: int, steps: int):
    obs, _ = env.reset(seed=seed)
    v_history = [env.v.copy()]
    shares = [float(env.v[:, env.target_trend].mean())]
    fatigue = [float(env.phi[:, env.target_trend].mean())]
    rewards = []
    energies = []
    for _ in range(steps):
        action = policy_fn(env)
        obs, reward, terminated, truncated, info = env.step(action)
        v_history.append(env.v.copy())
        shares.append(float(info["target_share"]))
        fatigue.append(float(info["target_fatigue"]))
        rewards.append(float(reward))
        energies.append(float(info["action_energy"]))
        if terminated or truncated:
            break
    return {
        "v": np.stack(v_history),
        "shares": np.asarray(shares, dtype=np.float32),
        "fatigue": np.asarray(fatigue, dtype=np.float32),
        "reward": float(np.sum(rewards)),
        "energy": float(np.mean(energies)) if energies else 0.0,
    }


def spectral_positions(eigenvectors: np.ndarray) -> np.ndarray:
    if eigenvectors.shape[1] >= 3:
        coords = eigenvectors[:, 1:3].copy()
    elif eigenvectors.shape[1] == 2:
        coords = eigenvectors[:, :2].copy()
    else:
        coords = np.column_stack([np.arange(eigenvectors.shape[0]), np.zeros(eigenvectors.shape[0])])
    coords -= coords.mean(axis=0, keepdims=True)
    scale = np.maximum(np.ptp(coords, axis=0, keepdims=True), 1e-8)
    return coords / scale


def get_positions(args, graph, node_features):
    cache = args.output_dir / f"layout_{args.layout}_n{graph.number_of_nodes()}_k{args.spectral_modes}.npz"
    if cache.exists():
        return np.load(cache)["pos"]

    if args.layout == "spring":
        initial = {
            i: xy for i, xy in enumerate(spectral_positions(node_features["eigenvectors"]))
        }
        pos_dict = nx.spring_layout(
            graph,
            pos=initial,
            iterations=args.spring_iterations,
            seed=args.seed,
        )
        pos = np.asarray([pos_dict[i] for i in range(graph.number_of_nodes())], dtype=np.float32)
    else:
        pos = spectral_positions(node_features["eigenvectors"]).astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, pos=pos)
    return pos


def edge_segments(graph, pos, edge_sample: int, seed: int):
    edges = np.asarray(list(graph.edges()), dtype=np.int64)
    if 0 < edge_sample < len(edges):
        rng = np.random.default_rng(seed)
        edges = edges[rng.choice(len(edges), size=edge_sample, replace=False)]
    return np.stack([pos[edges[:, 0]], pos[edges[:, 1]]], axis=1)


def draw_frame(ax, title, rollout, frame_idx, pos, segments, colors, node_size, target_trend):
    ax.clear()
    ax.add_collection(LineCollection(segments, colors="#d6d6d6", linewidths=0.15, alpha=0.18))
    v = rollout["v"][frame_idx]
    dominant = np.argmax(v, axis=1)
    target_share = float(v[:, target_trend].mean())
    target_strength = np.clip(v[:, target_trend], 0.0, 1.0)
    sizes = node_size + node_size * 2.5 * target_strength
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=dominant,
        cmap=ListedColormap(colors),
        vmin=0,
        vmax=len(colors) - 1,
        s=sizes,
        alpha=0.86,
        linewidths=0,
    )
    ax.set_title(f"{title}\nt={frame_idx}, target share={target_share:.3f}", fontsize=12)
    ax.set_aspect("equal")
    ax.axis("off")
    pad = 0.05
    ax.set_xlim(float(pos[:, 0].min() - pad), float(pos[:, 0].max() + pad))
    ax.set_ylim(float(pos[:, 1].min() - pad), float(pos[:, 1].max() + pad))


def save_share_plot(args, noop, frontier):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(noop["shares"], label="No intervention", linewidth=2)
    ax.plot(frontier["shares"], label="Raw frontier", linewidth=2)
    ax.axhline(args.target_share, color="black", linestyle="--", linewidth=1, label="Target")
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Trend {args.target_trend} mean share")
    ax.set_title("Target trend share over time")
    ax.legend()
    fig.tight_layout()
    path = args.output_dir / f"{args.output_name}_share_curve.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    static_inputs = build_inputs(args)
    graph, _, _, _, _, node_features = static_inputs

    noop_env = make_env(args, static_inputs, decoder_mode="raw_frontier")
    frontier_env = make_env(args, static_inputs, decoder_mode="raw_frontier")
    noop = run_rollout(noop_env, action_none, args.rollout_seed, args.steps)
    frontier = run_rollout(frontier_env, action_raw_frontier, args.rollout_seed, args.steps)

    pos = get_positions(args, graph, node_features)
    segments = edge_segments(graph, pos, args.edge_sample, args.seed)
    colors = TREND_COLORS[: args.num_trends]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    final_idx = min(len(noop["v"]), len(frontier["v"])) - 1
    draw_frame(
        axes[0],
        "No intervention",
        noop,
        final_idx,
        pos,
        segments,
        colors,
        args.node_size,
        args.target_trend,
    )
    draw_frame(
        axes[1],
        "Raw frontier policy",
        frontier,
        final_idx,
        pos,
        segments,
        colors,
        args.node_size,
        args.target_trend,
    )
    fig.tight_layout()
    final_png = args.output_dir / f"{args.output_name}_final.png"
    fig.savefig(final_png, dpi=args.dpi)

    share_png = save_share_plot(args, noop, frontier)
    gif_path = args.output_dir / f"{args.output_name}.gif"
    if not args.no_gif:
        frame_indices = list(range(0, final_idx + 1, max(args.frame_stride, 1)))
        if frame_indices[-1] != final_idx:
            frame_indices.append(final_idx)

        def animate(frame_idx):
            draw_frame(
                axes[0],
                "No intervention",
                noop,
                frame_idx,
                pos,
                segments,
                colors,
                args.node_size,
                args.target_trend,
            )
            draw_frame(
                axes[1],
                "Raw frontier policy",
                frontier,
                frame_idx,
                pos,
                segments,
                colors,
                args.node_size,
                args.target_trend,
            )
            return axes

        animation = FuncAnimation(fig, animate, frames=frame_indices, interval=1000 / args.fps)
        animation.save(gif_path, writer="pillow", fps=args.fps, dpi=args.dpi)
    plt.close(fig)

    print("Visualization complete")
    print(f"  final_png: {final_png}")
    print(f"  share_png: {share_png}")
    if not args.no_gif:
        print(f"  gif:       {gif_path}")
    print(
        "  no-op:    "
        f"final_share={noop['shares'][-1]:.4f}, "
        f"mean_share={noop['shares'].mean():.4f}, "
        f"reward={noop['reward']:.4f}, "
        f"energy={noop['energy']:.4f}"
    )
    print(
        "  frontier: "
        f"final_share={frontier['shares'][-1]:.4f}, "
        f"mean_share={frontier['shares'].mean():.4f}, "
        f"reward={frontier['reward']:.4f}, "
        f"energy={frontier['energy']:.4f}"
    )


if __name__ == "__main__":
    main()
