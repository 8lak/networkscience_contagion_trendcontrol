from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import to_rgb

from src.env import MultiTrendContagionEnv, RewardWeights
from src.graph import (
    assign_archetypes,
    compute_affinity_matrix,
    generate_trend_embeddings,
    load_facebook_graph,
    precompute_node_features,
    sample_node_interests,
)


TREND_COLORS = ["#d0d0d0", "#d94841", "#3478bf", "#2f9e44", "#8e5cc2", "#e39d25"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render spatial heatmaps of multi-trend attention over graph layout."
    )
    parser.add_argument("--graph", default=ROOT / "data/facebook_combined.txt", type=Path)
    parser.add_argument("--output-dir", default=ROOT / "results/visualizations", type=Path)
    parser.add_argument("--output-name", default="multi_trend_heatmap_snapshots")
    parser.add_argument("--target-trend", default=3, type=int)
    parser.add_argument("--target-share", default=0.3, type=float)
    parser.add_argument("--trend-strengths", nargs="*", default=[1.0, 1.25, 1.0, 0.75], type=float)
    parser.add_argument("--num-trends", default=4, type=int)
    parser.add_argument("--embedding-dim", default=10, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--snapshots", nargs="*", default=[0, 25, 50, 75, 100], type=int)
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
    parser.add_argument("--layout", choices=["forceatlas2", "spring", "spectral"], default="forceatlas2")
    parser.add_argument("--layout-iterations", default=100, type=int)
    parser.add_argument("--bins", default=150, type=int)
    parser.add_argument("--min-bin-count", default=1, type=int)
    parser.add_argument(
        "--mixture-mode",
        choices=["raw", "non_background", "dominant", "dominant_non_background"],
        default="dominant_non_background",
        help=(
            "How to color the all-trend heatmap. non_background removes trend 0 "
            "from the color blend so competing trends are visible."
        ),
    )
    parser.add_argument("--dpi", default=150, type=int)
    parser.add_argument("--show-nodes", action="store_true")
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
        decoder_mode="raw_frontier",
        node_features=node_features,
        reward_weights=RewardWeights(
            fatigue=args.fatigue_weight,
            fatigue_inequality=args.fatigue_inequality_weight,
            energy=args.energy_weight,
        ),
    )


def spectral_positions(eigenvectors: np.ndarray) -> np.ndarray:
    if eigenvectors.shape[1] >= 3:
        coords = eigenvectors[:, 1:3].copy()
    elif eigenvectors.shape[1] == 2:
        coords = eigenvectors[:, :2].copy()
    else:
        coords = np.column_stack([np.arange(eigenvectors.shape[0]), np.zeros(eigenvectors.shape[0])])
    return normalize_positions(coords)


def normalize_positions(pos: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    pos -= pos.mean(axis=0, keepdims=True)
    scale = np.maximum(np.ptp(pos, axis=0, keepdims=True), 1e-8)
    return pos / scale


def get_positions(args, graph, node_features):
    cache = (
        args.output_dir
        / f"layout_{args.layout}_n{graph.number_of_nodes()}_iter{args.layout_iterations}_seed{args.seed}.npz"
    )
    if cache.exists():
        cached = np.load(cache)["pos"]
        if np.all(np.isfinite(cached)):
            return cached

    initial_array = spectral_positions(node_features["eigenvectors"])
    initial = {i: initial_array[i] for i in range(graph.number_of_nodes())}
    if args.layout == "forceatlas2":
        try:
            pos_dict = nx.forceatlas2_layout(
                graph,
                pos=None,
                max_iter=args.layout_iterations,
                scaling_ratio=1.0,
                gravity=2.0,
                seed=args.seed,
            )
        except Exception:
            pos_dict = nx.spring_layout(
                graph,
                pos=initial,
                iterations=max(20, args.layout_iterations // 4),
                seed=args.seed,
            )
        pos = np.asarray([pos_dict[i] for i in range(graph.number_of_nodes())], dtype=np.float32)
        if not np.all(np.isfinite(pos)):
            pos_dict = nx.spring_layout(
                graph,
                pos=initial,
                iterations=max(20, args.layout_iterations // 4),
                seed=args.seed,
            )
            pos = np.asarray([pos_dict[i] for i in range(graph.number_of_nodes())], dtype=np.float32)
    elif args.layout == "spring":
        pos_dict = nx.spring_layout(
            graph,
            pos=initial,
            iterations=args.layout_iterations,
            seed=args.seed,
        )
        pos = np.asarray([pos_dict[i] for i in range(graph.number_of_nodes())], dtype=np.float32)
    else:
        pos = initial_array

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pos = normalize_positions(pos)
    np.savez_compressed(cache, pos=pos)
    return pos


def action_none(env):
    return np.zeros(env.action_space.shape, dtype=np.float32)


def action_raw_frontier(env):
    return np.ones(env.action_space.shape, dtype=np.float32)


def run_rollout(env, policy_fn, seed: int, steps: int):
    obs, _ = env.reset(seed=seed)
    history = [env.v.copy()]
    target_shares = [float(env.v[:, env.target_trend].mean())]
    for _ in range(steps):
        action = policy_fn(env)
        obs, _, terminated, truncated, info = env.step(action)
        history.append(env.v.copy())
        target_shares.append(float(info["target_share"]))
        if terminated or truncated:
            break
    return np.stack(history), np.asarray(target_shares, dtype=np.float32)


def trend_mixture_image(pos, v, colors, bins: int, min_count: int, mode: str):
    x = pos[:, 0]
    y = pos[:, 1]
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
    counts, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=[[extent[0], extent[1]], [extent[2], extent[3]]])

    palette = np.asarray([to_rgb(color) for color in colors], dtype=np.float32)
    trend_means = []
    for trend in range(v.shape[1]):
        weighted, _, _ = np.histogram2d(
            x,
            y,
            bins=[x_edges, y_edges],
            weights=v[:, trend],
        )
        trend_means.append(weighted / np.maximum(counts, 1.0))
    trend_means = np.stack(trend_means, axis=-1)

    if mode == "raw":
        weights = trend_means
        alpha_signal = np.ones((bins, bins), dtype=np.float32)
    elif mode == "dominant":
        dominant = np.argmax(trend_means, axis=-1)
        rgb = palette[dominant]
        dominance = np.max(trend_means, axis=-1)
        background = np.asarray(to_rgb("#f4f4f4"), dtype=np.float32)
        rgb = background + (rgb - background) * np.clip(dominance[:, :, None] * 2.2, 0.0, 1.0)
        alpha_signal = dominance
        weights = None
    elif mode == "dominant_non_background":
        active = trend_means[:, :, 1:]
        active_mass = np.sum(active, axis=-1)
        dominant_active = np.argmax(active, axis=-1) + 1
        rgb = palette[dominant_active]
        background = np.asarray(to_rgb("#f4f4f4"), dtype=np.float32)
        saturation = np.clip(0.15 + active_mass[:, :, None] * 3.5, 0.0, 1.0)
        rgb = background + (rgb - background) * saturation
        alpha_signal = active_mass
        weights = None
    else:
        non_background = trend_means[:, :, 1:]
        non_background_mass = np.sum(non_background, axis=-1)
        weights = np.zeros_like(trend_means)
        weights[:, :, 1:] = non_background / np.maximum(non_background_mass[:, :, None], 1e-8)
        alpha_signal = non_background_mass

    if weights is not None:
        rgb = np.zeros((bins, bins, 3), dtype=np.float32)
        for trend in range(v.shape[1]):
            rgb += weights[:, :, trend, None] * palette[trend]
        if mode == "non_background":
            background = np.asarray(to_rgb("#f4f4f4"), dtype=np.float32)
            saturation = np.clip(alpha_signal[:, :, None] * 2.4, 0.0, 1.0)
            rgb = background + (rgb - background) * saturation

    occupancy = counts >= min_count
    density_alpha = np.clip(np.log1p(counts) / np.log1p(counts.max()), 0.18, 1.0)
    signal_alpha = np.clip(0.35 + alpha_signal * 1.5, 0.35, 1.0)
    alpha = np.where(occupancy, density_alpha * signal_alpha, 0.0)
    image = np.dstack([np.clip(rgb, 0.0, 1.0), alpha])
    return np.swapaxes(image, 0, 1), extent


def plot_heatmaps(args, pos, noop_v, frontier_v, noop_share, frontier_share):
    colors = TREND_COLORS[: args.num_trends]
    snapshots = [step for step in args.snapshots if 0 <= step < len(noop_v) and 0 <= step < len(frontier_v)]
    if not snapshots:
        raise ValueError("No valid snapshots selected.")

    fig, axes = plt.subplots(
        2,
        len(snapshots),
        figsize=(3.2 * len(snapshots), 6.2),
        squeeze=False,
        constrained_layout=True,
    )
    rows = [
        ("No intervention", noop_v, noop_share),
        ("Raw frontier", frontier_v, frontier_share),
    ]

    for row_idx, (label, history, shares) in enumerate(rows):
        for col_idx, step in enumerate(snapshots):
            ax = axes[row_idx, col_idx]
            image, extent = trend_mixture_image(
                pos,
                history[step],
                colors,
                args.bins,
                args.min_bin_count,
                args.mixture_mode,
            )
            ax.imshow(image, origin="lower", extent=extent, interpolation="bilinear")
            if args.show_nodes:
                target = np.clip(history[step][:, args.target_trend], 0.0, 1.0)
                ax.scatter(pos[:, 0], pos[:, 1], c=target, cmap="Greens", s=1, alpha=0.25, linewidths=0)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=12)
            ax.set_title(f"t={step}\ntrend {args.target_trend}={shares[step]:.3f}", fontsize=10)
            ax.set_aspect("equal")
            ax.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color, markersize=8, label=f"Trend {i}")
        for i, color in enumerate(colors)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=args.num_trends,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    path = args.output_dir / f"{args.output_name}.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target_trend_heatmaps(args, pos, noop_v, frontier_v, noop_share, frontier_share):
    snapshots = [step for step in args.snapshots if 0 <= step < len(noop_v) and 0 <= step < len(frontier_v)]
    fig, axes = plt.subplots(
        2,
        len(snapshots),
        figsize=(3.2 * len(snapshots), 6.2),
        squeeze=False,
        constrained_layout=True,
    )
    rows = [
        ("No intervention", noop_v, noop_share),
        ("Raw frontier", frontier_v, frontier_share),
    ]
    x = pos[:, 0]
    y = pos[:, 1]
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

    last_image = None
    for row_idx, (label, history, shares) in enumerate(rows):
        for col_idx, step in enumerate(snapshots):
            counts, x_edges, y_edges = np.histogram2d(
                x,
                y,
                bins=args.bins,
                range=[[extent[0], extent[1]], [extent[2], extent[3]]],
            )
            weighted, _, _ = np.histogram2d(
                x,
                y,
                bins=[x_edges, y_edges],
                weights=history[step][:, args.target_trend],
            )
            target = weighted / np.maximum(counts, 1.0)
            target[counts < args.min_bin_count] = np.nan
            ax = axes[row_idx, col_idx]
            last_image = ax.imshow(
                target.T,
                origin="lower",
                extent=extent,
                cmap="Greens",
                vmin=0.0,
                vmax=0.6,
                interpolation="bilinear",
            )
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=12)
            ax.set_title(f"t={step}\ntrend {args.target_trend}={shares[step]:.3f}", fontsize=10)
            ax.set_aspect("equal")
            ax.axis("off")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes, location="right", shrink=0.8, label=f"Trend {args.target_trend} share")
    path = args.output_dir / f"{args.output_name}_target_only.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target_difference_heatmaps(args, pos, noop_v, frontier_v):
    snapshots = [step for step in args.snapshots if 0 <= step < len(noop_v) and 0 <= step < len(frontier_v)]
    fig, axes = plt.subplots(
        1,
        len(snapshots),
        figsize=(3.2 * len(snapshots), 3.4),
        squeeze=False,
        constrained_layout=True,
    )
    x = pos[:, 0]
    y = pos[:, 1]
    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

    last_image = None
    for col_idx, step in enumerate(snapshots):
        counts, x_edges, y_edges = np.histogram2d(
            x,
            y,
            bins=args.bins,
            range=[[extent[0], extent[1]], [extent[2], extent[3]]],
        )
        delta = frontier_v[step][:, args.target_trend] - noop_v[step][:, args.target_trend]
        weighted, _, _ = np.histogram2d(
            x,
            y,
            bins=[x_edges, y_edges],
            weights=delta,
        )
        mean_delta = weighted / np.maximum(counts, 1.0)
        mean_delta[counts < args.min_bin_count] = np.nan
        ax = axes[0, col_idx]
        last_image = ax.imshow(
            mean_delta.T,
            origin="lower",
            extent=extent,
            cmap="PiYG",
            vmin=-0.25,
            vmax=0.25,
            interpolation="bilinear",
        )
        ax.set_title(f"t={step}", fontsize=10)
        ax.set_aspect("equal")
        ax.axis("off")

    if last_image is not None:
        fig.colorbar(
            last_image,
            ax=axes,
            location="right",
            shrink=0.8,
            label=f"Frontier - no-op trend {args.target_trend} share",
        )
    path = args.output_dir / f"{args.output_name}_target_delta.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    static_inputs = build_inputs(args)
    graph, _, _, _, _, node_features = static_inputs
    pos = get_positions(args, graph, node_features)

    noop_env = make_env(args, static_inputs)
    frontier_env = make_env(args, static_inputs)
    noop_v, noop_share = run_rollout(noop_env, action_none, args.rollout_seed, args.steps)
    frontier_v, frontier_share = run_rollout(
        frontier_env,
        action_raw_frontier,
        args.rollout_seed,
        args.steps,
    )

    mixture_path = plot_heatmaps(args, pos, noop_v, frontier_v, noop_share, frontier_share)
    target_path = plot_target_trend_heatmaps(args, pos, noop_v, frontier_v, noop_share, frontier_share)
    delta_path = plot_target_difference_heatmaps(args, pos, noop_v, frontier_v)
    print("Heatmaps complete")
    print(f"  mixture:     {mixture_path}")
    print(f"  target-only: {target_path}")
    print(f"  delta:       {delta_path}")
    print(f"  no-op final target share:    {noop_share[-1]:.4f}")
    print(f"  frontier final target share: {frontier_share[-1]:.4f}")


if __name__ == "__main__":
    main()
