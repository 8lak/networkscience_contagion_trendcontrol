from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.dynamics import seed_trend, seed_trends, step, step_multi_trend
from src.graph import (
    assign_archetypes,
    compute_affinity_matrix,
    generate_trend_embeddings,
    load_facebook_graph,
    sample_affinity,
    sample_node_interests,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run and plot contagion dynamics.")
    parser.add_argument("--graph", default=ROOT / "data/facebook_combined.txt", type=Path)
    parser.add_argument("--steps", default=150, type=int)
    parser.add_argument("--num-seeds", default=20, type=int)
    parser.add_argument("--multi-trend", action="store_true")
    parser.add_argument("--num-trends", default=4, type=int)
    parser.add_argument("--embedding-dim", default=10, type=int)
    parser.add_argument("--beta", default=0.3, type=float)
    parser.add_argument("--eta", default=0.1, type=float)
    parser.add_argument("--dt", default=0.5, type=float)
    parser.add_argument("--seed-strength", default=0.5, type=float)
    parser.add_argument(
        "--trend-strengths",
        nargs="*",
        type=float,
        default=None,
        help="Optional per-trend affinity multipliers, e.g. 1.0 1.2 1.0 0.8.",
    )
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--output", default=ROOT / "results/dynamics_test.png", type=Path)
    return parser.parse_args()


def run_single_trend(args, graph, adjacency, n, sigma, rho):
    affinity = sample_affinity(
        n,
        {"distribution": "uniform", "low": 0.25, "high": 1.0},
        args.seed,
    )
    rng = np.random.default_rng(args.seed)
    v = seed_trend(n, args.num_seeds, rng)
    phi = np.zeros(n, dtype=np.float32)

    v_history = []
    phi_history = []
    for _ in range(args.steps):
        v, phi = step(v, phi, adjacency, sigma, affinity, rho)
        v_history.append(v.copy())
        phi_history.append(phi.copy())

    v_history = np.asarray(v_history)
    phi_history = np.asarray(phi_history)
    adopted = np.mean(v_history > 0.5, axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(v_history.mean(axis=1))
    axes[0, 0].set_title("Mean adoption")
    axes[0, 1].plot(phi_history.mean(axis=1))
    axes[0, 1].set_title("Mean fatigue")
    axes[1, 0].plot(adopted)
    axes[1, 0].set_title("Fraction adopted (V > 0.5)")
    axes[1, 1].hist(v_history[-1], bins=30)
    axes[1, 1].set_title("Final adoption distribution")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)

    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Initial mean(V): {v_history[0].mean():.4f}")
    print(f"Final mean(V): {v_history[-1].mean():.4f}")
    print(f"Final mean(Phi): {phi_history[-1].mean():.4f}")
    print(f"Saved plot to {args.output}")


def run_multi_trend(args, graph, adjacency, n, sigma, rho):
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

    v = seed_trends(
        n,
        args.num_trends,
        num_seeds_per_trend=args.num_seeds,
        rng=rng,
        seed_strength=args.seed_strength,
    )
    phi = np.zeros((n, args.num_trends), dtype=np.float32)

    v_history = []
    phi_history = []
    simplex_error = []
    for _ in range(args.steps):
        v, phi = step_multi_trend(
            v,
            phi,
            adjacency,
            sigma,
            affinity,
            rho,
            beta=args.beta,
            eta=args.eta,
            dt=args.dt,
        )
        v_history.append(v.copy())
        phi_history.append(phi.copy())
        simplex_error.append(float(np.max(np.abs(v.sum(axis=1) - 1.0))))

    v_history = np.asarray(v_history)
    phi_history = np.asarray(phi_history)
    trend_labels = [f"trend {idx}" for idx in range(args.num_trends)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for trend in range(args.num_trends):
        axes[0, 0].plot(v_history[:, :, trend].mean(axis=1), label=trend_labels[trend])
    axes[0, 0].set_title("Mean attention share by trend")
    axes[0, 0].set_ylabel("attention share")
    axes[0, 0].legend()

    for trend in range(args.num_trends):
        axes[0, 1].plot(phi_history[:, :, trend].mean(axis=1), label=trend_labels[trend])
    axes[0, 1].set_title("Mean fatigue by trend")
    axes[0, 1].legend()

    for trend in range(args.num_trends):
        axes[1, 0].hist(v_history[-1, :, trend], bins=30, alpha=0.45, label=trend_labels[trend])
    axes[1, 0].set_title("Final attention share distribution")
    axes[1, 0].legend()

    initial_dominant = np.argmax(v_history[0], axis=1)
    final_dominant = np.argmax(v_history[-1], axis=1)
    x = np.arange(args.num_trends)
    width = 0.4
    initial_counts = np.bincount(initial_dominant, minlength=args.num_trends)
    final_counts = np.bincount(final_dominant, minlength=args.num_trends)
    axes[1, 1].bar(x - width / 2, initial_counts, width=width, label="initial")
    axes[1, 1].bar(x + width / 2, final_counts, width=width, label="final")
    axes[1, 1].set_title("Dominant trend counts")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(trend_labels, rotation=30)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(args.output, dpi=160)

    mean_simplex = v_history.sum(axis=2).mean(axis=1)
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Initial mean shares: {v_history[0].mean(axis=0)}")
    print(f"Final mean shares: {v_history[-1].mean(axis=0)}")
    print(f"Final mean fatigue: {phi_history[-1].mean(axis=0)}")
    print(f"Mean simplex range: {mean_simplex.min():.6f} to {mean_simplex.max():.6f}")
    print(f"Max node simplex error: {max(simplex_error):.8f}")
    print(f"Saved plot to {args.output}")


def main():
    args = parse_args()
    graph, adjacency, nodes = load_facebook_graph(args.graph)
    n = len(nodes)

    sigma, rho, _ = assign_archetypes(n, seed=args.seed)
    if args.multi_trend:
        run_multi_trend(args, graph, adjacency, n, sigma, rho)
    else:
        run_single_trend(args, graph, adjacency, n, sigma, rho)


if __name__ == "__main__":
    main()
