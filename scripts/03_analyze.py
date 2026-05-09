from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from stable_baselines3 import PPO

from src.env import ContagionEnv, RewardWeights
from src.generators.random_surfer import multi_edge_random_surfer
from src.graph import assign_archetypes, load_facebook_graph, precompute_node_features, sample_affinity


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze trained contagion policies.")
    parser.add_argument("--graph", default=ROOT / "data/facebook_combined.txt", type=Path)
    parser.add_argument("--model-path", default=ROOT / "models/ppo_contagion", type=Path)
    parser.add_argument("--episodes", default=3, type=int)
    parser.add_argument("--max-steps", default=100, type=int)
    parser.add_argument("--budget", default=10, type=int)
    parser.add_argument("--boost-budget", default=None, type=int)
    parser.add_argument("--suppress-budget", default=None, type=int)
    parser.add_argument("--target-adoption", default=0.3, type=float)
    parser.add_argument("--fatigue-weight", default=0.5, type=float)
    parser.add_argument("--fatigue-inequality-weight", default=0.1, type=float)
    parser.add_argument("--energy-weight", default=0.1, type=float)
    parser.add_argument("--spectral-modes", default=20, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--output-dir", default=ROOT / "results", type=Path)
    return parser.parse_args()


def build_env(graph, adjacency, nodes, args, fatigue=True, heterogeneous=True):
    n = len(nodes)
    sigma, rho, _ = assign_archetypes(n, seed=args.seed)
    if not heterogeneous:
        sigma = np.full(n, 0.3, dtype=np.float32)
    if not fatigue:
        rho = np.ones(n, dtype=np.float32)
    affinity = sample_affinity(n, seed=args.seed)
    node_features = precompute_node_features(graph, adjacency, nodes, k=args.spectral_modes)
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
        eta=0.08 if fatigue else 0.0,
        node_features=node_features,
        reward_weights=RewardWeights(
            fatigue=args.fatigue_weight,
            fatigue_inequality=args.fatigue_inequality_weight,
            energy=args.energy_weight,
        ),
    )


def effective_laplacian(adjacency, sigma, affinity, phi):
    effector = sigma * affinity / (1.0 + phi + float(np.mean(phi)))
    weights = adjacency.multiply(effector)
    symmetric_weights = 0.5 * (weights + weights.T)
    return csgraph.laplacian(symmetric_weights, normed=False)


def fiedler_value(laplacian):
    if laplacian.shape[0] < 3:
        return 0.0
    vals = eigsh(laplacian, k=2, which="SM", return_eigenvectors=False)
    return float(np.sort(vals)[1])


def collect_rollout(env, model, seed):
    obs, _ = env.reset(seed=seed)
    done = False
    states, actions, selected_history, v_history, phi_history, fiedlers = [], [], [], [], [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        states.append(obs.copy())
        actions.append(np.asarray(action).copy())
        v_history.append(env.v.copy())
        phi_history.append(env.phi.copy())
        fiedlers.append(fiedler_value(effective_laplacian(env.adjacency, env.sigma, env.affinity, env.phi)))
        obs, _, terminated, truncated, _ = env.step(action)
        selected_history.append(env.last_selected.copy())
        done = terminated or truncated
    return {
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "selected": selected_history,
        "v": np.asarray(v_history),
        "phi": np.asarray(phi_history),
        "fiedler": np.asarray(fiedlers),
    }


def targeted_degree_correlation(env, selections):
    degree = np.asarray(env.adjacency.sum(axis=1)).reshape(-1)
    scores = []
    for selected in selections:
        if len(selected) == 0:
            continue
        scores.append(float(np.mean(degree[selected]) / max(np.mean(degree), 1e-6)))
    return float(np.mean(scores)) if scores else 0.0


def rollout_baseline(env):
    obs, _ = env.reset(seed=0)
    done = False
    rewards = []
    while not done:
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    return float(np.sum(rewards)), info


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    facebook_graph, facebook_adjacency, nodes = load_facebook_graph(args.graph)
    env = build_env(facebook_graph, facebook_adjacency, nodes, args)
    model = PPO.load(args.model_path, env=env, device=args.device)

    rollouts = [collect_rollout(env, model, args.seed + i) for i in range(args.episodes)]
    fiedler_mean = np.mean([r["fiedler"] for r in rollouts], axis=0)
    targeting = np.mean([targeted_degree_correlation(env, r["selected"]) for r in rollouts])

    random_graph = multi_edge_random_surfer(len(nodes), m=3, p=0.15, seed=args.seed)
    random_nodes = list(random_graph.nodes())
    random_adjacency = nx.to_scipy_sparse_array(random_graph, dtype=np.float32, format="csr")
    ablations = {
        "facebook": build_env(facebook_graph, facebook_adjacency, nodes, args),
        "fatigue_off": build_env(facebook_graph, facebook_adjacency, nodes, args, fatigue=False),
        "uniform_sigma": build_env(facebook_graph, facebook_adjacency, nodes, args, heterogeneous=False),
        "random_surfer": build_env(random_graph, random_adjacency, random_nodes, args),
    }
    for name, ablation_env in ablations.items():
        total_reward, info = rollout_baseline(ablation_env)
        print(f"{name}: reward={total_reward:.3f}, info={info}")

    plt.figure(figsize=(7, 4))
    plt.plot(fiedler_mean)
    plt.title("Effective Fiedler value over PPO rollout")
    plt.xlabel("Step")
    plt.ylabel("lambda_2")
    plt.tight_layout()
    plot_path = args.output_dir / "fiedler_over_time.png"
    plt.savefig(plot_path, dpi=160)

    np.savez_compressed(
        args.output_dir / "analysis_rollouts.npz",
        fiedler_mean=fiedler_mean,
        targeting=targeting,
        node_count=len(nodes),
        edge_count=facebook_graph.number_of_edges(),
    )
    print(f"Mean selected-node degree / network mean degree: {targeting:.3f}")
    print(f"Saved analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
