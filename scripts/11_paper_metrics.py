from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
from scipy import stats

from src.env import MultiTrendContagionEnv, RewardWeights
from src.graph import (
    assign_archetypes,
    compute_affinity_matrix,
    generate_trend_embeddings,
    load_facebook_graph,
    precompute_node_features,
    sample_node_interests,
)


POLICIES = ("none", "random", "greedy_degree", "raw_frontier")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute paper-ready summary metrics.")
    parser.add_argument("--graph", default=ROOT / "data/facebook_combined.txt", type=Path)
    parser.add_argument("--output-dir", default=ROOT / "results/paper_metrics", type=Path)
    parser.add_argument("--target-trend", default=3, type=int)
    parser.add_argument("--target-share", default=0.3, type=float)
    parser.add_argument("--strong-target-share", default=0.5, type=float)
    parser.add_argument("--trend-strengths", nargs="*", default=[1.0, 1.25, 1.0, 0.75], type=float)
    parser.add_argument("--num-trends", default=4, type=int)
    parser.add_argument("--embedding-dim", default=10, type=int)
    parser.add_argument("--episodes", default=10, type=int)
    parser.add_argument("--max-steps", default=100, type=int)
    parser.add_argument("--spectral-modes", default=5, type=int)
    parser.add_argument("--boost-budget", default=400, type=int)
    parser.add_argument("--suppress-budget", default=400, type=int)
    parser.add_argument("--strong-boost-budget", default=1200, type=int)
    parser.add_argument("--strong-suppress-budget", default=0, type=int)
    parser.add_argument("--action-scale", default=0.5, type=float)
    parser.add_argument("--strong-action-scale", default=2.0, type=float)
    parser.add_argument("--energy-weight", default=0.02, type=float)
    parser.add_argument("--fatigue-weight", default=0.5, type=float)
    parser.add_argument("--fatigue-inequality-weight", default=0.1, type=float)
    parser.add_argument("--num-seeds", default=20, type=int)
    parser.add_argument("--seed-strength", default=0.5, type=float)
    parser.add_argument("--beta", default=0.3, type=float)
    parser.add_argument("--eta", default=0.1, type=float)
    parser.add_argument("--dt", default=0.5, type=float)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--threshold", default=0.2, type=float)
    parser.add_argument(
        "--target-sweep",
        nargs="*",
        default=[0.2, 0.3, 0.4, 0.5, 0.6],
        type=float,
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def network_stats(graph):
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "avg_degree": 2 * graph.number_of_edges() / graph.number_of_nodes(),
        "connected_components": nx.number_connected_components(graph),
        "avg_clustering": nx.average_clustering(graph),
        "diameter": nx.diameter(graph) if nx.is_connected(graph) else "",
    }


def build_inputs(args):
    graph, adjacency, nodes = load_facebook_graph(args.graph)
    n = len(nodes)
    sigma, rho, _ = assign_archetypes(n, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    trend_embeddings = generate_trend_embeddings(args.num_trends, args.embedding_dim, rng)
    node_interests = sample_node_interests(n, args.embedding_dim, rng)
    affinity = compute_affinity_matrix(node_interests, trend_embeddings)
    strengths = np.asarray(args.trend_strengths, dtype=np.float32)
    affinity = np.clip(affinity * strengths[None, :], 0.0, 1.0)
    affinity[:, 0] = 0.5
    node_features = precompute_node_features(graph, adjacency, nodes, k=args.spectral_modes)
    return graph, adjacency, sigma, rho, affinity, node_features


def make_env(args, static_inputs, target_share=None, boost_budget=None, suppress_budget=None, action_scale=None):
    _, adjacency, sigma, rho, affinity, node_features = static_inputs
    return MultiTrendContagionEnv(
        adjacency,
        sigma,
        rho,
        affinity,
        target_trend=args.target_trend,
        target_share=args.target_share if target_share is None else target_share,
        boost_budget=args.boost_budget if boost_budget is None else boost_budget,
        suppress_budget=args.suppress_budget if suppress_budget is None else suppress_budget,
        max_steps=args.max_steps,
        num_seeds=args.num_seeds,
        seed_strength=args.seed_strength,
        action_scale=args.action_scale if action_scale is None else action_scale,
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


def action_for_policy(env, policy: str):
    if policy == "none":
        return np.zeros(env.action_space.shape, dtype=np.float32)
    if policy == "random":
        return env.rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
    if policy == "greedy_degree":
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        action[0] = 1.0
        action[-1] = 1.0
        return action
    if policy == "raw_frontier":
        return np.ones(env.action_space.shape, dtype=np.float32)
    raise ValueError(policy)


def first_time_at_or_above(values: np.ndarray, threshold: float):
    hits = np.flatnonzero(values >= threshold)
    return int(hits[0]) if hits.size else ""


def evaluate_policy(env, policy: str, episodes: int, threshold: float):
    rewards = []
    deviations = []
    fatigue = []
    energy = []
    action_magnitude = []
    intervention_rate = []
    final_share = []
    time_to_threshold = []
    final_all_shares = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        done = False
        total_reward = 0.0
        ep_dev = []
        ep_fatigue = []
        ep_energy = []
        ep_mag = []
        shares = [float(env.v[:, env.target_trend].mean())]
        interventions = 0
        steps = 0
        info = {}
        while not done:
            action = action_for_policy(env, policy)
            mag = float(np.mean(np.abs(action)))
            if mag > 0.1:
                interventions += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            ep_dev.append(float(info["target_deviation"]))
            ep_fatigue.append(float(info["target_fatigue"]))
            ep_energy.append(float(info["action_energy"]))
            ep_mag.append(mag)
            shares.append(float(info["target_share"]))
            steps += 1
        rewards.append(total_reward)
        deviations.append(float(np.mean(ep_dev)))
        fatigue.append(float(np.mean(ep_fatigue)))
        energy.append(float(np.mean(ep_energy)))
        action_magnitude.append(float(np.mean(ep_mag)))
        intervention_rate.append(interventions / max(steps, 1))
        final_share.append(float(info["target_share"]))
        time_to_threshold.append(first_time_at_or_above(np.asarray(shares), threshold))
        final_all_shares.append(env.v.mean(axis=0).copy())
    return {
        "policy": policy,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards, ddof=1)),
        "deviation_mean": float(np.mean(deviations)),
        "deviation_std": float(np.std(deviations, ddof=1)),
        "fatigue_mean": float(np.mean(fatigue)),
        "fatigue_std": float(np.std(fatigue, ddof=1)),
        "energy_mean": float(np.mean(energy)),
        "energy_std": float(np.std(energy, ddof=1)),
        "action_magnitude_mean": float(np.mean(action_magnitude)),
        "intervention_rate_mean": float(np.mean(intervention_rate)),
        "final_share_mean": float(np.mean(final_share)),
        "final_share_std": float(np.std(final_share, ddof=1)),
        "time_to_threshold_mean": float(
            np.mean([value for value in time_to_threshold if value != ""])
        )
        if any(value != "" for value in time_to_threshold)
        else "",
        "time_to_threshold_values": " ".join(str(value) for value in time_to_threshold),
        "_rewards": rewards,
        "_final_all_shares": final_all_shares,
    }


def clean_metric_row(row: dict):
    return {key: value for key, value in row.items() if not key.startswith("_")}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    static_inputs = build_inputs(args)
    graph = static_inputs[0]

    write_csv(args.output_dir / "network_stats.csv", [network_stats(graph)])

    base_env = make_env(args, static_inputs)
    policy_rows = []
    detailed = {}
    for policy in POLICIES:
        row = evaluate_policy(base_env, policy, args.episodes, args.threshold)
        detailed[policy] = row
        policy_rows.append(clean_metric_row(row))
    write_csv(args.output_dir / "multitrend_policy_stats_target03.csv", policy_rows)

    equilibrium = np.mean(detailed["none"]["_final_all_shares"], axis=0)
    write_csv(
        args.output_dir / "natural_equilibrium_target03.csv",
        [{"trend": idx, "final_share": float(value)} for idx, value in enumerate(equilibrium)],
    )

    strong_env = make_env(
        args,
        static_inputs,
        target_share=args.strong_target_share,
        boost_budget=args.strong_boost_budget,
        suppress_budget=args.strong_suppress_budget,
        action_scale=args.strong_action_scale,
    )
    strong_rows = []
    strong_detailed = {}
    for policy in ("none", "raw_frontier"):
        row = evaluate_policy(strong_env, policy, args.episodes, args.threshold)
        strong_detailed[policy] = row
        strong_rows.append(clean_metric_row(row))
    write_csv(args.output_dir / "presentation_policy_stats_target05.csv", strong_rows)

    t_stat, p_value = stats.ttest_rel(
        strong_detailed["raw_frontier"]["_rewards"],
        strong_detailed["none"]["_rewards"],
    )
    t03_stat, t03_p_value = stats.ttest_rel(
        detailed["raw_frontier"]["_rewards"],
        detailed["none"]["_rewards"],
    )
    write_csv(
        args.output_dir / "paired_tests.csv",
        [
            {
                "comparison": "raw_frontier_vs_none_target03",
                "t_stat": float(t03_stat),
                "p_value": float(t03_p_value),
            },
            {
                "comparison": "raw_frontier_vs_none_target05",
                "t_stat": float(t_stat),
                "p_value": float(p_value),
            }
        ],
    )

    sweep_rows = []
    natural_share = float(equilibrium[args.target_trend])
    for target_share in args.target_sweep:
        sweep_env = make_env(
            args,
            static_inputs,
            target_share=target_share,
            boost_budget=args.strong_boost_budget,
            suppress_budget=args.strong_suppress_budget,
            action_scale=args.strong_action_scale,
        )
        none_row = evaluate_policy(sweep_env, "none", args.episodes, args.threshold)
        frontier_row = evaluate_policy(sweep_env, "raw_frontier", args.episodes, args.threshold)
        t_sweep, p_sweep = stats.ttest_rel(frontier_row["_rewards"], none_row["_rewards"])
        sweep_rows.append(
            {
                "target_share": float(target_share),
                "natural_target_share": natural_share,
                "target_gap": float(abs(target_share - natural_share)),
                "none_reward_mean": none_row["reward_mean"],
                "frontier_reward_mean": frontier_row["reward_mean"],
                "reward_advantage": frontier_row["reward_mean"] - none_row["reward_mean"],
                "none_deviation_mean": none_row["deviation_mean"],
                "frontier_deviation_mean": frontier_row["deviation_mean"],
                "deviation_reduction": none_row["deviation_mean"] - frontier_row["deviation_mean"],
                "deviation_reduction_pct": (
                    (none_row["deviation_mean"] - frontier_row["deviation_mean"])
                    / max(none_row["deviation_mean"], 1e-8)
                ),
                "none_final_share_mean": none_row["final_share_mean"],
                "frontier_final_share_mean": frontier_row["final_share_mean"],
                "final_share_lift": frontier_row["final_share_mean"] - none_row["final_share_mean"],
                "none_time_to_threshold": none_row["time_to_threshold_mean"],
                "frontier_time_to_threshold": frontier_row["time_to_threshold_mean"],
                "time_speedup": (
                    none_row["time_to_threshold_mean"] / frontier_row["time_to_threshold_mean"]
                    if frontier_row["time_to_threshold_mean"] not in ("", 0)
                    and none_row["time_to_threshold_mean"] != ""
                    else ""
                ),
                "paired_t_stat": float(t_sweep),
                "paired_p_value": float(p_sweep),
            }
        )
    write_csv(args.output_dir / "target_regime_sweep.csv", sweep_rows)

    print(f"Wrote paper metrics to {args.output_dir}")
    print(f"Network: {network_stats(graph)}")
    print(f"Target 0.3 natural equilibrium: {equilibrium}")
    print("Target 0.3 policy stats:")
    for row in policy_rows:
        print(row)
    print("Target 0.5 presentation stats:")
    for row in strong_rows:
        print(row)
    print(f"Paired test raw_frontier vs none target 0.5: t={t_stat:.3f}, p={p_value:.6f}")
    print(f"Paired test raw_frontier vs none target 0.3: t={t03_stat:.3f}, p={t03_p_value:.6f}")
    print("Target-regime sweep:")
    for row in sweep_rows:
        print(row)


if __name__ == "__main__":
    main()
