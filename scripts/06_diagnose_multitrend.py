from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

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
    parser = argparse.ArgumentParser(description="Diagnose multi-trend env dynamics and action leverage.")
    parser.add_argument("--graph", default=ROOT / "data/facebook_combined.txt", type=Path)
    parser.add_argument("--num-trends", default=4, type=int)
    parser.add_argument("--embedding-dim", default=10, type=int)
    parser.add_argument("--target-trend", default=3, type=int)
    parser.add_argument("--target-share", default=0.3, type=float)
    parser.add_argument("--trend-strengths", nargs="*", default=[1.0, 1.25, 1.0, 0.75], type=float)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--budget", default=10, type=int)
    parser.add_argument("--boost-budget", default=None, type=int)
    parser.add_argument("--suppress-budget", default=None, type=int)
    parser.add_argument("--spectral-modes", default=5, type=int)
    parser.add_argument("--num-seeds", default=20, type=int)
    parser.add_argument("--seed-strength", default=0.5, type=float)
    parser.add_argument("--action-scale", default=0.1, type=float)
    parser.add_argument(
        "--decoder-mode",
        default="current",
        choices=["current", "additive", "spectral_frontier", "frontier", "raw_frontier"],
    )
    parser.add_argument("--beta", default=0.3, type=float)
    parser.add_argument("--eta", default=0.1, type=float)
    parser.add_argument("--dt", default=0.5, type=float)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--probe-action", choices=["zero", "degree", "random", "frontier"], default="degree")
    return parser.parse_args()


def build_env(args):
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
        max_steps=args.steps,
        num_seeds=args.num_seeds,
        seed_strength=args.seed_strength,
        action_scale=args.action_scale,
        decoder_mode=args.decoder_mode,
        beta=args.beta,
        eta=args.eta,
        dt=args.dt,
        node_features=node_features,
        reward_weights=RewardWeights(fatigue=0.5, fatigue_inequality=0.1, energy=0.02),
    )


def make_probe_action(env, kind: str):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    if kind == "degree":
        action[0] = 1.0
        action[-1] = 1.0
    elif kind == "random":
        action = env.rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
    return action


def direct_frontier_scores(env):
    target_v = env.v[:, env.target_trend]
    pressure = np.asarray(env.adjacency @ target_v).reshape(-1)
    frontier = (1.0 - target_v) * pressure
    if float(np.max(frontier)) <= 1e-8:
        return np.zeros(env.n, dtype=np.float32), np.zeros(env.n, dtype=np.float32)
    boost_scores = frontier / np.max(frontier)
    return boost_scores.astype(np.float32), np.zeros(env.n, dtype=np.float32)


def expected_action_delta(env, boosted, boost_strengths, suppressed, suppress_strengths):
    expected = 0.0
    for node, strength in zip(boosted, boost_strengths):
        target_mass = float(env.v[node, env.target_trend])
        other_mass = max(1.0 - target_mass, 0.0)
        expected += min(env.action_scale * float(strength), other_mass)
    for node, strength in zip(suppressed, suppress_strengths):
        target_mass = float(env.v[node, env.target_trend])
        expected -= min(env.action_scale * float(strength), target_mass)
    return expected / env.n


def run_noop(env, steps: int, seed: int):
    obs, info = env.reset(seed=seed)
    trajectory = [info["target_share"]]
    rewards = []
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        rewards.append(reward)
        trajectory.append(info["target_share"])
        if terminated or truncated:
            break
    trajectory = np.asarray(trajectory, dtype=np.float32)
    print("NO-OP")
    print(f"  initial target share: {trajectory[0]:.6f}")
    print(f"  final target share:   {trajectory[-1]:.6f}")
    print(f"  mean target share:    {trajectory.mean():.6f}")
    print(f"  mean deviation:       {np.mean(np.abs(trajectory - env.target_share)):.6f}")
    print(f"  total reward:         {np.sum(rewards):.6f}")
    print(f"  final all shares:     {env.v.mean(axis=0)}")
    print(f"  simplex error:        {np.max(np.abs(env.v.sum(axis=1) - 1.0)):.8f}")
    sample_idx = np.linspace(0, len(trajectory) - 1, num=min(10, len(trajectory)), dtype=int)
    print("  sampled trajectory:")
    for idx in sample_idx:
        print(f"    step {idx:03d}: {trajectory[idx]:.6f}")


def run_action_probe(env, args):
    obs, _ = env.reset(seed=args.seed)
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(max(args.steps // 2, 1)):
        obs, _, terminated, truncated, _ = env.step(zero)
        if terminated or truncated:
            break

    before = float(env.v[:, env.target_trend].mean())
    action = make_probe_action(env, args.probe_action)
    if args.probe_action == "frontier":
        boost_scores, suppress_scores = direct_frontier_scores(env)
    else:
        boost_scores, suppress_scores = env._decode_action(action)

    boost_candidates = np.flatnonzero(boost_scores > 1e-8)
    suppress_candidates = np.flatnonzero(suppress_scores < -1e-8)
    would_boost = boost_candidates[np.argsort(-boost_scores[boost_candidates])[: env.boost_budget]]
    would_suppress = suppress_candidates[np.argsort(suppress_scores[suppress_candidates])[: env.suppress_budget]]
    would_boost_strengths = boost_scores[would_boost].astype(np.float32)
    would_suppress_strengths = np.abs(suppress_scores[would_suppress]).astype(np.float32)
    expected_delta = expected_action_delta(
        env,
        would_boost,
        would_boost_strengths,
        would_suppress,
        would_suppress_strengths,
    )

    score_before = float(env.v[:, env.target_trend].mean())
    env._apply_action_scores(boost_scores, suppress_scores)
    after_action = float(env.v[:, env.target_trend].mean())
    selected_count = len(env.last_selected)
    boosted_count = len(env.last_boosted)
    suppressed_count = len(env.last_suppressed)
    action_energy = env.last_action_energy
    obs, reward, terminated, truncated, info = env.step(zero)
    after_dynamics = float(env.v[:, env.target_trend].mean())

    print("ACTION PROBE")
    print(f"  probe action:         {args.probe_action}")
    print(f"  action mean abs:      {np.mean(np.abs(action)):.6f}")
    print(f"  decoded boost max:    {float(np.max(boost_scores)):.6f}")
    print(f"  decoded suppress min: {float(np.min(suppress_scores)):.6f}")
    if len(would_boost_strengths):
        print(
            "  boost strength q:     "
            f"{np.quantile(would_boost_strengths, [0, .25, .5, .75, 1])}"
        )
    if len(would_suppress_strengths):
        print(
            "  suppress strength q:  "
            f"{np.quantile(would_suppress_strengths, [0, .25, .5, .75, 1])}"
        )
    print(f"  share before:         {before:.6f}")
    print(f"  score before:         {score_before:.6f}")
    print(f"  after action only:    {after_action:.6f}")
    print(f"  action delta:         {after_action - score_before:.6f}")
    print(f"  expected delta:       {expected_delta:.6f}")
    print(f"  measurement error:    {(after_action - score_before) - expected_delta:.8f}")
    print(f"  after dynamics:       {after_dynamics:.6f}")
    print(f"  dynamics delta:       {after_dynamics - after_action:.6f}")
    print(f"  selected/boost/supp:  {selected_count}/{boosted_count}/{suppressed_count}")
    print(f"  action energy:        {action_energy:.6f}")


def main():
    args = parse_args()
    print(f"target_trend={args.target_trend} target_share={args.target_share}")
    print(f"trend_strengths={args.trend_strengths}")
    print(f"action_scale={args.action_scale}")
    print(f"decoder_mode={args.decoder_mode}")
    env = build_env(args)
    print(f"obs_shape={env.observation_space.shape} action_shape={env.action_space.shape}")
    run_noop(env, args.steps, args.seed)
    run_action_probe(env, args)


if __name__ == "__main__":
    main()
