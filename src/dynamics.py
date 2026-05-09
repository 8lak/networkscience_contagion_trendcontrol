"""Contagion and fatigue dynamics."""

from __future__ import annotations

import numpy as np


def compute_effector_quality(
    sigma: np.ndarray,
    affinity: np.ndarray,
    phi: np.ndarray,
    phi_global: float = 0.0,
) -> np.ndarray:
    """Return fatigue-adjusted spreading quality for each node."""
    sigma = np.asarray(sigma, dtype=np.float32)
    affinity = np.asarray(affinity, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    if affinity.ndim == 2:
        return sigma[:, None] * affinity / (1.0 + phi + phi_global)
    return sigma * affinity / (1.0 + phi + phi_global)


def step(
    v: np.ndarray,
    phi: np.ndarray,
    adjacency,
    sigma: np.ndarray,
    affinity: np.ndarray,
    rho: np.ndarray,
    eta: float = 0.08,
    phi_global: float = 0.0,
    dt: float = 0.1,
):
    """Advance SIS-style adoption ``v`` and fatigue ``phi`` by one Euler step."""
    v = np.asarray(v, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.float32)
    affinity = np.asarray(affinity, dtype=np.float32)
    rho = np.asarray(rho, dtype=np.float32)

    effector = compute_effector_quality(sigma, affinity, phi, phi_global)
    pressure = adjacency @ (effector * v)

    susceptibility = 1.0 - v
    dv = affinity * sigma * susceptibility * pressure - phi * v
    dphi = -rho * phi + eta * v

    v_new = np.clip(v + dt * dv, 0.0, 1.0).astype(np.float32)
    phi_new = np.clip(phi + dt * dphi, 0.0, 1.0).astype(np.float32)
    return v_new, phi_new


def seed_trend(
    n: int,
    num_seeds: int = 10,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Create an initial adoption vector with randomly selected seed nodes."""
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    seed_count = min(max(int(num_seeds), 0), n)
    v = np.zeros(n, dtype=np.float32)
    if seed_count:
        seed_idx = generator.choice(n, size=seed_count, replace=False)
        v[seed_idx] = 1.0
    return v


def step_multi_trend(
    v: np.ndarray,
    phi: np.ndarray,
    adjacency,
    sigma: np.ndarray,
    affinity: np.ndarray,
    rho: np.ndarray,
    beta: float = 0.3,
    eta: float = 0.1,
    dt: float = 0.5,
    eps: float = 1e-8,
):
    """Advance competing attention shares and per-trend fatigue one step."""
    v = np.asarray(v, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.float32)
    affinity = np.asarray(affinity, dtype=np.float32)
    rho = np.asarray(rho, dtype=np.float32)

    effector = compute_effector_quality(sigma, affinity, phi)
    pressure = adjacency @ (effector * v)
    susceptibility = 1.0 - v

    dv = beta * affinity * sigma[:, None] * susceptibility * pressure - phi * v
    dphi = -rho[:, None] * phi + eta * v

    v_new = np.clip(v + dt * dv, 0.0, 1.0).astype(np.float32)
    phi_new = np.clip(phi + dt * dphi, 0.0, 1.0).astype(np.float32)

    row_sums = v_new.sum(axis=1, keepdims=True)
    v_new = v_new / np.maximum(row_sums, eps)
    return v_new.astype(np.float32), phi_new


def seed_trends(
    n: int,
    num_trends: int,
    num_seeds_per_trend: int = 20,
    rng: np.random.Generator | int | None = None,
    seed_strength: float = 0.5,
) -> np.ndarray:
    """Initialize background attention and seed competing trends."""
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    trend_count = max(int(num_trends), 1)
    seed_count = min(max(int(num_seeds_per_trend), 0), n)
    strength = float(np.clip(seed_strength, 0.0, 1.0))

    v = np.zeros((n, trend_count), dtype=np.float32)
    v[:, 0] = 1.0
    for trend in range(1, trend_count):
        if seed_count == 0:
            continue
        seed_idx = generator.choice(n, size=seed_count, replace=False)
        v[seed_idx, 0] = np.maximum(v[seed_idx, 0] - strength, 0.0)
        v[seed_idx, trend] = np.minimum(v[seed_idx, trend] + strength, 1.0)

    row_sums = v.sum(axis=1, keepdims=True)
    return (v / np.maximum(row_sums, 1e-8)).astype(np.float32)
