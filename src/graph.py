"""Graph loading and per-node parameter sampling utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh


def normalize01(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale non-constant values to [0, 1]."""
    values = np.asarray(values, dtype=np.float32)
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low < eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def normalize_signed(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale signed values by their maximum absolute magnitude."""
    values = np.asarray(values, dtype=np.float32)
    scale = float(np.max(np.abs(values)))
    if scale < eps:
        return np.zeros_like(values, dtype=np.float32)
    return (values / scale).astype(np.float32)


def normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize each row to unit length."""
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, eps)).astype(np.float32)


def generate_trend_embeddings(
    num_trends: int,
    embedding_dim: int,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Sample unit trend embeddings."""
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    embeddings = generator.standard_normal((num_trends, embedding_dim))
    return normalize_rows(embeddings)


def sample_node_interests(
    n: int,
    embedding_dim: int,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Sample unit node-interest embeddings."""
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    interests = generator.standard_normal((n, embedding_dim))
    return normalize_rows(interests)


def compute_affinity_matrix(
    node_interests: np.ndarray,
    trend_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute per-node per-trend affinities in [0, 1]."""
    raw_affinity = np.asarray(node_interests, dtype=np.float32) @ np.asarray(
        trend_embeddings,
        dtype=np.float32,
    ).T
    return np.clip((raw_affinity + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def load_facebook_graph(path: str | Path):
    """Load an undirected Facebook edge-list graph.

    Returns the NetworkX graph, a SciPy sparse adjacency matrix, and the node
    ordering used by that matrix.
    """
    graph = nx.read_edgelist(path, nodetype=int)
    graph = nx.Graph(graph)
    nodes = list(graph.nodes())
    adjacency = nx.to_scipy_sparse_array(
        graph,
        nodelist=nodes,
        dtype=np.float32,
        format="csr",
    )
    return graph, adjacency, nodes


def _fix_eigenvector_signs(eigenvectors: np.ndarray) -> np.ndarray:
    """Make eigenvector orientation deterministic for repeatable features."""
    eigenvectors = np.asarray(eigenvectors, dtype=np.float32).copy()
    for col in range(eigenvectors.shape[1]):
        vector = eigenvectors[:, col]
        anchor = int(np.argmax(np.abs(vector)))
        if vector[anchor] < 0:
            eigenvectors[:, col] *= -1.0
    return eigenvectors


def precompute_node_features(graph, adjacency, nodes, k: int = 20) -> dict[str, np.ndarray]:
    """Precompute normalized structural features aligned to adjacency rows."""
    n = len(nodes)
    mode_count = min(max(int(k), 1), max(n - 1, 1))

    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1).astype(np.float32)
    degree = normalize01(degree)

    eigenvector_scores = nx.eigenvector_centrality_numpy(graph)
    authority = np.asarray([eigenvector_scores[node] for node in nodes], dtype=np.float32)
    authority = normalize01(authority)

    laplacian = csgraph.laplacian(adjacency, normed=True)
    if n <= 2:
        eigenvalues = np.zeros(mode_count, dtype=np.float32)
        eigenvectors = np.ones((n, mode_count), dtype=np.float32)
    else:
        eigenvalues, eigenvectors = eigsh(laplacian, k=mode_count, which="SM")
        order = np.argsort(eigenvalues)
        eigenvalues = np.asarray(eigenvalues[order], dtype=np.float32)
        eigenvectors = _fix_eigenvector_signs(eigenvectors[:, order])

    return {
        "eigenvalues": normalize01(eigenvalues),
        "eigenvectors": eigenvectors.astype(np.float32),
        "degree": degree,
        "hub": authority.copy(),
        "authority": authority,
    }


def assign_archetypes(
    n: int,
    conductor_ratio: float = 0.1,
    fast_fatigue_ratio: float = 0.25,
    seed: int | None = None,
):
    """Sample node archetypes and map them to spreading/fatigue parameters."""
    rng = np.random.default_rng(seed)
    is_conductor = rng.random(n) < conductor_ratio
    is_fast = rng.random(n) < fast_fatigue_ratio

    sigma = np.where(is_conductor, 1.0, 0.3).astype(np.float32)
    rho = np.where(is_fast, 0.05, 0.15).astype(np.float32)

    labels = np.full(n, "regular", dtype=object)
    labels[is_conductor] = "conductor"
    labels[is_fast] = "fast_fatigue"
    labels[is_conductor & is_fast] = "conductor_fast_fatigue"
    return sigma, rho, labels


def sample_affinity(
    n: int,
    distribution_params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Sample per-node affinity for one trend.

    Supported distributions:
    - {"distribution": "uniform", "low": 0.2, "high": 1.0}
    - {"distribution": "normal", "mean": 0.6, "std": 0.15}
    """
    params = distribution_params or {}
    distribution = params.get("distribution", "uniform")
    rng = np.random.default_rng(seed)

    if distribution == "uniform":
        low = float(params.get("low", 0.2))
        high = float(params.get("high", 1.0))
        values = rng.uniform(low, high, size=n)
    elif distribution == "normal":
        mean = float(params.get("mean", 0.6))
        std = float(params.get("std", 0.15))
        values = rng.normal(mean, std, size=n)
    else:
        raise ValueError(f"Unsupported affinity distribution: {distribution!r}")

    return np.clip(values, 0.0, 1.0).astype(np.float32)
