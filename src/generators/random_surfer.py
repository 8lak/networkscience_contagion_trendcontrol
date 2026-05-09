"""Random-surfer graph generator."""

from __future__ import annotations

import networkx as nx
import numpy as np


def multi_edge_random_surfer(
    n: int,
    m: int = 3,
    p: float = 0.15,
    seed: int | None = None,
) -> nx.Graph:
    """Generate an undirected graph with random-surfer attachment."""
    if n < 1:
        raise ValueError("n must be positive")
    if m < 1:
        raise ValueError("m must be positive")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")

    rng = np.random.default_rng(seed)
    adjacency: dict[int, list[int]] = {0: []}

    for t in range(1, n):
        adjacency[t] = []
        targets: set[int] = set()

        for _ in range(m):
            u = int(rng.integers(0, t))
            while rng.random() > p and adjacency[u]:
                u = int(rng.choice(adjacency[u]))
            targets.add(u)

        for target in targets:
            adjacency[t].append(target)
            adjacency[target].append(t)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for source, targets in adjacency.items():
        for target in targets:
            graph.add_edge(source, target)
    return graph
