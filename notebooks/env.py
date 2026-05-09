import networkx as nx


DATASET_PATH = "facebook_combined.txt.gz"


def load_graph(path: str = DATASET_PATH) -> nx.Graph:
    return nx.read_edgelist(path, nodetype=int, create_using=nx.Graph())


def attach_placeholders(graph: nx.Graph) -> nx.Graph:
    placeholder_attrs = {
        "sentiment_vector": None,
        "conductivity": None,
        "fatigue_rate": None,
        "affinity": None,
    }

    for node in graph.nodes:
        graph.nodes[node].update(placeholder_attrs)

    return graph



if __name__ == "__main__":
    G = load_graph()
    attach_placeholders(G)

    sample_node = next(iter(G.nodes), None)

    print(f"nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    
    print(f"sample_node={G.nodes[0]}")
