"""
Builds the Task 2 transportation network graphs from the same World Cities
dataset used in Task 1, and provides a quick sanity check when run directly.

NOTE: this file imports data_loader.py (from Task 1). Make sure a copy of
data_loader.py is sitting in this same folder before running.
"""

from data_loader import load_cities, sample_cities
from graph_structures import build_knn_graph, build_undirected_version

DATA_PATH = r"D:\sem4\worldcities.csv"


def get_city_sample(n=300, seed=42):
    all_cities = load_cities(path=DATA_PATH)
    return sample_cities(all_cities, n, seed=seed)


if __name__ == "__main__":
    cities = get_city_sample(n=300)
    print(f"Sample size: {len(cities)} cities")

    g_directed = build_knn_graph(cities, k=6, directed=True)
    print(f"Directed graph:   {g_directed.num_nodes()} nodes, "
          f"{g_directed.num_edges()} edges, density={g_directed.density():.4f}")

    g_undirected = build_undirected_version(cities, k=6)
    print(f"Undirected graph: {g_undirected.num_nodes()} nodes, "
          f"{g_undirected.num_edges()} edges, density={g_undirected.density():.4f}")

    # sanity: every node should have at least k outgoing edges (directed graph)
    min_out_degree = min(len(g_directed.neighbors(n)) for n in g_directed.nodes())
    print(f"Min out-degree in directed graph: {min_out_degree} (expect >= 6)")