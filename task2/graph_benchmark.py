"""
Empirical benchmarking for Task 2: Dijkstra, Prim, Bellman-Ford.

Measures wall-clock runtime across graph sizes n in {100, 500, 1000, 2000}
(k=6 nearest-neighbour edges per node, so E ~= 6n - a consistently sparse
graph at every size, isolating the effect of n on runtime).

Also runs a separate dense-vs-sparse comparison at fixed n by varying k,
since the brief specifically asks for dense vs sparse suitability comparison.
"""
import csv
import os
import time

from build_city_graph import get_city_sample
from graph_structures import build_knn_graph, build_undirected_version, inject_negative_edges
from graph_algorithms import dijkstra, prim_mst, bellman_ford

SIZES = [100, 500, 1000, 2000]
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def time_it(func):
    start = time.perf_counter()
    result = func()
    return result, time.perf_counter() - start


def run_size_benchmark():
    rows = []
    for n in SIZES:
        print(f"\n=== n = {n} ===")
        cities = get_city_sample(n=n)
        source = cities[0]["name"]

        _, t_build_d = time_it(lambda: build_knn_graph(cities, k=6, directed=True))
        gd = build_knn_graph(cities, k=6, directed=True)
        gu = build_undirected_version(cities, k=6)
        g_neg = inject_negative_edges(gd, num_edges=max(3, n // 100))

        print(f"  Graph: {gd.num_nodes()} nodes, {gd.num_edges()} edges, "
              f"density={gd.density():.5f}")

        _, t_dij = time_it(lambda: dijkstra(gd, source))
        _, t_prim = time_it(lambda: prim_mst(gu, start=source))
        _, t_bf = time_it(lambda: bellman_ford(g_neg, source))

        print(f"  Dijkstra:     {t_dij:.6f}s")
        print(f"  Prim:         {t_prim:.6f}s")
        print(f"  Bellman-Ford: {t_bf:.6f}s")

        rows.append({
            "n": n, "edges": gd.num_edges(), "density": gd.density(),
            "dijkstra_s": t_dij, "prim_s": t_prim, "bellman_ford_s": t_bf,
        })

    with open(f"{RESULTS_DIR}/graph_algo_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "edges", "density",
                                                "dijkstra_s", "prim_s", "bellman_ford_s"])
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved results/graph_algo_benchmark.csv")


def run_density_benchmark(n=500):
    """Fixed n=500, varying k (neighbours per node) from sparse to dense,
    to compare algorithm suitability for sparse vs dense graphs directly."""
    print(f"\n=== Density comparison at n={n} ===")
    cities = get_city_sample(n=n)
    source = cities[0]["name"]
    rows = []
    for k in [3, 6, 12, 25, 50, 100]:
        gd = build_knn_graph(cities, k=k, directed=True)
        gu = build_undirected_version(cities, k=k)
        g_neg = inject_negative_edges(gd, num_edges=10)

        _, t_dij = time_it(lambda: dijkstra(gd, source))
        _, t_prim = time_it(lambda: prim_mst(gu, start=source))
        _, t_bf = time_it(lambda: bellman_ford(g_neg, source))

        density = gd.density()
        print(f"  k={k:3d}  edges={gd.num_edges():6d}  density={density:.4f}  "
              f"dijkstra={t_dij:.5f}s  prim={t_prim:.5f}s  bf={t_bf:.5f}s")
        rows.append({
            "k": k, "n": n, "edges": gd.num_edges(), "density": density,
            "dijkstra_s": t_dij, "prim_s": t_prim, "bellman_ford_s": t_bf,
        })

    with open(f"{RESULTS_DIR}/graph_density_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "n", "edges", "density",
                                                "dijkstra_s", "prim_s", "bellman_ford_s"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved results/graph_density_benchmark.csv")


if __name__ == "__main__":
    run_size_benchmark()
    run_density_benchmark()