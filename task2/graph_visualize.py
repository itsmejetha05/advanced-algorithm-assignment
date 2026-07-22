"""
Generates the Task 2 graphs:
  1. Runtime vs n (log-log) for Dijkstra / Prim / Bellman-Ford
  2. Runtime vs graph density (fixed n, varying k)
  3. Step-by-step Dijkstra shortest-path-tree construction (on a real map)
  4. Step-by-step Prim MST construction (on a real map)
  5. Theoretical complexity vs actual runtime (normalised)
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_city_graph import get_city_sample
from graph_structures import build_knn_graph, build_undirected_version
from graph_algorithms import dijkstra, prim_mst

RESULTS_DIR = "results"
GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)


def load_csv(name):
    rows = []
    with open(f"{RESULTS_DIR}/{name}") as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if k != "n" and k != "k" and k != "edges" else int(float(v)))
                         for k, v in row.items()})
    return rows


def plot_runtime_vs_n():
    rows = load_csv("graph_algo_benchmark.csv")
    ns = [r["n"] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(ns, [r["dijkstra_s"] for r in rows], marker="o", label="Dijkstra")
    plt.plot(ns, [r["prim_s"] for r in rows], marker="o", label="Prim (MST)")
    plt.plot(ns, [r["bellman_ford_s"] for r in rows], marker="o", label="Bellman-Ford")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Number of nodes (n), k=6 fixed (sparse)")
    plt.ylabel("Runtime (s, log scale)")
    plt.title("Algorithm Runtime vs Graph Size (sparse, k=6)")
    plt.legend(); plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/fig1_runtime_vs_n.png", dpi=150)
    plt.close()


def plot_runtime_vs_density():
    rows = load_csv("graph_density_benchmark.csv")
    densities = [r["density"] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(densities, [r["dijkstra_s"] for r in rows], marker="o", label="Dijkstra")
    plt.plot(densities, [r["prim_s"] for r in rows], marker="o", label="Prim (MST)")
    plt.plot(densities, [r["bellman_ford_s"] for r in rows], marker="o", label="Bellman-Ford")
    plt.xlabel("Graph density (n=500 fixed, k varied 3\u2192100)")
    plt.ylabel("Runtime (s)")
    plt.title("Algorithm Runtime vs Graph Density (Sparse \u2192 Dense)")
    plt.legend(); plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/fig2_runtime_vs_density.png", dpi=150)
    plt.close()


def plot_dijkstra_steps():
    cities = get_city_sample(n=150, seed=7)
    g = build_knn_graph(cities, k=6, directed=True)
    source = cities[0]["name"]
    dist, prev, steps = dijkstra(g, source, record_steps=True)

    lons = [c["lng"] for c in cities]
    lats = [c["lat"] for c in cities]
    name_to_coord = {c["name"]: (c["lng"], c["lat"]) for c in cities}

    plt.figure(figsize=(9, 6))
    plt.scatter(lons, lats, c="lightgrey", s=15, zorder=1)

    # colour nodes by the ORDER they were finalised (step index) - a proxy
    # for "distance layers" expanding outward from the source
    order_x, order_y, order_c = [], [], []
    for i, (node, d) in enumerate(steps):
        x, y = name_to_coord[node]
        order_x.append(x); order_y.append(y); order_c.append(i)

    sc = plt.scatter(order_x, order_y, c=order_c, cmap="viridis", s=25, zorder=2)
    plt.colorbar(sc, label="Order finalised (Dijkstra expansion step)")
    sx, sy = name_to_coord[source]
    plt.scatter([sx], [sy], c="red", s=120, marker="*", zorder=3, label=f"Source: {source}")

    # draw the shortest-path tree edges (prev pointers)
    for node, parent in prev.items():
        if parent is not None:
            x1, y1 = name_to_coord[node]
            x2, y2 = name_to_coord[parent]
            plt.plot([x1, x2], [y1, y2], color="steelblue", linewidth=0.4, alpha=0.5, zorder=1)

    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title(f"Dijkstra Shortest-Path Tree from {source}\n(colour = order node was finalised)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/fig3_dijkstra_shortest_path_tree.png", dpi=150)
    plt.close()


def plot_prim_steps():
    cities = get_city_sample(n=150, seed=7)
    g = build_undirected_version(cities, k=6)
    source = cities[0]["name"]
    mst_edges, total_weight, steps = prim_mst(g, start=source, record_steps=True)

    name_to_coord = {c["name"]: (c["lng"], c["lat"]) for c in cities}
    lons = [c["lng"] for c in cities]
    lats = [c["lat"] for c in cities]

    plt.figure(figsize=(9, 6))
    plt.scatter(lons, lats, c="lightgrey", s=15, zorder=1)

    order_x, order_y, order_c = [], [], []
    for i, (node, parent, w) in enumerate(steps):
        x, y = name_to_coord[node]
        order_x.append(x); order_y.append(y); order_c.append(i)

    sc = plt.scatter(order_x, order_y, c=order_c, cmap="plasma", s=25, zorder=2)
    plt.colorbar(sc, label="Order node was added to MST")
    sx, sy = name_to_coord[source]
    plt.scatter([sx], [sy], c="red", s=120, marker="*", zorder=3, label=f"Start: {source}")

    for u, v, w in mst_edges:
        x1, y1 = name_to_coord[u]
        x2, y2 = name_to_coord[v]
        plt.plot([x1, x2], [y1, y2], color="darkorange", linewidth=0.6, alpha=0.6, zorder=1)

    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.title(f"Prim's MST Construction (total weight = {total_weight:.0f} km)\n(colour = order node was added)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/fig4_prim_mst_construction.png", dpi=150)
    plt.close()


def plot_theory_vs_actual():
    import math
    rows = load_csv("graph_algo_benchmark.csv")
    plt.figure(figsize=(7, 5))
    for key, label, color in [
        ("dijkstra_s", "Dijkstra: time / ((n+E) log n)", "#1f77b4"),
        ("prim_s", "Prim: time / ((n+E) log n)", "#2ca02c"),
    ]:
        ns = [r["n"] for r in rows]
        es = [r["edges"] for r in rows]
        ts = [r[key] for r in rows]
        norm = [t / ((n + e) * math.log2(n)) * 1e6 for n, e, t in zip(ns, es, ts)]
        plt.plot(ns, norm, marker="o", label=label, color=color)

    ns = [r["n"] for r in rows]
    es = [r["edges"] for r in rows]
    ts = [r["bellman_ford_s"] for r in rows]
    norm_bf = [t / (n * e) * 1e9 for n, e, t in zip(ns, es, ts)]
    plt.plot(ns, norm_bf, marker="o", label="Bellman-Ford: time / (n\u00b7E)", color="#d62728")

    plt.xscale("log")
    plt.xlabel("Number of nodes (n)")
    plt.ylabel("Normalised time (arbitrary units)")
    plt.title("Theoretical vs Actual: Normalised Runtime\n(flat line = matches predicted complexity class)")
    plt.legend(); plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/fig5_theory_vs_actual.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    plot_runtime_vs_n()
    plot_runtime_vs_density()
    plot_dijkstra_steps()
    plot_prim_steps()
    plot_theory_vs_actual()
    print("All Task 2 graphs written to", GRAPH_DIR)