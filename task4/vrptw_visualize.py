import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vrptw import generate_instance, DEPOT, solution_total_distance
from vrptw_heuristics import greedy_construction, two_opt_improve_all_routes
from vrptw_sa import simulated_annealing

RESULTS_DIR = "results"
GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)


def load_csv(name):
    rows = []
    with open(f"{RESULTS_DIR}/{name}") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def plot_quality_comparison():
    rows = load_csv("vrptw_benchmark.csv")
    ns = [r["n"] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(ns, [r["greedy_dist"] for r in rows], marker="o", label="Greedy")
    plt.plot(ns, [r["two_opt_dist"] for r in rows], marker="o", label="Greedy + 2-opt")
    plt.plot(ns, [r["sa_dist"] for r in rows], marker="o", label="Greedy + 2-opt + SA")
    plt.xlabel("Number of customers (n)")
    plt.ylabel("Total route distance")
    plt.title("VRPTW Solution Quality: Greedy vs 2-opt vs Simulated Annealing")
    plt.legend(); plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/fig1_solution_quality.png", dpi=150)
    plt.close()


def plot_runtime_comparison():
    rows = load_csv("vrptw_benchmark.csv")
    ns = [r["n"] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(ns, [r["greedy_time_s"] for r in rows], marker="o", label="Greedy")
    plt.plot(ns, [r["two_opt_time_s"] for r in rows], marker="o", label="Greedy + 2-opt")
    plt.plot(ns, [r["sa_time_s"] for r in rows], marker="o", label="Greedy + 2-opt + SA")
    plt.yscale("log")
    plt.xlabel("Number of customers (n)")
    plt.ylabel("Runtime (s, log scale)")
    plt.title("VRPTW Runtime: Greedy vs 2-opt vs Simulated Annealing")
    plt.legend(); plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/fig2_runtime.png", dpi=150)
    plt.close()


def plot_routes_before_after():
    customers, capacity = generate_instance(num_customers=25, seed=7)
    greedy_routes = greedy_construction(customers, capacity)
    sa_routes = simulated_annealing(
        two_opt_improve_all_routes(greedy_routes, capacity), capacity, iterations=3000)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = plt.cm.tab20.colors

    for ax, routes, title in [(axes[0], greedy_routes, "Greedy Construction"),
                               (axes[1], sa_routes, "After 2-opt + Simulated Annealing")]:
        ax.scatter([DEPOT.x], [DEPOT.y], c="red", s=150, marker="s", zorder=3, label="Depot")
        for i, route in enumerate(routes):
            xs = [DEPOT.x] + [c.x for c in route] + [DEPOT.x]
            ys = [DEPOT.y] + [c.y for c in route] + [DEPOT.y]
            ax.plot(xs, ys, marker="o", color=colors[i % len(colors)], linewidth=1.2, markersize=4)
        dist = solution_total_distance(routes)
        ax.set_title(f"{title}\n{len(routes)} vehicles, distance={dist:.0f}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/fig3_routes_before_after.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    plot_quality_comparison()
    plot_runtime_comparison()
    plot_routes_before_after()
    print("All Task 4 graphs written to", GRAPH_DIR)
