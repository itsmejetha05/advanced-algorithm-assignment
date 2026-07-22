import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results"
GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)


def load(name):
    rows = []
    with open(f"{RESULTS_DIR}/{name}") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def to_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def plot_job_scheduling():
    rows = load("job_scheduling_benchmark.csv")
    small_n_rows = [r for r in rows if r["brute_force_s"]]
    ns = [int(r["n"]) for r in small_n_rows]
    dp = [to_float(r["dp_s"]) for r in small_n_rows]
    bf = [to_float(r["brute_force_s"]) for r in small_n_rows]

    plt.figure(figsize=(7, 5))
    plt.plot(ns, dp, marker="o", label="DP (bottom-up), O(n log n)")
    plt.plot(ns, bf, marker="o", label="Brute force, O(2^n)")
    plt.yscale("log")
    plt.xlabel("Number of jobs (n)")
    plt.ylabel("Runtime (s, log scale)")
    plt.title("Weighted Job Scheduling: DP vs Brute Force\n(low-overlap jobs - brute force's true worst case)")
    plt.legend(); plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/fig1_job_scheduling_dp_vs_brute.png", dpi=150)
    plt.close()

    # DP alone at large scale
    all_rows = [r for r in rows if not r["brute_force_s"]]
    ns2 = [int(r["n"]) for r in all_rows]
    dp2 = [to_float(r["dp_s"]) for r in all_rows]
    plt.figure(figsize=(7, 5))
    plt.plot(ns2, dp2, marker="o", color="#2ca02c")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Number of jobs (n)")
    plt.ylabel("Runtime (s, log scale)")
    plt.title("Weighted Job Scheduling: DP Scaling to Large n\n(brute force infeasible beyond ~n=25)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/fig2_job_scheduling_dp_scaling.png", dpi=150)
    plt.close()


def plot_min_platforms():
    rows = load("min_platforms_benchmark.csv")
    ns = [int(r["n"]) for r in rows]
    greedy = [to_float(r["greedy_s"]) for r in rows]
    brute = [to_float(r["brute_force_s"]) for r in rows]

    plt.figure(figsize=(7, 5))
    plt.plot(ns, greedy, marker="o", label="Greedy, O(n log n)")
    bf_ns = [n for n, b in zip(ns, brute) if b is not None]
    bf_vals = [b for b in brute if b is not None]
    plt.plot(bf_ns, bf_vals, marker="o", label="Brute force, O(n\u00b2)")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Number of trains (n)")
    plt.ylabel("Runtime (s, log scale)")
    plt.title("Minimum Platforms: Greedy vs Brute Force")
    plt.legend(); plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/fig3_min_platforms.png", dpi=150)
    plt.close()


def plot_knights_tour():
    rows = load("knights_tour_benchmark.csv")
    ns = [int(r["n"]) for r in rows]
    warn_calls = [int(r["warnsdorff_calls"]) for r in rows]
    plain_calls = [(int(r["n"]), int(r["plain_calls"])) for r in rows if r["plain_calls"]]

    plt.figure(figsize=(7, 5))
    plt.plot(ns, warn_calls, marker="o", label="Warnsdorff-guided backtracking")
    if plain_calls:
        pn, pc = zip(*plain_calls)
        plt.plot(pn, pc, marker="o", label="Plain (fixed-order) backtracking")
    plt.yscale("log")
    plt.xlabel("Board size (n x n)")
    plt.ylabel("Recursive calls made (log scale)")
    plt.title("Knight's Tour: Search Space Explored\n(Warnsdorff pruning vs plain backtracking)")
    plt.legend(); plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/fig4_knights_tour_calls.png", dpi=150)
    plt.close()

    # Warnsdorff scaling alone (since it's the only one that scales to n=20)
    plt.figure(figsize=(7, 5))
    plt.plot(ns, warn_calls, marker="o", color="#2ca02c")
    plt.xlabel("Board size (n x n)")
    plt.ylabel("Recursive calls made")
    plt.title("Knight's Tour: Warnsdorff-Guided Search Scales ~O(n\u00b2)\n(near-linear in board cells, almost no backtracking needed)")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/fig5_knights_tour_warnsdorff_scaling.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    plot_job_scheduling()
    plot_min_platforms()
    plot_knights_tour()
    print("All Task 3 graphs written to", GRAPH_DIR)
