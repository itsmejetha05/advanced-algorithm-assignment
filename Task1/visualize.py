"""
Generates the graphs used in the Task 1 report from the CSVs produced
by benchmark.py.
"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = r"D:\advanced assignment\Task1\results"
GRAPH_DIR = r"D:\advanced assignment\Task1\graphs"

COLORS = {
    "BST_random": "#1f77b4",
    "BST_sorted": "#d62728",
    "AVL": "#2ca02c",
    "Hash": "#9467bd",
    "MinHeap": "#ff7f0e",
}


def load_timings():
    rows = []
    with open(f"{RESULTS_DIR}/benchmark_results.csv") as f:
        for row in csv.DictReader(f):
            row["n"] = int(row["n"])
            row["total_time_s"] = float(row["total_time_s"])
            row["avg_time_ms"] = float(row["avg_time_ms"])
            rows.append(row)
    return rows


def load_heights():
    rows = []
    with open(f"{RESULTS_DIR}/tree_heights.csv") as f:
        for row in csv.DictReader(f):
            for k in row:
                row[k] = float(row[k])
            rows.append(row)
    return rows


def series_for(rows, operation, structure, value_key="total_time_s"):
    pts = [(r["n"], r[value_key]) for r in rows
           if r["operation"] == operation and r["structure"] == structure]
    pts.sort()
    return zip(*pts) if pts else ([], [])


def plot_insertion(rows):
    plt.figure(figsize=(7, 5))
    for struct in ["BST_random", "AVL", "Hash", "MinHeap"]:
        ns, ts = series_for(rows, "insert", struct)
        plt.plot(ns, ts, marker="o", label=struct, color=COLORS[struct])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of elements (n)")
    plt.ylabel("Total insertion time (s, log scale)")
    plt.title("Insertion Time vs n (average-case insertion order)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/1_insertion_time.png", dpi=150)
    plt.close()


def plot_bst_worst_case(rows):
    plt.figure(figsize=(7, 5))
    for struct in ["BST_random", "BST_sorted", "AVL"]:
        ns, ts = series_for(rows, "insert", struct)
        plt.plot(ns, ts, marker="o", label=struct, color=COLORS[struct])
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of elements (n)")
    plt.ylabel("Total insertion time (s, log scale)")
    plt.title("BST Worst Case: Sorted vs Random Insertion Order (vs AVL)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/2_bst_worst_case.png", dpi=150)
    plt.close()


def plot_search(rows):
    plt.figure(figsize=(7, 5))
    for struct in ["BST_random", "AVL", "Hash"]:
        ns, ts = series_for(rows, "search", struct, "avg_time_ms")
        plt.plot(ns, ts, marker="o", label=struct, color=COLORS[struct])
    plt.xscale("log")
    plt.xlabel("Number of elements (n)")
    plt.ylabel("Average search time (ms/op)")
    plt.title("Search Time vs n")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/3_search_time.png", dpi=150)
    plt.close()


def plot_delete(rows):
    plt.figure(figsize=(7, 5))
    for struct in ["BST_random", "AVL", "Hash"]:
        ns, ts = series_for(rows, "delete", struct, "avg_time_ms")
        plt.plot(ns, ts, marker="o", label=struct, color=COLORS[struct])
    plt.xscale("log")
    plt.xlabel("Number of elements (n)")
    plt.ylabel("Average delete time (ms/op)")
    plt.title("Deletion Time vs n")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/4_delete_time.png", dpi=150)
    plt.close()


def plot_heap(rows):
    plt.figure(figsize=(7, 5))
    ns, ts = series_for(rows, "insert", "MinHeap", "avg_time_ms")
    plt.plot(ns, ts, marker="o", label="Insert (push)", color="#ff7f0e")
    ns2, ts2 = series_for(rows, "extract_min", "MinHeap", "avg_time_ms")
    plt.plot(ns2, ts2, marker="s", label="Extract-min (pop)", color="#8c564b")
    plt.xscale("log")
    plt.xlabel("Number of elements (n)")
    plt.ylabel("Average time (ms/op)")
    plt.title("Min-Heap: Insert vs Extract-min Time")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/5_heap_ops.png", dpi=150)
    plt.close()


def plot_heights(heights):
    plt.figure(figsize=(7, 5))
    ns = [h["n"] for h in heights]
    plt.plot(ns, [h["BST_random_height"] for h in heights],
              marker="o", label="BST height (random insert)", color=COLORS["BST_random"])
    plt.plot(ns, [h["BST_sorted_height"] for h in heights],
              marker="o", label="BST height (sorted insert - worst case)", color=COLORS["BST_sorted"])
    plt.plot(ns, [h["AVL_height"] for h in heights],
              marker="o", label="AVL height (always balanced)", color=COLORS["AVL"])
    plt.plot(ns, [h["ideal_log2n"] for h in heights],
              linestyle="--", label="ideal log2(n)", color="black")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of elements (n)")
    plt.ylabel("Tree height (log scale)")
    plt.title("Tree Height Comparison: BST vs AVL")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/6_tree_heights.png", dpi=150)
    plt.close()


def plot_theory_vs_actual(rows):
    """Normalise insertion time by n*log2(n) (BST/AVL) and by n (Hash) to
    show that the ratio is roughly constant -- i.e. the actual timings
    track the theoretical complexity class, with a visible constant factor
    gap between structures."""
    import math
    plt.figure(figsize=(7, 5))
    for struct in ["BST_random", "AVL"]:
        ns, ts = series_for(rows, "insert", struct)
        ns, ts = list(ns), list(ts)
        normalised = [t / (n * math.log2(n)) * 1e6 for n, t in zip(ns, ts)]
        plt.plot(ns, normalised, marker="o", label=f"{struct}: time / (n log2 n)", color=COLORS[struct])
    ns, ts = series_for(rows, "insert", "Hash")
    ns, ts = list(ns), list(ts)
    normalised = [t / n * 1e6 for n, t in zip(ns, ts)]
    plt.plot(ns, normalised, marker="o", label="Hash: time / n", color=COLORS["Hash"])
    plt.xscale("log")
    plt.xlabel("Number of elements (n)")
    plt.ylabel("Normalised time (microseconds, arbitrary units)")
    plt.title("Theoretical vs Actual: Normalised Insertion Time\n(flat line = matches predicted complexity class)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/7_theory_vs_actual.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    timings = load_timings()
    heights = load_heights()
    plot_insertion(timings)
    plot_bst_worst_case(timings)
    plot_search(timings)
    plot_delete(timings)
    plot_heap(timings)
    plot_heights(heights)
    plot_theory_vs_actual(timings)
    print("Graphs written to", GRAPH_DIR)
