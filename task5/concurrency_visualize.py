import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)


def load():
    rows = []
    with open(f"{RESULTS_DIR}/concurrency_benchmark.csv") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def plot_speedup_vs_workers():
    rows = load()
    workers = [int(r["workers"]) for r in rows]
    threaded_speedup = [r["threaded_speedup"] for r in rows]
    mp_speedup = [r["multiprocess_speedup"] for r in rows]

    plt.figure(figsize=(7, 5))
    plt.plot(workers, threaded_speedup, marker="o", label="Threading (GIL-bound)")
    plt.plot(workers, mp_speedup, marker="o", label="Multiprocessing (true parallelism)")
    plt.plot(workers, workers, linestyle="--", color="grey", label="Ideal linear speedup", alpha=0.6)
    plt.xlabel("Number of workers (threads / processes)")
    plt.ylabel("Speedup vs sequential (x)")
    plt.title("Speedup vs Worker Count: Threading vs Multiprocessing")
    plt.legend(); plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/fig1_speedup_vs_workers.png", dpi=150)
    plt.close()


def plot_runtime_vs_workers():
    rows = load()
    workers = [int(r["workers"]) for r in rows]
    threaded = [r["threaded_s"] for r in rows]
    mp = [r["multiprocess_s"] for r in rows]
    seq = rows[0]["sequential_s"]

    plt.figure(figsize=(7, 5))
    plt.axhline(seq, linestyle="--", color="grey", label="Sequential baseline", alpha=0.7)
    plt.plot(workers, threaded, marker="o", label="Threading")
    plt.plot(workers, mp, marker="o", label="Multiprocessing")
    plt.xlabel("Number of workers")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs Worker Count")
    plt.legend(); plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{GRAPH_DIR}/fig2_runtime_vs_workers.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    plot_speedup_vs_workers()
    plot_runtime_vs_workers()
    print("All Task 5 graphs written to", GRAPH_DIR)
