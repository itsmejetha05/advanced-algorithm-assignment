"""
Task 5 benchmark: measures wall-clock speedup of threaded vs multiprocess
parallel sort at 1, 2, 4, and 8 workers, against a sequential baseline.

This deliberately contrasts threading (GIL-bound, limited real speedup for
CPU-bound work) against multiprocessing (true parallelism, real speedup)
on the SAME sorting workload, to make the difference concrete rather than
theoretical.
"""
import csv
import os
import random
import time

from parallel_sort import sequential_merge_sort, threaded_parallel_sort, multiprocess_parallel_sort

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

WORKER_COUNTS = [1, 2, 4, 8]


def time_it(func):
    start = time.perf_counter()
    func()
    return time.perf_counter() - start


def run_benchmark(n=200_000):
    rng = random.Random(42)
    arr = [rng.randint(0, 10_000_000) for _ in range(n)]

    print(f"\n=== n = {n} elements ===")
    t_seq = time_it(lambda: sequential_merge_sort(arr))
    print(f"  Sequential:            {t_seq:.4f}s")

    rows = []
    for workers in WORKER_COUNTS:
        t_threaded = time_it(lambda: threaded_parallel_sort(arr, num_threads=workers))
        speedup_threaded = t_seq / t_threaded
        print(f"  Threaded  ({workers} threads):  {t_threaded:.4f}s  "
              f"(speedup={speedup_threaded:.2f}x)")

        t_mp = time_it(lambda: multiprocess_parallel_sort(arr, num_processes=workers))
        speedup_mp = t_seq / t_mp
        print(f"  Multiproc ({workers} procs):    {t_mp:.4f}s  "
              f"(speedup={speedup_mp:.2f}x)")

        rows.append({
            "workers": workers,
            "sequential_s": t_seq,
            "threaded_s": t_threaded, "threaded_speedup": speedup_threaded,
            "multiprocess_s": t_mp, "multiprocess_speedup": speedup_mp,
        })

    with open(f"{RESULTS_DIR}/concurrency_benchmark.csv", "w", newline="") as f:
        fieldnames = ["workers", "sequential_s", "threaded_s", "threaded_speedup",
                      "multiprocess_s", "multiprocess_speedup"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved results/concurrency_benchmark.csv")


if __name__ == "__main__":
    run_benchmark()
