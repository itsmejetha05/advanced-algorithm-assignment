"""
Empirical benchmarking for Task 3:
  1. Weighted Job Scheduling: DP (bottom-up, O(n log n)) vs brute force (O(2^n))
  2. Minimum Platforms: greedy (O(n log n)) vs brute force (O(n^2))
  3. Knight's Tour: plain backtracking vs Warnsdorff-heuristic backtracking,
     across increasing board sizes - measuring both wall-clock time AND
     recursive calls made (a direct measure of search-space pruning).
"""
import csv
import os
import random
import time

from job_scheduling import Job, weighted_job_scheduling_bottom_up, brute_force_job_scheduling
from min_platforms import min_platforms_greedy, min_platforms_brute_force
from knights_tour import plain_knights_tour, warnsdorff_knights_tour

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def time_it(func):
    start = time.perf_counter()
    result = func()
    return result, time.perf_counter() - start


def benchmark_job_scheduling():
    print("\n=== Weighted Job Scheduling: DP vs Brute Force ===")
    rng = random.Random(42)
    rows = []

    # DP alone, scaling to large n (brute force would never finish here)
    for n in [100, 1000, 10000, 100000]:
        jobs = [Job(s := rng.randint(0, n * 2), s + rng.randint(1, 50), rng.randint(1, 1000))
                for _ in range(n)]
        _, t_dp = time_it(lambda: weighted_job_scheduling_bottom_up(jobs))
        print(f"  DP only   n={n:6d}  time={t_dp:.6f}s")
        rows.append({"n": n, "dp_s": t_dp, "brute_force_s": ""})

    # DP vs brute force, small n only (brute force is O(2^n)).
    # Jobs are spread far apart (low overlap) so few branches get pruned by
    # the non-overlap constraint - this is closer to brute force's TRUE
    # worst case than random/dense-overlap jobs, where many "take" branches
    # get cut early and runtime looks deceptively tame.
    for n in [15, 18, 20, 22, 24]:
        jobs = [Job(s := i * 100, s + rng.randint(1, 5), rng.randint(1, 1000)) for i in range(n)]
        _, t_dp = time_it(lambda: weighted_job_scheduling_bottom_up(jobs))
        _, t_bf = time_it(lambda: brute_force_job_scheduling(jobs))
        print(f"  DP vs BF  n={n:6d}  dp={t_dp:.6f}s  brute_force={t_bf:.6f}s  "
              f"(brute_force / dp = {t_bf/t_dp:.1f}x)")
        rows.append({"n": n, "dp_s": t_dp, "brute_force_s": t_bf})

    with open(f"{RESULTS_DIR}/job_scheduling_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "dp_s", "brute_force_s"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved results/job_scheduling_benchmark.csv")


def benchmark_min_platforms():
    print("\n=== Minimum Platforms: Greedy vs Brute Force ===")
    rng = random.Random(7)
    rows = []
    for n in [10, 50, 100, 500, 1000, 5000, 10000]:
        arrivals = [rng.randint(0, n * 10) for _ in range(n)]
        departures = [a + rng.randint(1, 100) for a in arrivals]

        _, t_greedy = time_it(lambda: min_platforms_greedy(arrivals, departures))
        if n <= 5000:  # O(n^2) brute force - cap to keep runtime reasonable
            _, t_brute = time_it(lambda: min_platforms_brute_force(arrivals, departures))
        else:
            t_brute = None

        bf_str = f"{t_brute:.6f}s" if t_brute is not None else "skipped (too slow)"
        print(f"  n={n:6d}  greedy={t_greedy:.6f}s  brute_force={bf_str}")
        rows.append({"n": n, "greedy_s": t_greedy, "brute_force_s": t_brute if t_brute else ""})

    with open(f"{RESULTS_DIR}/min_platforms_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "greedy_s", "brute_force_s"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved results/min_platforms_benchmark.csv")


def benchmark_knights_tour():
    print("\n=== Knight's Tour: Plain Backtracking vs Warnsdorff ===")
    rows = []
    MAX_CALLS = 2_000_000  # abort plain backtracking if it blows up

    for n in [5, 6, 7, 8, 10, 12, 15, 20]:
        _, t_warn = time_it(lambda: warnsdorff_knights_tour(n))
        path_w, calls_w = warnsdorff_knights_tour(n)

        if n <= 6:  # plain backtracking only feasible for very small boards
            t0 = time.perf_counter()
            path_p, calls_p = plain_knights_tour(n, max_calls=MAX_CALLS)
            t_plain = time.perf_counter() - t0
            plain_str = f"{t_plain:.4f}s ({calls_p} calls)"
        else:
            t_plain, calls_p = None, None
            plain_str = "skipped (infeasible)"

        found_w = "yes" if path_w else "no"
        print(f"  n={n:3d}  Warnsdorff: {t_warn:.6f}s ({calls_w} calls, tour found={found_w})  "
              f"| Plain: {plain_str}")

        rows.append({
            "n": n, "warnsdorff_s": t_warn, "warnsdorff_calls": calls_w,
            "plain_s": t_plain if t_plain else "", "plain_calls": calls_p if calls_p else "",
        })

    with open(f"{RESULTS_DIR}/knights_tour_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "warnsdorff_s", "warnsdorff_calls",
                                                "plain_s", "plain_calls"])
        writer.writeheader()
        writer.writerows(rows)
    print("Saved results/knights_tour_benchmark.csv")


if __name__ == "__main__":
    benchmark_job_scheduling()
    benchmark_min_platforms()
    benchmark_knights_tour()
