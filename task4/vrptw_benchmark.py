"""
Task 4 benchmark: compares Greedy, Greedy+2-opt, and Greedy+2-opt+SA across
increasing problem sizes, measuring both solution quality (total distance)
and runtime - the core trade-off the brief asks for.
"""
import csv
import os
import time

from vrptw import generate_instance, solution_total_distance, solution_is_feasible
from vrptw_heuristics import greedy_construction, two_opt_improve_all_routes
from vrptw_sa import simulated_annealing

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def time_it(func):
    start = time.perf_counter()
    result = func()
    return result, time.perf_counter() - start


def run_benchmark():
    rows = []
    for n in [20, 50, 100, 200]:
        print(f"\n=== n = {n} customers ===")
        customers, capacity = generate_instance(num_customers=n, seed=42)

        greedy_routes, t_greedy = time_it(lambda: greedy_construction(customers, capacity))
        greedy_dist = solution_total_distance(greedy_routes)
        assert solution_is_feasible(greedy_routes, capacity, customers)

        two_opt_routes, t_2opt = time_it(lambda: two_opt_improve_all_routes(greedy_routes, capacity))
        two_opt_dist = solution_total_distance(two_opt_routes)
        assert solution_is_feasible(two_opt_routes, capacity, customers)

        sa_routes, t_sa = time_it(lambda: simulated_annealing(two_opt_routes, capacity, iterations=2000))
        sa_dist = solution_total_distance(sa_routes)
        assert solution_is_feasible(sa_routes, capacity, customers)

        print(f"  Greedy:          dist={greedy_dist:8.1f}  time={t_greedy:.4f}s  "
              f"vehicles={len(greedy_routes)}")
        print(f"  Greedy+2opt:     dist={two_opt_dist:8.1f}  time={t_2opt:.4f}s  "
              f"({(1 - two_opt_dist/greedy_dist)*100:.1f}% better than greedy)")
        print(f"  Greedy+2opt+SA:  dist={sa_dist:8.1f}  time={t_sa:.4f}s  "
              f"({(1 - sa_dist/greedy_dist)*100:.1f}% better than greedy)")

        rows.append({
            "n": n,
            "greedy_dist": greedy_dist, "greedy_time_s": t_greedy, "greedy_vehicles": len(greedy_routes),
            "two_opt_dist": two_opt_dist, "two_opt_time_s": t_2opt,
            "sa_dist": sa_dist, "sa_time_s": t_sa,
        })

    with open(f"{RESULTS_DIR}/vrptw_benchmark.csv", "w", newline="") as f:
        fieldnames = ["n", "greedy_dist", "greedy_time_s", "greedy_vehicles",
                      "two_opt_dist", "two_opt_time_s", "sa_dist", "sa_time_s"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved results/vrptw_benchmark.csv")


if __name__ == "__main__":
    run_benchmark()
