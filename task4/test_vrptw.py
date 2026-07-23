"""
Correctness tests for Task 4: VRPTW greedy construction, 2-opt, and SA.
"""
from vrptw import (generate_instance, solution_is_feasible, solution_total_distance,
                    route_is_feasible, DEPOT)
from vrptw_heuristics import greedy_construction, two_opt_improve_all_routes
from vrptw_sa import simulated_annealing


def test_greedy_produces_feasible_solution():
    customers, capacity = generate_instance(num_customers=30, seed=1)
    routes = greedy_construction(customers, capacity)
    assert solution_is_feasible(routes, capacity, customers), "greedy solution infeasible"
    total_visited = sum(len(r) for r in routes)
    assert total_visited == len(customers), "not all customers visited exactly once"
    print(f"Greedy construction: OK - {len(routes)} routes, "
          f"distance={solution_total_distance(routes):.1f}")


def test_two_opt_improves_or_matches_greedy():
    customers, capacity = generate_instance(num_customers=30, seed=1)
    greedy_routes = greedy_construction(customers, capacity)
    greedy_dist = solution_total_distance(greedy_routes)

    improved_routes = two_opt_improve_all_routes(greedy_routes, capacity)
    assert solution_is_feasible(improved_routes, capacity, customers), "2-opt broke feasibility"
    improved_dist = solution_total_distance(improved_routes)

    assert improved_dist <= greedy_dist + 1e-9, \
        f"2-opt made things worse: {improved_dist} > {greedy_dist}"
    print(f"2-opt local search: OK - distance {greedy_dist:.1f} -> {improved_dist:.1f} "
          f"({'improved' if improved_dist < greedy_dist - 1e-6 else 'no change'})")


def test_simulated_annealing_feasible_and_not_worse():
    customers, capacity = generate_instance(num_customers=20, seed=2)
    greedy_routes = greedy_construction(customers, capacity)
    greedy_dist = solution_total_distance(greedy_routes)

    sa_routes = simulated_annealing(greedy_routes, capacity, seed=1, iterations=500)
    assert solution_is_feasible(sa_routes, capacity, customers), "SA solution infeasible"
    sa_dist = solution_total_distance(sa_routes)

    # SA starts from the greedy solution and only keeps improving moves as
    # its best-so-far, so it should never end up worse than where it started
    assert sa_dist <= greedy_dist + 1e-6, f"SA best solution worse than start: {sa_dist} > {greedy_dist}"
    print(f"Simulated Annealing: OK - distance {greedy_dist:.1f} -> {sa_dist:.1f}")


def test_capacity_violation_detected():
    customers, capacity = generate_instance(num_customers=10, seed=3)
    # deliberately construct a route that exceeds capacity
    overloaded = customers[:]  # all 10 customers in one route - almost certainly over capacity
    total_demand = sum(c.demand for c in overloaded)
    if total_demand <= capacity:
        # force it to exceed by inflating demand check manually
        assert not route_is_feasible(overloaded[:1], 0), "expected infeasible with zero capacity"
    else:
        assert not route_is_feasible(overloaded, capacity), "expected capacity violation to be caught"
    print("Capacity violation detection: OK")


def test_time_window_violation_detected():
    from vrptw import Customer
    depot = DEPOT
    # customer whose window closes before the vehicle could possibly arrive
    far_but_early_window = Customer(id=1, x=1000, y=1000, demand=1, earliest=0, latest=1, service_time=0)
    assert not route_is_feasible([far_but_early_window], capacity=100, depot=depot), \
        "expected time window violation to be caught"
    print("Time window violation detection: OK")


if __name__ == "__main__":
    test_greedy_produces_feasible_solution()
    test_two_opt_improves_or_matches_greedy()
    test_simulated_annealing_feasible_and_not_worse()
    test_capacity_violation_detected()
    test_time_window_violation_detected()
    print("\nAll Task 4 correctness tests passed.")
