"""
Task 4: Simulated Annealing for VRPTW.

Operates on a full multi-route solution (unlike two_opt_improve, which only
reorders WITHIN a route). Two move types are used, both of which can move
customers BETWEEN routes:
    - relocate: move one customer from its current route to a different
      position (possibly a different route)
    - swap: exchange two customers between two (possibly different) routes

A move is always accepted if it improves total distance; a worsening move
is accepted with probability exp(-delta / temperature), and temperature is
geometrically cooled each iteration. This lets the search escape the local
optima that two_opt_improve (a pure hill-climber) can get stuck in, at the
cost of needing many more iterations (and therefore more runtime) to converge.
"""
import copy
import math
import random

from vrptw import distance, route_distance, route_is_feasible, solution_total_distance, DEPOT


def _try_relocate(routes, capacity, depot, rng):
    routes = [list(r) for r in routes]
    non_empty = [i for i, r in enumerate(routes) if r]
    if not non_empty:
        return routes, False
    ri = rng.choice(non_empty)
    if not routes[ri]:
        return routes, False
    ci = rng.randrange(len(routes[ri]))
    customer = routes[ri][ci]

    rj = rng.randrange(len(routes))
    new_routes = [list(r) for r in routes]
    new_routes[ri].pop(ci)
    insert_pos = rng.randint(0, len(new_routes[rj]))
    new_routes[rj].insert(insert_pos, customer)
    new_routes = [r for r in new_routes if r]  # drop emptied routes

    for r in new_routes:
        if not route_is_feasible(r, capacity, depot):
            return routes, False
    return new_routes, True


def _try_swap(routes, capacity, depot, rng):
    non_empty = [i for i, r in enumerate(routes) if r]
    if len(non_empty) < 1:
        return routes, False
    ri = rng.choice(non_empty)
    rj = rng.choice(non_empty)
    if not routes[ri] or not routes[rj]:
        return routes, False
    ci = rng.randrange(len(routes[ri]))
    cj = rng.randrange(len(routes[rj]))
    if ri == rj and ci == cj:
        return routes, False

    new_routes = [list(r) for r in routes]
    new_routes[ri][ci], new_routes[rj][cj] = new_routes[rj][cj], new_routes[ri][ci]

    for r in new_routes:
        if not route_is_feasible(r, capacity, depot):
            return routes, False
    return new_routes, True


def simulated_annealing(routes, capacity, depot=DEPOT, seed=1,
                         initial_temp=50.0, cooling_rate=0.995, iterations=2000):
    """
    Returns the best solution (list of routes) found. Starts from the given
    `routes` (typically the greedy or 2-opt solution) and explores
    relocate/swap moves under a standard SA acceptance criterion.
    """
    rng = random.Random(seed)
    current = [list(r) for r in routes]
    current_cost = solution_total_distance(current, depot)
    best = [list(r) for r in current]
    best_cost = current_cost
    temp = initial_temp

    for _ in range(iterations):
        move_fn = rng.choice([_try_relocate, _try_swap])
        candidate, applied = move_fn(current, capacity, depot, rng)
        if not applied:
            temp *= cooling_rate
            continue

        candidate_cost = solution_total_distance(candidate, depot)
        delta = candidate_cost - current_cost

        if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-6)):
            current = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best = [list(r) for r in current]
                best_cost = current_cost

        temp *= cooling_rate

    return best
