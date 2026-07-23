"""
Task 4: Heuristics for VRPTW.

1. greedy_construction(): nearest-feasible-neighbour construction heuristic.
   Builds one route at a time; repeatedly appends the closest remaining
   customer that keeps the route capacity- and time-window-feasible. When
   no remaining customer can be feasibly appended, the vehicle returns to
   the depot and a new route/vehicle begins. Simple, fast, but can leave
   obviously-improvable crossed/zig-zag routes.

2. two_opt_improve(): classic 2-opt local search applied WITHIN each route
   (since customers are partitioned by route, and reordering across routes
   would need a more complex move set - within-route 2-opt is the standard
   "hill climbing" improvement step for a constructed VRP solution). For
   every pair of edges in a route, tries reversing the segment between them
   and keeps the change if it (a) still satisfies time windows and (b)
   strictly reduces the route's distance. Repeats until no improving move
   is found (local optimum) or a max-iteration cap is hit.
"""
from vrptw import distance, route_distance, route_is_feasible, DEPOT


def greedy_construction(customers, capacity, depot=DEPOT):
    """Returns a list of routes (each a list of Customer)."""
    unvisited = set(c.id for c in customers)
    by_id = {c.id: c for c in customers}
    routes = []

    while unvisited:
        route = []
        current = depot
        current_load = 0
        current_time = 0.0

        while True:
            best_next = None
            best_dist = float("inf")
            for cid in unvisited:
                cand = by_id[cid]
                if current_load + cand.demand > capacity:
                    continue
                d = distance(current, cand)
                arrival = current_time + d
                if arrival > cand.latest:
                    continue  # would arrive too late
                if d < best_dist:
                    best_dist = d
                    best_next = cand

            if best_next is None:
                break  # no feasible next customer - close this route

            route.append(best_next)
            unvisited.discard(best_next.id)
            current_load += best_next.demand
            arrival = current_time + best_dist
            current_time = max(arrival, best_next.earliest) + best_next.service_time
            current = best_next

        if route:
            routes.append(route)
        else:
            # safety valve: a single remaining customer is infeasible from
            # the depot alone (shouldn't happen with sane generated
            # instances, but avoids an infinite loop if it does)
            stranded = unvisited.pop()
            routes.append([by_id[stranded]])

    return routes


def two_opt_improve(route, capacity, depot=DEPOT, max_iterations=200):
    """
    Within-route 2-opt: repeatedly tries reversing route[i:j] for all
    i < j, keeping the reversal if it reduces distance and remains
    time-window feasible. Returns the improved route.
    """
    best = list(route)
    best_dist = route_distance(best, depot)
    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        n = len(best)
        for i in range(n - 1):
            for j in range(i + 1, n):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                cand_dist = route_distance(candidate, depot)
                if cand_dist < best_dist - 1e-9 and route_is_feasible(candidate, capacity, depot):
                    best = candidate
                    best_dist = cand_dist
                    improved = True
        # loop again from scratch if any improvement was found this pass
    return best


def two_opt_improve_all_routes(routes, capacity, depot=DEPOT):
    return [two_opt_improve(r, capacity, depot) for r in routes]
