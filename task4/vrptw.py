"""
Task 4: Vehicle Routing Problem with Time Windows (VRPTW).

Problem: a fleet of vehicles, each with limited capacity, must deliver to a
set of customers, each with a demand, a time window [earliest, latest], and
a service duration. All routes start and end at a single depot. Minimise
total distance travelled across all vehicles subject to:
    - vehicle capacity never exceeded
    - every customer served within its time window
    - every customer served exactly once

Why VRPTW is NP-Hard
---------------------
VRPTW generalises the Travelling Salesman Problem (TSP): if there is exactly
one vehicle with infinite capacity and every time window is [0, infinity),
VRPTW reduces EXACTLY to TSP (find the shortest route visiting all
customers once and returning to the depot). Since TSP is a well-known
NP-Hard problem (no polynomial-time exact algorithm is known, and it is
NP-Complete in decision form), and VRPTW contains TSP as a special case,
VRPTW is at least as hard as TSP - i.e. NP-Hard. Adding capacity
constraints and time windows only restricts the feasible solution space
further; it does not make the underlying combinatorial search easier.
"""
import math
import random


class Customer:
    __slots__ = ("id", "x", "y", "demand", "earliest", "latest", "service_time")

    def __init__(self, id, x, y, demand, earliest, latest, service_time=10):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.earliest = earliest
        self.latest = latest
        self.service_time = service_time

    def __repr__(self):
        return f"C{self.id}(d={self.demand},tw=[{self.earliest},{self.latest}])"


DEPOT = Customer(id=0, x=50, y=50, demand=0, earliest=0, latest=10_000, service_time=0)


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def generate_instance(num_customers=50, vehicle_capacity=100, seed=42,
                       area=100, max_demand=25, tw_width_range=(20, 60)):
    """
    Generates a synthetic VRPTW instance: customers scattered randomly in a
    square area around the depot, with random demand and a random time
    window of width tw_width_range. Distances double as travel times (unit
    speed) for time-window checks.

    IMPORTANT: `earliest` is generated as an offset AFTER the customer's
    direct travel time from the depot (not independently of it), so every
    customer's window is guaranteed individually reachable in isolation
    (arriving straight from the depot never arrives after `latest`). Without
    this, a customer could be randomly placed far from the depot but given
    an early time window that no vehicle could ever reach in time, making
    the instance infeasible regardless of routing algorithm - a data
    generation bug, not a heuristic failure.
    """
    rng = random.Random(seed)
    customers = []
    for i in range(1, num_customers + 1):
        x = rng.uniform(0, area)
        y = rng.uniform(0, area)
        demand = rng.randint(5, max_demand)
        tw_width = rng.randint(*tw_width_range)
        dist_from_depot = math.hypot(x - DEPOT.x, y - DEPOT.y)
        earliest = dist_from_depot + rng.uniform(0, area)  # always reachable directly
        latest = earliest + tw_width
        customers.append(Customer(i, x, y, demand, earliest, latest))
    return customers, vehicle_capacity


def route_distance(route, depot=DEPOT):
    """route: list of Customer, depot-to-depot. Returns total distance."""
    if not route:
        return 0.0
    total = distance(depot, route[0])
    for a, b in zip(route, route[1:]):
        total += distance(a, b)
    total += distance(route[-1], depot)
    return total


def route_is_feasible(route, capacity, depot=DEPOT):
    """
    Checks capacity and time-window feasibility for a single route in the
    given visiting order. Arrival time = departure time from previous stop
    + travel time; if arrival is before a customer's earliest window, the
    vehicle waits (no penalty, just idles); arriving after `latest` is
    infeasible.
    """
    total_demand = sum(c.demand for c in route)
    if total_demand > capacity:
        return False

    t = 0.0
    prev = depot
    for c in route:
        t += distance(prev, c)
        if t > c.latest:
            return False
        t = max(t, c.earliest)  # wait if early
        t += c.service_time
        prev = c
    return True


def solution_total_distance(routes, depot=DEPOT):
    return sum(route_distance(r, depot) for r in routes)


def solution_is_feasible(routes, capacity, all_customers, depot=DEPOT):
    """Every customer visited exactly once, across feasible routes."""
    visited = []
    for r in routes:
        if not route_is_feasible(r, capacity, depot):
            return False
        visited.extend(c.id for c in r)
    if sorted(visited) != sorted(c.id for c in all_customers):
        return False
    return True
