"""
Task 2: Graph representation for a transportation network optimisation problem.

Represents the network as a weighted DIRECTED graph using an ADJACENCY LIST.

Why adjacency list over adjacency matrix?
------------------------------------------
A real transportation network built from k-nearest-neighbour city connections
is SPARSE: each city connects to only k (~6) of the other n cities, so the
number of edges E is O(k*n) = O(n), not O(n^2). For n = 1000 cities:
    - Adjacency matrix: O(n^2) = 1,000,000 cells, almost all zero/unused.
    - Adjacency list:    O(n + E) ~= O(n + 6n) = O(7n) = ~7,000 entries.
An adjacency list therefore uses far less memory and, crucially, iterating a
node's neighbours (the core operation in Dijkstra/Prim/Bellman-Ford) is O(deg(v))
instead of O(n) as it would be scanning a matrix row. The only thing a matrix
would be better at - O(1) "is there an edge (u,v)?" lookup - is not a
bottleneck operation for any of the three algorithms implemented in Task 2.
"""

import math
import random


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


class Graph:
    """Weighted directed graph stored as an adjacency list."""

    def __init__(self, directed=True):
        self.directed = directed
        self.adj = {}          # node -> list of (neighbour, weight)
        self.node_data = {}    # node -> metadata dict (name, lat, lng, ...)

    def add_node(self, node, **metadata):
        if node not in self.adj:
            self.adj[node] = []
        self.node_data[node] = metadata

    def add_edge(self, u, v, weight):
        self.add_node(u)
        self.add_node(v)
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))

    def neighbors(self, u):
        return self.adj.get(u, [])

    def nodes(self):
        return list(self.adj.keys())

    def edges(self):
        result = []
        seen = set()
        for u in self.adj:
            for v, w in self.adj[u]:
                if self.directed:
                    result.append((u, v, w))
                else:
                    key = tuple(sorted((u, v)))
                    if key not in seen:
                        seen.add(key)
                        result.append((u, v, w))
        return result

    def num_nodes(self):
        return len(self.adj)

    def num_edges(self):
        return len(self.edges())

    def density(self):
        n = self.num_nodes()
        max_edges = n * (n - 1) if self.directed else n * (n - 1) / 2
        return self.num_edges() / max_edges if max_edges else 0


def build_knn_graph(cities, k=6, directed=True, asymmetry=0.15, seed=1, connect=True):
    """
    Builds a weighted graph by connecting each city to its k nearest
    neighbours (by haversine distance). This models a realistic sparse
    transportation network from raw geographic point data.

    If directed=True, the reverse edge gets a randomly perturbed weight
    (+/- `asymmetry` fraction) to simulate real-world directional cost
    differences (e.g. prevailing wind/elevation/traffic), producing a
    genuinely directed graph rather than a symmetric one relabelled.

    If connect=True (default), any disconnected components left over from
    k-NN construction (e.g. geographically isolated city clusters) are
    bridged with the cheapest possible connecting edge - see
    ensure_connected() for why this matters.
    """
    rng = random.Random(seed)
    g = Graph(directed=directed)
    for c in cities:
        g.add_node(c["name"], lat=c["lat"], lng=c["lng"], population=c["population"])

    for c in cities:
        dists = []
        for other in cities:
            if other["name"] == c["name"]:
                continue
            d = haversine_km(c["lat"], c["lng"], other["lat"], other["lng"])
            dists.append((d, other["name"]))
        dists.sort()
        for d, name in dists[:k]:
            g.add_edge(c["name"], name, round(d, 2))
            if directed:
                perturb = 1 + rng.uniform(-asymmetry, asymmetry)
                g.add_edge(name, c["name"], round(d * perturb, 2))

    if connect:
        ensure_connected(g, cities)

    return g


def build_undirected_version(cities, k=6, seed=1, connect=True):
    """Undirected graph (required for Prim's MST) using the same k-NN
    connectivity, weight = plain haversine distance (symmetric).
    connect=True bridges disconnected components - see ensure_connected()."""
    g = Graph(directed=False)
    for c in cities:
        g.add_node(c["name"], lat=c["lat"], lng=c["lng"], population=c["population"])

    added = set()
    for c in cities:
        dists = []
        for other in cities:
            if other["name"] == c["name"]:
                continue
            d = haversine_km(c["lat"], c["lng"], other["lat"], other["lng"])
            dists.append((d, other["name"]))
        dists.sort()
        for d, name in dists[:k]:
            key = tuple(sorted((c["name"], name)))
            if key not in added:
                added.add(key)
                g.add_edge(c["name"], name, round(d, 2))

    if connect:
        ensure_connected(g, cities)

    return g


def find_connected_components(g):
    """Returns a list of sets, each set being the nodes of one connected
    component (treats the graph as undirected for this check, since
    connectivity for Prim's MST is about the undirected reachability)."""
    undirected_adj = {}
    for u in g.adj:
        undirected_adj.setdefault(u, set())
        for v, w in g.adj[u]:
            undirected_adj.setdefault(v, set())
            undirected_adj[u].add(v)
            undirected_adj[v].add(u)

    visited = set()
    components = []
    for start in undirected_adj:
        if start in visited:
            continue
        comp = {start}
        stack = [start]
        visited.add(start)
        while stack:
            u = stack.pop()
            for v in undirected_adj[u]:
                if v not in visited:
                    visited.add(v)
                    comp.add(v)
                    stack.append(v)
        components.append(comp)
    return components


def ensure_connected(g, cities):
    """
    k-NN graphs built from geographically scattered points can leave
    isolated clusters disconnected from the rest (e.g. an island city whose
    nearest neighbours are all each other). Since Prim's MST fundamentally
    requires a CONNECTED graph, this bridges any disconnected components by
    repeatedly adding the single cheapest (shortest-distance) edge between
    the largest component and the nearest other component, until only one
    component remains. Mutates and returns the same graph.
    """
    coord = {c["name"]: (c["lat"], c["lng"]) for c in cities}
    components = find_connected_components(g)

    while len(components) > 1:
        components.sort(key=len, reverse=True)
        main = components[0]
        best = None  # (dist, u_in_main, v_in_other, other_idx)
        for oi, other in enumerate(components[1:], start=1):
            for u in main:
                ulat, ulng = coord[u]
                for v in other:
                    vlat, vlng = coord[v]
                    d = haversine_km(ulat, ulng, vlat, vlng)
                    if best is None or d < best[0]:
                        best = (d, u, v, oi)
        d, u, v, oi = best
        g.add_edge(u, v, round(d, 2))
        if g.directed:
            g.add_edge(v, u, round(d, 2))
        merged = main | components[oi]
        components = [merged] + [c for i, c in enumerate(components) if i not in (0, oi)]

    return g


def haversine_km_public(lat1, lon1, lat2, lon2):
    return haversine_km(lat1, lon1, lat2, lon2)


def inject_negative_edges(g, num_edges=5, min_weight=-50, max_weight=-5, seed=2):
    """Returns a NEW graph (copy) with a few artificial negative-weight edges
    added, to demonstrate Bellman-Ford's ability to handle negative weights
    (representing e.g. a subsidised/discounted route)."""
    rng = random.Random(seed)
    g2 = Graph(directed=True)
    for n, meta in g.node_data.items():
        g2.add_node(n, **meta)
    for u in g.adj:
        for v, w in g.adj[u]:
            g2.add_edge(u, v, w)

    nodes = g2.nodes()
    added = 0
    attempts = 0
    while added < num_edges and attempts < num_edges * 20:
        attempts += 1
        u, v = rng.sample(nodes, 2)
        w = rng.uniform(min_weight, max_weight)
        g2.add_edge(u, v, round(w, 2))
        added += 1
    return g2


def inject_negative_cycle(g, cycle_length=3, seed=3):
    """Returns a NEW graph containing a deliberate negative-weight cycle
    among `cycle_length` existing nodes, for testing negative-cycle
    detection in Bellman-Ford."""
    rng = random.Random(seed)
    g2 = Graph(directed=True)
    for n, meta in g.node_data.items():
        g2.add_node(n, **meta)
    for u in g.adj:
        for v, w in g.adj[u]:
            g2.add_edge(u, v, w)

    nodes = rng.sample(g2.nodes(), cycle_length)
    # build a cycle nodes[0] -> nodes[1] -> ... -> nodes[0] with a very
    # negative total weight so it is unambiguously a negative cycle
    per_edge_weight = -100.0
    for i in range(cycle_length):
        u = nodes[i]
        v = nodes[(i + 1) % cycle_length]
        g2.add_edge(u, v, per_edge_weight)
    return g2, nodes
