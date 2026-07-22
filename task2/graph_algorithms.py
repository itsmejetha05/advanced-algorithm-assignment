"""
Task 2: Graph Algorithms - Dijkstra (shortest path), Prim (MST), Bellman-Ford
(shortest path with negative weights + negative-cycle detection).

A binary min-heap (built from scratch, same design as Task 1's MinHeap) is
used as the priority queue for Dijkstra and Prim, since both need repeated
extract-minimum access - exactly the operation a heap is designed for.
"""

import math


class _MinHeap:
    """Lightweight binary min-heap for (priority, node) pairs, with lazy
    deletion to emulate a decrease-key operation (we simply push an updated,
    lower-priority entry and skip stale ones when popped)."""

    def __init__(self):
        self._heap = []

    def __len__(self):
        return len(self._heap)

    def push(self, priority, node):
        self._heap.append((priority, node))
        self._sift_up(len(self._heap) - 1)

    def pop(self):
        root = self._heap[0]
        last = self._heap.pop()
        if self._heap:
            self._heap[0] = last
            self._sift_down(0)
        return root

    def _sift_up(self, i):
        h = self._heap
        while i > 0:
            parent = (i - 1) // 2
            if h[i][0] < h[parent][0]:
                h[i], h[parent] = h[parent], h[i]
                i = parent
            else:
                break

    def _sift_down(self, i):
        h = self._heap
        n = len(h)
        while True:
            l, r = 2 * i + 1, 2 * i + 2
            smallest = i
            if l < n and h[l][0] < h[smallest][0]:
                smallest = l
            if r < n and h[r][0] < h[smallest][0]:
                smallest = r
            if smallest == i:
                break
            h[i], h[smallest] = h[smallest], h[i]
            i = smallest


def dijkstra(graph, source, record_steps=False):
    """
    Single-source shortest path for graphs with non-negative edge weights.
    Returns (dist, prev) dicts. If record_steps=True, also returns a list of
    (node_just_finalised, dist) tuples in the order nodes were finalised -
    used to visualise the shortest-path tree growing outward from source.
    """
    if source not in graph.adj:
        raise ValueError(f"Source node '{source}' does not exist in the graph.")

    dist = {n: math.inf for n in graph.nodes()}
    prev = {n: None for n in graph.nodes()}
    dist[source] = 0
    visited = set()
    heap = _MinHeap()
    heap.push(0, source)
    steps = []

    while len(heap):
        d, u = heap.pop()
        if u in visited:
            continue
        visited.add(u)
        if record_steps:
            steps.append((u, d))
        for v, w in graph.neighbors(u):
            if v in visited:
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heap.push(nd, v)

    if record_steps:
        return dist, prev, steps
    return dist, prev


def reconstruct_path(prev, source, target):
    if prev.get(target) is None and target != source:
        return None  # unreachable
    path = [target]
    while path[-1] != source:
        p = prev[path[-1]]
        if p is None:
            return None
        path.append(p)
    return list(reversed(path))


def prim_mst(graph, start=None, record_steps=False):
    """
    Minimum Spanning Tree via Prim's algorithm. Requires an UNDIRECTED
    connected graph. Returns (mst_edges, total_weight[, steps]).

    Raises RuntimeError if the graph turns out to be disconnected (Prim's
    algorithm is only defined for connected graphs; build_undirected_version()
    already guards against this via ensure_connected(), but this check
    catches misuse if that step is skipped).
    """
    nodes = graph.nodes()
    if not nodes:
        return [], 0
    start = start or nodes[0]
    if start not in graph.adj:
        raise ValueError(f"Start node '{start}' does not exist in the graph.")

    visited = {start}
    mst_edges = []
    total_weight = 0
    heap = _MinHeap()
    steps = []

    def push_frontier(u):
        for v, w in graph.neighbors(u):
            if v not in visited:
                heap.push(w, (u, v))

    push_frontier(start)
    if record_steps:
        steps.append((start, None, 0))

    while len(heap) and len(visited) < len(nodes):
        w, (u, v) = heap.pop()
        if v in visited:
            continue
        visited.add(v)
        mst_edges.append((u, v, w))
        total_weight += w
        if record_steps:
            steps.append((v, u, w))
        push_frontier(v)

    if len(visited) < len(nodes):
        raise RuntimeError(
            f"Prim's algorithm reached only {len(visited)}/{len(nodes)} nodes - "
            "the graph is disconnected. Build it with build_undirected_version() "
            "(connect=True) or call ensure_connected() first."
        )

    if record_steps:
        return mst_edges, total_weight, steps
    return mst_edges, total_weight


def bellman_ford(graph, source):
    """
    Single-source shortest path allowing negative edge weights.
    Returns (dist, prev, has_negative_cycle: bool).
    If a negative cycle is detected, dist/prev reflect the state after
    V-1 relaxations (i.e. before the cycle could distort them further) and
    has_negative_cycle is True.
    """
    nodes = graph.nodes()
    if source not in graph.adj:
        raise ValueError(f"Source node '{source}' does not exist in the graph.")

    dist = {n: math.inf for n in nodes}
    prev = {n: None for n in nodes}
    dist[source] = 0

    edges = []
    for u in graph.adj:
        for v, w in graph.adj[u]:
            edges.append((u, v, w))

    for _ in range(len(nodes) - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != math.inf and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                updated = True
        if not updated:
            break  # early exit - no more relaxations possible

    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != math.inf and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break

    return dist, prev, has_negative_cycle