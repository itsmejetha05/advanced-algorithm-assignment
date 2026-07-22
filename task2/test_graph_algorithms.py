"""
Correctness tests for Graph, Dijkstra, Prim, Bellman-Ford - using small,
hand-verifiable graphs (not the city dataset) so expected results are known.
"""
from graph_structures import Graph
from graph_algorithms import dijkstra, reconstruct_path, prim_mst, bellman_ford


def test_graph_adjacency_list():
    g = Graph(directed=True)
    g.add_edge("A", "B", 5)
    g.add_edge("A", "C", 2)
    g.add_edge("C", "B", 1)
    assert g.num_nodes() == 3
    assert g.num_edges() == 3
    assert ("B", 5) in g.neighbors("A")
    assert ("B", 1) in g.neighbors("C")
    print("Graph adjacency list: OK")


def test_dijkstra_known_graph():
    # classic textbook example with a known shortest path
    g = Graph(directed=True)
    edges = [
        ("A", "B", 4), ("A", "C", 1),
        ("C", "B", 2), ("B", "D", 1),
        ("C", "D", 5),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)

    dist, prev = dijkstra(g, "A")
    # A->C->B->D = 1+2+1 = 4, cheaper than A->B->D = 4+1 = 5
    assert dist["D"] == 4, f"expected 4, got {dist['D']}"
    assert dist["B"] == 3, f"expected 3 (A->C->B), got {dist['B']}"
    path = reconstruct_path(prev, "A", "D")
    assert path == ["A", "C", "B", "D"], f"unexpected path: {path}"
    print("Dijkstra (known graph): OK - shortest A->D =", dist["D"], "via", path)


def test_prim_known_graph():
    # undirected graph, known minimum spanning tree weight
    g = Graph(directed=False)
    edges = [
        ("A", "B", 1), ("A", "C", 4),
        ("B", "C", 2), ("B", "D", 6),
        ("C", "D", 3),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)
    # MST: A-B(1), B-C(2), C-D(3) = total 6
    mst_edges, total = prim_mst(g, start="A")
    assert total == 6, f"expected MST weight 6, got {total}"
    assert len(mst_edges) == 3
    print("Prim MST (known graph): OK - total weight =", total)


def test_bellman_ford_negative_weights():
    g = Graph(directed=True)
    edges = [
        ("A", "B", 4), ("A", "C", 1),
        ("C", "B", -3),  # negative edge, still no negative cycle
        ("B", "D", 2),
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)
    dist, prev, has_cycle = bellman_ford(g, "A")
    assert has_cycle is False
    # A->C->B = 1 + (-3) = -2, cheaper than A->B = 4
    assert dist["B"] == -2, f"expected -2, got {dist['B']}"
    assert dist["D"] == 0, f"expected 0 (A->C->B->D = -2+2=0), got {dist['D']}"
    print("Bellman-Ford (negative weight, no cycle): OK")


def test_bellman_ford_negative_cycle():
    g = Graph(directed=True)
    edges = [
        ("A", "B", 1),
        ("B", "C", -1),
        ("C", "B", -1),  # B->C->B is a negative cycle (-2 total)
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)
    dist, prev, has_cycle = bellman_ford(g, "A")
    assert has_cycle is True, "expected negative cycle to be detected"
    print("Bellman-Ford (negative cycle detection): OK")


def test_defensive_checks():
    g = Graph(directed=True)
    g.add_edge("A", "B", 5)

    try:
        dijkstra(g, "Z")
        assert False, "expected ValueError for missing source"
    except ValueError:
        pass

    try:
        bellman_ford(g, "Z")
        assert False, "expected ValueError for missing source"
    except ValueError:
        pass

    # disconnected undirected graph should raise RuntimeError in Prim
    g2 = Graph(directed=False)
    g2.add_edge("A", "B", 1)
    g2.add_node("C")  # isolated node, no edges -> graph is disconnected
    try:
        prim_mst(g2, start="A")
        assert False, "expected RuntimeError for disconnected graph"
    except RuntimeError:
        pass

    print("Defensive checks (missing node / disconnected graph): OK")


if __name__ == "__main__":
    test_graph_adjacency_list()
    test_dijkstra_known_graph()
    test_prim_known_graph()
    test_bellman_ford_negative_weights()
    test_bellman_ford_negative_cycle()
    test_defensive_checks()
    print("\nAll Task 2 correctness tests passed.")