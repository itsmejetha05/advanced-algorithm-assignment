"""
Sanity/correctness tests for BST, AVL, MinHeap, HashTable.
Not a formal pytest suite (kept dependency-free) - run directly.
"""
import random
from data_structures import BinarySearchTree, AVLTree, MinHeap, HashTable


def test_bst():
    t = BinarySearchTree()
    keys = ["Paris", "Berlin", "Zurich", "Amman", "Tokyo", "Dublin"]
    for k in keys:
        t.insert(k, {"name": k})
    assert len(t) == len(keys)
    for k in keys:
        assert t.search(k)["name"] == k
    assert t.search("Nowhere") is None
    assert t.delete("Berlin") is True
    assert t.search("Berlin") is None
    assert len(t) == len(keys) - 1
    print("BST: OK")


def test_avl_balanced():
    t = AVLTree()
    n = 2000
    keys = [f"city_{i:05d}" for i in range(n)]
    for k in keys:  # sorted/ascending insert -> worst case for plain BST
        t.insert(k, k)
    import math
    assert t.height() <= 1.45 * math.log2(n + 2), "AVL height exceeded 1.45*log2(n) bound"
    for k in random.sample(keys, 50):
        assert t.search(k) == k
    for k in random.sample(keys, 50):
        assert t.delete(k) is True
    assert len(t) == n - 50
    print(f"AVL: OK (height={t.height()} for n={n}, well within O(log n) bound)")


def test_minheap():
    h = MinHeap()
    values = [50, 10, 40, 5, 100, 1, 77]
    for v in values:
        h.insert(v, v)
    out = []
    while len(h):
        out.append(h.extract_min()[0])
    assert out == sorted(values), "heap did not produce sorted order"
    print("MinHeap: OK (heap-sort order verified)")


def test_hashtable():
    ht = HashTable(capacity=4)  # small capacity forces resizes
    for i in range(500):
        ht.insert(f"key_{i}", i)
    assert len(ht) == 500
    for i in range(500):
        assert ht.search(f"key_{i}") == i
    assert ht.load_factor() <= 0.75
    for i in range(0, 500, 2):
        ht.delete(f"key_{i}")
    assert len(ht) == 250
    for i in range(1, 500, 2):
        assert ht.search(f"key_{i}") == i
    for i in range(0, 500, 2):
        assert ht.search(f"key_{i}") is None
    print("HashTable: OK (insert/search/delete/resize verified)")


if __name__ == "__main__":
    test_bst()
    test_avl_balanced()
    test_minheap()
    test_hashtable()
    print("\nAll correctness tests passed.")
