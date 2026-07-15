"""
Empirical benchmarking for Task 1.

For each dataset size n in {100, 1000, 10000} we measure wall-clock time
(using time.perf_counter) for:

    - insertion (BST built from random order, BST built from sorted order
      to show the worst case, AVL, Hash Table, Min-Heap)
    - search    (BST, AVL, Hash Table)
    - deletion  (BST, AVL, Hash Table)
    - extract_min (Min-Heap)

Results are written to results/benchmark_results.csv and
results/tree_heights.csv for later analysis / plotting.
"""

import csv
import random
import time

from data_loader import load_cities, sample_cities
from data_structures import BinarySearchTree, AVLTree, MinHeap, HashTable

SIZES = [100, 1000, 10000]
SEARCH_SAMPLE = 200      # number of search probes per structure/size
RESULTS_DIR = r"D:\advanced assignment\Task1\results"


def time_it(func):
    start = time.perf_counter()
    func()
    return time.perf_counter() - start


def build_and_time_insertions(cities):
    """Returns dict of structure -> (populated structure, insert_time_s)."""
    results = {}

    # --- BST, random insertion order (typical/average case) ---
    bst = BinarySearchTree()
    t = time_it(lambda: [bst.insert(c["name"], c) for c in cities])
    results["BST_random"] = (bst, t)

    # --- BST, sorted insertion order (adversarial / worst case) ---
    sorted_cities = sorted(cities, key=lambda c: c["name"])
    bst_sorted = BinarySearchTree()
    t = time_it(lambda: [bst_sorted.insert(c["name"], c) for c in sorted_cities])
    results["BST_sorted"] = (bst_sorted, t)

    # --- AVL Tree ---
    avl = AVLTree()
    t = time_it(lambda: [avl.insert(c["name"], c) for c in cities])
    results["AVL"] = (avl, t)

    # --- Hash Table ---
    ht = HashTable()
    t = time_it(lambda: [ht.insert(c["name"], c) for c in cities])
    results["Hash"] = (ht, t)

    # --- Min-Heap (priority = distance from hub) ---
    heap = MinHeap()
    t = time_it(lambda: [heap.insert(c["distance"], c) for c in cities])
    results["MinHeap"] = (heap, t)

    return results


def time_search(structure, keys):
    start = time.perf_counter()
    for k in keys:
        structure.search(k)
    return time.perf_counter() - start


def time_delete(structure, keys):
    start = time.perf_counter()
    for k in keys:
        structure.delete(k)
    return time.perf_counter() - start


def time_extract_all(heap):
    start = time.perf_counter()
    n = len(heap)
    for _ in range(n):
        heap.extract_min()
    return time.perf_counter() - start


def run():
    print("Loading full dataset...")
    all_cities = load_cities()
    print(f"Total unique cities available: {len(all_cities)}")

    timing_rows = []
    height_rows = []

    for n in SIZES:
        print(f"\n=== n = {n} ===")
        sample = sample_cities(all_cities, n, seed=42)
        rng = random.Random(7)
        search_keys = [c["name"] for c in rng.sample(sample, min(SEARCH_SAMPLE, n))]
        delete_keys_bst = [c["name"] for c in rng.sample(sample, min(SEARCH_SAMPLE, n))]

        built = build_and_time_insertions(sample)

        # ---- Insertion timings ----
        for struct_name, (structure, insert_time) in built.items():
            timing_rows.append({
                "structure": struct_name, "n": n, "operation": "insert",
                "total_time_s": insert_time,
                "avg_time_ms": (insert_time / n) * 1000,
            })
            print(f"  Insert {struct_name:12s}: total={insert_time:.6f}s "
                  f"avg={(insert_time/n)*1000:.6f} ms/op")

        # ---- Tree heights (structural comparison, not raw timing) ----
        import math
        bst_rand_h = built["BST_random"][0].height()
        bst_sorted_h = built["BST_sorted"][0].height()
        avl_h = built["AVL"][0].height()
        ideal_h = math.log2(n + 1)
        height_rows.append({
            "n": n, "BST_random_height": bst_rand_h,
            "BST_sorted_height": bst_sorted_h, "AVL_height": avl_h,
            "ideal_log2n": ideal_h,
        })
        print(f"  Heights -> BST(random)={bst_rand_h}  BST(sorted)={bst_sorted_h}  "
              f"AVL={avl_h}  log2(n)~={ideal_h:.1f}")

        # ---- Search timings (BST_random, AVL, Hash) ----
        for struct_name in ["BST_random", "AVL", "Hash"]:
            structure = built[struct_name][0]
            t = time_search(structure, search_keys)
            timing_rows.append({
                "structure": struct_name, "n": n, "operation": "search",
                "total_time_s": t,
                "avg_time_ms": (t / len(search_keys)) * 1000,
            })
            print(f"  Search {struct_name:12s}: total={t:.6f}s "
                  f"avg={(t/len(search_keys))*1000:.6f} ms/op")

        # ---- Delete timings (BST_random, AVL, Hash) - delete ALL n items ----
        for struct_name in ["BST_random", "AVL", "Hash"]:
            structure = built[struct_name][0]
            keys_to_delete = [c["name"] for c in sample]
            t = time_delete(structure, keys_to_delete)
            timing_rows.append({
                "structure": struct_name, "n": n, "operation": "delete",
                "total_time_s": t,
                "avg_time_ms": (t / n) * 1000,
            })
            print(f"  Delete {struct_name:12s}: total={t:.6f}s "
                  f"avg={(t/n)*1000:.6f} ms/op")

        # ---- Min-Heap extract_min (all n) ----
        heap = built["MinHeap"][0]
        t = time_extract_all(heap)
        timing_rows.append({
            "structure": "MinHeap", "n": n, "operation": "extract_min",
            "total_time_s": t,
            "avg_time_ms": (t / n) * 1000,
        })
        print(f"  ExtractMin MinHeap  : total={t:.6f}s avg={(t/n)*1000:.6f} ms/op")

    # ---- write CSVs ----
    with open(f"{RESULTS_DIR}/benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["structure", "n", "operation",
                                                "total_time_s", "avg_time_ms"])
        writer.writeheader()
        writer.writerows(timing_rows)

    with open(f"{RESULTS_DIR}/tree_heights.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "BST_random_height",
                                                "BST_sorted_height", "AVL_height",
                                                "ideal_log2n"])
        writer.writeheader()
        writer.writerows(height_rows)

    print("\nSaved results/benchmark_results.csv and results/tree_heights.csv")


if __name__ == "__main__":
    run()
