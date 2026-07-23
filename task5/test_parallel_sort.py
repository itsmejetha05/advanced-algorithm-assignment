"""
Correctness tests for Task 5: sequential, threaded, and multiprocess sort.
Must be run as a script (multiprocessing on some platforms requires
`if __name__ == "__main__":` guarding, which is why tests are invoked that way).
"""
import random

from parallel_sort import sequential_merge_sort, threaded_parallel_sort, multiprocess_parallel_sort


def test_sequential_matches_builtin():
    rng = random.Random(1)
    for trial in range(10):
        arr = [rng.randint(0, 1000) for _ in range(rng.randint(0, 200))]
        assert sequential_merge_sort(arr) == sorted(arr)
    print("Sequential merge sort matches builtin sorted(): OK")


def test_threaded_matches_builtin():
    rng = random.Random(2)
    for trial in range(10):
        n = rng.randint(1, 500)
        arr = [rng.randint(0, 10000) for _ in range(n)]
        for num_threads in [1, 2, 4, 8]:
            result = threaded_parallel_sort(arr, num_threads=num_threads)
            assert result == sorted(arr), \
                f"threaded sort mismatch at n={n}, threads={num_threads}"
    print("Threaded parallel sort matches builtin sorted() (1/2/4/8 threads): OK")


def test_multiprocess_matches_builtin():
    rng = random.Random(3)
    for trial in range(5):
        n = rng.randint(1, 500)
        arr = [rng.randint(0, 10000) for _ in range(n)]
        for num_processes in [1, 2, 4]:
            result = multiprocess_parallel_sort(arr, num_processes=num_processes)
            assert result == sorted(arr), \
                f"multiprocess sort mismatch at n={n}, processes={num_processes}"
    print("Multiprocess parallel sort matches builtin sorted() (1/2/4 processes): OK")


def test_sort_with_key_function():
    # sort tuples by second element, to confirm the `key` parameter works
    # correctly through chunking + merging (not just plain numbers)
    data = [(i, -i) for i in range(50)]
    random.Random(4).shuffle(data)
    expected = sorted(data, key=lambda t: t[1])
    assert sequential_merge_sort(data, key=lambda t: t[1]) == expected
    assert threaded_parallel_sort(data, key=lambda t: t[1], num_threads=4) == expected
    print("Sort with custom key function: OK")


def test_empty_and_single_element():
    assert sequential_merge_sort([]) == []
    assert sequential_merge_sort([5]) == [5]
    assert threaded_parallel_sort([], num_threads=4) == []
    assert threaded_parallel_sort([5], num_threads=4) == [5]
    print("Empty and single-element edge cases: OK")


def test_threaded_repeated_runs_are_consistent():
    """Run the same input through the threaded sort many times to catch
    any intermittent race condition that a single run might miss."""
    rng = random.Random(5)
    arr = [rng.randint(0, 1000) for _ in range(300)]
    expected = sorted(arr)
    for _ in range(30):
        assert threaded_parallel_sort(arr, num_threads=8) == expected
    print("Threaded sort: 30 repeated runs, all consistent (no race conditions detected): OK")


if __name__ == "__main__":
    test_sequential_matches_builtin()
    test_threaded_matches_builtin()
    test_multiprocess_matches_builtin()
    test_sort_with_key_function()
    test_empty_and_single_element()
    test_threaded_repeated_runs_are_consistent()
    print("\nAll Task 5 correctness tests passed.")
