"""
Task 5: Concurrent Programming - Parallel Merge Sort.

Parallelises the merge-sort algorithm (sorting numeric city attributes, as
used in Task 1) across multiple workers. Two concurrency models are
implemented deliberately, to allow an honest comparison:

1. threaded_parallel_sort() - uses Python's `threading` module.
   Critical section: each worker thread writes its locally-sorted chunk
   into a SHARED results list and increments a shared completed-count.
   This is protected by a threading.Condition (which wraps a Lock): the
   lock prevents two threads writing/incrementing at the same time (a
   genuine race condition otherwise - list writes and counter increments
   are not atomic compound operations), and the condition variable lets
   the main thread block efficiently until all workers signal completion,
   instead of busy-polling or relying only on Thread.join().

2. multiprocess_parallel_sort() - uses Python's `multiprocessing` module.
   Each worker is a separate OS process with its own Python interpreter
   and its own GIL, so this achieves GENUINE parallel CPU execution across
   cores - unlike the threaded version (see note below).

Why both versions are implemented (the GIL)
---------------------------------------------
Python's Global Interpreter Lock (GIL) allows only one thread to execute
Python bytecode at a time, even on a multi-core machine. This means
threading.Thread does NOT provide real parallelism for CPU-bound work like
sorting - the threaded version below is included specifically to
demonstrate correct synchronisation primitives (as required), but its
benchmark results are expected to show little or no speedup, and are
compared directly against multiprocessing (which sidesteps the GIL by
using separate processes) to make this limitation concrete rather than
theoretical.
"""
import threading
import multiprocessing


def _identity(x):
    """Default sort key. Must be a module-level (picklable) function, not
    a lambda - multiprocessing sends functions to worker processes via
    pickle, and lambdas cannot be pickled. This is a genuine practical
    constraint of process-based parallelism that thread-based parallelism
    does not have (threads share memory directly, so any callable works)."""
    return x


def sequential_merge_sort(arr, key=_identity):
    """Standard recursive merge sort. O(n log n) time, O(n) space."""
    if len(arr) <= 1:
        return list(arr)
    mid = len(arr) // 2
    left = sequential_merge_sort(arr[:mid], key)
    right = sequential_merge_sort(arr[mid:], key)
    return _merge(left, right, key)


def _merge(left, right, key):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def _merge_k_sorted_chunks(chunks, key):
    """K-way merge of already-sorted chunks into one fully sorted list."""
    import heapq
    heap = []
    iterators = [iter(c) for c in chunks]
    for ci, it in enumerate(iterators):
        try:
            val = next(it)
            heapq.heappush(heap, (key(val), ci, val))
        except StopIteration:
            pass

    result = []
    while heap:
        _, ci, val = heapq.heappop(heap)
        result.append(val)
        try:
            nxt = next(iterators[ci])
            heapq.heappush(heap, (key(nxt), ci, nxt))
        except StopIteration:
            pass
    return result


def _chunk(arr, n_chunks):
    chunk_size = (len(arr) + n_chunks - 1) // n_chunks
    return [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]


def threaded_parallel_sort(arr, key=_identity, num_threads=4):
    """
    Splits arr into num_threads chunks, sorts each chunk in its own thread,
    then merges the sorted chunks. See module docstring for the
    synchronisation strategy (Condition variable guarding shared results).
    """
    if not arr:
        return []
    chunks = _chunk(arr, num_threads)
    n = len(chunks)
    results = [None] * n
    completed = [0]
    condition = threading.Condition()

    def worker(index, chunk):
        sorted_chunk = sequential_merge_sort(chunk, key)
        # --- critical section: shared `results` list and `completed`
        # counter are mutated here, so this must be lock-protected ---
        with condition:
            results[index] = sorted_chunk
            completed[0] += 1
            condition.notify_all()

    threads = [threading.Thread(target=worker, args=(i, c)) for i, c in enumerate(chunks)]
    for t in threads:
        t.start()

    with condition:
        while completed[0] < n:
            condition.wait()

    for t in threads:
        t.join()

    return _merge_k_sorted_chunks(results, key)


def _sort_chunk_for_multiprocessing(args):
    chunk, key = args
    return sequential_merge_sort(chunk, key)


def multiprocess_parallel_sort(arr, key=_identity, num_processes=4):
    """
    Same divide-and-conquer strategy as threaded_parallel_sort, but using
    separate OS processes (multiprocessing.Pool) instead of threads. Each
    process has its own interpreter/GIL, so this achieves real parallel
    CPU execution - at the cost of inter-process communication overhead
    (each chunk and its sorted result must be pickled and sent between
    processes, unlike threads which share memory directly).
    """
    if not arr:
        return []
    chunks = _chunk(arr, num_processes)
    with multiprocessing.Pool(processes=num_processes) as pool:
        sorted_chunks = pool.map(_sort_chunk_for_multiprocessing, [(c, key) for c in chunks])
    return _merge_k_sorted_chunks(sorted_chunks, key)
