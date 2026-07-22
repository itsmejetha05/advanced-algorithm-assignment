"""
Task 3 - Greedy: Minimum Number of Platforms.

Problem: given train arrival and departure times, find the minimum number
of platforms needed so no two trains occupying the station at the same
time share a platform.

Greedy choice
--------------
Sort arrivals and departures independently. Sweep through events in time
order using two pointers: whenever the next event (by time) is an arrival,
increment the platform count in use; whenever it's a departure, decrement
it. Track the running maximum - that maximum IS the answer.

Why the greedy choice is optimal (not just "usually good")
-------------------------------------------------------------
This is a classic EXACT greedy, not a heuristic that can go wrong. At any
instant in time, the number of platforms required is exactly the number of
trains simultaneously present at the station. The sweep computes the true
maximum simultaneous overlap across all instants. Since no schedule can
ever need FEWER platforms than the busiest instant demands, and the sweep
never invents an overlap that doesn't exist, the computed maximum is both
a lower bound (necessary) and achievable (sufficient - just assign platforms
in arrival order, reusing any platform that has freed up). Hence greedy = optimal
here, unlike the weighted interval-scheduling variant where greedy-by-some-rule
can be provably suboptimal.

Complexity: sorting arrivals and departures is O(n log n); the sweep itself
is O(n). Total: O(n log n) time, O(n) space (for the sorted copies).
"""


def min_platforms_greedy(arrivals, departures):
    """
    Returns (min_platforms_needed, timeline) where timeline is a list of
    (time, event_type, running_count) tuples for visualisation.
    O(n log n) time.
    """
    arr = sorted(arrivals)
    dep = sorted(departures)
    n = len(arr)

    i = j = 0
    platforms = 0
    max_platforms = 0
    timeline = []

    while i < n and j < n:
        if arr[i] <= dep[j]:
            platforms += 1
            max_platforms = max(max_platforms, platforms)
            timeline.append((arr[i], "arrival", platforms))
            i += 1
        else:
            platforms -= 1
            timeline.append((dep[j], "departure", platforms))
            j += 1

    return max_platforms, timeline


def min_platforms_brute_force(arrivals, departures):
    """
    Exact O(n^2) baseline: for every train i, count how many trains
    (including itself) are already at the station at time arr[i], i.e.
    arr[j] <= arr[i] <= dep[j]. The answer is the maximum such count across
    all i. Used to verify the greedy result and to compare runtime scaling.

    NOTE on boundary convention: a train arriving at the EXACT same instant
    another departs is treated as needing a SEPARATE platform (inclusive
    "<=" on both sides) - matching the convention used by min_platforms_greedy
    (which processes an arrival before a same-time departure). This must be
    kept consistent between both implementations, or they disagree on
    boundary-touching schedules even though both are "correct" under their
    own (different) convention.
    """
    n = len(arrivals)
    max_overlap = 0
    for i in range(n):
        count = 0
        for j in range(n):
            if arrivals[j] <= arrivals[i] <= departures[j]:
                count += 1
        max_overlap = max(max_overlap, count)
    return max_overlap
