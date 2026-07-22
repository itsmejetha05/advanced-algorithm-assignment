"""
Task 3 - Dynamic Programming: Weighted Job Scheduling.

Problem: given n jobs, each with a start time, end time, and profit, select
a subset of NON-OVERLAPPING jobs that maximises total profit.

Subproblem definition
----------------------
Sort jobs by end time. Let dp[i] = the maximum profit achievable using only
the first i jobs (in end-time order). For job i (1-indexed, sorted), let
p(i) = the index of the latest job that finishes at or before job i starts
(found via binary search over the sorted end times - since jobs are sorted
by end time, this is a valid binary search target).

Recurrence relation
--------------------
    dp[0] = 0
    dp[i] = max( dp[i-1],                      # skip job i
                 profit[i] + dp[p(i) + 1] )     # take job i
    answer = dp[n]

Bottom-up strategy: build dp[0..n] iteratively left to right - each dp[i]
only depends on strictly smaller indices, so a single forward pass over a
1D table suffices (no recursion needed).

Complexity: sorting is O(n log n); each of the n jobs does one binary
search over up to n elements, O(log n); total = O(n log n) time, O(n) space
for the dp table. The hidden constant: binary search's O(log n) has extra
constant overhead (branch mispredictions, non-local memory access) versus
a simple linear scan - for very small n a linear scan of recent jobs can
be faster in practice despite worse asymptotic complexity.
"""


class Job:
    __slots__ = ("start", "end", "profit")

    def __init__(self, start, end, profit):
        self.start = start
        self.end = end
        self.profit = profit

    def __repr__(self):
        return f"Job({self.start},{self.end},{self.profit})"


def _latest_non_conflicting(jobs, i):
    """Binary search for the rightmost job index j < i with jobs[j].end <= jobs[i].start.
    Returns -1 if none exists. jobs must be sorted by end time."""
    lo, hi = 0, i - 1
    result = -1
    target = jobs[i].start
    while lo <= hi:
        mid = (lo + hi) // 2
        if jobs[mid].end <= target:
            result = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return result


def weighted_job_scheduling_bottom_up(jobs):
    """
    Returns (max_profit, selected_jobs, dp_table).
    O(n log n) time, O(n) space.
    """
    jobs = sorted(jobs, key=lambda j: j.end)
    n = len(jobs)
    dp = [0] * (n + 1)
    choice = [False] * (n + 1)  # did we take job i in the optimal solution up to i?

    for i in range(1, n + 1):
        job = jobs[i - 1]
        incl_profit = job.profit
        l = _latest_non_conflicting(jobs, i - 1)
        if l != -1:
            incl_profit += dp[l + 1]

        if incl_profit > dp[i - 1]:
            dp[i] = incl_profit
            choice[i] = True
        else:
            dp[i] = dp[i - 1]
            choice[i] = False

    # reconstruct selected jobs by walking the table backwards
    selected = []
    i = n
    while i > 0:
        if choice[i]:
            job = jobs[i - 1]
            selected.append(job)
            l = _latest_non_conflicting(jobs, i - 1)
            i = l + 1
        else:
            i -= 1
    selected.reverse()

    return dp[n], selected, dp


def weighted_job_scheduling_top_down(jobs):
    """
    Memoised (top-down) version of the same recurrence, for comparison.
    Same O(n log n) time complexity but uses O(n) extra call-stack depth
    (recursion) instead of the bottom-up version's flat iterative loop.
    """
    jobs = sorted(jobs, key=lambda j: j.end)
    n = len(jobs)
    memo = {}

    def solve(i):
        if i == 0:
            return 0
        if i in memo:
            return memo[i]
        job = jobs[i - 1]
        incl_profit = job.profit
        l = _latest_non_conflicting(jobs, i - 1)
        if l != -1:
            incl_profit += solve(l + 1)
        result = max(solve(i - 1), incl_profit)
        memo[i] = result
        return result

    return solve(n)


def brute_force_job_scheduling(jobs):
    """
    Exhaustive O(2^n) baseline for correctness-checking on small inputs only
    (used in tests, NOT in the benchmark - infeasible beyond ~20 jobs).

    IMPORTANT: jobs must be processed in a fixed chronological order (here,
    sorted by start time) for the `last_end` tracking to correctly represent
    the rightmost boundary of already-selected jobs. Processing jobs in
    arbitrary/input order breaks this invariant and silently produces wrong
    (too-low) answers, since a later-processed job with an earlier start
    time could be wrongly rejected against a `last_end` that doesn't
    actually represent a chronologically-earlier selection.
    """
    jobs = sorted(jobs, key=lambda j: j.start)
    best = [0]

    def backtrack(i, last_end, profit_so_far):
        if i == len(jobs):
            best[0] = max(best[0], profit_so_far)
            return
        # skip job i
        backtrack(i + 1, last_end, profit_so_far)
        # take job i if it doesn't conflict with last taken job
        if jobs[i].start >= last_end:
            backtrack(i + 1, jobs[i].end, profit_so_far + jobs[i].profit)

    backtrack(0, float("-inf"), 0)
    return best[0]
