"""
Correctness tests for Task 3: Weighted Job Scheduling (DP), Minimum
Platforms (Greedy), Knight's Tour (Backtracking).
"""
from job_scheduling import (Job, weighted_job_scheduling_bottom_up,
                             weighted_job_scheduling_top_down,
                             brute_force_job_scheduling)
from min_platforms import min_platforms_greedy, min_platforms_brute_force
from knights_tour import plain_knights_tour, warnsdorff_knights_tour, verify_tour


def test_job_scheduling_known_case():
    # classic textbook example: expected optimal profit = 250
    # (jobs (1,3,50) + (3,5,100) + (6,19,100) are NOT the standard set;
    # use a well known worked example instead)
    jobs = [
        Job(1, 3, 50), Job(3, 5, 20), Job(6, 19, 100),
        Job(2, 100, 200),
    ]
    profit, selected, dp = weighted_job_scheduling_bottom_up(jobs)
    # optimal: just job (2,100,200) alone beats combining smaller ones (50+20+100=170 < 200)
    assert profit == 200, f"expected 200, got {profit}"
    assert len(selected) == 1 and selected[0].profit == 200
    print("Weighted Job Scheduling (known case): OK - optimal profit =", profit)


def test_job_scheduling_bottom_up_matches_top_down_and_brute_force():
    import random
    rng = random.Random(11)
    for trial in range(20):
        n = rng.randint(1, 12)
        jobs = []
        for _ in range(n):
            s = rng.randint(0, 20)
            e = s + rng.randint(1, 10)
            p = rng.randint(1, 100)
            jobs.append(Job(s, e, p))
        bu_profit, _, _ = weighted_job_scheduling_bottom_up(jobs)
        td_profit = weighted_job_scheduling_top_down(jobs)
        bf_profit = brute_force_job_scheduling(jobs)
        assert bu_profit == td_profit == bf_profit, \
            f"mismatch trial {trial}: bottom_up={bu_profit} top_down={td_profit} brute_force={bf_profit}"
    print("Weighted Job Scheduling (20 random trials): bottom-up == top-down == brute-force - OK")


def test_job_scheduling_selected_jobs_non_overlapping():
    import random
    rng = random.Random(5)
    jobs = [Job(rng.randint(0, 30), 0, rng.randint(1, 50)) for _ in range(15)]
    jobs = [Job(j.start, j.start + rng.randint(1, 8), j.profit) for j in jobs]
    _, selected, _ = weighted_job_scheduling_bottom_up(jobs)
    selected_sorted = sorted(selected, key=lambda j: j.start)
    for a, b in zip(selected_sorted, selected_sorted[1:]):
        assert a.end <= b.start, f"selected jobs overlap: {a} and {b}"
    print("Weighted Job Scheduling: selected jobs are non-overlapping - OK")


def test_min_platforms_known_case():
    arrivals = [900, 940, 950, 1100, 1500, 1800]
    departures = [910, 1200, 1120, 1130, 1900, 2000]
    platforms, _ = min_platforms_greedy(arrivals, departures)
    assert platforms == 3, f"expected 3, got {platforms}"
    print("Minimum Platforms (known case): OK - needs", platforms, "platforms")


def test_min_platforms_matches_brute_force():
    import random
    rng = random.Random(3)
    for trial in range(20):
        n = rng.randint(1, 30)
        arrivals = [rng.randint(0, 100) for _ in range(n)]
        departures = [a + rng.randint(1, 20) for a in arrivals]
        greedy_result, _ = min_platforms_greedy(arrivals, departures)
        brute_result = min_platforms_brute_force(arrivals, departures)
        assert greedy_result == brute_result, \
            f"trial {trial}: greedy={greedy_result} brute_force={brute_result}"
    print("Minimum Platforms (20 random trials): greedy == brute-force - OK")


def test_knights_tour_small_board():
    # 5x5 board is known to have an open knight's tour
    path, calls = warnsdorff_knights_tour(5, start=(0, 0))
    assert verify_tour(path, 5), "Warnsdorff tour failed verification on 5x5"
    print(f"Knight's Tour (Warnsdorff, 5x5): OK - found in {calls} recursive calls")

    path2, calls2 = plain_knights_tour(5, start=(0, 0))
    assert verify_tour(path2, 5), "Plain backtracking tour failed verification on 5x5"
    print(f"Knight's Tour (plain backtracking, 5x5): OK - found in {calls2} recursive calls")


def test_knights_tour_3x3_impossible():
    # it is a well-known result that no knight's tour exists on a 3x3 board
    path, calls = warnsdorff_knights_tour(3, start=(0, 0), max_calls=10000)
    assert path is None, "3x3 should have NO valid knight's tour"
    print("Knight's Tour (3x3, known impossible): correctly returned no solution - OK")


if __name__ == "__main__":
    test_job_scheduling_known_case()
    test_job_scheduling_bottom_up_matches_top_down_and_brute_force()
    test_job_scheduling_selected_jobs_non_overlapping()
    test_min_platforms_known_case()
    test_min_platforms_matches_brute_force()
    test_knights_tour_small_board()
    test_knights_tour_3x3_impossible()
    print("\nAll Task 3 correctness tests passed.")
