"""
Task 3 - Backtracking: Knight's Tour.

Problem: find a sequence of knight moves on an n x n board that visits
every square exactly once (an open tour - does not need to return to start).

Two implementations are provided to demonstrate the effect of pruning:

1. plain_knights_tour(): tries the 8 knight moves in a FIXED order at each
   step, backtracking on dead ends. This is the naive exponential-worst-case
   approach - O(8^(n^2)) in the absolute worst case, since at each of the
   n^2 squares up to 8 branches could be explored before backtracking.

2. warnsdorff_knights_tour(): at each step, greedily tries the candidate
   move that has the FEWEST onward moves available (Warnsdorff's rule),
   still with backtracking as a fallback if a dead end is reached. This
   heuristic ordering is a pruning strategy: by visiting the most
   "constrained" square first, it avoids stranding hard-to-reach squares
   for later, which empirically eliminates nearly all backtracking for
   board sizes where plain backtracking is already infeasible.

Both implementations count the number of squares_tried (recursive calls
made) so the search-space reduction from pruning can be measured directly,
not just inferred from wall-clock time.
"""

MOVES = [(1, 2), (2, 1), (2, -1), (1, -2),
         (-1, -2), (-2, -1), (-2, 1), (-1, 2)]


def _in_bounds(x, y, n):
    return 0 <= x < n and 0 <= y < n


def plain_knights_tour(n, start=(0, 0), max_calls=None):
    """
    Fixed move-order backtracking. Returns (path_or_None, calls_made).
    max_calls: optional cap to abort early (plain backtracking can be
    infeasibly slow for n >= 6) - returns (None, calls_made) if exceeded.
    """
    board = [[-1] * n for _ in range(n)]
    path = [start]
    board[start[0]][start[1]] = 0
    calls = [0]
    aborted = [False]

    def backtrack(x, y, move_count):
        calls[0] += 1
        if max_calls is not None and calls[0] > max_calls:
            aborted[0] = True
            return True  # unwind immediately
        if move_count == n * n:
            return True
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if _in_bounds(nx, ny, n) and board[nx][ny] == -1:
                board[nx][ny] = move_count
                path.append((nx, ny))
                if backtrack(nx, ny, move_count + 1):
                    return True
                board[nx][ny] = -1
                path.pop()
        return False

    found = backtrack(start[0], start[1], 1)
    if aborted[0]:
        return None, calls[0]
    return (path if found else None), calls[0]


def _count_onward_moves(board, x, y, n):
    count = 0
    for dx, dy in MOVES:
        nx, ny = x + dx, y + dy
        if _in_bounds(nx, ny, n) and board[nx][ny] == -1:
            count += 1
    return count


def warnsdorff_knights_tour(n, start=(0, 0), max_calls=None):
    """
    Warnsdorff's-rule-guided backtracking: at each step, sort candidate
    moves by their onward-move count (ascending) before trying them, so
    the most constrained square is visited first. Still backtracks if a
    branch dead-ends (this is NOT a pure greedy - it's greedy ordering
    WITH a backtracking fallback, hence still correct/complete).
    """
    board = [[-1] * n for _ in range(n)]
    path = [start]
    board[start[0]][start[1]] = 0
    calls = [0]
    aborted = [False]

    def backtrack(x, y, move_count):
        calls[0] += 1
        if max_calls is not None and calls[0] > max_calls:
            aborted[0] = True
            return True
        if move_count == n * n:
            return True

        candidates = []
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if _in_bounds(nx, ny, n) and board[nx][ny] == -1:
                degree = _count_onward_moves(board, nx, ny, n)
                candidates.append((degree, nx, ny))
        candidates.sort(key=lambda c: c[0])  # Warnsdorff: fewest onward moves first

        for degree, nx, ny in candidates:
            board[nx][ny] = move_count
            path.append((nx, ny))
            if backtrack(nx, ny, move_count + 1):
                return True
            board[nx][ny] = -1
            path.pop()
        return False

    found = backtrack(start[0], start[1], 1)
    if aborted[0]:
        return None, calls[0]
    return (path if found else None), calls[0]


def verify_tour(path, n):
    """Sanity check: path visits every square exactly once, and every
    consecutive pair of squares is a valid knight move."""
    if path is None:
        return False
    if len(path) != n * n:
        return False
    if len(set(path)) != n * n:
        return False  # duplicate square
    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        if (abs(x1 - x2), abs(y1 - y2)) not in [(1, 2), (2, 1)]:
            return False
    return True
