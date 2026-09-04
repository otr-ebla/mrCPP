from types import SimpleNamespace

import numpy as np

from src.algorithms.bosco import _balance_by_swap


def _grid_graph(h: int, w: int):
    neighbors = np.full((h * w, 4), -1, dtype=np.int32)
    for row in range(h):
        for col in range(w):
            cell = row * w + col
            candidates = (
                (row + 1, col),
                (row - 1, col),
                (row, col + 1),
                (row, col - 1),
            )
            for direction, (next_row, next_col) in enumerate(candidates):
                if 0 <= next_row < h and 0 <= next_col < w:
                    neighbors[cell, direction] = next_row * w + next_col
    return SimpleNamespace(neighbors=neighbors, n_cells=h * w)


def _perimeter(graph, owner):
    total = 0
    for cell, robot in enumerate(owner):
        total += sum(
            neighbor < 0 or owner[int(neighbor)] != robot
            for neighbor in graph.neighbors[cell]
        )
    return total


def test_balancing_prefers_the_swap_with_the_smaller_perimeter():
    graph = _grid_graph(3, 3)
    owner = np.asarray([
        0, 0, 0,
        0, 0, 1,
        0, 1, 1,
    ], dtype=np.int32)
    starts = np.asarray([0, 5], dtype=np.int32)
    rows, cols = np.divmod(np.arange(9), 3)
    dist = np.stack([
        np.abs(rows - rows[start]) + np.abs(cols - cols[start])
        for start in starts
    ]).astype(np.float32)

    balanced = _balance_by_swap(
        graph,
        owner,
        starts,
        dist,
        target=np.asarray([4.5, 4.5]),
        n_robots=2,
        n_cells=9,
    )

    # Cell 2 is as close to robot 1 but would protrude from its region. Cell 4
    # shares two sides with each region, so transferring it balances the areas
    # without increasing the inter-region boundary.
    assert balanced[2] == 0
    assert balanced[4] == 1
    assert _perimeter(graph, balanced) <= _perimeter(graph, owner)
    assert sorted(np.bincount(balanced, minlength=2).tolist()) == [4, 5]
