#!/usr/bin/env python3
"""
BOSCO as a *guide* for a learning policy instead of as a controller.

`BoscoExpert` answers "what should the robot do now" with a motor command. This
module answers the weaker question "where should the robot go next", and leaves
the driving to the policy:

    BoscoGuide.update(positions, coverage_grid)
        -> target (N,) flat cell index, -1 when the robot has nothing left to do
           reached (N,) bool, True on the step the robot entered the cell it was
                              pointed at

The target is always a cell the robot can reach in a single move — a
graph-adjacent neighbour of the cell it currently occupies — so "get there" is a
one-step objective the policy can actually be rewarded for, not a distant goal
it has to plan towards.

What is reused, and what is not
------------------------------
Everything upstream of the pure-pursuit controller is inherited unchanged from
`BoscoExpert`: the traversability graph, the balanced ownership partition, the
boustrophedon tours, the revisit-priced mop-up (`_replan`) and the splice past
ground a teammate already swept (`_skip_covered`). Only the cursor discipline is
new, because the cursor is no longer moved by a controller that tracks the tour
exactly — it is moved by a policy that may ignore it. Three cases the expert
never had to handle:

* the policy cuts a corner of its own tour — the cursor jumps to the furthest
  node inside a small window that the robot is standing on, so it can never fall
  behind the robot and start pointing backwards;
* the policy wanders off the tour entirely — a revisit-priced path from wherever
  it ended up is spliced in front of the remaining tour (`_reroute`), so the
  waypoint becomes adjacent again instead of dangling several rooms away;
* the waypoint would be the cell the robot is already in (every splice re-anchors
  the tour on the current cell) — the cursor is pushed past it, which is what
  makes an arrival a genuine cell transition rather than free reward.

Cost
----
The bookkeeping is gated on the robot changing cell. At v_max * dt = 0.1 m per
step against a 0.5 m cell a robot needs ~5 steps to cross one, so ~80% of the
updates return the cached target and touch nothing. The consequence is that a
teammate covering the stretch ahead is noticed at the next cell boundary rather
than immediately, which costs at most a few steps of stale guidance.
"""

from __future__ import annotations

import copy

import numpy as np

from src.algorithms.bosco import BoscoExpert

# Guidance block appended to each robot's observation:
#   [cos(bearing), sin(bearing), distance / cell_size, target_is_valid]
# Egocentric on purpose — the actor's parameters are shared by every robot, so a
# world-frame goal would have to be re-learned per position, while a bearing is
# the same quantity everywhere on the map. An invalid target zeroes the whole
# block, which is unambiguous: cos and sin are never both zero for a real one.

# The waypoint is an adjacent cell, so the distance never exceeds ~1.5 cells;
# the clip only bounds the channel while a reroute is pending.
_MAX_GUIDE_DIST = 2.0


class BoscoGuide(BoscoExpert):
    """BOSCO's tour, exposed as a one-cell-ahead waypoint the policy may ignore."""

    def __init__(self, env, *, snap_window: int = 8, **kwargs):
        super().__init__(env, **kwargs)
        # How far ahead of the cursor a corner cut is still recognised. One cell
        # per step is the physical limit, so this only has to cover the nodes a
        # diagonal shortcut can bypass; it is a window rather than a scan of the
        # whole tour because the cost is paid on every cell change.
        self.snap_window = int(snap_window)
        self.target = np.full(self.n, -1, dtype=np.int64)

    # -- episode setup ------------------------------------------------------

    def reset(self, positions: np.ndarray) -> dict:
        info = super().reset(positions)
        n = self.n
        self.target = np.full(n, -1, dtype=np.int64)
        # -1 forces the first `update` to do the full bookkeeping for every robot.
        self._prev_cell = np.full(n, -1, dtype=np.int64)
        # Cell a reroute was last attempted from, so a robot standing in a pocket
        # no path leaves does not pay a Dijkstra every single step.
        self._route_from = np.full(n, -1, dtype=np.int64)
        # Team coverage count at the last update that found nothing to do. A
        # robot with no target only retries once the map has actually changed.
        self._fail_cov = np.full(n, -1, dtype=np.int64)
        return info

    # -- per-step guidance --------------------------------------------------

    def update(self, positions: np.ndarray,
               coverage_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,2), (H,W) -> (target (N,) int64, reached (N,) bool).

        `reached` refers to the target handed out by the *previous* call, which
        is the one the policy was acting on, so it is exactly the event the
        arrival reward should pay for.
        """
        pos = np.asarray(positions, dtype=np.float32)
        covered = np.asarray(coverage_grid, dtype=np.float32).ravel() > 0.5
        cur = self.graph.cell_of(pos).astype(np.int64)

        # The cursor invariant guarantees target != the cell the robot was in, so
        # an arrival is always a real cell transition.
        reached = (self.target >= 0) & (cur == self.target)

        n_covered = int(covered.sum())
        stale = (self.target < 0) & (self._fail_cov != n_covered)
        for r in np.flatnonzero((cur != self._prev_cell) | stale):
            self._advance(int(r), int(cur[r]), covered, pos, n_covered)
        self._prev_cell = cur
        return self.target.copy(), reached

    # -- helpers ------------------------------------------------------------

    def _forward(self, r: int, cell: int) -> int:
        """Push the cursor past every node the robot is already standing on.

        Every splice re-anchors the tour on the current cell (`_adopt` writes it
        as node 0), so without this the waypoint would be the robot's own cell —
        reached for free, on the step it is issued.
        """
        tour = self.tours[r]
        i = int(self.idx[r])
        while i < len(tour) and tour[i] == cell:
            i += 1
        self.idx[r] = i
        return i

    def _reroute(self, r: int, cell: int, covered: np.ndarray, pos: np.ndarray) -> None:
        """Stitch a path from where the policy actually is back onto the tour.

        Priced by `REVISIT_PENALTY` like every other route in the planner, so the
        way back is dragged over fresh ground where that is possible and the
        detour sweeps whatever it crosses. The remainder of the tour is kept, not
        replanned: the policy straying does not invalidate the boustrophedon
        structure of the cells it has yet to reach.
        """
        tour = self.tours[r]
        i = int(self.idx[r])
        cost = np.where(covered, 1.0 + self.revisit_penalty, 1.0)
        _, came = self.graph.weighted_paths(cell, cost)
        path = self.graph.trace(came, tour[i])
        if path:
            self._adopt(r, path + tour[i + 1:], cell, pos)

    def _advance(self, r: int, cell: int, covered: np.ndarray,
                 positions: np.ndarray, n_covered: int) -> None:
        """Re-derive robot r's waypoint from the cell it is actually in."""
        g = self.graph

        # 1. Cursor: jump past the furthest node in the window the robot occupies.
        #    Index `i` is the waypoint it was aiming at, so this is where a normal
        #    arrival retires; a hit further along means the policy cut a corner of
        #    its own tour, and leaving the cursor behind it would point the robot
        #    back at ground it has already crossed.
        tour = self.tours[r]
        i = int(self.idx[r])
        for k in range(min(i + self.snap_window, len(tour)) - 1, i - 1, -1):
            if tour[k] == cell:
                i = k + 1
                break
        self.idx[r] = i
        i = self._forward(r, cell)

        # 2. Ground a teammate already swept is worth going around, not over.
        tour = self.tours[r]
        if i < len(tour) and covered[tour[i]]:
            self._skip_covered(r, cell, covered, positions[r])
            i = self._forward(r, cell)

        # 3. Tour finished: head for the nearest cell still pending.
        if i >= len(self.tours[r]):
            self._replan(r, cell, covered, positions)
            i = self._forward(r, cell)

        # 4. Off route: the waypoint has to stay one move away, or the arrival
        #    reward is unearnable and the bearing points at nothing reachable.
        tour = self.tours[r]
        if (i < len(tour) and tour[i] not in g.neighbors[cell]
                and cell != self._route_from[r]):
            self._route_from[r] = cell
            self._reroute(r, cell, covered, positions[r])
            i = self._forward(r, cell)

        tour = self.tours[r]
        self.target[r] = tour[i] if i < len(tour) else -1
        self._fail_cov[r] = n_covered if self.target[r] < 0 else -1




def make_guides(env, n: int) -> list[BoscoGuide]:
    """One guide per parallel env, sharing the (read-only) traversability graph.

    Building the graph sweeps every candidate edge of the map and is identical
    for all envs — they run the same layout. Every mutable field (tours, cursors,
    targets) is rebound by `reset`, so the shallow copies never share a writable
    array as long as each guide is reset before its first `update`.
    """
    base = BoscoGuide(env)
    return [base] + [copy.copy(base) for _ in range(n - 1)]
