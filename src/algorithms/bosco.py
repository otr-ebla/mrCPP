#!/usr/bin/env python3
"""
BOSCO — Boustrophedon Oracle with Spatial Cell Ownership.

A deterministic, learning-free multi-robot coverage planner used as the expert
(oracle) for imitation learning / behaviour cloning on
`MultiRobotCoverageEnv`. No neural network, no randomness: given the same map
and the same spawn positions it always produces the same joint trajectory.

Pipeline
--------
0. Traversability graph (once per map, vectorised numpy)
   Nodes are the coverable cell centres of the environment. An edge between two
   4-adjacent cells exists only if the *robot disc* can slide between the two
   centres without touching a wall. This matters: two cells flanking a thin
   inner wall are 4-adjacent in the environment's free mask, yet the straight
   move between them cuts through the wall. Planning on the raw free mask would
   drive robots into walls; planning on this graph cannot.

1. Spatial cell ownership (partition)
   Balanced multi-source wavefront from the robots' spawn cells: at every
   iteration the robot owning the *fewest* cells expands its frontier by one.
   Regions are therefore connected by construction (a robot only ever claims a
   cell adjacent to something it already owns), contain the robot's own spawn,
   and end up near-equal in area. This is a lightweight, deterministic cousin of
   DARP that needs no iterative optimisation.

2. Boustrophedon tour (per robot, offline)
   The owned cells are swept back and forth along the region's *longer* axis
   (fewer turns). Consecutive sweep targets that are not adjacent — because a
   wall, an obstacle or the region boundary breaks the lane — are reconnected
   with A* restricted to the robot's own region. The result is a single ordered
   cell tour that visits every owned cell.

3. Execution (online, closed loop)
   A pure-pursuit controller turns the next tour waypoint into the environment's
   normalised action:
       action[0] -> v     in [0, v_max]      (a0 = 2v/v_max - 1)
       action[1] -> omega in [-omega_max, omega_max]
   Four reactive layers keep the open-loop tour honest:
     * line-of-sight shortcutting — aim at the farthest tour node still reachable
       in a straight, wall-free line *whose segment still crosses every skipped
       cell*, so smoothing never sacrifices coverage;
     * priority yielding plus a one-step kinematic safety shield — on a near
       encounter the higher-id robot yields, then predicted overlapping commands
       are stopped before they reach the environment;
     * stuck recovery + mop-up — a robot that stops making progress pulls clear of
       any teammate it is wedged against, replans, and spins if it keeps failing;
       a robot whose tour is finished walks to the nearest cell still uncovered
       (own region first, then the whole map), so the team converges to full
       coverage instead of idling;
     * an idle watchdog on the pose alone. The layers above act on the robot's
       *plan*, so none of them can see a robot the planner itself has told to
       stand still, and that is the failure mode that ends a sweep: any robot
       which has not left an `arrive_tol` ball in `idle_limit` steps is pushed
       through recovery whatever its plan believes it is doing.

Measured on the default 12x8 m indoor map (384 coverable cells, 3 robots):
100% coverage on every episode, ~1600 steps to finish, and no robot ever
stationary for more than the recovery budget. The step count is dominated by
turning: at omega_max = 1 rad/s a 90° lane change costs 16 steps, and roughly
half of all steps are spent turning in place.

Usage
-----
    python -m src.algorithms.bosco --episodes 5
    python -m src.algorithms.bosco --episodes 200 --save data/bosco_expert.npz

The saved archive is a flat behaviour-cloning dataset:
    obs      (T, obs_dim)   per-agent observations, exactly env.get_obs
    actions  (T, 2)         expert actions in [-1, 1]
    episode  (T,)           episode id
    agent    (T,)           agent id
"""

from __future__ import annotations

import argparse
import heapq
import os
import sys
import time
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_TWO_PI = 2.0 * np.pi


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap to (-pi, pi]."""
    return (a + np.pi) % _TWO_PI - np.pi


# ---------------------------------------------------------------------------
# Phase 0 — traversability graph
# ---------------------------------------------------------------------------

class TraversabilityGraph:
    """Cell-centre graph on which the robot disc can actually move.

    Built once per map with vectorised numpy: every candidate edge is sampled
    along its length and every sample is tested against every wall AABB in one
    broadcast, so construction is a couple of array ops rather than a loop.
    """

    # 4-neighbourhood as (drow, dcol)
    _DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def __init__(self, env, edge_samples: int = 11, los_step: float = 0.05,
                 map_id: int = 0):
        self.h, self.w = env.grid_h, env.grid_w
        self.cell_size = env.cell_size
        self.n_cells = self.h * self.w
        self.map_id = int(map_id)
        self.los_step = float(los_step)

        walls = np.asarray(env.walls[self.map_id], dtype=np.float32)
        self._wx0, self._wy0 = walls[None, :, 0], walls[None, :, 1]
        self._wx1, self._wy1 = walls[None, :, 2], walls[None, :, 3]
        # Exactly the environment's own collision radius. Inflating it would
        # sterilise the cell lines flanking the inner walls, which clear the
        # wall face by only 0.01 m, and those lines must stay coverable.
        self._r2 = float(env.robot_radius) ** 2

        # Cell centres, flat-indexed as row * W + col (the env's own layout).
        rows, cols = np.divmod(np.arange(self.n_cells), self.w)
        self.centers = np.stack([
            (cols + 0.5) * self.cell_size,
            (rows + 0.5) * self.cell_size,
        ], axis=1).astype(np.float32)
        self.rows, self.cols = rows, cols
        self.free = np.asarray(env.free_mask_np[self.map_id], dtype=np.float32).ravel() > 0.5

        room_masks = np.asarray(env.room_masks[self.map_id]) > 0.5
        self.room_id = np.full(self.n_cells, -1, dtype=np.int32)
        for ri in range(len(room_masks)):
            self.room_id[room_masks[ri].ravel()] = ri

        self.neighbors, self.degree = self._build_edges(edge_samples)
        self.component = self._connected_components()

    # -- geometry -----------------------------------------------------------

    def wall_distance(self, pts: np.ndarray) -> np.ndarray:
        """pts (K, 2) -> (K,) distance from each point to the nearest wall."""
        cx = np.clip(pts[:, 0:1], self._wx0, self._wx1)
        cy = np.clip(pts[:, 1:2], self._wy0, self._wy1)
        d2 = (pts[:, 0:1] - cx) ** 2 + (pts[:, 1:2] - cy) ** 2
        return np.sqrt(d2.min(axis=1))

    def clear(self, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """pts (K, 2) -> (K,) bool: robot centre may stand there."""
        cx = np.clip(pts[:, 0:1], self._wx0, self._wx1)
        cy = np.clip(pts[:, 1:2], self._wy0, self._wy1)
        d2 = (pts[:, 0:1] - cx) ** 2 + (pts[:, 1:2] - cy) ** 2
        r2 = self._r2 if margin == 0.0 else (np.sqrt(self._r2) + margin) ** 2
        return ~np.any(d2 < r2, axis=1)

    def _segment_points(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Uniform samples along p->q, spaced at most `los_step` apart."""
        n = max(2, int(np.ceil(np.linalg.norm(q - p) / self.los_step)) + 1)
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        return p[None, :] * (1.0 - t) + q[None, :] * t

    def los_clear(self, p: np.ndarray, q: np.ndarray, margin: float = 0.0) -> bool:
        """True if the robot disc can travel the straight segment p -> q."""
        return bool(np.all(self.clear(self._segment_points(p, q), margin)))

    def cells_on_segment(self, p: np.ndarray, q: np.ndarray) -> set[int]:
        pts = self._segment_points(p, q)
        c = np.clip((pts[:, 0] / self.cell_size).astype(np.int32), 0, self.w - 1)
        r = np.clip((pts[:, 1] / self.cell_size).astype(np.int32), 0, self.h - 1)
        return set((r * self.w + c).tolist())

    def cell_of(self, pos: np.ndarray) -> np.ndarray:
        """pos (N, 2) -> (N,) flat cell index."""
        c = np.clip((pos[:, 0] / self.cell_size).astype(np.int32), 0, self.w - 1)
        r = np.clip((pos[:, 1] / self.cell_size).astype(np.int32), 0, self.h - 1)
        return r * self.w + c

    # -- construction -------------------------------------------------------

    def _build_edges(self, samples: int) -> tuple[np.ndarray, np.ndarray]:
        """(C, 4) neighbour indices (-1 = no edge) and (C,) degrees."""
        nbr = np.full((self.n_cells, 4), -1, dtype=np.int32)
        t = np.linspace(0.0, 1.0, samples, dtype=np.float32)[None, :, None]

        for d, (dr, dc) in enumerate(self._DIRS):
            nr, nc = self.rows + dr, self.cols + dc
            inside = (nr >= 0) & (nr < self.h) & (nc >= 0) & (nc < self.w)
            nidx = np.where(inside, nr * self.w + nc, 0)
            ok = inside & self.free & self.free[nidx]

            src = self.centers                       # (C, 2)
            dst = self.centers[nidx]                 # (C, 2)
            pts = src[:, None, :] * (1.0 - t) + dst[:, None, :] * t   # (C, S, 2)
            swept = self.clear(pts.reshape(-1, 2)).reshape(self.n_cells, -1)
            ok &= np.all(swept, axis=1)
            nbr[ok, d] = nidx[ok]

        return nbr, (nbr >= 0).sum(axis=1).astype(np.int32)

    def _connected_components(self) -> np.ndarray:
        """(C,) component id, -1 for non-free cells."""
        comp = np.full(self.n_cells, -1, dtype=np.int32)
        cid = 0
        for seed in np.flatnonzero(self.free):
            if comp[seed] >= 0:
                continue
            queue = deque([seed])
            comp[seed] = cid
            while queue:
                c = queue.popleft()
                for nb in self.neighbors[c]:
                    if nb >= 0 and comp[nb] < 0:
                        comp[nb] = cid
                        queue.append(nb)
            cid += 1
        return comp

    # -- search -------------------------------------------------------------

    def weighted_paths(self, start: int, cost: np.ndarray,
                       allowed: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Dijkstra from `start`, where *entering* cell c costs `cost[c]`.

        Returns (dist (C,) with inf where unreachable, came (C,) int32 with -1
        at the source and -2 where unreached).

        The per-cell cost is what makes every route in this planner
        redundancy-aware: charging extra for a cell that has already been swept
        prices a transit by the number of *covered* cells it crosses, so a
        detour over fresh ground is free and a shortcut over swept ground is
        not. Plain hop-count search cannot express that distinction — it is
        indifferent between a path that re-sweeps ten cells and one that covers
        ten new ones.

        One run answers every "cheapest way from here to any of those" query of
        a planning iteration, which is why the tour builder can afford to
        re-derive its sweep strips from scratch after every leg.
        """
        n = self.n_cells
        dist = np.full(n, np.inf)
        came = np.full(n, -2, dtype=np.int32)
        start = int(start)
        dist[start] = 0.0
        came[start] = -1
        heap = [(0.0, start)]
        while heap:
            d, cur = heapq.heappop(heap)
            if d > dist[cur]:
                continue
            for nb in self.neighbors[cur]:
                if nb < 0:
                    continue
                nb = int(nb)
                if allowed is not None and not allowed[nb]:
                    continue
                nd = d + cost[nb]
                if nd < dist[nb]:
                    dist[nb] = nd
                    came[nb] = cur
                    heapq.heappush(heap, (nd, nb))
        return dist, came

    @staticmethod
    def trace(came: np.ndarray, goal: int) -> list[int]:
        """Source -> goal cell path from a `weighted_paths` predecessor array."""
        goal = int(goal)
        if came[goal] == -2:
            return []
        path = [goal]
        while came[path[-1]] >= 0:
            path.append(int(came[path[-1]]))
        return path[::-1]

    def nearest_path(self, start: int, targets: np.ndarray, cost: np.ndarray,
                     allowed: np.ndarray | None = None) -> list[int] | None:
        """Cheapest path from `start` to any cell where `targets` is True.

        "Cheapest" under `cost`, not shortest: among the reachable targets this
        picks the one whose approach crosses the least already-swept ground,
        and the path it returns sweeps whatever fresh cells lie on the way.
        """
        start = int(start)
        if allowed is not None and not allowed[start]:
            allowed = None                     # robot drifted out of its region
        dist, came = self.weighted_paths(start, cost, allowed=allowed)
        d = np.where(targets, dist, np.inf)
        d[start] = np.inf                      # a target underfoot is not a goal
        goal = int(np.argmin(d))
        if not np.isfinite(d[goal]):
            return None
        return self.trace(came, goal)


# ---------------------------------------------------------------------------
# Phase 1 & 2 — ownership and boustrophedon tours
# ---------------------------------------------------------------------------

def _repair_connectivity(graph: TraversabilityGraph, owner: np.ndarray,
                         starts: np.ndarray, dist: np.ndarray,
                         n_robots: int) -> np.ndarray:
    """Re-assign stray (disconnected) components to adjacent robot regions.

    Voronoi partitions can produce disconnected pockets in corridor-heavy maps:
    a robot may "win" a cluster of cells separated from its spawn by another
    robot's territory.  BFS from each spawn within its own region identifies
    those strays; each stray cell is then re-assigned to the adjacent robot with
    the shortest BFS distance to it, so total travel time barely changes.
    The outer loop repeats until no strays remain (cascade-safe).
    """
    owner = owner.copy()
    changed = True
    while changed:
        changed = False
        for r in range(n_robots):
            s = int(starts[r])
            mask = owner == r
            if not mask[s]:
                continue            # spawn was re-assigned — skip safely
            # BFS within r's territory from its spawn cell
            visited = np.zeros(graph.n_cells, dtype=bool)
            queue = deque([s])
            visited[s] = True
            while queue:
                c = queue.popleft()
                for nb in graph.neighbors[c]:
                    nb = int(nb)
                    if nb >= 0 and mask[nb] and not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            # Cells owned by r but unreachable from its spawn are strays
            for c in np.flatnonzero(mask & ~visited):
                best_r, best_d = -1, np.inf
                # Prefer a graph-adjacent neighbour already in another region
                for nb in graph.neighbors[c]:
                    nb = int(nb)
                    if nb >= 0 and owner[nb] != r:
                        nr = int(owner[nb])
                        if nr >= 0 and dist[nr, c] < best_d:
                            best_d, best_r = dist[nr, c], nr
                if best_r < 0:      # no adjacent neighbour — fall back to global min
                    for r2 in range(n_robots):
                        if r2 != r and dist[r2, c] < best_d:
                            best_d, best_r = dist[r2, c], r2
                if best_r >= 0:
                    owner[c] = best_r
                    changed = True
    return owner


def _is_connected_after_removal(graph: TraversabilityGraph, owner: np.ndarray,
                                 r: int, cell: int, spawn: int) -> bool:
    """True if robot r's region minus `cell` is still connected from `spawn`."""
    mask = (owner == r)
    mask[cell] = False
    if not mask[spawn]:
        return False                # removing the spawn itself — never allowed
    visited = np.zeros(graph.n_cells, dtype=bool)
    queue = deque([spawn])
    visited[spawn] = True
    while queue:
        c = queue.popleft()
        for nb in graph.neighbors[c]:
            nb = int(nb)
            if nb >= 0 and mask[nb] and not visited[nb]:
                visited[nb] = True
                queue.append(nb)
    return bool(np.all(visited[mask]))


def _balance_by_swap(graph: TraversabilityGraph, owner: np.ndarray,
                     starts: np.ndarray, dist: np.ndarray,
                     target: np.ndarray, n_robots: int,
                     n_cells: int) -> np.ndarray:
    """Greedily transfer boundary cells to balance region sizes.

    Evaluates all borders between regions and diffuses cells from larger
    regions to smaller ones. This allows cells to flow across intermediate
    robots, fixing cases where a robot is "boxed in" by the wavefront.
    Runs until every robot's count is within 1 of its target or no swap
    is possible.
    """
    owner = owner.copy()
    counts = np.array([(owner == r).sum() for r in range(n_robots)],
                      dtype=np.float64)

    for _outer in range(n_cells * 5):
        if np.max(np.abs(counts - target)) <= 1:
            break
            
        # Find all valid boundary swaps
        # A valid swap is a cell `c` owned by `donor` adjacent to a cell
        # owned by `recv`, where counts[donor] > counts[recv] + 1.
        candidates = []
        for c in range(n_cells):
            donor = int(owner[c])
            if donor < 0:
                continue
            for nb in graph.neighbors[c]:
                nb = int(nb)
                if nb < 0:
                    continue
                recv = int(owner[nb])
                if recv < 0 or recv == donor:
                    continue
                
                count_diff = counts[donor] - counts[recv]
                if count_diff > 1.1:  # strictly greater than 1 (allowing for float inaccuracies)
                    # We want to maximize count_diff, and minimize distance
                    # We store (-count_diff, dist) to use min-heap / sort
                    cost = float(dist[recv, c])
                    candidates.append((-count_diff, cost, c, donor, recv))
                    
        if not candidates:
            break  # No border cells can be swapped to improve balance
            
        # Sort candidates: primarily by largest count difference (most balancing),
        # secondarily by smallest distance (keeps regions compact).
        candidates.sort()
        
        swapped = False
        for _, _, c, donor, recv in candidates:
            if _is_connected_after_removal(graph, owner, donor, c, int(starts[donor])):
                owner[c] = recv
                counts[donor] -= 1
                counts[recv] += 1
                swapped = True
                break
                
        if not swapped:
            break  # All balancing swaps break connectivity

    return owner


def partition_cells(graph: TraversabilityGraph, starts: np.ndarray,
                    max_iter: int = 100, alpha: float = 0.5) -> np.ndarray:
    """Balanced, makespan-aware cell partition.  Returns (C,) owner id (int32).

    Cells unreachable by any robot are left at -1, matching the old interface.

    Algorithm
    ---------
    One BFS per robot computes the hop-count distance from each spawn to every
    free cell.  A single unified priority-queue wavefront then assigns cells
    using a two-level priority key:

        key = (count[r] / target[r],  dist[r, c])

    Primary key — fractional fill: the robot furthest below its fair share
    always claims the next cell.  This guarantees that final region sizes
    satisfy |count[r] - target[r]| ≤ 1 for every robot by construction, with
    no post-hoc rebalancing pass.

    Secondary key — BFS distance: when two robots have the same fill ratio
    (common at the start and at integer multiples of the target), the one
    closest to the frontier cell wins.  Closeness = shorter travel time, so
    this secondary sort directly minimises the makespan — the time the last
    robot needs to finish — without sacrificing the fairness guarantee.

    Regions are connected by construction (a robot only ever claims a cell
    adjacent to one it already owns), so `_repair_connectivity` is not needed
    in normal operation but is kept as a safety net.

    The old helper functions (`_balance_by_swap`, `_repair_connectivity`) are
    retained and still callable; this function no longer invokes either.
    """
    n_robots = len(starts)
    n_cells = graph.n_cells

    # ------------------------------------------------------------------ #
    # Step 1: BFS hop-count distances from each spawn                     #
    # ------------------------------------------------------------------ #
    INF = np.float32(np.inf)
    dist = np.full((n_robots, n_cells), INF, dtype=np.float32)
    for r, s in enumerate(starts):
        s = int(s)
        dist[r, s] = 0.0
        queue = deque([s])
        while queue:
            c = queue.popleft()
            nd = dist[r, c] + 1.0
            for nb in graph.neighbors[c]:
                nb = int(nb)
                if nb >= 0 and dist[r, nb] == INF:
                    dist[r, nb] = nd
                    queue.append(nb)

    free_count = int(graph.free.sum())
    if n_robots == 0 or free_count == 0:
        return np.full(n_cells, -1, dtype=np.int32)

    # ------------------------------------------------------------------ #
    # Step 2: Target counts                                               #
    # ------------------------------------------------------------------ #
    fair = free_count // n_robots
    remainder = free_count % n_robots
    # First `remainder` robots get one extra cell so the total is exact.
    target = np.array([fair + (1 if r < remainder else 0)
                       for r in range(n_robots)], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Step 3: Unified priority-queue wavefront                            #
    #                                                                     #
    # A robot-level min-heap drives expansion.  Each entry is:           #
    #   (fill_ratio, min_frontier_dist, robot_id)                        #
    # so the most under-loaded robot always expands next (fairness), and  #
    # among equal-fill robots the one with the cheapest frontier cell     #
    # expands (makespan: closest robot to unclaimed territory wins).      #
    #                                                                     #
    # Each expansion pops the cheapest reachable unowned cell from that   #
    # robot's BFS frontier, claims it, and pushes the robot back with     #
    # its updated fill ratio and new minimum frontier distance.           #
    # Regions are connected by construction.                              #
    # ------------------------------------------------------------------ #
    owner = np.full(n_cells, -1, dtype=np.int32)
    counts = np.zeros(n_robots, dtype=np.float64)
    # Per-robot BFS frontier: min-heap of (bfs_dist, cell)
    frontiers: list[list] = [[] for _ in range(n_robots)]

    for r, s in enumerate(starts):
        s = int(s)
        if owner[s] >= 0:
            continue
        owner[s] = r
        counts[r] = 1.0
        for nb in graph.neighbors[s]:
            nb = int(nb)
            if nb >= 0 and graph.free[nb]:
                heapq.heappush(frontiers[r], (float(dist[r, nb]), nb))

    # Robot-level heap: (count, robot_id) — count drives fairness (smallest
    # count always expands next), robot_id gives a stable deterministic tiebreak.
    # Makespan awareness comes from each robot's per-frontier heap, which
    # always expands the nearest unclaimed cell first.
    robot_heap = []
    for r in range(n_robots):
        if frontiers[r]:
            heapq.heappush(robot_heap, (int(counts[r]), r))

    while robot_heap:
        count, r = heapq.heappop(robot_heap)
        # Drain stale frontier entries (already claimed by another robot)
        while frontiers[r] and owner[frontiers[r][0][1]] >= 0:
            heapq.heappop(frontiers[r])
        if not frontiers[r]:
            continue                # this robot's frontier is exhausted

        # Claim the cheapest available frontier cell
        d, c = heapq.heappop(frontiers[r])
        while owner[c] >= 0:       # stale: keep draining
            if not frontiers[r]:
                break
            d, c = heapq.heappop(frontiers[r])
        if owner[c] >= 0:
            continue

        owner[c] = r
        counts[r] += 1.0
        for nb in graph.neighbors[c]:
            nb = int(nb)
            if nb >= 0 and graph.free[nb] and owner[nb] < 0:
                heapq.heappush(frontiers[r], (float(dist[r, nb]), nb))

        # Drain stale, then re-push robot if frontier not empty
        while frontiers[r] and owner[frontiers[r][0][1]] >= 0:
            heapq.heappop(frontiers[r])
        if frontiers[r]:
            heapq.heappush(robot_heap, (int(counts[r]), r))

    # Any free cell still unowned sits in a traversability component that
    # no robot spawned in.  Assign each to the most under-loaded robot whose
    # current territory is adjacent to the cell; if none is, leave it at -1
    # (it is genuinely isolated — no robot can reach it via the graph).
    for c in np.flatnonzero((owner < 0) & graph.free):
        best_r, best_fill = -1, np.inf
        for nb in graph.neighbors[c]:
            nb = int(nb)
            if nb >= 0 and owner[nb] >= 0:
                r2 = int(owner[nb])
                f = counts[r2] / target[r2]
                if f < best_fill:
                    best_fill, best_r = f, r2
        if best_r >= 0:
            owner[c] = best_r
            counts[best_r] += 1.0

    # ------------------------------------------------------------------ #
    # Step 4: Diffusion balancing                                         #
    #                                                                     #
    # The wavefront can "box in" a robot before it reaches its quota.     #
    # _balance_by_swap iteratively transfers cells across borders from    #
    # over-loaded to under-loaded regions, subject to connectivity,       #
    # fixing any geometric trapping.                                      #
    # ------------------------------------------------------------------ #
    owner = _balance_by_swap(graph, owner, starts, dist, target, n_robots, n_cells)

    return owner


# Extra cost of entering a cell that has already been swept, in units of one
# cell of travel. Every revisit the planner accepts must therefore buy it a
# detour shorter than this many fresh cells, so the value is the exchange rate
# between the two things the sweep trades off: redundant passes and path length.
# 20 exceeds the width of any single region on this map, which makes the search
# effectively lexicographic — it minimises revisits first and only then distance
# — while still refusing a detour so long that it wanders the whole floor.
REVISIT_PENALTY = 20.0


def _lane_runs(graph: TraversabilityGraph, remaining: np.ndarray,
               dir_idx: int) -> list[np.ndarray]:
    """Maximal straight, connected runs of `remaining` along one axis.

    `dir_idx` indexes `TraversabilityGraph._DIRS`: 2 steps one column right, so
    lanes are rows; 0 steps one row down, so lanes are columns.

    A run is a stretch the robot sweeps end to end with no turn and no revisit —
    the atom a boustrophedon path is built from. This is the boustrophedon
    *decomposition* the old global lexsort skipped: sorting all the region's
    cells by (lane, +-position) produces an order whose consecutive entries can
    sit either side of a wall, and reconnecting those with a shortest path is
    precisely what dragged the robot back across ground it had already swept.

    Splitting on *graph* edges rather than index adjacency is what keeps the
    sweep honest: two cells flanking a thin inner wall are index-adjacent yet
    have no edge, so they land in separate runs and get a real transit between
    them instead of a move through the wall.
    """
    nxt = graph.neighbors[:, dir_idx].astype(np.int64)
    ok = nxt >= 0
    link = np.zeros(graph.n_cells, dtype=bool)          # c -> c's successor
    link[ok] = remaining[ok] & remaining[nxt[ok]]
    entered = np.zeros(graph.n_cells, dtype=bool)
    entered[nxt[link]] = True                           # has a predecessor

    runs = []
    for head in np.flatnonzero(remaining & ~entered):
        run = [int(head)]
        while link[run[-1]]:
            run.append(int(nxt[run[-1]]))
        runs.append(np.asarray(run, dtype=np.int64))
    return runs


def _sweep_tour(graph: TraversabilityGraph, region: np.ndarray, start: int,
                dir_idx: int, penalty: float) -> tuple[list[int], int]:
    """Boustrophedon tour of `region` along one axis. Returns (tour, transits).

    Greedy strip chaining: decompose what is left of the region into sweepable
    runs, drive to the cheapest run *endpoint* under the revisit-priced metric,
    sweep that run, repeat. Entering a run anywhere but at an endpoint would
    strand the shorter half and cost a second pass over it, so only the two
    endpoints of each run are ever considered as entry points.

    The decomposition is redone every iteration because a transit is allowed to
    cut through fresh cells — that is how it stays free — and doing so splits
    the run it crossed. Recomputing is O(region) and there are only as many
    iterations as there are runs, so the whole tour costs far less than the
    single A* per cell the previous planner paid.
    """
    visited = np.zeros(graph.n_cells, dtype=bool)
    visited[start] = True
    tour = [int(start)]
    transits = 0

    while True:
        remaining = region & ~visited
        if not remaining.any():
            break

        cost = np.where(visited, 1.0 + penalty, 1.0)
        dist, came = graph.weighted_paths(tour[-1], cost, allowed=region)
        if not np.isfinite(dist[remaining]).any():
            # Own region momentarily cut in two by nothing but geometry: allow
            # the detour through a teammate's area rather than abandon cells.
            dist, came = graph.weighted_paths(tour[-1], cost)

        best_key, best_run = None, None
        for run in _lane_runs(graph, remaining, dir_idx):
            for ordered in (run, run[::-1]):
                d = dist[ordered[0]]
                if not np.isfinite(d):
                    continue
                # Cheapest approach first; among equals sweep the longest run,
                # which buys the most coverage before the next transit is due.
                key = (d, -len(run), int(ordered[0]))
                if best_key is None or key < best_key:
                    best_key, best_run = key, ordered
        if best_run is None:
            break                       # nothing left this robot can reach

        for c in graph.trace(came, best_run[0])[1:]:
            tour.append(c)
            visited[c] = True
        transits += 1

        # Sweep the run while it is still fresh. The transit that just landed
        # here may have eaten into it; the next iteration re-derives the runs
        # from whatever survived.
        for c in best_run[1:]:
            c = int(c)
            if visited[c] or c not in graph.neighbors[tour[-1]]:
                break
            tour.append(c)
            visited[c] = True

    return tour, transits


def plan_tour(graph: TraversabilityGraph, region: np.ndarray, start: int,
              penalty: float = REVISIT_PENALTY) -> list[int]:
    """Ordered cell tour covering every cell of `region`, beginning at `start`.

    Both sweep axes are planned and the better one kept. Guessing the axis from
    the region's bounding box, as the previous version did, optimises the wrong
    quantity: the bounding box says nothing about where the walls cut the lanes,
    and it is the cuts — not the region's proportions — that force transits.
    Planning a tour is a few milliseconds, so the axis is *measured* instead:
    fewest revisits wins, and ties go to the tour with fewer transits, since
    each transit is a lane change and a 90 degree turn costs ~16 steps.
    """
    start = int(start)
    if not region.any():
        return [start]

    best_key, best_tour = None, None
    for dir_idx in (2, 0):                      # rows as lanes, then columns
        tour, transits = _sweep_tour(graph, region, start, dir_idx, penalty)
        key = (len(tour) - len(set(tour)), transits)
        if best_key is None or key < best_key:
            best_key, best_tour = key, tour
    return best_tour


def plan_tour_room_aware(graph: TraversabilityGraph, region: np.ndarray, start: int,
                         penalty: float = REVISIT_PENALTY) -> list[int]:
    """Come plan_tour, ma esaurisce ogni stanza prima di passare alla successiva.

    Ordine stanze: quella di partenza per prima (se dentro region), poi le
    restanti in ordine "più vicina prossima" secondo il costo pesato dal punto
    in cui il tour si trova in quel momento — così ogni transito tra stanze è
    comunque il più economico disponibile, ma non può più "rubare" celle di
    un'altra stanza mentre quella corrente non è finita.
    """
    start = int(start)
    room = graph.room_id
    rooms_here = sorted({int(room[c]) for c in np.flatnonzero(region) if room[c] >= 0})
    if not rooms_here:
        return plan_tour(graph, region, start, penalty)   # fallback: nessuna info stanza

    def _best_axis_sweep(sub_region: np.ndarray, entry: int) -> list[int]:
        best_key, best_tour = None, None
        for dir_idx in (2, 0):
            tour, transits = _sweep_tour(graph, sub_region, entry, dir_idx, penalty)
            key = (len(tour) - len(set(tour)), transits)
            if best_key is None or key < best_key:
                best_key, best_tour = key, tour
        return best_tour

    tour = [start]
    remaining = set(rooms_here)
    cur_room = int(room[start]) if room[start] >= 0 else None
    if cur_room in remaining:
        tour += _best_axis_sweep(region & (room == cur_room), start)[1:]
        remaining.discard(cur_room)

    cur = tour[-1]
    while remaining:
        cost = np.ones(graph.n_cells)               # solo transito, nessuna penalità revisit
        dist, came = graph.weighted_paths(cur, cost, allowed=region)
        best_room, best_d, best_entry = None, np.inf, None
        for rm in remaining:
            cells = np.flatnonzero(region & (room == rm))
            d = dist[cells]
            if d.size == 0 or not np.isfinite(d.min()):
                continue
            j = int(np.argmin(d))
            if d[j] < best_d:
                best_d, best_room, best_entry = d[j], rm, int(cells[j])
        if best_room is None:
            break                                    # stanze restanti irraggiungibili da qui
        tour += graph.trace(came, best_entry)[1:]
        tour += _best_axis_sweep(region & (room == best_room), tour[-1])[1:]
        cur = tour[-1]
        remaining.discard(best_room)

    return tour


# ---------------------------------------------------------------------------
# Phase 3 — the expert policy
# ---------------------------------------------------------------------------

class BoscoExpert:
    """Deterministic joint policy: `reset(positions)` then `act(...)` per step."""

    def __init__(self, env, *, map_id: int = 0, lookahead: int = 16, pursuit_dist: float = 0.25,
                 turn_threshold: float = 0.10, open_turn_threshold: float = 0.7,
                 open_clearance: float = 0.25, k_omega: float = 3.0,
                 arrive_tol: float = 0.06, shortcut_margin: float = 0.015,
                 yield_radius_factor: float = 4.0, yield_limit: int = 25,
                 stuck_limit: int = 6, spin_steps: int = 8, fail_limit: int = 3,
                 revisit_penalty: float = REVISIT_PENALTY, skip_horizon: int = 4,
                 idle_limit: int = 80, collision_margin: float = 0.02):
        self.env = env
        self.n = env.num_robots
        self.dt = env.dt
        self.v_max = env.v_max
        self.omega_max = env.omega_max
        self.robot_radius = float(env.robot_radius)
        self.collision_margin = float(collision_margin)

        self.graph = TraversabilityGraph(env, map_id=map_id)
        self.lookahead = int(lookahead)
        self.pursuit_dist = float(pursuit_dist)
        self.turn_threshold = float(turn_threshold)
        self.open_turn_threshold = float(open_turn_threshold)
        self.open_clearance = float(open_clearance)
        self.k_omega = float(k_omega)
        self.arrive_tol = float(arrive_tol)
        self.shortcut_margin = float(shortcut_margin)
        self.fail_limit = int(fail_limit)
        self.yield_radius = yield_radius_factor * env.robot_radius
        self.yield_limit = int(yield_limit)
        self.stuck_limit = int(stuck_limit)
        self.spin_steps = int(spin_steps)
        self.revisit_penalty = float(revisit_penalty)
        self.skip_horizon = int(skip_horizon)
        # Longest run of steps a robot may legitimately spend without leaving an
        # `arrive_tol` ball: a yield (`yield_limit`) followed by a turn in place
        # (pi / omega_max seconds) is the worst honest case, so anything past
        # this is a livelock, not a manoeuvre.
        self.idle_limit = int(idle_limit)
        # Centre distance below which two robots count as wedged: their discs are
        # in contact at 2 * robot_radius, and the environment rejects every move
        # that would close the gap further.
        self.wedge_dist = 2.4 * env.robot_radius

        # Cells the env counts as coverable but no edge can reach (pockets sealed
        # by a wall the disc cannot squeeze past). Reported, never planned for.
        self._isolated = self.graph.free & (self.graph.degree == 0)

        self.owner: np.ndarray | None = None
        self.tours: list[list[int]] = []

    # -- episode setup ------------------------------------------------------

    def reset(self, positions: np.ndarray, map_id: int | None = None) -> dict:
        if map_id is not None and int(map_id) != self.graph.map_id:
            self.graph = TraversabilityGraph(self.env, map_id=int(map_id))
            self._isolated = self.graph.free & (self.graph.degree == 0)
        g = self.graph
        starts = g.cell_of(np.asarray(positions, dtype=np.float32))
        self.owner = partition_cells(g, starts)
        self.regions = [self.owner == r for r in range(self.n)]
        # Free cells the wavefront never reached: they sit in a traversability
        # component no robot spawned in, so they are unwinnable this episode and
        # must stay out of every mop-up search.
        self.unreachable = g.free & (self.owner < 0)
        self.tours = [plan_tour_room_aware(g, self.regions[r], int(starts[r]), self.revisit_penalty)
              for r in range(self.n)]

        self.idx = np.zeros(self.n, dtype=np.int64)          # cursor into the tour
        self.tgt = np.full(self.n, -1, dtype=np.int64)       # committed leg end
        # Length of the covered head of the current plan: the transit the robot
        # has already accepted, which `_skip_covered` must not try to avoid again.
        self.commit = np.zeros(self.n, dtype=np.int64)
        # Anchor of the very first leg of a plan, frozen at plan time. Anchoring
        # it on the live pose instead would move the reference line under the
        # controller every step and leave the robot creeping in a limit cycle.
        self.anchor0 = np.asarray(positions, dtype=np.float32).copy()
        self.prev_pos = np.asarray(positions, dtype=np.float32).copy()
        self.prev_v = np.zeros(self.n, dtype=np.float32)
        self.stuck = np.zeros(self.n, dtype=np.int32)
        # Pose-only idle watchdog: last position the robot demonstrably left,
        # and how long ago it left it.
        self.watch_pos = np.asarray(positions, dtype=np.float32).copy()
        self.idle = np.zeros(self.n, dtype=np.int32)
        self.fails = np.zeros(self.n, dtype=np.int32)
        self.spin = np.zeros(self.n, dtype=np.int32)
        self.yielding = np.zeros(self.n, dtype=np.int32)
        self.done = np.zeros(self.n, dtype=bool)

        sizes = np.array([int(m.sum()) for m in self.regions])
        # Revisits the offline tours already commit to: every tour node beyond
        # the first visit of a cell is a pass over ground this robot has swept.
        planned_revisits = int(sum(len(t) - len(set(t)) for t in self.tours))
        return {
            'region_sizes': sizes,
            'tour_lengths': np.array([len(t) for t in self.tours]),
            'unassigned_cells': int((self.owner < 0).sum() - (~g.free).sum()),
            'unreachable_cells': int(self._isolated.sum()),
            'planned_revisits': planned_revisits,
        }

    # -- per-step control ---------------------------------------------------

    def act(self, positions: np.ndarray, headings: np.ndarray,
            coverage_grid: np.ndarray) -> np.ndarray:
        """(N,2), (N,), (H,W) -> (N,2) actions in [-1, 1]."""
        g = self.graph
        pos = np.asarray(positions, dtype=np.float32)
        hdg = np.asarray(headings, dtype=np.float32)
        covered = np.asarray(coverage_grid, dtype=np.float32).ravel() > 0.5
        cur_cell = g.cell_of(pos)

        # Progress bookkeeping: a robot that was *told to drive* and did not move
        # is blocked by a wall it should not be touching or shoved by a teammate;
        # either way the open-loop tour is stale and must be re-derived from the
        # true pose. Turning in place and yielding command v = 0 on purpose and
        # must never be mistaken for being stuck.
        moved = np.linalg.norm(pos - self.prev_pos, axis=1)
        driving = self.prev_v > 1e-3
        blocked = driving & (moved < 0.3 * self.prev_v * self.dt)
        self.stuck = np.where(blocked, self.stuck + 1, 0)
        self.prev_pos = pos.copy()

        # Pose-only idle watchdog. The test above only sees a robot that was
        # *told to drive*, so every livelock in which the expert itself commands
        # v = 0 is invisible to it — a plan whose next node is the cell already
        # underfoot, a leg with no distance left to run, a splice that rebuilds
        # itself every step. Those are precisely the failures that used to park a
        # robot for the rest of the episode. This watchdog watches the pose and
        # nothing else: a robot that has not left an `arrive_tol` ball in
        # `idle_limit` steps is not sweeping, whatever its plan believes, and is
        # forced through the same spin-and-replan recovery as a blocked one.
        drifted = np.linalg.norm(pos - self.watch_pos, axis=1) > 2.0 * self.arrive_tol
        self.watch_pos = np.where(drifted[:, None], pos, self.watch_pos)
        self.idle = np.where(drifted, 0, self.idle + 1)
        idle_out = (self.idle >= self.idle_limit) & ~self.done
        if idle_out.any():
            self.idle = np.where(idle_out, 0, self.idle)
            self.spin = np.where(idle_out, self.spin_steps, self.spin)

        aim = np.zeros((self.n, 2), dtype=np.float32)      # pure-pursuit point
        remaining = np.zeros(self.n, dtype=np.float32)     # distance left on leg
        halt = np.zeros(self.n, dtype=bool)
        spin_now = np.zeros(self.n, dtype=bool)

        for r in range(self.n):
            tour = self.tours[r]

            # Recovery: spin in place, then rebuild the plan from where we are.
            if self.spin[r] > 0:
                self.spin[r] -= 1
                spin_now[r] = True
                aim[r] = pos[r]
                if self.spin[r] == 0 and not self._unwedge(r, int(cur_cell[r]), pos):
                    self._replan(r, int(cur_cell[r]), covered, pos)
                continue
            # Recovery ladder: pull clear of a teammate first — while two bodies
            # are in contact no route is drivable, so replanning one is wasted —
            # then a fresh plan for a one-off block (a teammate crossing, a
            # clipped corner). Spinning is reserved for a robot that keeps
            # failing, since it throws away a known heading.
            if self.stuck[r] >= self.stuck_limit:
                self.stuck[r] = 0
                self.fails[r] += 1
                if self.fails[r] >= self.fail_limit:
                    self.fails[r] = 0
                    self.spin[r] = self.spin_steps
                    spin_now[r] = True
                    aim[r] = pos[r]
                    continue
                if not self._unwedge(r, int(cur_cell[r]), pos):
                    self._replan(r, int(cur_cell[r]), covered, pos)
                tour = self.tours[r]     # the old plan is gone; do not index it

            i0 = int(self.idx[r])
            i = self._retire_reached(r, pos[r])
            if i > i0:                               # real progress: forgive
                self.fails[r] = 0

            # Ground someone else already swept is worth going around, not over.
            # Only past the accepted transit, though: the head of a plan is
            # covered by construction — a transit crosses swept ground — so
            # re-testing it here would splice the same detour every step and put
            # the cursor back to 0 for ever, which is a robot frozen on the spot.
            if i < len(tour) and covered[tour[i]] and i >= int(self.commit[r]):
                self._skip_covered(r, int(cur_cell[r]), covered, pos[r])
                tour = self.tours[r]        # spliced, or the cursor was retired
                i = self._retire_reached(r, pos[r])

            if i >= len(tour):                       # tour finished -> mop up
                self._replan(r, int(cur_cell[r]), covered, pos)
                tour = self.tours[r]
                i = self._retire_reached(r, pos[r])
                if i >= len(tour):                   # nothing left anywhere
                    self.done[r] = True
                    halt[r] = True
                    aim[r] = pos[r]
                    continue

            # The leg is anchored on the *cell centre* just left behind, not on
            # the robot's actual pose, so a straight run down a lane is an exact
            # lattice line. The wall-adjacent lanes clear the wall by 0.01 m, so
            # the robot must track that line rather than merely head for its far
            # end — the cross-track term below is what keeps it off the walls.
            prev = tour[i - 1] if i > 0 else -1
            anchor = g.centers[prev] if prev >= 0 else self.anchor0[r]
            j = self._shortcut(anchor, tour, i)
            self.tgt[r] = j
            goal = g.centers[tour[j]]

            leg = goal - anchor
            leg_len = float(np.linalg.norm(leg))
            if leg_len < 1e-6:
                # The plan is asking the robot to drive to the point it is
                # already standing on. Retiring the node is the only answer that
                # makes progress: commanding v = 0 and keeping the cursor put
                # reproduces the same degenerate leg on the next step, which is a
                # robot parked for good.
                self.idx[r] = i + 1
                aim[r] = pos[r]
                remaining[r] = 0.0
                continue
            u = leg / leg_len
            # Project the pose onto the leg and look ahead along it: lateral
            # error decays as the robot converges on the line.
            t = float(np.clip(np.dot(pos[r] - anchor, u), 0.0, leg_len))
            aim[r] = anchor + u * min(leg_len, t + self.pursuit_dist)
            # Distance still to travel, used below to keep the robot from
            # overshooting the leg end. It is the larger of the run left along
            # the line and the true distance to the goal: a robot shoved past the
            # end of its leg has zero of the former but still has to drive back,
            # and pricing that at zero would cap its speed at zero for good.
            remaining[r] = max(leg_len - t, float(np.linalg.norm(goal - pos[r])))

        # --- Pure pursuit, vectorised over robots ---
        delta = aim - pos
        look = np.maximum(np.linalg.norm(delta, axis=1), 1e-6)
        err = _wrap_angle(np.arctan2(delta[:, 1], delta[:, 0]) - hdg)

        # Turning in place is expensive — at omega_max = 1 rad/s a 90° lane change
        # costs 16 steps — so it is only imposed where it is actually needed. With
        # room around it the robot may arc through the turn; hugging a wall it may
        # not, because there the whole lane clears the wall by ~0.01 m.
        room = self.graph.wall_distance(pos) - self.env.robot_radius
        threshold = np.where(room > self.open_clearance,
                             self.open_turn_threshold, self.turn_threshold)

        # The chord that reaches the look-ahead point has curvature 2 sin(err)/L.
        # Capping v at omega_max * L / (2 |sin err|) is what stops the robot from
        # orbiting: the minimum turn radius at full speed is v_max/omega_max = 1 m,
        # far wider than the 0.25 m look-ahead, so an uncapped robot would circle
        # its target forever instead of converging on it.
        curvature = 2.0 * np.sin(err) / look
        v = self.v_max * np.maximum(np.cos(err), 0.0)
        v = np.where(np.abs(err) > threshold, 0.0, v)
        v = np.minimum(v, np.maximum(remaining, 0.0) / self.dt)
        v = np.minimum(v, self.omega_max * look / (2.0 * np.abs(np.sin(err)) + 1e-6))

        omega = np.where(
            v > 1e-6,
            v * curvature,                                  # follow the arc
            np.clip(self.k_omega * err, -self.omega_max, self.omega_max),
        )
        omega = np.clip(omega, -self.omega_max, self.omega_max)

        # Recovery spin: rotate on the spot, no translation.
        v = np.where(spin_now, 0.0, v)
        omega = np.where(spin_now, self.omega_max, omega)

        # --- Priority yielding: the higher-id robot gives way ---
        diff = pos[None, :, :] - pos[:, None, :]
        d = np.linalg.norm(diff, axis=2) + np.eye(self.n) * 1e9
        ahead = (diff[:, :, 0] * np.cos(hdg)[:, None]
                 + diff[:, :, 1] * np.sin(hdg)[:, None]) > 0.0
        lower_id = np.tril(np.ones((self.n, self.n), dtype=bool), -1)
        conflict = np.any((d < self.yield_radius) & ahead & lower_id, axis=1)
        # A yield that never resolves would stall the sweep, so it expires — and
        # the counter keeps running while the encounter lasts, so an expired
        # yield stays expired. Resetting it on expiry would re-arm the yield on
        # the very next step and stall the robot for the rest of the episode.
        self.yielding = np.where(conflict, self.yielding + 1, 0)
        yielded = conflict & (self.yielding <= self.yield_limit)

        stop = yielded | halt
        v = np.where(stop, 0.0, v)
        omega = np.where(stop, 0.0, omega)

        # Yielding handles ordinary encounters early, but an ID-priority rule
        # alone cannot make the final command safe: the lower-ID robot may still
        # drive directly into the higher-ID robot that stopped for it. Simulate
        # the actual next kinematic step and suppress translation before either
        # disc can overlap.
        v = self._collision_shield(pos, hdg, v, omega)

        self.prev_v = v.astype(np.float32)
        actions = np.stack([
            2.0 * v / self.v_max - 1.0,                      # v -> [-1, 1]
            np.clip(omega / self.omega_max, -1.0, 1.0),
        ], axis=1)
        return actions.astype(np.float32)

    # -- helpers ------------------------------------------------------------

    def _predict_positions(self, positions: np.ndarray, headings: np.ndarray,
                           v: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Mirror the environment's differential-drive position update."""
        straight = np.abs(omega) < 1e-6
        omega_safe = np.where(straight, 1.0, omega)
        turn_heading = headings + omega * self.dt
        radius = v / omega_safe
        pos_straight = positions + np.stack([
            v * np.cos(headings), v * np.sin(headings)
        ], axis=-1) * self.dt
        pos_turn = positions + radius[:, None] * np.stack([
            np.sin(turn_heading) - np.sin(headings),
            np.cos(headings) - np.cos(turn_heading),
        ], axis=-1)
        return np.where(straight[:, None], pos_straight, pos_turn)

    def _collision_shield(self, positions: np.ndarray, headings: np.ndarray,
                          v: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Stop unsafe translations before the environment rejects the move."""
        safe_v = np.asarray(v, dtype=np.float32).copy()
        min_dist = 2.0 * self.robot_radius + self.collision_margin

        # Stopping one robot can expose a conflict with a third robot that had
        # expected it to move away, so recompute candidates until no new stop is
        # needed. At least one velocity becomes zero on every non-final pass.
        for _ in range(self.n):
            candidate = self._predict_positions(positions, headings, safe_v, omega)
            changed = False
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    moving_i = safe_v[i] > 1e-6
                    moving_j = safe_v[j] > 1e-6
                    if not (moving_i or moving_j):
                        continue
                    if np.linalg.norm(candidate[i] - candidate[j]) >= min_dist:
                        continue

                    stop_i_safe = np.linalg.norm(positions[i] - candidate[j]) >= min_dist
                    stop_j_safe = np.linalg.norm(candidate[i] - positions[j]) >= min_dist
                    if stop_i_safe and stop_j_safe:
                        # Preserve BOSCO's deterministic ID priority when either
                        # robot can safely yield.
                        safe_v[j] = 0.0
                    elif stop_i_safe:
                        safe_v[i] = 0.0
                    elif stop_j_safe:
                        safe_v[j] = 0.0
                    else:
                        safe_v[i] = 0.0
                        safe_v[j] = 0.0
                    changed = True
            if not changed:
                break
        return safe_v

    def _retire_reached(self, r: int, p: np.ndarray) -> int:
        """Retire every tour node the robot is already standing on. Returns the cursor.

        A node counts as reached only when the robot is *on its centre*, not
        merely inside its cell: advancing on cell entry would anchor the next leg
        up to a quarter-cell off the lattice line, and the lanes flanking the
        inner walls have only 0.01 m to give.

        This runs again after every mid-step replan, which is what keeps a fresh
        plan from producing a zero-length first leg: a plan always starts at the
        cell the robot occupies, and if it is already on that centre the node is
        nothing left to drive to.
        """
        g, tour = self.graph, self.tours[r]
        i = int(self.idx[r])
        # Arriving at the committed leg end retires every node it shortcut past;
        # those cells were crossed by the leg, which is the condition the shortcut
        # was accepted under. Without this the cursor would wait forever on a
        # centre the straight leg legitimately bypassed.
        j = int(self.tgt[r])
        if i <= j < len(tour) and np.linalg.norm(
                p - g.centers[tour[j]]) < self.arrive_tol:
            i = j + 1
        while i < len(tour) and np.linalg.norm(
                p - g.centers[tour[i]]) < self.arrive_tol:
            i += 1
        self.idx[r] = i
        return i

    def _shortcut(self, p: np.ndarray, tour: list[int], i: int) -> int:
        """Farthest tour node reachable in a straight line without skipping cells.

        Returns the tour *index* to aim at, never a cell id.

        Aiming several nodes ahead removes the stop-and-turn at every cell centre
        on a straight lane. Coverage is preserved by requiring the straight
        segment to pass through *every* cell it skips, and the small clearance
        margin keeps shortcuts out of the wall-hugging lanes: those fall back to
        single-cell legs, whose edges were validated at the exact robot radius.
        """
        g = self.graph
        j_max = min(i + self.lookahead, len(tour) - 1)
        for j in range(j_max, i, -1):
            q = g.centers[tour[j]]
            if not g.los_clear(p, q, self.shortcut_margin):
                continue
            crossed = g.cells_on_segment(p, q)
            if all(c in crossed for c in tour[i:j]):
                return j
        return i

    def _adopt(self, r: int, path: list[int], cell: int, pos: np.ndarray,
               covered: np.ndarray | None = None) -> None:
        """Install `path` as robot r's tour and put the cursor back on the lattice.

        Cursor 0 makes the robot re-acquire its own cell centre first, which
        puts it back on the lattice line before the new leg starts.

        The covered head of the new plan is recorded as a *commitment*. Every
        path this planner builds starts on the cell the robot occupies and reaches
        its goal across whatever ground lies between, so its first cells are
        covered by construction. Without the commitment `_skip_covered` reads that
        head as a stretch worth detouring around, rebuilds the very same transit,
        resets the cursor to 0, and does it again on the next step — a robot
        replanning itself into a standstill it can never leave.
        """
        self.tours[r] = path if path else [int(cell)]
        self.idx[r] = 0
        self.tgt[r] = -1
        self.anchor0[r] = pos
        # A robot with somewhere to go is not finished. Leaving the flag latched
        # would exempt it from the idle watchdog for the rest of the episode.
        self.done[r] = not path
        k = 0
        if covered is not None:
            tour = self.tours[r]
            while k < len(tour) and covered[tour[k]]:
                k += 1
        self.commit[r] = k

    def _unwedge(self, r: int, cell: int, positions: np.ndarray) -> bool:
        """Break a body-to-body wedge by driving away from the teammate.

        Two robots pressed together at 2 * robot_radius cannot be separated by
        replanning: every plan starts at the cell the robot occupies, so both keep
        aiming at a centre the other one is sitting on and the environment rejects
        every move as a collision. What they need is not a better route but a
        metre of clearance, so the escape leg deliberately ignores coverage and
        heads for whichever neighbouring cell opens the gap fastest.

        The leg runs from the live pose, not from a cell centre — the robot is
        wedged precisely because it cannot reach its own centre — so the straight
        line is checked against the walls at the full robot radius before it is
        accepted. Returns False when nothing is wedged or nothing opens up.
        """
        g = self.graph
        pos = positions[r]
        others = np.ones(len(positions), dtype=bool)
        others[r] = False
        if not others.any():
            return False
        rel = positions[others] - pos[None, :]
        d = np.linalg.norm(rel, axis=1)
        if d.min() > self.wedge_dist:
            return False

        blocker = positions[others][int(np.argmin(d))]
        best, best_gap = None, -np.inf
        for nb in g.neighbors[cell]:
            nb = int(nb)
            if nb < 0:
                continue
            centre = g.centers[nb]
            gap = float(np.linalg.norm(centre - blocker))
            if gap > best_gap and g.los_clear(pos, centre):
                best, best_gap = nb, gap
        if best is None:
            return False
        # A single-node plan whose leg starts at the pose: the head that would
        # normally re-acquire this cell's centre is exactly what cannot be done.
        self._adopt(r, [best], cell, pos)
        return True

    def _replan(self, r: int, cell: int, covered: np.ndarray,
                positions: np.ndarray) -> None:
        """Head for the nearest uncovered cell: own region first, then anywhere.

        "Nearest" is priced by `REVISIT_PENALTY`, so of two equally plausible
        holes the robot takes the one it can reach over fresh ground, and the
        approach it drives sweeps whatever it crosses instead of re-treading the
        lane it came down. Mop-up is where a plain shortest-path search wastes
        the most: by definition it starts on ground the team has already swept.

        Teammates are holes in the map here. A robot whose target sits behind a
        parked teammate would otherwise replan onto the same blocked cell every
        time it is pushed back, which is exactly how two robots lock each other
        up for the rest of an episode.
        """
        g = self.graph
        # Teammates only. Masking every robot's cell and then clearing `cell`
        # "so you never block yourself" also clears a teammate standing in the
        # *same* cell as this robot — and two robots in one cell then plan the
        # identical route to the identical cell centre, which is a wedge neither
        # of them can leave. Excluding self by index instead keeps the teammate
        # visible whether or not it shares the cell.
        occupied = np.zeros(g.n_cells, dtype=bool)
        others = np.ones(len(positions), dtype=bool)
        others[r] = False
        occupied[g.cell_of(positions[others])] = True
        # The robot may always stand where it already stands, even if a teammate
        # is pressed into the same cell; otherwise the search has no source.
        passable = ~occupied
        passable[cell] = True

        cost = np.where(covered, 1.0 + self.revisit_penalty, 1.0)
        pending = g.free & ~covered & ~self._isolated & ~self.unreachable & ~occupied
        path = g.nearest_path(cell, pending & self.regions[r], cost,
                              allowed=self.regions[r] & passable)
        if path is None:
            # Own region done: help elsewhere, but only with the leftovers this
            # robot is closest to. Without that split every free robot converges
            # on the same hole and they spend the rest of the episode blocking
            # each other out of it.
            d = np.linalg.norm(g.centers[None, :, :] - positions[:, None, :], axis=2)
            path = g.nearest_path(cell, pending & (np.argmin(d, axis=0) == r), cost,
                                  allowed=passable)
        if path is None:
            # The split is a preference, not a rule. Holding to it when it leaves
            # this robot nothing to do parks a working robot next to cells no one
            # else has claimed — the nearest-robot assignment is only a tie-break,
            # so once it comes up empty the whole remainder is fair game.
            path = g.nearest_path(cell, pending, cost, allowed=passable)
        if path is None:
            # Last resort: teammates standing on the way are transient, so retry
            # without treating them as walls rather than declaring the map done.
            path = g.nearest_path(cell, g.free & ~covered & ~self._isolated
                                  & ~self.unreachable, cost)
        self._adopt(r, path, cell, positions[r], covered)

    def _skip_covered(self, r: int, cell: int, covered: np.ndarray,
                      pos: np.ndarray) -> bool:
        """Splice past a stretch of the tour a teammate has already covered.

        The tour is planned offline against an empty map, so it stays optimal
        only as long as nobody else touches it. A teammate spilling over a
        region border, or this robot's own earlier mop-up, can leave a long
        stretch of it already swept — and driving that stretch is a pass over
        covered ground for every cell of it.

        The remainder is *spliced*, not thrown away: only the covered prefix is
        dropped and a revisit-priced transit is stitched to the first cell still
        pending, so the boustrophedon structure of everything after it survives.
        Replanning outright would degrade the sweep into greedy
        nearest-uncovered, which is exactly the behaviour that costs revisits.

        Short stretches are left alone — under `skip_horizon` cells, driving
        through is cheaper than the turn the detour would cost.
        """
        tour = self.tours[r]
        i = int(self.idx[r])
        k = i
        while k < len(tour) and covered[tour[k]]:
            k += 1
        if k - i < self.skip_horizon:
            return False
        if k >= len(tour):              # nothing pending left on this tour
            self.idx[r] = len(tour)     # let the mop-up branch take over
            return False

        g = self.graph
        cost = np.where(covered, 1.0 + self.revisit_penalty, 1.0)
        _, came = g.weighted_paths(cell, cost)
        path = g.trace(came, tour[k])
        if not path:
            return False
        self._adopt(r, path + tour[k + 1:], cell, pos, covered)
        return True


# ---------------------------------------------------------------------------
# Rollout / dataset generation
# ---------------------------------------------------------------------------

def run_episode(env, expert: BoscoExpert, key, collect: bool = False):
    """One deterministic BOSCO episode. Returns (stats, obs, actions)."""
    import jax
    import jax.numpy as jnp

    step_fn = getattr(env, '_bosco_step', None)
    if step_fn is None:
        step_fn = jax.jit(env.step)
        obs_fn = jax.jit(env.get_obs)
        env._bosco_step, env._bosco_obs = step_fn, obs_fn
    obs_fn = env._bosco_obs

    state = env.reset(key)
    plan = expert.reset(np.asarray(state.robot_positions),
                        map_id=int(jax.device_get(state.map_id)))

    obs_buf, act_buf = [], []
    collisions = 0
    team_return = 0.0
    # Redundancy accounting. A "cell entry" is a step on which a robot's centre
    # crosses into a different cell; the entry is a *revisit* when that cell was
    # already covered before the step. Steps taken inside one cell are excluded
    # on purpose: they are the price of the 0.1 m/step kinematics, not a planning
    # defect, and counting them would drown the signal the planner controls.
    revisits = 0
    entries = 0
    prev_cell = expert.graph.cell_of(np.asarray(state.robot_positions))
    for t in range(env.max_steps):
        actions = expert.act(
            np.asarray(state.robot_positions),
            np.asarray(state.robot_headings),
            np.asarray(state.coverage_grid),
        )
        if collect:
            obs_buf.append(np.asarray(obs_fn(state)))
            act_buf.append(actions)

        covered_before = np.asarray(state.coverage_grid).ravel() > 0.5
        state, rewards, terminated, truncated = step_fn(state, jnp.asarray(actions))
        new_cell = expert.graph.cell_of(np.asarray(state.robot_positions))
        entered = new_cell != prev_cell
        entries += int(entered.sum())
        revisits += int(np.sum(entered & covered_before[new_cell]))
        prev_cell = new_cell
        # The env zeroes the reported linear velocity of any robot whose move was
        # rejected, so "commanded v > 0 but reported v == 0" is exactly the set of
        # robots charged the collision penalty on this step. Checking positions
        # after the step cannot see it: a blocked robot never moves into the wall.
        v_cmd = (actions[:, 0] + 1.0) * 0.5 * env.v_max
        collisions += int(np.sum((v_cmd > 1e-6)
                                 & (np.asarray(state.robot_velocities)[:, 0] <= 0.0)))
        team_return += float(np.sum(np.asarray(rewards)))
        if bool(terminated) or bool(truncated):
            break

    covered = float(np.sum(np.asarray(state.coverage_grid)))
    free_total = float(np.asarray(env.free_totals)[int(np.asarray(state.map_id))])
    stats = {
        'steps': t + 1,
        'coverage': covered / free_total,
        'covered_cells': covered,
        'collisions': collisions,
        'team_return': team_return,
        'region_sizes': plan['region_sizes'],
        'unreachable_cells': plan['unreachable_cells'],
        'revisits': revisits,
        'cell_entries': entries,
        # 1.0 means every cell entry discovered a new cell — a perfect,
        # revisit-free sweep. The ratio is the planner's own efficiency score.
        'sweep_efficiency': (entries - revisits) / max(entries, 1),
        'planned_revisits': plan['planned_revisits'],
    }
    obs = np.concatenate(obs_buf, axis=0) if collect else None
    acts = np.concatenate(act_buf, axis=0) if collect else None
    return stats, obs, acts


def main() -> None:
    ap = argparse.ArgumentParser(description='BOSCO deterministic coverage expert')
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--robots', type=int, default=3)
    # The env's own default of 500 steps is far short of a full sweep: three
    # robots need roughly 2100 steps to finish this map, most of it paid to the
    # 1 rad/s turn rate.
    ap.add_argument('--max-steps', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save', type=str, default=None,
                    help='write the behaviour-cloning dataset to this .npz')
    ap.add_argument('--humans', nargs='?', type=int, const=3, default=0, help='Number of humans')
    args = ap.parse_args()

    import jax

    from src.envs.coverage_vector_env import MultiRobotCoverageEnv

    env = MultiRobotCoverageEnv({
        'num_robots': args.robots,
        'max_steps': args.max_steps,
        'terminate_on_collision': False,
        'num_humans': args.humans,
    })
    expert = BoscoExpert(env)

    print(f"map: {env.grid_h}x{env.grid_w} cells, {int(env.free_totals[0])} coverable, "
          f"{int(expert._isolated.sum())} unreachable by the robot disc")

    all_obs, all_act, ep_id, ag_id = [], [], [], []
    coverages, steps, collisions, returns = [], [], [], []
    revisits, effs, planned = [], [], []
    t0 = time.time()
    for ep in range(args.episodes):
        key = jax.random.PRNGKey(args.seed + ep)
        stats, obs, acts = run_episode(env, expert, key, collect=args.save is not None)
        coverages.append(stats['coverage'])
        steps.append(stats['steps'])
        collisions.append(stats['collisions'])
        returns.append(stats['team_return'])
        revisits.append(stats['revisits'])
        effs.append(stats['sweep_efficiency'])
        planned.append(stats['planned_revisits'])
        print(f"ep {ep:3d} | steps {stats['steps']:5d} | coverage "
              f"{stats['coverage'] * 100:6.2f}% | collisions {stats['collisions']:4d} "
              f"| revisits {stats['revisits']:4d} (plan {stats['planned_revisits']:4d}) "
              f"| eff {stats['sweep_efficiency'] * 100:5.1f}% "
              f"| return {stats['team_return']:8.1f} "
              f"| regions {stats['region_sizes'].tolist()}")
        if args.save is not None:
            n_steps = obs.shape[0] // env.num_robots
            all_obs.append(obs)
            all_act.append(acts)
            ep_id.append(np.full(obs.shape[0], ep, dtype=np.int32))
            ag_id.append(np.tile(np.arange(env.num_robots, dtype=np.int32), n_steps))

    dt = time.time() - t0
    print(f"\n{args.episodes} episodes in {dt:.1f}s | "
          f"coverage {np.mean(coverages) * 100:.2f}% +- {np.std(coverages) * 100:.2f} | "
          f"steps {np.mean(steps):.0f} | collisions {np.mean(collisions):.1f} | "
          f"revisits {np.mean(revisits):.1f} (plan {np.mean(planned):.1f}) | "
          f"sweep efficiency {np.mean(effs) * 100:.1f}% | "
          f"return {np.mean(returns):.1f}")

    if args.save is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        np.savez_compressed(
            args.save,
            obs=np.concatenate(all_obs).astype(np.float32),
            actions=np.concatenate(all_act).astype(np.float32),
            episode=np.concatenate(ep_id),
            agent=np.concatenate(ag_id),
        )
        total = sum(o.shape[0] for o in all_obs)
        print(f"saved {total} (obs, action) pairs -> {args.save}")


if __name__ == '__main__':
    main()
