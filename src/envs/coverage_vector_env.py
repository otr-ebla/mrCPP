"""
JAX multi-robot indoor coverage environment.

The environment is a pure function of an immutable `EnvState` pytree, so
`reset` / `step` / `get_obs` can be jitted, vmapped over parallel environments,
and scanned over time without leaving the accelerator.

Observation per robot (o_i):
    [v_i, w_i,  rel_teammates (k*2),  rel_humans (m*2),  lidar (n_rays),
     local_coverage (L*L, optional)]

Lidar detects both walls and other robots as obstacles, giving a direct
proximity signal identical to wall avoidance so agents can learn inter-robot
collision avoidance with the same mechanism.

Centralised critic state (s):
    [robot_positions (N*2), robot_headings (N), human_positions (M*2),
     coverage_grid (H*W)]

Actions: (N, 2) in [-1, 1].
    action[:,0] -> linear  velocity in [0, v_max]   (remapped from [-1,1])
    action[:,1] -> angular velocity in [-omega_max, omega_max]

Reward (per-agent):
    team part  :  alpha*dn  -  beta*rho  -  tau  +  room_bonus
    agent part :  -  kappa*collision_i  -  psi*proximity_i
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from .map_layouts import IndoorMapLayout

_TWO_PI = 2.0 * np.pi
_BIG = 1.0e9


@struct.dataclass
class EnvState:
    """Complete environment state. All fields are JAX arrays (a valid pytree)."""

    robot_positions:  jax.Array   # (N, 2)   float32
    robot_headings:   jax.Array   # (N,)     float32
    robot_velocities: jax.Array   # (N, 2)   float32  — [v, omega] commanded
    human_positions:  jax.Array   # (M, 2)   float32
    coverage_grid:    jax.Array   # (H, W)   float32  — 0.0 / 1.0
    room_completed:   jax.Array   # (R,)     bool
    robot_alive:      jax.Array   # (N,)     bool
    step_count:       jax.Array   # ()       int32
    key:              jax.Array   # PRNG key carried for auto-reset


class MultiRobotCoverageEnv:
    """Static configuration plus pure `reset` / `step` / observation functions.

    Instances hold only compile-time constants, so they can be closed over by
    jitted functions without being traced.
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}

        # -- Environment parameters --
        self.num_robots      = int(cfg.get('num_robots',      3))
        self.num_humans      = int(cfg.get('num_humans',      0))
        self.k_teammates     = int(cfg.get('k_teammates',     2))
        self.m_humans        = int(cfg.get('m_humans',        1))
        self.n_rays          = int(cfg.get('n_rays',          36))
        self.max_lidar_range = float(cfg.get('max_lidar_range', 5.0))
        self.cell_size       = float(cfg.get('cell_size',       0.5))
        self.sensing_radius  = float(cfg.get('sensing_radius',  5.0))
        self.robot_radius    = float(cfg.get('robot_radius',    0.20))
        self.dt              = float(cfg.get('dt',              0.1))
        self.max_steps       = int(cfg.get('max_steps',       500))
        self.v_max           = float(cfg.get('v_max',           1.0))
        self.omega_max       = float(cfg.get('omega_max',       1.0))

        # -- Collision handling --
        # When True, a robot that collides (wall or robot-robot) dies: it
        # freezes permanently at its last position for the rest of the episode
        # and stops receiving any reward from that point on. Other (still
        # alive) robots are unaffected and keep the episode running until
        # truncation or until every robot has died.
        self.terminate_on_collision = bool(cfg.get('terminate_on_collision', False))

        # -- Local coverage patch in actor observation --
        self.use_local_coverage_obs = bool(cfg.get('use_local_coverage_obs', True))
        self.local_coverage_size    = int(cfg.get('local_coverage_size',    7))

        # -- Reward weights --
        self.alpha = float(cfg.get('alpha', 10.0))
        self.beta  = float(cfg.get('beta',  0.1))
        self.kappa = float(cfg.get('kappa', 20.0))
        self.tau   = float(cfg.get('tau',   0.01))

        # -- Proximity penalty: soft repulsion before hard collision --
        self.psi        = float(cfg.get('psi',              1.5))
        self._safe_dist = float(cfg.get('safe_dist_factor', 5.0)) * self.robot_radius

        # -- Room completion bonus --
        self.room_completion_bonus     = float(cfg.get('room_completion_bonus',     50.0))
        self.room_completion_threshold = float(cfg.get('room_completion_threshold', 0.85))

        # -- Map --
        self.map_layout = IndoorMapLayout()
        walls_np = np.asarray(self.map_layout.get_walls(), dtype=np.float32)
        self.walls = jnp.asarray(walls_np)
        self._wall_x0 = jnp.asarray(walls_np[:, 0])
        self._wall_y0 = jnp.asarray(walls_np[:, 1])
        self._wall_x1 = jnp.asarray(walls_np[:, 2])
        self._wall_y1 = jnp.asarray(walls_np[:, 3])

        # -- Grid --
        self.grid_w = int(np.ceil(self.map_layout.width  / self.cell_size))
        self.grid_h = int(np.ceil(self.map_layout.height / self.cell_size))

        # -- Room definitions: 5 bounding boxes matching the map layout --
        outer_t = 0.20
        inner_t = 0.15
        x_left  = 4.0
        x_right = 6.0
        W, H    = self.map_layout.width, self.map_layout.height
        self._rooms = [
            (outer_t,             outer_t,          x_left  - inner_t/2, 4.0 - inner_t/2),
            (outer_t,             4.0 + inner_t/2,  x_left  - inner_t/2, H   - outer_t),
            (x_left  + inner_t/2, outer_t,          x_right - inner_t/2, H   - outer_t),
            (x_right + inner_t/2, outer_t,          W       - outer_t,   4.0 - inner_t/2),
            (x_right + inner_t/2, 4.0 + inner_t/2,  W       - outer_t,   H   - outer_t),
        ]
        xs = np.arange(self.grid_w) * self.cell_size + self.cell_size * 0.5
        ys = np.arange(self.grid_h) * self.cell_size + self.cell_size * 0.5
        xv, yv = np.meshgrid(xs, ys)
        masks_np = np.stack([
            (xv >= rx0) & (xv < rx1) & (yv >= ry0) & (yv < ry1)
            for rx0, ry0, rx1, ry1 in self._rooms
        ]).astype(np.float32)                      # (R, H, W)
        self.num_rooms   = masks_np.shape[0]
        self.room_masks  = jnp.asarray(masks_np)
        self.room_totals = jnp.asarray(masks_np.sum(axis=(1, 2)))

        # -- Derived dims --
        local_cov_dim  = self.local_coverage_size ** 2 if self.use_local_coverage_obs else 0
        self.obs_dim   = (2 + self.k_teammates * 2 + self.m_humans * 2
                          + self.n_rays + local_cov_dim)
        self.state_dim = (self.num_robots * 3
                          + self.num_humans * 2
                          + self.grid_w * self.grid_h)
        self.action_dim = 2

        # -- Lidar ray angles (relative to robot heading) --
        self._ray_angles = jnp.asarray(
            np.linspace(0.0, _TWO_PI, self.n_rays, endpoint=False, dtype=np.float32)
        )

        # -- Spawn candidates --
        self._spawn_candidates = jnp.asarray(self._precompute_spawn_candidates())
        self._num_candidates   = int(self._spawn_candidates.shape[0])
        if self._num_candidates < self.num_robots:
            raise RuntimeError(
                f"Not enough spawn candidates ({self._num_candidates}) "
                f"for {self.num_robots} robots."
            )
        # Candidates sit on a `cell_size` lattice, so any two distinct candidates
        # are already at least `cell_size` apart. When that exceeds the required
        # clearance a random subset is always valid and the greedy scan is skipped.
        self._spawn_clearance = 2.0 * self.robot_radius + 0.05
        self._spawn_needs_greedy = self.cell_size < self._spawn_clearance

        # -- Cached teammate/human slot counts --
        self._k_eff = min(self.k_teammates, max(self.num_robots - 1, 0))
        self._m_eff = min(self.m_humans, self.num_humans)

    # ------------------------------------------------------------------
    # Static precomputation (numpy, runs once at construction)
    # ------------------------------------------------------------------

    def _precompute_spawn_candidates(self) -> np.ndarray:
        xs = np.arange(self.cell_size, self.map_layout.width,  self.cell_size)
        ys = np.arange(self.cell_size, self.map_layout.height, self.cell_size)
        xv, yv = np.meshgrid(xs, ys)
        cands = np.stack([xv.ravel(), yv.ravel()], axis=1).astype(np.float32)

        walls = np.asarray(self.map_layout.get_walls(), dtype=np.float32)
        cx = np.clip(cands[:, 0:1], walls[None, :, 0], walls[None, :, 2])
        cy = np.clip(cands[:, 1:2], walls[None, :, 1], walls[None, :, 3])
        d2 = (cands[:, 0:1] - cx) ** 2 + (cands[:, 1:2] - cy) ** 2
        free = ~np.any(d2 < self.robot_radius ** 2, axis=1)
        return cands[free]

    # ------------------------------------------------------------------
    # Geometry helpers (vectorised over robots)
    # ------------------------------------------------------------------

    def _wall_collision(self, pos: jax.Array) -> jax.Array:
        """pos: (N, 2) -> (N,) bool"""
        cx = jnp.clip(pos[:, 0:1], self._wall_x0[None, :], self._wall_x1[None, :])
        cy = jnp.clip(pos[:, 1:2], self._wall_y0[None, :], self._wall_y1[None, :])
        d2 = (pos[:, 0:1] - cx) ** 2 + (pos[:, 1:2] - cy) ** 2
        return jnp.any(d2 < self.robot_radius ** 2, axis=1)

    @staticmethod
    def _pairwise_sq_dist(pos: jax.Array) -> jax.Array:
        """pos: (N, 2) -> (N, N) squared distances with +inf on the diagonal."""
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = jnp.sum(diff * diff, axis=-1)
        n = pos.shape[0]
        return d2 + jnp.eye(n, dtype=d2.dtype) * _BIG

    def _pos_to_cell(self, pos: jax.Array) -> tuple[jax.Array, jax.Array]:
        """pos: (N, 2) -> (cols, rows) int32, each (N,)"""
        col = jnp.clip(jnp.floor(pos[:, 0] / self.cell_size), 0, self.grid_w - 1)
        row = jnp.clip(jnp.floor(pos[:, 1] / self.cell_size), 0, self.grid_h - 1)
        return col.astype(jnp.int32), row.astype(jnp.int32)

    # ------------------------------------------------------------------
    # Physics — exact differential-drive integration
    # ------------------------------------------------------------------

    def _diff_drive(
        self, pos: jax.Array, heading: jax.Array, v: jax.Array, omega: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        straight = jnp.abs(omega) < 1e-6
        # Guard the division so the unused turning branch never produces NaN,
        # which would otherwise poison the jnp.where result.
        omega_safe = jnp.where(straight, 1.0, omega)

        turn_heading = heading + omega * self.dt
        radius       = v / omega_safe

        pos_straight = pos + jnp.stack(
            [v * jnp.cos(heading), v * jnp.sin(heading)], axis=-1
        ) * self.dt
        pos_turn = pos + radius[:, None] * jnp.stack([
            jnp.sin(turn_heading) - jnp.sin(heading),
            jnp.cos(heading) - jnp.cos(turn_heading),
        ], axis=-1)

        new_pos     = jnp.where(straight[:, None], pos_straight, pos_turn)
        new_heading = jnp.where(straight, heading, turn_heading)
        return new_pos, jnp.mod(new_heading, _TWO_PI)

    # ------------------------------------------------------------------
    # Sensing — vectorised ray-AABB (walls) + ray-sphere (robots)
    # ------------------------------------------------------------------

    def _cast_lidar_single(
        self, pos: jax.Array, heading: jax.Array,
        all_pos: jax.Array, other_mask: jax.Array,
    ) -> jax.Array:
        """Distances for one robot, normalised to [0, 1]. Returns (n_rays,).

        all_pos / other_mask cover every robot; `other_mask` excludes self so a
        robot never detects its own body.
        """
        angles = heading + self._ray_angles
        dx = jnp.cos(angles)
        dy = jnp.sin(angles)

        # --- Wall ray-AABB slab intersection ---
        # Rays parallel to an axis give +-inf (or NaN when the origin lies exactly
        # on the slab plane); both are rejected by the `valid` mask below.
        tx0 = (self._wall_x0[None, :] - pos[0]) / dx[:, None]      # (R, W)
        tx1 = (self._wall_x1[None, :] - pos[0]) / dx[:, None]
        ty0 = (self._wall_y0[None, :] - pos[1]) / dy[:, None]
        ty1 = (self._wall_y1[None, :] - pos[1]) / dy[:, None]

        t_near = jnp.maximum(jnp.minimum(tx0, tx1), jnp.minimum(ty0, ty1))
        t_far  = jnp.minimum(jnp.maximum(tx0, tx1), jnp.maximum(ty0, ty1))

        valid = (t_near <= t_far + 1e-9) & (t_far > 1e-6)
        t_hit = jnp.where(valid, jnp.maximum(t_near, 1e-6), jnp.inf)
        t_min = jnp.min(t_hit, axis=1)                              # (R,)

        # --- Robot ray-sphere intersection ---
        # Solve |pos + t*dir - center|^2 = r^2  ->  t^2 + b*t + c = 0
        dirs = jnp.stack([dx, dy], axis=1)                          # (R, 2)
        w    = pos[None, :] - all_pos                               # (K, 2)
        b    = 2.0 * (w @ dirs.T)                                   # (K, R)
        c    = jnp.sum(w * w, axis=1, keepdims=True) - self.robot_radius ** 2
        disc = b ** 2 - 4.0 * c
        t_r  = (-b - jnp.sqrt(jnp.maximum(disc, 0.0))) * 0.5
        t_r  = jnp.where((disc >= 0) & (t_r > 1e-6) & other_mask[:, None], t_r, jnp.inf)
        t_min = jnp.minimum(t_min, jnp.min(t_r, axis=0))

        dist = jnp.clip(t_min, 0.0, self.max_lidar_range)
        return dist / self.max_lidar_range

    def _cast_lidar_all(self, positions: jax.Array, headings: jax.Array) -> jax.Array:
        """(N, 2), (N,) -> (N, n_rays)"""
        n = self.num_robots
        not_self = ~jnp.eye(n, dtype=bool)
        return jax.vmap(
            self._cast_lidar_single, in_axes=(0, 0, None, 0)
        )(positions, headings, positions, not_self)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _sample_spawns(self, key: jax.Array) -> jax.Array:
        """Pick `num_robots` mutually clear spawn positions. Returns (N, 2)."""
        cands = self._spawn_candidates
        if not self._spawn_needs_greedy:
            idx = jax.random.permutation(key, self._num_candidates)[: self.num_robots]
            return cands[idx]

        # Greedy pass over a random permutation, mirroring the sequential
        # accept-if-clear rule; only needed when the lattice spacing is finer
        # than the required inter-robot clearance.
        shuffled = jax.random.permutation(key, cands, axis=0)
        slots = jnp.arange(self.num_robots)

        def body(carry, cand):
            chosen, count = carry
            d = jnp.sqrt(jnp.sum((chosen - cand[None, :]) ** 2, axis=1))
            active = slots < count
            clear = jnp.all(jnp.where(active, d >= self._spawn_clearance, True))
            take = clear & (count < self.num_robots)
            write = take & (slots == count)
            chosen = jnp.where(write[:, None], cand[None, :], chosen)
            return (chosen, count + take.astype(count.dtype)), None

        init = (jnp.zeros((self.num_robots, 2), jnp.float32), jnp.int32(0))
        (chosen, _), _ = jax.lax.scan(body, init, shuffled)
        return chosen

    def reset(self, key: jax.Array) -> EnvState:
        key, spawn_key, hdg_key = jax.random.split(key, 3)
        return EnvState(
            robot_positions  = self._sample_spawns(spawn_key),
            robot_headings   = jax.random.uniform(
                hdg_key, (self.num_robots,), minval=0.0, maxval=_TWO_PI
            ),
            robot_velocities = jnp.zeros((self.num_robots, 2), jnp.float32),
            human_positions  = jnp.zeros((self.num_humans, 2), jnp.float32),
            coverage_grid    = jnp.zeros((self.grid_h, self.grid_w), jnp.float32),
            room_completed   = jnp.zeros((self.num_rooms,), bool),
            robot_alive      = jnp.ones((self.num_robots,), bool),
            step_count       = jnp.int32(0),
            key              = key,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, state: EnvState, joint_actions: jax.Array
    ) -> tuple[EnvState, jax.Array, jax.Array, jax.Array]:
        """
        joint_actions : (N, 2) in [-1, 1]
        Returns (next_state, rewards (N,), terminated (), truncated ())

        Rewards are per-agent:
          - team coverage reward shared equally across all currently-alive robots
          - collision kappa penalty applied only to the robot(s) that collided
          - proximity psi penalty applied per robot proportional to nearness

        Collision handling is per-robot, not per-episode: when
        terminate_on_collision is True, a robot that collides dies (freezes
        permanently and earns no further reward) while every other robot keeps
        running. The episode-level `terminated` flag only fires once *all*
        robots have died.
        """
        alive     = state.robot_alive
        v_cmds    = (joint_actions[:, 0] + 1.0) * 0.5 * self.v_max
        omega_cmds = joint_actions[:, 1] * self.omega_max
        prev_grid = state.coverage_grid

        # --- Tentative moves: each alive robot independent, dead robots frozen ---
        cand_pos, cand_hdg = self._diff_drive(
            state.robot_positions, state.robot_headings, v_cmds, omega_cmds
        )
        wall_hit = self._wall_collision(cand_pos) & alive

        blocked  = wall_hit | ~alive
        next_pos = jnp.where(blocked[:, None], state.robot_positions, cand_pos)
        next_hdg = jnp.where(blocked, state.robot_headings, cand_hdg)

        # --- Pairwise robot-robot collision on tentative non-wall positions ---
        # Dead robots remain solid obstacles at their frozen position but cannot
        # themselves newly collide again.
        d2       = self._pairwise_sq_dist(next_pos)
        pair_ok  = (~wall_hit)[:, None] & (~wall_hit)[None, :]
        close    = (d2 < (2.0 * self.robot_radius) ** 2) & pair_ok
        robot_hit = jnp.any(close, axis=1) & alive

        collided = (wall_hit | robot_hit) & alive
        alive_next = alive & ~collided if self.terminate_on_collision else alive

        # --- Apply movement to alive robots that did not collide ---
        moved   = alive & ~collided
        new_pos = jnp.where(moved[:, None], next_pos, state.robot_positions)
        new_hdg = jnp.where(moved, next_hdg, state.robot_headings)

        new_vel = jnp.stack([
            jnp.where(alive, v_cmds,     0.0),
            jnp.where(alive, omega_cmds, 0.0),
        ], axis=-1)

        # --- Coverage update: only for robots that successfully moved ---
        cols, rows = self._pos_to_cell(new_pos)
        already    = prev_grid[rows, cols] > 0.0
        delta_n    = jnp.sum(moved & ~already)
        rho        = jnp.sum(moved & already)
        # `.max` leaves cells of ineligible robots untouched and is safe under
        # duplicate indices (several robots landing on the same cell).
        new_grid = prev_grid.at[rows, cols].max(jnp.where(moved, 1.0, 0.0))

        # --- Room completion bonus: fires once per room at the threshold ---
        covered   = jnp.sum(new_grid[None, :, :] * self.room_masks, axis=(1, 2))
        ratio     = covered / jnp.maximum(self.room_totals, 1.0)
        newly     = (~state.room_completed) & (self.room_totals > 0) \
                    & (ratio >= self.room_completion_threshold)
        room_bonus = self.room_completion_bonus * jnp.sum(newly)
        room_completed = state.room_completed | newly

        # --- Proximity penalty: soft repulsion before hard collision ---
        # Diagonal carries +inf from _pairwise_sq_dist, so self-pairs never fire.
        dist     = jnp.sqrt(self._pairwise_sq_dist(new_pos))
        pen      = self.psi * (1.0 - dist / self._safe_dist) * (dist < self._safe_dist)
        prox_pen = jnp.sum(pen, axis=1) * alive

        # --- Compose per-agent rewards ---
        # Robots already dead at the start of this step get no reward at all; a
        # robot dying on this step still receives its final team_r/coll_pen.
        team_r   = self.alpha * delta_n - self.beta * rho - self.tau + room_bonus
        coll_pen = self.kappa * collided
        rewards  = jnp.where(alive, team_r - coll_pen - prox_pen, 0.0).astype(jnp.float32)

        step_count = state.step_count + 1
        truncated  = step_count >= self.max_steps
        terminated = (~jnp.any(alive_next) if self.terminate_on_collision
                      else jnp.bool_(False))

        next_state = state.replace(
            robot_positions  = new_pos,
            robot_headings   = new_hdg,
            robot_velocities = new_vel,
            coverage_grid    = new_grid,
            room_completed   = room_completed,
            robot_alive      = alive_next,
            step_count       = step_count,
        )
        return next_state, rewards, terminated, truncated

    # ------------------------------------------------------------------
    # Observation / state builders
    # ------------------------------------------------------------------

    def get_obs(self, state: EnvState) -> jax.Array:
        """(N, obs_dim) float32"""
        n     = self.num_robots
        parts = [state.robot_velocities]

        # --- k nearest teammates within the sensing radius ---
        rel = state.robot_positions[None, :, :] - state.robot_positions[:, None, :]
        if self._k_eff > 0:
            d2 = self._pairwise_sq_dist(state.robot_positions)
            # top_k on the negated distance yields the nearest neighbours in order.
            neg_d2, idx = jax.lax.top_k(-d2, self._k_eff)              # (N, k_eff)
            near = jnp.take_along_axis(rel, idx[:, :, None], axis=1)   # (N, k_eff, 2)
            visible = (-neg_d2) <= self.sensing_radius ** 2
            near = jnp.where(visible[:, :, None], near, 0.0)
            parts.append(near.reshape(n, self._k_eff * 2))
        pad_k = self.k_teammates - self._k_eff
        if pad_k > 0:
            parts.append(jnp.zeros((n, pad_k * 2), jnp.float32))

        # --- m nearest humans within the sensing radius ---
        if self._m_eff > 0:
            rel_h = state.human_positions[None, :, :] - state.robot_positions[:, None, :]
            dh2   = jnp.sum(rel_h * rel_h, axis=-1)                    # (N, M)
            neg_dh2, idx_h = jax.lax.top_k(-dh2, self._m_eff)
            near_h = jnp.take_along_axis(rel_h, idx_h[:, :, None], axis=1)
            vis_h  = (-neg_dh2) <= self.sensing_radius ** 2
            near_h = jnp.where(vis_h[:, :, None], near_h, 0.0)
            parts.append(near_h.reshape(n, self._m_eff * 2))
        pad_m = self.m_humans - self._m_eff
        if pad_m > 0:
            parts.append(jnp.zeros((n, pad_m * 2), jnp.float32))

        # --- Lidar sees walls AND other robots as obstacles ---
        parts.append(self._cast_lidar_all(state.robot_positions, state.robot_headings))

        # --- Local coverage patch centred on each robot ---
        if self.use_local_coverage_obs:
            r = self.local_coverage_size // 2
            s = self.local_coverage_size
            # Out-of-map padding reads as covered so robots are not drawn outside.
            padded = jnp.pad(state.coverage_grid, r, constant_values=1.0)
            cols, rows = self._pos_to_cell(state.robot_positions)
            patches = jax.vmap(
                lambda rw, cl: jax.lax.dynamic_slice(padded, (rw, cl), (s, s))
            )(rows, cols)
            parts.append(patches.reshape(n, s * s))

        return jnp.concatenate(parts, axis=1).astype(jnp.float32)

    def get_global_state(self, state: EnvState) -> jax.Array:
        """Centralised critic state: robot poses + human positions + coverage grid."""
        return jnp.concatenate([
            state.robot_positions.ravel(),
            state.robot_headings,
            state.human_positions.ravel(),
            state.coverage_grid.ravel(),
        ]).astype(jnp.float32)

    def get_info(self, state: EnvState) -> dict:
        """Diagnostics. All values are arrays so the dict survives jit/vmap."""
        covered = jnp.sum(state.coverage_grid)
        total   = float(self.grid_w * self.grid_h)
        room_cov = jnp.sum(state.coverage_grid[None, :, :] * self.room_masks, axis=(1, 2))
        info = {
            'coverage_ratio':   covered / total,
            'covered_cells':    covered,
            'total_cells':      jnp.float32(total),
            'step':             state.step_count,
            'robots_alive':     state.robot_alive,
            'num_robots_alive': jnp.sum(state.robot_alive),
        }
        for ri in range(self.num_rooms):
            info[f'room_{ri}_ratio'] = room_cov[ri] / jnp.maximum(self.room_totals[ri], 1.0)
        return info
