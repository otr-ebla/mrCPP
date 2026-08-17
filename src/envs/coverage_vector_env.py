"""
JAX multi-robot indoor coverage environment.

The environment is a pure function of an immutable `EnvState` pytree, so
`reset` / `step` / `get_obs` can be jitted, vmapped over parallel environments,
and scanned over time without leaving the accelerator.

Observation per robot (o_i), decentralised execution:
    [v_i, w_i,  rel_teammates (k*2),  rel_humans (m*2),  lidar (n_rays),
     local_coverage (L*L, optional)]

The binary local-coverage patch is deliberately the tail of the vector: it is
the only non-continuous block, so callers can normalise `obs[..., :norm_dim]`
with a single slice and leave the binary cells untouched.

Lidar detects both walls and other robots as obstacles, giving a direct
proximity signal identical to wall avoidance so agents can learn inter-robot
collision avoidance with the same mechanism.

Centralised critic state (`GlobalState`), training only:
    coverage   (H, W)      team coverage grid
    occupancy  (N, H, W)   one-hot cell of each robot
    kinematics (N, 6)      normalised pose + commanded velocities
`critic_inputs` expands it into the per-agent multi-channel tensor
[walls, coverage, self, teammates] plus the joint kinematic vector.

Actions: (N, 2) in [-1, 1].
    action[:,0] -> linear  velocity in [0, v_max]   (remapped from [-1,1])
    action[:,1] -> angular velocity in [-omega_max, omega_max]

Reward (difference reward, per agent):
    R_i = alpha*new_cell_i - beta*redundant_i - tau - kappa*collided_i
          - psi*proximity_i + team_bonus
Only the robot that flips a cell from unvisited to visited is paid for it, so a
robot that stands still while the team works collects nothing. `team_bonus`
carries the two cooperative events (room completion, full coverage) that no
single agent can be credited for.
"""

from __future__ import annotations

from collections import deque

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
    # Diagnostics for the step that produced this state, kept in the state so
    # `get_info` can report them without changing the `step` signature. float32
    # rather than bool: jax-metal 0.1.1 drops bool leaves out of `lax.scan`.
    wall_hits:        jax.Array   # (N,)     float32  — 0.0 / 1.0
    robot_hits:       jax.Array   # (N,)     float32  — 0.0 / 1.0


@struct.dataclass
class GlobalState:
    """Centralised critic state, kept as small spatial tensors.

    The static wall map is *not* stored here: it is identical in every
    transition, so `critic_inputs` broadcasts it from the environment instead of
    paying for it once per step in the rollout buffer.
    """

    coverage:   jax.Array   # (H, W)     float32
    occupancy:  jax.Array   # (N, H, W)  float32, one-hot per robot
    kinematics: jax.Array   # (N, 6)     float32, normalised


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
        # When True, a collision is fatal for the whole team: as soon as any
        # robot hits a wall or another robot the episode terminates for all
        # robots on that same step. When False, collisions are soft — the
        # offender is blocked for one step, pays the kappa penalty and the
        # episode continues. Soft is the default: under hard termination the
        # early random policy ends every episode after a few dozen steps, so the
        # team never collects the experience needed to reach the far rooms.
        self.terminate_on_collision = bool(cfg.get('terminate_on_collision', False))

        # -- Local coverage patch in actor observation --
        self.use_local_coverage_obs = bool(cfg.get('use_local_coverage_obs', True))
        self.local_coverage_size    = int(cfg.get('local_coverage_size',    5))

        # -- Reward weights --
        self.alpha = float(cfg.get('alpha', 10.0))
        self.beta  = float(cfg.get('beta',   0.005))
        self.kappa = float(cfg.get('kappa', 20.0))
        self.tau   = float(cfg.get('tau',    0.05))
        if self.beta >= self.tau:
            raise ValueError(
                f"redundancy weight beta={self.beta} must be strictly smaller than "
                f"the time penalty tau={self.tau}; otherwise standing still is "
                "cheaper than driving across already-covered cells."
            )

        # -- Proximity penalty: soft repulsion before hard collision --
        self.psi        = float(cfg.get('psi',              1.5))
        self._safe_dist = float(cfg.get('safe_dist_factor', 5.0)) * self.robot_radius

        # -- Cooperative bonuses --
        self.room_completion_bonus     = float(cfg.get('room_completion_bonus',     50.0))
        self.room_completion_threshold = float(cfg.get('room_completion_threshold', 0.85))
        self.completion_bonus          = float(cfg.get('completion_bonus',         200.0))

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
        self.num_cells = self.grid_h * self.grid_w

        # -- Free space: the only cells that count towards coverage --
        # Computed once here (the map is static) rather than per reset.
        free_np = self._compute_free_mask()                      # (H, W) float32
        self.free_mask_np = free_np                              # host copy, for rendering
        self.free_mask   = jnp.asarray(free_np)
        self._free_flat  = jnp.asarray(free_np.ravel())
        self.free_total  = float(free_np.sum())
        # Obstacle channel for the critic and "nothing to do here" marker for the
        # actor's local patch.
        self.wall_grid   = jnp.asarray(1.0 - free_np)
        if self.free_total < self.num_robots:
            raise RuntimeError("Free space is too small for the requested robot count.")

        # -- Room definitions: 5 bounding boxes matching the map layout --
        # Taken from the layout rather than restated, so a change of wall
        # thickness cannot silently desynchronise the room boxes from the walls.
        outer_t = self.map_layout.outer_t
        inner_t = self.map_layout.inner_t
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
        xv, yv = self._cell_centers()
        # Intersecting each room box with the free mask is what makes the 0.85
        # threshold reachable: counting walled cells in the denominator caps
        # every room below it.
        masks_np = np.stack([
            ((xv >= rx0) & (xv < rx1) & (yv >= ry0) & (yv < ry1)) & (free_np > 0.0)
            for rx0, ry0, rx1, ry1 in self._rooms
        ]).astype(np.float32)                      # (R, H, W)
        self.num_rooms   = masks_np.shape[0]
        self.room_masks  = jnp.asarray(masks_np)
        self.room_totals = jnp.asarray(masks_np.sum(axis=(1, 2)))

        # -- Derived dims --
        # Layout: [continuous block | lidar | binary patch]. `norm_dim` is the
        # length of the prefix that observation normalisation may touch.
        self.obs_vec_dim   = 2 + self.k_teammates * 2 + self.m_humans * 2
        self.patch_dim     = (self.local_coverage_size ** 2
                              if self.use_local_coverage_obs else 0)
        self.norm_dim      = self.obs_vec_dim + self.n_rays
        self.obs_dim       = self.norm_dim + self.patch_dim
        self.action_dim    = 2
        # Critic input shapes: [walls, coverage, self, teammates] per agent.
        self.critic_channels = 4
        self.critic_vec_dim  = 6 + 6 * self.num_robots

        # -- Lidar ray angles (relative to robot heading) --
        self._ray_angles = jnp.asarray(
            np.linspace(0.0, _TWO_PI, self.n_rays, endpoint=False, dtype=np.float32)
        )

        # -- Spawn candidates: the free cell centres, so a robot can never start
        #    inside a pocket that is walled off from the coverable region --
        centers = np.stack([xv.ravel(), yv.ravel()], axis=1).astype(np.float32)
        self._spawn_candidates = jnp.asarray(centers[free_np.ravel() > 0.0])
        self._num_candidates   = int(self._spawn_candidates.shape[0])
        # Candidates sit on a `cell_size` lattice, so any two distinct candidates
        # are already at least `cell_size` apart. When that exceeds the required
        # clearance a random subset is always valid and the greedy scan is skipped.
        self._spawn_clearance = 2.0 * self.robot_radius + 0.05
        self._spawn_needs_greedy = self.cell_size < self._spawn_clearance

        # -- Cached teammate/human slot counts --
        self._k_eff = min(self.k_teammates, max(self.num_robots - 1, 0))
        self._m_eff = min(self.m_humans, self.num_humans)

        self._robot_ids = jnp.arange(self.num_robots, dtype=jnp.int32)

    # ------------------------------------------------------------------
    # Static precomputation (numpy, runs once at construction)
    # ------------------------------------------------------------------

    def _cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        xs = (np.arange(self.grid_w) + 0.5) * self.cell_size
        ys = (np.arange(self.grid_h) + 0.5) * self.cell_size
        return np.meshgrid(xs, ys)

    def _compute_free_mask(self) -> np.ndarray:
        """Cells a robot centre can occupy AND reach. Returns (H, W) float32.

        A cell is coverable only if the robot body fits at its centre and the
        cell is connected to the rest of the map: a geometrically clear cell
        sealed behind walls would otherwise sit in the denominator forever and
        make 100% coverage unattainable.
        """
        xv, yv = self._cell_centers()
        centers = np.stack([xv.ravel(), yv.ravel()], axis=1).astype(np.float32)

        walls = np.asarray(self.map_layout.get_walls(), dtype=np.float32)
        cx = np.clip(centers[:, 0:1], walls[None, :, 0], walls[None, :, 2])
        cy = np.clip(centers[:, 1:2], walls[None, :, 1], walls[None, :, 3])
        d2 = (centers[:, 0:1] - cx) ** 2 + (centers[:, 1:2] - cy) ** 2
        clear = (~np.any(d2 < self.robot_radius ** 2, axis=1))
        clear = clear.reshape(self.grid_h, self.grid_w)

        # Largest 4-connected component of `clear`. Runs once at construction on
        # a few hundred cells, so a plain BFS is cheaper than pulling in scipy.
        seen = np.zeros_like(clear)
        best = np.zeros_like(clear)
        best_size = 0
        for r0 in range(self.grid_h):
            for c0 in range(self.grid_w):
                if not clear[r0, c0] or seen[r0, c0]:
                    continue
                comp = np.zeros_like(clear)
                queue = deque([(r0, c0)])
                seen[r0, c0] = True
                size = 0
                while queue:
                    r, c = queue.popleft()
                    comp[r, c] = True
                    size += 1
                    for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                        if (0 <= nr < self.grid_h and 0 <= nc < self.grid_w
                                and clear[nr, nc] and not seen[nr, nc]):
                            seen[nr, nc] = True
                            queue.append((nr, nc))
                if size > best_size:
                    best, best_size = comp, size
        return best.astype(np.float32)

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
            wall_hits        = jnp.zeros((self.num_robots,), jnp.float32),
            robot_hits       = jnp.zeros((self.num_robots,), jnp.float32),
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

        Rewards follow the difference-reward principle: discovery, redundancy,
        collision and proximity are all charged to the individual robot, so a
        robot that lets the team do the work earns only the time penalty. The
        two genuinely cooperative events (room completion, full coverage) are
        the only shared terms.

        Collision handling is team-wide when terminate_on_collision is True:
        a single robot hitting a wall or another robot terminates the episode
        for everyone on that step. The offender is still marked dead in the
        terminal state so the visualiser can identify it, but no robot gets to
        act again. With the flag False, collisions are soft: the offender keeps
        its position for that step, has its linear velocity zeroed, pays kappa,
        and the episode runs on to full coverage or truncation.

        Rotation is never blocked. The body is a disc, so turning on the spot
        cannot create a new overlap, and letting a blocked robot still apply its
        commanded omega is what makes a soft collision recoverable: a robot
        pressed against a wall can turn away instead of grinding into it for the
        rest of the episode.
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
        # Heading follows the command for every alive robot, blocked or not.
        next_hdg = jnp.where(alive, cand_hdg, state.robot_headings)

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
        new_hdg = next_hdg

        # A blocked robot travelled nothing this step: reporting the commanded
        # v would tell the actor it is cruising while it is stuck against a
        # wall. The applied omega is reported as-is, since rotation went through.
        new_vel = jnp.stack([
            jnp.where(moved, v_cmds,     0.0),
            jnp.where(alive, omega_cmds, 0.0),
        ], axis=-1)

        # --- Per-agent coverage credit ---
        cols, rows = self._pos_to_cell(new_pos)
        flat       = rows * self.grid_w + cols                     # (N,)
        coverable  = self._free_flat[flat] > 0.0
        already    = prev_grid[rows, cols] > 0.0
        eligible   = moved & coverable & ~already
        # Two robots can enter the same unvisited cell on the same step; the cell
        # is discovered once, so exactly one of them may be paid. A scatter-max
        # over agent ids picks a single winner without any sequential pass.
        ids   = self._robot_ids
        claim = jnp.zeros((self.num_cells,), jnp.int32).at[flat].max(
            jnp.where(eligible, ids + 1, 0)
        )
        discovered = eligible & (claim[flat] == ids + 1)
        # Everything else an active robot does — re-entering a covered cell,
        # sitting still, or bumping into a wall footprint — is redundant work.
        redundant  = moved & ~discovered

        # `.max` leaves cells of ineligible robots untouched and is safe under
        # duplicate indices (several robots landing on the same cell).
        new_grid = prev_grid.at[rows, cols].max(
            jnp.where(moved & coverable, 1.0, 0.0)
        )

        # --- Cooperative bonuses: room completion and full coverage ---
        covered   = jnp.sum(new_grid[None, :, :] * self.room_masks, axis=(1, 2))
        ratio     = covered / jnp.maximum(self.room_totals, 1.0)
        newly     = (~state.room_completed) & (self.room_totals > 0) \
                    & (ratio >= self.room_completion_threshold)
        room_completed = state.room_completed | newly

        # The grid only ever holds free cells, so this reaches free_total exactly
        # when the whole coverable area has been visited.
        complete   = jnp.sum(new_grid) >= self.free_total - 0.5
        team_bonus = (self.room_completion_bonus * jnp.sum(newly)
                      + self.completion_bonus * complete)

        # --- Proximity penalty: soft repulsion before hard collision ---
        # Diagonal carries +inf from _pairwise_sq_dist, so self-pairs never fire.
        dist     = jnp.sqrt(self._pairwise_sq_dist(new_pos))
        pen      = self.psi * (1.0 - dist / self._safe_dist) * (dist < self._safe_dist)
        prox_pen = jnp.sum(pen, axis=1) * alive

        # --- Compose per-agent rewards ---
        rewards = jnp.where(
            alive,
            self.alpha * discovered
            - self.beta * redundant
            - self.tau
            - self.kappa * collided
            - prox_pen
            + team_bonus,
            0.0,
        ).astype(jnp.float32)

        step_count = state.step_count + 1
        truncated  = step_count >= self.max_steps
        # Shared fate: one collision anywhere in the team ends the episode, and
        # so does finishing the map — there is nothing left to earn.
        terminated = complete | (jnp.any(collided) if self.terminate_on_collision
                                 else jnp.bool_(False))

        next_state = state.replace(
            robot_positions  = new_pos,
            robot_headings   = new_hdg,
            robot_velocities = new_vel,
            coverage_grid    = new_grid,
            room_completed   = room_completed,
            robot_alive      = alive_next,
            step_count       = step_count,
            wall_hits        = wall_hit.astype(jnp.float32),
            robot_hits       = robot_hit.astype(jnp.float32),
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

        # --- Local coverage patch centred on each robot (binary, kept last) ---
        if self.use_local_coverage_obs:
            r = self.local_coverage_size // 2
            s = self.local_coverage_size
            # Non-free cells read as covered: they carry no reward, so showing
            # them as pending would advertise work that does not exist.
            source = jnp.maximum(state.coverage_grid, self.wall_grid)
            # Out-of-map padding reads as covered so robots are not drawn outside.
            padded = jnp.pad(source, r, constant_values=1.0)
            cols, rows = self._pos_to_cell(state.robot_positions)
            patches = jax.vmap(
                lambda rw, cl: jax.lax.dynamic_slice(padded, (rw, cl), (s, s))
            )(rows, cols)
            parts.append(patches.reshape(n, s * s))

        return jnp.concatenate(parts, axis=1).astype(jnp.float32)

    def get_global_state(self, state: EnvState) -> GlobalState:
        """Centralised critic state as compact spatial tensors."""
        cols, rows = self._pos_to_cell(state.robot_positions)
        occupancy = jnp.zeros(
            (self.num_robots, self.grid_h, self.grid_w), jnp.float32
        ).at[self._robot_ids, rows, cols].set(state.robot_alive.astype(jnp.float32))

        kinematics = jnp.stack([
            state.robot_positions[:, 0] / self.map_layout.width,
            state.robot_positions[:, 1] / self.map_layout.height,
            jnp.cos(state.robot_headings),
            jnp.sin(state.robot_headings),
            state.robot_velocities[:, 0] / self.v_max,
            state.robot_velocities[:, 1] / self.omega_max,
        ], axis=-1)

        return GlobalState(state.coverage_grid, occupancy, kinematics)

    def critic_inputs(self, gs: GlobalState) -> tuple[jax.Array, jax.Array]:
        """Expand a `GlobalState` into per-agent critic inputs.

        Accepts any number of leading batch dimensions and returns
            grid : (..., N, 4, H, W)  channels [walls, coverage, self, teammates]
            vec  : (..., N, 6 + 6N)   own kinematics followed by the joint vector

        `robot_velocities` are the commanded actions of the previous step, so the
        joint action history the centralised critic needs is already inside the
        kinematic vector. The *current* joint action is deliberately excluded:
        conditioning on it would turn this into Q(s, a), which cannot be used as
        the GAE baseline that the per-agent advantage is built from.
        """
        occ  = gs.occupancy                                   # (..., N, H, W)
        me   = occ[..., :, None, :, :]                        # (..., N, 1, H, W)
        rest = (occ.sum(axis=-3, keepdims=True) - occ)[..., :, None, :, :]
        cov  = jnp.broadcast_to(gs.coverage[..., None, None, :, :], me.shape)
        wall = jnp.broadcast_to(self.wall_grid, me.shape)
        grid = jnp.concatenate([wall, cov, me, rest], axis=-3)

        joint = gs.kinematics.reshape(*gs.kinematics.shape[:-2], 1, -1)
        vec = jnp.concatenate(
            [gs.kinematics, jnp.broadcast_to(joint, (*gs.kinematics.shape[:-1],
                                                     joint.shape[-1]))],
            axis=-1,
        )
        return grid, vec

    def get_info(self, state: EnvState) -> dict:
        """Diagnostics. All values are arrays so the dict survives jit/vmap."""
        covered = jnp.sum(state.coverage_grid)
        room_cov = jnp.sum(state.coverage_grid[None, :, :] * self.room_masks, axis=(1, 2))
        info = {
            # Denominator is the reachable free space, so 1.0 means "done".
            'coverage_ratio':   covered / self.free_total,
            'covered_cells':    covered,
            'total_cells':      jnp.float32(self.free_total),
            'step':             state.step_count,
            'robots_alive':     state.robot_alive,
            'num_robots_alive': jnp.sum(state.robot_alive),
            # Fraction of the team that collided on the step leading to this
            # state, split by cause (the two are mutually exclusive: a robot
            # blocked by a wall is excluded from the pairwise test).
            'wall_collision_rate':  jnp.mean(state.wall_hits),
            'robot_collision_rate': jnp.mean(state.robot_hits),
            # Episode-end causes, as 0/1 floats so they survive vmap/scan and can
            # be averaged directly into rates. `complete` mirrors the termination
            # test in `step`; `timeout` mirrors the truncation test.
            'complete':         (covered >= self.free_total - 0.5).astype(jnp.float32),
            'timeout':          (state.step_count >= self.max_steps).astype(jnp.float32),
        }
        for ri in range(self.num_rooms):
            info[f'room_{ri}_ratio'] = room_cov[ri] / jnp.maximum(self.room_totals[ri], 1.0)
        return info
