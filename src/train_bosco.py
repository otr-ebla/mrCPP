#!/usr/bin/env python3
"""Entry point: train MAPPO under BOSCO guidance (JAX).

Same algorithm, environment and networks as `src.train_simple`; two things are
added, both of them the BOSCO planner's next waypoint:

1. Observation. Each robot's observation gains a 4-channel egocentric encoding
   of the cell BOSCO would send it to next — bearing (cos, sin), distance, and a
   validity flag. The block is appended to the *tail* of the observation, after
   the binary coverage patch, so `RunningMeanStd` (which only ever touches the
   continuous prefix, see `rms_normalize`) leaves it alone. That is the right
   treatment: all four channels are already O(1) and bounded by construction.

2. Reward. A robot that actually enters the cell it was pointed at is paid
   `--guide-bonus` on that step, on top of the environment's own difference
   reward. The waypoint is always one move away, so this is a dense, per-agent,
   one-step objective rather than a distant goal — which is what makes it usable
   as shaping instead of another exploration problem.

Why the waypoint has to be in the observation as well as the reward
------------------------------------------------------------------
Paying for arrival alone would be a non-stationary reward the policy cannot see
the cause of: two identical observations, one of which is worth a bonus, differ
only in the planner's hidden cursor. Feeding the same waypoint the bonus is
computed from makes the shaped reward a function of the observation again, so it
is learnable rather than noise. It is also the only channel that carries global
information — the tour is planned on the whole map — into an otherwise strictly
local, decentralised observation.

Cost: the rollout is a python loop
----------------------------------
`train_simple` collects a whole update inside one `lax.scan`. That is impossible
here: BOSCO is a host-side numpy planner (heap Dijkstra, python tours), so the
waypoint is produced between two device calls and the scan degrades into a
python loop. Everything that can stay on device still does — the actor, the
critic and the entire vectorised environment transition are one jitted call each
— and the host bookkeeping is gated on a robot actually changing cell (~5 steps
at v_max against a 0.5 m cell), so most steps cost only the device→host transfer
of poses and the coverage grid. Measured at 8 envs x 3 robots on CPU that is
~2.6x `train_simple`'s wall clock per update. The PPO update itself is untouched
and still runs as a single jitted scan.

Usage
-----
    python -m src.train_bosco
    python -m src.train_bosco --guide-bonus 4.0 --wandb-name guided
    python -m src.pretrain_bc && \
        python -m src.train_bosco --resume checkpoints/bc_pretrained.pkl

The last form is the intended pipeline: `--resume` accepts a checkpoint from
`src.pretrain_bc` or `src.train_simple` even though its actor is narrower, by
zero-padding the trunk rows the guidance block feeds into. The widened actor
computes exactly what the loaded one did, so the behaviour-cloned policy
transfers intact and PPO learns what the waypoint is worth from there.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.bosco_guide import make_guides
from src.algorithms.mappo import (
    MAPPO,
    RunningMeanStd,
    Transition,
    _tanh_normal_log_prob,
    compute_gae,
    rms_init,
    rms_normalize,
    rms_update,
)
from src.envs.vec_env import VecEnv
from src.models.actor_critic import Actor, Critic
from src.train_simple import (
    _COLLISION,
    _SUCCESS,
    _TIMEOUT,
    _WINDOW,
    init_wandb,
    linear_lr_decay,
    wandb,
    window,
)
from src.utils.config_parser import load_config
from src.utils.jax_device import describe, select_device


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, update: int, actor_state, critic_state, rms,
                    guide_dim: int) -> None:
    """`train_simple`'s payload plus the guidance width the actor was built for.

    The extra key makes the observation layout self-describing — an actor trained
    here does not accept a bare environment observation — and is ignored by
    `train_simple.load_checkpoint`, so a guided checkpoint stays loadable there.
    """
    payload = {
        'update':        update,
        'actor_params':  jax.device_get(actor_state.params),
        'critic_params': jax.device_get(critic_state.params),
        'actor_opt':     jax.device_get(actor_state.opt_state),
        'critic_opt':    jax.device_get(critic_state.opt_state),
        'obs_rms':       jax.device_get(rms),
        'guide_dim':     int(guide_dim),
    }
    with open(path, 'wb') as f:
        pickle.dump(payload, f)

def _trunk_key(params, in_dim: int) -> str | None:
    """Name of the actor layer whose kernel takes `in_dim` inputs, if unique."""
    hits = [k for k, v in params['params'].items()
            if isinstance(v, dict) and 'kernel' in v
            and np.ndim(v['kernel']) == 2 and np.shape(v['kernel'])[0] == in_dim]
    return hits[0] if len(hits) == 1 else None


def load_checkpoint(path: str, actor_state, critic_state, device,
                    trunk_in: int) -> tuple[object, object, RunningMeanStd, int]:
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)

    params = ckpt['actor_params']
    if _trunk_key(params, trunk_in) is None:
        raise SystemExit(
            f"'{path}' has no actor trunk taking {trunk_in} inputs — it was trained with a different "
            "model config."
        )

    params = jax.device_put(params, device)
    actor_state = actor_state.replace(
        params=params,
        opt_state=jax.device_put(ckpt['actor_opt'], device),
    )
    critic_state = critic_state.replace(
        params=jax.device_put(ckpt['critic_params'], device),
        opt_state=jax.device_put(ckpt['critic_opt'], device),
    )
    rms = RunningMeanStd(*jax.device_put(tuple(ckpt['obs_rms']), device))
    return actor_state, critic_state, rms, int(ckpt['update'])


# ---------------------------------------------------------------------------
# Guided rollout
# ---------------------------------------------------------------------------

class GuidedCarry(NamedTuple):
    """Rollout state threaded between updates. Not a pytree: `guides` are host
    objects and `targets` is numpy, so this is never tree_map'd."""

    env_state: object
    obs:       jax.Array      # already carries the guidance block
    gstate:    object
    rms:       RunningMeanStd
    targets:   np.ndarray     # (E, N) flat cell index, -1 when idle


class GuidedRollout:
    """Collects T steps with the BOSCO waypoint in the loop.

    One jitted call produces the action, the log-probability and the value; one
    jitted call steps every environment; one `device_get` brings back the three
    host arrays the planner reads (poses, headings, coverage grid). The waypoint
    and the arrival bonus are computed between them, which is the only reason
    this is a python loop at all.
    """

    def __init__(self, mappo: MAPPO, vec_env: VecEnv, guides, guide_bonus: float):
        self.mappo = mappo
        self.env = vec_env
        self.guides = guides
        self.graph = guides[0].graph
        self.bonus = float(guide_bonus)
        e, n = vec_env.E, vec_env.num_robots

        @jax.jit
        def act(actor_params, critic_params, rms, obs, gstate, key):
            # Statistics are refreshed on the raw observation before it is
            # normalised, exactly as in `MAPPO.rollout`. `rms` covers only the
            # continuous prefix, so the guidance tail passes through untouched.
            rms = rms_update(rms, obs.reshape(e * n, -1))
            obs_n = rms_normalize(rms, obs)

            mean, log_std = mappo.actor.apply(actor_params, obs_n.reshape(e * n, -1))
            std = jnp.exp(log_std)
            z = mean + std * jax.random.normal(key, mean.shape)
            action = jnp.tanh(z)
            log_prob = _tanh_normal_log_prob(z, mean, std, action).reshape(e, n)
            value = mappo._values(critic_params, gstate)
            return (obs_n, action.reshape(e, n, -1), z.reshape(e, n, -1),
                    log_prob, value, rms)

        self._act = act
        self._value_fn = jax.jit(mappo._values)

    def _get_target_coords(self, pos: np.ndarray, targets: np.ndarray) -> np.ndarray:
        valid = targets >= 0
        safe_targets = np.maximum(targets, 0)
        coords = self.graph.centers[safe_targets]
        return np.where(valid[..., None], coords, pos)

    @staticmethod
    def _host(state, done=None):
        """The planner's three arrays (and optionally `done`) in one transfer."""
        payload = (state.robot_positions, state.robot_headings, state.coverage_grid)
        return jax.device_get(payload if done is None else payload + (done,))

    def start(self, key: jax.Array) -> GuidedCarry:
        state, obs, gstate, _ = self.env.reset(key)
        pos, hdg, cov = self._host(state)
        targets = np.full((self.env.E, self.env.num_robots), -1, dtype=np.int64)
        for e in range(self.env.E):
            self.guides[e].reset(pos[e])
            targets[e], _ = self.guides[e].update(pos[e], cov[e])
            
        target_coords = self._get_target_coords(pos, targets)
        state, obs, gstate = self.env.update_bosco(state, jnp.asarray(target_coords, dtype=jnp.float32))
        
        return GuidedCarry(state, obs, gstate, rms_init(self.env.norm_dim), targets)

    def run(self, actor_params, critic_params, carry: GuidedCarry, num_steps: int,
            key: jax.Array):
        """-> (carry, trajectory, bootstrap value, waypoint hits (T, E, N))."""
        E, N = self.env.E, self.env.num_robots
        state, obs, gstate, rms, targets = carry
        keys = jax.random.split(key, num_steps)
        buf, hits = [], np.zeros((num_steps, E, N), np.float32)

        for t in range(num_steps):
            obs_n, action, z, log_prob, value, rms = self._act(
                actor_params, critic_params, rms, obs, gstate, keys[t]
            )
            next_state, _, reward, term, done, info, _ = self.env.step(
                state, action
            )

            pos, hdg, cov, done_np = self._host(next_state, done)
            reached = np.zeros((E, N), dtype=bool)
            for e in range(E):
                if done_np[e]:
                    self.guides[e].reset(pos[e])
                targets[e], reached[e] = self.guides[e].update(pos[e], cov[e])
            hits[t] = reached

            target_coords = self._get_target_coords(pos, targets)
            next_state, next_obs, next_gstate = self.env.update_bosco(next_state, jnp.asarray(target_coords, dtype=jnp.float32))

            buf.append(Transition(
                obs=obs_n,
                gstate=gstate,
                action=action,
                z=z,
                log_prob=log_prob,
                reward=(reward + self.bonus * jnp.asarray(reached, jnp.float32))
                * self.mappo.reward_scale,
                value=value,
                term=term.astype(jnp.float32),
                done=done.astype(jnp.float32),
                coverage=info['coverage_ratio'],
                wall_hit=info['wall_collision_rate'],
                robot_hit=info['robot_collision_rate'],
                complete=info['complete'],
                timeout=info['timeout'],
            ))

            state, obs, gstate = next_state, next_obs, next_gstate

        traj = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *buf)
        last_value = self._value_fn(critic_params, gstate)
        return GuidedCarry(state, obs, gstate, rms, targets), traj, last_value, hits


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# Kept apart from `train_simple`'s own outputs: the two policies are not
# interchangeable — a guided actor rejects a bare environment observation — so
# writing both to `checkpoint_final.pkl` would silently destroy the baseline.
CHECKPOINT_NAME = 'checkpoint_bosco.pkl'
LOG_NAME        = 'training_log_bosco.csv'


class _WidenedEnv:
    """`vec_env` with the guided observation width.

    `MAPPO.create_train_states` reads `obs_dim` off the environment to shape the
    dummy observation it initialises the actor with, and the environment cannot
    report a width that includes a block it does not produce. Only that one field
    differs, so everything else is forwarded.
    """

    def __init__(self, vec_env: VecEnv, obs_dim: int):
        self._env = vec_env
        self.obs_dim = obs_dim

    def __getattr__(self, name):
        return getattr(self._env, name)


def train(config_path: str, save_dir: str, resume: str | None,
          backend: str | None = None, guide_bonus: float = 2.0,
          wandb_overrides: dict | None = None):
    config = load_config(config_path)
    # Must run before any array is created so implicit placement follows the
    # selected backend: Metal on Apple Silicon, else CUDA, else CPU.
    device = select_device(backend)
    print(f"Device: {describe(device)}  |  requested: {backend or 'auto'}")

    env_cfg   = config.get('env',   {})
    model_cfg = config.get('model', {})
    train_cfg = config.get('train', {})

    vec_env    = VecEnv(train_cfg.get('num_envs', 4), env_cfg)
    env        = vec_env.env
    E          = vec_env.E
    N          = vec_env.num_robots
    action_dim = vec_env.action_dim
    # The guidance block rides in the observation tail, where the actor's `patch`
    # slice picks it up and feeds it straight to the trunk alongside the binary
    # coverage patch. Both are inputs no normaliser should touch.
    tail_dim = env.patch_dim + GUIDE_DIM
    obs_dim  = env.obs_dim + GUIDE_DIM

    lidar_embed = model_cfg.get('lidar_embed',  64)
    hidden_size = model_cfg.get('hidden_size', 128)
    trunk_in    = lidar_embed + env.obs_vec_dim + tail_dim

    print(f"Parallel envs: {E}  |  robots/env: {N}  |  obs_dim: {obs_dim} "
          f"({env.obs_dim} env + {GUIDE_DIM} guidance)  |  critic map: "
          f"{vec_env.critic_channels}x{vec_env.grid_h}x{vec_env.grid_w}"
          f" + {vec_env.critic_vec_dim}")
    print(f"Coverable cells: {int(env.free_total)} / {env.num_cells} "
          f"({env.free_total / env.num_cells:.1%} of the grid)")
    print(f"Guidance: BOSCO next-cell waypoint, arrival bonus {guide_bonus} "
          f"(alpha={env.alpha} per discovered cell)")

    actor = Actor(
        action_dim=action_dim,
        vec_dim=env.obs_vec_dim,
        n_rays=env.n_rays,
        patch_dim=tail_dim,
        lidar_embed=lidar_embed,
        hidden_size=hidden_size,
    )
    critic = Critic(
        hidden_size=model_cfg.get('critic_hidden',    256),
        map_embed=model_cfg.get('critic_map_embed', 128),
    )
    mappo = MAPPO(actor, critic, vec_env, train_cfg, device=device)
    # `MAPPO.create_train_states` sizes its dummy observation from `vec_env`,
    # which knows nothing about the guidance block; the actor is initialised here
    # against the real width instead.
    mappo.env = _WidenedEnv(vec_env, obs_dim)

    T             = train_cfg.get('rollout_steps',  256)
    total_updates = train_cfg.get('total_updates',  3000)
    log_interval  = train_cfg.get('log_interval',   10)
    gamma         = train_cfg.get('gamma',          0.99)
    gae_lambda    = train_cfg.get('gae_lambda',     0.95)
    normalize_obs = train_cfg.get('normalize_obs',  True)
    lr_decay      = train_cfg.get('lr_decay',       True)
    lr_actor_0    = train_cfg.get('lr_actor',       3e-4)
    lr_critic_0   = train_cfg.get('lr_critic',      1e-3)
    reward_scale  = train_cfg.get('reward_scale',   1.0)

    key = jax.random.PRNGKey(train_cfg.get('seed', 0))
    key, init_key, reset_key = jax.random.split(key, 3)

    actor_state, critic_state = mappo.create_train_states(init_key)

    guides = make_guides(env, E)
    print(f"Map: {env.grid_h}x{env.grid_w} cells, "
          f"{int(guides[0]._isolated.sum())} unreachable by the robot disc")
    rollout = GuidedRollout(mappo, vec_env, guides, guide_bonus)
    carry = rollout.start(reset_key)

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, LOG_NAME)

    start_update = 1
    if resume:
        actor_state, critic_state, rms, last = load_checkpoint(
            resume, actor_state, critic_state, device, trunk_in
        )
        carry = carry._replace(rms=rms)
        start_update = last + 1
        print(f"Resumed from {resume}, continuing at update {start_update}")

    run = init_wandb(
        config,
        wandb_overrides or {},
        extra={
            'device':           describe(device),
            'num_envs':         E,
            'num_robots':       N,
            'obs_dim':          obs_dim,
            'action_dim':       action_dim,
            'coverable_cells':  int(env.free_total),
            'steps_per_update': T * E,
            'guide_bonus':      guide_bonus,
            'guide_dim':        GUIDE_DIM,
        },
    )
    if run is not None:
        print(f"W&B run: {run.url or run.dir}")

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['update', 'episodes', 'env_steps',
                                'mean_ep_reward', 'mean_ep_coverage',
                                'coverage_ratio', 'mean_ep_length',
                                'completion_rate', 'timeout_rate',
                                'collision_end_rate',
                                'wall_collision_rate', 'robot_collision_rate',
                                'wall_collisions_per_episode',
                                'robot_collisions_per_episode',
                                'guide_hits_per_episode', 'guide_hit_rate',
                                'actor_loss', 'critic_loss', 'entropy', 'std'])

    # Per-env accumulators for the episode currently in flight; the rollout is
    # cut at an arbitrary step, so these must survive across updates.
    ep_reward = np.zeros(E, dtype=np.float64)
    ep_len    = np.zeros(E, dtype=np.int64)
    ep_wall   = np.zeros(E, dtype=np.float64)
    ep_robot  = np.zeros(E, dtype=np.float64)
    ep_hit    = np.zeros(E, dtype=np.float64)   # team-mean waypoints reached

    ep_rewards: list[float] = []
    ep_coverages: list[float] = []
    ep_lengths: list[int] = []
    ep_walls: list[float] = []
    ep_robots: list[float] = []
    ep_hits: list[float] = []
    ep_outcomes: list[int] = []
    ep_count = 0
    best_mean_reward = -np.inf

    for update in range(start_update, total_updates + 1):
        lr_a = linear_lr_decay(lr_actor_0,  update, total_updates) if lr_decay else lr_actor_0
        lr_c = linear_lr_decay(lr_critic_0, update, total_updates) if lr_decay else lr_critic_0

        key, rollout_key = jax.random.split(key)
        prev_rms = carry.rms
        carry, traj, last_value, hits = rollout.run(
            actor_state.params, critic_state.params, carry, T, rollout_key
        )
        if not normalize_obs:
            carry = carry._replace(rms=prev_rms)

        advantages, returns = compute_gae(traj, last_value, gamma, gae_lambda)

        actor_state, critic_state, metrics = mappo.update(
            actor_state, critic_state, traj, advantages, returns, lr_a, lr_c
        )

        # ----------------------------------------------------------------
        # Episode bookkeeping (host-side, from the stacked rollout)
        # ----------------------------------------------------------------
        # Includes the arrival bonus: it is part of what the policy optimises, so
        # hiding it from the logged return would make the reward curve describe a
        # different objective than the one being maximised. `guide_hits` reports
        # the shaping separately, and `mean_ep_coverage` stays the success metric.
        team_rewards = np.asarray(jnp.mean(traj.reward, axis=-1)) / reward_scale
        dones     = np.asarray(traj.done)          # (T, E)
        coverage  = np.asarray(traj.coverage)      # (T, E)
        wall_hit  = np.asarray(traj.wall_hit)      # (T, E)
        robot_hit = np.asarray(traj.robot_hit)
        complete  = np.asarray(traj.complete)
        timeout   = np.asarray(traj.timeout)
        hit_rate_t = hits.mean(axis=2)             # (T, E) team-mean per step
        for t in range(T):
            ep_reward += team_rewards[t]
            ep_len    += 1
            ep_wall   += wall_hit[t]
            ep_robot  += robot_hit[t]
            ep_hit    += hit_rate_t[t]
            for e in np.nonzero(dones[t])[0]:
                ep_rewards.append(float(ep_reward[e]))
                ep_coverages.append(float(coverage[t, e]))
                ep_lengths.append(int(ep_len[e]))
                ep_walls.append(float(ep_wall[e]))
                ep_robots.append(float(ep_robot[e]))
                ep_hits.append(float(ep_hit[e]))
                ep_outcomes.append(
                    _SUCCESS   if complete[t, e] > 0.5 else
                    _TIMEOUT   if timeout[t, e]  > 0.5 else
                    _COLLISION
                )
                ep_reward[e] = ep_wall[e] = ep_robot[e] = ep_hit[e] = 0.0
                ep_len[e] = 0
                ep_count += 1

        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------
        if update % log_interval == 0:
            mean_ep_r   = window(ep_rewards)
            mean_ep_cov = window(ep_coverages)
            mean_ep_len = window(ep_lengths)
            last_cov    = float(coverage[-1].mean())
            ep_wall_mean  = window(ep_walls)
            ep_robot_mean = window(ep_robots)
            ep_hit_mean   = window(ep_hits)
            wall_rate  = float(wall_hit.mean())
            robot_rate = float(robot_hit.mean())
            # Fraction of robot-steps that ended on the waypoint. A robot needs
            # ~5 steps to cross a 0.5 m cell at v_max, so 0.2 is the arithmetic
            # ceiling, but BOSCO itself only reaches 0.095 on this map — a 90°
            # lane change costs ~16 steps of turning in place, which buys no
            # waypoint. That measured 0.095, not the ceiling, is what a policy
            # tracking the tour as well as the planner does looks like.
            guide_rate = float(hits.mean())
            outcomes = np.asarray(ep_outcomes[-_WINDOW:], dtype=np.int64)
            if outcomes.size:
                completion_rate    = float(np.mean(outcomes == _SUCCESS))
                timeout_rate       = float(np.mean(outcomes == _TIMEOUT))
                collision_end_rate = float(np.mean(outcomes == _COLLISION))
            else:
                completion_rate = timeout_rate = collision_end_rate = 0.0
            env_steps = update * T * E
            losses = jax.device_get(metrics)
            print(
                f"Update {update:5d}/{total_updates} | "
                f"episodes={ep_count:6d} | "
                f"mean_ep_r={mean_ep_r:8.3f} | "
                f"success={mean_ep_cov:6.2%} | "
                f"coverage={last_cov:.2%} | "
                f"guide={guide_rate:6.2%} | "
                f"actor={float(losses['actor_loss']):7.4f} | "
                f"critic={float(losses['critic_loss']):7.4f} | "
                f"entropy={float(losses['entropy']):6.4f} | "
                f"std={float(losses['std']):5.3f} | "
                f"wall={wall_rate:6.2%} | "
                f"rr={robot_rate:6.2%} | "
                f"timeout={timeout_rate:6.2%}",
                flush=True,
            )
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    update, ep_count, env_steps,
                    round(mean_ep_r,   4), round(mean_ep_cov, 4),
                    round(last_cov,    4), round(mean_ep_len, 1),
                    round(completion_rate,    4),
                    round(timeout_rate,       4),
                    round(collision_end_rate, 4),
                    round(wall_rate,      6),
                    round(robot_rate,     6),
                    round(ep_wall_mean,   4),
                    round(ep_robot_mean,  4),
                    round(ep_hit_mean,    4),
                    round(guide_rate,     6),
                    round(float(losses['actor_loss']),  4),
                    round(float(losses['critic_loss']), 4),
                    round(float(losses['entropy']),     4),
                    round(float(losses['std']),         4),
                ])
            if run is not None:
                wandb.log({
                    'env_steps':                      env_steps,
                    'update':                         update,
                    'episode/count':                  ep_count,
                    'episode/mean_reward':            mean_ep_r,
                    'episode/mean_coverage':          mean_ep_cov,
                    'episode/mean_length':            mean_ep_len,
                    'episode/coverage_last_step':     last_cov,
                    'rate/completion':                completion_rate,
                    'rate/timeout':                   timeout_rate,
                    'rate/collision_end':             collision_end_rate,
                    'collision/wall_per_robot_step':  wall_rate,
                    'collision/robot_per_robot_step': robot_rate,
                    'collision/wall_per_episode':     ep_wall_mean,
                    'collision/robot_per_episode':    ep_robot_mean,
                    'guide/hit_per_robot_step':       guide_rate,
                    'guide/hits_per_episode':         ep_hit_mean,
                    'loss/actor':                     float(losses['actor_loss']),
                    'loss/critic':                    float(losses['critic_loss']),
                    'loss/entropy':                   float(losses['entropy']),
                    'policy/std':                     float(losses['std']),
                    'lr/actor':                       lr_a,
                    'lr/critic':                      lr_c,
                }, step=update)
            if ep_count > 0 and mean_ep_r > best_mean_reward:
                best_mean_reward = mean_ep_r
                save_checkpoint(os.path.join(save_dir, CHECKPOINT_NAME),
                                update, actor_state, critic_state, carry.rms,
                                GUIDE_DIM)
                print(f"  → best policy saved (mean_ep_r={mean_ep_r:.3f})")
                if run is not None:
                    run.summary['best_mean_ep_reward'] = mean_ep_r
                    run.summary['best_update'] = update

    save_checkpoint(os.path.join(save_dir, CHECKPOINT_NAME),
                    total_updates, actor_state, critic_state, carry.rms, GUIDE_DIM)
    if run is not None:
        run.finish()
    return actor_state, critic_state, carry.rms


if __name__ == '__main__':
    _default_cfg  = os.path.join(os.path.dirname(__file__), '..', 'config',
                                 'mappo_baseline.yaml')
    _default_save = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')

    parser = argparse.ArgumentParser(
        description='Train MAPPO for indoor coverage under BOSCO waypoint guidance')
    parser.add_argument('--config',   default=_default_cfg,
                        help='Path to YAML config file')
    parser.add_argument('--save-dir', default=_default_save,
                        help='Directory for checkpoints and training log')
    parser.add_argument('--resume',   default=None,
                        help='Checkpoint to resume from; a non-guided actor '
                             '(src.pretrain_bc / src.train_simple) is widened')
    parser.add_argument('--guide-bonus', type=float, default=2.0,
                        help='Reward paid to a robot that enters the cell BOSCO '
                             'pointed it at. Keep it below the environment\'s '
                             'alpha (10.0): the waypoint is usually an '
                             'undiscovered cell, so the two are collected '
                             'together and a larger bonus would make following '
                             'the tour worth more than covering the map')
    parser.add_argument('--backend',  default='auto',
                        choices=['auto', 'metal', 'cuda', 'gpu', 'cpu'],
                        help='Force a JAX backend. Default "auto": Metal on Apple '
                             'Silicon, else CUDA when an NVIDIA GPU is present, else CPU')
    parser.add_argument('--wandb', dest='wandb_enabled', action='store_true',
                        default=None, help='Enable Weights & Biases logging')
    parser.add_argument('--no-wandb', dest='wandb_enabled', action='store_false',
                        default=None, help='Disable Weights & Biases logging')
    parser.add_argument('--wandb-project', default=None, help='W&B project name')
    parser.add_argument('--wandb-entity',  default=None, help='W&B team / user')
    parser.add_argument('--wandb-name',    default=None, help='W&B run name')
    parser.add_argument('--wandb-group',   default=None, help='W&B run group')
    parser.add_argument('--wandb-mode',    default=None,
                        choices=['online', 'offline', 'disabled'],
                        help='W&B mode; "offline" logs locally with no network')
    args = parser.parse_args()
    train(args.config, args.save_dir, args.resume,
          None if args.backend == 'auto' else args.backend,
          args.guide_bonus,
          wandb_overrides={
              'enabled': args.wandb_enabled,
              'project': args.wandb_project,
              'entity':  args.wandb_entity,
              'name':    args.wandb_name,
              'group':   args.wandb_group,
              'mode':    args.wandb_mode,
          })
