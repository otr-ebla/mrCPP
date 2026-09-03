#!/usr/bin/env python3
"""Entry point: train MAPPO agents on the indoor coverage task (JAX)."""

from __future__ import annotations

import argparse
import csv
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from src.algorithms.mappo import MAPPO, RunningMeanStd, compute_gae
from src.envs.vec_env import VecEnv
from src.models.actor_critic import Actor, Critic
from src.utils.config_parser import load_config
from src.utils.jax_device import describe, select_device
from src.utils.human_curriculum import ghost_robot_probability

try:                        # Optional dependency: training runs without it.
    import wandb
except ImportError:         # pragma: no cover
    wandb = None


def linear_lr_decay(initial_lr: float, current_update: int, total_updates: int) -> float:
    return initial_lr * max(0.0, 1.0 - current_update / total_updates)


# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------

# Episode-end causes, in the order the trajectory flags are tested.
_SUCCESS, _TIMEOUT, _COLLISION = 0, 1, 2

# All episode statistics are reported over the same trailing window, so reward,
# coverage and the end-cause rates always describe the same set of episodes.
_WINDOW = 20


def window(values: list, default: float = 0.0) -> float:
    """Mean over the last `_WINDOW` finished episodes."""
    return float(np.mean(values[-_WINDOW:])) if values else default


def init_wandb(config: dict, overrides: dict, extra: dict):
    """Start a W&B run, or return None if logging is off or unavailable.

    Never fatal: a missing package or a failed handshake degrades to console and
    CSV logging rather than killing a run that may already be hours from a GPU.
    """
    cfg = dict(config.get('wandb', {}))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if not cfg.get('enabled', False):
        return None
    if wandb is None:
        print("W&B logging requested but `wandb` is not installed — "
              "run `pip install wandb`. Continuing without it.")
        return None
    try:
        run = wandb.init(
            project = cfg.get('project', 'mrcpp-mappo'),
            entity  = cfg.get('entity',  None),
            name    = cfg.get('name',    None),
            group   = cfg.get('group',   None),
            mode    = cfg.get('mode',    'online'),
            config  = {**config, **extra},
        )
    except Exception as exc:                                  # noqa: BLE001
        print(f"W&B init failed ({exc}); continuing without it.")
        return None

    # Every metric is also plottable against environment steps, which is the
    # only x-axis comparable across runs with different num_envs / rollout_steps.
    wandb.define_metric('env_steps')
    wandb.define_metric('*', step_metric='env_steps')
    return run


def save_checkpoint(path: str, update: int, actor_state, critic_state, rms) -> None:
    payload = {
        'update':        update,
        'actor_params':  jax.device_get(actor_state.params),
        'critic_params': jax.device_get(critic_state.params),
        'actor_opt':     jax.device_get(actor_state.opt_state),
        'critic_opt':    jax.device_get(critic_state.opt_state),
        'obs_rms':       jax.device_get(rms),
    }
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_checkpoint(path: str, actor_state, critic_state, device):
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    actor_state = actor_state.replace(
        params=jax.device_put(ckpt['actor_params'], device),
        opt_state=jax.device_put(ckpt['actor_opt'], device),
    )
    critic_state = critic_state.replace(
        params=jax.device_put(ckpt['critic_params'], device),
        opt_state=jax.device_put(ckpt['critic_opt'], device),
    )
    rms = RunningMeanStd(*jax.device_put(tuple(ckpt['obs_rms']), device))
    return actor_state, critic_state, rms, int(ckpt['update'])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config_path: str, save_dir: str, resume: str | None, backend: str | None = None,
          wandb_overrides: dict | None = None, num_humans: int = 0):
    config = load_config(config_path)
    # Must run before any array is created so implicit placement follows the
    # selected backend: Metal on Apple Silicon, else CUDA, else CPU.
    device = select_device(backend)
    print(f"Device: {describe(device)}  |  requested: {backend or 'auto'}")

    env_cfg   = config.get('env',   {})
    if num_humans > 0:
        env_cfg['num_humans'] = num_humans
    model_cfg = config.get('model', {})
    train_cfg = config.get('train', {})

    num_envs = train_cfg.get('num_envs', 4)

    vec_env    = VecEnv(num_envs, env_cfg)
    env        = vec_env.env
    E          = vec_env.E
    N          = vec_env.num_robots
    obs_dim    = vec_env.obs_dim
    action_dim = vec_env.action_dim

    print(f"Parallel envs: {E}  |  robots/env: {N}  |  obs_dim: {obs_dim}  |  "
          f"critic map: {vec_env.critic_channels}x{vec_env.grid_h}x{vec_env.grid_w}"
          f" + {vec_env.critic_vec_dim}")
    print(f"Coverable cells: {int(env.free_total)} / {env.num_cells} "
          f"({env.free_total / env.num_cells:.1%} of the grid)")

    actor = Actor(
        action_dim=action_dim,
        vec_dim=env.obs_vec_dim,
        n_rays=env.n_rays,
        tail_dim=env.patch_dim,
        lidar_embed=model_cfg.get('lidar_embed',  64),
        hidden_size=model_cfg.get('hidden_size', 128),
    )
    critic = Critic(
        hidden_size=model_cfg.get('critic_hidden',    256),
        map_embed=model_cfg.get('critic_map_embed', 128),
    )
    mappo = MAPPO(actor, critic, vec_env, train_cfg, device=device)

    T             = train_cfg.get('rollout_steps',  256)
    total_updates = train_cfg.get('total_updates',  3000)
    log_interval  = train_cfg.get('log_interval',   10)
    gamma         = train_cfg.get('gamma',          0.99)
    gae_lambda    = train_cfg.get('gae_lambda',     0.95)
    normalize_obs = train_cfg.get('normalize_obs',  True)
    lr_decay      = train_cfg.get('lr_decay',       True)
    lr_actor_0    = train_cfg.get('lr_actor',       3e-4)
    lr_critic_0   = train_cfg.get('lr_critic',      1e-3)
    # The trajectory carries rewards already scaled for the critic; undo it for
    # reporting so the logged episode reward stays in environment units and
    # remains comparable across runs with different scales.
    reward_scale  = train_cfg.get('reward_scale',   1.0)

    key = jax.random.PRNGKey(train_cfg.get('seed', 0))
    key, init_key, reset_key = jax.random.split(key, 3)

    actor_state, critic_state = mappo.create_train_states(init_key)
    carry = mappo.init_carry(reset_key)

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'training_log.csv')

    start_update = 1
    if resume:
        actor_state, critic_state, rms, last = load_checkpoint(
            resume, actor_state, critic_state, device
        )
        carry = carry._replace(rms=rms)
        start_update = last + 1
        print(f"Resumed from {resume}, continuing at update {start_update}")

    run = init_wandb(
        config,
        wandb_overrides or {},
        extra={
            'device':         describe(device),
            'num_envs':       E,
            'num_robots':     N,
            'obs_dim':        obs_dim,
            'action_dim':     action_dim,
            'coverable_cells': int(env.free_total),
            'steps_per_update': T * E,
        },
    )
    if run is not None:
        # Offline runs have no URL; the local directory is what can be synced.
        print(f"W&B run: {run.url or run.dir}")

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['update', 'episodes', 'env_steps',
                                'mean_ep_reward', 'mean_ep_coverage',
                                'coverage_ratio', 'mean_ep_length',
                                'completion_rate', 'timeout_rate',
                                'collision_end_rate',
                                'wall_collision_rate', 'robot_collision_rate',
                                'human_collision_rate',
                                'wall_collisions_per_episode',
                                'robot_collisions_per_episode',
                                'human_collisions_per_episode',
                                'actor_loss', 'critic_loss', 'entropy', 'std'])

    # Per-env accumulators for the episode currently in flight; the rollout is
    # cut at an arbitrary step, so these must survive across updates.
    ep_reward = np.zeros(E, dtype=np.float64)
    ep_len    = np.zeros(E, dtype=np.int64)
    ep_wall   = np.zeros(E, dtype=np.float64)   # team-mean wall hits, summed
    ep_robot  = np.zeros(E, dtype=np.float64)   # team-mean robot-robot hits
    ep_human  = np.zeros(E, dtype=np.float64)   # team-mean robot-human hits

    ep_rewards: list[float] = []
    ep_coverages: list[float] = []              # coverage reached at episode end
    ep_lengths: list[int] = []
    ep_walls: list[float] = []                  # wall collisions per robot, per episode
    ep_robots: list[float] = []
    ep_humans: list[float] = []
    ep_outcomes: list[int] = []                 # _SUCCESS / _TIMEOUT / _COLLISION
    ep_count = 0
    best_mean_reward = -np.inf

    for update in range(start_update, total_updates + 1):
        lr_a = linear_lr_decay(lr_actor_0,  update, total_updates) if lr_decay else lr_actor_0
        lr_c = linear_lr_decay(lr_critic_0, update, total_updates) if lr_decay else lr_critic_0

        # ----------------------------------------------------------------
        # Collect T environment steps across all E envs in a single call
        # ----------------------------------------------------------------
        
        # Start with ghost robots, then phase in human avoidance as training
        # collision and coverage signals rise.
        if update == start_update:
            smoothed_col_rate = 0.0
            smoothed_coverage_rate = 0.0
        else:
            total_col_rate = float(traj.wall_hit.mean()) + float(traj.robot_hit.mean())
            smoothed_col_rate = 0.9 * smoothed_col_rate + 0.1 * total_col_rate
            smoothed_coverage_rate = (
                0.9 * smoothed_coverage_rate + 0.1 * float(traj.coverage.mean())
            )

        ghost_prob = ghost_robot_probability(
            smoothed_col_rate, smoothed_coverage_rate
        )
        carry = carry._replace(state=vec_env.update_ghost_robot_prob(
            carry.state, jnp.full((E,), ghost_prob, dtype=jnp.float32)
        ))

        key, rollout_key = jax.random.split(key)
        prev_rms = carry.rms
        carry, traj, last_value = mappo.rollout(
            actor_state.params, critic_state.params, carry, T, rollout_key
        )
        if not normalize_obs:
            carry = carry._replace(rms=prev_rms)

        advantages, returns = compute_gae(traj, last_value, gamma, gae_lambda)

        # ----------------------------------------------------------------
        # Policy update
        # ----------------------------------------------------------------
        actor_state, critic_state, metrics = mappo.update(
            actor_state, critic_state, traj, advantages, returns, lr_a, lr_c
        )

        # ----------------------------------------------------------------
        # Episode bookkeeping (host-side, from the stacked rollout)
        # ----------------------------------------------------------------
        host_stats = jax.device_get((
            jnp.mean(traj.reward, axis=-1), traj.done, traj.coverage,
            traj.wall_hit, traj.robot_hit, traj.human_hit,
            traj.complete, traj.timeout,
        ))
        (team_rewards, dones, coverage, wall_hit, robot_hit, human_hit,
         complete, timeout) = map(np.asarray, host_stats)
        team_rewards /= reward_scale
        # Already averaged over the team inside the env, so a value of 0.1 means
        # "one robot in ten collided on this step".
        for t in range(T):
            ep_reward += team_rewards[t]
            ep_len    += 1
            ep_wall   += wall_hit[t]
            ep_robot  += robot_hit[t]
            ep_human  += human_hit[t]
            finished = np.nonzero(dones[t])[0]
            for e in finished:
                ep_rewards.append(float(ep_reward[e]))
                # Pre-reset info, so this is the coverage the episode finished on.
                ep_coverages.append(float(coverage[t, e]))
                ep_lengths.append(int(ep_len[e]))
                ep_walls.append(float(ep_wall[e]))
                ep_robots.append(float(ep_robot[e]))
                ep_humans.append(float(ep_human[e]))
                # Full coverage wins over a simultaneous horizon hit; anything
                # else that ends an episode is a terminal collision, which only
                # happens when terminate_on_collision is set.
                ep_outcomes.append(
                    _SUCCESS   if complete[t, e] > 0.5 else
                    _TIMEOUT   if timeout[t, e]  > 0.5 else
                    _COLLISION
                )
                ep_reward[e] = 0.0
                ep_len[e]    = 0
                ep_wall[e]   = 0.0
                ep_robot[e]  = 0.0
                ep_human[e]  = 0.0
                ep_count += 1

        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------
        if update % log_interval == 0:
            mean_ep_r = window(ep_rewards)
            # Success metric: cells covered when the episode ended, averaged
            # over the same 20-episode window as the reward.
            mean_ep_cov = window(ep_coverages)
            mean_ep_len = window(ep_lengths)
            last_cov = float(coverage[-1].mean())
            # Per-episode collision totals, per robot: how many times an average
            # robot bumped into something during one episode.
            ep_wall_mean  = window(ep_walls)
            ep_robot_mean = window(ep_robots)
            ep_human_mean = window(ep_humans)
            # Same events as an instantaneous rate: the fraction of robot-steps
            # in *this* rollout that ended in a collision. Independent of the
            # episode window, so it reacts immediately to a policy change.
            wall_rate  = float(wall_hit.mean())
            robot_rate = float(robot_hit.mean())
            human_rate = float(human_hit.mean())
            # Episode-end causes over the window; the three sum to 1.
            # `completion_rate` is the strict success measure — the episode ended
            # because the whole map was covered — as opposed to `mean_ep_cov`,
            # which is how much got covered whatever the ending.
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
                f"actor={float(losses['actor_loss']):7.4f} | "
                f"critic={float(losses['critic_loss']):7.4f} | "
                f"entropy={float(losses['entropy']):6.4f} | "
                f"std={float(losses['std']):5.3f} | "
                f"rr={robot_rate:6.2%} | "
                f"rh={human_rate:6.2%} | "
                f"rw={wall_rate:6.2%} | "
                f"timeout={timeout_rate:6.2%}"
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
                    round(human_rate,     6),
                    round(ep_wall_mean,   4),
                    round(ep_robot_mean,  4),
                    round(ep_human_mean,  4),
                    round(float(losses['actor_loss']),  4),
                    round(float(losses['critic_loss']), 4),
                    round(float(losses['entropy']),     4),
                    round(float(losses['std']),         4),
                ])
            if run is not None:
                wandb.log({
                    'env_steps':                     env_steps,
                    'update':                        update,
                    'episode/count':                 ep_count,
                    'episode/mean_reward':           mean_ep_r,
                    'episode/mean_coverage':         mean_ep_cov,
                    'episode/mean_length':           mean_ep_len,
                    'episode/coverage_last_step':    last_cov,
                    'rate/completion':               completion_rate,
                    'rate/timeout':                  timeout_rate,
                    'rate/collision_end':            collision_end_rate,
                    'collision/wall_per_robot_step':  wall_rate,
                    'collision/robot_per_robot_step': robot_rate,
                    'collision/human_per_robot_step': human_rate,
                    'collision/wall_per_episode':     ep_wall_mean,
                    'collision/robot_per_episode':    ep_robot_mean,
                    'collision/human_per_episode':    ep_human_mean,
                    'loss/actor':                    float(losses['actor_loss']),
                    'loss/critic':                   float(losses['critic_loss']),
                    'loss/entropy':                  float(losses['entropy']),
                    'policy/std':                    float(losses['std']),
                    'lr/actor':                      lr_a,
                    'lr/critic':                     lr_c,
                }, step=update)
            if ep_count > 0 and mean_ep_r > best_mean_reward:
                best_mean_reward = mean_ep_r
                save_checkpoint(os.path.join(save_dir, 'checkpoint_final.pkl'),
                                update, actor_state, critic_state, carry.rms)
                print(f"  → best policy saved (mean_ep_r={mean_ep_r:.3f})")
                if run is not None:
                    run.summary['best_mean_ep_reward'] = mean_ep_r
                    run.summary['best_update'] = update

    save_checkpoint(os.path.join(save_dir, 'checkpoint_final.pkl'),
                    total_updates, actor_state, critic_state, carry.rms)
    if run is not None:
        run.finish()
    return actor_state, critic_state, carry.rms


if __name__ == '__main__':
    _default_cfg  = os.path.join(os.path.dirname(__file__), '..', 'config',
                                 'mappo_baseline.yaml')
    _default_save = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')

    parser = argparse.ArgumentParser(description='Train MAPPO for indoor coverage')
    parser.add_argument('--config',   default=_default_cfg,
                        help='Path to YAML config file')
    parser.add_argument('--save-dir', default=_default_save,
                        help='Directory for checkpoints and training log')
    parser.add_argument('--resume',   default=None,
                        help='Checkpoint path to resume from')
    parser.add_argument('--backend',  default='auto',
                        choices=['auto', 'metal', 'cuda', 'gpu', 'cpu'],
                        help='Force a JAX backend. Default "auto": Metal on Apple '
                             'Silicon, else CUDA when an NVIDIA GPU is present, else CPU')
    # W&B: absent flags fall back to the `wandb:` block of the YAML config.
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
    parser.add_argument('--humans', nargs='?', type=int, const=3, default=0, help='Number of humans')
    args = parser.parse_args()
    backend = None if args.backend == 'auto' else args.backend
    train(args.config, args.save_dir, args.resume, backend, wandb_overrides={
        'enabled': args.wandb_enabled,
        'project': args.wandb_project,
        'entity':  args.wandb_entity,
        'name':    args.wandb_name,
        'group':   args.wandb_group,
        'mode':    args.wandb_mode,
    }, num_humans=args.humans)
