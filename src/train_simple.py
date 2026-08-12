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


def linear_lr_decay(initial_lr: float, current_update: int, total_updates: int) -> float:
    return initial_lr * max(0.0, 1.0 - current_update / total_updates)


def save_checkpoint(path: str, update: int, actor_state, critic_state, rms) -> None:
    payload = {
        'update':        update,
        'actor_params':  jax.device_get(actor_state.params),
        'critic_params': jax.device_get(critic_state.params),
        'actor_opt':     jax.device_get(actor_state.opt_state),
        'critic_opt':    jax.device_get(critic_state.opt_state),
        'obs_rms':       jax.device_get(rms),
    }
    with open(path, 'wb') as f:
        pickle.dump(payload, f)


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

def train(config_path: str, save_dir: str, resume: str | None, backend: str | None = None):
    config = load_config(config_path)
    # Must run before any array is created so implicit placement follows the
    # selected backend: Metal on Apple Silicon, else CUDA, else CPU.
    device = select_device(backend)
    print(f"Device: {describe(device)}  |  requested: {backend or 'auto'}")

    env_cfg   = config.get('env',   {})
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
        patch_dim=env.patch_dim,
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

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['update', 'episodes', 'mean_ep_reward',
                                'mean_ep_coverage', 'coverage_ratio',
                                'actor_loss', 'critic_loss', 'entropy', 'std'])

    ep_reward = np.zeros(E, dtype=np.float64)   # per-env accumulator
    ep_rewards: list[float] = []
    ep_coverages: list[float] = []              # coverage reached at episode end
    ep_count = 0
    best_mean_reward = -np.inf

    for update in range(start_update, total_updates + 1):
        lr_a = linear_lr_decay(lr_actor_0,  update, total_updates) if lr_decay else lr_actor_0
        lr_c = linear_lr_decay(lr_critic_0, update, total_updates) if lr_decay else lr_critic_0

        # ----------------------------------------------------------------
        # Collect T environment steps across all E envs in a single call
        # ----------------------------------------------------------------
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
        team_rewards = np.asarray(jnp.mean(traj.reward, axis=-1)) / reward_scale
        dones = np.asarray(traj.done)                               # (T, E)
        coverage = np.asarray(traj.coverage)                        # (T, E)
        for t in range(T):
            ep_reward += team_rewards[t]
            finished = np.nonzero(dones[t])[0]
            for e in finished:
                ep_rewards.append(float(ep_reward[e]))
                # Pre-reset info, so this is the coverage the episode finished on.
                ep_coverages.append(float(coverage[t, e]))
                ep_reward[e] = 0.0
                ep_count += 1

        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------
        if update % log_interval == 0:
            window = ep_rewards[-20:] if ep_rewards else [0.0]
            mean_ep_r = float(np.mean(window))
            # Success metric: cells covered when the episode ended, averaged
            # over the same 20-episode window as the reward.
            cov_window = ep_coverages[-20:] if ep_coverages else [0.0]
            mean_ep_cov = float(np.mean(cov_window))
            last_cov = float(coverage[-1].mean())
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
                f"std={float(losses['std']):5.3f}"
            )
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    update, ep_count, round(mean_ep_r, 4),
                    round(mean_ep_cov, 4), round(last_cov, 4),
                    round(float(losses['actor_loss']),  4),
                    round(float(losses['critic_loss']), 4),
                    round(float(losses['entropy']),     4),
                    round(float(losses['std']),         4),
                ])
            if ep_count > 0 and mean_ep_r > best_mean_reward:
                best_mean_reward = mean_ep_r
                save_checkpoint(os.path.join(save_dir, 'checkpoint_final.pkl'),
                                update, actor_state, critic_state, carry.rms)
                print(f"  → best policy saved (mean_ep_r={mean_ep_r:.3f})")

    save_checkpoint(os.path.join(save_dir, 'checkpoint_final.pkl'),
                    total_updates, actor_state, critic_state, carry.rms)
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
    args = parser.parse_args()
    backend = None if args.backend == 'auto' else args.backend
    train(args.config, args.save_dir, args.resume, backend)
