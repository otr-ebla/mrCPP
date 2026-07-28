#!/usr/bin/env python3
"""Entry point: train MAPPO agents on the indoor coverage task."""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

from src.envs.vec_env import SubprocVecEnv
from src.models.actor_critic_mlp import Actor, Critic
from src.algorithms.mappo import MAPPO, RolloutBuffer
from src.utils.config_parser import load_config


# ---------------------------------------------------------------------------
# Observation normalisation (Welford online algorithm)
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Tracks running mean and variance for online observation normalisation."""

    def __init__(self, shape: tuple, eps: float = 1e-4):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        """x : (B, obs_dim) — a flat batch of observations."""
        x = x.reshape(-1, *self.mean.shape)
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)
        batch_count = x.shape[0]
        total       = self.count + batch_count
        delta       = batch_mean - self.mean
        self.mean   = self.mean + delta * batch_count / total
        m_a         = self.var * self.count
        m_b         = batch_var * batch_count
        M2          = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var    = M2 / total
        self.count  = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalise x: (..., obs_dim), preserving leading dimensions."""
        orig = x.shape
        out  = ((x.reshape(-1, self.mean.shape[0]) - self.mean)
                / np.sqrt(self.var + 1e-8)).astype(np.float32)
        return out.reshape(orig)

    def state_dict(self) -> dict:
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean, self.var, self.count = d['mean'], d['var'], d['count']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def linear_lr_decay(
    optimizer: torch.optim.Optimizer,
    initial_lr: float,
    current_update: int,
    total_updates: int,
) -> None:
    frac = max(0.0, 1.0 - current_update / total_updates)
    for group in optimizer.param_groups:
        group['lr'] = initial_lr * frac


def save_checkpoint(
    path: str,
    update: int,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    mappo: MAPPO,
    obs_rms: RunningMeanStd | None,
) -> None:
    payload = {
        'update':           update,
        'actor_state':      actor.state_dict(),
        'critic_state':     critic.state_dict(),
        'actor_opt_state':  mappo.actor_opt.state_dict(),
        'critic_opt_state': mappo.critic_opt.state_dict(),
    }
    if obs_rms is not None:
        payload['obs_rms'] = obs_rms.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    mappo: MAPPO,
    obs_rms: RunningMeanStd | None,
) -> int:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    actor.load_state_dict(ckpt['actor_state'])
    critic.load_state_dict(ckpt['critic_state'])
    mappo.actor_opt.load_state_dict(ckpt['actor_opt_state'])
    mappo.critic_opt.load_state_dict(ckpt['critic_opt_state'])
    if obs_rms is not None and 'obs_rms' in ckpt:
        obs_rms.load_state_dict(ckpt['obs_rms'])
    return int(ckpt['update'])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config_path: str, save_dir: str, resume: str | None) -> tuple:
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    env_cfg   = config.get('env',   {})
    model_cfg = config.get('model', {})
    train_cfg = config.get('train', {})

    num_envs = train_cfg.get('num_envs', 4)

    vec_env   = SubprocVecEnv(num_envs, env_cfg)
    E         = vec_env.E
    N         = vec_env.num_robots
    obs_dim   = vec_env.obs_dim
    state_dim = vec_env.state_dim
    action_dim = 2

    print(f"Parallel envs: {E}  |  robots/env: {N}  |  "
          f"obs_dim: {obs_dim}  |  state_dim: {state_dim}")

    hidden_size   = model_cfg.get('hidden_size',   128)
    gru_hidden    = model_cfg.get('gru_hidden',    128)
    critic_hidden = model_cfg.get('critic_hidden', 256)

    actor  = Actor(obs_dim, action_dim, hidden_size, gru_hidden)
    critic = Critic(state_dim, critic_hidden)
    mappo  = MAPPO(actor, critic, train_cfg, device=str(device))

    T             = train_cfg.get('rollout_steps',  256)
    total_updates = train_cfg.get('total_updates',  3000)
    log_interval  = train_cfg.get('log_interval',   10)
    save_interval = train_cfg.get('save_interval',  100)
    gamma         = train_cfg.get('gamma',          0.99)
    gae_lambda    = train_cfg.get('gae_lambda',     0.95)
    normalize_obs = train_cfg.get('normalize_obs',  True)
    lr_decay      = train_cfg.get('lr_decay',       True)
    lr_actor_0    = train_cfg.get('lr_actor',       3e-4)
    lr_critic_0   = train_cfg.get('lr_critic',      1e-3)

    buffer  = RolloutBuffer(T, E, N, obs_dim, state_dim, action_dim,
                            gru_hidden, torch.device(str(device)))
    obs_rms = RunningMeanStd(shape=(obs_dim,)) if normalize_obs else None

    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'training_log.csv')

    start_update = 1
    if resume:
        start_update = load_checkpoint(resume, actor, critic, mappo, obs_rms) + 1
        print(f"Resumed from {resume}, continuing at update {start_update}")

    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['update', 'episodes', 'mean_ep_reward',
                                 'coverage_ratio', 'actor_loss', 'critic_loss', 'entropy'])

    # -- Initial state --
    obs, state, info_list = vec_env.reset()
    # hx: (E, N, gru_hidden) — one hidden state per env × agent
    hx = actor.init_hidden(E * N, device=str(device)).reshape(E, N, -1)

    ep_reward   = np.zeros(E, dtype=np.float64)   # per-env accumulator
    ep_rewards: list[float] = []
    ep_count    = 0
    last_infos  = info_list
    best_mean_reward = -np.inf

    for update in range(start_update, total_updates + 1):

        if lr_decay:
            linear_lr_decay(mappo.actor_opt,  lr_actor_0,  update, total_updates)
            linear_lr_decay(mappo.critic_opt, lr_critic_0, update, total_updates)

        # ----------------------------------------------------------------
        # Collect T environment steps across all E envs
        # ----------------------------------------------------------------
        for _ in range(T):
            if obs_rms is not None:
                obs_rms.update(obs.reshape(E * N, obs_dim))
                obs_in = obs_rms.normalize(obs)   # (E, N, obs_dim)
            else:
                obs_in = obs

            actions, log_probs, values, new_hx = mappo.get_actions(obs_in, state, hx)
            # actions : (E, N, 2)   log_probs : (E, N)
            # values  : (E,)        new_hx    : (E, N, gru_hidden)

            next_obs, rewards, terms, dones, last_infos, next_state = vec_env.step(actions)
            # next_obs : (E, N, obs_dim)   rewards : (E, N)
            # terms    : (E,) bool — collision termination (GAE mask)
            # dones    : (E,) bool — any episode end (hx reset)
            # next_state : (E, state_dim)

            buffer.insert(
                torch.from_numpy(obs_in).float().to(device),
                torch.from_numpy(state).float().to(device),
                torch.from_numpy(actions).float().to(device),
                log_probs.to(device),
                torch.from_numpy(rewards).float().to(device),
                values.to(device),
                # GAE masking: only zero the bootstrap on hard termination (collision).
                # Timeout (truncation) still bootstraps — V(s_{timeout}) is meaningful.
                torch.from_numpy(terms.astype(np.float32)).to(device),
                hx,
            )

            ep_reward += rewards.mean(axis=-1)   # (E,) — per-env mean agent reward
            for e in range(E):
                if dones[e]:
                    ep_rewards.append(float(ep_reward[e]))
                    ep_reward[e] = 0.0
                    ep_count    += 1

            obs   = next_obs
            state = next_state
            # Zero hx for ALL episode endings (collision or timeout)
            done_mask = torch.from_numpy(dones.astype(np.float32)).to(device)  # (E,)
            hx = new_hx * (1.0 - done_mask[:, None, None])

        # ----------------------------------------------------------------
        # Bootstrap: V(s_T) for each env
        # ----------------------------------------------------------------
        if obs_rms is not None:
            obs_in_last = obs_rms.normalize(obs)
        else:
            obs_in_last = obs
        _, _, last_values, _ = mappo.get_actions(obs_in_last, state, hx)
        buffer.compute_returns(last_values.to(device), gamma, gae_lambda)

        # ----------------------------------------------------------------
        # Policy update
        # ----------------------------------------------------------------
        losses = mappo.update(buffer)

        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------
        if update % log_interval == 0:
            window    = ep_rewards[-20:] if ep_rewards else [0.0]
            mean_ep_r = float(np.mean(window))
            last_cov  = float(np.mean([i['coverage_ratio'] for i in last_infos]))
            print(
                f"Update {update:5d}/{total_updates} | "
                f"episodes={ep_count:6d} | "
                f"mean_ep_r={mean_ep_r:8.3f} | "
                f"coverage={last_cov:.2%} | "
                f"actor={losses['actor_loss']:7.4f} | "
                f"critic={losses['critic_loss']:7.4f} | "
                f"entropy={losses['entropy']:6.4f}"
            )
            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    update, ep_count, round(mean_ep_r, 4),
                    round(last_cov, 4),
                    round(losses['actor_loss'],  4),
                    round(losses['critic_loss'], 4),
                    round(losses['entropy'],     4),
                ])
            if ep_count > 0 and mean_ep_r > best_mean_reward:
                best_mean_reward = mean_ep_r
                best_path = os.path.join(save_dir, 'checkpoint_final.pt')
                save_checkpoint(best_path, update, actor, critic, mappo, obs_rms)
                print(f"  → best policy saved (mean_ep_r={mean_ep_r:.3f})")

    save_checkpoint(os.path.join(save_dir, 'checkpoint_final.pt'),
                    total_updates, actor, critic, mappo, obs_rms)
    vec_env.close()
    return actor, critic, obs_rms


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
    args = parser.parse_args()
    train(args.config, args.save_dir, args.resume)
